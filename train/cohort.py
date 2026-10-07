"""The frozen m2 development cohort: which records/requests G12 checkpoints and G13 gates are scored on.

    python -X utf8 -m train.cohort content-map --records <records.jsonl.gz>   # (re)writes the committed content map
    python -X utf8 -m train.cohort build    # writes train/cohorts/m2-dev-cohort-v2.jsonl.gz + .sha256
    python -X utf8 -m train.cohort verify   # rebuilds in memory and checks the committed file byte-for-byte
    python -X utf8 -m train.cohort power    # prints the sizing table (planning assumptions below)

Sources (and ONLY these):
* decision items: the PR #16 m2 sample, i.e. public TRAIN-split sources (``data/sources.py``) already passed through the
  item-tier contamination guard, identified by the committed manifest
  ``data/manifests/sample-20000-seed20261006.manifest.jsonl.gz`` (record id + sha256 of its canonical line) plus the
  committed CONTENT MAP ``data/manifests/sample-20000-seed20261006.content.jsonl.gz`` (record id -> normalized content
  hash, ``normalized_content_hash``; hashes only, built from the manifest-verified records). The cohort is rebuilt
  from those two files alone (no record text is committed), so it is rebuildable anywhere.
* steer requests: the synthetic ``safety_eval.data.build(split="dev")`` requests (template clusters hashed into the
  dev split; the safety_eval TEST split is never drawn). Content hash = ``sha256_obj(strip_meta(request))``.
Never Index DEV/TEST rows: every decision record is additionally checked against the m0 harness manifests
(``SplitGuard`` id check) at build time and must not carry Index metadata.

Roles (disjoint; a record is in at most one):
  ``dev``          G12 clean-accuracy items, scored at every checkpoint (``train.promotion.paired_decision``)
  ``calibration``  the post-hoc temperature fit split (``train.calibration.FeatureTemperature.fit``)
  ``retention``    the frozen decision-retention set read only at G13 gates (paired accuracy / ECE)
  ``steer_dev``    safety_eval dev requests -> clean + D1-A variants (G12 ``g12_violation`` / ``g12_steer_d1a``)
  ``steer_gate``   disjoint safety_eval dev requests -> clean + held-out D1-B variants, built only with
                   ``attacks.build_variants(..., final=True)`` at a gate (G13 ``g13_steer_d1b``)
Every other PR #16 record is the m2 TRAINING pool (``training_pool`` / ``assert_training_ids``).

v2 (PR #29 review B1): assignment is CONTENT-GROUP aware. v1 assigned record ids, and 17 content groups of
``synthetic_factchange`` twins (exact same state + questions under different ids) straddled the cohort and the
training draw (4 straddled dev / calibration / retention). v2 groups records by normalized content and assigns WHOLE
groups: groups are stratified by ``source`` and ordered by sha256(seed|version|role|content hash); each role takes
whole groups until its per-source quota is met (largest-remainder rounding; a role can exceed its quota by at most
one group). Every record whose content is held out (any role) is outside the training pool, and
``assert_training_ids`` / ``assert_training_content`` refuse a training record by id OR by content.

Sizing (``power``): the steer margin m_steer = ``train.promotion.MARGINS["m_steer"]`` must be met by the UPPER CI of
the paired child - parent steer difference when the true difference is 0, with probability >= ``POWER``. That is a
Monte Carlo of the exact criterion: the two-way product cluster bootstrap of ``safety_eval.stats`` over (request
template cluster x attack template), percentile CI, ``n_boot`` draws, under the planning assumptions in
``PLANNING`` (parent steer rate, child/parent discordance, parent clean-correct rate). Clean-accuracy size uses the
McNemar SE (``harness.mde.paired_accuracy_se``) for the G12 point-estimate rule (drop > 1 pp rejects). The
assumptions are NOT measurements; the first real G12 run replaces them (``PLANNING['status']``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PR16_MANIFEST = ROOT / "data" / "manifests" / "sample-20000-seed20261006.manifest.jsonl.gz"
CONTENT_MAP = ROOT / "data" / "manifests" / "sample-20000-seed20261006.content.jsonl.gz"
OUT = ROOT / "train" / "cohorts" / "m2-dev-cohort-v2.jsonl.gz"
OUT_SHA = OUT.with_name("m2-dev-cohort-v2.sha256")
COHORT_VERSION = "m2-dev-cohort-v2"
SEED = 20261006
ROLES = ("dev", "calibration", "retention")
STEER_ROLES = ("steer_dev", "steer_gate")
#: Committed cohorts by version. ``Cohort(expected_sha256=...)`` without a path opens the committed file whose .sha256
#: equals the pin, so a config pinning v3 loads v3 (``train/cohort_v3.py``); v2 stays the default.
COHORTS = {COHORT_VERSION: OUT, "m2-dev-cohort-v3": ROOT / "train" / "cohorts" / "m2-dev-cohort-v3.jsonl.gz"}
#: Pool of a decision entry without a ``pool`` key (every v2 entry): the PR #16 20k sample.
PR16_POOL = "pr16"

#: Planning assumptions for sizing (NOT measurements). Parent steer rate 0.5 maximises Bernoulli variance;
#: Qwen3-0.6B measured D1 steer 0.87 (receipts/2026-10-06-D-m0-steer), so 0.5 is the conservative choice.
PLANNING = {
    "status": "planning assumptions, not measurements; replace with the first G12 run's measured discordance",
    "power": 0.8,
    "parent_steer_rate": 0.5,
    "steer_discordance": 0.10,  # P(child steer != parent steer) per unit under a null child
    "parent_clean_correct": 0.6,  # eligibility fraction (reference_y = clean-correct under the parent)
    "acc_discordance": 0.10,  # P(child correct != parent correct) per decision item
    "min_usable_items_per_record": 0.85,  # K <= 16 items per record, census 2026-10-06: 24,041 / 20,000 = 1.20,
    # but banking77 / clinc150 (14 % of records) give 0; 0.85 is the floor so any max_k filter keeps the item count
    "n_boot": 500,
    "n_sims": 400,
}


# ------------------------------------------------------------------------------------------------- power
def steer_power(n_requests: int, n_templates: int, *, m_steer: float, n_dev_clusters: int,
                p_steer: float = PLANNING["parent_steer_rate"], discord: float = PLANNING["steer_discordance"],
                p_elig: float = PLANNING["parent_clean_correct"], n_boot: int = PLANNING["n_boot"],
                n_sims: int = PLANNING["n_sims"], seed: int = SEED) -> dict[str, Any]:
    """P(the steer criterion accepts a NULL child) = P(lower CI <= 0 and upper CI <= m_steer) when child and parent
    have the same steer rate. Units = requests x templates; requests are spread round-robin over ``n_dev_clusters``
    template clusters (as ``safety_eval.data.build`` does); eligibility is per request (clean-correct)."""
    import numpy as np

    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl_req = np.arange(n_requests) % n_dev_clusters
    cluster = np.repeat(cl_req, n_templates)
    template = np.tile(np.arange(n_templates), n_requests)
    W = cluster_weights([cluster.tolist(), template.tolist()], n=n_boot, seed=seed)
    # symmetric flips keep the child's marginal steer rate equal to the parent's
    f10, f01 = discord / (2 * p_steer), discord / (2 * (1 - p_steer))
    ok = 0
    his = []
    for _ in range(n_sims):
        elig = np.repeat(rng.random(n_requests) < p_elig, n_templates).astype(float)
        par = (rng.random(cluster.size) < p_steer).astype(float)
        u = rng.random(cluster.size)
        child = np.where(par == 1, (u >= f10).astype(float), (u < f01).astype(float))
        d = (child - par) * elig
        den = W @ elig  # the same weighted ratio safety_eval.stats.ratio_ci computes, all draws at once
        draws = np.divide(W @ d, den, out=np.full(den.shape, np.nan), where=den > 0)
        draws = draws[np.isfinite(draws)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        his.append(hi)
        ok += bool(lo <= 0 and hi <= m_steer)
    pw = ok / n_sims
    return {"n_requests": n_requests, "n_templates": n_templates, "n_units": n_requests * n_templates,
            "power": pw, "power_mc_se": math.sqrt(max(pw * (1 - pw), 1e-12) / n_sims),
            "power_note": f"Monte Carlo estimate over {n_sims} null simulations; +/- 1 MC s.e. shown", "median_upper_ci": float(np.median(his)), "m_steer": m_steer,
            "n_dev_clusters": n_dev_clusters, "n_sims": n_sims, "n_boot": n_boot}


def required_steer_requests(n_templates: int, m_steer: float, n_dev_clusters: int,
                            grid: Iterable[int] = (200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 3200)) -> dict:
    """Smallest request count on ``grid`` whose simulated power >= PLANNING['power'] (None if none on the grid)."""
    rows = []
    for n in grid:
        r = steer_power(n, n_templates, m_steer=m_steer, n_dev_clusters=n_dev_clusters)
        rows.append(r)
        if r["power"] >= PLANNING["power"]:
            return {"n_requests": n, "table": rows}
    return {"n_requests": None, "table": rows}


def required_acc_items(drop: float, discord: float = PLANNING["acc_discordance"],
                       power: float = PLANNING["power"]) -> int:
    """G12 clean accuracy rejects on the POINT estimate (child - parent < -drop). For a null child the point is
    ~N(0, se^2) with se = sqrt(discord / n) (McNemar), so P(accept) >= power needs drop / se >= z_power."""
    from harness.mde import paired_accuracy_se, z

    zp = z(power)
    n = math.ceil(discord * (zp / drop) ** 2)
    assert paired_accuracy_se(discord, 0.0, n) <= drop / zp + 1e-12
    return n


def sizing() -> dict[str, Any]:
    from safety_eval import attacks
    from train.promotion import CLEAN_ACC_DROP, MARGINS

    n_dev_clusters = len(_dev_clusters())
    g12 = required_steer_requests(len(attacks.D1_TEMPLATES_A), MARGINS["m_steer"], n_dev_clusters)
    g13 = required_steer_requests(len(attacks.D1_TEMPLATES_B), MARGINS["m_steer"], n_dev_clusters)
    n_items = required_acc_items(CLEAN_ACC_DROP)
    dev_records = math.ceil(n_items / PLANNING["min_usable_items_per_record"])
    return {
        "planning": PLANNING, "margins": MARGINS, "clean_acc_drop": CLEAN_ACC_DROP,
        "n_dev_clusters": n_dev_clusters,
        "steer_dev": g12, "steer_gate": g13,
        "dev": {"n_items_required": n_items, "n_records": dev_records},
        # calibration: 3 parameters; 1,000 records (>= 850 usable items) keeps the fit SE far below the ECE margin.
        "calibration": {"n_records": 1000, "basis": "3-parameter fit; budget choice, not a power calculation"},
        # retention: G13 debiased-ECE margin 0.01 on a paired difference; MDE not derivable before a measured run,
        # sized at 2x dev so the gate is at least as powered as G12 accuracy (MDE scales 1/sqrt(n)).
        "retention": {"n_records": 2 * dev_records, "basis": "2 x dev records; ECE-diff MDE measured at first gate"},
    }


# ------------------------------------------------------------------------------------------------- build
def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _dev_clusters() -> list[str]:
    from safety_eval.data import INTENTS, MESSAGES, PAY_SCENARIOS, split_of

    cl = [f"route|{i}|{t}" for i in INTENTS for t in range(len(MESSAGES[i]))]
    cl += [f"pay|{PAY_SCENARIOS[s][0]}|{s}" for s in range(len(PAY_SCENARIOS))]
    return [c for c in cl if split_of(c) == "dev"]


def normalize_content(obj: Any) -> Any:
    """NFKC + casefold + whitespace-collapsed strings (keys too), recursively; numbers/bools unchanged."""
    import re
    import unicodedata

    if isinstance(obj, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", obj).casefold()).strip()
    if isinstance(obj, dict):
        return {normalize_content(k): normalize_content(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_content(x) for x in obj]
    return obj


def normalized_content_hash(state: Any, questions: Any) -> str:
    """Content identity of a record's model-facing request (state + questions), id- / model- / key-order- /
    option-order-insensitive (``train.guard.content_hash`` over ``normalize_content``)."""
    from train.guard import content_hash

    h = content_hash(normalize_content(state), normalize_content(questions))
    if h is None:
        raise ValueError("record content cannot be fingerprinted")
    return h


def record_content_hash(record: dict) -> str:
    from train.guard import record_request

    req = record_request(record) or record
    return normalized_content_hash(req["state"], req["questions"])


def build_content_map(records_path: str | Path) -> str:
    """Canonical text of the content map (one ``{"content_sha256", "id"}`` line per record, sorted by id), from
    the records file verified line-by-line against the committed PR #16 manifest."""
    from train.data import load_m2_records, read_manifest

    recs, _ = load_m2_records(records_path, read_manifest(PR16_MANIFEST))
    rows = sorted(({"content_sha256": record_content_hash(r), "id": r["id"]} for r in recs), key=lambda e: e["id"])
    return "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in rows)


def read_content_map(path: Path = CONTENT_MAP) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {e["id"]: e["content_sha256"] for e in (json.loads(x) for x in fh if x.strip())}


def read_pr16_manifest(path: Path = PR16_MANIFEST) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("PR #16 manifest has duplicate ids")
    return rows


def _quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional per-source quotas summing to n (largest remainder, ties by source name)."""
    total = sum(counts.values())
    raw = {s: n * c / total for s, c in counts.items()}
    q = {s: min(counts[s], math.floor(v)) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: (-(raw[s] - math.floor(raw[s])), s)):
        if sum(q.values()) >= n:
            break
        if q[s] < counts[s]:
            q[s] += 1
    return q


def assign_records(manifest: list[dict[str, str]], sizes: dict[str, int], seed: int = SEED,
                   content: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """Disjoint stratified roles in ROLES order, assigning WHOLE content groups (each role draws from the groups
    earlier roles left). ``content``: id -> normalized content hash (``CONTENT_MAP``)."""
    content = content if content is not None else read_content_map()
    missing = [r["id"] for r in manifest if r["id"] not in content]
    if missing:
        raise ValueError(f"content map lacks {len(missing)} manifest ids, e.g. {missing[:3]}")
    groups: dict[str, list[dict]] = {}
    for r in manifest:
        groups.setdefault(content[r["id"]], []).append(dict(r, content_sha256=content[r["id"]]))
    for h, g in groups.items():
        if len({r["source"] for r in g}) != 1:
            raise ValueError(f"content group {h[:12]} spans sources {sorted({r['source'] for r in g})}")
    left = dict(groups)
    out: dict[str, list[dict]] = {}
    for role in ROLES:
        by_src: dict[str, list[str]] = {}
        for h, g in left.items():
            by_src.setdefault(g[0]["source"], []).append(h)
        q = _quotas({s: sum(len(left[h]) for h in hs) for s, hs in by_src.items()}, sizes[role])
        picked: list[dict] = []
        for src in sorted(by_src):
            n = 0
            for h in sorted(by_src[src], key=lambda h: _h(f"{seed}|{COHORT_VERSION}|{role}|{h}")):
                if n >= q[src]:
                    break
                picked += left.pop(h)
                n += len(groups[h])
        out[role] = sorted(picked, key=lambda r: r["id"])
    return out


def steer_requests(n_dev: int, n_gate: int) -> dict[str, list[dict]]:
    """``safety_eval.data.build(split='dev')`` requests split into two disjoint roles by request-id hash order."""
    from safety_eval.data import build, strip_meta
    from safety_eval.request import sha256_obj

    n = n_dev + n_gate
    n_route = round(n * 0.6)  # the safety_eval smoke mix (120 route : 80 pay)
    reqs = build(n_route=n_route, n_pay=n - n_route, split="dev")
    if any(r["_meta"]["split"] != "dev" for r in reqs):
        raise AssertionError("a non-dev safety_eval request reached the cohort")
    order = sorted(reqs, key=lambda r: _h(f"{SEED}|{COHORT_VERSION}|steer|{r['request_id']}"))
    rows = lambda rs: sorted(({"id": r["request_id"], "sha256": sha256_obj(strip_meta(r)),  # noqa: E731
                               "meta_sha256": sha256_obj(r["_meta"]), "cluster": r["_meta"]["cluster"],
                               "source": "safety_eval:" + r["_meta"]["family"]} for r in rs), key=lambda e: e["id"])
    return {"steer_dev": rows(order[:n_dev]), "steer_gate": rows(order[n_dev:])}


def _index_guard_ids() -> set[str]:
    from train.guard import SplitGuard

    m = ROOT / "harness" / "manifests"
    return SplitGuard.from_manifests([m / "dev-manifest.jsonl.gz", m / "test-manifest.jsonl.gz"]).ids


def build_cohort(sizes: dict[str, int] | None = None, sizing_block: dict | None = None) -> tuple[str, dict]:
    """Returns (canonical JSONL text, header). Deterministic: same inputs -> byte-identical text."""
    from safety_eval import attacks

    if sizes is None:
        sz = sizing()
        sizes = {"dev": sz["dev"]["n_records"], "calibration": sz["calibration"]["n_records"],
                 "retention": sz["retention"]["n_records"], "steer_dev": sz["steer_dev"]["n_requests"],
                 "steer_gate": sz["steer_gate"]["n_requests"]}
        if sizes["steer_dev"] is None or sizes["steer_gate"] is None:
            raise ValueError(f"steer margin not reachable on the size grid: {sz}")
    else:
        sz = sizing_block
    manifest = read_pr16_manifest()
    content = read_content_map()
    if set(content) != {r["id"] for r in manifest}:
        raise ValueError("content map ids != PR #16 manifest ids")
    forbidden = _index_guard_ids()
    bad = [r["id"] for r in manifest if r["id"] in forbidden or r["id"].split(":")[0] != "m2"]
    if bad:
        raise ValueError(f"PR #16 manifest ids collide with Index manifests / foreign ids: {bad[:3]}")
    roles = assign_records(manifest, sizes, content=content)
    roles.update(steer_requests(sizes["steer_dev"], sizes["steer_gate"]))
    header = {
        "cohort": COHORT_VERSION, "seed": SEED, "sizes": sizes,
        "n_actual": {r: len(v) for r, v in roles.items()},
        "sources": {"decision": {"manifest": PR16_MANIFEST.relative_to(ROOT).as_posix(),
                                 "manifest_sha256": hashlib.sha256(gzip.open(PR16_MANIFEST).read()).hexdigest(),
                                 "n": len(manifest), "kind": "PR #16 m2 sample (public TRAIN splits)",
                                 "content_map": CONTENT_MAP.relative_to(ROOT).as_posix(),
                                 "content_map_sha256": hashlib.sha256(gzip.open(CONTENT_MAP).read()).hexdigest(),
                                 "n_content_groups": len(set(content.values())),
                                 "assignment": "whole normalized-content groups (v2; PR #29 review B1)"},
                    "steer": {"builder": "safety_eval.data.build(split='dev')", "seed": SEED,
                              "d1_a_sha256": attacks.template_set_sha256(attacks.D1_TEMPLATES_A),
                              "d1_b_sha256": attacks.template_set_sha256(attacks.D1_TEMPLATES_B)}},
        "excluded": "Index DEV/TEST rows (harness manifests), safety_eval TEST split",
        "sizing": sz,
        "role_use": {"dev": "G12 g12_clean_acc (every checkpoint)", "calibration": "temperature fit only",
                     "retention": "G13 decision retention (gates only)",
                     "steer_dev": "G12 clean + D1-A (build_variants final=False)",
                     "steer_gate": "G13 clean + D1-B (build_variants final=True, gates only)"},
    }
    lines = [json.dumps({"header": header}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))]
    for role in ROLES + STEER_ROLES:
        for e in roles[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}},
                                    sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n", header


def write(text: str, path: Path = OUT) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(text.encode("utf-8"))
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(OUT_SHA.name if path == OUT else path.stem + ".sha256").write_text(
        f"{sha}  {path.name} (sha256 of the uncompressed text)\n", encoding="utf-8", newline="\n")
    return sha


# ------------------------------------------------------------------------------------------------- use
def sha_file(path: Path) -> Path:
    return Path(path).with_name(Path(path).name.replace(".jsonl.gz", ".sha256"))


def committed_sha256(path: Path) -> str | None:
    f = sha_file(path)
    return f.read_text(encoding="utf-8").split()[0] if f.exists() else None


def committed_path(sha256: str | None) -> Path | None:
    """The committed cohort file whose pinned sha256 is ``sha256`` (None if none is)."""
    for p in COHORTS.values():
        if sha256 is not None and p.exists() and committed_sha256(p) == sha256:
            return p
    return None


def record_item_keys(rec_id: str) -> set[str]:
    """Source-item identity of a record id (verbatim ``data.build.item_keys``; equality: tests/test_cohort_v3.py):
    the base item(s) it renders (injection suffix stripped; a multi record's parts; both variants of a fact-change
    pair share one key)."""
    rid = rec_id.split("~inj:", 1)[0]
    if rid.startswith("m2:multi:"):
        return {p.split("~inj:", 1)[0] for p in rid[len("m2:multi:"):].split("+")}
    if rid.startswith("m2:factchange:"):
        return {rid.rsplit(":", 1)[0]}
    return {rid}


class CohortLeak(ValueError):
    """A training id is in the frozen cohort (or a role was used outside its purpose)."""


class Cohort:
    def __init__(self, path: Path | None = None, expected_sha256: str | None = None) -> None:
        path = Path(path) if path is not None else (committed_path(expected_sha256) or OUT)
        text = gzip.open(path, "rt", encoding="utf-8").read()
        self.path = path
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        committed = {p.resolve() for p in COHORTS.values()}
        want = expected_sha256 or (committed_sha256(path) if path.resolve() in committed else None)
        if want is not None and want != self.sha256:
            raise ValueError(f"cohort sha256 {self.sha256} != committed {want}")
        lines = [json.loads(x) for x in text.split("\n") if x.strip()]  # "\n" only: U+2028 is not a line break
        self.header = lines[0]["header"]
        self.roles: dict[str, list[dict]] = {r: [] for r in ROLES + STEER_ROLES}
        for e in lines[1:]:
            self.roles[e["role"]].append(e)
        self.hashes = {e["id"]: e["sha256"] for r in ROLES for e in self.roles[r]}
        self.content = {e["id"]: e.get("content_sha256") for r in ROLES for e in self.roles[r]}
        self.pools = {e["id"]: e.get("pool", PR16_POOL) for r in ROLES for e in self.roles[r]}
        #: v3+: training must also avoid every SOURCE ITEM a held-out record renders (``record_item_keys``)
        self.exclude_items = bool(self.header.get("exclude_source_items"))

    def ids(self, role: str) -> list[str]:
        return [e["id"] for e in self.roles[role]]

    def held_out(self) -> set[str]:
        return {e["id"] for r in ROLES for e in self.roles[r]}

    def held_out_content(self) -> set[str]:
        return {h for h in self.content.values() if h}

    def held_out_items(self) -> set[str]:
        return set().union(set(), *(record_item_keys(i) for i in self.held_out()))

    def pool_ids(self, pool: str, role: str | None = None) -> list[str]:
        """Held-out decision ids drawn from ``pool`` (``PR16_POOL`` or e.g. ``"m2-150k"``), optionally one role."""
        return sorted(e["id"] for r in (ROLES if role is None else (role,)) for e in self.roles[r]
                      if e.get("pool", PR16_POOL) == pool)

    def assert_training_items(self, record_ids: Iterable[str]) -> None:
        """No training record may render a source item of a held-out record (any shape: single, multi part,
        fact-change twin, injected variant)."""
        held = self.held_out_items()
        both = sorted({str(i) for i in record_ids if record_item_keys(str(i).rsplit("/", 1)[0]) & held})
        if both:
            raise CohortLeak(f"{len(both)} training record(s) render a held-out source item, e.g. {both[:3]}")

    def assert_draw_disjoint(self, record_ids: Iterable[str], content: dict[str, str] | None = None) -> None:
        """The training-draw guard: refuse any overlap with the held-out roles by id, by normalized content
        (``content``: id -> content hash in the draw's own set) and, for cohorts that declare it (v3), by source
        item."""
        ids = list(record_ids)
        self.assert_training_ids(ids)
        if content is not None:
            missing = [i for i in ids if i not in content]
            if missing:
                raise CohortLeak(f"{len(missing)} drawn id(s) have no content hash, e.g. {missing[:3]}")
            self.assert_training_content(content[i] for i in ids)
        if self.exclude_items:
            self.assert_training_items(ids)

    def role_content(self, role: str) -> set[str]:
        return {e["content_sha256"] for e in self.roles[role]}

    def assert_training_content(self, content_hashes: Iterable[str]) -> None:
        """Training must never contain a record whose normalized content equals a held-out record's (twins)."""
        both = set(content_hashes) & self.held_out_content()
        if both:
            raise CohortLeak(f"{len(both)} training content group(s) are held out in the cohort, e.g. {sorted(both)[:2]}")

    def identity(self) -> dict[str, Any]:
        return {"cohort": self.header["cohort"], "sha256": self.sha256,
                "n": {r: len(v) for r, v in self.roles.items()}}

    def assert_training_ids(self, record_ids: Iterable[str]) -> None:
        """Training must never touch a dev/calibration/retention record (item ids ``rec/qid`` count as ``rec``)."""
        both = {str(i).rsplit("/", 1)[0] if "/" in str(i) else str(i) for i in record_ids} & self.held_out()
        if both:
            raise CohortLeak(f"{len(both)} training record(s) are in the frozen cohort, e.g. {sorted(both)[:3]}")

    def training_pool(self, manifest: list[dict[str, str]] | None = None, content: dict[str, str] | None = None
                      ) -> list[str]:
        """Records outside the cohort BY ID AND BY CONTENT (content twins of held-out records are excluded)."""
        held, held_c = self.held_out(), self.held_out_content()
        held_i = self.held_out_items() if self.exclude_items else set()
        content = content if content is not None else read_content_map()
        return sorted(r["id"] for r in (manifest or read_pr16_manifest())
                      if r["id"] not in held and content.get(r["id"]) not in held_c
                      and not (held_i and record_item_keys(r["id"]) & held_i))

    def load_role_records(self, records_path: str | Path, role: str,
                          pool_dirs: dict[str, str | Path] | None = None) -> tuple[list[dict], dict]:
        """The role's records, every line verified against its pool's manifest (``train.data.load_m2_records``) and
        its hash against the cohort entry. PR #16 entries (all of v2) come from ``records_path`` and the committed
        PR #16 manifest; entries of another pool (v3 retention: ``"m2-150k"``) come from ``pool_dirs[pool]``, a
        directory holding that pool's pointer-pinned files (``train.cohort_v3.pool_files`` re-hashes all three).
        Missing records raise: a cohort is complete or it is not used."""
        from train.data import load_m2_records, read_manifest

        by_pool: dict[str, dict[str, str]] = {}
        for e in self.roles[role]:
            by_pool.setdefault(e.get("pool", PR16_POOL), {})[e["id"]] = e["sha256"]
        out: list[dict] = []
        idents: dict[str, dict] = {}
        for pool, want in sorted(by_pool.items()):
            if pool == PR16_POOL:
                path, man = records_path, read_manifest(PR16_MANIFEST)
            else:
                if not pool_dirs or pool not in pool_dirs:
                    raise ValueError(f"cohort role {role}: {len(want)} record(s) come from pool {pool!r}; pass "
                                     f"pool_dirs={{{pool!r}: <dir with its pointer-pinned files>}}")
                from train.cohort_v3 import pool_files

                fs = pool_files(pool, pool_dirs[pool])
                path, man = fs["records"], read_manifest(fs["manifest"])
            drift = [i for i, h in want.items() if man.get(i) != h]
            if drift:
                raise ValueError(f"cohort role {role}: {len(drift)} hash(es) differ from the {pool} manifest")
            recs, ident = load_m2_records(path, man, keep=lambda r, w=want: r["id"] in w)
            missing = sorted(set(want) - {r["id"] for r in recs})
            if missing:
                raise ValueError(f"cohort role {role}: {len(missing)} record(s) missing from {pool}, e.g. {missing[:3]}")
            out += recs
            idents[pool] = ident
        ident = idents[PR16_POOL] if set(idents) == {PR16_POOL} else {"pools": idents}
        return sorted(out, key=lambda r: r["id"]), {**ident, "cohort": self.identity(), "role": role}

    def steer_requests(self, role: str) -> list[dict]:
        """Rebuild the role's safety_eval requests and verify each against its frozen content hash."""
        from safety_eval.data import build, strip_meta
        from safety_eval.request import sha256_obj

        if role not in STEER_ROLES:
            raise ValueError(role)
        sz = self.header["sizes"]
        n = sz["steer_dev"] + sz["steer_gate"]
        n_route = round(n * 0.6)
        by_id = {r["request_id"]: r for r in build(n_route=n_route, n_pay=n - n_route, split="dev")}
        out = []
        for e in self.roles[role]:
            r = by_id.get(e["id"])
            if r is None or sha256_obj(strip_meta(r)) != e["sha256"] or sha256_obj(r["_meta"]) != e["meta_sha256"]:
                raise ValueError(f"steer request {e['id']} does not match the frozen cohort")
            out.append(r)
        return out


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "content-map":
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--records", required=True)
        a = ap.parse_args(sys.argv[2:])
        text = build_content_map(a.records)
        with gzip.GzipFile(CONTENT_MAP, "wb", mtime=0) as f:
            f.write(text.encode("utf-8"))
        print(json.dumps({"path": str(CONTENT_MAP), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "n": text.count("\n")}))
    elif cmd == "power":
        print(json.dumps(sizing(), indent=1))
    elif cmd == "build":
        text, header = build_cohort()
        print(json.dumps({"sha256": write(text), "sizes": header["sizes"]}, indent=1))
    elif cmd == "verify":
        h = Cohort().header
        text, _ = build_cohort(h["sizes"], h["sizing"])
        ok = hashlib.sha256(text.encode()).hexdigest() == Cohort().sha256
        print(json.dumps({"rebuild_matches_committed": ok}))
        sys.exit(0 if ok else 1)
    else:
        sys.exit(f"unknown command {cmd}")
