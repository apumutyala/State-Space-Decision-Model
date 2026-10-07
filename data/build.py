"""End-to-end sample build: load -> render/augment -> inject -> validate -> held-out guard -> Index dedup -> write.

Outputs (outside git): ``<out>/records.jsonl.gz``. Committed: a manifest (record id + sha256 per line) and the
receipt (dedup log, summary). Quotas are counts of KEPT records, so dropped records are replaced by the next item
of the same source; the dedup log counts every record checked.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from . import augment as A
from . import config, convert as C, injection as I
from .dedup import DedupLog, Fingerprint
from .text import MIN_SEGMENT_CHARS, canonical_json, normalize, strings_in, words

# Kept-record shares of the sample (sum 1.0).
SHARES = {"banking77": 0.07, "clinc150": 0.07, "anli": 0.09, "hellaswag": 0.08, "winogrande": 0.07,
          "gsm8k": 0.07, "mmlu_aux": 0.09, "sst5": 0.05, "boolq": 0.06, "when2call": 0.06, "toolace": 0.08,
          "nq": 0.09, "multi": 0.06, "factchange": 0.06}
MULTI_FROM = ["banking77", "clinc150", "anli", "winogrande", "hellaswag", "mmlu_aux", "gsm8k", "sst5", "boolq"]
# The 150k mix (Track D m2-data-150k): kept-record COUNTS per bucket for n=150,000 (shares = count / n). Weighted
# toward knowledge / retrieval / tools (Index area weights K 25.8, L 25.8, R 20, T 18.3, A 10); each count is
# capped below the source's measured post-guard, post-cohort capacity (receipts/2026-10-06-D-m2-data-150k).
COUNTS_150K = {"mmlu_aux": 31000, "gsm8k": 5500,                                        # knowledge
               "anli": 10000, "hellaswag": 9000, "winogrande": 7000, "sst5": 4000,      # language
               "nq": 10000, "banking77": 7000, "clinc150": 9000, "boolq": 8000,         # retrieval
               "glaive_fc": 15000, "toolace": 6000, "when2call": 5500,                  # tools
               "humicroedit_s1": 5000, "humicroedit_s2": 6500,                          # arts
               "multi": 8500, "factchange": 3000}
SHARES_150K = {k: v / 150000 for k, v in COUNTS_150K.items()}
#: Index area of each source (SOURCES.md section 1; BoolQ = passage-evidence verification like HoVer -> retrieval;
#: fact-change pairs = agentic decisions under untrusted tool facts -> tools; multi is attributed per part).
AREA = {"mmlu_aux": "knowledge", "gsm8k": "knowledge", "anli": "language", "hellaswag": "language",
        "winogrande": "language", "sst5": "language", "nq": "retrieval", "banking77": "retrieval",
        "clinc150": "retrieval", "boolq": "retrieval", "glaive_fc": "tools", "toolace": "tools", "when2call": "tools",
        "synthetic_factchange": "tools", "humicroedit_s1": "arts", "humicroedit_s2": "arts"}
INDEX_AREA_WEIGHTS = {"knowledge": 25.8, "language": 25.8, "retrieval": 20.0, "tools": 18.3, "arts": 10.0}
P_INJECT = 0.10         # in-place injection on single-question records (label unchanged)
P_INJECT_FACT = 0.15     # one variant of a fact-change pair also carries an injection
P_PAD = 0.08


def _rec_sha(rec: dict) -> str:
    return hashlib.sha256(canonical_json(rec).encode("utf-8")).hexdigest()


def _len_words(rec: dict) -> int:
    return len(words(canonical_json({"state": rec["state"], "questions": rec["questions"]})))


def item_keys(rec_id: str) -> set[str]:
    """Source-item identity of a record id: the base item(s) it renders (injection suffix stripped; a multi
    record's parts; both variants of a fact-change pair share one key). Keeps every source item of a held-out
    cohort record out of training, whatever record shape it reappears in."""
    rid = rec_id.split("~inj:", 1)[0]
    if rid.startswith("m2:multi:"):
        return {p.split("~inj:", 1)[0] for p in rid[len("m2:multi:"):].split("+")}
    if rid.startswith("m2:factchange:"):
        return {rid.rsplit(":", 1)[0]}
    return {rid}


class Exclusion:
    """Records that must never enter this training set: the frozen m2 dev cohort (``train/cohorts``) held-out
    roles (dev / calibration / retention), by record id, by normalized content (twins) and by source item."""

    def __init__(self, ids: set[str], contents: set[str], items: set[str], identity: dict,
                 probes: set[str] | None = None):
        self.ids, self.contents, self.items, self.identity = ids, contents, items, identity
        self.probes = probes or set()

    @classmethod
    def from_cohort(cls, seed: int = config.SEED, with_probes: bool = True) -> "Exclusion":
        from train.cohort import Cohort  # the repo root must be importable (the build CLI runs from the repo)
        c = Cohort()  # verifies the committed sha256
        ids = c.held_out()
        items = set().union(*(item_keys(i) for i in ids))
        probes, pinfo = held_out_probes(items, seed) if with_probes else (set(), {})
        return cls(ids, c.held_out_content(), items,
                   {**c.identity(), "n_held_out_ids": len(ids), "n_held_out_content": len(c.held_out_content()),
                    "n_held_out_items": len(items), "probe_texts": pinfo}, probes)

    def reason(self, rec: dict, content: str | None) -> str | None:
        if rec["id"] in self.ids:
            return "id"
        if content is not None and content in self.contents:
            return "content"
        if item_keys(rec["id"]) & self.items:
            return "item"
        if self.probes and rec.get("source") != "synthetic_factchange" and any(
                p in self.probes for p in probe_norms(rec.get("_probe") or [])):
            return "probe"
        return None


def probe_norms(texts) -> set[str]:
    """Normalised item texts long enough to identify an item (the exact matcher's segment floor)."""
    out = set()
    for t in texts:
        n = normalize(t)
        if len(n) >= MIN_SEGMENT_CHARS:
            out.add(n)
    return out


def held_out_probes(items: set[str], seed: int) -> tuple[set[str], dict]:
    """Normalised ITEM-field texts of every held-out source item (PR #44 review finding 1: the same item text can
    recur under another id, e.g. duplicate When2Call questions or BoolQ questions that are NQ queries). Each held
    source item is re-derived from its pinned loader (deterministic order), so no record text is needed. Fact-change
    items are synthetic templates and are excluded by id/item/content only."""
    want: dict[str, set[str]] = collections.defaultdict(set)
    for k in items:
        parts = k.split(":")
        if parts[1] != "factchange":
            want[parts[1]].add(k)
    probes, found = set(), collections.Counter()
    for sid, keys in sorted(want.items()):
        left = set(keys)
        for base in C.LOADERS[sid](seed):
            if base["id"] in left:
                left.discard(base["id"])
                found[sid] += 1
                probes |= probe_norms(t for f in base["item_fields"] for t in strings_in(base["fields"].get(f)))
                if not left:
                    break
        if left:
            raise RuntimeError(f"held-out items not found in the {sid} loader: {sorted(left)[:3]}")
    return probes, {"n_items": sum(found.values()), "n_texts": len(probes), "by_source": dict(sorted(found.items())),
                    "rule": f"normalize(text) >= {MIN_SEGMENT_CHARS} chars; candidate _probe (item tier) vs held "
                            "item fields; fact-change excluded"}


def content_guard():
    """``train.guard.SplitGuard`` over the harness DEV/TEST manifests plus the content index, refused unless the
    index and manifests match ``jobs/m2_pilot/guard_pin.json``. Returns (guard, identity)."""
    from train.guard import SplitGuard
    root = config.REPO_ROOT
    pin = json.loads((root / "jobs" / "m2_pilot" / "guard_pin.json").read_text(encoding="utf-8"))
    mans = {k: root / k for k in pin["harness_manifests_sha256"]}
    g = SplitGuard.from_manifests(list(mans.values()), content_index=config.M2_ROOT / "index-content-hashes-v1.txt.gz")
    want = pin["content_index"]
    if g.content_index["sha256"] != want["sha256"] or g.content_index["n"] != want["n_hashes"]:
        raise RuntimeError(f"content index {g.content_index} != pin {want}")
    for k, p in mans.items():
        if g.manifests[str(p)] != pin["harness_manifests_sha256"][k]:
            raise RuntimeError(f"{k}: sha256 != guard pin")
    ident = g.identity()
    ident["manifests"] = {k: pin["harness_manifests_sha256"][k] for k in mans}
    return g, {**ident, "pin": want}


def build_sample(n_target: int = 20000, out: str | None = None, receipt: str | None = None,
                 seed: int = config.SEED, fingerprint: Fingerprint | None = None, sources: dict | None = None,
                 log=print, allow_test_fingerprint: bool = False, exclusion: Exclusion | None = None,
                 guard=None, keep_records: bool = True, dedup_factchange: bool = False,
                 manifest_dir: str | None = None) -> dict:
    """``allow_test_fingerprint`` is for unit tests with synthetic fingerprints only; a receipt build refuses it.

    Additive options, all off by default so the PR #16 20k build is byte-identical: ``exclusion`` skips held-out
    cohort records (id / content / source item) before the guard; ``guard`` (a ``train.guard.SplitGuard`` with
    the Index content index) is a sixth item-tier matcher, logged as ``item_content_index``;
    ``keep_records=False`` streams records to disk without holding them; ``dedup_factchange`` skips a fact-change
    pair whose variant content already occurs; ``manifest_dir`` writes the manifest and the content map there."""
    t0 = time.time()
    fp = fingerprint or Fingerprint.load(config.FINGERPRINT)
    if not allow_test_fingerprint:
        fp.assert_production()
    elif receipt:
        raise ValueError("a receipt build needs the production fingerprint")
    dlog = DedupLog(fp.meta)
    shares = sources or SHARES
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"shares sum to {sum(shares.values())}, not 1")
    quotas = {k: int(round(v * n_target)) for k, v in shares.items()}
    quotas[max(quotas, key=quotas.get)] += n_target - sum(quotas.values())
    rng_master = random.Random(seed)
    iters = {sid: C.LOADERS[sid](seed) for sid in C.LOADERS if sid in quotas or sid in MULTI_FROM}
    pad_pool = []
    if "nq" in C.LOADERS:
        rows = C.nq_pairs(seed)
        r = random.Random(f"{seed}|pad")
        pad_pool = [rows[r.randrange(len(rows))]["answer"] for _ in range(4000)]
    kept: list[dict] = []
    stats = collections.defaultdict(collections.Counter)
    held_out_checked = 0
    excluded = collections.defaultdict(collections.Counter)
    need_content = exclusion is not None or dedup_factchange or manifest_dir is not None
    if need_content:
        from train.cohort import normalized_content_hash
    out_dir = Path(out or config.BUILD_DIR / f"sample-{n_target}-seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl.gz"
    tmp_path = out_dir / "records.jsonl.gz.part"
    writer = gzip.open(tmp_path, "wt", encoding="utf-8", newline="\n", compresslevel=6)
    manifest_lines: list[str] = []
    content_lines: list[tuple[str, str]] = []
    kept_ids: set[str] = set()
    kept_content = collections.Counter()
    area_items = collections.Counter()

    def hits_of(rec: dict) -> tuple[dict, dict]:
        nonlocal held_out_checked
        A.validate(rec)
        I.assert_no_heldout(rec["state"], rec["questions"], rec["id"])
        held_out_checked += 1
        item, full = fp.check(rec["_probe"], {}), fp.check(rec["state"], rec["questions"])
        if guard is not None:
            from train.guard import ForbiddenRow
            try:
                guard.check(rec)
            except ForbiddenRow:
                item = {**item, "content_index": ["index-content-hash"]}
        return item, full

    def bucket_of(rec: dict) -> str:
        return rec["source"] if rec["family"] != "multi" else "multi"

    def chash(rec: dict) -> str | None:
        # hashed on the record AS WRITTEN (canonical JSON, sorted keys): normalize_content casefolds keys, so keys
        # that collide under casefold (2 ToolACE schemas) make the hash key-order dependent; train.cohort's content
        # map is computed from records read back from disk, so the build must hash the same form
        if not need_content:
            return None
        rt = json.loads(canonical_json({"state": rec["state"], "questions": rec["questions"]}))
        return normalized_content_hash(rt["state"], rt["questions"])

    def excluded_by_cohort(rec: dict, ch: str | None) -> bool:
        why = exclusion.reason(rec, ch) if exclusion is not None else None
        if why:
            excluded[bucket_of(rec)][why] += 1
        return bool(why)

    def accept(rec: dict) -> bool:
        if excluded_by_cohort(rec, chash(rec)):
            return False
        item, full = hits_of(rec)
        return dlog.add(bucket_of(rec), rec["id"], item, full)

    def emit(rec: dict, bucket: str):
        note(rec, bucket)
        if rec["id"] in kept_ids:
            raise AssertionError(f"duplicate record id: {rec['id']}")
        kept_ids.add(rec["id"])
        if rec["family"] == "multi":
            for p in rec["aug"]["parts"]:
                area_items[AREA.get(p.split(":")[1], "unmapped")] += 1
        else:
            area_items[AREA.get(rec["source"], "unmapped")] += 1
        ch = chash(rec)
        rec.pop("_probe", None)
        line = canonical_json(rec)
        writer.write(line + "\n")
        manifest_lines.append(canonical_json({"id": rec["id"], "sha256": hashlib.sha256(line.encode()).hexdigest(),
                                              "source": rec["source"], "family": rec["family"]}))
        if ch is not None:
            content_lines.append((rec["id"], ch))
            kept_content[ch] += 1
        if keep_records:
            kept.append(rec)

    def note(rec: dict, bucket: str):
        s = stats[bucket]
        s["records"] += 1
        s[f"family:{rec['family']}"] += 1
        a = rec.get("aug", {})
        for k in ("placement", "state_shape", "id_style", "shape"):
            if k in a:
                s[f"{k}:{a[k]}"] += 1
        if "injection" in a:
            s[f"injection:{a['injection']['family']}:{a['injection']['template']}"] += 1
        if a.get("padding_passages"):
            s["padded"] += 1
        for q in rec["questions"].values():
            s[f"qtype:{q['type']}"] += 1
            if q["type"] == "choice":
                n = len(q["criteria"])
                b = "2" if n == 2 else "3-4" if n <= 4 else "5-10" if n <= 10 else "11-64" if n <= 64 else \
                    "65-151" if n <= 151 else "152-255"
                s[f"options:{b}"] += 1
        L = _len_words(rec)
        lb = "<64" if L < 64 else "64-255" if L < 256 else "256-1023" if L < 1024 else "1024-4095" if L < 4096 else ">=4096"
        s[f"words:{lb}"] += 1

    # ---- single-source records
    for sid in [s for s in quotas if s in C.LOADERS]:
        got = 0
        for base in iters[sid]:
            if got >= quotas[sid]:
                break
            rng = random.Random(f"{seed}|render|{base['id']}")
            rec = A.render(base, rng, pad_pool=pad_pool, p_pad=P_PAD)
            if rec["aug"].get("shape") == "single" and rng.random() < P_INJECT:
                inj = I.inject(rec, rng)
                rec = inj or rec
            if accept(rec):
                emit(rec, sid); got += 1
        if got < quotas[sid]:
            log(f"  {sid}: source exhausted at {got}/{quotas[sid]}")
        log(f"  {sid}: kept {got} ({time.time() - t0:.0f}s)")

    # ---- multi-question records (items drawn from the same iterators: no item reused)
    got, k = 0, 0
    while got < quotas.get("multi", 0):
        rng = random.Random(f"{seed}|multi|{k}"); k += 1
        m = rng.choice([2, 2, 3, 4])
        parts = []
        for _ in range(m):
            sid = rng.choice(MULTI_FROM)
            base = next(iters[sid], None)
            if base is None:
                continue
            parts.append(A.render(base, random.Random(f"{seed}|render|{base['id']}"), placement="state"))
        if len(parts) < 2:
            continue
        rec = A.merge(parts, rng)
        if accept(rec):
            emit(rec, "multi"); got += 1
    log(f"  multi: kept {got}")

    # ---- fact-change pairs (both variants kept or both dropped)
    got, k = 0, 0
    fams = sorted(I.FACT_FAMILIES)
    fc_seen: set[str] = set()
    fc_twins = 0
    while got < quotas.get("factchange", 0):
        if dedup_factchange and k >= 50 * max(1, quotas["factchange"]):
            log(f"  factchange: distinct pairs exhausted at {got}/{quotas['factchange']} after {k} draws")
            break
        fam = fams[k % len(fams)]
        pair = I.fact_change_pair(fam, k, seed)
        # the SAME option permutation/relabelling for both variants (fresh RNG with one seed per pair), so the
        # pair differs only in the untrusted fact and the label (PR #16 review: a shared advancing RNG gave the
        # two variants different option orders/ids)
        pair = [A.reshuffle_options(p, random.Random(f"{seed}|fcaug|{k}")) for p in pair]
        if pair[0]["questions"] != pair[1]["questions"]:
            raise AssertionError(f"fact-change pair {k}: questions differ after augmentation")
        rng = random.Random(f"{seed}|fcinj|{k}"); k += 1
        if rng.random() < P_INJECT_FACT:
            v = rng.randrange(2)
            inj = I.inject(pair[v], rng, family="d1a")
            if inj is not None:
                pair[v] = inj
        chs = [chash(p) for p in pair]
        if dedup_factchange and any(c in fc_seen for c in chs):
            fc_twins += 1
            continue
        if any([excluded_by_cohort(p, c) for p, c in zip(pair, chs)]):
            continue
        fc_seen.update(c for c in chs if c)
        # decide on the pair first, then log: a clean partner of a dropped variant is logged as dropped
        hs = [hits_of(p) for p in pair]
        pair_ok = all(not any(h[0].values()) for h in hs)
        for p, (item, full) in zip(pair, hs):
            dlog.add(bucket_of(p), p["id"], item, full, partner_dropped=not pair_ok)
        if pair_ok:
            for p in pair:
                emit(p, "factchange"); got += 1
    log(f"  factchange: kept {got}" + (f" (twin pairs skipped: {fc_twins})" if dedup_factchange else ""))

    # ---- finish the streamed write (record ids were asserted unique as they were emitted)
    writer.close()
    tmp_path.replace(rec_path)
    h = hashlib.sha256()
    with gzip.open(rec_path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    total = collections.Counter()
    for c in stats.values():
        total.update(c)
    summary = {
        "n_target": n_target, "n_kept": len(manifest_lines), "seed": seed, "quotas": quotas,
        "records_uncompressed_sha256": h.hexdigest(), "records_path": str(rec_path),
        "held_out_template_guard_checked": held_out_checked,
        "dedup_totals": dlog.to_json()["totals"],
        "by_bucket": {k: dict(sorted(v.items())) for k, v in sorted(stats.items())},
        "overall": dict(sorted(total.items())),
        "seconds": round(time.time() - t0, 1),
    }
    if exclusion is not None:
        summary["cohort_exclusion"] = {"cohort": exclusion.identity,
                                       "skipped_by_bucket": {k: dict(v) for k, v in sorted(excluded.items())},
                                       "skipped_total": dict(sum(excluded.values(), collections.Counter()))}
    if guard is not None:
        summary["content_index_guard_fired"] = dlog.to_json()["totals"].get("item_content_index", 0)
    if need_content:
        summary["content"] = {"n_distinct_normalized_content": len(kept_content),
                              "records_in_twin_groups": sum(v for v in kept_content.values() if v > 1),
                              "twin_groups": sum(1 for v in kept_content.values() if v > 1)}
    if exclusion is not None or sources is SHARES_150K:
        tot = sum(area_items.values())
        summary["area_balance"] = {"unit": "source items (a multi record counts each part in its source's area)",
                                   "items": dict(sorted(area_items.items())),
                                   "share_pct": {a: round(100 * v / tot, 2) for a, v in sorted(area_items.items())},
                                   "index_area_weight_pct": INDEX_AREA_WEIGHTS}
    res = {"summary": summary, "dedup_log": dlog.to_json(), "manifest_lines": manifest_lines, "records": kept}
    if content_lines:
        res["content_lines"] = [json.dumps({"content_sha256": c, "id": i}, sort_keys=True, separators=(",", ":"))
                                for i, c in sorted(content_lines)]
    if receipt:
        rd = Path(receipt)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "dedup-log.json").write_text(json.dumps(res["dedup_log"], indent=1) + "\n", encoding="utf-8")
        (rd / "build-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        md = Path(manifest_dir) if manifest_dir else config.MANIFEST_DIR
        md.mkdir(parents=True, exist_ok=True)
        mp = md / f"sample-{n_target}-seed{seed}.manifest.jsonl.gz"
        with gzip.GzipFile(mp, "wb", mtime=0) as f:
            f.write(("\n".join(manifest_lines) + "\n").encode("utf-8"))
        if manifest_dir and content_lines:  # id -> normalized content hash, the format train.cohort reads
            with gzip.GzipFile(md / f"sample-{n_target}-seed{seed}.content.jsonl.gz", "wb", mtime=0) as f:
                f.write("".join(x + "\n" for x in res["content_lines"]).encode("utf-8"))
    return res
