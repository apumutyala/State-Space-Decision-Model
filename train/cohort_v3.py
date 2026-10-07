"""m2 development cohort v3: cohort v2 with the G13 RETENTION role expanded from the PR #44 150k m2 set.

    python -X utf8 -m train.cohort_v3 build  --train-dir <dir>   # writes train/cohorts/m2-dev-cohort-v3.jsonl.gz + .sha256
    python -X utf8 -m train.cohort_v3 verify --train-dir <dir>   # rebuilds in memory, compares byte-for-byte

``<dir>`` holds the 150k set's three files under the names its committed pointer gives
(``data/manifests/sample-150000-seed20261006.pointer.json``); every file's uncompressed sha256 must equal the pointer.

Why (issue #36, Codex 6028537932; PR #102 ``train/margin_sim.py``): with the approved margins (m_ret_acc 0.01, lower CI
>= -0.01; retention ECE upper CI <= +0.01) an unchanged candidate needs ~9.3k retention items for accuracy and ~30.7k
for ECE to pass with probability >= 0.8. v2 has 1,670 records (~1.4k items). Margins never move for power; the
evaluation is expanded instead.

What v3 is:
* Every v2 entry (dev, calibration, retention, steer_dev, steer_gate) unchanged, byte for byte and in order.
* The retention role additionally holds ``N_RETENTION_NEW`` records of the 150k set, tagged ``"pool": "m2-150k"``
  (v2 entries carry no ``pool`` key: they are PR #16 records). Selection follows the v2 construction: WHOLE groups,
  stratified, each stratum's groups in sha256(seed|version|role|group key) order until its largest-remainder quota is
  met. Groups are the connected components of "same normalized content" OR "same source item"
  (``record_item_keys``: injection suffix stripped, multi-record parts, both fact-change variants), so no content twin
  and no source item straddles retention and training. Strata = source x word-length bin (sources nest in areas).
* ``exclude_source_items``: training must avoid held-out records by id, content AND source item.

Disjointness, all checked at build time and recorded in the header (a selected record failing any raises):
1. Training draws used or planned. Every m2 training draw from the 150k set (first cell n=24,000 seed 0,
   jobs/m2_first_cell; the q9_feature_cache rows of that draw; the Clef teacher export, jobs/clef_baseline
   teacher_scope.json) is ``train.run_cell.split_ids(sorted(set - heldout_excluded), n, 0, seed=0)``, i.e. the first n
   of the pool in sha256("0:" + id) order. No group with any member among the first ``DRAW_RESERVE`` of that order is
   eligible, so (a) the drawn 24,000 (``drawn_ids_sha256`` 2bbbf180...) are disjoint, and (b) every seed-0 draw of up
   to ``DRAW_RESERVE`` records is byte-identical whether or not it excludes v3 (excluding records that sort after the
   prefix never changes the prefix). From now on every draw from the set excludes v3 (``training_exclusion`` /
   ``assert_draw_disjoint``, wired into ``baselines.clef_teacher``), so larger or other-seed draws stay disjoint.
2. Index DEV/TEST rows: ``train.guard.SplitGuard`` over the harness DEV + TEST manifests (id, row / payload hash) and
   the pinned content index (``data.build.content_guard``: ``jobs/m2_pilot/guard_pin.json``).
3. D1-A / D1-B safety sets: injection-augmented records (``~inj:`` ids: D1-A verbatim and the training-only hijack
   family) are never eligible; ``train.heldout.scan`` over every candidate with the D1-B AND the D1-A template sets;
   the 3 pinned D1-B exclusions of the 150k set are never eligible. safety_eval requests are a disjoint synthetic
   source (v2 steer roles unchanged).
4. ``harness.heldout.check_record`` (heldout-draw-v1, the 9 frozen held-out families): zero ``FrozenFamilyError``.
5. Cohort v2 held-out records (by id, content and source item): none (the 150k build already excluded them).
6. The PR #16 20k sample: every m2 PILOT draw comes from it (jobs/m2_pilot cells: 3,000 of the v2 training pool, seed
   0; jobs/rwkv7 pilot_cell: legacy ``split_ids`` 2,000 + 300). Ids are source-item based across builds, so the same
   source item can sit in both sets. Any group sharing a source item or a normalized content with ANY PR #16 record
   (committed manifest + content map) is ineligible, so v3 is disjoint from every PR #16 draw of any size or seed.
Candidates refused by a guard are excluded with their whole group and listed in the header (``guard_excluded``).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from train.cohort import (COHORTS, OUT, ROLES, SEED, STEER_ROLES, Cohort, CohortLeak, _h, _quotas, committed_sha256,
                          read_content_map, record_item_keys, sha_file)

ROOT = Path(__file__).resolve().parents[1]
COHORT_V3 = "m2-dev-cohort-v3"
OUT_V3 = COHORTS[COHORT_V3]
POOL_150K = "m2-150k"
POINTERS = {POOL_150K: ROOT / "data" / "manifests" / "sample-150000-seed20261006.pointer.json"}
#: The 150k set's pre-registered D1-B exclusions (jobs/m2_first_cell, jobs/clef_baseline/teacher_scope.json): never
#: drawn for training, never eligible here.
HELDOUT_EXCLUDED_150K = ("m2:mmlu_aux:21118", "m2:mmlu_aux:39640", "m2:mmlu_aux:39643")
DRAW_SEED = 0
#: Seed-0 draws of up to this many records are untouched by v3 (2.5 x the first cell's 24,000).
DRAW_RESERVE = 60_000
#: Records added to the retention role (sized so the MEASURED usable items, K <= 16 and T <= 2,048 tokens under BOTH
#: parents' tokenizers, clear the ~30.7k retention-ECE requirement with margin; receipts/2026-10-07-D-cohort-v3).
N_RETENTION_NEW = 38_000
LENGTH_BINS = ((64, "<64"), (256, "64-255"), (1024, "256-1023"), (4096, "1024-4095"), (None, ">=4096"))
#: Pinned: the drawn ids of the m2 first cell / teacher / q9-cache draw (n 24,000, seed 0).
FIRST_CELL_DRAW = {"n": 24_000, "seed": 0,
                   "drawn_ids_sha256": "2bbbf1808c0bdd6cf0ba3ecd236cf9a64110c5e429e22f8b8e9cd2df7507556b"}


class PoolMismatch(ValueError):
    """The pool files are not the ones the committed pointer pins."""


def draw_key(rec_id: str, seed: int = DRAW_SEED) -> str:
    """The sort key of ``train.run_cell.split_ids``."""
    return hashlib.sha256(f"{seed}:{rec_id}".encode()).hexdigest()


def length_bin(n_words: int) -> str:
    for hi, name in LENGTH_BINS:
        if hi is None or n_words < hi:
            return name
    raise AssertionError


def read_pointer(pool: str) -> dict[str, Any]:
    p = POINTERS[pool]
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_pointer_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    d["_pointer_path"] = p.relative_to(ROOT).as_posix()
    return d


def _usha(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


_VERIFIED: dict[tuple[str, str], dict] = {}


def pool_files(pool: str, pool_dir: str | Path, verify: bool = True) -> dict[str, Any]:
    """Paths of the pool's records / manifest / content map in ``pool_dir``, each re-hashed (uncompressed sha256)
    against the committed pointer (once per process)."""
    ptr = read_pointer(pool)
    d = Path(pool_dir)
    fs = {"records": d / "records.jsonl.gz", "manifest": d / ptr["manifest"], "content_map": d / ptr["content_map"]}
    key = (pool, str(d.resolve()))
    if verify and key not in _VERIFIED:
        want = {"records": ptr["records_uncompressed_sha256"], "manifest": ptr["manifest_uncompressed_sha256"],
                "content_map": ptr["content_map_uncompressed_sha256"]}
        got = {}
        for k, p in fs.items():
            if not p.exists():
                raise PoolMismatch(f"{pool} {k} file missing: {p}")
            got[k] = _usha(p)
            if got[k] != want[k]:
                raise PoolMismatch(f"{pool} {k} uncompressed sha256 {got[k]} != pointer {want[k]}")
        _VERIFIED[key] = got
    return {**fs, "files_uncompressed_sha256": _VERIFIED.get(key), "pointer": ptr}


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise PoolMismatch("pool manifest has duplicate ids")
    return rows


# ------------------------------------------------------------------------------------------------- build
class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.p.setdefault(x, x)
        while p != self.p[p]:
            self.p[p] = self.p[self.p[p]]
            p = self.p[p]
        self.p[x] = p
        return p

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _record_stats(rec: dict) -> dict[str, Any]:
    from data.build import _len_words

    qs = rec["questions"]
    return {"source": rec["source"], "family": rec.get("family"), "shape": (rec.get("aug") or {}).get("shape"),
            "types": [q["type"] for q in qs.values()],
            "ks": [len(q.get("criteria") or {}) if q["type"] == "choice" else 2 for q in qs.values()],
            "words": _len_words(rec)}


def _part_sources(rec_id: str, source: str) -> list[str]:
    if rec_id.startswith("m2:multi:"):
        return [k.split(":")[1] for k in sorted(record_item_keys(rec_id))]
    return [source]


def composition(ids: Iterable[str], stats: dict[str, dict]) -> dict[str, Any]:
    from data.build import AREA

    ids = list(ids)
    src, area_items, qtypes, bins, shapes, fam = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    n_q = n_k16 = 0
    for i in ids:
        s = stats[i]
        src[s["source"]] += 1
        for ps in _part_sources(i, s["source"]):
            area_items[AREA.get(ps, "unknown")] += 1
        qtypes.update(s["types"])
        bins[length_bin(s["words"])] += 1
        shapes[str(s["shape"] or "multi")] += 1
        fam[str(s["family"])] += 1
        n_q += len(s["types"])
        n_k16 += sum(1 for k in s["ks"] if k <= 16)
    srt = lambda c: dict(sorted(c.items()))  # noqa: E731
    return {"n_records": len(ids), "n_questions": n_q, "n_questions_k_le_16": n_k16, "by_source": srt(src),
            "items_by_area": srt(area_items), "question_types": srt(qtypes),
            "length_words": {name: bins.get(name, 0) for _, name in LENGTH_BINS}, "shapes": srt(shapes),
            "families": srt(fam)}


def scan_guards(records: Iterable[dict], guard) -> dict[str, list[str]]:
    """{record id: [reasons]} for every record a guard refuses (DEV/TEST SplitGuard, frozen held-out families,
    D1-B / D1-A template fragments)."""
    _ensure_kit()
    from harness.heldout import FrozenFamilyError, check_record
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B
    from train.guard import ForbiddenRow
    from train.heldout import record_text, scan

    bad: dict[str, list[str]] = defaultdict(list)
    texts = []
    for rec in records:
        try:
            guard.check(rec)
        except ForbiddenRow as e:
            bad[rec["id"]].append(f"split_guard: {e}")
        try:
            check_record(rec)
        except FrozenFamilyError as e:
            bad[rec["id"]].append(f"frozen_family: {e}")
        texts.append((rec["id"], record_text(rec)))
    for name, tpl in (("d1_b", D1_TEMPLATES_B), ("d1_a", D1_TEMPLATES_A)):
        for h in scan(texts, templates=tpl)["hits"]:
            bad[h["id"]].append(f"{name}_fragment: {h['template']}")
    return dict(bad)


def _ensure_kit() -> None:
    """``harness.heldout`` imports the official kit (``decision_index``); add the pinned local clone if needed."""
    from scripts.m2_e2e_dryrun import ensure_kit

    ensure_kit()


def guard_identity(guard) -> dict[str, Any]:
    _ensure_kit()
    from harness import config as hconfig
    from harness.heldout import frozen
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, template_set_sha256
    from train.heldout import K_WORDS, MIN_WORDS

    gi = guard.identity()
    return {"split_guard": {"manifests": {Path(k).relative_to(ROOT).as_posix() if Path(k).is_relative_to(ROOT) else k: v
                                          for k, v in gi["manifests"].items()},
                            "content_index": {k: v for k, v in gi["content_index"].items() if k != "path"}
                            if isinstance(gi["content_index"], dict) else gi["content_index"]},
            "frozen_families": {"heldout_draw_v1_sha256": hconfig.HELDOUT_DRAW_V1_SHA256, "families": sorted(frozen())},
            "template_scan": {"k_words": K_WORDS, "min_words": MIN_WORDS,
                              "d1_a_sha256": template_set_sha256(D1_TEMPLATES_A),
                              "d1_b_sha256": template_set_sha256(D1_TEMPLATES_B)}}


def production_guard():
    """The pinned DEV/TEST guard with the content index (``data.build.content_guard``; refuses unless it matches
    ``jobs/m2_pilot/guard_pin.json``)."""
    from data.build import content_guard

    return content_guard()[0]


def pr16_exclusion() -> tuple[set[str], set[str]]:
    """(source items, normalized contents) of every record of the PR #16 20k sample (committed manifest + map)."""
    from train.cohort import read_pr16_manifest

    cm = read_content_map()
    ids = [r["id"] for r in read_pr16_manifest()]
    return set().union(*(record_item_keys(i) for i in ids)), {cm[i] for i in ids}


def select_retention(manifest: list[dict], content: dict[str, str], stats: dict[str, dict], n_new: int,
                     v2: Cohort, guard_bad: dict[str, list[str]], reserve: int = DRAW_RESERVE,
                     seed: int = SEED, pr16: tuple[set[str], set[str]] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Pure selection (no I/O): returns (selected ids sorted, selection report)."""
    ids = [r["id"] for r in manifest]
    missing_ex = sorted(set(HELDOUT_EXCLUDED_150K) - set(ids))
    if missing_ex:
        raise PoolMismatch(f"pinned D1-B exclusions not in the pool: {missing_ex}")
    pool = sorted(set(ids) - set(HELDOUT_EXCLUDED_150K))
    order = sorted(pool, key=draw_key)
    reserve = min(reserve, len(order))
    reserved = set(order[:reserve])
    cutoff = draw_key(order[reserve - 1])
    uf = _UF()
    for i in ids:
        uf.find(i)
        uf.union(i, "c:" + content[i])
        for k in record_item_keys(i):
            uf.union(i, "i:" + k)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[uf.find(i)].append(i)
    v2_ids, v2_c, v2_i = v2.held_out(), v2.held_out_content(), v2.held_out_items()
    p16_i, p16_c = pr16 if pr16 is not None else (set(), set())
    why: Counter = Counter()
    eligible: dict[str, list[str]] = {}
    for g, mem in groups.items():
        reasons = set()
        for i in mem:
            if i in reserved:
                reasons.add("draw_reserve")
            if i in HELDOUT_EXCLUDED_150K:
                reasons.add("heldout_excluded")
            if "~inj:" in i:
                reasons.add("injected")
            if i in v2_ids or content[i] in v2_c or record_item_keys(i) & v2_i:
                reasons.add("cohort_v2")
            if i in guard_bad:
                reasons.add("guard")
            if record_item_keys(i) & p16_i or content[i] in p16_c:
                reasons.add("pr16_sample")
        for r in reasons:
            why[r] += len(mem)
        if not reasons:
            eligible[g] = sorted(mem)
    strata: dict[str, list[str]] = defaultdict(list)
    for g, mem in eligible.items():
        s = stats[mem[0]]
        strata[f"{s['source']}|{length_bin(s['words'])}"].append(g)
    counts = {s: sum(len(eligible[g]) for g in gs) for s, gs in strata.items()}
    if n_new > sum(counts.values()):
        raise ValueError(f"n_new {n_new} > eligible records {sum(counts.values())}")
    q = _quotas(counts, n_new)
    picked: list[str] = []
    for s in sorted(strata):
        n = 0
        gkey = lambda g: _h(f"{seed}|{COHORT_V3}|retention|{min(content[i] for i in eligible[g])}")  # noqa: E731
        for g in sorted(strata[s], key=gkey):
            if n >= q[s]:
                break
            picked += eligible[g]
            n += len(eligible[g])
    picked = sorted(picked)
    rep = {"n_pool_records": len(ids), "n_groups": len(groups), "n_eligible_records": sum(counts.values()),
           "n_eligible_groups": len(eligible), "ineligible_records_by_reason": dict(sorted(why.items())),
           "strata": {s: {"eligible": counts[s], "quota": q[s]} for s in sorted(counts)},
           "draw": {"seed": DRAW_SEED, "pool": "sorted(set(manifest ids) - HELDOUT_EXCLUDED_150K)",
                    "rule": "train.run_cell.split_ids order: sha256(f'{seed}:{id}')", "reserve": reserve,
                    "reserve_cutoff_key": cutoff,
                    "reserve_ids_sha256": hashlib.sha256("\n".join(order[:reserve]).encode()).hexdigest(),
                    "first_cell_draw_ids_sha256": hashlib.sha256("\n".join(order[:FIRST_CELL_DRAW["n"]]).encode()
                                                                 ).hexdigest()}}
    return picked, rep


def build_cohort_v3(train_dir: str | Path, n_new: int = N_RETENTION_NEW, guard=None, log=print
                    ) -> tuple[str, dict, dict]:
    """(canonical JSONL text, header, per-record stats of the selected records). Deterministic."""
    from data.build import AREA  # noqa: F401  (fail early if the data package is not importable)

    v2 = Cohort(OUT)
    fs = pool_files(POOL_150K, train_dir)
    ptr = fs["pointer"]
    manifest = read_manifest_rows(fs["manifest"])
    man = {r["id"]: r["sha256"] for r in manifest}
    content = read_content_map(fs["content_map"])
    if set(content) != set(man) or len(man) != int(ptr["n_records"]):
        raise PoolMismatch("content-map ids != manifest ids, or n != pointer n_records")
    guard = guard if guard is not None else production_guard()
    pool = sorted(set(man) - set(HELDOUT_EXCLUDED_150K))
    cutoff = draw_key(sorted(pool, key=draw_key)[DRAW_RESERVE - 1])
    stats: dict[str, dict] = {}
    cands: list[dict] = []
    n_lines = 0
    with gzip.open(fs["records"], "rt", encoding="utf-8", newline="\n") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            n_lines += 1
            rec = json.loads(line)
            if man.get(rec["id"]) != hashlib.sha256(line.encode("utf-8")).hexdigest():
                raise PoolMismatch(f"{rec['id']}: line sha256 does not match the manifest")
            stats[rec["id"]] = _record_stats(rec)
            if draw_key(rec["id"]) > cutoff and "~inj:" not in rec["id"] and rec["id"] not in HELDOUT_EXCLUDED_150K:
                cands.append(rec)
    if n_lines != len(man) or set(stats) != set(man):
        raise PoolMismatch(f"records file has {n_lines} lines / {len(stats)} ids, manifest {len(man)}")
    log(f"read {n_lines} records; guarding {len(cands)} candidates beyond the draw reserve")
    bad = scan_guards(cands, guard)
    del cands
    p16 = pr16_exclusion()
    picked, rep = select_retention(manifest, content, stats, n_new, v2, bad, pr16=p16)
    # fail-closed re-checks on the selection itself
    sel = set(picked)
    assert all(draw_key(i) > rep["draw"]["reserve_cutoff_key"] for i in picked)
    assert not sel & set(HELDOUT_EXCLUDED_150K) and not any("~inj:" in i for i in picked)
    assert not sel & set(bad)
    v2.assert_draw_disjoint(picked, content)
    v2.assert_training_items(picked)
    if any(record_item_keys(i) & p16[0] or content[i] in p16[1] for i in picked):
        raise CohortLeak("a selected record shares a source item / content with the PR #16 sample")
    src = {r["id"]: r["source"] for r in manifest}
    new = [{"content_sha256": content[i], "id": i, "pool": POOL_150K, "role": "retention", "sha256": man[i],
            "source": src[i]} for i in picked]
    header = json.loads(json.dumps(v2.header))
    v2_n = {r: len(v) for r, v in v2.roles.items()}
    header.update({
        "cohort": COHORT_V3,
        "extends": {"cohort": v2.header["cohort"], "sha256": v2.sha256,
                    "note": "every v2 entry unchanged and in order; retention += pool m2-150k records"},
        "exclude_source_items": True,
        "sizes": {**v2.header["sizes"], "retention": v2.header["sizes"]["retention"] + n_new},
        "n_actual": {**v2_n, "retention": v2_n["retention"] + len(new)},
    })
    header["sources"]["retention_150k"] = {
        "pool": POOL_150K, "pointer": ptr["_pointer_path"], "pointer_sha256": ptr["_pointer_sha256"],
        "build": ptr.get("build"), "files_uncompressed_sha256": fs["files_uncompressed_sha256"],
        "n_records": len(man), "kind": "PR #44 150k m2 set (public TRAIN splits, cohort-v2 excluded at build)",
        "heldout_excluded": list(HELDOUT_EXCLUDED_150K), "n_new": len(new),
        "assignment": "whole groups (normalized content OR source item), stratified source x word-length bin",
        "selection": rep, "guards": guard_identity(guard),
        "guard_excluded": {i: bad[i] for i in sorted(bad)},
        "composition_new": composition(picked, stats)}
    header["excluded"] = (header["excluded"] + "; retention_150k: m2 training draws (seed-0 reserve), injected "
                          "(D1-A / hijack) records, D1-A/D1-B fragments, frozen held-out families, cohort v2, any PR #16 source item / "
                          "content (pilot draws)")
    header["role_use"]["retention"] = "G13 decision retention (gates only); v2 records + pool m2-150k records"
    header["sizing"]["retention_v3"] = {
        "basis": "train/margin_sim.py (PR #102): unchanged candidate passes g13_retention_acc (m 0.01) with "
                 "~9.3k items and g13_retention_ece (m 0.01) with ~30.7k items at P >= 0.8; v3 is sized so the "
                 "MEASURED usable items (K <= 16, T <= 2,048, both parents' tokenizers) clear 32,000",
        "n_new_records": n_new}
    lines = [json.dumps({"header": header}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))]
    for role in ROLES + STEER_ROLES:
        rows = list(v2.roles[role]) + (new if role == "retention" else [])
        for e in rows:
            lines.append(json.dumps(e, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n", header, {i: stats[i] for i in picked}


def write_v3(text: str, path: Path = OUT_V3) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(text.encode("utf-8"))
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sha_file(path).write_text(f"{sha}  {path.name} (sha256 of the uncompressed text)\n", encoding="utf-8",
                              newline="\n")
    return sha


# ------------------------------------------------------------------------------------------------- training guard
def cohort_v3() -> Cohort:
    """The committed v3 (sha256 verified against ``m2-dev-cohort-v3.sha256``)."""
    return Cohort(OUT_V3)


def training_exclusion(manifest_ids: Iterable[str], content: dict[str, str], cohort: Cohort | None = None
                       ) -> dict[str, Any]:
    """Records of a training set a draw must leave out because cohort v3 holds them, their content or a source item
    they render. Every m2 draw from the 150k set removes these from its pool BEFORE ``split_ids`` (then calls
    ``assert_draw_disjoint``). Also refuses a set that lacks a v3 pool record (not the pinned build)."""
    co = cohort if cohort is not None else cohort_v3()
    ids = list(manifest_ids)
    held, held_c, held_i = co.held_out(), co.held_out_content(), co.held_out_items()
    ex = {i for i in ids if i in held or content.get(i) in held_c or record_item_keys(i) & held_i}
    return {"ids": ex, "n": len(ex), "cohort": co.identity(),
            "rule": "id OR normalized content OR source item held out by the cohort (train.cohort_v3)"}


def assert_draw_disjoint(drawn_ids: Iterable[str], content: dict[str, str], cohort: Cohort | None = None) -> None:
    (cohort if cohort is not None else cohort_v3()).assert_draw_disjoint(list(drawn_ids), content)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.cohort_v3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--train-dir", required=True, help="dir with the 150k set's pointer-pinned files")
        s.add_argument("--n-new", type=int, default=N_RETENTION_NEW)
        s.add_argument("--stats-out", default=None, help="write the selected records' stats (receipt input)")
    a = ap.parse_args(argv)
    text, header, stats = build_cohort_v3(a.train_dir, a.n_new)
    if a.stats_out:
        Path(a.stats_out).write_text(json.dumps(stats, sort_keys=True), encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    if a.cmd == "build":
        print(json.dumps({"sha256": write_v3(text), "n_actual": header["n_actual"]}, indent=1))
        return 0
    ok = sha == committed_sha256(OUT_V3)
    print(json.dumps({"rebuild_matches_committed": ok, "sha256": sha}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
