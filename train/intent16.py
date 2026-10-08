"""Training-only intent shortlists, verified against the original guarded 24k draw.

The derived set is never an evaluation source. Both original and derived payloads
are checked; a fresh pointer does not make transformed source items independent.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

VERSION = "intent16-v1"
SOURCES = ("banking77", "clinc150")
SIDE = "intent16_lineage"
SCHEDULE = {"kind": VERSION, "baseline_rows": 7, "intent_rows": 1,
            "sources": list(SOURCES)}
PARENT_CONFIG = "jobs/m2_first_cell/m2_first_cell_rwkv7.json"
BUILD = "sample-24000-seed0-intent16-v1"
POINTER = f"data/manifests/{BUILD}.pointer.json"
ROOT = Path(__file__).resolve().parents[1]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold().replace("_", " ")))


def utterance(record: dict, question: dict) -> str:
    state = record["state"]
    if isinstance(state, str) and state:
        return state
    if isinstance(state, dict):
        for key in ("customer_message", "message", "query", "utterance", "request_text",
                    "user_request", "command", "text", "input", "content"):
            if isinstance(state.get(key), str) and state[key]:
                return state[key]
    ins = question.get("instructions")
    if isinstance(ins, str) and "\n" in ins:
        return ins.split("\n", 1)[1]
    raise ValueError(f"{record['id']}: cannot identify the intent utterance")


def construct(record: dict, original_sha256: str, seed: int = 0) -> dict:
    """Preserve IDs/labels/text; select 16 choices and hash-permute their rendering.

    Single-question intent examples only. Mixed/multi rows remain baseline rows.
    Required negatives (OOS and injection target) count toward the 15 negatives.
    """
    from train.cohort import record_content_hash

    if SIDE in record:
        raise ValueError("intent views cannot be derived recursively")
    out = copy.deepcopy(record)
    side = {"version": VERSION, "original_id": record["id"],
            "original_line_sha256": original_sha256,
            "original_content_sha256": record_content_hash(record),
            "source": record["source"], "source_id": record["source_id"], "role": "baseline"}
    out[SIDE] = side
    if (record["source"] not in SOURCES or len(record["questions"]) != 1
            or record["family"] not in ("intent_banking", "intent_general")):
        return out
    qid, question = next(iter(record["questions"].items()))
    opts = question.get("criteria")
    if question.get("type") != "choice" or not isinstance(opts, dict) or len(opts) < 16:
        raise ValueError(f"{record['id']}: malformed intent choice domain")
    if any(not isinstance(d, str) for d in opts.values()):
        raise ValueError("intent descriptions must be strings")
    gold = record["labels"][qid]
    if gold not in opts:
        raise ValueError("gold is outside the original option domain")
    required = {gold}
    injection = record.get("aug", {}).get("injection")
    if injection:
        if injection.get("question") != qid or injection.get("target") not in opts:
            raise ValueError("injection target/question must be preserved")
        required.add(injection["target"])
    if record["source"] == "clinc150":
        oos = [o for o, d in opts.items() if d == "out of scope: none of the supported intents"]
        if len(oos) != 1:
            raise ValueError("CLINC must have exactly one OOS option")
        required.add(oos[0])
    query = tokens(utterance(record, question))
    key = lambda oid: digest(f"{VERSION}:{seed}:{record['id']}:{oid}")
    def similarity(oid):
        label = tokens(opts[oid])
        return len(query & label) / max(1, len(query | label))
    remaining = set(opts) - required
    hard = sorted(remaining, key=lambda oid: (-similarity(oid), key(oid)))[:7]
    selected = required | set(hard)
    selected.update(sorted(set(opts) - selected, key=key)[:16 - len(selected)])
    order = sorted(selected, key=lambda oid: digest(f"{VERSION}:order:{seed}:{record['id']}:{oid}"))
    out["id"] += f"~view:{VERSION}"
    out["questions"][qid]["criteria"] = {oid: opts[oid] for oid in order}
    side.update(role="intent", question=qid, rendered_option_ids=order)
    return out


def validate_config(cfg: dict) -> None:
    if cfg.get("data_schedule") != SCHEDULE:
        raise ValueError("intent16 requires its declared 7:1 alternating-source schedule")
    if cfg.get("seed") != 0 or cfg.get("n_train_records") != 24000:
        raise ValueError("intent16-v1 is the seed-0 24k original draw")
    if cfg.get("batch_size") != 1 or cfg.get("grad_accum") != 8 or cfg.get("max_k") != 16:
        raise ValueError("intent16 requires batch_size=1, grad_accum=8 and max_k=16")
    if cfg.get("teacher") or cfg.get("kd_alpha") or cfg.get("order_consistency") or cfg.get("order_consistency_weight"):
        raise ValueError("intent16 payloads have no matching KD/order-consistency teachers")


class Schedule:
    """Stateless global row cursor; source permutations cycle only after exhaustion."""
    def __init__(self, items, seed: int):
        self.seed, self.pools, self.cache = seed, {"baseline": [], **{s: [] for s in SOURCES}}, {}
        for index, item in enumerate(items):
            side = item.row.meta.get(SIDE)
            if not isinstance(side, dict) or side.get("version") != VERSION:
                raise ValueError("scheduled items must carry verified intent16 lineage")
            pool = side["source"] if side["role"] == "intent" else "baseline"
            if pool not in self.pools:
                raise ValueError("unknown intent16 schedule source")
            self.pools[pool].append((item.id, index))
        if any(not v for v in self.pools.values()):
            raise ValueError("intent16 schedule requires baseline and both intent pools after K/T filtering")

    def _at(self, pool, offset):
        values = self.pools[pool]
        cycle, position = divmod(offset, len(values))
        cached = self.cache.get(pool)
        if cached is None or cached[0] != cycle:
            order = sorted(values, key=lambda v: digest(f"{VERSION}:schedule:{self.seed}:{pool}:{cycle}:{v[0]}"))
            self.cache[pool] = (cycle, order)
        return self.cache[pool][1][position][1]

    def take(self, cursor: int, count: int) -> list[int]:
        indices = []
        for pos in range(cursor, cursor + count):
            group, slot = divmod(pos, 8)
            if slot == 7:
                indices.append(self._at(SOURCES[group % 2], group // 2))
            else:
                indices.append(self._at("baseline", group * 7 + slot))
        return indices


def _parent(ptr: dict, train_dir: Path, cohort=None):
    """Reverify original files, original draw, cohort/item/text and source hashes."""
    from train.train_set import draw, files, read_pointer, TrainSetMismatch
    from train.data import load_m2_records, read_manifest

    spec = ptr["derivation"]
    if spec["version"] != VERSION or spec.get("seed") != 0 or spec["parent_config"] != PARENT_CONFIG:
        raise TrainSetMismatch("unknown intent16 derivation")
    raw = (ROOT / PARENT_CONFIG).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["parent_config_sha256"]:
        raise TrainSetMismatch("original config drift")
    cfg = json.loads(raw)
    parent_ptr = read_pointer(cfg["train_set"]["pointer"])
    if parent_ptr["_pointer_sha256"] != spec["parent_pointer_sha256"]:
        raise TrainSetMismatch("original pointer drift")
    name = spec["parent_directory"]
    if Path(name).name != name or name in (".", ".."):
        raise TrainSetMismatch("parent directory must be a sibling basename")
    parent_dir = train_dir.parent / name
    _, ids, identity = draw(cfg, parent_dir, cohort=cohort, verify=True)
    if identity["drawn_ids_sha256"] != spec["parent_drawn_ids_sha256"]:
        raise TrainSetMismatch("original training draw drift")
    fs = files(parent_dir, parent_ptr)
    manifest = read_manifest(fs["manifest"])
    want = set(ids)
    records, _ = load_m2_records(fs["records"], manifest, keep=lambda r: r["id"] in want)
    return {r["id"]: r for r in records}, ids, manifest, identity


def _check(records, originals, cohort, guard):
    from train.cohort import record_content_hash, Cohort
    from train.cohort_v3 import cohort_v3, _ensure_kit
    from train.heldout import record_text, scan

    _ensure_kit()
    from harness.heldout import check_record
    for co in (cohort or Cohort(), cohort_v3()):
        for group in (originals, records):
            # Use originals' IDs for source-item guards, including transformed IDs.
            ids = [r.get(SIDE, {}).get("original_id", r["id"]) for r in group]
            content = {i: record_content_hash(r) for i, r in zip(ids, group)}
            co.assert_draw_disjoint(ids, content, records=group)
            co.assert_training_items(ids)
    for group in (originals, records):
        for record in group:
            guard.check(record)
            check_record(record)
        if scan((r["id"], record_text(r)) for r in group)["hits"]:
            raise ValueError("original/derived records carry a held-out D1-B fragment")


def draw_derived(cfg, train_dir, ptr, cohort=None):
    """Load only if every record reconstructs exactly from its verified original."""
    from train.train_set import verify_files, files, TrainSetMismatch
    from train.data import load_m2_records, read_manifest
    from train.cohort import read_content_map, record_content_hash
    from train.run_cell import to_train_record
    from data.build import content_guard

    validate_config(cfg)
    train_dir = Path(train_dir)
    sha = verify_files(train_dir, ptr)
    original, parent_ids, parent_man, identity = _parent(ptr, train_dir, cohort)
    if cfg.get("cohort_sha256") != identity["cohort"]["sha256"]:
        raise TrainSetMismatch("derived config must pin the original draw's cohort")
    expected = [construct(original[i], parent_man[i]) for i in parent_ids]
    fs = files(train_dir, ptr)
    manifest = read_manifest(fs["manifest"])
    rows, loaded = load_m2_records(fs["records"], manifest)
    content = read_content_map(fs["content_map"])
    if (len(rows) != 24000 or len(manifest) != 24000 or ptr["n_records"] != 24000
            or len({r["id"] for r in rows}) != 24000
            or set(manifest) != set(content) or rows != expected):
        raise TrainSetMismatch("derived records/order do not reconstruct from the guarded original draw")
    if any(content[r["id"]] != record_content_hash(r) for r in rows):
        raise TrainSetMismatch("derived content map drift")
    guard, _ = content_guard()
    _check(rows, [original[i] for i in parent_ids], cohort, guard)
    ids = [r["id"] for r in rows]
    identity = {**identity, "parent_training_set": identity, "train_set": BUILD,
                "pointer": ptr["_pointer_path"], "pointer_sha256": ptr["_pointer_sha256"],
                "files_uncompressed_sha256": sha, "n_pool": len(rows), "n_drawn": len(rows),
                "drawn_ids_sha256": digest("\n".join(ids)), "derivation": ptr["derivation"],
                "lines_manifest_verified": loaded["lines_read"]}
    return {r["id"]: to_train_record(r) for r in rows}, ids, identity


def build(parent_dir: Path, out: Path):
    """CPU data construction only. Refuse to replace an existing artifact."""
    from train.train_set import read_pointer
    from train.cohort import record_content_hash
    from data.build import content_guard

    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    cfg_raw = (ROOT / PARENT_CONFIG).read_bytes()
    parent_cfg = json.loads(cfg_raw)
    parent_ptr = read_pointer(parent_cfg["train_set"]["pointer"])
    if parent_dir.resolve().parent != out.resolve().parent:
        raise ValueError("derived and original data directories must be siblings")
    spec = {"version": VERSION, "seed": 0, "parent_config": PARENT_CONFIG,
            "parent_config_sha256": hashlib.sha256(cfg_raw).hexdigest(),
            "parent_pointer_sha256": parent_ptr["_pointer_sha256"], "parent_directory": parent_dir.name,
            "parent_drawn_ids_sha256": "2bbbf1808c0bdd6cf0ba3ecd236cf9a64110c5e429e22f8b8e9cd2df7507556b"}
    originals, ids, manifest, identity = _parent({"derivation": spec}, out)
    records = [construct(originals[i], manifest[i]) for i in ids]
    guard, gi = content_guard()
    _check(records, [originals[i] for i in ids], None, guard)
    out.mkdir(parents=True)
    names = {"records": "records.jsonl.gz", "manifest": f"{BUILD}.manifest.jsonl.gz",
             "content_map": f"{BUILD}.content.jsonl.gz"}
    hashes = {}
    lines = [canonical(r) for r in records]
    rows_by_name = {"records": lines,
        "manifest": [canonical({"id": r["id"], "sha256": digest(line), "source": r["source"],
                                "family": r["family"]}) for r, line in zip(records, lines)],
        "content_map": [canonical({"id": r["id"], "content_sha256": record_content_hash(r)}) for r in records]}
    for key, rows in rows_by_name.items():
        blob = ("\n".join(rows) + "\n").encode("utf-8")
        hashes[key] = hashlib.sha256(blob).hexdigest()
        with gzip.open(out / names[key], "wb") as handle:
            handle.write(blob)
    pointer = {"build": BUILD, "location": f"ssdm-data/m2/build/{BUILD}/", "n_records": len(records),
               "manifest": names["manifest"], "content_map": names["content_map"], "derivation": spec,
               **{f"{k}_uncompressed_sha256": v for k, v in hashes.items()}}
    pointer_path = ROOT / POINTER
    if pointer_path.exists():
        if json.loads(pointer_path.read_text(encoding="utf-8")) != pointer:
            raise ValueError("rebuild does not match the committed derived pointer")
    else:
        pointer_path.write_text(json.dumps(pointer, indent=1) + "\n", encoding="utf-8")
    report = {"build": BUILD, "n_records": len(records), "parent": identity,
              "roles": dict(Counter(r[SIDE]["role"] for r in records)),
              "intent_sources": dict(Counter(r["source"] for r in records if r[SIDE]["role"] == "intent")),
              "injected_intent_views": sum(bool(r.get("aug", {}).get("injection")) for r in records
                                           if r[SIDE]["role"] == "intent"),
              "guard": gi, "checks": "original + derived: cohort id/content/source-item/text; SplitGuard; frozen; D1-B"}
    (out / "build-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = build(args.parent_dir, args.out)
    print(json.dumps({k: report[k] for k in ("build", "n_records", "roles", "intent_sources", "injected_intent_views")}))
