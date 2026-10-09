"""Independent FULL100K compact verifier. Full scan requires coordinator pins and --authorized-full-scan."""
from __future__ import annotations
import argparse
import builtins
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(r"C:\Users\apumu\research\CHLabs\ssdm-wt\agent-c-multirecord-synth")
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
SOURCE_MANIFEST = DATA / "c-multirecord-synth-100k-v2-cached/train_set/c-multirecord-synth-100k-v2.manifest.jsonl.gz"
SOURCE_MANIFEST_SHA = "c031940a0bdd3d81f4b09e2b082383fd52ca1643f20034f75fdefbdfd10601e8"
GENERATOR_SHA = "a5f49e431361671687bab453f4b5f7b280d8d5dfc43530b7b9b6fcc7efba669b"
POOLS_SHA = "a8cd7ecb63d876a541037fa6ed530fe985da1bd4d848b8f5ae8ab0ea432380aa"
TRAINER_COMMIT = "9c3e4350ccb41bec1385d58b9fe7002184f69626"
NAMES = ("records", "manifest", "content_map")
N = 100000

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def bootstrap():
    require(file_sha(ROOT / "data/c_multirecord_synth.py") == GENERATOR_SHA, "generator pin changed")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name in ("train.trainer", "heads.model"):
            raise RuntimeError("model import forbidden: " + name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded
    sys.path[:0] = [str(ROOT), str(DATA / "decision-index")]
    from data.c_multirecord_synth import canonical, make_record
    from train.cohort import record_content_hash
    from harness.heldout import check_record
    return canonical, make_record, record_content_hash, check_record

def inverse_full_record(row, old, pools):
    """Independent inverse only: never call the production transform during verification."""
    parent_id = old["id"]
    require(row["id"] == parent_id + "~view:compact-single-token-v1", "compact ID/origin mismatch")
    old_qids, new_qids = sorted(old["questions"]), sorted(row["questions"])
    require(len(old_qids) == 7 and new_qids == list("abcdefg"), "question count or names changed")
    qmap = dict(zip(old_qids, new_qids))
    qback = {value: key for key, value in qmap.items()}
    choices = [(q, m) for q, m in old["aug"]["question_roles"].items() if m["kind"] == "top_profile"]
    require(len(choices) == 1, "parent choice-role count changed")
    choice, meta = choices[0]
    style = meta["id_style"]
    old_ids = sorted(old["questions"][choice]["criteria"])
    new_ids = sorted(row["questions"][qmap[choice]]["criteria"])
    k = len(old_ids)
    require(2 <= k <= 77 and new_ids == [entry["id"] for entry in pools["pools"][style]][:k], "option pool differs")
    view = {"version": 1, "source_record_id": parent_id, "pool_sha256": pools["pool_ids_sha256"][style]}
    require(row["aug"]["compact_view"] == view, "compact origin metadata differs")
    oback = dict(zip(new_ids, old_ids))
    restored = deepcopy(row)
    restored["id"] = parent_id
    restored["questions"] = {q: deepcopy(row["questions"][qmap[q]]) for q in old_qids}
    restored["questions"][choice]["criteria"] = {oback[oid]: value for oid, value in restored["questions"][choice]["criteria"].items()}
    restored["labels"] = {q: row["labels"][qmap[q]] for q in old_qids}
    restored["labels"][choice] = oback[restored["labels"][choice]]
    restored["aug"].pop("compact_view")
    restored["aug"]["question_roles"] = {q: deepcopy(row["aug"]["question_roles"][qmap[q]]) for q in old_qids}
    for key in ("case_exact", "multi_select"):
        restored["aug"][key]["question_ids"] = [qback[q] for q in row["aug"][key]["question_ids"]]
    require(restored == old, "COMPLETE inverse record differs from regenerated parent")
    pos = new_ids.index(row["labels"][qmap[choice]])
    require(pos == meta["gold_position"], "canonical gold position changed")
    return k, style, pos

def self_test(canonical, make_record, check_record, pools):
    # Production transform only supplies tiny fixtures; the independent inverse
    # is then challenged with changes to meaning, gold, and nonvisible metadata.
    from data.c_multirecord_compact import compact_record
    fixtures = 0
    rejected = 0
    for k in (2, 40, 77):
        for style in range(3):
            old = make_record(20261008, k - 2 + 76 * style + 228 * 71, revision=2)
            row = compact_record(old, pools)
            require(inverse_full_record(row, old, pools)[0] == k, "fixture K mismatch")
            check_record(row)
            fixtures += 1
            for mutation in ("state", "source_id", "instruction", "label", "role"):
                bad = deepcopy(row)
                qid = sorted(bad["questions"])[0]
                if mutation == "state":
                    bad["state"] = {"changed": True}
                elif mutation == "source_id":
                    bad["source_id"] += "-changed"
                elif mutation == "instruction":
                    bad["questions"][qid]["instructions"] += " changed"
                elif mutation == "label":
                    choice = next(q for q, m in bad["aug"]["question_roles"].items() if m["kind"] == "top_profile")
                    ids = sorted(bad["questions"][choice]["criteria"])
                    bad["labels"][choice] = next(oid for oid in ids if oid != bad["labels"][choice])
                else:
                    bad["aug"]["question_roles"][qid]["unexpected"] = True
                try:
                    inverse_full_record(bad, old, pools)
                except (ValueError, KeyError):
                    rejected += 1
                else:
                    raise ValueError("inverse accepted mutated " + mutation)
    require("torch" not in sys.modules, "Torch imported")
    return {"status": "self_test_passed", "fixtures": fixtures, "semantic_or_metadata_mutations_rejected": rejected, "full_output_scan_ran": False, "torch_imported": False}

def verify(args, canonical, make_record, content_hash, check_record, pools):
    pkg, pointer_path = Path(args.package), Path(args.pointer)
    pointer_raw = pointer_path.read_bytes()
    summary_raw = (pkg / "summary.json").read_bytes()
    require(sha(pointer_raw) == args.pointer_sha256, "completion pointer pin mismatch")
    require(sha(summary_raw) == args.summary_sha256, "completion summary pin mismatch")
    pointer, summary = json.loads(pointer_raw), json.loads(summary_raw)
    require(summary["pointer"] == pointer and summary["pointer_sha256"] == args.pointer_sha256, "receipt/pointer disagree")
    require((pkg / "train_set/pointer.json").read_bytes() == pointer_raw, "external pointer differs")
    require(DATA.parent / pointer["location"] == pkg / "train_set", "pointer location differs")
    receipt = ROOT / pointer["receipt"]
    require(receipt.is_file() and receipt.read_bytes() == summary_raw, "repo receipt missing or differs")
    require(summary["records"] == pointer["n_records"] == N and summary["questions"] == 7 * N, "completed full counts differ")
    require(summary["cohort_mode"] == "all_worlds" and summary["source_validation_passes"] == 2 and summary["retained_raw_source_payload_bytes"] == 0, "full streaming mode not qualified")
    require(summary["trainer_commit"] == TRAINER_COMMIT and summary["pools_sha256"] == POOLS_SHA, "trainer/pools differ")
    for field, pool_field in (("trainer_source_sha256", "trainer_source_sha256"), ("tokenizer_sha256", "tokenizer_sha256"), ("codebook_metadata_sha256", "metadata_sha256"), ("generator_source_sha256", "generator_source_sha256")):
        require(summary[field] == pools[pool_field], "qualification pin differs: " + field)
    require(summary["layout"] == "g1v2" and summary["templates"] == ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"], "template/layout differs")
    for field in ("all_worlds_retained", "inverse_alpha_semantics_passed", "verify_record_check_record_passed", "direct_trainer_row_equality_passed", "original_full_c_source_preserved"):
        require(summary[field] is True, "producer qualification failed: " + field)
    require(summary["torch_imported"] is False and summary["b_inputs_read"] is False, "producer forbidden input/import")
    producer_max = {key: summary["lengths"][key]["max"] for key in ("serialized", "v2f", "v3w")}
    require(all(type(n) is int and 1 <= n <= 2048 for n in producer_max.values()), "producer token bound failed")
    require(all(summary["lengths"][key]["n"] == N for key in producer_max), "producer measurements incomplete")
    # Read only previously verified synthetic source manifest hash metadata.
    parents = [None] * N
    source_digest = hashlib.sha256()
    source_rows = 0
    parent_pattern = re.compile(r"m2:c_multirecord_synth_v2:20261008:(\d{6})\Z")
    with gzip.open(SOURCE_MANIFEST, "rb") as stream:
        for line in stream:
            source_digest.update(line)
            entry = json.loads(line)
            match = parent_pattern.fullmatch(entry["id"])
            require(match is not None, "source manifest ID outside pinned cohort")
            index = int(match[1])
            require(0 <= index < N and parents[index] is None, "source manifest duplicate/outside index")
            parents[index] = bytes.fromhex(entry["sha256"])
            source_rows += 1
    require(source_digest.hexdigest() == SOURCE_MANIFEST_SHA and source_rows == N and all(parents), "source manifest metadata pin/count failed")
    paths = {"records": pkg / "train_set/records.jsonl.gz", "manifest": pkg / "train_set" / pointer["manifest"], "content_map": pkg / "train_set" / pointer["content_map"]}
    compressed = {name: file_sha(path) for name, path in paths.items()}
    require(compressed == summary["compressed_sha256"], "output gzip pins failed")
    digests = {name: hashlib.sha256() for name in paths}
    counts, per_k, styles, strata = Counter(), Counter(), Counter(), Counter()
    ids, worlds, contents = set(), set(), set()
    questions = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, "rb")) for path in paths.values()]
        for position, lines in enumerate(zip_longest(*streams)):
            require(all(line is not None and line.endswith(b"\n") for line in lines), "output alignment/newline failed")
            for name, line in zip(paths, lines):
                digests[name].update(line)
                counts[name] += 1
            raw, manifest_raw, content_raw = lines
            row, manifest, content = (json.loads(line) for line in lines)
            rid, world = row["id"], row["source_id"]
            id_digest, world_digest = hashlib.sha256(rid.encode()).digest(), hashlib.sha256(world.encode()).digest()
            require(id_digest not in ids and world_digest not in worlds, "duplicate output ID/world")
            ids.add(id_digest); worlds.add(world_digest)
            require(manifest == {"id": rid, "sha256": sha(raw[:-1]), "source": row["source"], "family": row["family"]}, "output manifest row differs")
            ch = content_hash(row)
            ch_bytes = bytes.fromhex(ch)
            require(content == {"id": rid, "content_sha256": ch} and ch_bytes not in contents, "output content mismatch/duplicate")
            contents.add(ch_bytes)
            require(row["aug"]["seed"] == 20261008 and row["aug"]["record_index"] == position and position < N, "source seed/index/order differs")
            old = make_record(20261008, position, revision=2)
            require(hashlib.sha256(canonical(old)).digest() == parents[position], "regenerated parent byte hash differs")
            k, style, gold = inverse_full_record(row, old, pools)
            per_k[k] += 1; styles[style] += 1; strata[k, style, gold] += 1
            check_record(row)
            questions += len(row["questions"])
            if (position + 1) % 25000 == 0:
                print(f"independent full compact scan: {position + 1}/{N}", flush=True)
    uncompressed = {name: digest.hexdigest() for name, digest in digests.items()}
    require(all(uncompressed[name] == pointer[name + "_uncompressed_sha256"] == summary["uncompressed_sha256"][name] for name in paths), "output raw pins failed")
    require(dict(counts) == {name: N for name in NAMES} and questions == 7 * N, "actual complete output count failed")
    require(per_k == {k: 1316 if k <= 61 else 1315 for k in range(2, 78)}, "all-K source distribution differs")
    expected_strata = {(k, style, gold) for k in range(2, 78) for style in ("letters", "words", "opaque") for gold in range(k)}
    require(set(strata) == expected_strata, "all K/style/gold positions not covered")
    require({str(k): value for k, value in per_k.items()} == summary["per_k_records"] and dict(styles) == summary["per_style_records"], "producer distribution differs")
    require("torch" not in sys.modules, "Torch imported")
    report = {"schema": "ssdm.c_multirecord_full_compact_independent_verification.v1", "status": "verified", "package": str(pkg), "pointer_path": str(pointer_path), "pointer_sha256": sha(pointer_raw), "summary_sha256": sha(summary_raw), "receipt_path": str(receipt), "counts": dict(counts), "questions": questions, "per_k_records": dict(per_k), "per_style_records": dict(styles), "all_gold_position_strata": len(strata), "min_stratum_count": min(strata.values()), "max_stratum_count": max(strata.values()), "compressed_sha256": compressed, "uncompressed_sha256": uncompressed, "source_manifest_uncompressed_sha256": source_digest.hexdigest(), "source_metadata_rows": source_rows, "regenerated_parents_hash_matched": len(ids), "complete_inverse_records": len(ids), "distinct_worlds": len(worlds), "distinct_contents": len(contents), "check_record_passed": len(ids), "refused": 0, "producer_pinned_trainer_flags_verified": True, "producer_max_tokens": producer_max, "pools_sha256": POOLS_SHA, "path_issues": [], "independent_output_scans": 1, "full_source_record_rescans": 0, "b_inputs_read": False, "protected_rows_read": False, "torch_imported": False, "retokenized": False, "builder_commit": pointer["builder"]["commit"], "verifier_source_sha256": file_sha(__file__)}
    report_dir = Path(tempfile.mkdtemp(prefix="c-full100k-compact-final-verification-"))
    report_path = report_dir / "verification.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(report, verifier_json=str(report_path)), indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorized-full-scan", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--pointer")
    parser.add_argument("--pointer-sha256")
    parser.add_argument("--summary-sha256")
    args = parser.parse_args()
    if args.self_test == args.authorized_full_scan:
        parser.error("choose self-test or coordinator-authorized full scan")
    if args.authorized_full_scan and any(not getattr(args, name) for name in ("package", "pointer", "pointer_sha256", "summary_sha256")):
        parser.error("full scan requires all completion paths and independent byte pins")
    canonical, make_record, content_hash, check_record = bootstrap()
    pools_raw = (ROOT / "data/c_multirecord_compact_pools.json").read_bytes()
    require(sha(pools_raw) == POOLS_SHA, "pool byte pin changed")
    pools = json.loads(pools_raw)
    if args.self_test:
        print(json.dumps(self_test(canonical, make_record, check_record, pools), indent=2))
    else:
        verify(args, canonical, make_record, content_hash, check_record, pools)

if __name__ == "__main__":
    main()
