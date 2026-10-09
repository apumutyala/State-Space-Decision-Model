"""Lean stdlib byte/coverage verification; source RECORDS are never opened."""
import argparse
from collections import Counter
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMES = ("records", "manifest", "content_map")
POINTER_PIN = "2c4659464e3ddaab28145b45628c59081964300f64d855f402edc1f6e1b81618"
SOURCE_POINTER_PIN = "531faa15a6a54e58cd14f44c6af7a5bf8edd4046ca5a16a54ae7f84db26172cc"
SOURCE_PROOF_PIN = "8e6e780bee43a53391fd04403c79909a3949e7a661a3a3ed54e659c98c7b7c79"
ID_PATTERN = re.compile(r"m2:c_multirecord_synth_v2:[0-9]+:([0-9]{6})(?:~view:compact-single-token-v1)?\Z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, pin):
    raw = path.read_bytes()
    require(sha(raw) == pin, f"metadata byte pin differs: {path}")
    return json.loads(raw)


def scheduled_bucket(rid):
    # The fixed generator schedule is data/c_multirecord_synth.py:260-264;
    # compact alpha views preserve the record index and gold position.
    match = ID_PATTERN.fullmatch(rid)
    require(match is not None, "source ID outside the verified C v2 schedule")
    index = int(match.group(1))
    k = 2 + index % 76
    style = ("letters", "words", "opaque")[(index // 76) % 3]
    return k, style, (index // 228) % k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("C:/Users/apumu/research/CHLabs/ssdm-data/c-multirecord-7k-v2-allk-quota"))
    parser.add_argument("--source-pointer", type=Path, default=ROOT / "data/manifests/c-multirecord-13575-v2-allk-admitted.pointer.json")
    parser.add_argument("--source-proof", type=Path, default=ROOT / "receipts/2026-10-08-C-multirecord-synth/c-multirecord-13575-v2-allk-admitted/independent-verification.json")
    parser.add_argument("--out", type=Path, default=HERE / "independent-verification.json")
    args = parser.parse_args()
    summary_path = args.artifact / "summary.json"
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    pointer = pinned_json(args.artifact / "train_set/pointer.json", POINTER_PIN)
    require(pointer == summary["pointer"] and summary["pointer_sha256"] == POINTER_PIN, "output pointer/summary mismatch")
    source_pointer = pinned_json(args.source_pointer, SOURCE_POINTER_PIN)
    source_proof = pinned_json(args.source_proof, SOURCE_PROOF_PIN)
    require(summary["source_pointer_sha256"] == SOURCE_POINTER_PIN and summary["source_verification_sha256"] == SOURCE_PROOF_PIN, "source provenance mismatch")
    require(source_proof["status"] == "verified" and source_proof["b_inputs_read"] is False and source_proof["protected_rows_read"] is False, "unverified source proof")
    require(source_proof["pointer_sha256"] == SOURCE_POINTER_PIN, "source proof/pointer mismatch")
    require(source_proof["uncompressed_sha256"] == summary["source_uncompressed_sha256"], "source hash provenance mismatch")
    source = Path(summary["source_directory"])
    sidecars, source_hashes = {}, {}
    for name in ("manifest", "content_map"):
        require(Path(source_pointer[name]).name == source_pointer[name], "source sidecar is not an immediate filename")
        path = source / source_pointer[name]
        require(sha(path.read_bytes()) == source_proof["compressed_sha256"][name], "source compressed sidecar differs")
        digest, rows = hashlib.sha256(), {}
        with gzip.open(path, "rb") as stream:
            for raw in stream:
                require(raw.endswith(b"\n"), "unterminated source sidecar")
                digest.update(raw)
                rid = json.loads(raw)["id"]
                require(rid not in rows, "duplicate source sidecar ID")
                rows[rid] = raw
        require(len(rows) == source_pointer["n_records"] == 13575, "source sidecar count mismatch")
        require(digest.hexdigest() == source_pointer[name + "_uncompressed_sha256"] == source_proof["uncompressed_sha256"][name], "source sidecar raw hash mismatch")
        source_hashes[name], sidecars[name] = digest.hexdigest(), rows
    require(set(sidecars["manifest"]) == set(sidecars["content_map"]), "source sidecar IDs differ")
    source_groups = {scheduled_bucket(rid)[:2] for rid in sidecars["manifest"]}
    require(len(source_groups) == summary["source_groups"] == 204, "source K/style group mismatch")
    selected_ids, worlds, contents = [], set(), set()
    groups, per_k, per_style, gold_strata, cases, scores = Counter(), Counter(), Counter(), set(), Counter(), Counter()
    names = {"records": "records.jsonl.gz", **{name: pointer[name] for name in NAMES[1:]}}
    hashes = {name: hashlib.sha256() for name in NAMES}
    compressed = {name: sha((args.artifact / "train_set" / filename).read_bytes()) for name, filename in names.items()}
    require(compressed == summary["compressed_sha256"], "output compressed hash mismatch")
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(args.artifact / "train_set" / names[name], "rb")) for name in NAMES]
        for raw in zip_longest(*streams):
            require(all(line is not None and line.endswith(b"\n") for line in raw), "output stream count/newline mismatch")
            for name, line in zip(NAMES, raw):
                hashes[name].update(line)
            record, manifest, content = map(json.loads, raw)
            rid = record["id"]
            require(rid in sidecars["manifest"], "output record absent from source")
            require(raw[1] == sidecars["manifest"][rid] and raw[2] == sidecars["content_map"][rid], "output sidecar bytes changed")
            require(manifest["id"] == content["id"] == rid and sha(raw[0][:-1]) == manifest["sha256"], "output record bytes differ from verified source manifest")
            require(record["source"] == "synthetic_c_multirecord_v2" and record["source_split"] == "synthetic", "output is not synthetic C v2")
            require(len(record["questions"]) == 7 and set(record["questions"]) == set(record["labels"]), "whole record lost questions/labels")
            world, digest = record["source_id"], content["content_sha256"]
            require(world not in worlds and digest not in contents, "duplicate output world/content")
            worlds.add(world); contents.add(digest); selected_ids.append(rid)
            roles = record["aug"]["question_roles"]
            choices = [(qid, meta) for qid, meta in roles.items() if meta["kind"] == "top_profile"]
            require(len(choices) == 1, "output choice metadata missing")
            qid, meta = choices[0]
            bucket = (meta["k"], meta["id_style"], meta["gold_position"])
            require(bucket == scheduled_bucket(rid), "output K/style/position differs from fixed source schedule")
            ids = sorted(record["questions"][qid]["criteria"])
            require(len(ids) == bucket[0] and ids.index(record["labels"][qid]) == bucket[2], "canonical choice structure differs")
            groups[bucket[:2]] += 1; per_k[bucket[0]] += 1; per_style[bucket[1]] += 1; gold_strata.add(bucket)
            for qid, role in roles.items():
                if role["kind"] == "case_exact":
                    cases[record["labels"][qid]] += 1
                elif role["kind"] == "joint_score":
                    scores[record["labels"][qid]] += 1
    actual_raw = {name: digest.hexdigest() for name, digest in hashes.items()}
    require(actual_raw == {name: pointer[name + "_uncompressed_sha256"] for name in NAMES}, "output raw hash mismatch")
    require(selected_ids == summary["selected_ids"] and len(set(selected_ids)) == 7000, "selected ID order/count differs")
    require(len(selected_ids) == pointer["n_records"] == summary["records"] == 7000 and summary["questions"] == 49000, "output count mismatch")
    expected_quotas = {(entry["k"], entry["style"]): entry["records"] for entry in summary["group_quotas"]}
    require(dict(groups) == expected_quotas and set(groups) == source_groups and len(groups) == summary["selected_groups"], "group quotas/coverage differ")
    require(set(per_k) == set(range(2, 78)) and dict(per_k) == {int(k): v for k, v in summary["per_k_records"].items()}, "all-K coverage/counts differ")
    require(len(gold_strata) == summary["selected_gold_strata"] == 4772, "gold stratum count differs")
    require(summary["token_bounds"] == source_proof["token_bounds_verified"], "inherited trainer bounds changed")
    report = {"schema": "ssdm.c_multirecord_quota_independent_verification.v1", "status": "verified",
              "artifact": str(args.artifact.resolve()), "summary_sha256": sha(summary_raw), "pointer_sha256": POINTER_PIN,
              "source_pointer_sha256": SOURCE_POINTER_PIN, "source_verification_sha256": SOURCE_PROOF_PIN,
              "records": 7000, "questions": 49000, "unique_worlds": len(worlds), "unique_content": len(contents),
              "all_k": sorted(per_k), "source_groups": len(source_groups), "selected_groups": len(groups),
              "selected_gold_strata": len(gold_strata), "group_quotas_match": True, "selected_ids_order_match": True,
              "source_manifest_record_byte_matches": 7000, "exact_source_sidecar_line_matches": {name: 7000 for name in sidecars},
              "compressed_sha256": compressed, "uncompressed_sha256": actual_raw, "source_sidecar_uncompressed_sha256": source_hashes,
              "per_k_records": dict(sorted(per_k.items())), "per_style_records": dict(sorted(per_style.items())),
              "case_exact_labels": dict(cases), "joint_score_labels": dict(scores),
              "token_bounds": summary["token_bounds"], "token_bound_scope": "inherited verified complete-source upper bounds; not retokenized",
              "source_record_text_read": False, "b_inputs_read": False, "protected_rows_read": False,
              "retokenized": False, "semantic_reaudit": False, "gpu_used": False,
              "verifier_sha256": sha(Path(__file__).read_bytes())}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "records", "questions", "selected_groups", "selected_gold_strata", "per_style_records", "joint_score_labels")}))


if __name__ == "__main__":
    main()
