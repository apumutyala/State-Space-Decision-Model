"""Replace only B's pinned C v1 quota with byte-preserved, seeded C v2 records."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import heapq
from itertools import zip_longest
import json
from pathlib import Path
import subprocess

from .c_multirecord_subset import _pinned_json, _rank, _sha, stratum
from .c_multirecord_synth import GENERATOR_SPEC, _sha_file, canonical, verify_record

V1, V2 = "synthetic_c_multirecord_v1", "synthetic_c_multirecord_v2"
FAMILY = "synthetic_multirecord_telemetry"
PAYLOAD_LIMIT = 80 * 1024 * 1024
NAMES = ("records", "manifest", "content_map")
QUARANTINED_B_POINTERS = frozenset({"10206b05e976bd0a043653233abd95f77575c4ddaed6b1f309a1dc97be935895"})


def _c_version(record):
    source, family = record["source"], record["family"]
    generator = record.get("aug", {}).get("generator", "")
    is_c = (source.startswith("synthetic_c_multirecord") or family.startswith("synthetic_multirecord")
            or generator.startswith("c_multirecord") or record["id"].startswith("m2:c_multirecord"))
    if not is_c:
        return None
    expected_generator = {V1: "c_multirecord_synth_v1", V2: "c_multirecord_synth_v2"}.get(source)
    if source not in (V1, V2) or family != FAMILY or generator != expected_generator:
        raise ValueError("unknown C source version or family")
    return source


def _triples(directory, pointer):
    """Stream aligned records/sidecars; validate line identities and full hashes on exhaustion."""
    paths = [Path(directory) / "records.jsonl.gz"]
    for name in NAMES[1:]:
        if Path(pointer[name]).name != pointer[name]:
            raise ValueError("sidecars must be immediate filenames")
        paths.append(Path(directory) / pointer[name])
    hashes = [hashlib.sha256() for _ in NAMES]
    n = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, "rb")) for path in paths]
        for raw in zip_longest(*streams):
            if any(line is None or not line.endswith(b"\n") for line in raw):
                raise ValueError("source sidecar count/newline mismatch")
            for digest, line in zip(hashes, raw):
                digest.update(line)
            record, manifest, content = map(json.loads, raw)
            if (manifest.get("id") != record["id"] or manifest.get("sha256") != _sha(raw[0][:-1])
                    or manifest.get("source") != record["source"] or manifest.get("family") != record["family"]
                    or content.get("id") != record["id"] or len(content.get("content_sha256", "")) != 64):
                raise ValueError("record bytes or sidecar identity mismatch")
            n += 1
            yield record, content["content_sha256"], raw
    if n != pointer["n_records"] or any(d.hexdigest() != pointer[name + "_uncompressed_sha256"] for name, d in zip(NAMES, hashes)):
        raise ValueError("source count or uncompressed byte pin mismatch")


def _census(summary_path, summary_sha, census_path, pointer, pointer_sha):
    summary = _pinned_json(summary_path, summary_sha)
    if (summary.get("schema") != "ssdm.c_multirecord_trainer_census.v1"
            or summary.get("source_pointer_sha256") != pointer_sha
            or summary.get("source_records_sha256") != pointer["records_uncompressed_sha256"]
            or summary.get("records") != pointer["n_records"] or summary.get("layout") != "g1v2"
            or summary.get("length_scope") != "prompt plus longest continuation"
            or any(summary.get("lengths", {}).get(name, {}).get("n") != pointer["n_records"] for name in ("v2f", "v3w"))):
        raise ValueError("completed C trainer census does not bind the pinned pool/layout")
    path = Path(census_path) if census_path else Path(summary_path).parent / "census.jsonl.gz"
    if _sha_file(path) != summary["census_compressed_sha256"]:
        raise ValueError("C census compressed byte pin mismatch")
    lengths = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (set(row) != {"id", "v2f", "v3w"} or not isinstance(row["id"], str) or not row["id"]
                    or row["id"] in lengths or any(type(row[name]) is not int or row[name] <= 0 for name in ("v2f", "v3w"))):
                raise ValueError("C census requires unique IDs and positive scalar lengths without errors")
            lengths[row["id"]] = row["v2f"], row["v3w"]
    if len(lengths) != pointer["n_records"]:
        raise ValueError("C census record count mismatch")
    return summary, path, lengths


def _select_c(source, pointer, census_lengths, quota, seed, other_ids=(), other_contents=()):
    from train.cohort import record_content_hash

    heap, c_ids, c_contents, payload_bytes = [], set(), set(), 0
    eligible, excluded = Counter(), Counter()
    for record, content, raw in _triples(source, pointer):
        rid, content_digest = record["id"], bytes.fromhex(content)
        if (_c_version(record) != V2 or record.get("source_split") != "synthetic"):
            raise ValueError("fresh pool contains a non-v2 synthetic C record")
        verify_record(record)
        bucket = stratum(record)
        if record_content_hash(record) != content:
            raise ValueError("C normalized content differs from sidecar")
        if rid in c_ids or content_digest in c_contents or rid in other_ids or content_digest in other_contents:
            raise ValueError("duplicate C ID/content or overlap with retained source")
        c_ids.add(rid)
        c_contents.add(content_digest)
        if rid not in census_lengths:
            raise ValueError("C pool record missing from pinned census")
        if max(census_lengths[rid]) > 2048:
            excluded[bucket[:2]] += 1
            continue
        eligible[bucket[:2]] += 1
        candidate = (-int.from_bytes(_rank(seed, "refresh-c", rid), "big"), rid, raw)
        cost = sum(map(len, raw))
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
            payload_bytes += cost
        elif candidate > heap[0]:
            old = heapq.heapreplace(heap, candidate)
            payload_bytes += cost - sum(map(len, old[2]))
        if payload_bytes > PAYLOAD_LIMIT:
            raise MemoryError("selected C raw payload exceeds 80 MiB")
    if set(census_lengths) != c_ids:
        raise ValueError("C census IDs are not exactly the source pool IDs")
    if len(heap) != quota:
        raise ValueError(f"insufficient whole-record 2k C capacity: {sum(eligible.values())} eligible, {quota} required")
    selected_max = {name: max(census_lengths[rid][i] for _, rid, _ in heap) for i, name in enumerate(("v2f", "v3w"))}
    return heap, payload_bytes, eligible, excluded, selected_max


def refresh(b_source, b_pointer, b_pointer_sha256, b_report, b_report_sha256,
            b_exclusion, b_exclusion_sha256, c_source, c_pointer, c_pointer_sha256,
            c_summary, c_summary_sha256, c_census_summary, c_census_summary_sha256, out,
            *, c_census_file=None, quota=7000, expected_records=229891,
            seed=20261009, data_root=None, pointer_out=None):
    from .config import DATA_ROOT
    from harness.heldout import check_generator
    from train.cohort import record_content_hash

    out, data_root = Path(out), Path(data_root or DATA_ROOT).resolve()
    relative = out.resolve().relative_to(data_root)
    pointer_out = Path(pointer_out) if pointer_out else None
    if out.exists() or (pointer_out and pointer_out.exists()):
        raise FileExistsError("refusing existing refresh output or pointer")
    if any(type(x) is not int or x < 1 for x in (quota, expected_records)) or type(seed) is not int or seed < 0:
        raise ValueError("positive integer sizes and nonnegative seed required")
    bp = _pinned_json(b_pointer, b_pointer_sha256)
    if b_pointer_sha256 in QUARANTINED_B_POINTERS:
        raise ValueError("quarantined B pointer; refuse before any source file hash or record reads")
    br = _pinned_json(b_report, b_report_sha256)["select"]
    if any(source.lower().startswith("nli4ct") and n > 0 for source, n in br["by_source"].items()):
        raise ValueError("quarantined B mix: positive NLI4CT source; refuse before any B record reads")
    be = _pinned_json(b_exclusion, b_exclusion_sha256)
    cp = _pinned_json(c_pointer, c_pointer_sha256)
    cs = _pinned_json(c_summary, c_summary_sha256)
    if (bp["n_records"] != expected_records or br["n_records"] != expected_records
            or br["quotas"]["P3:extra"] != quota or br["by_source"].get(V1) != quota
            or any(br[name + "_uncompressed_sha256"] != bp[name + "_uncompressed_sha256"] for name in NAMES)
            or be.get("pass") is not True or be.get("n_rows") != expected_records or be.get("failures") != {}
            or str(be["pointer"]).replace("\\", "/").split("/")[-1] != Path(b_pointer).name):
        raise ValueError("B build/exclusion receipts do not bind the pinned source and quota")
    if (cs["schema"] != "ssdm.c_multirecord_subset.v1" or cs["pointer"] != cp
            or cs["pointer_sha256"] != c_pointer_sha256 or cs["records"] != cp["n_records"]
            or cp["n_records"] < quota or cs.get("record_bytes_preserved") is not True
            or cs.get("generator_spec_check_passed") is not True):
        raise ValueError("C receipt does not bind the pinned completed v2 pool")
    census_summary, census_path, census_lengths = _census(c_census_summary, c_census_summary_sha256,
                                                         c_census_file, cp, c_pointer_sha256)
    check_generator({**GENERATOR_SPEC, "name": "c_multirecord_synth_v2", "version": 2})
    # First pass proves the entire reviewed B input before creating any output.
    ids, contents, by_source = set(), set(), Counter()
    kept_hashes = [hashlib.sha256() for _ in NAMES]
    removed = kept = 0
    for record, content, raw in _triples(b_source, bp):
        rid, digest = record["id"], bytes.fromhex(content)
        if rid in ids or digest in contents:
            raise ValueError("duplicate B source ID or normalized content")
        ids.add(rid)
        contents.add(digest)
        by_source[record["source"]] += 1
        version = _c_version(record)
        if version is not None:
            if version != V1:
                raise ValueError("B source already contains another C version")
            removed += 1
        else:
            kept += 1
            for digest, line in zip(kept_hashes, raw):
                digest.update(line)
    if removed != quota or dict(by_source) != br["by_source"]:
        raise ValueError("actual B source mix or C removal quota differs from receipt")
    kept_proof = [digest.hexdigest() for digest in kept_hashes]
    heap, payload_bytes, eligible_by_bucket, excluded_by_bucket, selected_max = _select_c(
        c_source, cp, census_lengths, quota, seed, ids, contents)
    del ids, contents
    # Recheck metadata pins before writing; second B pass detects any data change.
    for path, pin in ((b_pointer, b_pointer_sha256), (b_report, b_report_sha256), (b_exclusion, b_exclusion_sha256),
                      (c_pointer, c_pointer_sha256), (c_summary, c_summary_sha256), (c_census_summary, c_census_summary_sha256)):
        if _sha(Path(path).read_bytes()) != pin:
            raise ValueError("input metadata changed during refresh")
    if _sha_file(census_path) != census_summary["census_compressed_sha256"]:
        raise ValueError("C census changed during selection")
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    names = {"records": "records.jsonl.gz", "manifest": out.name + ".manifest.jsonl.gz", "content_map": out.name + ".content.jsonl.gz"}
    hashes, kept_again = [hashlib.sha256() for _ in NAMES], [hashlib.sha256() for _ in NAMES]
    label_counts = {"case_exact": Counter(), "joint_score": Counter(), "multi_select_member": Counter()}
    new_sources, per_k, shapes, fresh_strata = by_source.copy(), Counter(), Counter(), set()
    new_sources.pop(V1)
    new_sources[V2] = quota
    n_written = n_kept = 0
    with ExitStack() as stack:
        streams = []
        for name in NAMES:
            handle = stack.enter_context((train_dir / names[name]).open("xb"))
            streams.append(stack.enter_context(gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0)))
        def emit(raw):
            for stream, digest, line in zip(streams, hashes, raw):
                stream.write(line)
                digest.update(line)
        for record, _, raw in _triples(b_source, bp):
            if _c_version(record) is None:
                emit(raw)
                for digest, line in zip(kept_again, raw):
                    digest.update(line)
                n_kept += 1
                n_written += 1
        if n_kept != kept or [d.hexdigest() for d in kept_again] != kept_proof:
            raise ValueError("retained B record/sidecar bytes changed between passes")
        for _, _, raw in sorted(heap, key=lambda entry: (-entry[0], entry[1])):
            record = json.loads(raw[0])
            verify_record(record)
            bucket = stratum(record)
            fresh_strata.add(bucket)
            per_k[bucket[0]] += 1
            shapes[record["aug"]["schema_shape"]] += 1
            for qid, meta in record["aug"]["question_roles"].items():
                if meta["kind"] in label_counts:
                    label_counts[meta["kind"]][record["labels"][qid]] += 1
            emit(raw)
            n_written += 1
    if n_written != expected_records:
        raise ValueError("refresh changed total record count")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    pointer = {**bp, "build": out.name + " (python -m data.c_multirecord_refresh)",
               "location": (Path(data_root.name) / relative / "train_set").as_posix() + "/",
               "manifest": names["manifest"], "content_map": names["content_map"],
               "excludes_cohort": "retained B bytes inherit pinned B exclusion receipt; fresh C is synthetic; consuming peer review/guard still required",
               "builder": {"module": "data.c_multirecord_refresh", "commit": commit},
               "receipt": f"receipts/2026-10-08-C-multirecord-synth/{out.name}/summary.json",
               **{name + "_uncompressed_sha256": d.hexdigest() for name, d in zip(NAMES, hashes)}}
    pointer_raw = json.dumps(pointer, indent=1).encode() + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    if pointer_out:
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as handle:
            handle.write(pointer_raw)
    report = {"schema": "ssdm.c_multirecord_refresh.v1", "records": n_written, "removed_c_v1": removed, "added_c_v2": quota, "retained_non_c": kept,
              "seed": seed, "by_source": dict(new_sources), "fresh_c_label_counts": {k: dict(v) for k, v in label_counts.items()},
              "selection_rank": "lowest SHA256(canonical JSON [seed, 'refresh-c', record ID]) among whole-record v2f/v3w <=2048; fresh records appended in rank order",
              "fresh_c_per_k": dict(sorted(per_k.items())), "fresh_c_strata_covered": len(fresh_strata), "fresh_c_schema_shapes": dict(shapes),
              "coverage_scope": "seeded quota sample of balanced candidate pool; no full-stratum/class balancing promise for final quota",
              "source_pins": {"b_pointer": b_pointer_sha256, "b_build_report": b_report_sha256, "b_exclusion": b_exclusion_sha256,
                              "c_pointer": c_pointer_sha256, "c_summary": c_summary_sha256,
                              "c_census_summary": c_census_summary_sha256, "c_census_compressed": census_summary["census_compressed_sha256"]},
              "c_eligibility": {"rule": "whole record: max(v2f,v3w) prompt plus longest continuation <= 2048; no truncated questions/options",
                                "eligible_records": sum(eligible_by_bucket.values()), "excluded_records": sum(excluded_by_bucket.values()),
                                "per_k_style": [{"k": k, "id_style": style, "eligible": eligible_by_bucket[k, style], "excluded": excluded_by_bucket[k, style]}
                                                for k, style in sorted(set(eligible_by_bucket) | set(excluded_by_bucket))],
                                "selected_max_effective_tokens": selected_max, "selected_whole_record_2k": True,
                                "trainer_commit": census_summary.get("trainer_commit"), "trainer_source_sha256": census_summary.get("trainer_source_sha256"),
                                "original_full_c_sources_preserved": True},
              "source_files": {"b": {n: bp[n + "_uncompressed_sha256"] for n in NAMES}, "c": {n: cp[n + "_uncompressed_sha256"] for n in NAMES}},
              "retained_byte_proof": {"scope": "all non-C record and original sidecar bytes in source order", "source": dict(zip(NAMES, kept_proof)), "output": dict(zip(NAMES, [d.hexdigest() for d in kept_again]))},
              "compressed_sha256": {n: _sha_file(train_dir / names[n]) for n in NAMES},
              "uncompressed_sha256": {n: d.hexdigest() for n, d in zip(NAMES, hashes)}, "fresh_c_payload_bytes": payload_bytes,
              "exclusion_scope": "B receipt inherited only for byte-identical retained rows; fresh synthetic C verified; no protected rows or B checker read",
              "token_bounds": {"retained_b": {"scope": "source maxima apply to unchanged retained subset; source percentiles are not refreshed measurements", "source_census": br["tokens"]},
                               "fresh_c": {"layout": "g1v2", "templates": ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"], "selected_max_effective_tokens": selected_max},
                               "applicability": "pinned B source maxima apply to retained B; completed pinned trainer census proves every selected fresh C record <=2048 for both templates; full C pool is not all-fit"},
              "pointer": pointer, "pointer_sha256": _sha(pointer_raw)}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def c_only(c_source, c_pointer, c_pointer_sha256, c_summary, c_summary_sha256,
           c_census_summary, c_census_summary_sha256, out, *, c_census_file=None,
           quota=7000, seed=20261009, data_root=None, pointer_out=None):
    """Pure synthetic C quota for a clean B rebuild; no B inputs are read."""
    from .config import DATA_ROOT
    from harness.heldout import check_generator

    out, data_root = Path(out), Path(data_root or DATA_ROOT).resolve()
    relative = out.resolve().relative_to(data_root)
    pointer_out = Path(pointer_out) if pointer_out else None
    if out.exists() or (pointer_out and pointer_out.exists()):
        raise FileExistsError("refusing existing C-only output or pointer")
    if type(quota) is not int or quota < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer quota and nonnegative seed required")
    cp = _pinned_json(c_pointer, c_pointer_sha256)
    cs = _pinned_json(c_summary, c_summary_sha256)
    if (cs.get("schema") != "ssdm.c_multirecord_subset.v1" or cs.get("pointer") != cp
            or cs.get("pointer_sha256") != c_pointer_sha256 or cs.get("records") != cp["n_records"]
            or cs.get("record_bytes_preserved") is not True or cs.get("generator_spec_check_passed") is not True):
        raise ValueError("C receipt does not bind the completed synthetic pool")
    summary, census_path, lengths = _census(c_census_summary, c_census_summary_sha256,
                                            c_census_file, cp, c_pointer_sha256)
    check_generator({**GENERATOR_SPEC, "name": "c_multirecord_synth_v2", "version": 2})
    heap, payload_bytes, eligible, excluded, selected_max = _select_c(c_source, cp, lengths, quota, seed)
    for path, pin in ((c_pointer, c_pointer_sha256), (c_summary, c_summary_sha256), (c_census_summary, c_census_summary_sha256)):
        if _sha(Path(path).read_bytes()) != pin:
            raise ValueError("C input metadata changed during selection")
    if _sha_file(census_path) != summary["census_compressed_sha256"]:
        raise ValueError("C census changed during selection")
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    names = {"records": "records.jsonl.gz", "manifest": out.name + ".manifest.jsonl.gz", "content_map": out.name + ".content.jsonl.gz"}
    hashes = [hashlib.sha256() for _ in NAMES]
    labels = {kind: Counter() for kind in ("case_exact", "joint_score", "multi_select_member")}
    per_k, per_style, shapes = Counter(), Counter(), Counter()
    with ExitStack() as stack:
        streams = []
        for name in NAMES:
            handle = stack.enter_context((train_dir / names[name]).open("xb"))
            streams.append(stack.enter_context(gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0)))
        for _, _, raw in sorted(heap, key=lambda entry: (-entry[0], entry[1])):
            record = json.loads(raw[0])
            verify_record(record)
            k, style, _ = stratum(record)
            per_k[k] += 1
            per_style[style] += 1
            shapes[record["aug"]["schema_shape"]] += 1
            for qid, meta in record["aug"]["question_roles"].items():
                if meta["kind"] in labels:
                    labels[meta["kind"]][record["labels"][qid]] += 1
            for stream, digest, line in zip(streams, hashes, raw):
                stream.write(line)
                digest.update(line)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    pointer = {"build": out.name + " (python -m data.c_multirecord_refresh --c-only)", "n_records": quota,
               "location": (Path(data_root.name) / relative / "train_set").as_posix() + "/", "manifest": names["manifest"], "content_map": names["content_map"],
               "builder": {"module": "data.c_multirecord_refresh", "commit": commit},
               "excludes_cohort": "pure synthetic C; no B/Index/TEST/cohort records read; consuming clean B rebuild owns its guard",
               "receipt": f"receipts/2026-10-08-C-multirecord-synth/{out.name}/summary.json",
               **{name + "_uncompressed_sha256": digest.hexdigest() for name, digest in zip(NAMES, hashes)}}
    pointer_raw = json.dumps(pointer, indent=1).encode() + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    if pointer_out:
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as handle:
            handle.write(pointer_raw)
    report = {"schema": "ssdm.c_multirecord_admitted_subset.v1", "records": quota, "seed": seed, "source_records": cp["n_records"], "by_source": {V2: quota},
              "selection_rank": "lowest SHA256(canonical JSON [seed, 'refresh-c', record ID]) among whole-record v2f/v3w <=2048",
              "source_pins": {"c_pointer": c_pointer_sha256, "c_summary": c_summary_sha256, "c_census_summary": c_census_summary_sha256,
                              "c_census_compressed": summary["census_compressed_sha256"]},
              "source_files_uncompressed_sha256": {n: cp[n + "_uncompressed_sha256"] for n in NAMES},
              "eligible_records": sum(eligible.values()), "excluded_records": sum(excluded.values()),
              "eligibility_per_k_style": [{"k": k, "id_style": style, "eligible": eligible[k, style], "excluded": excluded[k, style]}
                                         for k, style in sorted(set(eligible) | set(excluded))],
              "per_k_records": dict(sorted(per_k.items())), "per_style_records": dict(per_style), "schema_shapes": dict(shapes),
              "label_counts": {k: dict(v) for k, v in labels.items()}, "raw_payload_bytes": payload_bytes,
              "record_and_sidecar_bytes_preserved": True, "b_inputs_read": False, "original_full_c_sources_preserved": True,
              "token_bounds": {"layout": "g1v2", "templates": ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"],
                               "length_scope": "prompt plus longest continuation", "selected_max_effective_tokens": selected_max,
                               "selected_whole_record_2k": True, "trainer_commit": summary.get("trainer_commit"), "trainer_source_sha256": summary.get("trainer_source_sha256")},
              "scope": "whole eligible records only; no question/option truncation; no all-K or class-balancing claim; full source remains preserved",
              "uncompressed_sha256": {name: digest.hexdigest() for name, digest in zip(NAMES, hashes)},
              "compressed_sha256": {n: _sha_file(train_dir / names[n]) for n in NAMES}, "pointer": pointer, "pointer_sha256": _sha(pointer_raw)}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix, fields in (("b", ("source", "pointer", "pointer-sha256", "report", "report-sha256", "exclusion", "exclusion-sha256")),
                           ("c", ("source", "pointer", "pointer-sha256", "summary", "summary-sha256", "census-summary", "census-summary-sha256"))):
        for field in fields:
            parser.add_argument(f"--{prefix}-{field}", required=prefix == "c")
    parser.add_argument("--c-only", action="store_true", help="pure synthetic admitted C quota; no B inputs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--c-census-file", help="default: census.jsonl.gz beside pinned census summary")
    parser.add_argument("--quota", type=int, default=7000)
    parser.add_argument("--expected-records", type=int, default=229891)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--data-root")
    parser.add_argument("--pointer-out")
    args = vars(parser.parse_args(argv))
    only = args.pop("c_only")
    if only:
        args = {k: v for k, v in args.items() if not k.startswith("b_") and k != "expected_records"}
        result = c_only(**args)
    else:
        if any(value is None for key, value in args.items() if key.startswith("b_")):
            parser.error("B inputs/pins are required unless --c-only is set")
        result = refresh(**args)
    print(json.dumps({k: result[k] for k in ("records", "pointer_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
