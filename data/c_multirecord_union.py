"""One exact C-only P3 pool: admitted 7k, then qualified compact high-K worlds."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from .c_multirecord_refresh import FAMILY, NAMES, V2, _triples
from .c_multirecord_subset import _pinned_json, _sha, stratum
from .c_multirecord_synth import _sha_file, canonical

BASE_RECORDS, HIGH_RECORDS = 7000, 6575
ALL_K = set(range(2, 78))
TRAINER = "9c3e4350ccb41bec1385d58b9fe7002184f69626"
TEMPLATES = ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"]
PAYLOAD_LIMIT = 80 * 1024 * 1024


def union(base_source, base_pointer, base_pointer_sha256, base_summary, base_summary_sha256,
          base_census_summary, high_source, high_pointer, high_pointer_sha256,
          high_summary, high_summary_sha256, out, *, data_root=None, pointer_out=None):
    from .config import DATA_ROOT
    from train.cohort import record_content_hash

    out, data_root = Path(out), Path(data_root or DATA_ROOT).resolve()
    relative = out.resolve().relative_to(data_root)
    pointer_out = Path(pointer_out) if pointer_out else None
    if out.exists() or (pointer_out and pointer_out.exists()):
        raise FileExistsError("refusing existing union output or pointer")
    bp, hp = _pinned_json(base_pointer, base_pointer_sha256), _pinned_json(high_pointer, high_pointer_sha256)
    bs, hs = _pinned_json(base_summary, base_summary_sha256), _pinned_json(high_summary, high_summary_sha256)
    for summary, pointer, pin, n in ((bs, bp, base_pointer_sha256, BASE_RECORDS), (hs, hp, high_pointer_sha256, HIGH_RECORDS)):
        if summary.get("pointer") != pointer or summary.get("pointer_sha256") != pin or summary.get("records") != n or pointer["n_records"] != n:
            raise ValueError("component receipt/pointer/count mismatch")
    if (bs.get("schema") != "ssdm.c_multirecord_admitted_subset.v1" or bs.get("b_inputs_read") is not False
            or hs.get("schema") != "ssdm.c_multirecord_compact.v1" or hs.get("b_inputs_read") is not False
            or any(hs.get(key) is not True for key in ("all_worlds_retained", "inverse_alpha_semantics_passed", "verify_record_check_record_passed", "direct_trainer_row_equality_passed"))
            or hs.get("torch_imported") is not False):
        raise ValueError("both components must be completed qualified C-only artifacts")
    bounds = bs["token_bounds"]
    if (bounds.get("selected_whole_record_2k") is not True or bounds.get("layout") != "g1v2"
            or hs.get("layout") != "g1v2" or bounds.get("templates") != TEMPLATES or hs.get("templates") != TEMPLATES
            or bounds.get("trainer_commit") != TRAINER or hs.get("trainer_commit") != TRAINER
            or bounds.get("trainer_source_sha256") != hs.get("trainer_source_sha256") or not hs.get("trainer_source_sha256")):
        raise ValueError("component trainer/layout/template qualifications differ")
    census_pin = bs["source_pins"]["c_census_summary"]
    census = _pinned_json(base_census_summary, census_pin)
    if (census.get("source_pointer_sha256") != bs["source_pins"]["c_pointer"]
            or census.get("source_records_sha256") != bs["source_files_uncompressed_sha256"]["records"]
            or census.get("records") != bs["source_records"] or census.get("trainer_commit") != TRAINER
            or census.get("trainer_source_sha256") != hs["trainer_source_sha256"]
            or census.get("tokenizer_sha256") != hs.get("tokenizer_sha256") or not hs.get("tokenizer_sha256")):
        raise ValueError("base census does not bind the same source/trainer/tokenizer")
    maxima = {key: max(bounds["selected_max_effective_tokens"][key], hs["lengths"][key]["max"]) for key in ("v2f", "v3w")}
    if any(type(n) is not int or not 1 <= n <= 2048 for n in (*bounds["selected_max_effective_tokens"].values(), *(hs["lengths"][key]["max"] for key in ("v2f", "v3w")))):
        raise ValueError("component complete-record 2k bound failed")
    payload, ids, worlds, contents, per_k, styles, labels, component_proofs = [], set(), set(), set(), Counter(), Counter(), {k: Counter() for k in ("case_exact", "joint_score", "multi_select_member")}, {}
    payload_bytes = questions = 0
    for name, source, pointer in (("base", base_source, bp), ("high", high_source, hp)):
        digests = [hashlib.sha256() for _ in NAMES]
        component_worlds = set()
        for record, content, raw in _triples(source, pointer):
            rid, world = record["id"], record["source_id"]
            if record["source"] != V2 or record["family"] != FAMILY or record["source_split"] != "synthetic" or record["aug"]["generator"] != "c_multirecord_synth_v2":
                raise ValueError("union accepts only synthetic C v2 records")
            if rid in ids or world in worlds or content in contents:
                raise ValueError("component IDs/world source IDs/content are not disjoint")
            if not isinstance(world, str) or not world or record_content_hash(record) != content:
                raise ValueError("invalid world/content identity")
            ids.add(rid); worlds.add(world); contents.add(content); component_worlds.add(world)
            k, style, _ = stratum(record)
            if (name == "base" and k > 72) or (name == "high" and k not in range(73, 78)):
                raise ValueError("component K scope differs from qualification")
            if len(record["questions"]) != 7 or set(record["labels"]) != set(record["questions"]):
                raise ValueError("component lost a complete question/label")
            questions += 7; per_k[k] += 1; styles[style] += 1
            for qid, meta in record["aug"]["question_roles"].items():
                if meta["kind"] in labels:
                    labels[meta["kind"]][record["labels"][qid]] += 1
            payload_bytes += sum(map(len, raw))
            if payload_bytes > PAYLOAD_LIMIT:
                raise MemoryError("exact union payload exceeds 80 MiB; no rows discarded")
            payload.append(raw)
            for digest, line in zip(digests, raw):
                digest.update(line)
        if name == "high" and set(hs["source_world_ids"]) != component_worlds:
            raise ValueError("high component world metadata differs from records")
        component_proofs[name] = {key: digest.hexdigest() for key, digest in zip(NAMES, digests)}
    if set(per_k) != ALL_K or len(payload) != BASE_RECORDS + HIGH_RECORDS:
        raise ValueError("complete union K/world cohort missing; no quota/drop permitted")
    for path, pin in ((base_pointer, base_pointer_sha256), (base_summary, base_summary_sha256), (base_census_summary, census_pin), (high_pointer, high_pointer_sha256), (high_summary, high_summary_sha256)):
        if _sha(Path(path).read_bytes()) != pin:
            raise ValueError("component metadata changed during union")
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    names = {"records": "records.jsonl.gz", "manifest": out.name + ".manifest.jsonl.gz", "content_map": out.name + ".content.jsonl.gz"}
    digests = [hashlib.sha256() for _ in NAMES]
    with ExitStack() as stack:
        streams = []
        for key in NAMES:
            file = stack.enter_context((train_dir / names[key]).open("xb"))
            streams.append(stack.enter_context(gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0)))
        for raw in payload:
            for stream, digest, line in zip(streams, digests, raw):
                stream.write(line); digest.update(line)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    pointer = {"build": out.name + " (python -m data.c_multirecord_union)", "n_records": len(payload),
               "location": (Path(data_root.name) / relative / "train_set").as_posix() + "/", "manifest": names["manifest"], "content_map": names["content_map"],
               "builder": {"module": "data.c_multirecord_union", "commit": commit}, "receipt": f"receipts/2026-10-08-C-multirecord-synth/{out.name}/summary.json",
               "excludes_cohort": "qualified pure synthetic C only; no B/Index/TEST/cohort reads; consuming peer owns its guard",
               **{key + "_uncompressed_sha256": digest.hexdigest() for key, digest in zip(NAMES, digests)}}
    pointer_raw = json.dumps(pointer, indent=1).encode() + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    if pointer_out:
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as file:
            file.write(pointer_raw)
    report = {"schema": "ssdm.c_multirecord_union.v1", "records": len(payload), "questions": questions,
              "order": "all base component records in original order, then all high component records in original order; no quota/drop",
              "component_pins": {"base_pointer": base_pointer_sha256, "base_summary": base_summary_sha256, "base_census_summary": census_pin, "high_pointer": high_pointer_sha256, "high_summary": high_summary_sha256},
              "component_raw_byte_proofs": component_proofs, "record_and_sidecar_bytes_preserved": True,
              "world_source_ids": sorted(worlds), "world_source_ids_sha256": _sha(canonical(sorted(worlds))), "worlds_disjoint": True, "ids_content_unique": True,
              "per_k_records": dict(sorted(per_k.items())), "per_style_records": dict(styles), "label_counts": {k: dict(v) for k, v in labels.items()},
              "token_bounds": {"layout": "g1v2", "templates": TEMPLATES, "max_effective_tokens": maxima, "trainer_commit": TRAINER,
                               "trainer_source_sha256": hs["trainer_source_sha256"], "tokenizer_sha256": hs["tokenizer_sha256"], "scope": "inherited complete-record bounds; no retokenization"},
              "raw_payload_bytes": payload_bytes, "b_inputs_read": False,
              "uncompressed_sha256": {k: d.hexdigest() for k, d in zip(NAMES, digests)}, "compressed_sha256": {k: _sha_file(train_dir / names[k]) for k in NAMES}, "pointer": pointer, "pointer_sha256": _sha(pointer_raw)}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("base", "high"):
        for field in ("source", "pointer", "pointer-sha256", "summary", "summary-sha256"):
            parser.add_argument(f"--{prefix}-{field}", required=True)
    parser.add_argument("--base-census-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--pointer-out")
    report = union(**vars(parser.parse_args(argv)))
    print(json.dumps({k: report[k] for k in ("records", "questions", "pointer_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
