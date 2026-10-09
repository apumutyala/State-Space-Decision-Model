"""Export byte-preserved C v2 records balanced over (K, ID style, gold position)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import subprocess

from .c_multirecord_synth import GENERATOR_SPEC, _sha_file, canonical, verify_record

STYLES = ("letters", "words", "opaque")
EXPECTED_STRATA = {(k, style, pos) for k in range(2, 78) for style in STYLES for pos in range(k)}
PAYLOAD_LIMIT = 80 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _rank(seed, purpose, value):
    return hashlib.sha256(canonical([seed, purpose, value])).digest()


def stratum(record):
    roles = record["aug"]["question_roles"]
    choices = [(qid, meta) for qid, meta in roles.items() if meta["kind"] == "top_profile"]
    if len(choices) != 1:
        raise ValueError("exactly one top_profile question required")
    qid, meta = choices[0]
    question = record["questions"][qid]
    ids = sorted(question["criteria"])
    position = ids.index(record["labels"][qid])
    if (question["type"] != "choice" or meta["k"] != len(ids)
            or meta["gold_position"] != position or meta["id_style"] not in STYLES):
        raise ValueError("top_profile metadata differs from canonical choices")
    return len(ids), meta["id_style"], position


def allocations(counts, count, seed):
    """Two per bucket, then a seeded third; extra counts differ by <= 1 across styles."""
    if type(count) is not int or not 2 * len(counts) <= count <= 3 * len(counts):
        raise ValueError("subset size must permit exactly two or three records per stratum")
    if not counts or min(counts.values()) < 2:
        raise ValueError("every expected stratum needs at least two source records")
    result = {bucket: 2 for bucket in counts}
    extra = count - 2 * len(counts)
    styles = sorted({bucket[1] for bucket in counts}, key=lambda s: (_rank(seed, "style", s), s))
    for i, style in enumerate(styles):
        take = extra // len(styles) + (i < extra % len(styles))
        eligible = sorted((b for b, n in counts.items() if b[1] == style and n >= 3),
                          key=lambda b: (_rank(seed, "extra-stratum", b), b))
        if len(eligible) < take:
            raise ValueError("insufficient third records for balanced style allocation")
        for bucket in eligible[:take]:
            result[bucket] = 3
    return result


def _pinned_json(path, expected_sha):
    raw = Path(path).read_bytes()
    if len(expected_sha) != 64 or _sha(raw) != expected_sha:
        raise ValueError(f"byte pin mismatch: {path}")
    return json.loads(raw)


def _source_scan(source, pointer, seed, prefix_count):
    """Verify all source bytes/semantics; retain at most three small identities per bucket."""
    from train.cohort import record_content_hash

    counts, prefix, best = Counter(), Counter(), defaultdict(list)
    seen_ids, seen_content = set(), set()
    hashes = {name: hashlib.sha256() for name in ("records", "manifest", "content_map")}
    paths = [source / "records.jsonl.gz", source / pointer["manifest"], source / pointer["content_map"]]
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, "rb")) for path in paths]
        for lines in zip_longest(*streams):
            if any(line is None or not line.endswith(b"\n") for line in lines):
                raise ValueError("source files differ in count or contain an unterminated line")
            for name, line in zip(hashes, lines):
                hashes[name].update(line)
            raw, manifest_raw, content_raw = lines
            record, manifest, content = (json.loads(line) for line in lines)
            rid = record["id"]
            line_sha = _sha(raw[:-1])
            content_sha = record_content_hash(record)
            if (record["source"] != "synthetic_c_multirecord_v2"
                    or record["aug"]["generator"] != "c_multirecord_synth_v2"
                    or record["source_split"] != "synthetic"):
                raise ValueError("only completed synthetic C v2 source records accepted")
            if (manifest != {"id": rid, "sha256": line_sha, "source": record["source"], "family": record["family"]}
                    or content != {"id": rid, "content_sha256": content_sha}):
                raise ValueError("source manifest/content identity mismatch")
            if rid in seen_ids or content_sha in seen_content:
                raise ValueError("duplicate source record ID or normalized content")
            seen_ids.add(rid)
            seen_content.add(content_sha)
            verify_record(record)
            bucket = stratum(record)
            if bucket not in EXPECTED_STRATA:
                raise ValueError("unexpected source stratum")
            counts[bucket] += 1
            if len(seen_ids) <= prefix_count:
                prefix[bucket] += 1
            candidates = best[bucket]
            candidates.append((_rank(seed, "within-stratum", rid), rid, line_sha, content_sha))
            candidates.sort()
            del candidates[3:]
    actual = {name: digest.hexdigest() for name, digest in hashes.items()}
    if any(actual[name] != pointer[name + "_uncompressed_sha256"] for name in actual):
        raise ValueError("source uncompressed bytes differ from pinned pointer")
    if sum(counts.values()) != pointer["n_records"] or set(counts) != EXPECTED_STRATA:
        raise ValueError("source count or complete expected stratum coverage missing")
    return counts, best, actual, prefix


def export_subset(source, source_pointer, source_pointer_sha256, source_summary,
                  source_summary_sha256, out, *, count=20000, seed=20261009,
                  data_root=None, pointer_out=None):
    """Two source passes; no tokenizer, model, protected datasets, or rewritten records."""
    from .config import DATA_ROOT
    from harness.heldout import check_generator
    from train.cohort import record_content_hash

    source, out = Path(source), Path(out)
    source_pointer, source_summary = Path(source_pointer), Path(source_summary)
    data_root = Path(data_root or DATA_ROOT).resolve()
    relative_out = out.resolve().relative_to(data_root)
    pointer_out = Path(pointer_out) if pointer_out else None
    if out.exists() or (pointer_out is not None and pointer_out.exists()):
        raise FileExistsError("refusing existing subset output or pointer")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    pointer = _pinned_json(source_pointer, source_pointer_sha256)
    summary = _pinned_json(source_summary, source_summary_sha256)
    if (summary["pointer"] != pointer or summary["pointer_sha256"] != source_pointer_sha256
            or summary["records"] != pointer["n_records"]
            or summary["label_method_verified_records"] != pointer["n_records"]
            or summary["generator"]["version"] != 2):
        raise ValueError("source receipt does not describe the pinned completed v2 build")
    for name in ("manifest", "content_map"):
        if Path(pointer[name]).name != pointer[name]:
            raise ValueError("source sidecars must be immediate filenames")
    check_generator({**GENERATOR_SPEC, "name": "c_multirecord_synth_v2", "version": 2})
    counts, best, source_hashes, prefix = _source_scan(source, pointer, seed, count)
    quota = allocations(counts, count, seed)
    selected = [candidate for bucket in sorted(quota) for candidate in best[bucket][:quota[bucket]]]
    del best
    selected.sort(key=lambda candidate: (_rank(seed, "output-order", candidate[1]), candidate[1]))
    wanted = {candidate[1]: candidate for candidate in selected}
    payload, payload_bytes, digest = {}, 0, hashlib.sha256()
    with gzip.open(source / "records.jsonl.gz", "rb") as stream:
        for raw in stream:
            digest.update(raw)
            rid = json.loads(raw)["id"]
            if rid in wanted:
                if rid in payload or _sha(raw[:-1]) != wanted[rid][2]:
                    raise ValueError("selected source record changed between passes")
                payload_bytes += len(raw)
                if payload_bytes > PAYLOAD_LIMIT:
                    raise MemoryError("selected raw payload exceeds 80 MiB; no rows silently dropped")
                payload[rid] = raw
    if digest.hexdigest() != source_hashes["records"] or set(payload) != set(wanted):
        raise ValueError("source changed or selected records missing on second pass")
    if _sha(source_pointer.read_bytes()) != source_pointer_sha256:
        raise ValueError("source pointer changed during selection")
    # No output exists until both passes and all input pins have passed.
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    build = out.name
    names = {"records": "records.jsonl.gz", "manifest": build + ".manifest.jsonl.gz",
             "content_map": build + ".content.jsonl.gz"}
    hashes = {name: hashlib.sha256() for name in names}
    per_k, per_style, case_labels, score_labels, shapes = (Counter() for _ in range(5))
    with ExitStack() as stack:
        streams = {}
        for name, filename in names.items():
            raw_stream = stack.enter_context((train_dir / filename).open("xb"))
            streams[name] = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0))
        for _, rid, line_sha, content_sha in selected:
            raw = payload.pop(rid)
            record = json.loads(raw)
            verify_record(record)
            if record_content_hash(record) != content_sha:
                raise ValueError("selected output content changed")
            k, style, _ = stratum(record)
            per_k[k] += 1
            per_style[style] += 1
            shapes[record["aug"]["schema_shape"]] += 1
            for qid, meta in record["aug"]["question_roles"].items():
                if meta["kind"] == "case_exact":
                    case_labels[record["labels"][qid]] += 1
                elif meta["kind"] == "joint_score":
                    score_labels[record["labels"][qid]] += 1
            lines = {"records": raw,
                     "manifest": canonical({"id": rid, "sha256": line_sha, "source": record["source"], "family": record["family"]}) + b"\n",
                     "content_map": canonical({"id": rid, "content_sha256": content_sha}) + b"\n"}
            for name, line in lines.items():
                hashes[name].update(line)
                streams[name].write(line)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    result_pointer = {"build": build + " (python -m data.c_multirecord_subset)",
                      "location": (Path(data_root.name) / relative_out / "train_set").as_posix() + "/",
                      "n_records": count, "manifest": names["manifest"], "content_map": names["content_map"],
                      "excludes_cohort": pointer["excludes_cohort"],
                      "builder": {"module": "data.c_multirecord_subset", "commit": commit},
                      "receipt": f"receipts/2026-10-08-C-multirecord-synth/{build}/summary.json",
                      **{name + "_uncompressed_sha256": digest.hexdigest() for name, digest in hashes.items()}}
    pointer_raw = json.dumps(result_pointer, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    if pointer_out is not None:
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as stream:
            stream.write(pointer_raw)
    report = {"schema": "ssdm.c_multirecord_subset.v1", "records": count, "seed": seed,
              "source_directory": str(source.resolve()), "source_pointer": str(source_pointer.resolve()),
              "source_summary": str(source_summary.resolve()),
              "source_pointer_sha256": source_pointer_sha256, "source_summary_sha256": source_summary_sha256,
              "source_files_uncompressed_sha256": source_hashes, "source_records": pointer["n_records"],
              "selection": "two lowest seeded hashes per (K, ID style, canonical gold position); seeded third in style-balanced extra strata; output seeded hash order",
              "weighting": "K-weighted by number of gold-position strata, not uniform K; no class-balancing promise; full source unchanged",
              "coverage_scope": "exported candidate pool only; later subsampling such as a 12000-record quota may not preserve stratum coverage",
              "strata": len(quota), "stratum_count_histogram": dict(Counter(quota.values())),
              "per_k_records": dict(sorted(per_k.items())), "per_style_records": dict(sorted(per_style.items())),
              "case_exact_labels": dict(case_labels), "joint_score_labels": {str(i): score_labels[str(i)] for i in range(5)},
              "schema_shape_counts": dict(shapes),
              "source_prefix_strata": {"records": sum(prefix.values()), "covered": len(prefix),
                                       "min_count": min(prefix.get(b, 0) for b in EXPECTED_STRATA),
                                       "max_count": max(prefix.values()), "per_k_records": dict(sorted(Counter({k: sum(n for (kk, _, _), n in prefix.items() if kk == k) for k in {b[0] for b in prefix}}).items()))},
              "per_stratum": [{"k": k, "id_style": style, "gold_position": pos, "source": counts[k, style, pos], "selected": n}
                              for (k, style, pos), n in sorted(quota.items())],
              "raw_payload_bytes": payload_bytes, "payload_limit_bytes": PAYLOAD_LIMIT,
              "record_bytes_preserved": True, "duplicate_ids": 0, "duplicate_normalized_content": 0,
              "generator_spec_check_passed": True,
              "compressed_sha256": {name: _sha_file(train_dir / filename) for name, filename in names.items()},
              "reused_token_bounds": {"scope": "source census upper bounds apply to unchanged records; subset not retokenized", "source_token_limits": summary["token_limits"]},
              "pointer": result_pointer, "pointer_sha256": _sha(pointer_raw)}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-pointer", "source-pointer-sha256", "source-summary", "source-summary-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--data-root")
    parser.add_argument("--pointer-out")
    args = vars(parser.parse_args(argv))
    report = export_subset(**args)
    print(json.dumps({key: report[key] for key in ("records", "strata", "stratum_count_histogram", "per_style_records", "raw_payload_bytes", "pointer_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
