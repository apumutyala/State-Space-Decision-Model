"""Keep every synthetic C K73..77 world via proven, compact identifier bijections."""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import string
import subprocess
import sys
import types

from .c_multirecord_refresh import NAMES, V2, _triples
from .c_multirecord_subset import _pinned_json, _sha, stratum
from .c_multirecord_synth import (BoundedEncodingCache, GENERATOR_SPEC, _histogram_summary,
                                _load_encoder, _load_layout_without_package_initializers,
                                _sha_file, canonical, verify_record)

ROOT = Path(__file__).resolve().parents[1]
POOLS_PATH = Path(__file__).with_name("c_multirecord_compact_pools.json")
POOLS_SHA256 = "a8cd7ecb63d876a541037fa6ed530fe985da1bd4d848b8f5ae8ab0ea432380aa"
TRAINER_COMMIT = "9c3e4350ccb41bec1385d58b9fe7002184f69626"
SOURCE_RECORDS, WORLDS_PER_K = 100000, 1315
HIGH_K = range(73, 78)
PAYLOAD_LIMIT = 80 * 1024 * 1024
SAFE_SYMBOLS = set(string.punctuation) - set("<>`\"'\\")
VIEW_SUFFIX = "~view:compact-single-token-v1"


def load_pools(path=POOLS_PATH, sha256=POOLS_SHA256):
    manifest = _pinned_json(path, sha256)
    if (manifest.get("schema") != "ssdm.c_multirecord_compact_pools.v1"
            or manifest.get("trainer_commit") != TRAINER_COMMIT
            or set(manifest.get("pools", {})) != {"letters", "words", "opaque"}):
        raise ValueError("unknown compact pool/trainer identity")
    for style, entries in manifest["pools"].items():
        ids = [entry["id"] for entry in entries]
        tokens = [entry["answer_token"] for entry in entries]
        if (len(ids) != 77 or ids != sorted(set(ids)) or len(set(tokens)) != 77
                or any(type(token) is not int or token <= 0 for token in tokens)
                or _sha(canonical(ids)) != manifest["pool_ids_sha256"][style]):
            raise ValueError("compact pool IDs/tokens/hash differ")
        for oid in ids:
            valid = (bool(re.fullmatch("[A-Z]{1,2}", oid)) if style == "letters" else
                     bool(re.fullmatch("[a-z]{3,8}", oid)) if style == "words" else
                     2 <= len(oid) <= 8 and all(char in SAFE_SYMBOLS for char in oid))
            if (not valid or oid in {"true", "false", "null", "none"}
                    or oid.casefold() in manifest["excluded_alias_values"]):
                raise ValueError("compact pool contains reserved, alias, or dishonest style ID")
    return manifest


def compact_record(old, manifest):
    """Return only an alpha view; inverse maps exist locally, never in record metadata."""
    verify_record(old)
    k, style, position = stratum(old)
    if k not in HIGH_K or old["source"] != V2 or old["source_split"] != "synthetic":
        raise ValueError("only original high-K synthetic C v2 records accepted")
    if old["aug"]["generator"] != "c_multirecord_synth_v2" or "compact_view" in old["aug"]:
        raise ValueError("source is not an original C v2 world")
    row = deepcopy(old)
    old_qids = sorted(old["questions"])
    qmap = dict(zip(old_qids, string.ascii_lowercase[:7], strict=True))
    choice = next(q for q, meta in old["aug"]["question_roles"].items() if meta["kind"] == "top_profile")
    old_ids = sorted(old["questions"][choice]["criteria"])
    new_ids = [entry["id"] for entry in manifest["pools"][style]][:k]
    omap = dict(zip(old_ids, new_ids, strict=True))
    row["questions"][choice]["criteria"] = {omap[oid]: deepcopy(old["questions"][choice]["criteria"][oid]) for oid in old_ids}
    row["labels"][choice] = omap[old["labels"][choice]]
    row["questions"] = {qmap[q]: row["questions"][q] for q in old_qids}
    row["labels"] = {qmap[q]: row["labels"][q] for q in old_qids}
    row["aug"]["question_roles"] = {qmap[q]: deepcopy(old["aug"]["question_roles"][q]) for q in old_qids}
    for name in ("case_exact", "multi_select"):
        row["aug"][name]["question_ids"] = [qmap[q] for q in old["aug"][name]["question_ids"]]
    row["id"] = old["id"] + VIEW_SUFFIX
    row["aug"]["compact_view"] = {"version": 1, "source_record_id": old["id"],
                                    "pool_sha256": manifest["pool_ids_sha256"][style]}
    # Prove all original meanings, labels, positions, and metadata references by inversion.
    inverse = {value: key for key, value in omap.items()}
    restored = {q: deepcopy(row["questions"][qmap[q]]) for q in old_qids}
    restored[choice]["criteria"] = {inverse[oid]: value for oid, value in restored[choice]["criteria"].items()}
    labels = {q: row["labels"][qmap[q]] for q in old_qids}
    labels[choice] = inverse[labels[choice]]
    aug = deepcopy(row["aug"])
    aug.pop("compact_view")
    aug["question_roles"] = {q: aug["question_roles"][qmap[q]] for q in old_qids}
    back = {value: key for key, value in qmap.items()}
    for name in ("case_exact", "multi_select"):
        aug[name]["question_ids"] = [back[q] for q in aug[name]["question_ids"]]
    if (restored != old["questions"] or labels != old["labels"] or aug != old["aug"]
            or row["state"] != old["state"] or row["source_id"] != old["source_id"]
            or sorted(row["questions"][qmap[choice]]["criteria"]).index(row["labels"][qmap[choice]]) != position):
        raise ValueError("compact alpha inverse changed world/question/label/position semantics")
    verify_record(row)
    return row


class TrainerRuntime:
    """Pinned trainer adapter + tokenizer only; the model modules are never imported."""
    def __init__(self, manifest, trainer_root, tokenizer_dir):
        self.manifest, self.trainer = manifest, Path(trainer_root)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.assert_pins()
        original = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise RuntimeError("model/Torch imports forbidden in compact export")
            return original(name, *args, **kwargs)
        builtins.__import__ = no_torch
        self.build_row = _load_layout_without_package_initializers(self.trainer)
        package = types.ModuleType("train")
        package.__path__ = [str(self.trainer / "train")]
        sys.modules["train"] = package
        wanted = {"RwkvSegmentEncoder", "parent_encoder", "host_provenance", "to_train_record"}
        source = self.trainer / "train/run_cell.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
        if {node.name for node in nodes} != wanted:
            raise ValueError("pinned trainer adapter definitions missing")
        env = {"hashlib": hashlib, "json": json}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
        tokenizer, _ = _load_encoder(self.tokenizer_dir)
        encode = env["RwkvSegmentEncoder"](tokenizer, set(tokenizer.all_special_ids))
        self.encoder = BoundedEncodingCache(encode)
        self.encoder.token_bytes, self.encoder.identity = encode.token_bytes, encode.identity
        self.encoder.is_rwkv7_g1 = encode.is_rwkv7_g1
        self.to_train_record = env["to_train_record"]
        from heads.layout import V3W_ANSWER_CTX, v3w_first_token
        from decide.prompt_g1v3 import load_v3_codes
        entries, table_sha = load_v3_codes()
        if table_sha != manifest["v3_code_table_sha256"]:
            raise ValueError("frozen v3 code table differs from pool proof")
        codes = {entry["code"]: entry["token_id"] for entry in entries}
        base = self.encoder(V3W_ANSWER_CTX, True)
        special = set(tokenizer.all_special_ids) | {0}
        for style, pool in manifest["pools"].items():
            for entry in pool:
                oid, expected = entry["id"], entry["answer_token"]
                ids = self.encoder(V3W_ANSWER_CTX + " " + oid, True)
                if (ids[:len(base)] != base or len(ids) != len(base) + 1
                        or v3w_first_token(self.encoder, oid) != expected or ids[-1] != expected
                        or expected in special or (style == "letters" and codes.get(oid) != expected)):
                    raise ValueError("compact ID is not its proven unique complete answer token")

    def assert_pins(self):
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.trainer, text=True).strip()
        if commit != TRAINER_COMMIT:
            raise ValueError("trainer commit differs from pin")
        for name, pin in {**self.manifest["trainer_source_sha256"], **self.manifest["metadata_sha256"]}.items():
            if _sha_file(self.trainer / name) != pin:
                raise ValueError("pinned trainer/codebook source changed")
        for name, pin in self.manifest["tokenizer_sha256"].items():
            if _sha_file(self.tokenizer_dir / name) != pin:
                raise ValueError("tokenizer file differs from pool proof")
        if _sha_file(ROOT / "data/c_multirecord_synth.py") != self.manifest["generator_source_sha256"]:
            raise ValueError("C generator differs from the pool proof")
        tokenizer_source = "third_party/rwkv7_hf/tokenization_rwkv7.py"
        if _sha_file(ROOT / tokenizer_source) != self.manifest["trainer_source_sha256"][tokenizer_source]:
            raise ValueError("imported tokenizer implementation differs from pinned trainer")

    def measure(self, record):
        from train.data import items_from_records
        result = {"serialized": len(self.encoder(canonical(record).decode("utf-8"), True))}
        training = self.to_train_record(record)
        for key, template in (("v2f", "jrt-rwkv-g1-v2f"), ("v3w", "jrt-rwkv-g1-v3w")):
            items = items_from_records([training], self.encoder, layout="g1v2", template=template)
            if len(items) != 7 or {item.row.qid for item in items} != set(record["questions"]):
                raise ValueError("trainer route lost a complete question")
            longest = 0
            for item in items:
                from decide.schema import canonical_options
                expected_ids = [oid for oid, _ in canonical_options(record["questions"][item.row.qid])]
                if item.row.option_ids != expected_ids or item.label != expected_ids.index(str(record["labels"][item.row.qid])):
                    raise ValueError("trainer route changed complete options or gold index")
                direct = self.build_row(training["request"], item.row.qid, self.encoder, layout="g1v2", template=template)
                for name in ("ids", "option_ids", "continuations", "option_spans", "state_span", "field_span"):
                    if getattr(direct, name) != getattr(item.row, name):
                        raise ValueError("direct/trainer row route differs")
                cont = max(map(len, item.row.continuations))
                if not 1 <= cont <= 16 or item.row.n_options != len(direct.option_ids):
                    raise ValueError("invalid continuation/options in trainer route")
                longest = max(longest, len(item.row.ids) + cont)
            result[key] = longest
        if any(type(value) is not int or not 1 <= value <= 2048 for value in result.values()):
            raise ValueError(f"compact world exceeds exact complete-record limits: {result}")
        if "torch" in sys.modules:
            raise RuntimeError("Torch imported by compact export")
        return result


def export_compact(source, source_pointer, source_pointer_sha256, source_summary,
                   source_summary_sha256, trainer_root, tokenizer_dir, out, *, data_root=None,
                   pointer_out=None, pools=POOLS_PATH, pools_sha256=POOLS_SHA256):
    from .config import DATA_ROOT
    from harness.heldout import check_generator, check_record

    out, data_root = Path(out), Path(data_root or DATA_ROOT).resolve()
    relative = out.resolve().relative_to(data_root)
    pointer_out = Path(pointer_out) if pointer_out else None
    if out.exists() or (pointer_out and pointer_out.exists()):
        raise FileExistsError("refusing existing compact output or pointer")
    manifest = load_pools(pools, pools_sha256)
    pointer = _pinned_json(source_pointer, source_pointer_sha256)
    summary = _pinned_json(source_summary, source_summary_sha256)
    if (pointer["n_records"] != SOURCE_RECORDS or summary["pointer"] != pointer
            or summary["pointer_sha256"] != source_pointer_sha256 or summary["records"] != SOURCE_RECORDS
            or summary["label_method_verified_records"] != SOURCE_RECORDS or summary["generator"]["version"] != 2
            or summary["source_sha256"]["data/c_multirecord_synth.py"] != manifest["generator_source_sha256"]):
        raise ValueError("source receipt does not bind the complete original C v2 artifact")
    check_generator({**GENERATOR_SPEC, "name": "c_multirecord_synth_v2", "version": 2})
    runtime = TrainerRuntime(manifest, trainer_root, tokenizer_dir)
    from train.cohort import record_content_hash
    selected, ids, contents, counts, payload_bytes = [], set(), set(), Counter(), 0
    for record, content, raw in _triples(source, pointer):
        if (record["source"] != V2 or record["source_split"] != "synthetic"
                or record["aug"]["generator"] != "c_multirecord_synth_v2"):
            raise ValueError("source contains a non-original synthetic C v2 record")
        rid = hashlib.sha256(record["id"].encode()).digest()
        ch = bytes.fromhex(content)
        if rid in ids or ch in contents:
            raise ValueError("duplicate source ID or normalized content")
        ids.add(rid)
        contents.add(ch)
        k, _, _ = stratum(record)
        if k in HIGH_K:
            if record_content_hash(record) != content:
                raise ValueError("selected source normalized content differs from sidecar")
            verify_record(record)
            check_record(record)
            counts[k] += 1
            selected.append(raw[0])
            payload_bytes += len(raw[0])
            if payload_bytes > PAYLOAD_LIMIT:
                raise MemoryError("high-K source payload exceeds 80 MiB")
    del ids, contents
    if counts != {k: WORLDS_PER_K for k in HIGH_K}:
        raise ValueError("complete K73..77 world cohort missing; no worlds discarded")
    for path, pin in ((source_pointer, source_pointer_sha256), (source_summary, source_summary_sha256), (pools, pools_sha256)):
        if _sha(Path(path).read_bytes()) != pin:
            raise ValueError("input metadata changed during selection")
    # Only after all three source files pass their pins is an output created.
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    names = {"records": "records.jsonl.gz", "manifest": out.name + ".manifest.jsonl.gz", "content_map": out.name + ".content.jsonl.gz"}
    hashes = {name: hashlib.sha256() for name in NAMES}
    hist = {name: [0] * 2049 for name in ("v2f", "v3w", "serialized")}
    world_ids, record_ids, out_contents, styles, shapes, buckets = [], [], set(), Counter(), Counter(), Counter()
    labels = {kind: Counter() for kind in ("case_exact", "joint_score", "multi_select_member")}
    with ExitStack() as stack:
        streams = {}
        for name in NAMES:
            handle = stack.enter_context((train_dir / names[name]).open("xb"))
            streams[name] = stack.enter_context(gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0))
        for index, raw in enumerate(selected):
            old = json.loads(raw)
            row = compact_record(old, manifest)
            check_record(row)
            lengths = runtime.measure(row)  # failure refuses the whole export, never drops a question/world
            if set(lengths) != {"v2f", "v3w", "serialized"} or any(type(n) is not int or not 1 <= n <= 2048 for n in lengths.values()):
                raise ValueError("compact whole-record length measurement failed")
            for name, value in lengths.items():
                hist[name][value] += 1
            k, style, position = stratum(row)
            styles[style] += 1
            shapes[row["aug"]["schema_shape"]] += 1
            buckets[k, style, position] += 1
            world_ids.append(row["source_id"])
            record_ids.append(old["id"])
            content = record_content_hash(row)
            if content in out_contents:
                raise ValueError("compact views duplicate normalized content")
            out_contents.add(content)
            for qid, meta in row["aug"]["question_roles"].items():
                if meta["kind"] in labels:
                    labels[meta["kind"]][row["labels"][qid]] += 1
            line = canonical(row)
            emitted = {"records": line + b"\n", "manifest": canonical({"id": row["id"], "source": row["source"], "family": row["family"], "sha256": _sha(line)}) + b"\n",
                       "content_map": canonical({"id": row["id"], "content_sha256": content}) + b"\n"}
            for name, line in emitted.items():
                hashes[name].update(line)
                streams[name].write(line)
            if (index + 1) % 250 == 0:
                print(f"compact complete worlds: {index + 1}/{len(selected)}", flush=True)
    runtime.assert_pins()
    if len(set(world_ids)) != len(world_ids):
        raise ValueError("source world IDs are not distinct")
    if any(sum(buckets[k, style, position] for position in range(k)) == 0 for k in HIGH_K for style in manifest["pools"]):
        raise ValueError("full high-K/style cohort coverage absent")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result_pointer = {"build": out.name + " (python -m data.c_multirecord_compact)", "n_records": len(selected),
                      "location": (Path(data_root.name) / relative / "train_set").as_posix() + "/", "manifest": names["manifest"], "content_map": names["content_map"],
                      "excludes_cohort": "fully synthetic C alpha views; no B/Index/TEST/cohort records read; consuming peer owns its guard",
                      "builder": {"module": "data.c_multirecord_compact", "commit": commit},
                      "receipt": f"receipts/2026-10-08-C-multirecord-synth/{out.name}/summary.json",
                      **{name + "_uncompressed_sha256": digest.hexdigest() for name, digest in hashes.items()}}
    pointer_raw = json.dumps(result_pointer, indent=1).encode() + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    if pointer_out:
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as handle:
            handle.write(pointer_raw)
    report = {"schema": "ssdm.c_multirecord_compact.v1", "records": len(selected), "questions": 7 * len(selected),
              "source_pointer_sha256": source_pointer_sha256, "source_summary_sha256": source_summary_sha256,
              "source_files_uncompressed_sha256": {name: pointer[name + "_uncompressed_sha256"] for name in NAMES},
              "pools_sha256": pools_sha256, "pool_ids_sha256": manifest["pool_ids_sha256"],
              "prototype_source_sha256": manifest["prototype_source_sha256"], "prototype_proof_sha256": manifest["prototype_proof_sha256"],
              "trainer_commit": manifest["trainer_commit"], "trainer_source_sha256": manifest["trainer_source_sha256"],
              "tokenizer_sha256": manifest["tokenizer_sha256"], "codebook_metadata_sha256": manifest["metadata_sha256"],
              "generator_source_sha256": manifest["generator_source_sha256"], "per_k_records": dict(counts), "per_style_records": dict(styles),
              "schema_shape_counts": dict(shapes), "gold_strata_counts": [{"k": k, "id_style": s, "gold_position": p, "records": n} for (k, s, p), n in sorted(buckets.items())],
              "source_world_ids": world_ids, "source_record_ids": record_ids, "source_world_ids_sha256": _sha(canonical(world_ids)),
              "label_counts": {kind: dict(values) for kind, values in labels.items()},
              "lengths": {name: _histogram_summary(values) for name, values in hist.items()}, "length_scope": "whole record prompt plus longest continuation; serialized record separately",
              "layout": "g1v2", "templates": ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"], "all_worlds_retained": True,
              "inverse_alpha_semantics_passed": True, "verify_record_check_record_passed": True, "direct_trainer_row_equality_passed": True,
              "original_full_c_source_preserved": True, "b_inputs_read": False, "torch_imported": "torch" in sys.modules,
              "raw_source_payload_bytes": payload_bytes, "uncompressed_sha256": {n: d.hexdigest() for n, d in hashes.items()},
              "compressed_sha256": {n: _sha_file(train_dir / names[n]) for n in NAMES}, "pointer": result_pointer, "pointer_sha256": _sha(pointer_raw)}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-pointer", "source-pointer-sha256", "source-summary", "source-summary-sha256", "trainer-root", "tokenizer-dir", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--pointer-out")
    parser.add_argument("--pools", default=str(POOLS_PATH))
    parser.add_argument("--pools-sha256", default=POOLS_SHA256)
    report = export_compact(**vars(parser.parse_args(argv)))
    print(json.dumps({key: report[key] for key in ("records", "questions", "per_k_records", "per_style_records", "pointer_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
