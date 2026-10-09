"""Tokenizer-only admission census using pinned trainer code, without importing a model."""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.c_multirecord_synth import BoundedEncodingCache, TOKENIZER_PINS, _load_encoder, _load_layout_without_package_initializers


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0], "p50": ordered[(len(ordered)-1)//2],
            "p95": ordered[min(len(ordered)-1, (95*len(ordered)+99)//100-1)], "max": ordered[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("trainer-root", "trainer-commit", "records", "pointer", "pointer-sha256", "tokdir", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    trainer, out = Path(a.trainer_root).resolve(), Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=trainer, text=True).strip()
    if actual_commit != a.trainer_commit:
        raise ValueError("trainer checkout commit differs from pin")
    pointer_path = Path(a.pointer)
    if file_sha(pointer_path) != a.pointer_sha256:
        raise ValueError("source pointer byte pin mismatch")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    tokdir = Path(a.tokdir)
    if {name: file_sha(tokdir/name) for name in TOKENIZER_PINS} != TOKENIZER_PINS:
        raise ValueError("tokenizer byte pins differ")

    # Execute the original pure adapter definitions, not a reimplementation or train.run_cell's model imports.
    source = trainer / "train/run_cell.py"
    names = {"RwkvSegmentEncoder", "parent_encoder", "host_provenance", "to_train_record"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("trainer adapter definitions missing")
    env = {"hashlib": hashlib, "json": json}
    os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TORCH="0", USE_TF="0", USE_FLAX="0")
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("model/Torch imports forbidden in tokenizer census")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = no_torch
    build_row = _load_layout_without_package_initializers(trainer)
    package = types.ModuleType("train")
    package.__path__ = [str(trainer / "train")]
    sys.modules["train"] = package
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    # The trainer and census tokenizer implementation must be exactly the same source bytes.
    tokenizer_file = "third_party/rwkv7_hf/tokenization_rwkv7.py"
    if file_sha(ROOT/tokenizer_file) != file_sha(trainer/tokenizer_file):
        raise ValueError("tokenizer source differs from pinned trainer")
    tokenizer, _ = _load_encoder(tokdir)
    encoder = env["RwkvSegmentEncoder"](tokenizer, set(tokenizer.all_special_ids))
    cached = BoundedEncodingCache(encoder)
    cached.token_bytes, cached.identity = encoder.token_bytes, encoder.identity
    cached.is_rwkv7_g1 = encoder.is_rwkv7_g1
    from train.data import items_from_records
    files = ("train/run_cell.py", "train/data.py", "heads/layout.py", "decide/prompt.py",
             "decide/prompt_g1v3.py", tokenizer_file)
    code_pins = {name: file_sha(trainer/name) for name in files}
    out.mkdir(parents=True, exist_ok=False)
    lengths = {key: [] for key in ("v2f", "v3w")}
    max_cont, max_k, ids, source_hash = 0, 0, set(), hashlib.sha256()
    proof_records = 0
    with gzip.open(a.records, "rb") as records, gzip.open(out/"census.jsonl.gz", "wt", encoding="utf-8") as census:
        for raw in records:
            source_hash.update(raw)
            record = json.loads(raw)
            if record["id"] in ids or record["source"] != "synthetic_c_multirecord_v2":
                raise ValueError("duplicate or non-v2 record")
            ids.add(record["id"])
            training_record = env["to_train_record"](record)
            result = {"id": record["id"]}
            for key, template in (("v2f", "jrt-rwkv-g1-v2f"), ("v3w", "jrt-rwkv-g1-v3w")):
                items = items_from_records([training_record], cached, layout="g1v2", template=template)
                # First record proves direct builder calls match the actual training-items route, field for field.
                if proof_records == 0:
                    for item in items:
                        qid = item.row.qid
                        direct = build_row(training_record["request"], qid, cached, layout="g1v2", template=template)
                        for name in ("ids", "option_ids", "continuations", "option_spans", "state_span", "field_span"):
                            if getattr(direct, name) != getattr(item.row, name):
                                raise ValueError("direct/trainer route mismatch")
                longest = max(len(item.row.ids) + max(map(len, item.row.continuations)) for item in items)
                max_cont = max(max_cont, max(max(map(len, item.row.continuations)) for item in items))
                max_k = max(max_k, max(item.row.n_options for item in items))
                lengths[key].append(longest)
                result[key] = longest
            proof_records += 1
            census.write(json.dumps(result, sort_keys=True) + "\n")
            if proof_records % 1000 == 0:
                print(f"trainer census: {proof_records}/{pointer['n_records']}", flush=True)
    if source_hash.hexdigest() != pointer["records_uncompressed_sha256"] or len(ids) != pointer["n_records"]:
        raise ValueError("source records differ from pinned pointer")
    if code_pins != {name: file_sha(trainer/name) for name in files}:
        raise ValueError("trainer source changed during census")
    passed = max(max(vals) for vals in lengths.values()) <= 2048 and max_k <= 77 and max_cont <= 16
    report = {"schema": "ssdm.c_multirecord_trainer_census.v1", "records": len(ids), "admitted": passed,
              "layout": "g1v2", "length_scope": "prompt plus longest continuation",
              "lengths": {key: stats(vals) for key, vals in lengths.items()}, "max_options": max_k,
              "max_continuation_tokens": max_cont, "source_pointer_sha256": a.pointer_sha256,
              "source_records_sha256": source_hash.hexdigest(), "trainer_commit": actual_commit,
              "trainer_source_sha256": code_pins, "tokenizer_sha256": TOKENIZER_PINS,
              "direct_builder_training_route_equality_records": 1, "torch_imported": "torch" in sys.modules,
              "census_compressed_sha256": file_sha(out/"census.jsonl.gz")}
    (out/"summary.json").write_text(json.dumps(report, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
