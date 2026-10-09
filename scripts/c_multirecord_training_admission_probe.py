"""CPU/tokenizer-only C quota admission probe against the actual D13 adapter/filter."""
from __future__ import annotations
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
from data.c_multirecord_synth import BoundedEncodingCache, _load_encoder, _load_layout_without_package_initializers, TOKENIZER_PINS
from data.c_multirecord_subset import stratum

TRAINER = ROOT.parent / "d13"
COMMIT = "8d92f04f6bbc46ec40cf19b2ab5786435e139266"
DATA = ROOT.parent.parent / "ssdm-data"
ARTIFACT = DATA / "c-multirecord-7k-v2-allk-quota"
POINTER = ROOT / "data/manifests/c-multirecord-7k-v2-allk-quota.pointer.json"
POINTER_SHA = "2c4659464e3ddaab28145b45628c59081964300f64d855f402edc1f6e1b81618"
OUT = ROOT / "receipts/2026-10-09-C-multirecord-quota/training-admission.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", USE_TORCH="0", USE_TF="0", USE_FLAX="0")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TRAINER, text=True).strip() == COMMIT
    assert sha(POINTER) == POINTER_SHA
    pointer = json.loads(POINTER.read_bytes())
    files = ("train/run_cell.py", "train/data.py", "heads/layout.py", "heads/model.py",
             "decide/prompt.py", "decide/prompt_g1v3.py", "third_party/rwkv7_hf/tokenization_rwkv7.py")
    pins = {name: sha(TRAINER / name) for name in files}
    assert pins[files[-1]] == sha(ROOT / files[-1])
    config_path = TRAINER / "jobs/m2_first_cell/m2_first_cell_rwkv7_d13_joint_safe_nofla.json"
    cfg = json.loads(config_path.read_bytes())
    override = {"head": "joint_record", "layout": "g1v2", "template": "jrt-rwkv-g1-v3w",
                "train_max_k": 255, "row_options": 160}
    candidate = {**cfg, **override}
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("Torch/model import forbidden")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = no_torch
    _load_layout_without_package_initializers(TRAINER)
    package = types.ModuleType("train")
    package.__path__ = [str(TRAINER / "train")]
    sys.modules["train"] = package
    source = TRAINER / "train/run_cell.py"
    wanted = {"RwkvSegmentEncoder", "host_provenance", "to_train_record", "train_max_k", "keep_item", "_items_chunk"}
    nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    env = {"hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    tokdir = DATA / "tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba"
    assert {name: sha(tokdir / name) for name in TOKENIZER_PINS} == TOKENIZER_PINS
    tok, _ = _load_encoder(tokdir)
    enc = env["RwkvSegmentEncoder"](tok, set(tok.all_special_ids))
    cache = BoundedEncodingCache(enc)
    cache.token_bytes, cache.identity, cache.is_rwkv7_g1 = enc.token_bytes, enc.identity, enc.is_rwkv7_g1
    samples, kcounts, digest = {}, Counter(), hashlib.sha256()
    with gzip.open(ARTIFACT / "train_set/records.jsonl.gz", "rb") as stream:
        for raw in stream:
            digest.update(raw)
            record = json.loads(raw)
            k, style, _ = stratum(record)
            kcounts[k] += 1
            samples.setdefault((k, style), record)
    assert digest.hexdigest() == pointer["records_uncompressed_sha256"]
    assert sum(kcounts.values()) == 7000
    cap = env["train_max_k"](cfg)
    dropped = sum(n for k, n in kcounts.items() if not env["keep_item"](types.SimpleNamespace(n_options=k, ids=[1], continuations=[[1]]), cap, 2048))
    sample_results, max_effective, max_cont, max_options_sum = [], 0, 0, 0
    for (k, style), record in sorted(samples.items()):
        training = env["to_train_record"](record)
        items = env["_items_chunk"]([training], candidate, cache, None, 2048, env["train_max_k"](candidate))
        assert len(items) == 7 and {x.row.qid for x in items} == set(record["questions"])
        assert {x.row.meta["record"] for x in items} == {record["id"]}
        longest = max(len(x.row.ids) + max(map(len, x.row.continuations)) for x in items)
        assert longest <= 2048
        options_sum = sum(x.row.n_options for x in items)
        assert options_sum <= candidate["row_options"]
        max_effective = max(max_effective, longest)
        max_options_sum = max(max_options_sum, options_sum)
        max_cont = max(max_cont, max(max(map(len, x.row.continuations)) for x in items))
        control_items = [x for x in items if env["keep_item"](x.row, cap, 2048)]
        assert len(control_items) == (7 if k <= cap else 6)
        sample_results.append({"id": record["id"], "k": k, "style": style, "candidate_questions": len(items),
                               "same_format_cap16_questions": len(control_items), "effective_tokens": longest})
    assert pins == {name: sha(TRAINER / name) for name in files}
    report = {"schema": "ssdm.c_training_admission.v1", "trainer_commit": COMMIT, "trainer_source_sha256": pins,
              "config": str(config_path), "config_sha256": sha(config_path), "source_pointer_sha256": POINTER_SHA,
              "source_records_uncompressed_sha256": digest.hexdigest(), "config_override_candidate": override,
              "source_records": 7000, "source_questions": 49000, "k_only_filter": {"cap": cap,
                  "top_profile_questions_dropped": dropped, "incomplete_records": dropped,
                  "complete_records_upper_bound": 7000-dropped,
                  "scope": "exact structural K filter count; excludes original v1 length effects"},
              "actual_candidate_sample": {"records": len(samples), "questions": len(samples)*7,
                  "selection": "first record per available K/style; no Index/protected sampling",
                  "max_effective_tokens": max_effective, "max_continuation_tokens": max_cont,
                  "max_record_options_sum": max_options_sum, "records_detail": sample_results},
              "evaluation_max_k_unchanged": candidate["max_k"] == cfg["max_k"] == 16,
              "training_record_row_budget": cfg["batch_size"] * cfg["grad_accum"],
              "limitations": ["C-only tokenizer/loader proof; no actual backbone training or Index measurement",
                  "not a B mixed-draw admission proof; clean B v2d and exact C retention remain required",
                  "D13 g1v2 ordinal gap feature fix must be integrated before use; see D15 b565ec2",
                  "actual token proof sampled all available K/style groups; full-pool v3w bounds inherited from pinned original proof"],
              "torch_imported": "torch" in sys.modules, "gpu_spend_usd": 0}
    OUT.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"k_only_filter": report["k_only_filter"], "sample_records": len(samples),
                      "sample_questions": len(samples)*7, "max_effective_tokens": max_effective, "max_options_sum": max_options_sum}))


if __name__ == "__main__":
    main()
