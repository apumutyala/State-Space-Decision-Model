"""EXECUTES jobs/q9_feature_cache/cache_job.sh end to end in a subprocess (Q9C_DRYRUN=1, CPU, a tiny random Qwen3.5
stand-in with a byte-level tokenizer, synthetic toy training records, fake DEV rows carrying real SELECT / HOLDOUT run
ids), then trains every proxy head family on the cache it wrote and writes kit-format results. LESSONS L-10: a
Python-level test does not count; this one runs the .sh. $0, no data, no GPU."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "jobs/q9_feature_cache/cache_job.sh"
BASH = shutil.which("bash")


def _tiny_model_dir(d: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast

    byte_vocab = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    tok = Tokenizer(models.BPE(vocab=byte_vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>"])
    fast.save_pretrained(d)
    torch.manual_seed(0)
    kw = dict(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
              num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
              linear_key_head_dim=8, linear_value_head_dim=8)
    m = tf.Qwen3_5ForCausalLM(tf.Qwen3_5TextConfig(**kw))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    m.save_pretrained(d)


def _subset(base: Path, ids: list[str], out: Path) -> str:
    from jobs.common import dev_subset as D

    d = json.loads(base.read_text(encoding="utf-8"))
    d.update(run_ids=sorted(ids), ids_sha256=D.ids_sha256(ids))
    out.write_text(json.dumps(d), encoding="utf-8")
    return str(out)


def _fake_dev_row(rid: str, i: int) -> dict:
    qs = {"pick": {"type": "choice", "instructions": "which option fits?",
                   "criteria": {f"option_{j}": f"choice {j}" for j in range(3 + i % 3)}},
          "ok": {"type": "noul", "instructions": "is it fine?"}}
    if i % 2:
        qs["lvl"] = {"type": "score", "instructions": "how much?", "criteria": ["low", "mid", "high"]}
    return {"id": rid, "state": f"record {i}: value {i * 7 % 11}; note ok", "questions": qs,
            "_evaluation": {"run_id": rid, "catalog_id": 1}}


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_runner_dry_run_end_to_end_then_proxy_heads(tmp_path):
    from train.data import toy_records

    data, work, out = tmp_path / "data", tmp_path / "work", tmp_path / "output"
    (data / "slices").mkdir(parents=True)
    model = tmp_path / "model"
    _tiny_model_dir(model)
    sel = json.loads((ROOT / "jobs/common/dev_subset_m1.json").read_text(encoding="utf-8"))["run_ids"][::400][:5]
    hol = json.loads((ROOT / "jobs/q9_feature_cache/dev_holdout.json").read_text(encoding="utf-8"))["run_ids"][::400][:5]
    rows = [_fake_dev_row(r, i) for i, r in enumerate(sel + hol)]
    with gzip.open(data / "slices" / "dev-rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import hashlib

    sha = hashlib.sha256((data / "slices" / "dev-rows.jsonl.gz").read_bytes()).hexdigest()
    (tmp_path / "pins.sha256").write_text(f"{sha}  slices/dev-rows.jsonl.gz\n", encoding="utf-8", newline="\n")
    recs = [{"id": r["id"], "request": r["request"], "labels": r["labels"]} for r in toy_records(12, seed=1)]
    with gzip.open(data / "toy-train.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    cfg = json.loads((ROOT / "jobs/q9_feature_cache/cache_config.json").read_text(encoding="utf-8"))
    cfg["d_model"] = 32
    cfg["rich_rows"] = 6
    cfg["bank_chunks"] = 4
    cfg["shard_requests"] = 4
    cfg["arms"]["stock"]["train_rows"] = 16
    cfg["dev_splits"] = {"select": _subset(ROOT / "jobs/common/dev_subset_m1.json", sel, tmp_path / "sel.json"),
                         "holdout": _subset(ROOT / "jobs/q9_feature_cache/dev_holdout.json", hol, tmp_path / "hol.json")}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "ctx.json").write_text("{}", encoding="utf-8")
    import re

    def run_part(mode):
        """One job = one part (review B1): each writes its own output directory, as the platform packages each."""
        out, work = tmp_path / f"output_{mode}", tmp_path / f"work_{mode}"
        env = {**os.environ, "Q9C_MODE": mode, "Q9C_DRYRUN": "1", "Q9C_CODE": str(ROOT), "Q9C_DATA": str(data),
               "Q9C_WORK": str(work), "Q9C_MODEL_DIR": str(model), "Q9C_CONFIG": str(tmp_path / "cfg.json"),
               "Q9C_DATA_PINS": str(tmp_path / "pins.sha256"), "Q9C_CONTEXT_MANIFEST": str(tmp_path / "ctx.json"),
               "GMN_OUTPUT_DIR": str(out), "GMN_RESULT_PATH": str(out / "result.json"),
               "GMN_METADATA_URL": "http://127.0.0.1:9", "HF_HUB_CACHE": str(tmp_path / "hf"),
               "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
        if os.name == "nt":  # GNU tar (Git Bash) reads "C:\..." as host:path without --force-local; Linux needs nothing
            env["TAR_OPTIONS"] = "--force-local"
        cmd = [BASH, "jobs/q9_feature_cache/cache_job.sh"]  # == RUNNER, spelled out for the launch lint (L-10)
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
        log = (work / "logs" / "run.log").read_text(encoding="utf-8", errors="replace") if (work / "logs/run.log").exists() else ""
        arm = work / "logs" / "arm_stock.log"
        arm_log = arm.read_text(encoding="utf-8", errors="replace") if arm.exists() else ""
        assert p.returncode == 0, (mode, p.returncode, log[-2000:], arm_log[-3000:])
        assert (out / "Q9_CACHE_DONE").exists()
        assert "TEST OVERRIDES SET" in log
        res = json.loads((work / "out" / "cache_result.json").read_text(encoding="utf-8"))
        assert res["valid"] and res["output_bytes"] <= res["max_bytes"]
        assert res["arms"]["stock"]["dry_run"] == (mode == "stock_train")  # toy training records only
        # parsed by regex: Git Bash's sha256sum prefixes the hash with "\\" for a Windows path holding backslashes
        man = (out / "MANIFEST.jsonl").read_text().splitlines()
        assert {re.search(r'"phase": "(\w+)"', x).group(1) for x in man} >= {"setup", "guard", "arm_stock", "final"}
        assert all('"ok": true' in x for x in man)
        return out, res

    out_dev, res_dev = run_part("stock_dev")
    sp = res_dev["arms"]["stock"]["splits"]
    assert sp["select"]["requests"] == 5 and sp["holdout"]["requests"] == 5 and "train" not in sp
    assert sp["select"]["lex_vocab_ok"] and not (out_dev / "cache" / "stock" / "train").exists()
    out_tr, res_tr = run_part("stock_train")
    assert res_tr["arms"]["stock"]["splits"]["train"]["questions"] == 16
    # the parts unpack into ONE cache directory (as the artifacts would be downloaded)
    out = tmp_path / "merged"
    shutil.copytree(out_dev / "cache", out / "cache")
    shutil.copytree(out_tr / "cache", out / "cache", dirs_exist_ok=True)
    # a tampered part file is caught by the report validator
    from jobs.q9_feature_cache.report import arm_report, valid_result

    shard = next((out / "cache" / "stock" / "select").glob("shard_*.pt"))
    shard.write_bytes(shard.read_bytes() + b"x")
    r = {"arms": {"stock": arm_report(out / "cache" / "stock", ["select", "holdout"])}, "output_bytes": 1, "max_bytes": 2}
    assert not valid_result(r)
    shutil.copy(out_dev / "cache" / "stock" / "select" / shard.name, shard)
    r = {"arms": {"stock": arm_report(out / "cache" / "stock", ["select", "holdout"])}, "output_bytes": 1, "max_bytes": 2}
    assert valid_result(r) and not valid_result({**r, "output_bytes": 3})

    # --- the proxy loop on the cache the runner wrote: every family trains, predicts and writes kit rows
    from train import proxy_loop as PL
    from train.proxy_rule import RECIPES

    train = PL.load_split(out / "cache", "stock", "train")
    dev = PL.load_split(out / "cache", "stock", "select")
    assert all(q.label >= 0 for r in train for q in r.questions)
    budget = {**PL.TRAIN, "steps": 4, "grad_accum": 2, "warmup_steps": 1}
    dev_rows = {r["_evaluation"]["run_id"]: r for r in rows if r["_evaluation"]["run_id"] in set(sel)}
    for rid in ("R00", "R01", "R04", "R11"):
        rec = next(r for r in RECIPES if r["id"] == rid)
        head, info = PL.train_head(rec, train, seed=0, budget=budget)
        assert info["steps"] >= 1 and all(map(lambda v: v == v, info["loss_last"]))
        counts = PL.write_results(tmp_path / f"{rid}.jsonl", dev_rows, dev, PL.predict(head, dev), rid)
        assert counts == {"ok": 5, "error": 0}
    stock = PL.write_results(tmp_path / "stock.jsonl", dev_rows, dev, PL.stock_predictions(dev), "stock")
    assert stock["ok"] == 5
    line = json.loads((tmp_path / "R01.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["status"] == "ok" and set(line["response"]["answers"]) == set(dev_rows[line["run_id"]]["questions"])
