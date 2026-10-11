"""jobs/q9af_v2 (Q9-AF v2: all 8 attention layers -> GDN jointly): rule, mixture builder, preflight, shell dry run."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs.q9af_v2 import rule  # noqa: E402
from jobs.q9af_v2.rule import (ADVANCE_POINT, ATTENTION_LAYERS, BOOTSTRAP_B, BOOTSTRAP_SEED,  # noqa: E402
                               MARGINAL_POINT, MAX_HELDOUT_KL, PROXY_IDS_SHA256, RULE_VERSION, STRONG_LO, STRONG_POINT)

CFG = json.loads((ROOT / "configs/q9af_v2_h100.json").read_text(encoding="utf-8"))
PARAMS = json.loads((ROOT / "jobs/q9af_v2/job_params.json").read_text(encoding="utf-8"))
RUNNER = (ROOT / "jobs/q9af_v2/q9af_v2_job.sh").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- the pre-registered rule
def test_rule_constants_are_pinned():
    assert RULE_VERSION == "q9af-v2-rule-v1"
    assert ATTENTION_LAYERS == [3, 7, 11, 15, 19, 23, 27, 31] == CFG["attention_layers"]
    assert MAX_HELDOUT_KL == 0.25
    assert (STRONG_POINT, STRONG_LO, ADVANCE_POINT, MARGINAL_POINT) == (-4.0, -8.0, -10.0, -17.0)
    assert PROXY_IDS_SHA256 == json.loads((ROOT / "jobs/common/dev_proxy600.json").read_text())["ids_sha256"]
    assert (BOOTSTRAP_B, BOOTSTRAP_SEED) == (2000, 0)
    assert set(rule.ACTIONS) == set(rule.VERDICTS)


def _export(**kw):
    return {"ok": True, "mode": "final", "converted": list(ATTENTION_LAYERS), "attention_free": True, **kw}


def _cmp(point, lo, sha=PROXY_IDS_SHA256, B=2000, seed=0):
    return {"subset_ids_sha256": sha, "B": B, "seed": seed, "diff": {"index": {"point": point, "lo": lo, "hi": point + 4}}}


def _res(kl=0.1):
    return {"final_heldout": {"mix": {"kl": kl, "delta_nll": 0.05, "hidden_cos": 0.97}}}


@pytest.mark.parametrize("kl,point,lo,want", [
    (0.10, -2.0, -6.0, "STRONG"), (0.10, -2.0, -8.5, "ADVANCE"), (0.10, -4.0, -8.0, "STRONG"),
    (0.10, -10.0, -14.0, "ADVANCE"), (0.10, -10.01, -14.0, "MARGINAL"), (0.10, -17.0, -21.0, "MARGINAL"),
    (0.10, -17.01, -21.0, "FOLD"), (0.10, -26.0, -28.9, "FOLD"), (0.2501, 5.0, 1.0, "FOLD_KL"),
    (0.25, -1.0, -5.0, "STRONG")])
def test_verdict_ladder(kl, point, lo, want):
    d = rule.decide(_res(kl), _cmp(point, lo), _export())
    assert d["verdict"] == want and d["action"] == rule.ACTIONS[want] and rule.valid_decision(d)


def test_q9h2_numbers_fold():
    """job-2xe5w's 6-layer hybrid (Index -26.0 [-28.9, -22.2], wikitext KL 0.094) folds under v2's rule too."""
    assert rule.decide(_res(0.094), _cmp(-26.0, -28.9), _export())["verdict"] == "FOLD"


@pytest.mark.parametrize("bad", [
    {"export": None}, {"export": _export(converted=[3, 7, 11, 15, 19, 23, 31])}, {"export": _export(attention_free=False)},
    {"export": _export(mode="salvage")}, {"export": _export(ok=False)}, {"cmp": _cmp(-1, -3, sha="0" * 64)},
    {"cmp": _cmp(-1, -3, B=1000)}, {"cmp": None}, {"res": {}}])
def test_no_measurement(bad):
    d = rule.decide(bad.get("res", _res()), bad.get("cmp", _cmp(-1, -3)) if "cmp" in bad else _cmp(-1, -3),
                    bad["export"] if "export" in bad else _export())
    assert d["verdict"] == "NO_MEASUREMENT" and d["problems"] and not rule.valid_decision(d)


# ----------------------------------------------------------------------------- mixture builder
class FakeTok:
    """Whitespace tokenizer with a stable vocabulary (hash of the word) + the one special the builder needs."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in t.split()] for t in texts]}

    def convert_tokens_to_ids(self, t):
        assert t == "<|endoftext|>"
        return 7


def test_ngram_hashes_match_a_reference():
    from convert.mixdata import NGRAM, ngram_hashes

    ids = np.arange(40, dtype=np.int64) * 7919
    h = ngram_hashes(ids)
    assert len(h) == 40 - NGRAM + 1
    ref = []
    for i in range(len(h)):
        v = 0
        for j in range(NGRAM):
            v = (v * 1000003 + int(ids[i + j])) % 2 ** 64
        ref.append(v)
    assert [int(x) for x in h] == ref
    assert len(ngram_hashes(ids[:NGRAM - 1])) == 0


def test_contamination_filter_drops_only_docs_sharing_a_13gram():
    from convert.mixdata import build_ngram_set, contaminated

    tok = FakeTok()
    dev = " ".join(f"devword{i}" for i in range(30))
    ng = build_ngram_set(tok, [dev])
    docs = [" ".join(f"devword{i}" for i in range(5, 18)),          # 13 consecutive DEV words: dropped
            " ".join(f"devword{i}" for i in range(5, 17)) + " x",   # 12 + other: kept
            "completely unrelated text " * 10]
    ids = tok(docs)["input_ids"]
    assert contaminated(ids, ng).tolist() == [True, False, False]


def test_render_chat_matches_the_readout_template():
    from convert.mixdata import THINK_EMPTY, render_chat
    from decide.prompt import TEMPLATES

    t = render_chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], random.Random(1))
    tail = TEMPLATES["jrt-chatml-v1"].tail
    assert "<|im_start|>assistant\n" + THINK_EMPTY + "hello<|im_end|>\n" in t
    assert tail.startswith("<|im_end|>\n<|im_start|>assistant\n" + THINK_EMPTY.rstrip("\n"))
    assert t.count("<|im_start|>user\nhi<|im_end|>\n") == 1


def test_window_stream_never_repeats_within_a_pass_and_reports_passes():
    pytest.importorskip("torch")
    from convert.mixdata import WindowStream

    toks = np.arange(10 * 16 + 1, dtype=np.uint32)
    ws = WindowStream(toks, 16, 2, seed=0, device="cpu")
    seen = [tuple(r[:1].tolist()) for _ in range(5) for r in next(ws)]
    assert len(set(seen)) == 10 and ws.passes == 0 and ws.windows_used == 10
    next(ws)
    assert ws.passes == 1
    ws2, ref = WindowStream(toks, 16, 2, seed=0, device="cpu"), WindowStream(toks, 16, 2, seed=0, device="cpu")
    ws2.skip(4)
    next(ref), next(ref)
    assert ws2.windows_used == 4 and next(ws2).tolist() == next(ref).tolist()


def test_mixture_build_end_to_end(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from convert.mixdata import HELDOUT_GLOBS, SOURCES, TRAIN_GLOBS, build

    roots = {}
    for s in SOURCES:
        r = tmp_path / s
        roots[s] = str(r)
        for g in TRAIN_GLOBS[s] + HELDOUT_GLOBS[s]:
            p = r / g.replace("*", "x")
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            if s == "chat":
                rows = [{"messages": [{"role": "user", "content": f"q{k} " * 20},
                                      {"role": "assistant", "content": f"a{k} " * 30}]} for k in range(60)]
            else:
                rows = [{"text": f"{s}{k} " * 40} for k in range(60)]
            pq.write_table(pa.Table.from_pylist(rows), p, row_group_size=10)
    dev = tmp_path / "dev-rows.jsonl.gz"
    with gzip.open(dev, "wt", encoding="utf-8") as f:  # DEV row whose state equals one web doc -> that doc dropped
        f.write(json.dumps({"state": "web3 " * 40, "questions": {"q": {"type": "choice"}}}) + "\n")
    cfg = {"model_dir": "unused", "mix": {"train_tokens": 3000, "shares": {"web": 0.5, "chat": 0.3, "math": 0.2},
                                           "heldout_tokens": {"web": 100, "chat": 100, "math": 100}}}
    meta = build(cfg, str(dev), str(tmp_path / "mix"), tok=FakeTok(), roots=roots)
    assert meta["sources"]["web"]["docs_dropped_dev_ngram"] == 1
    assert all(meta["sources"][s]["tokens"] >= meta["sources"][s]["quota"] for s in SOURCES)
    train = np.fromfile(tmp_path / "mix" / "train.u32", dtype=np.uint32)
    assert len(train) == meta["train_tokens"] and (train == 7).sum() >= 3
    for s in ("web", "math"):  # held-out = the LAST row group; training stopped before it
        assert meta["heldout"][s]["units"][0][1] == 5
        assert meta["sources"][s]["row_groups"] <= 5
    assert meta["heldout"]["chat"]["units"][0][0].startswith("data/test_sft")
    assert build(cfg, str(dev), str(tmp_path / "mix"), tok=FakeTok(), roots=roots) == meta  # idempotent


# ----------------------------------------------------------------------------- preflight, params, registration
def test_preflight_passes_and_budget():
    from jobs.common.costs import PARAMS as REG, worst_case_usd
    from jobs.q9af_v2.preflight import check

    res = check(PARAMS, CFG)
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    w = worst_case_usd(PARAMS)
    assert w == 37.37 and w <= 40.0
    assert "37.37" in (ROOT / "jobs/q9af_v2/JOB_SPEC.md").read_text(encoding="utf-8")
    assert REG["q9af_v2"] == "jobs/q9af_v2/job_params.json"
    assert PARAMS["decision_rule"] == "jobs/q9af_v2/rule.py" and PARAMS["max_restarts"] == 0


def test_runner_dry_seam_is_refused_in_the_image():
    assert 'if [ -d /opt/ssdm/m1 ] || [ -z "${Q9AF2_DRYRUN_ROOT:-}" ]; then echo "Q9AF2_DRYRUN refused here"; exit 5; fi' \
        in RUNNER
    assert not any(k.startswith("Q9AF2") for k in PARAMS["submit"]["env"])


# ----------------------------------------------------------------------------- the SHELL-LEVEL dry run
BASH = shutil.which("bash")


def _to_posix(p):
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if re.match(r"^[A-Za-z]:/", s) else s


def _dry_root(tmp_path):
    root = tmp_path / "root"
    (root / "data" / "slices").mkdir(parents=True)
    ids = json.loads((ROOT / "jobs/common/dev_proxy600.json").read_text())["run_ids"]
    rows = root / "data" / "slices" / "dev-rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as f:
        for i in ids + ["not-in-proxy-1"]:
            f.write(json.dumps({"_evaluation": {"run_id": i}}) + "\n")
    sha = hashlib.sha256(rows.read_bytes()).hexdigest()
    (root / "data.sha256").write_text(f"{sha}  slices/dev-rows.jsonl.gz\n", encoding="utf-8", newline="\n")
    (root / "CONTEXT_MANIFEST.json").write_text(json.dumps({"dry_run": True}))
    shim = root / "shim"
    shim.mkdir()
    (shim / "curl").write_text('#!/usr/bin/env bash\nexit 0\n', newline="\n")
    (shim / "curl").chmod(0o755)
    return root


def run_dry(root, env_extra=None, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GMN_", "Q9AF2_"))}
    path = [_to_posix(root / "shim"), _to_posix(os.path.dirname(sys.executable))]
    env.update({"Q9AF2_DRYRUN": "1", "Q9AF2_DRYRUN_ROOT": _to_posix(root), "CUDA_VISIBLE_DEVICES": "",
                "GMN_METADATA_URL": "http://metadata.invalid", "Q9AF2_PATH_PREFIX": ":".join(path)})
    if os.name == "nt":
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p.endswith("site-packages")]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(env_extra or {})
    cmd = [BASH, "-c", 'export PATH="$Q9AF2_PATH_PREFIX:$PATH"; exec bash jobs/q9af_v2/q9af_v2_job.sh']
    with open(root / "console.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    return rc, root / "output"


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("point,lo,want_rc,want", [("-6.0", "-9.5", 0, "ADVANCE"), ("-26.0", "-28.9", 0, "FOLD")])
def test_shell_dry_run(tmp_path, point, lo, want_rc, want):
    pytest.importorskip("torch")
    root = _dry_root(tmp_path)
    rc, out = run_dry(root, {"Q9AF2_STUB_POINT": point, "Q9AF2_STUB_LO": lo})
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == want_rc, log[-4000:]
    res = json.loads((out / "result.json").read_text())
    assert res["ok"] and (out / "Q9AF2_DONE").exists()
    dec = json.loads((out / "decision.json").read_text())
    assert dec["verdict"] == want and dec["attention_free"]
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert {m["phase"] for m in man} >= {"setup", "D", "P", "X", "I_stock", "I_conv", "S", "final"}
    assert all(m["ok"] for m in man)


# ----------------------------------------------------------------------------- convert.joint on a tiny Qwen3.5 (CPU)
def _tiny_model():
    """tests/full8_dryrun/tiny.py's stand-in (32 layers, attention at 3, 7, ..., 31) with the 9B HEAD LAYOUT ratios
    (nk == H, dk < hd, nv / nk == hd / dv), so the reuse_attn init takes the 9B "proj" path."""
    import torch
    import torch.nn as nn
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from types import SimpleNamespace

    from convert.model import ConvertibleQwen, replace_native_gdn

    cfg = Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=32, vocab_size=101,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    lm = Mq.Qwen3_5TextModel(cfg).float()
    for p in lm.parameters():
        torch.nn.init.normal_(p, std=0.05)
    full = SimpleNamespace(model=SimpleNamespace(language_model=lm), lm_head=nn.Linear(32, 101, bias=False))
    m = ConvertibleQwen(full, kernel="chunk")
    replace_native_gdn(m, kernel="chunk")
    for p in list(lm.parameters()) + list(full.lm_head.parameters()):
        p.requires_grad_(False)
    return m.eval()


def test_joint_pipeline_tiny_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert import joint
    from probes import core as pcore
    from probes import throughput as pthr
    from tests.full8_dryrun.tiny import mmlu_rows, stub_probes, tokenizer

    model = _tiny_model()
    monkeypatch.setattr(joint, "load", lambda *a, **k: (model, None, "tiny"))
    rng = np.random.default_rng(0)
    wt = {s: rng.integers(0, 101, 4000).astype(np.uint32) for s in ("eval", "gate")}
    monkeypatch.setattr(joint.data, "build", lambda tok, s, max_tokens=None: (s, {"split": s}))
    real_load = joint.data.load_tokens
    monkeypatch.setattr(joint.data, "load_tokens", lambda p: wt[p] if p in wt else real_load(p))
    monkeypatch.setattr(pcore, "run_all", stub_probes)
    real_mmlu, ttok, rows = pcore.mmlu_full, tokenizer(), mmlu_rows(40)
    monkeypatch.setattr(pcore, "mmlu_full", lambda m, tok, **k: real_mmlu(m, ttok, rows=rows, mult=32, **k))
    monkeypatch.setattr(pthr, "measure", lambda m, modes, lengths, reps=2: {str(L): {"tok_s": 22144.5} for L in lengths})
    monkeypatch.setattr(pthr, "call_latency", lambda m, modes: {"pack_call_s": 0.24, "small_call_s": 0.05})
    mix = tmp_path / "mix"
    mix.mkdir()
    rng.integers(0, 101, 24 * 400 + 1).astype(np.uint32).tofile(mix / "train.u32")
    for s in ("web", "chat", "math"):
        rng.integers(0, 101, 24 * 9).astype(np.uint32).tofile(mix / f"heldout_{s}.u32")
    (mix / "mix.json").write_text(json.dumps({"dry_run": True}))
    c = json.loads(json.dumps(CFG))
    c.update(seq_len=24, batch=4, micro_batch=2, model_dir="tiny", ckpt_dir=str(tmp_path / "ck" / "gdn"),
             heldout_windows={"web": 8, "chat": 4, "math": 4}, curve_windows_per_source=4, wikitext_gate_windows=4)
    c.pop("vram_fraction")
    c["init_cmp"]["steps"] = 2
    c["stage1"]["guard"] = {"max_minutes": 0.03}
    c["stage2"]["guard"] = {"max_minutes": 0.08}
    c["stage2"]["eval_every_min"] = 0.03
    c["throughput"] = {"lengths": [24], "reps": 1}
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(c))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert.joint", "--config", str(cfgp), "--mix", str(mix), "--out", str(out)])
    joint.main()
    R = json.loads((out / "results.json").read_text())
    assert R["init_cmp"]["chosen"] in ("reuse_attn", "clone_neighbor")
    assert set(R["init_cmp"]["results"]) == {"reuse_attn", "clone_neighbor"}
    assert R["stage1"]["reuse_info"]["3"]["key_map"] == "proj" or R["stage1"]["init"] == "clone_neighbor"
    s2 = R["stage2"]
    assert s2["layers"] == ATTENTION_LAYERS and s2["tokens"] > 0 and len(s2["curve"]) >= 1
    assert s2["windows_used_after"] * 24 == s2["train_tokens_total"] and s2["stream_passes"] == 0
    fh = R["final_heldout"]
    assert set(fh) == {"web", "chat", "math", "mix"} and np.isfinite(fh["mix"]["kl"]) and fh["mix"]["kl"] >= 0
    assert R["mmlu_full"]["final"]["n"] == 40 and "converted_8" in R["throughput"]
    sd = torch.load(tmp_path / "ck" / "gdn" / "final.pt", weights_only=True)
    assert sorted(sd) == ATTENTION_LAYERS
    # the trained GDNs are not the stage-1 ones (stage 2 actually updated every layer)
    s1 = torch.load(tmp_path / "ck" / "gdn" / "stage1.pt", weights_only=True)
    assert all(not torch.equal(sd[i]["out_proj.weight"], s1[i]["out_proj.weight"]) for i in ATTENTION_LAYERS)
    d = rule.decide(R, _cmp(-3.0, -7.0), _export())
    assert d["verdict"] in ("STRONG", "FOLD_KL") and rule.valid_decision(d)
