"""jobs/q9af_v2 mode clefflash (README_clefflash.md): the clean v2 recipe on Clef-flash's weights, then proxy600 under
Clef's joint head (pair A) and the LM-logit v3w readout (pair B). CPU only."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs.q9af_v2 import diag  # noqa: E402

CFG_CLEAN = json.loads((ROOT / "configs/q9af_v2_clean_h100.json").read_text(encoding="utf-8"))
CFG_CF = json.loads((ROOT / "configs/q9af_v2_clefflash_h100.json").read_text(encoding="utf-8"))
P_CLEAN = json.loads((ROOT / "jobs/q9af_v2/job_params_clean.json").read_text(encoding="utf-8"))
P_CF = json.loads((ROOT / "jobs/q9af_v2/job_params_clefflash.json").read_text(encoding="utf-8"))
ARMS_SH = (ROOT / "jobs/q9af_v2/clefflash_arms.sh").read_text(encoding="utf-8")
RUNNER = (ROOT / "jobs/q9af_v2/q9af_v2_job.sh").read_text(encoding="utf-8")
CLEF_REV = "17f0b0ad64efb65d273590632833508766b2aae6"
Q9B_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


# ----------------------------------------------------------------------------- the recipe is 7ji99's (clean) recipe
def test_config_is_the_clean_recipe_on_clef_flash_weights():
    """Only WHICH weights change (model_dir -> Clef-flash; tokenizer pinned to the 9B's, so every window is 7ji99's)."""
    strip = {"_comment", "model_dir", "tokenizer_dir"}
    assert {k: v for k, v in CFG_CF.items() if k not in strip} == {k: v for k, v in CFG_CLEAN.items() if k not in strip}
    assert set(CFG_CF) - set(CFG_CLEAN) == {"tokenizer_dir"}
    assert CFG_CF["model_dir"] == "${SSDM_DATA_ROOT}/hf/models--Cloudflare--clef-flash/snapshots"
    assert CFG_CF["tokenizer_dir"] == CFG_CLEAN["model_dir"] == "${SSDM_DATA_ROOT}/hf/models--Qwen--Qwen3.5-9B/snapshots"
    assert CFG_CF["mix"]["decontam"] == "full-eval-v1" and CFG_CF["stage2"]["guard"]["max_minutes"] == 385


def test_fetch_stage_pins_clef_flash_and_the_9b_tokenizer_only():
    import ast

    src = (ROOT / "scripts/sf_fetch.py").read_text(encoding="utf-8")
    items = ast.literal_eval(re.search(r"^ITEMS = (\{.*?^\})", src, re.S | re.M).group(1).replace("MAMBA3_TOK, ", "'x', ")
                             .replace("MAMBA3_TOK_REV", "'x'"))
    cf, v2 = items["q9af_v2_clefflash"], items["q9af_v2"]
    by = {x[0]: x for x in cf}
    assert by["CLEF_FLASH_DIR"][1:] == ("Cloudflare/clef-flash", "model", None, CLEF_REV)
    assert by["Q9B_TOK_DIR"][1:] == ("Qwen/Qwen3.5-9B", "model", ["*.json", "*.txt", "*.jinja"], Q9B_REV)
    # every mixture / probe source exactly as job-7ji99 fetched it
    assert [x for x in cf if x[0] not in ("CLEF_FLASH_DIR", "Q9B_TOK_DIR")] == [x for x in v2 if x[0] != "Q9B_DIR"]


def test_tokenizer_dir_is_opt_in():
    from convert.model import tokenizer_dir

    a = {"model_dir": str(ROOT / "jobs")}
    with pytest.raises(FileNotFoundError):  # no config.json under jobs/: the model_dir is what it resolves
        tokenizer_dir(a)
    assert "tokenizer_dir" not in CFG_CLEAN  # 7ji99's config: the model's own tokenizer, unchanged


def test_params_keep_the_conversion_caps_and_fit_900_min():
    from jobs.common.costs import PARAMS as REG, QUOTE_USD_PER_MIN, quoted_max_cost_usd, worst_case_usd
    from jobs.q9af_v2.preflight import check

    res = check(P_CF, CFG_CF)
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    for ph in ("weights", "D", "P", "X"):
        assert P_CF["phase_caps_minutes"][ph] == P_CLEAN["phase_caps_minutes"][ph]
    for k in ("measured", "estimates_not_measured", "margin_min_frac", "reserve_minutes", "max_restarts",
              "build_timeout_minutes", "chip", "chip_count"):
        assert P_CF[k] == P_CLEAN[k], k
    assert set(P_CF["phase_caps_minutes"]) == {"setup", "weights", "D", "P", "X", "C_stock", "C_conv", "L_stock",
                                               "L_conv", "S"}
    assert P_CF["max_duration_minutes"] == 638 <= 900
    assert worst_case_usd(P_CF) == 43.97 <= P_CF["hard_cap_usd"] == 44.0
    assert P_CF["batch_usd_per_min"] == QUOTE_USD_PER_MIN and quoted_max_cost_usd(P_CF) == 40.19
    assert REG["q9af_v2_clefflash"] == "jobs/q9af_v2/job_params_clefflash.json"
    assert P_CF["model"] == f"Cloudflare/clef-flash@{CLEF_REV}" and P_CF["decision_rule"] == "jobs/q9af_v2/diag.py"
    assert P_CF["submit"]["command"].endswith("q9af_v2_job.sh clefflash")
    assert P_CF["submit"]["key"] not in (P_CLEAN["submit"]["key"],) and P_CF["submit"]["checks"] == P_CLEAN["submit"]["checks"]


def test_context_is_the_clean_image_plus_one_layer_and_pinned_tokenizer():
    ctx, base = P_CF["context"], P_CLEAN["context"]
    assert ctx["dockerfile"] == base["dockerfile"] == "jobs/q9af_v2/Dockerfile"
    (old, new), = ctx["rewrites"]
    df = (ROOT / ctx["dockerfile"]).read_text(encoding="utf-8")
    assert df.count(old) == 1 and new.endswith(old) and "transformers==5.10.2" in new and "torchvision==0.28.0" in new
    assert "--system-site-packages" in new and "'5.17.0', '2.13.0'" in new  # the system stack stays the conversion's
    assert [d["src"] for d in ctx["data"][:13]] == [d["src"] for d in base["data"]]
    pin = (ROOT / "jobs/q9af_v2/data_clefflash.sha256").read_text(encoding="utf-8")
    assert pin.startswith((ROOT / "jobs/q9af_v2/data_clean.sha256").read_text(encoding="utf-8"))
    assert all(f"/data/{d['dest']}" in pin for d in ctx["data"]) and len(ctx["data"]) == 15
    assert f"tokenizers/Qwen--Qwen3.5-9B@{Q9B_REV}/tokenizer.json" in pin


def test_runner_mode_and_arms():
    assert "clefflash) PARAMS=$CODE/jobs/q9af_v2/job_params_clefflash.json" in RUNNER
    assert 'VALIDATOR=jobs.q9af_v2.diag:valid_decision' in RUNNER and '"$VALIDATOR"' in RUNNER
    assert RUNNER.index("clef_arms") > RUNNER.index('LAYERS=${LAYERS%') and RUNNER.index("clef_arms") < RUNNER.index(
        "# ------------------------------------------------------------------ I: Index proxy600")
    assert "baselines.clef_index_engine:ClefIndexEngine" in ARMS_SH and "jobs.q9af_v2.clef_af:ClefAFIndexEngine" in ARMS_SH
    assert "--option gdn_path=reference" in ARMS_SH and "family=q9af-probe-v3w" in ARMS_SH
    assert re.search(r"V3W_IDENTITY=" + P_CF["clef_arms"]["v3w_identity_sha256"], ARMS_SH)
    assert "--edition 0.3 --omit-gsm8k03" in ARMS_SH and "--edition 0.2.1" in ARMS_SH


def test_v3w_check_receipt():
    """scripts/check_clefflash_v3w.py on the 644 proxy600 rows: 255 single-token codes, every scored token single and
    distinct, the readout identity pinned in the job; Clef-flash's own tokenizer differs on 1 of 3,730 questions."""
    r = json.loads((ROOT / "analysis/check_clefflash_v3w.json").read_text(encoding="utf-8"))
    assert r["ok"] and r["identity_sha256"] == P_CF["clef_arms"]["v3w_identity_sha256"]
    c = r["counts"]
    assert c["rows"] == 644 and c["q_scored_tokens_single_and_distinct"] == c["q_words"] + c["q_codes"] == c["questions"]
    assert r["codes"]["same_codes_and_ids"] and c["q_fallback_multitoken"] == 0
    vol = json.loads((ROOT / P_CF["clef_arms"]["v3w_volume"]).read_text(encoding="utf-8"))
    assert vol["identity_sha256"] == r["identity_sha256"] and vol["totals"]["fork"] == r["volume_fork_totals"]
    assert vol["rows"] + vol["refused"] == 644


# ----------------------------------------------------------------------------- the diagnostic report
def _write_idx(tmp, point=-5.0, sha=diag.PROXY_IDS_SHA256, drop=None):
    for sfx in ("", "_03"):
        for pair in ("clef", "v3w"):
            name = f"compare_{pair}_conv_vs_stock{sfx}.json"
            if name != drop:
                (tmp / name).write_text(json.dumps({"subset_ids_sha256": sha, "B": 2000, "seed": 0,
                                                    "diff": {"index": {"point": point, "lo": point - 3, "hi": point + 3}}}))
            for arm in ("stock", "conv"):
                (tmp / f"score_{pair}_{arm}{sfx}.json").write_text(json.dumps(
                    {"slice_index": {"index": {"point": 50.0 + (point if arm == "conv" else 0)}}}))


EXP = {"ok": True, "mode": "final", "attention_free": True, "converted": diag.ATTENTION_LAYERS}
RES = {"final_heldout": {"mix": {"kl": 0.12}}}


def test_diag_complete_measurement(tmp_path):
    _write_idx(tmp_path)
    d = diag.decide(RES, tmp_path, EXP)
    assert d["verdict"] == "DIAGNOSTIC" and diag.valid_decision(d) and not d["problems"]
    assert d["numbers"]["delta_A_joint_head_0.2.1_point"] == -5.0 and d["numbers"]["score_v3w_conv_0.3"] == 45.0
    assert d["report"]["keep_discard_rule"] is None
    assert d["attention_free_system"] == {"A_joint_head": False, "B_v3w_lm_logit": True}


@pytest.mark.parametrize("bad", ["sha", "drop", "export", "kl"])
def test_diag_incomplete_is_no_measurement(tmp_path, bad):
    _write_idx(tmp_path, sha="0" * 64 if bad == "sha" else diag.PROXY_IDS_SHA256,
               drop="compare_v3w_conv_vs_stock_03.json" if bad == "drop" else None)
    d = diag.decide({} if bad == "kl" else RES, tmp_path, {**EXP, "converted": [3]} if bad == "export" else EXP)
    assert d["verdict"] == "NO_MEASUREMENT" and d["problems"] and not diag.valid_decision(d)


# ----------------------------------------------------------------------------- arm-pair B backend (prefix fork)
class _Ops:
    """Fake prefix-fork ops: next-token logits are a deterministic function of the full id sequence."""

    V = 50

    def __init__(self):
        self.extends = []

    def _logits(self, seq):
        import torch

        g = torch.Generator().manual_seed(hash(tuple(seq)) % (2 ** 31))
        return torch.randn(self.V, generator=g)

    def prefill(self, ids):
        return None, list(ids[0].tolist())

    def extend(self, state, ids, n_last):
        import torch

        self.extends.append(ids.shape[0])
        out = torch.zeros(ids.shape[0], ids.shape[1], self.V)
        for b in range(ids.shape[0]):
            row = ids[b].tolist()
            for t in range(ids.shape[1]):
                out[b, t] = self._logits(state + row[: t + 1])
        return out, state


def _fake_v3w(prefix_len=40):
    pre = list(range(1, prefix_len + 1))
    qs = [{"qid": f"q{i}", "prompt": pre + [10 + i, 20 + i], "options": ["a", "b", "c"], "coded": True,
           "tokens": [30, 31, 32 + i], "words": False} for i in range(5)]
    return SimpleNamespace(codes=[{"code": "A", "token_id": 30}], table=["A"], questions=lambda req: qs,
                           identity=lambda: {"id": "jrt-v3w-generic:test", "sha256": "ab" * 32}), qs


def _plugin(ops, budget=None):
    from decide.backends.hf_readout import ModelPlugin

    return ModelPlugin(family="qwen3.5", model_id="tiny", snapshot="s", encode=lambda t, trusted=True: [1],
                       forward=None, state_ops=ops, device="cpu", fork_state_token_budget=budget,
                       files={"weights": "w" * 64, "gdn_ckpt": "g" * 64})


@pytest.mark.parametrize("budget,want_packs", [(None, [5]), (80, [2, 2, 1])])
def test_v3w_backend_scores_the_v3w_tokens(budget, want_packs):
    torch = pytest.importorskip("torch")
    from decide.backends.v3w_generic_backend import V3WGenericBackend

    ops = _Ops()
    v, qs = _fake_v3w()
    be = V3WGenericBackend(_plugin(ops, budget), v)
    res = be.score({"questions": {}}, lambda: None)
    assert ops.extends == want_packs  # KV fork budget: rows x prefix (40) <= 80 -> 2 rows per pack
    for q in qs:
        lp = torch.log_softmax(ops._logits(q["prompt"]).float(), -1)
        for o, t in zip(q["options"], q["tokens"]):
            assert math.isclose(res.logits[q["qid"]][o], float(lp[t]), rel_tol=1e-5, abs_tol=1e-5)
    rev = be.revision()
    assert rev["backend"] == "v3w_generic" and ";gdn=gggggggggggg;" in rev["revision_id"]
    assert "prompt=jrt-v3w-generic:test.abababab" in rev["revision_id"]
    plan = be.plan({"questions": {}})
    assert plan.inference_calls == 1 + len(want_packs) and plan.max_call_input_tokens == 43


# ----------------------------------------------------------------------------- arm-pair A: the converted Clef backbone
def _tiny_clef():
    """A ClefModel-shaped wrapper (language_model.model.language_model = a tiny 32-layer Qwen3.5 text model with the 9B
    head-layout ratios; attention at 3, 7, ..., 31)."""
    import torch
    import torch.nn as nn
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

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
    inner = nn.Module()
    inner.language_model = lm
    base = nn.Module()
    base.model = inner
    clef = nn.Module()
    clef.language_model = base
    return clef.eval(), lm, cfg


def test_swap_converted_equals_the_conversions_student_path(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5")
    from types import SimpleNamespace as NS

    import torch.nn as nn

    from convert.gdn import GDN
    from convert.model import ConvertibleQwen
    from jobs.q9af_v2.clef_af import ATTENTION_LAYERS, StudentMixer, swap_converted

    clef, lm, cfg = _tiny_clef()
    torch.manual_seed(1)
    sd = {}
    for i in ATTENTION_LAYERS:
        g = GDN.from_config(cfg, kernel="chunk")
        for p in g.parameters():
            torch.nn.init.normal_(p, std=0.1)
        sd[i] = {k: v.clone() for k, v in g.state_dict().items()}
    ck = tmp_path / "gdn_converted.pt"
    torch.save(sd, ck)
    ids = torch.randint(0, 101, (1, 23))
    # reference: the conversion's own student path (convert.model.ConvertibleQwen, every attention layer -> student)
    import copy

    ref_lm = copy.deepcopy(lm)
    m = ConvertibleQwen(NS(model=NS(language_model=ref_lm), lm_head=nn.Linear(32, 101, bias=False)), kernel="chunk")
    for i in ATTENTION_LAYERS:
        g = GDN.from_config(cfg, kernel="chunk")
        g.load_state_dict(sd[i])
        m.swaps[i].gdn = g
    m.set_modes({i: "student" for i in ATTENTION_LAYERS})
    with torch.no_grad():
        want = m.hidden(ids)
        stock = lm(input_ids=ids, use_cache=False).last_hidden_state
    info = swap_converted(clef, ck, ",".join(map(str, ATTENTION_LAYERS)), kernel="chunk")
    assert info["attention_free_backbone"] and not info["attention_free_system"]
    assert all(isinstance(lm.layers[i].self_attn, StudentMixer) for i in ATTENTION_LAYERS)
    with torch.no_grad():
        got = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    assert torch.allclose(got, want, atol=1e-5) and not torch.allclose(got, stock, atol=1e-3)
    with pytest.raises(ValueError):  # anything but ALL attention layers is refused
        swap_converted(_tiny_clef()[0], ck, "3,7", kernel="chunk")


# ----------------------------------------------------------------------------- shell dry run (mode clefflash)
@pytest.mark.skipif(__import__("shutil").which("bash") is None, reason="bash not available")
def test_shell_dry_run_clefflash_mode(tmp_path):
    pytest.importorskip("torch")
    from tests.test_q9af_v2 import _dry_root, run_dry

    root = _dry_root(tmp_path)
    rc, out = run_dry(root, {"Q9AF2_STUB_POINT": "-6.0", "Q9AF2_STUB_LO": "-9.5"}, mode="clefflash")
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, log[-4000:]
    assert "clefflash mode: configs/q9af_v2_clefflash_h100.json" in log
    dec = json.loads((out / "decision.json").read_text())
    assert dec["verdict"] == "DIAGNOSTIC" and dec["rule_version"] == diag.RULE_VERSION
    assert dec["numbers"]["delta_A_joint_head_0.2.1_point"] == -6.0
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert {m["phase"] for m in man} >= {"setup", "D", "P", "X", "C_stock", "C_conv", "L_stock", "L_conv", "S", "final"}
    assert not {m["phase"] for m in man} & {"I_stock", "I_conv"} and all(m["ok"] for m in man)
    assert json.loads((out / "result.json").read_text())["ok"] and (out / "Q9AF2_DONE").exists()
    pre = json.loads((root / "work" / "out" / "preflight.json").read_text())
    assert pre["checks"]["schedule"]["max_duration"] == 638 and pre["checks"]["hard_cap"]["worst_case_usd"] == 43.97
