"""jobs/q9af_v3 (Q9-AF v3, recall redesign: widened GDN, recall-heavy KD incl. teacher-generated retrieval / tool /
knowledge rows and synthetic recall rows, recall-ordered per-group conversion gated on MQAR / NIAH + proxy600 with a STOP
rule, the cheap per-layer sensitivity job S, 32k long context + on-policy; gate = proxy600 under the generic v3w readout
+ recall): rules, data plumbing, losses, cached sampling, the tiny-model pipeline end to end (CPU), preflight / params /
registration, artifact pins, and the shell dry runs of S, A and B."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobs.q9af_v3 import rule  # noqa: E402
from jobs.q9af_v3.rule import (ATTENTION_LAYERS, BOOTSTRAP_B, BOOTSTRAP_SEED, CUM_MAX_DROP, PROMOTE_LO,  # noqa: E402
                               PROXY_IDS_SHA256, REFERENCES, RULE_VERSION, SENS_GO_LAYERS, SENS_MAX_DROP,
                               SENS_VERSION, STAGE_A_VERSION, STAGE_INDEX_FLOOR, STAGE_MAX_DROP, V3W_IDENTITY)

CFG = json.loads((ROOT / "configs/q9af_v3_h100.json").read_text(encoding="utf-8"))
PA = json.loads((ROOT / "jobs/q9af_v3/job_params_A.json").read_text(encoding="utf-8"))
PB = json.loads((ROOT / "jobs/q9af_v3/job_params_B.json").read_text(encoding="utf-8"))
PS = json.loads((ROOT / "jobs/q9af_v3/job_params_S.json").read_text(encoding="utf-8"))
CFG_S = json.loads((ROOT / "configs/q9af_v3_sens_h100.json").read_text(encoding="utf-8"))
RUNNER = (ROOT / "jobs/q9af_v3/q9af_v3_job.sh").read_text(encoding="utf-8")
SHA = "ab" * 32


# ----------------------------------------------------------------------------- the pre-registered gate
def test_rule_constants_are_pinned():
    assert RULE_VERSION == "q9af-v3-rule-v1" and STAGE_A_VERSION == "q9af-v3-stageA-v1"
    assert ATTENTION_LAYERS == [3, 7, 11, 15, 19, 23, 27, 31] == CFG["attention_layers"]
    assert PROMOTE_LO == 0.0
    assert PROXY_IDS_SHA256 == json.loads((ROOT / "jobs/common/dev_proxy600.json").read_text())["ids_sha256"]
    assert (BOOTSTRAP_B, BOOTSTRAP_SEED) == (2000, 0)
    assert REFERENCES == ("v2", "stock")
    assert V3W_IDENTITY == "2e4907ee4302d9d196052fc04c5618e88fc8058446644715345eaad93a3a2543"
    assert json.loads((ROOT / PB["v3w_volume"]).read_text())["identity_sha256"] == V3W_IDENTITY
    assert set(rule.ACTIONS) == set(rule.VERDICTS)
    # recall gates (pre-registered after v2's recall fold: MQAR drop 0.59, NIAH drop 0.525)
    assert SENS_VERSION == "q9af-v3-sens-v1" and (SENS_MAX_DROP, SENS_GO_LAYERS) == (0.05, 6)
    assert (STAGE_MAX_DROP, CUM_MAX_DROP, STAGE_INDEX_FLOOR) == (0.05, 0.20, -10.0)


def _exp(**kw):
    return {"ok": True, "mode": "final", "converted": list(ATTENTION_LAYERS), "attention_free": True, **kw}


def _cmp(point, lo, sha=PROXY_IDS_SHA256, B=2000, seed=0):
    return {"subset_ids_sha256": sha, "B": B, "seed": seed, "diff": {"index": {"point": point, "lo": lo, "hi": point + 4}}}


def _params(v2_pinned):
    return {"v2": {"artifact_sha256": SHA if v2_pinned else None}}


def _res():
    return {"final": {"heldout": {"mix": {"kl": 0.08}}, "long": {"kl": 0.15}, "gen_heldout": {"kl": 0.1},
                      "recall_drop_vs_stock": {"mqar_drop": 0.1, "niah_drop": 0.05}},
            "entry": {"heldout": {"mix": {"kl": 0.09}}, "long": {"kl": 0.2}}}


def test_reference_is_fixed_by_the_committed_params():
    assert rule.expected_reference(_params(True)) == "v2" and rule.expected_reference(_params(False)) == "stock"
    assert rule.expected_reference(PB) == ("v2" if PB["v2"]["artifact_sha256"] else "stock")


@pytest.mark.parametrize("v2,point,lo,want", [
    (True, 2.0, 0.01, "PROMOTE"), (True, 2.0, 0.0, "REPORT"), (True, -1.0, -4.0, "REPORT"),
    (False, 3.0, 0.5, "PROMOTE"), (False, -6.0, -9.5, "REPORT"), (False, 0.5, -0.0001, "REPORT")])
def test_gate(v2, point, lo, want):
    d = rule.decide(_params(v2), _res(), _exp(), _exp() if v2 else None, _cmp(point, lo), _cmp(-5, -8),
                    _cmp(-6, -9) if v2 else None)
    assert d["verdict"] == want and d["reference"] == ("v2" if v2 else "stock") and rule.valid_decision(d)
    assert d["numbers"]["heldout_kl_mix"] == 0.08 and d["numbers"]["diff_vs_ref_lo"] == lo


@pytest.mark.parametrize("bad", [
    {"export": None}, {"export": _exp(converted=[3, 7, 11, 15, 19, 23, 31])}, {"export": _exp(mode="salvage")},
    {"v2_export": None}, {"v2_export": _exp(attention_free=False)}, {"cmp": _cmp(1, 0.5, sha="0" * 64)},
    {"cmp": _cmp(1, 0.5, B=1000)}, {"cmp": None}, {"reference": "stock"}, {"cmp": {"subset_ids_sha256": PROXY_IDS_SHA256,
                                                                                  "B": 2000, "seed": 0}}])
def test_no_measurement(bad):
    d = rule.decide(_params(True), _res(), bad.get("export", _exp()) if "export" in bad else _exp(),
                    bad["v2_export"] if "v2_export" in bad else _exp(), bad["cmp"] if "cmp" in bad else _cmp(1, 0.5),
                    _cmp(0, -1), _cmp(0, -1), reference=bad.get("reference"))
    assert d["verdict"] == "NO_MEASUREMENT" and d["problems"] and not rule.valid_decision(d)


def test_recall_is_binding_for_promotion():
    res = _res()
    res["final"]["recall_drop_vs_stock"]["niah_drop"] = 0.25
    d = rule.decide(_params(False), res, _exp(), None, _cmp(3, 1), _cmp(3, 1), None)
    assert d["verdict"] == "REPORT" and not d["recall_ok"] and rule.valid_decision(d)
    del res["final"]["recall_drop_vs_stock"]
    assert rule.decide(_params(False), res, _exp(), None, _cmp(3, 1), _cmp(3, 1), None)["verdict"] == "NO_MEASUREMENT"


def _stages(drops, stock=(0.9, 0.95)):
    st, mq, nh = {}, stock[0], stock[1]
    for k, (dm, dn) in enumerate(drops, 1):
        mq, nh = mq - dm, nh - dn
        st[f"stage{k}"] = {"adds": [k], "converted": list(range(k)), "recall": {"mqar_mean": mq, "niah_mean": nh}}
    return {"recall_stock": {"mqar_mean": stock[0], "niah_mean": stock[1]}, "stages": st}


@pytest.mark.parametrize("drops,k,point,want,why", [
    ([(0.02, 0.01)], 1, -3.0, "CONTINUE", None), ([(0.06, 0.0)], 1, -3.0, "STOP", "mqar stage"),
    ([(0.0, 0.051)], 1, -3.0, "STOP", "niah stage"), ([(0.0, 0.0)], 1, -10.5, "STOP", "proxy600"),
    ([(0.05, 0.05)] * 4, 4, -9.0, "CONTINUE", None), ([(0.05, 0.05)] * 3 + [(0.051, 0.0)], 4, -9.0, "STOP", "cumulative"),
    ([(0.04, 0.0), (0.0, 0.04)], 2, -10.0, "CONTINUE", None)])
def test_stage_gate(drops, k, point, want, why):
    g = rule.stage_gate(_stages(drops), k, _cmp(point, point - 3))
    assert g["verdict"] == want, g
    assert (why is None and not g["breaches"]) or any(why in b for b in g["breaches"])
    assert rule.stage_gate(_stages(drops), k, None)["verdict"] == "NO_MEASUREMENT"


def test_stopped_stage_a_record_is_valid_and_never_chains():
    gates = [rule.stage_gate(_stages([(0.0, 0.0), (0.2, 0.0)]), k, _cmp(-2, -5)) for k in (1, 2)]
    d = rule.stage_a({"plan": {"order": ATTENTION_LAYERS}}, {"completion_tokens": 5}, _exp(mode="salvage"), gates)
    assert d["verdict"] == "STOPPED" and d["stopped_at_stage"] == 2 and rule.valid_stage_a(d)
    assert "mqar" in " ".join(d["breaches"]) and "no job B" in d["action"]


def _table(drops, base=None):
    t = {f"wide_L{i}": {"mqar_drop": m, "niah_drop": n, "kl": 0.01} for i, (m, n) in zip(ATTENTION_LAYERS, drops)}
    for i, (m, n) in (base or {}).items():
        t[f"base_L{i}"] = {"mqar_drop": m, "niah_drop": n}
    return {"recall_stock": {"mqar_mean": 0.9, "niah_mean": 0.95}, "table": t}


@pytest.mark.parametrize("drops,want,n", [
    ([(0.01, 0.0)] * 8, "GO", 8), ([(0.05, 0.05)] * 6 + [(0.3, 0.1)] * 2, "GO", 6),
    ([(0.0, 0.0)] * 5 + [(0.06, 0.0)] * 3, "PARTIAL", 5), ([(0.051, 0.0)] * 8, "NO_GO", 0)])
def test_sensitivity_verdicts(drops, want, n):
    d = rule.sensitivity(_table(drops, {27: (0.4, 0.3)}))
    assert d["verdict"] == want and d["n_pass"] == n and rule.valid_sensitivity(d)
    assert d["capacity_control"]["27"]["base_mqar_drop"] == 0.4
    bad = rule.sensitivity(_table(drops[:7]))
    assert bad["verdict"] == "NO_MEASUREMENT" and not rule.valid_sensitivity(bad)


def test_kl_never_gates():
    """A model far from its teacher in KL but ahead on the Index is still promoted (KL is reported, v2's lesson)."""
    res = _res()
    res["final"]["heldout"]["mix"]["kl"] = 0.9  # recall within bounds
    assert rule.decide(_params(False), res, _exp(), None, _cmp(2, 0.5), _cmp(2, 0.5), None)["verdict"] == "PROMOTE"


def test_stage_a_record():
    res = {"plan": {"order": [27, 3, 19, 7, 31, 11, 23, 15], "stages": [{}] * 4},
           "stages": {f"stage{k}": {} for k in range(1, 5)}, "final": {"heldout": {"mix": {"kl": 0.1}}}}
    d = rule.stage_a(res, {"completion_tokens": 10}, _exp())
    assert d["verdict"] == "STAGE_A_DONE" and rule.valid_stage_a(d)
    # the runner passes phase G's whole results.json ({"gen": {...}, "env": ...})
    assert rule.stage_a(res, {"env": {}, "gen": {"completion_tokens": 10}}, _exp())["verdict"] == "STAGE_A_DONE"
    d2 = rule.stage_a({**res, "stages": {"stage1": {}}}, {"completion_tokens": 10}, _exp())
    assert d2["verdict"] == "NO_MEASUREMENT" and not rule.valid_stage_a(d2)
    assert not rule.valid_stage_a(rule.stage_a(res, {}, _exp(converted=[3])))


# ----------------------------------------------------------------------------- data plumbing (convert.gendata)
def test_block_split_disjoint_and_deterministic():
    from convert.gendata import BlockSplit

    s = BlockSplit(100 * 32 + 5, 4, 32, 6, seed=3)
    assert len(s.lc_blocks) == 6 and s.identity()["lc_tokens"] == 192
    w = s.short_windows()
    assert len(w) == (100 * 32 + 4) // 4 - 6 * 8
    assert not set((w // 8).tolist()) & set(s.lc_blocks.tolist())
    assert (BlockSplit(100 * 32 + 5, 4, 32, 6, seed=3).lc_blocks == s.lc_blocks).all()
    b = int(s.lc_blocks[0])
    assert not s.doc_ok(b * 32 - 2, b * 32 + 1) and s.doc_ok(*(((b + 1) * 32, (b + 1) * 32 + 3)
                                                             if not s.is_lc[b + 1] else (0, 0 + 1)))
    toks = np.arange(100 * 32 + 5, dtype=np.uint32)
    lc = s.lc_tokens(toks)
    assert len(lc) == 192 and lc[0] == b * 32
    with pytest.raises(ValueError):
        BlockSplit(1000, 3, 32, 1, 0)


def _stream(eos=0):
    """web docs [5..], chat docs with a header, math docs; returns toks, mix, header."""
    rng = np.random.default_rng(1)
    header = [90, 91, 92]
    toks, mix = [], {"sources": {}}
    for s, nd in (("web", 30), ("chat", 30), ("math", 20)):
        a = len(toks)
        for k in range(nd):
            body = list(rng.integers(5, 80, int(rng.integers(10, 30))))
            toks += (body[:4] + header + body[4:] if s == "chat" else body) + [eos]
        mix["sources"][s] = {"tokens": len(toks) - a}
    return np.asarray(toks, np.uint32), mix, header


def test_prompts_and_split_counts():
    from convert.gendata import BlockSplit, doc_spans, select_prompts, source_bounds, split_counts

    toks, mix, header = _stream()
    split = BlockSplit(len(toks), 4, 16, 2, seed=0)
    spans = doc_spans(toks, 0)
    assert (toks[spans[:, 1] - 1] == 0).all() and spans[0, 0] == 0
    cfg = {"doc_prompt_tokens": 6, "min_continuation": 2, "max_prompt_tokens": 40}
    counts = split_counts(20, {"chat": 0.6, "web": 0.25, "math": 0.15})
    assert counts == {"web": 5, "chat": 12, "math": 3} and sum(counts.values()) == 20
    p, used, st = select_prompts(toks, spans, source_bounds(mix), split, header, counts, cfg, seed=4)
    for q in p:
        a, b = spans[q["doc"]]
        assert split.doc_ok(int(a), int(b))
        if q["src"] == "chat":
            assert list(q["ids"][-3:]) == header and len(q["ids"]) == 4 + 3
        else:
            assert len(q["ids"]) == 6 and list(q["ids"]) == list(toks[a:a + 6])
    p2, _, _ = select_prompts(toks, spans, source_bounds(mix), split, header, counts, cfg, seed=4)
    assert [q["doc"] for q in p] == [q["doc"] for q in p2]
    p3, _, _ = select_prompts(toks, spans, source_bounds(mix), split, header, counts, cfg, seed=5, exclude=used)
    assert not used & {q["doc"] for q in p3}


def test_pack_masks_only_completions(tmp_path):
    from convert.gendata import load_prompts, pack, pred_weights, save_prompts

    R, M, st = pack([([1, 2], [3, 4, 5]), ([6], [7]), ([8] * 9, [9]), ([1], [])], seq_len=8, pad_id=0)
    assert st["samples"] == 2 and st["dropped_too_long"] == 1 and st["dropped_empty"] == 1
    assert R.tolist() == [[1, 2, 3, 4, 5, 6, 7, 0]] and M.tolist() == [[0, 0, 1, 1, 1, 0, 1, 0]]
    assert pred_weights(M).tolist() == [[0, 1, 1, 1, 0, 1, 0, 0]]
    ps = [{"src": "chat", "doc": 3, "ids": np.asarray([1, 2, 3], np.uint32)},
          {"src": "web", "doc": 9, "ids": np.asarray([4], np.uint32)}]
    save_prompts(tmp_path / "p.npz", ps)
    back = load_prompts(tmp_path / "p.npz")
    assert [(q["src"], q["doc"], q["ids"].tolist()) for q in back] == [("chat", 3, [1, 2, 3]), ("web", 9, [4])]


def test_mixed_stream_schedule_reads_each_gen_row_once():
    pytest.importorskip("torch")
    from convert.gendata import IndexedWindowStream, MixedStream

    toks = np.arange(64 * 4 + 1, dtype=np.uint32)
    g = IndexedWindowStream(toks, 4, 2, np.arange(0, 64, 2), seed=0)
    rows = np.arange(10 * 4, dtype=np.uint32).reshape(10, 4)
    ms = MixedStream(g, rows, np.ones((10, 4), np.uint8), 2, 0.25, seed=0)
    kinds, seen = [], []
    for _ in range(40):
        x, w, k = next(ms)
        kinds.append(k)
        if k == "gen":
            seen += x[:, 0].tolist()
            assert w.shape == x.shape and float(w[:, -1].sum()) == 0
        else:
            assert all(int(v) // 4 % 2 == 0 for v in x[:, 0])  # only allowed (even) windows
    assert kinds[:8] == ["generic", "generic", "generic", "gen"] * 2
    assert sorted(seen) == sorted(rows[:, 0].tolist()) and ms.state()["gen_rows_used"] == 10
    assert kinds.count("gen") == 5 and g.passes >= 0


# ----------------------------------------------------------------------------- the order rule
def test_conversion_plan_is_descending_sensitivity_with_share_budgets():
    from convert.distill import conversion_plan

    sens = {3: 0.02, 7: 0.05, 11: 0.01, 15: 0.01, 19: 0.03, 23: 0.04, 27: 0.09, 31: 0.0}
    p = conversion_plan(sens, 2, 360, 45)
    assert p["order"] == [27, 7, 23, 19, 3, 11, 15, 31]
    assert [s["adds"] for s in p["stages"]] == [[27, 7], [23, 19], [3, 11], [15, 31]]
    assert p["stages"][-1]["converted"] == ATTENTION_LAYERS
    assert abs(sum(s["minutes"] for s in p["stages"]) - 360) < 1e-6
    assert all(s["minutes"] >= 45 for s in p["stages"]) and p["stages"][0]["minutes"] > p["stages"][-1]["minutes"]
    assert abs(p["stages"][0]["minutes"] - (45 + 180 * 0.14 / 0.25)) < 1e-3
    eq = conversion_plan({i: 0.0 for i in ATTENTION_LAYERS}, 2, 360, 45)
    assert eq["order"] == ATTENTION_LAYERS and all(abs(s["minutes"] - 90) < 1e-9 for s in eq["stages"])
    with pytest.raises(ValueError):
        conversion_plan(sens, 2, 100, 45)
    asc = conversion_plan(sens, 2, 520, 60, descending=False, tiebreak={11: 0.5, 15: 0.1})
    assert asc["order"] == [31, 15, 11, 3, 19, 23, 7, 27]  # recall-hardest (27) LAST; tie 11/15 by KL
    assert asc["stages"][-1]["adds"] == [7, 27] and asc["stages"][-1]["minutes"] > asc["stages"][0]["minutes"]
    dc = CFG["distill"]
    assert (dc["group_size"], dc["kd_minutes"], dc["floor_minutes"], dc["ce_weight"], dc["order"]) == \
        (2, 520, 60, 0.25, "ascending_recall")
    assert CFG["mix_fracs"] == {"generic": 0.35, "gen": 0.25, "recall": 0.40} and CFG["gdn"]["dk"] == 256


# ----------------------------------------------------------------------------- losses + cached sampling (tiny model)
def _tiny():
    from tests.test_q9af_v2 import _tiny_model

    return _tiny_model()


def _attach_all(m, how="reuse_attn"):
    from convert.model import attach_gdn, init_gdn

    for i in m.attn_layers:
        attach_gdn(m, i, init_gdn(m, i, how, 0))


def test_weighted_losses_match_direct_computation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert.distill import distill_accumulate

    m = _tiny()
    _attach_all(m)
    torch.manual_seed(0)
    x = torch.randint(0, 101, (4, 24))
    w = (torch.rand(4, 24) > 0.4).float()
    modes = {3: "student", 11: "student"}
    with torch.no_grad():
        with m.modes({}):
            lt = m.logits(x).float().log_softmax(-1)
        with m.modes(modes):
            ls = m.logits(x).float().log_softmax(-1)
    wv = w.clone()
    wv[:, -1] = 0
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    fkl = ((lt.exp() * (lt - ls)).sum(-1) * wv).sum() / wv.sum()
    rkl = ((ls.exp() * (ls - lt)).sum(-1) * wv).sum() / wv.sum()
    ce = (-(ls.gather(-1, y[..., None])[..., 0]) * wv).sum() / wv.sum()
    grads = {}
    for mode, mb in (("fkl", None), ("fkl", 1), ("rkl", 2)):
        for p in m.gdn_parameters():
            p.grad = None
        r = distill_accumulate(m, modes, x, w, mb, mode, ce_weight=0.5 if mode == "fkl" else 0.0, chunk=7)
        want = fkl if mode == "fkl" else rkl
        assert abs(r["div"] - float(want)) < 1e-4 and abs(r["weight"] - float(wv.sum())) < 1e-6
        if mode == "fkl":
            assert abs(r["ce"] - float(ce)) < 1e-4
        grads[(mode, mb)] = torch.cat([p.grad.flatten() for p in m.gdn_parameters([3, 11])])
    # micro-batching is exact (the loss is normalised by the FULL batch's weight)
    assert torch.allclose(grads[("fkl", None)], grads[("fkl", 1)], atol=1e-6, rtol=1e-4)
    assert grads[("rkl", 2)].abs().sum() > 0


@pytest.mark.parametrize("modes", [{}, "all"])
def test_cached_greedy_generation_equals_full_recompute(modes):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert.distill import generate

    m = _tiny()
    _attach_all(m)
    if modes == "all":
        modes = {i: "student" for i in m.attn_layers}
    prompts = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14]]
    comps, st = generate(m, modes, prompts, max_new=6, stop_ids=[100], pad_id=0, temperature=0.0)
    P = 5
    for p, c in zip(prompts, comps):
        seq = [0] * (P - len(p)) + p
        want = []
        with torch.no_grad(), m.modes(modes):
            for _ in range(6):
                nxt = int(m.logits(torch.tensor([seq])).float()[0, -1].argmax())
                want.append(nxt)
                seq.append(nxt)
                if nxt == 100:
                    break
        assert c == want, (c, want)
    assert st["completion_tokens"] == sum(len(c) for c in comps)
    # sampling is seeded and stops at a stop token
    g = torch.Generator().manual_seed(1)
    a, _ = generate(m, modes, prompts, 6, [100], 0, 1.0, g)
    g = torch.Generator().manual_seed(1)
    b, _ = generate(m, modes, prompts, 6, [100], 0, 1.0, g)
    assert a == b


def test_recurrent_kernel_is_autocast_safe():
    torch = pytest.importorskip("torch")
    from convert.gdn import gdr_recurrent

    q, k = torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4)
    v, g, b = torch.randn(1, 3, 2, 4), -torch.rand(1, 3, 2), torch.rand(1, 3, 2)
    o, S = gdr_recurrent(q, k, v, g, b)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o2, S2 = gdr_recurrent(q, k, v, g, b)
    assert S2.dtype == torch.float32 and torch.allclose(S, S2) and torch.allclose(o, o2.float())


# ----------------------------------------------------------------------------- the pipeline end to end (tiny, CPU)
def _tiny_cfg(tmp_path):
    c = json.loads(json.dumps(CFG))
    c.update(seq_len=24, batch=4, micro_batch=2, model_dir="tiny", ckpt_dir=str(tmp_path / "ck" / "gdn"),
             heldout_windows={"web": 8, "chat": 4, "math": 4}, curve_windows_per_source=4)
    c.pop("vram_fraction")
    c["lc"] = {"seq_len": 96, "blocks": 6}
    c["gen"].update(n_prompts=40, batch=8, max_new=8, max_minutes=0.5, target_completion_tokens=10 ** 9,
                    heldout_rows=2, doc_prompt_tokens=8, min_continuation=4, max_prompt_tokens=16)
    c["gen"]["onpolicy"].update(n_prompts=20, max_prompt_tokens=16)
    c["gen"].update(check_templates=False, max_prompt_tokens=80)
    c["gen"]["instruct"] = {"doc_tokens": 8, "min_doc_tokens": 10}
    c["gdn"] = {"dk": 16}  # tiny: 2 heads x 16 = the attention head dim (the 9B's 16 x 256), 2x the config's 8
    c["recall_data"] = {"kv_pairs": [2, 4], "kv_queries": 2, "quote_questions": 1, "task_shares": [0.35, 0.35, 0.3],
                        "key_pool": 30, "key_scan": 3000, "key_filter": "any", "min_key_pool": 8}
    c["stage1"]["guard"] = {"max_minutes": 0.02}
    c["distill"].update(kd_minutes=0.16, floor_minutes=0.02, eval_every_min=0.02, guard={})
    c["long_eval"] = {"source": "web", "seq_len": 96, "windows": 2}
    c["long_context"].update(seq_len=96, batch=2, micro_batch=1, eval_every_min=0.02, guard={"max_minutes": 0.05})
    c["onpolicy"].update(rollout_batch=4, max_new=6, micro_batch=2, eval_every_min=0.02, guard={"max_minutes": 0.05})
    c["throughput"] = None
    return c


def _fake_recall(monkeypatch, v3):
    """probes.core MQAR / NIAH are CUDA-only: CPU stand-ins whose accuracy falls with each converted layer (a
    layer-specific amount), so the probe, the order rule and the gates see real, mode-dependent numbers."""
    from probes import core as pcore

    def acc(model):
        st = [i for i, md in model.get_modes().items() if md == "student"]
        return max(0.0, 0.95 - sum(0.01 + 0.004 * (i % 7) for i in st))

    monkeypatch.setattr(pcore, "mqar", lambda model, tok, n_pairs, n_seq: {"mean_acc": acc(model)})
    monkeypatch.setattr(pcore, "niah", lambda model, tok, eval_toks, lengths, n_per: {"mean_acc": acc(model) + 0.02})
    real_build, real_load = v3.data.build, v3.data.load_tokens
    monkeypatch.setattr(v3.data, "build", lambda tok, s, max_tokens=None: ("eval-split", {}))
    monkeypatch.setattr(v3.data, "load_tokens", lambda p: np.arange(4000, dtype=np.uint32) % 90 if p == "eval-split"
                        else real_load(p))
    return real_build


def _tiny_mix(tmp_path, tok):
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    header = tok("<|im_start|>assistant\n<think>\n\n</think>", add_special_tokens=False)["input_ids"]
    rng = np.random.default_rng(0)
    mix = tmp_path / "mix"
    mix.mkdir()
    toks, srcs = [], {}
    for s_, nd in (("web", 160), ("chat", 80), ("math", 60)):
        a = len(toks)
        for _ in range(nd):
            body = list(rng.integers(0, 80, int(rng.integers(14, 40))))
            toks += (body[:5] + header + body[5:] if s_ == "chat" else body) + [eos]
        srcs[s_] = {"tokens": len(toks) - a}
    np.asarray(toks, np.uint32).tofile(mix / "train.u32")
    for s_ in ("web", "chat", "math"):
        rng.integers(0, 80, 24 * 10 + 200).astype(np.uint32).tofile(mix / f"heldout_{s_}.u32")
    (mix / "mix.json").write_text(json.dumps({"sources": srcs, "train_sha256": "x"}))
    return mix


def test_v3_pipeline_tiny_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert import v3
    from tests.full8_dryrun.tiny import tokenizer

    tok, model = tokenizer(), _tiny()
    monkeypatch.setattr(v3, "load", lambda *a, **k: (model, tok, "tiny"))
    _fake_recall(monkeypatch, v3)
    mix = _tiny_mix(tmp_path, tok)
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(_tiny_cfg(tmp_path)))
    # ---- G
    v3.main(["gen", "--config", str(cfgp), "--mix", str(mix), "--out", str(tmp_path / "gen")])
    G = json.loads((tmp_path / "gen" / "results.json").read_text())["gen"]
    assert G["completion_tokens"] > 0 and G["prompts"]["stats"]["chat"]["taken"] > 0
    assert all(G["prompts"]["stats"][k]["taken"] >= G["prompts"]["wanted"][k] / 2 for k in ("retrieval", "tool",
                                                                                            "knowledge"))
    assert G["teacher"].startswith("stock Qwen3.5-9B") and G["split"]["n_lc"] == 6
    lc = np.fromfile(tmp_path / "gen" / "lc.u32", dtype=np.uint32)
    assert len(lc) == 6 * 96
    rows, mask = np.load(tmp_path / "gen" / "gen_rows.npy"), np.load(tmp_path / "gen" / "gen_mask.npy")
    assert rows.shape[1] == 24 and mask.sum() > 0 and len(rows) == G["train_rows"]
    # ---- A, one stage per invocation (as the runner gates it), then the rest: resumes from results.json
    v3.main(["stageA", "--config", str(cfgp), "--mix", str(mix), "--gen", str(tmp_path / "gen"), "--out",
             str(tmp_path / "A"), "--until-stage", "1"])
    R1 = json.loads((tmp_path / "A" / "results.json").read_text())
    assert list(R1["stages"]) == ["stage1"] and "final" not in R1
    g1 = rule.stage_gate(R1, 1, _cmp(-2.0, -5.0))
    assert g1["verdict"] in ("CONTINUE", "STOP") and set(g1["numbers"]) >= {"mqar_stage_drop", "niah_cum_drop"}
    v3.main(["stageA", "--config", str(cfgp), "--mix", str(mix), "--gen", str(tmp_path / "gen"), "--out",
             str(tmp_path / "A")])
    R = json.loads((tmp_path / "A" / "results.json").read_text())
    assert R["stages"]["stage1"] == R1["stages"]["stage1"]  # not re-run
    assert R["stages"]["stage2"]["stream"]["steps"] > R1["stages"]["stage1"]["stream"]["steps"]
    plan = R["plan"]
    sens = {int(k): v for k, v in plan["sensitivity"].items()}
    kl = {int(k): v["kl"] for k, v in plan["detail"].items()}
    assert plan["order"] == sorted(sens, key=lambda i: (sens[i], kl[i], i)) and sorted(plan["order"]) == ATTENTION_LAYERS
    assert plan["order"][-1] == max(sens, key=lambda i: (sens[i], kl[i], i))  # the recall-hardest layer converts last
    assert all(m.swaps[i].gdn.dk == 16 for m in [model] for i in ATTENTION_LAYERS)  # widened key dim
    assert all(st["recall"]["mqar_mean"] <= R["recall_stock"]["mqar_mean"] for st in R["stages"].values())
    kinds = {r.get("kind") for st in R["stages"].values() for r in st["hist"]}
    assert R["final"]["stream"]["taken"]["recall"] > 0 and R["final"]["stream"]["taken"]["generic"] > 0
    assert len(R["stages"]) == 4 and R["stages"]["stage4"]["converted"] == ATTENTION_LAYERS
    for k, st in R["stages"].items():
        crits = [st["entry"]["crit"]] + [c["crit"] for c in st["curve"]]
        assert abs(st["best"]["crit"] - min(crits)) < 1e-12  # keep-best per stage
    assert kinds and R["final"]["stream"]["gen_rows_used"] <= R["final"]["stream"]["gen_rows_total"]
    assert np.isfinite(R["final"]["heldout"]["mix"]["kl"]) and R["final"]["gen_heldout"]["kl"] >= 0
    sdA = torch.load(tmp_path / "ck" / "gdn" / "stageA.pt", weights_only=True)
    assert sorted(sdA) == ATTENTION_LAYERS
    # ---- B (from a stage-A artifact dir, as job B gets it)
    adir = tmp_path / "aart"
    adir.mkdir()
    shutil.copy(tmp_path / "ck" / "gdn" / "stageA.pt", adir / "gdn_converted.pt")
    for f in ("lc.u32", "onpolicy.npz", "gen_heldout.npz"):
        shutil.copy(tmp_path / "gen" / f, adir / f)
    for s in ("web", "chat", "math"):
        shutil.copy(mix / f"heldout_{s}.u32", adir)
    v3.main(["stageB", "--config", str(cfgp), "--a", str(adir), "--out", str(tmp_path / "B")])
    RB = json.loads((tmp_path / "B" / "results.json").read_text())
    assert RB["long_context"]["windows_used"] <= 6 and RB["long_context"]["passes"] == 0
    assert RB["onpolicy"]["prompts_used"] > 0 and RB["onpolicy"]["rollout_completion_tokens"] > 0
    for k in ("long_context", "onpolicy"):
        crits = [RB[k]["curve"][0]["crit"]] + [c["crit"] for c in RB[k]["curve"]]
        assert RB[k]["best"]["crit"] <= min(crits) + 1e-12
    assert np.isfinite(RB["final"]["long"]["kl"]) and np.isfinite(RB["final"]["heldout"]["mix"]["kl"])
    assert set(RB["final"]["recall_drop_vs_stock"]) == {"mqar_drop", "niah_drop"}
    sdB = torch.load(tmp_path / "ck" / "gdn" / "final.pt", weights_only=True)
    assert sorted(sdB) == ATTENTION_LAYERS


# ----------------------------------------------------------------------------- preflight, params, registration
@pytest.mark.parametrize("mode", ["S", "A", "B"])
def test_preflight_params_and_budget(mode):
    from jobs.common.costs import PARAMS as REG
    from jobs.common.costs import quoted_max_cost_usd, worst_case_usd
    from jobs.q9af_v3.preflight import check

    p = {"A": PA, "B": PB, "S": PS}[mode]
    res = check(p, CFG_S if mode == "S" else CFG)
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    want = {"S": (203, 22.09, 12.79, 25.0), "A": (863, 77.53, 54.37, 78.0), "B": (219, 23.44, 13.8, 24.0)}[mode]
    assert (p["max_duration_minutes"], worst_case_usd(p), quoted_max_cost_usd(p), p["hard_cap_usd"]) == want
    assert p["max_duration_minutes"] <= 900 and p["max_restarts"] == 0 and p["batch_usd_per_min"] == 0.084
    spec = (ROOT / "jobs/q9af_v3/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"{want[1]:.2f}" in spec and f"{want[2]:.2f}" in spec
    assert REG[f"q9af_v3_{mode}"] == f"jobs/q9af_v3/job_params_{mode}.json"
    if mode == "S":
        assert worst_case_usd(p) <= 25.0  # the cheap first job
    assert p["decision_rule"] == "jobs/q9af_v3/rule.py" and p["v3_mode"] == mode
    assert p["submit"]["command"] == f"bash /opt/ssdm/m1/jobs/q9af_v3/q9af_v3_job.sh {mode}"
    old, new = p["context"]["cmd_rewrite"]
    assert old in (ROOT / p["context"]["dockerfile"]).read_text(encoding="utf-8") and mode in new
    assert p["test_env"]["transformers"] == "5.17.0"
    assert not any(k.startswith("Q9AF3") for k in p["submit"]["env"])


def test_data_pins_are_the_v2_clean_and_clefflash_files():
    clean = (ROOT / "jobs/q9af_v2/data_clean.sha256").read_text(encoding="utf-8")
    assert (ROOT / "jobs/q9af_v3/data_S.sha256").read_text(encoding="utf-8") == clean
    a = (ROOT / "jobs/q9af_v3/data_A.sha256").read_text(encoding="utf-8")
    assert a.startswith(clean) and a.count("/data/tokenizers/") == 2
    b = (ROOT / "jobs/q9af_v3/data_B.sha256").read_text(encoding="utf-8").splitlines()
    assert b[0].startswith("d1202ab8") and "tokenizer.json" in b[1] and b[1].startswith("5f9e4d49")
    for p, pin in ((PA, "jobs/q9af_v3/data_A.sha256"), (PB, "jobs/q9af_v3/data_B.sha256"),
                   (PS, "jobs/q9af_v3/data_S.sha256")):
        txt = (ROOT / pin).read_text(encoding="utf-8")
        assert all(d["pin"] == pin and f"/data/{d['dest']}" in txt for d in p["context"]["data"])
    assert CFG["mix"] == json.loads((ROOT / "configs/q9af_v2_clean_h100.json").read_text())["mix"]
    assert CFG["mix"]["decontam"] == "full-eval-v1" == CFG_S["mix"]["decontam"]
    assert {**CFG_S["mix"], "train_tokens": None} == {**CFG["mix"], "train_tokens": None}


def test_runner_dry_seam_is_refused_in_the_image():
    assert 'if [ -d /opt/ssdm/m1 ] || [ -z "${Q9AF3_DRYRUN_ROOT:-}" ]; then echo "Q9AF3_DRYRUN refused here"; exit 5; fi' \
        in RUNNER
    assert "jobs.common.rate_guard" in RUNNER and "--option family=q9af-probe-v3w" in RUNNER


# ----------------------------------------------------------------------------- artifact pins (A -> B chaining)
def _tar(tmp_path, members, name="art.tar"):
    p = tmp_path / name
    with tarfile.open(p, "w") as t:
        for m, b in members.items():
            ti = tarfile.TarInfo(m)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    return p


def test_artifact_fill_fetch_and_tamper(tmp_path):
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError
    from jobs.q9af_v3 import artifact

    members = {m: f"payload {m}".encode() for m in artifact.MEMBERS["stageA"]}
    tar = _tar(tmp_path, members)
    params = tmp_path / "pb.json"
    params.write_text(json.dumps(PB))
    with pytest.raises(CheckpointError) as e:
        artifact.fetch(json.loads(params.read_text()), "stageA", tmp_path / "d0")
    assert e.value.code == 5
    artifact.fill(str(params), "stageA", str(tar), "art-test1", "job-test1")
    p = json.loads(params.read_text())
    assert artifact.pinned(p["stageA"]) and p["stageA"]["members"]["./stageA/lc.u32"] == \
        hashlib.sha256(members["./stageA/lc.u32"]).hexdigest()
    assert p["submit"]["env"]["STAGEA_URL"].startswith("<") and rule.expected_reference(p) == "stock"
    url = tar.resolve().as_uri()
    rec = artifact.fetch(p, "stageA", tmp_path / "d1", url=url, work=tmp_path)
    assert (tmp_path / "d1" / "gdn_converted.pt").read_bytes() == members["./stageA/gdn_converted.pt"]
    assert not rec["reused"] and artifact.fetch(p, "stageA", tmp_path / "d1", url="file:///nope")["reused"]
    bad = _tar(tmp_path, {**members, "./stageA/lc.u32": b"tampered"}, "bad.tar")
    with pytest.raises(CheckpointError) as e:
        artifact.fetch(p, "stageA", tmp_path / "d2", url=bad.resolve().as_uri(), work=tmp_path)
    assert e.value.code == 10
    v2 = _tar(tmp_path, {"./gdn_converted.pt": b"v2", "./export.json": b"{}"}, "v2.tar")
    artifact.fill(str(params), "v2", str(v2), "art-v2", "job-7ji99")
    assert rule.expected_reference(json.loads(params.read_text())) == "v2"


# ----------------------------------------------------------------------------- the SHELL-LEVEL dry runs
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
    for m in "SAB":
        (root / f"data_{m}.sha256").write_text(f"{sha}  slices/dev-rows.jsonl.gz\n", encoding="utf-8", newline="\n")
    (root / "CONTEXT_MANIFEST.json").write_text(json.dumps({"dry_run": True}))
    shim = root / "shim"
    shim.mkdir()
    (shim / "curl").write_text('#!/usr/bin/env bash\nexit 0\n', newline="\n")
    (shim / "curl").chmod(0o755)
    return root


def run_dry(root, mode, env_extra=None, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GMN_", "Q9AF3_"))}
    path = [_to_posix(root / "shim"), _to_posix(os.path.dirname(sys.executable))]
    env.update({"Q9AF3_DRYRUN": "1", "Q9AF3_DRYRUN_ROOT": _to_posix(root), "CUDA_VISIBLE_DEVICES": "",
                "GMN_METADATA_URL": "http://metadata.invalid", "Q9AF3_PATH_PREFIX": ":".join(path)})
    if os.name == "nt":
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p.endswith("site-packages")]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(env_extra or {})
    cmd = [BASH, "-c", 'export PATH="$Q9AF3_PATH_PREFIX:$PATH"; exec bash jobs/q9af_v3/q9af_v3_job.sh ' + mode]
    with open(root / "console.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    return rc, root / "output"


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("drop,want,n", [("0.03", "GO", 8), ("0.2", "NO_GO", 0)])
def test_shell_dry_run_job_s(tmp_path, drop, want, n):
    pytest.importorskip("torch")
    root = _dry_root(tmp_path)
    rc, out = run_dry(root, "S", {"Q9AF3_STUB_SENS_DROP": drop})
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, log[-4000:]
    res = json.loads((out / "result.json").read_text())
    assert res["ok"] and (out / "Q9AF3_DONE").exists() and res["verdict"] == want
    dec = json.loads((out / "decision.json").read_text())
    assert dec["n_pass"] == n and set(dec["capacity_control"]) == {"27", "3"}
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert {m["phase"] for m in man} >= {"setup", "D", "P", "final"} and all(m["ok"] for m in man)
    pre = json.loads((root / "work" / "out" / "preflight.json").read_text())
    assert pre["ok"] and pre["mode"] == "S" and pre["checks"]["schedule"]["max_duration"] == 203
    assert not (root / "work" / "out" / "index" / "proxy-rows.json").exists()  # S has no Index arm


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("stop_at,stage_point", [(0, "-3.0"), (2, "-3.0"), (0, "-12.0")])
def test_shell_dry_run_job_a(tmp_path, stop_at, stage_point):
    pytest.importorskip("torch")
    root = _dry_root(tmp_path)
    rc, out = run_dry(root, "A", {"Q9AF3_STUB_STOP_AT": str(stop_at), "Q9AF3_STUB_STAGE_POINT": stage_point})
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, log[-4000:]
    res = json.loads((out / "result.json").read_text())
    stopped = stop_at or float(stage_point) < -10
    assert res["ok"] and (out / "Q9AF3_DONE").exists() and res["verdict"] == ("STOPPED" if stopped else "STAGE_A_DONE")
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    phases = {m["phase"] for m in man}
    assert phases >= {"setup", "D", "G", "I_stock", "P1", "I_stage1", "gate1", "X", "final"} and all(m["ok"] for m in man)
    gates = json.loads((root / "work" / "out" / "gates.json").read_text())
    want_stop = stop_at or 1
    if stopped:
        dec = json.loads((out / "decision.json").read_text())
        assert dec["stopped_at_stage"] == want_stop and len(gates) == want_stop and gates[-1]["verdict"] == "STOP"
        assert f"P{want_stop + 1}" not in phases and not (out / "stageA" / "gdn_converted.pt").exists()
        assert (out / "gdn_converted.pt").exists()  # the stub stands in for the salvage export
    else:
        assert len(gates) == 4 and all(g["verdict"] == "CONTINUE" for g in gates)
        names = {p.name for p in (out / "stageA").iterdir()}
        from jobs.q9af_v3.artifact import MEMBERS

        assert {Path(m).name for m in MEMBERS["stageA"]} <= names and "MEMBERS.sha256" in names
    pre = json.loads((root / "work" / "out" / "preflight.json").read_text())
    assert pre["ok"] and pre["mode"] == "A" and pre["checks"]["schedule"]["max_duration"] == 863


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_shell_dry_run_job_b_refuses_an_unpinned_stage_a(tmp_path):
    pytest.importorskip("torch")
    root = _dry_root(tmp_path)
    rc, out = run_dry(root, "B")
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 10 and "stage-A artifact is not pinned" in log, log[-3000:]
    assert not (out / "Q9AF3_DONE").exists()


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("v2,point,lo,want", [(False, "1.5", "0.4", "PROMOTE"), (True, "1.0", "-0.5", "REPORT")])
def test_shell_dry_run_job_b(tmp_path, v2, point, lo, want):
    pytest.importorskip("torch")
    root = _dry_root(tmp_path)
    p = json.loads(json.dumps(PB))
    p["stageA"].update(artifact="art-dry", artifact_sha256=SHA,
                       members={m: SHA for m in __import__("jobs.q9af_v3.artifact", fromlist=["x"]).MEMBERS["stageA"]})
    if v2:
        p["v2"].update(artifact="art-v2", artifact_sha256=SHA, members={"./gdn_converted.pt": SHA, "./export.json": SHA})
    pp = root / "params_B.json"
    pp.write_text(json.dumps(p))
    rc, out = run_dry(root, "B", {"Q9AF3_DRYRUN_PARAMS": _to_posix(pp), "Q9AF3_STUB_REF_POINT": point,
                                  "Q9AF3_STUB_REF_LO": lo})
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, log[-4000:]
    dec = json.loads((out / "decision.json").read_text())
    assert dec["verdict"] == want and dec["reference"] == ("v2" if v2 else "stock") and dec["attention_free"]
    assert (out / "Q9AF3_DONE").exists()
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    phases = {m["phase"] for m in man}
    assert phases >= {"setup", "F", "P", "X", "L_stock", "L_v3", "S", "final"} and ("L_v2" in phases) == v2
    idx = root / "work" / "out" / "index"
    assert (idx / "compare_v3_vs_ref.json").exists() and (idx / "compare_v2_vs_stock.json").exists() == v2
    assert json.loads((idx / "compare_v3_vs_ref.json").read_text())["baseline"].endswith(
        ("v3w_v2/results.jsonl" if v2 else "v3w_stock/results.jsonl"))


def test_sensitivity_job_tiny_cpu(tmp_path, monkeypatch):
    """mode S end to end on the tiny model: stock recall, stage 1 (widened), every layer alone, 2 base-width controls,
    then the pre-registered sensitivity rule on the real results.json."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert import v3
    from tests.full8_dryrun.tiny import tokenizer

    tok, model = tokenizer(), _tiny()
    monkeypatch.setattr(v3, "load", lambda *a, **k: (model, tok, "tiny"))
    _fake_recall(monkeypatch, v3)
    mix = _tiny_mix(tmp_path, tok)
    c = _tiny_cfg(tmp_path)
    s = json.loads(json.dumps(CFG_S))
    for k in ("seq_len", "batch", "micro_batch", "model_dir", "ckpt_dir", "heldout_windows", "curve_windows_per_source",
              "gdn", "recall_data", "stage1"):
        s[k] = c[k]
    s.pop("vram_fraction")
    s["lc"] = {"seq_len": 96, "blocks": 0}
    s["distill"]["guard"] = {}
    s["sens"] = {"minutes_per_layer": 0.01, "control_layers": 2, "control_stage1_minutes": 0.01}
    cfgp = tmp_path / "s.json"
    cfgp.write_text(json.dumps(s))
    v3.main(["sens", "--config", str(cfgp), "--mix", str(mix), "--out", str(tmp_path / "S")])
    R = json.loads((tmp_path / "S" / "results.json").read_text())
    t = R["table"]
    assert sorted(k for k in t if k.startswith("wide_")) == sorted(f"wide_L{i}" for i in ATTENTION_LAYERS)
    assert all(t[f"wide_L{i}"]["dk"] == 16 for i in ATTENTION_LAYERS)
    base = [k for k in t if k.startswith("base_")]
    assert len(base) == 2 and all(t[k]["dk"] == 8 for k in base)
    worst = sorted(ATTENTION_LAYERS, key=lambda i: -(t[f"wide_L{i}"]["mqar_drop"] + t[f"wide_L{i}"]["niah_drop"]))[:2]
    assert sorted(base) == sorted(f"base_L{i}" for i in worst)
    d = rule.sensitivity(R)
    assert d["verdict"] in ("GO", "PARTIAL", "NO_GO") and rule.valid_sensitivity(d)
    assert d["n_pass"] == sum(t[f"wide_L{i}"]["mqar_drop"] <= 0.05 and t[f"wide_L{i}"]["niah_drop"] <= 0.05
                              for i in ATTENTION_LAYERS)


def test_recall_rows_answer_only_and_recallable():
    pytest.importorskip("transformers")
    from convert.recall import RecallSynth
    from tests.full8_dryrun.tiny import tokenizer

    tok = tokenizer()
    toks = (np.arange(64 * 40) % 85).astype(np.uint32)
    cfg = {"kv_pairs": [4, 8], "kv_queries": 3, "quote_questions": 2, "task_shares": [0.4, 0.3, 0.3],
           "key_pool": 40, "key_scan": 2000, "key_filter": "any", "min_key_pool": 8}
    rs = RecallSynth(tok, toks, np.arange(39), 64, seed=0, cfg=cfg, exclude_words=("w3",))
    assert tok.convert_tokens_to_ids("w3") not in set(rs.keys.tolist())
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        x, w, task = rs._row(rng)
        seen.add(task)
        assert x.shape == (64,) and w.shape == (64,) and w[-1] == 0
        if task == "kv":  # every answered value was written earlier in the row (recall is possible)
            idx = np.flatnonzero(w)
            assert len(idx) and all(int(x[i + 1]) in set(x[: i - 3].tolist()) for i in idx)
    assert {"kv", "needle", "quote"} <= seen
    a1, b1 = rs.rows(3)
    rs2 = RecallSynth(tok, toks, np.arange(39), 64, seed=0, cfg=cfg, exclude_words=("w3",))
    a2, b2 = rs2.rows(3)
    assert (a1 == a2).all() and (b1 == b2).all()  # deterministic per call index


def test_multistream_fractions_exhaustion_and_restore():
    pytest.importorskip("torch")
    from convert.gendata import IndexedWindowStream
    from convert.recall import MultiStream

    class FakeRec:
        calls = 0

        def rows(self, n):
            self.calls += 1
            return np.full((n, 4), self.calls, np.int64), np.ones((n, 4), np.float32)

    toks = np.arange(64 * 4 + 1, dtype=np.uint32)
    rows = np.arange(6 * 4, dtype=np.uint32).reshape(6, 4)

    def mk():
        return MultiStream(IndexedWindowStream(toks, 4, 2, np.arange(64), seed=0),
                           {"generic": 0.35, "gen": 0.25, "recall": 0.40}, 2, 0, rows, np.ones((6, 4), np.uint8),
                           FakeRec())

    ms = mk()
    kinds = [next(ms)[2] for _ in range(40)]
    assert kinds[:20].count("recall") == 8 and kinds[20:].count("recall") == 8  # recall keeps 40 % throughout
    assert kinds.count("gen") == 3  # 6 rows / batch 2, then gen is exhausted
    assert ms.state()["gen_rows_used"] == 6 and kinds[-10:].count("gen") == 0
    st = ms.state()
    ms2 = mk()
    ms2.restore(st)
    assert ms2.state() == st and next(ms2)[2] == next(ms)[2]
    with pytest.raises(ValueError):
        MultiStream(None, {"generic": 0.5, "recall": 0.4}, 2, 0)


def test_widened_gdn_init_reload_and_cached_decode():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert.distill import generate
    from convert.model import attach_from_state, attach_gdn, gdn_dk_of_state, init_gdn

    m = _tiny()
    for i in m.attn_layers:
        attach_gdn(m, i, init_gdn(m, i, "reuse_attn", 0, dk=16))
    g = m.swaps[3].gdn
    assert g.dk == 16 and g.reuse_info["key_map"] == "split" and g.reuse_info["retained_energy"] == 1.0
    extra = sum(p.numel() for p in g.parameters()) - sum(p.numel() for p in init_gdn(m, 3, "reuse_attn", 0).parameters())
    assert extra == 2 * 2 * 8 * 32 + 2 * 2 * 8 * 4  # in_proj_qkv rows + depthwise conv taps for the extra key dims
    sd = {k: v.clone() for k, v in g.state_dict().items()}
    assert gdn_dk_of_state(m, sd) == 16
    attach_from_state(m, 7, sd)
    assert m.swaps[7].gdn.dk == 16 and torch.equal(m.swaps[7].gdn.in_proj_qkv.weight, sd["in_proj_qkv.weight"])
    modes = {i: "student" for i in m.attn_layers}
    prompts = [[5, 6, 7, 8], [9, 10]]
    comps, _ = generate(m, modes, prompts, 5, [100], 0, 0.0)
    for p, c in zip(prompts, comps):
        seq = [0] * (4 - len(p)) + p
        with torch.no_grad(), m.modes(modes):
            for t in c:
                assert int(m.logits(torch.tensor([seq])).float()[0, -1].argmax()) == t
                seq.append(t)
    with pytest.raises(ValueError):
        init_gdn(m, 3, "clone_neighbor", 0, dk=16)
