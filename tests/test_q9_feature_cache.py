"""jobs/q9_feature_cache: the extractor equals the m2 feature definitions and a plain recompute; volume model = calls
actually made; option-key classes / E5; shard round trip. CPU, tiny random Qwen3.5 stand-ins, no data, $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from heads import ByteEncoder  # noqa: E402
from heads.backbone import HiddenStateBackbone, tiny_hf_backbone  # noqa: E402
from heads.features import build_head_input  # noqa: E402
from heads.layout import build_row  # noqa: E402
from heads.option_keys import e5_labels, e5_recode, key_class, short_code  # noqa: E402
from jobs.q9_feature_cache import extract as X  # noqa: E402
from train.data import TOY_TEMPLATE  # noqa: E402

ENC = ByteEncoder()
REQ = {
    "model": "t",
    "state": "ticket 41: printer offline since monday; user tried restart; queue shows 3 jobs",
    "questions": {
        "area": {"type": "choice", "instructions": "which team?",
                 "criteria": {f"option_{i}": f"team {i}" for i in range(12)}},
        "urgent": {"type": "noul", "instructions": "urgent?"},
        "sev": {"type": "score", "instructions": "severity", "criteria": ["low", "mid", "high"]},
    },
}


@pytest.fixture(scope="module")
def tiny():
    bb = tiny_hf_backbone("qwen3.5", seed=3, hidden=32)
    with torch.no_grad():
        for p in bb.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    lm = bb.model.eval()
    return lm.model, lm.lm_head.weight.detach(), lm


def _rows(req=REQ):
    return [build_row(req, q, ENC, template=TOY_TEMPLATE, layout="ordered") for q in sorted(req["questions"])]


def _reference(text, w, row, k):
    seq = row.ids + row.continuations[k]
    h = text(input_ids=torch.tensor([seq]), use_cache=False).last_hidden_state[0]
    lp = torch.log_softmax(h[len(row.ids) - 1: len(seq) - 1] @ w.T, -1)
    return h[-1], float(lp.gather(-1, torch.tensor(row.continuations[k]).unsqueeze(-1)).sum())


@pytest.mark.parametrize("min_saved,kv_budget", [(10**9, X.KV_COPY_TOKENS), (0, X.KV_COPY_TOKENS), (0, 600)])
def test_extract_equals_recompute_and_m2_features(tiny, monkeypatch, min_saved, kv_budget):
    """one-stage everywhere / two-stage everywhere / two-stage with the long-row KV-copy cap forcing small chunks"""
    text, w, _ = tiny
    monkeypatch.setattr(X, "TWO_STAGE_MIN_SAVED", min_saved)
    monkeypatch.setattr(X, "KV_COPY_TOKENS", kv_budget)
    rows = _rows()
    feats, banks = X.extract_request(text, w, rows, opt_chunk=5, bank_chunks=4)
    hb = HiddenStateBackbone(text, fork_mode="recompute")
    for r, f in zip(rows, feats):
        x = build_head_input(hb, [r], need_fork=True, pad_id=0)  # the m2 definitions (plain path, recompute fork)
        assert torch.allclose(f["fork"].float(), x.fork[0, : r.n_options], atol=2e-2)  # stored bf16
        assert torch.allclose(f["e"].float(), x.e[0, : r.n_options], atol=2e-2)
        assert torch.allclose(f["q_pre"].float(), x.q_pre[0], atol=2e-2)
        assert torch.allclose(f["q_end"].float(), x.q_end[0], atol=2e-2)
        for k in range(r.n_options):
            href, lref = _reference(text, w, r, k)
            assert abs(float(f["lm_lp"][k]) - lref) < 1e-3, (r.qid, k)
        assert f["key_class"] == key_class(r.option_ids)
        assert f["bank_key"] == "state" and banks["state"].shape[0] == 4
        chk = X.recompute_check(text, w, r, f, r.n_options - 1)
        assert chk["fork_rel_diff"] < 0.01 and chk["lm_abs_diff"] < 1e-3


@pytest.mark.parametrize("kv_budget", [X.KV_COPY_TOKENS, 600])  # 600: forces 1-3 option chunks (long-row memory cap)
@pytest.mark.parametrize("min_saved", [10**9, 0, X.TWO_STAGE_MIN_SAVED])
def test_volume_model_counts_the_calls_and_positions_made(tiny, monkeypatch, min_saved, kv_budget):
    text, w, _ = tiny
    monkeypatch.setattr(X, "TWO_STAGE_MIN_SAVED", min_saved)
    monkeypatch.setattr(X, "KV_COPY_TOKENS", kv_budget)
    rows = _rows()
    seen = {"calls": 0, "pos": 0}

    def hook(mod, args, kwargs):
        ids = kwargs.get("input_ids")
        seen["calls"] += 1
        seen["pos"] += ids.numel()

    h = text.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        X.extract_request(text, w, rows, opt_chunk=5)
    finally:
        h.remove()
    pos, calls = X.n_chunks_calls(rows, 5)
    assert (pos, calls) == (seen["pos"], seen["calls"])


def test_empty_state_row_banks_its_reread_instruction(tiny):
    text, w, _ = tiny
    req = {**REQ, "state": ""}
    rows = _rows(req)
    feats, banks = X.extract_request(text, w, rows, opt_chunk=8, bank_chunks=3)
    assert all(f["evidence"] == "reread_instructions" and f["bank_key"] == f"q:{f['qid']}" for f in feats)
    assert set(banks) == {f"q:{f['qid']}" for f in feats}


def test_key_classes_and_e5_round_trip():
    assert key_class(["A", "B", "C"]) == "letter"
    assert key_class(["option_0", "option_10"]) == "option_n"
    assert key_class(["chord_0", "chord_1"]) == "option_n"
    assert key_class(["k_2cf4ee4e", "k_bbeeacda", "k_08527554"]) == "hash"
    assert key_class(["yes", "no"]) == "semantic" and key_class(["c8a6", "c8a8"]) == "semantic"
    assert key_class(["1", "2"]) == "digits"
    assert [short_code(i) for i in (0, 25, 26, 27, 701, 702)] == ["A", "Z", "AA", "AB", "ZZ", "AAA"]
    req, back = e5_recode(REQ)
    assert set(back) == {"area"} and req["questions"]["urgent"] == REQ["questions"]["urgent"]
    codes = sorted(req["questions"]["area"]["criteria"])
    # canonical (sorted) original order is kept: option_0, option_1, option_10, option_11, option_2, ...
    assert [back["area"][c] for c in codes] == sorted(REQ["questions"]["area"]["criteria"])
    assert req["questions"]["area"]["criteria"]["C"] == REQ["questions"]["area"]["criteria"]["option_10"]
    assert e5_labels({"area": "option_10", "sev": "1"}, back) == {"area": "C", "sev": "1"}
    assert REQ["questions"]["area"]["criteria"].get("option_0") == "team 0"  # input untouched


def test_shards_round_trip_and_detect_tampering(tiny, tmp_path):
    text, w, _ = tiny
    feats, banks = X.extract_request(text, w, _rows(), opt_chunk=4, bank_chunks=2)
    wr = X.ShardWriter(tmp_path / "s", 1, tmp_path / "p.jsonl")
    for i in range(3):
        wr.add({"rid": f"r{i}", "split": "train", "questions": feats, "banks": banks, "skipped": {}, "e5_back": {}})
    wr.flush()
    got = X.load_shards(tmp_path / "s")
    assert len(got) == 3 and torch.equal(got[1]["questions"][0]["fork"], feats[0]["fork"])
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 3
    m = [json.loads(x) for x in (tmp_path / "s" / "MANIFEST.jsonl").read_text().splitlines()]
    assert [x["requests"] for x in m] == [1, 1, 1]
    (tmp_path / "s" / m[0]["file"]).write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256"):
        X.load_shards(tmp_path / "s")


def test_convert_layers_partial(tiny):
    donor = tiny_hf_backbone("q9af", seed=9, hidden=32).model.model
    stock = tiny_hf_backbone("qwen3.5", seed=3, hidden=32).model.model
    attn = [i for i, t in enumerate(stock.config.layer_types) if t == "full_attention"]
    sd = {attn[0]: donor.layers[attn[0]].linear_attn.state_dict()}
    X.convert_layers(stock, sd, [attn[0]])
    assert stock.config.layer_types[attn[0]] == "linear_attention"
    assert not hasattr(stock.layers[attn[0]], "self_attn")
    with pytest.raises(ValueError):
        X.convert_layers(stock, sd, [0 if attn[0] else 1])  # not an attention layer


def test_lexical_feature_is_rebuilt_from_the_vocab_table(tiny):
    """Review B1: the cache stores option token ids + one lm_head table instead of a [K, D] lex per option."""
    text, w, _ = tiny
    rows = _rows()
    feats, _ = X.extract_request(text, w, rows, opt_chunk=8)
    table = X.lex_vocab(w, X.option_tokens(rows))
    assert len(table["ids"]) == len(X.option_tokens(rows)) and "lex" not in feats[0]
    for r, f in zip(rows, feats):
        want = torch.stack([w[r.ids[a:b]].float().mean(0) for a, b in r.option_spans])
        assert torch.allclose(X.lexical(f, table).float(), want, atol=2e-2)


# ------------------------------------------------------------------ self-check v2 (LESSONS L-28; job-8vjve / job-b7d3n)
SC = json.loads((Path(__file__).resolve().parents[1] / "jobs/q9_feature_cache/cache_config.json").read_text())["selfcheck"]


SMALL = {"model": "t", "state": "ticket 7: printer offline; restart tried",
         "questions": {"area": {"type": "choice", "instructions": "team?",
                                "criteria": {f"option_{i}": f"team {i}" for i in range(4)}},
                       "urgent": {"type": "noul", "instructions": "urgent?"}}}


def _bf16_tiny():
    """A tiny Qwen3.5 in bf16 whose GDN layers keep a long memory (A_log = -4: decay ~ 1 per token), so a lost or
    corrupted recurrent state at the fork is visible, as it is in the trained 9B."""
    bb = tiny_hf_backbone("qwen3.5", seed=3, hidden=32)
    with torch.no_grad():
        for p in bb.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for layer in bb.model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.fill_(-4.0)
    lm = bb.model.eval().to(torch.bfloat16)
    return lm.model, lm.lm_head.weight.detach()


def _corrupting_clone(kind):
    orig = X._clone

    def clone(cache):
        c = orig(cache)
        for layer in c.layers:
            if kind in ("conv_zero", "rec_zero"):
                name = "conv_states" if kind == "conv_zero" else "recurrent_states"
                slots = getattr(layer, name, None)
                items = slots.items() if isinstance(slots, dict) else enumerate(slots) if isinstance(slots, list) else ()
                for k, v in list(items):
                    if isinstance(v, torch.Tensor) and v.ndim > 0:
                        slots[k] = torch.zeros_like(v)
            else:  # kv_drop1: the last cached position lost from every attention layer (an off-by-one)
                for name in ("keys", "values"):
                    v = getattr(layer, name, None)
                    if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[-2] > 1:
                        setattr(layer, name, v[..., :-1, :].contiguous())
        return c
    return clone


def test_selfcheck_v2_passes_the_exact_fork_in_bf16():
    """The production dtype (bf16) cached fork passes; its logic check (fp32, reference kernels) is at fp32 rounding."""
    text, w = _bf16_tiny()
    ok, recs, info = X.selfcheck(text, w, [_rows(SMALL)], 3, SC)
    assert ok, recs
    assert {r["row"] for r in recs} == {0, 1}  # every row of the request
    assert {r["k"] for r in recs if r["row"] == 0} == {0, 2, 3, None}  # options {0, K//2, K-1} + the row features
    fork = [r for r in recs if r["k"] is not None]
    for r in fork:  # S3: ratios and applied bounds are in every record
        assert {"fork_ratio", "lm_ratio", "fla_fork_ratio", "bound_fork_cache", "bound_lm_cache", "bound_fork_fla"} <= set(r)
    assert max(r["q_end_rel"] for r in recs if r["k"] is None) < SC["feature_rel_max"]
    assert info["fla_gating"] is False and set(info["kernels"]) == set(X.reference_kernels.NAMES)
    assert max(r["fork_rel_cache32_vs_truth"] for r in recs if r["k"] is not None) < 1e-4
    assert all(p.dtype == torch.bfloat16 for p in text.parameters())  # weights restored to bf16


@pytest.mark.parametrize("bug", ["conv_zero", "rec_zero", "kv_drop1"])
def test_selfcheck_v2_catches_injected_fork_bugs(monkeypatch, bug):
    text, w = _bf16_tiny()
    monkeypatch.setattr(X, "_clone", _corrupting_clone(bug))
    ok, recs, _ = X.selfcheck(text, w, [_rows(SMALL)], 3, SC)
    assert not ok and not all(r["logic_ok"] for r in recs), (bug, recs)


def test_fp32_weights_restores_every_tensor_bitwise_and_reference_kernels_restore():
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    text, w = _bf16_tiny()
    text.layers[0].linear_attn.A_log.data = text.layers[0].linear_attn.A_log.data.float()  # a kept-fp32 parameter
    before = {n: t.detach().clone() for n, t in list(text.named_parameters()) + list(text.named_buffers())}
    funcs = {n: getattr(M, n) for n in X.reference_kernels.NAMES if hasattr(M, n)}
    with X.fp32_weights(text), X.reference_kernels():
        assert all(t.dtype == torch.float32 for t in text.parameters())
        assert all(getattr(M, n) is getattr(f, "__wrapped__", f) for n, f in funcs.items())
        assert not torch.backends.cudnn.allow_tf32
    for n, t in list(text.named_parameters()) + list(text.named_buffers()):
        assert t.dtype == before[n].dtype and torch.equal(t, before[n]), n
    assert all(getattr(M, n) is f for n, f in funcs.items())



# ------------------------------------------------------------------ the PR #111 reviewer's missed bugs (S1)
def _patched_extract(old, new):
    """extract_request with one source edit (the reviewer's injection method, rev111_bugprobe.py)."""
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(X.extract_request.__wrapped__ if hasattr(X.extract_request, "__wrapped__")
                                            else X.extract_request))
    assert old in src, old
    ns = {}
    exec(compile(src.replace(old, new, 1), "patched_extract", "exec"), X.__dict__, ns)
    return torch.no_grad()(ns["extract_request"])


MISSED = {
    "q_end_offby1": ('"q_end": hrow[r.answer_pos]', '"q_end": hrow[r.answer_pos - 1]'),
    "q_pre_span_shift": ('"q_pre": _span_mean(hrow, r.field_span)',
                         '"q_pre": _span_mean(hrow, (r.field_span[0] + 1, r.field_span[1] + 1))'),
    "e_option_span_shift": ("e = torch.stack([_span_mean(hrow, sp_) for sp_ in r.option_spans])",
                            "e = torch.stack([_span_mean(hrow, (sp_[0] + 1, sp_[1] + 1)) for sp_ in r.option_spans])"),
    "later_row_lm_offby1": ("pp = torch.tensor([max(off + t - 1, 0) for t in tt], device=dev)",
                            "pp = torch.tensor([max(off + t - 1 - (0 if r is rows[0] else 1), 0) for t in tt], device=dev)"),
    "middle_option_swap": ("fork[c0:c0 + c] = h[torch.arange(c, device=dev), last].to(fork_dtype)",
                           "fork[c0:c0 + c] = h[torch.arange(c, device=dev), last].to(fork_dtype)\n"
                           "            if c0 == 0 and c >= 3: fork[[1, 2]] = fork[[2, 1]]"),
}


@pytest.mark.parametrize("bug", sorted(MISSED))
def test_selfcheck_v3_catches_the_reviewers_missed_bugs(monkeypatch, bug):
    text, w = _bf16_tiny()
    monkeypatch.setattr(X, "extract_request", _patched_extract(*MISSED[bug]))
    ok, recs, _ = X.selfcheck(text, w, [_rows(SMALL)], 3, SC)
    assert not ok and not all(r["logic_ok"] for r in recs), bug


def test_reference_kernels_fail_closed_when_a_kernel_cannot_be_swapped(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    before = {n: getattr(M, n) for n in X.reference_kernels.NAMES}
    monkeypatch.setattr(M, "causal_conv1d_update", lambda *a, **k: None)  # no __wrapped__: not swappable
    with pytest.raises(X.KernelSwapError, match="causal_conv1d_update"):
        with X.reference_kernels():
            pass
    for n in X.reference_kernels.NAMES:  # nothing left swapped
        if n != "causal_conv1d_update":
            assert getattr(M, n) is before[n], n


def test_fla_pass_is_record_only_unless_gated():
    text, w = _bf16_tiny()
    strict = {**SC, "fla_ratio": 0.0, "fla_fork_abs": -1.0}  # an impossible fla bound
    ok, recs, info = X.selfcheck(text, w, [_rows(SMALL)], 3, {**strict, "fla_gate": False})
    assert ok and not info["fla_ok_all"]  # recorded, not gating
    ok, _, _ = X.selfcheck(text, w, [_rows(SMALL)], 3, {**strict, "fla_gate": True})
    assert not ok
