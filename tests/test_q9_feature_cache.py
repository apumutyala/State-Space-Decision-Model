"""jobs/q9_feature_cache: the extractor equals the m2 feature definitions and a plain recompute; volume model = calls
actually made; option-key classes / E5; shard round trip. CPU, tiny random Qwen3.5 stand-ins, no data, $0."""

from __future__ import annotations

import json

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


@pytest.mark.parametrize("min_saved", [10**9, 0])  # one-stage everywhere / two-stage everywhere
def test_extract_equals_recompute_and_m2_features(tiny, monkeypatch, min_saved):
    text, w, _ = tiny
    monkeypatch.setattr(X, "TWO_STAGE_MIN_SAVED", min_saved)
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


@pytest.mark.parametrize("min_saved", [10**9, 0, X.TWO_STAGE_MIN_SAVED])
def test_volume_model_counts_the_calls_and_positions_made(tiny, monkeypatch, min_saved):
    text, w, _ = tiny
    monkeypatch.setattr(X, "TWO_STAGE_MIN_SAVED", min_saved)
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
