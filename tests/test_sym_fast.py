"""heads.sym_fast: the sym-fork fast readout (docs/SYM_FAST.md). CPU, tiny random models, fp32 (runs in the remote CPU
test job; never on the laptop).

* bitwise: ``fork2`` on a one-question request == the training cache fork (``fork_mode="cache"``,
  ``shared_prefix=False``: ``HiddenStateBackbone.fork_features``); ``packed1`` == the recompute fork
  (``fork_mode="recompute"``, ``shared_prefix=False``). Fork features and log-probabilities, ``torch.equal``.
* close: both == the OLD served path (``shared_prefix=True``) up to float rounding, one- and multi-question requests;
* invariance: served probabilities bit-identical under random reorder + relabel (Decider end to end, the evidence
  harness adapters, jevkit perturb ops via ``jobs.sym_eval.perturb_eval``), incl. the multi-question fused call;
  an axis-permuted row (``option_order``) gets EXACTLY permuted fork features (the old path: <= 1e-6 only);
* latency model: the number of backbone calls equals ``predicted_calls`` (old 3 / fork2 2 / packed1 1 per
  one-question decision);
* scope: training (grad on) never takes the fast path; the backend revision names the mode only when it is set.
"""
from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402
from heads.features import build_head_input  # noqa: E402
from heads.symmetric import SYM_LAYOUT  # noqa: E402
from tests.test_sym_invariance import make_request, transform  # noqa: E402
from train.data import TOY_TEMPLATE  # noqa: E402

ENC = ByteEncoder()
TOL = 1e-5
FAMILIES = ["qwen3.5", "q9af", "rwkv7"]

REQ_A = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme; invoice INV-7 overdue",
         "questions": {"dept": {"type": "choice", "instructions": "team?",
                                "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
                       "down": {"type": "noul", "instructions": "is a service down?"},
                       "urg": {"type": "score", "criteria": ["later", "week", "today"]},
                       "inv": {"type": "choice", "instructions": "invoice?", "criteria": {"paid": None, "late": None}}}}


def _rows(req, qids=None, layout=SYM_LAYOUT, **kw):
    return [build_row(req, q, ENC, template=TOY_TEMPLATE, layout=layout, **kw) for q in (qids or req["questions"])]


def _single(req, qid):
    """``req`` reduced to one question (a JevBench-shaped decision)."""
    return {**req, "questions": {qid: req["questions"][qid]}}


def _model(family="qwen3.5", fork_mode="cache", seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_hf_backbone(family, seed=seed, fork_mode=fork_mode), "fork", seed=seed).float().eval()


def _run(m, rows, sym_exec=None, shared_prefix=False):
    m.backbone.sym_exec = sym_exec
    try:
        with torch.no_grad():
            x = build_head_input(m.backbone, rows, need_fork=True, shared_prefix=shared_prefix)
            lp = m(rows, shared_prefix=shared_prefix).log_probs
    finally:
        m.backbone.sym_exec = None
    return x, lp


def _canonical(row) -> bool:
    from heads.sym_fast import content_order

    return content_order(row.continuations) == list(range(len(row.continuations)))


class _Calls:
    """Counts top-level forward calls of the wrapped LM (every backbone call goes through ``backbone.model``)."""

    def __init__(self, m):
        self.n = 0
        self._h = m.backbone.model.register_forward_pre_hook(self._hook)

    def _hook(self, *_):
        self.n += 1

    def close(self):
        self._h.remove()


# --------------------------------------------------------------------------------------- bitwise vs reference
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("qid", list(REQ_A["questions"]))
def test_fork2_one_question_is_bitwise_the_training_cache_fork(family, qid):
    m = _model(family, "cache")
    rows = _rows(_single(REQ_A, qid))
    ref, lp_ref = _run(m, rows, None, shared_prefix=False)  # build_head_input cache-fork branch: fork_features
    new, lp_new = _run(m, rows, "fork2", shared_prefix=True)
    if _canonical(rows[0]):  # same calls, same shapes, same batch order
        assert torch.equal(new.fork, ref.fork)
        assert torch.equal(lp_new, lp_ref)
        assert torch.equal(new.h, ref.h)
    else:  # noul / score whose batch order is content-sorted: same values up to batch-row order
        assert torch.allclose(new.fork, ref.fork, atol=1e-6, rtol=1e-6)
        assert torch.allclose(lp_new, lp_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_packed1_fork_is_bitwise_the_recompute_fork(family):
    m = _model(family, "recompute")
    for qid in REQ_A["questions"]:
        rows = _rows(_single(REQ_A, qid))
        ref, lp_ref = _run(m, rows, None, shared_prefix=False)  # build_head_input recompute branch (fork_chunk 64)
        new, lp_new = _run(m, rows, "packed1", shared_prefix=True)
        if _canonical(rows[0]):
            assert torch.equal(new.fork, ref.fork), qid
            assert torch.equal(lp_new, lp_ref), qid
        assert torch.allclose(new.h, ref.h, atol=TOL, rtol=TOL)  # h: read from a different call (float-close only)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["fork2", "packed1"])
def test_fast_equals_the_old_served_path_up_to_rounding(family, mode):
    m = _model(family, "cache")
    reqs = [REQ_A] + [_single(REQ_A, q) for q in REQ_A["questions"]]
    for req in reqs:
        rows = _rows(req)
        _, old = _run(m, rows, None, shared_prefix=True)  # heads.prefix_features (the served path before)
        _, new = _run(m, rows, mode, shared_prefix=True)
        fin = torch.isfinite(old)
        assert torch.equal(fin, torch.isfinite(new))
        assert torch.allclose(new[fin], old[fin], atol=TOL, rtol=TOL), (new - old).abs()[fin].max()


def test_fork2_chunked_copies_equal_one_call():
    from heads.sym_fast import sym_fast_features

    m = _model("qwen3.5", "cache")
    rows = _rows(REQ_A)
    with torch.no_grad():
        h1, f1 = sym_fast_features(m.backbone, rows, "fork2", max_copies=256)
        h2, f2 = sym_fast_features(m.backbone, rows, "fork2", max_copies=3)
    assert torch.allclose(f1, f2, atol=TOL, rtol=TOL) and torch.allclose(h1, h2, atol=TOL, rtol=TOL)


# ------------------------------------------------------------------------------------------- latency model
@pytest.mark.parametrize("mode", [None, "fork2", "packed1"])
def test_backbone_call_count_matches_the_latency_model(mode):
    from heads.sym_fast import predicted_calls

    m = _model("qwen3.5", "cache")
    want_one = {None: 3, "fork2": 2, "packed1": 1}[mode]
    for req in [_single(REQ_A, "dept"), _single(REQ_A, "down"), REQ_A]:
        rows = _rows(req)
        c = _Calls(m)
        try:
            _run(m, rows, mode, shared_prefix=True)
        finally:
            c.close()
        n = c.n // 2  # _run calls the readout twice (build_head_input, then the model forward)
        assert n == predicted_calls(rows, mode, backbone=m.backbone), (mode, len(rows), n)
        if len(rows) == 1:
            assert n == want_one, (mode, n)
        elif mode is not None:
            assert n == {"fork2": 2, "packed1": 1}[mode]  # 4 questions, still one prefix + one batched call


# ------------------------------------------------------------------------------------------------ invariance
def _decider(mode, model=None):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend

    return Decider(DecisionModelBackend(model or _model(), ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT,
                                        sym_exec=mode))


def _served(dec, req):
    resp = dec.decide(copy.deepcopy(req))
    ss = resp.get("ssdm") or {}
    assert ss.get("status", "ok") == "ok", ss
    return {q: {str(k): float(v) for k, v in d.items()} for q, d in ss["served_probabilities"].items()}


@pytest.mark.parametrize("mode", ["fork2", "packed1"])
def test_end_to_end_reorder_relabel_bit_identical(mode):
    dec = _decider(mode)
    rng = random.Random(11)
    for _ in range(10):
        req = make_request(rng, rng.randrange(2, 8))
        p0 = _served(dec, req)["q"]
        for _ in range(2):
            new, mp = transform(req, rng, relabel=True, reorder=True)
            p1 = _served(dec, new)["q"]
            assert all(p1[mp[o]] == v for o, v in p0.items())  # tol 0


@pytest.mark.parametrize("mode", ["fork2", "packed1"])
def test_multi_question_fused_call_bit_identical_under_reorder_relabel(mode):
    """Q > 1: every question's options share ONE batched call; permuting / relabeling the options of one question must
    leave every question's served probabilities bit-identical (the batch is content-determined)."""
    dec = _decider(mode)
    rng = random.Random(5)
    for _ in range(6):
        a, b = make_request(rng, rng.randrange(2, 7)), make_request(rng, rng.randrange(2, 7))
        req = {**a, "questions": {"q": a["questions"]["q"], "r": b["questions"]["q"],
                                  "n": {"type": "noul", "instructions": "is it urgent?"}}}
        p0 = _served(dec, req)
        new, mp = transform({**req, "questions": {"q": req["questions"]["q"]}}, rng, relabel=True, reorder=True)
        moved = {**req, "questions": {"q": new["questions"]["q"], "r": req["questions"]["r"], "n": req["questions"]["n"]}}
        p1 = _served(dec, moved)
        assert all(p1["q"][mp[o]] == v for o, v in p0["q"].items())
        assert p1["r"] == p0["r"] and p1["n"] == p0["n"]


def test_axis_permuted_row_gets_exactly_permuted_fork_features():
    from heads.sym_fast import sym_fast_features

    m = _model("qwen3.5", "cache")
    rng = random.Random(2)
    for _ in range(5):
        req = make_request(rng, rng.randrange(3, 8))
        base = build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT)
        perm = list(range(base.n_options))
        rng.shuffle(perm)
        moved = build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT, option_order=perm)
        for mode in ("fork2", "packed1"):
            with torch.no_grad():
                _, f0 = sym_fast_features(m.backbone, [base], mode)
                _, f1 = sym_fast_features(m.backbone, [moved], mode)
            assert torch.equal(f1[0], f0[0][torch.tensor(perm)]), mode


@pytest.mark.parametrize("fn", ["sym_fast_fork2_score", "sym_fast_packed1_score"])
def test_evidence_harness_adapter_fast_paths_exact(fn):
    from jobs.sym_eval import inv_adapter as A

    score = getattr(A, fn)
    opts = [("opt_1", "card was declined"), ("opt_2", "exchange rate wrong"), ("opt_3", "lost card")]
    base = score("my card payment failed", opts)
    sig = {"opt_1": "x9_2", "opt_2": "x1_0", "opt_3": "x5_1"}
    moved = score("my card payment failed", [(sig[i], c) for i, c in opts[::-1]])
    assert all(moved[sig[k]] == v for k, v in base.items())
    plain = A.sym_fork_score("my card payment failed", opts)  # the old served path, same tiny model
    assert max(abs(plain[k] - v) for k, v in base.items()) <= 1e-5


@pytest.mark.parametrize("mode", ["fork2", "packed1"])
def test_perturb_eval_g1_passes_on_the_fast_path(tmp_path, mode):
    from jobs.sym_eval.perturb_eval import run
    from tests.test_sym_perturb_eval import _items, _row_fn

    rep = run(_decider(mode), _items(5), tmp_path, row_fn=_row_fn(SYM_LAYOUT))
    per = rep["flip_report"]["per_operator"]
    assert rep["g1_pass"] and rep["rows_all_identical"], rep["g1_failures"]
    assert all(v["flips"] == 0 and v["max_abs_dp"] == 0.0 for v in per.values()), per


# ---------------------------------------------------------------------------------------------------- scope
def test_training_never_takes_the_fast_path():
    from heads import sym_fast as SF

    m = _model()
    m.backbone.sym_exec = "fork2"
    rows = _rows(_single(REQ_A, "dept"))
    with torch.enable_grad():
        assert not SF.eligible(m.backbone, rows, True)  # autograd on: the training path, untouched
    with torch.no_grad():
        assert SF.eligible(m.backbone, rows, True)
        assert not SF.eligible(m.backbone, rows, False)  # no fork wanted
        assert not SF.eligible(m.backbone, _rows(_single(REQ_A, "dept"), layout="ordered"), True)
    m.backbone.sym_exec = "nonsense"
    with pytest.raises(ValueError):
        SF.eligible(m.backbone, rows, True)
    m.backbone.sym_exec = None


def test_backend_revision_names_the_mode_only_when_set():
    from heads.decide_backend import DecisionModelBackend

    m = _model()
    old = DecisionModelBackend(m, ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT).revision()
    assert "sym_exec" not in old
    new = DecisionModelBackend(_model(), ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT, sym_exec="fork2").revision()
    assert new["sym_exec"] == "fork2" and new["revision_id"] != old["revision_id"]
    with pytest.raises(ValueError):
        DecisionModelBackend(_model(), ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT, sym_exec="fast")


def test_packed1_reports_the_tokens_it_really_runs():
    from heads.decide_backend import DecisionModelBackend
    from heads.symmetric import row_input_tokens

    req = _single(REQ_A, "dept")
    (row,) = _rows(req)
    plain = DecisionModelBackend(_model(), ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT, sym_exec="fork2")
    packed = DecisionModelBackend(_model(), ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT, sym_exec="packed1")
    assert plain.plan(req).total_input_tokens == row_input_tokens(row)  # fork2 is token-neutral
    assert packed.plan(req).total_input_tokens == row.n_options * len(row.ids) + sum(map(len, row.continuations))


def test_timing_harness_smoke(tmp_path):
    import json

    from scripts.sym_fast_timing import main

    out = tmp_path / "t.json"
    assert main(["--tiny", "--n", "4", "--warmup", "1", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    arms = rep["arms"]
    assert {a: arms[a]["calls_per_decision"] for a in arms} == {"old": 3, "fork2": 2, "packed1": 1}
    assert arms["fork2"]["max_abs_dp_vs_old"] <= 1e-5 and arms["packed1"]["max_abs_dp_vs_old"] <= 1e-5
    assert arms["fork2"]["argmax_agree_vs_old"] == 1.0
