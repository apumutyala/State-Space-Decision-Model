"""jobs.sym_eval.perturb_eval (jevkit perturb ops 4-6 on a decision-head checkpoint) and the content-based decisions of
sym rows (decide.schema.wire_answer ``tie``): duplicates form a tie SET, empty descriptions fall back to the id.
CPU, tiny random Qwen3.5 hybrid, synthetic JevBench-shaped items (never the public items with a byte model)."""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from heads.layout import build_row  # noqa: E402
from heads.symmetric import SYM_LAYOUT  # noqa: E402
from tests.test_sym_invariance import ENC, make_request, tiny_model, transform  # noqa: E402
from train.data import TOY_TEMPLATE  # noqa: E402


def _decider(layout: str, model=None):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend

    return Decider(DecisionModelBackend(model or tiny_model("qwen3.5", "cache"), ENC, template=TOY_TEMPLATE,
                                        layout=layout))


def _items(n: int, seed: int = 3):
    from jevkit.items import Item

    rng = random.Random(seed)
    out = []
    for i in range(n):
        req = make_request(rng, rng.randrange(3, 6))
        q = req["questions"]["q"]
        labels = list(q["criteria"])
        out.append(Item(id=f"syn-{i}", family="intent", state=req["state"], question=q, labels=labels,
                        expected=labels[0], split="public").validate())
    out.append(Item(id="syn-noul", family="fact", state="the card was declined twice",
                    question={"type": "noul", "instructions": "Was the card declined?"}, labels=["no", "yes"],
                    expected="yes", split="public").validate())
    return out


def _row_fn(layout):
    return lambda req: build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=layout)


def test_perturb_eval_sym_fork_passes_g1_with_identical_rows(tmp_path):
    from jobs.sym_eval.perturb_eval import run

    rep = run(_decider(SYM_LAYOUT), _items(5), tmp_path, row_fn=_row_fn(SYM_LAYOUT))
    per = rep["flip_report"]["per_operator"]
    assert set(per) == {"option_reverse", "option_shuffle", "key_relabel"}
    assert rep["g1_pass"] and rep["rows_all_identical"]
    assert all(v["flips"] == 0 and v["max_abs_dp"] == 0.0 for v in per.values()), per
    po = rep["presented_order"]
    assert all(po[op]["request_changed"] > 0 for op in per), po  # the perturbations really reached the decider
    assert all(po[op]["model_order_changed"] == 0 for op in per), po  # sym rows: canonical by construction
    # the written perturbation file keeps the PERTURBED option order (no sort_keys)
    import json as _json

    from jevkit import perturb as _P

    rows_f = [_json.loads(x) for x in (tmp_path / "perturbed.jsonl").read_text(encoding="utf-8").splitlines()]
    ps = _P.make_perturbations(_items(5), ["option_reverse", "option_shuffle", "key_relabel"], [0])
    assert [list((r.get("item") or r).get("question", {}).get("criteria") or {}) for r in rows_f] == \
        [list(p.item.question.get("criteria") or {}) for p in ps]
    for f in ("base.preds.jsonl", "perturbed.jsonl", "perturbed.preds.jsonl", "perturb_report.json"):
        assert (tmp_path / f).is_file()
    # the written files are the ones `jevkit perturb score` reads, and it agrees
    from jevkit import cli

    it_path = tmp_path / "items.jsonl"
    from jevkit.items import write_items

    write_items(_items(5), it_path)
    assert cli.main(["perturb", "score", "--items", str(it_path), "--perturbed", str(tmp_path / "perturbed.jsonl"),
                     "--pred", str(tmp_path / "base.preds.jsonl"), "--pred-perturbed",
                     str(tmp_path / "perturbed.preds.jsonl"), "--out", str(tmp_path / "cli.json")]) == 0


def test_g1_verdict_fails_closed():
    """Every requested operator must be present with n > 0, the dp bound defaults to 0 (bitwise), and the rows must
    be checked and identical (SYM_CHECK.md section 3)."""
    from functools import partial

    from jobs.sym_eval import perturb_eval as PE
    from jobs.sym_eval.perturb_eval import REORDER_RELABEL_OPS

    ok = {"n": 4, "flips": 0, "invalid": 0, "max_abs_dp": 0.0}
    per = {op: dict(ok) for op in REORDER_RELABEL_OPS}
    rows = {op: {"n": 4, "identical": 4} for op in REORDER_RELABEL_OPS}
    order = {op: {"n": 4, "request_changed": 4, "model_order_changed": 0} for op in REORDER_RELABEL_OPS}
    g1_verdict = partial(PE.g1_verdict, order=order)
    assert g1_verdict(REORDER_RELABEL_OPS, per, rows)["g1_pass"]
    # the perturbation never reached the decider (sort_keys put options back in base order) -> vacuous -> FAIL
    for op in REORDER_RELABEL_OPS:
        v = PE.g1_verdict(REORDER_RELABEL_OPS, per, rows, order={**order, op: {**order[op], "request_changed": 0}})
        assert not v["g1_pass"] and any(op in f and "presented" in f for f in v["g1_failures"])
    assert not PE.g1_verdict(REORDER_RELABEL_OPS, per, rows)["g1_pass"]  # no order check = no pass
    # an operator that produced no perturbation (e.g. key_relabel on a noul-only item set) -> vacuous -> FAIL
    part = {k: v for k, v in per.items() if k != "key_relabel"}
    v = g1_verdict(REORDER_RELABEL_OPS, part, rows)
    assert not v["g1_pass"] and any("key_relabel" in f for f in v["g1_failures"])
    assert not g1_verdict(REORDER_RELABEL_OPS, {**per, "key_relabel": {**ok, "n": 0}}, rows)["g1_pass"]
    # bitwise by default; the observed max |dp| is reported apart
    jit = {**per, "option_shuffle": {**ok, "max_abs_dp": 1e-9}}
    v = g1_verdict(REORDER_RELABEL_OPS, jit, rows)
    assert not v["g1_pass"] and v["max_abs_dp_observed"] == 1e-9
    assert g1_verdict(REORDER_RELABEL_OPS, jit, rows, max_dp=1e-3)["g1_pass"]
    # rows: missing check, one differing row, or a missing operator -> FAIL
    assert not g1_verdict(REORDER_RELABEL_OPS, per, None)["g1_pass"]
    assert not g1_verdict(REORDER_RELABEL_OPS, per, {**rows, "option_reverse": {"n": 4, "identical": 3}})["g1_pass"]
    assert not g1_verdict(REORDER_RELABEL_OPS, per, {k: v for k, v in rows.items() if k != "key_relabel"})["g1_pass"]
    assert not g1_verdict([], {}, {})["g1_pass"]


def test_perturb_eval_noul_only_items_fail_g1_as_vacuous(tmp_path):
    """The SYM_CHECK repro: noul items only -> key_relabel yields nothing -> G1 must not pass."""
    from jobs.sym_eval.perturb_eval import run

    rep = run(_decider(SYM_LAYOUT), _items(0), tmp_path, row_fn=_row_fn(SYM_LAYOUT))
    assert "key_relabel" not in rep["flip_report"]["per_operator"]
    assert not rep["g1_pass"] and any("key_relabel" in f for f in rep["g1_failures"])


def test_select_actions_on_sym_rows_uses_content_ties_invariant_under_relabel():
    """The sym Decider puts the content keys on the policy-conditioned stage; select_actions then decides duplicates as
    a tie_set (invariant as a set under reorder + relabel) and never by id."""
    from decide import select_actions

    dec = _decider(SYM_LAYOUT)
    rng = random.Random(11)
    for t in range(3):
        req = make_request(rng, 4)
        crit = req["questions"]["q"]["criteria"]
        a, b = list(crit)[:2]
        crit[b] = crit[a]
        req["ssdm"] = {"policy_ref": "p1", "context_ref": "c1", "request_id": f"r{t}",
                       "policy_restriction": {"q": sorted(crit)}}
        sel = select_actions(req, dec.decide(req))
        assert sel["action"] == "execute", sel
        new, m = transform({k: v for k, v in req.items() if k != "ssdm"}, rng, relabel=True, reorder=True)
        new["ssdm"] = {**req["ssdm"], "policy_restriction": {"q": sorted(new["questions"]["q"]["criteria"])}}
        sel2 = select_actions(new, dec.decide(new))
        assert sorted(m[o] for o in sel["answers"]["q"]["tie_set"]) == sel2["answers"]["q"]["tie_set"]
        assert all(sel2["answers"]["q"]["probabilities"][m[o]] == p for o, p in sel["answers"]["q"]["probabilities"].items())


def test_perturb_eval_teeth_ordered_layout_moves_under_key_relabel(tmp_path):
    from jobs.sym_eval.perturb_eval import run

    rep = run(_decider("ordered"), _items(5), tmp_path, row_fn=_row_fn("ordered"))
    assert not rep["rows_all_identical"]
    assert rep["flip_report"]["per_operator"]["key_relabel"]["max_abs_dp"] > 1e-6


def test_duplicate_options_form_a_tie_set_invariant_as_a_set():
    dec = _decider(SYM_LAYOUT)
    rng = random.Random(5)
    for t in range(4):
        req = make_request(rng, 4)
        crit = req["questions"]["q"]["criteria"]
        a, b = list(crit)[:2]
        crit[b] = crit[a]  # exact duplicate content
        ans = dec.decide(req)["answers"]["q"]
        assert ans["probabilities"][a] == ans["probabilities"][b]  # equal content -> equal probability (mass split)
        new, m = transform(req, rng, relabel=True, reorder=True)
        ans2 = dec.decide(new)["answers"]["q"]
        assert all(ans2["probabilities"][m[o]] == p for o, p in ans["probabilities"].items())
        assert sorted(m[o] for o in ans["tie_set"]) == ans2["tie_set"]  # the decision is invariant AS A SET
        if len(ans["tie_set"]) > 1:
            assert set(ans["tie_set"]) == {a, b}


def test_near_ties_between_different_options_break_by_content_not_id():
    from decide.schema import wire_answer

    q = {"type": "choice", "instructions": "x", "criteria": {"zz": "first text", "aa": "second text"}}
    tie = {"content_key": {"zz": "00", "aa": "ff"}, "dup_groups": [], "digits": 12}
    ans = wire_answer(q, {"zz": 0.5, "aa": 0.5}, tie)
    assert ans["choice"] == "zz" and ans["tie_set"] == ["zz"]  # content hash order, not the smallest id ("aa")
    assert wire_answer(q, {"zz": 0.5, "aa": 0.5})["choice"] == "aa"  # default path unchanged (smallest id)
    q2 = {"type": "choice", "instructions": "x", "criteria": {"p": "t", "r": "u"}}
    near = wire_answer(q2, {"p": 0.500001, "r": 0.499999}, {"content_key": {"p": "1", "r": "0"}, "digits": 12})
    assert near["choice"] == "p" and near["probabilities"]["p"] != near["probabilities"]["r"]  # 12 digits: no wire tie


def test_evidence_harness_adapter_sym_exact_and_ordered_not():
    """jobs.sym_eval.inv_adapter (the SSDM_INV_HEAD plug of the evidence lane's harness): sym-fork served probabilities
    are bit-identical under reorder + relabel; the ordered layout moves under relabel."""
    from jobs.sym_eval.inv_adapter import ordered_score, sym_fork_score

    opts = [("opt_1", "card was declined"), ("opt_2", "exchange rate wrong"), ("opt_3", "lost card")]
    base = sym_fork_score("my card payment failed", opts)
    sig = {"opt_1": "x9_2", "opt_2": "x1_0", "opt_3": "x5_1"}
    moved = sym_fork_score("my card payment failed", [(sig[i], c) for i, c in opts[::-1]])
    assert all(moved[sig[k]] == v for k, v in base.items())
    ob = ordered_score("my card payment failed", opts)
    om = ordered_score("my card payment failed", [(sig[i], c) for i, c in opts])
    assert max(abs(om[sig[k]] - v) for k, v in ob.items()) > 1e-6


def test_empty_descriptions_reorder_invariant_and_flagged_as_id_content():
    dec = _decider(SYM_LAYOUT)
    rng = random.Random(9)
    req = make_request(rng, 4)
    crit = req["questions"]["q"]["criteria"]
    for o in list(crit)[:2]:
        crit[o] = "" if o == list(crit)[0] else None
    row = build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT)
    assert sorted(row.meta["id_as_content"]) == sorted(list(crit)[:2])
    ans = dec.decide(req)["answers"]["q"]
    new, m = transform(req, rng, relabel=False, reorder=True)
    ans2 = dec.decide(new)["answers"]["q"]
    assert ans2["probabilities"] == {m[o]: p for o, p in ans["probabilities"].items()} or all(
        ans2["probabilities"][m[o]] == p for o, p in ans["probabilities"].items())
    assert ans["choice"] == ans2["choice"]
