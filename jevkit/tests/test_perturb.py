import hashlib
import json
import math

import pytest

from decide import Decider
from decide.backends import OracleBackend
from jevkit import perturb as P
from jevkit import systemone as S1
from jevkit.answers import answer_from_probs
from jevkit.cli import main

from ._fixtures import choice, mixed_set, noul, score


def _h(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def content_model(item):
    """Probabilities from option DESCRIPTIONS only: invariant to order and to key names, sensitive to wording."""
    crit = item.question.get("criteria")
    if item.type == "noul":
        return {"type": "noul", "noul": 0.1 + 0.8 * _h(json.dumps(item.state, sort_keys=True))}
    descs = list(crit.values()) if isinstance(crit, dict) else list(crit)
    logits = [3 * _h(str(d)) for d in descs]
    z = sum(math.exp(x) for x in logits)
    probs = dict(zip(item.labels if item.type == "score" else list(crit), (math.exp(x) / z for x in logits)))
    return answer_from_probs(item, probs)


def first_listed_model(item):
    """Position-biased: 0.7 on whichever option is listed first."""
    if item.type != "choice":
        return content_model(item)
    keys = list(item.question["criteria"])
    rest = 0.3 / (len(keys) - 1)
    return answer_from_probs(item, {k: (0.7 if n == 0 else rest) for n, k in enumerate(keys)})


def test_eight_operators_and_determinism():
    assert len(P.OPERATORS) == 8
    its = mixed_set()
    a = [p.to_dict() for p in P.make_perturbations(its, seeds=(0, 1))]
    b = [p.to_dict() for p in P.make_perturbations(its, seeds=(0, 1))]
    assert a == b and {d["perturbation"]["op"] for d in a} == set(P.OPERATORS)


def test_operator_semantics():
    c = choice("c", k=4, gold=2)
    rev = P.op_option_reverse(c, 0)
    assert list(rev.item.question["criteria"]) == ["o3", "o2", "o1", "o0"] and rev.item.expected == "o2"
    rl = P.op_key_relabel(c, 0)
    assert set(rl.item.labels) == {"opt_1", "opt_2", "opt_3", "opt_4"}
    assert rl.item.expected == rl.label_map["o2"]
    assert rl.item.question["criteria"][rl.label_map["o2"]] == c.question["criteria"]["o2"]
    sp = P.op_state_paraphrase(c, 0)
    assert "isn't" in sp.item.state and sp.item.question == c.question
    ip = P.op_instruction_paraphrase(c, 0)
    assert "don't" in ip.item.question["instructions"].lower() or "do not" in ip.item.question["instructions"].lower()
    s = score("s")
    assert P.op_option_reverse(s, 0) is None and P.op_option_shuffle(s, 0) is None and P.op_key_relabel(s, 0) is None
    n = noul("n")
    assert list(P.op_option_reverse(n, 0).item.question["criteria"]) == ["false", "true"]
    assert P.op_markup_noise(n, 0).item.state == {"detail": "It is approved.", "note": "record n"}
    assert P.op_whitespace_noise(c, 0).item.state.startswith("\n")
    for p in P.make_perturbations([c], seeds=range(5)):
        if p.op == "option_shuffle":
            assert list(p.item.question["criteria"]) != list(c.question["criteria"])


def test_reorder_invariant_model_has_zero_reorder_flips():
    its = mixed_set()
    rep = P.run_battery(its, content_model, seeds=(0, 1, 2))
    assert rep["reorder_flips"] == 0
    for op in P.REORDER_OPS:
        assert rep["per_operator"][op]["max_abs_dp"] == pytest.approx(0.0, abs=1e-12)
    assert rep["per_operator"]["option_paraphrase"]["n"] > 0


def test_position_biased_model_flips_under_reorder():
    its = [choice(f"c{i}", k=4) for i in range(6)]
    rep = P.run_battery(its, first_listed_model, ops=["option_reverse", "option_shuffle"])
    assert rep["per_operator"]["option_reverse"]["flip_rate"] == 1.0
    assert rep["per_operator"]["option_reverse"]["mean_tv_shift"] == pytest.approx(0.6)
    assert rep["average_flip_rate"] > 0.5


def test_decider_oracle_is_reorder_invariant_through_the_adapter():
    ad = S1.SystemOneAdapter(Decider(OracleBackend()))
    rep = P.run_battery(mixed_set(), lambda it: ad.answer_item(it)["answer"], ops=["option_reverse", "option_shuffle"])
    assert rep["reorder_flips"] == 0 and rep["pooled"]["max_abs_dp"] <= 1e-3  # wire rounding to 4 decimals


def test_invalid_answers_counted_separately():
    its = [choice("c", k=3)]
    rep = P.run_battery(its, lambda it: {"type": "choice", "probabilities": {"x": 1.0}}, ops=["option_reverse"])
    assert rep["per_operator"]["option_reverse"]["invalid"] == 1 and rep["per_operator"]["option_reverse"]["n"] == 0


def test_cli_make_and_score(tmp_path):
    its = mixed_set()
    ipath = tmp_path / "items.jsonl"
    ipath.write_text("".join(json.dumps(i.to_dict()) + "\n" for i in its), encoding="utf-8")
    ppath = tmp_path / "pert.jsonl"
    assert main(["perturb", "make", "--items", str(ipath), "--out", str(ppath), "--seeds", "0"]) == 0
    pert = [json.loads(x) for x in ppath.read_text(encoding="utf-8").splitlines()]
    from jevkit.items import Item

    base_pred = tmp_path / "b.jsonl"
    base_pred.write_text("".join(json.dumps({"id": i.id, "answer": content_model(i), "status": "ok"}) + "\n"
                                 for i in its), encoding="utf-8")
    pp = tmp_path / "p.jsonl"
    rows = []
    for d in pert:
        d2 = {k: v for k, v in d.items() if k != "perturbation"}
        rows.append(json.dumps({"id": d["id"], "answer": content_model(Item.from_dict(d2)), "status": "ok"}))
    pp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert main(["perturb", "score", "--items", str(ipath), "--perturbed", str(ppath), "--pred", str(base_pred),
                 "--pred-perturbed", str(pp), "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["reorder_flips"] == 0 and set(rep["per_operator"]) == set(P.OPERATORS)
