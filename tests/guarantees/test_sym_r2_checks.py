"""SYM_CHECK round 2 (sym-fork v3 @ 72af9886): perturb_eval's fail-closed G1 gate, the jevkit CLI perturbation file,
and the batch-vs-single residual against decision margins. CPU; the margin test uses a tiny random model."""
from __future__ import annotations

import json
import random

import pytest

pe = pytest.importorskip("jobs.sym_eval.perturb_eval")
if not hasattr(pe, "g1_verdict"):
    pytest.skip("perturb_eval without g1_verdict (pre-v3)", allow_module_level=True)

OPS = ("option_reverse", "option_shuffle", "key_relabel")


def _per(n=3, flips=0, invalid=0, dp=0.0):
    return {"n": n, "flips": flips, "invalid": invalid, "max_abs_dp": dp}


def _good():
    per = {op: _per() for op in OPS}
    rows = {op: {"n": 3, "identical": 3} for op in OPS}
    order = {op: {"n": 3, "request_changed": 3} for op in OPS}
    return per, rows, order


def test_g1_passes_only_the_fully_checked_case():
    per, rows, order = _good()
    assert pe.g1_verdict(OPS, per, rows, 0.0, order)["g1_pass"]


@pytest.mark.parametrize("mutate", [
    "no_ops", "op_missing", "op_n0", "flip", "invalid", "dp_tiny", "dp_none", "rows_none", "rows_missing_op",
    "rows_not_identical", "rows_n0", "order_missing", "order_unchanged"])
def test_g1_fails_closed(mutate):
    per, rows, order = _good()
    ops = list(OPS)
    if mutate == "no_ops":
        ops = []
    elif mutate == "op_missing":
        del per["key_relabel"]
    elif mutate == "op_n0":
        per["key_relabel"]["n"] = 0
    elif mutate == "flip":
        per["option_shuffle"]["flips"] = 1
    elif mutate == "invalid":
        per["option_reverse"]["invalid"] = 1
    elif mutate == "dp_tiny":
        per["option_reverse"]["max_abs_dp"] = 1e-15
    elif mutate == "dp_none":
        per["option_reverse"]["max_abs_dp"] = None
    elif mutate == "rows_none":
        rows = None
    elif mutate == "rows_missing_op":
        del rows["key_relabel"]
    elif mutate == "rows_not_identical":
        rows["option_reverse"]["identical"] = 2
    elif mutate == "rows_n0":
        rows["option_reverse"] = {"n": 0, "identical": 0}
    elif mutate == "order_missing":
        del order["option_shuffle"]
    elif mutate == "order_unchanged":
        order["option_reverse"]["request_changed"] = 0
    v = pe.g1_verdict(ops, per, rows, 0.0, order)
    assert not v["g1_pass"] and v["g1_failures"], (mutate, v)


def test_g1_partial_presentation_still_passes_documented_weakness():
    """request_changed > 0 is required per operator, not == n: 1 changed perturbation out of 3 passes. Not a hole for
    jevkit's own operators (reverse/shuffle rotate an identity draw, relabel always renames), but worth knowing."""
    per, rows, order = _good()
    order["option_reverse"]["request_changed"] = 1
    assert pe.g1_verdict(OPS, per, rows, 0.0, order)["g1_pass"]


def test_round1_noul_repro_now_fails_closed():
    from jevkit import perturb
    from jevkit.items import Item

    it = Item(id="n1", family="fact", state="the card was declined",
              question={"type": "noul", "instructions": "Declined?", "criteria": {"true": "yes it was", "false": "no"}},
              labels=["no", "yes"], expected="yes", split="public").validate()
    ps = perturb.make_perturbations([it], list(OPS), [0])
    assert sorted({p.op for p in ps}) == ["option_reverse", "option_shuffle"]  # key_relabel is choice-only
    ans = {"n1": {"type": "noul", "noul": 0.7}}
    rep = perturb.flip_report([it], ps, ans, {p.item.id: {"type": "noul", "noul": 0.7} for p in ps})
    order = pe.presented_order([it], ps, [p.item for p in ps])
    rows = {op: {"n": 1, "identical": 1} for op in ("option_reverse", "option_shuffle")}
    v = pe.g1_verdict(OPS, rep["per_operator"], rows, 0.0, order)
    assert not v["g1_pass"] and any("key_relabel" in f for f in v["g1_failures"]), v


@pytest.mark.xfail(strict=True, reason="jevkit/cli.py `perturb make` dumps with sort_keys=True, which puts every "
                                       "reordered criteria dict back in base order (reorder flip rates via the CLI are "
                                       "vacuous); strict: flips to FAIL once cli.py is fixed")
def test_jevkit_cli_perturb_make_preserves_presented_order(tmp_path):
    from jevkit import cli
    from jevkit.items import Item, write_items

    it = Item(id="c1", family="intent", state="s",
              question={"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": "y", "c": "z"}},
              labels=["a", "b", "c"], expected="a", split="public").validate()
    write_items([it], tmp_path / "items.jsonl")
    out = tmp_path / "pert.jsonl"
    assert cli.main(["perturb", "make", "--items", str(tmp_path / "items.jsonl"), "--ops", "option_reverse",
                     "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    item = rec.get("item", rec)
    assert list(item["question"]["criteria"]) == ["c", "b", "a"], item["question"]["criteria"]


# ------------------------------------------------------------------------------------------- batch vs single margin
def test_batch_vs_single_residual_vs_decision_margin():
    """A5: a question scored alone vs co-batched with sibling questions. Records max |dp| and the smallest top-2
    margin; a decision can flip only if the margin is below the residual. Asserts no decision flips."""
    torch = pytest.importorskip("torch")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE

    worst, min_margin, flips, n = 0.0, 1.0, 0, 0
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = DecisionModel(tiny_hf_backbone("qwen3.5", seed=seed, fork_mode="cache"), "fork", seed=seed).float().eval()
        dec = Decider(DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE, layout="sym-fork"))
        rng = random.Random(seed)
        for t in range(10):
            k = rng.randrange(2, 7)
            crit = {f"o{j}": f"option {rng.randrange(10 ** 6)} " + "w" * rng.randrange(0, 40) for j in range(k)}
            q = {"type": "choice", "instructions": "Which option fits?", "criteria": crit}
            st = f"state {seed}-{t} " * rng.randrange(1, 6)
            alone = dec.decide({"model": "m", "state": st, "questions": {"q": q}})
            sib = {f"s{i}": {"type": "choice", "instructions": "Another?",
                             "criteria": {"x": "long " * rng.randrange(1, 60), "y": "n"}} for i in range(rng.randrange(1, 4))}
            both = dec.decide({"model": "m", "state": st, "questions": {"q": q, **sib}})
            pa, pb = alone["ssdm"]["served_probabilities"]["q"], both["ssdm"]["served_probabilities"]["q"]
            worst = max(worst, max(abs(pa[o] - pb[o]) for o in pa))
            top = sorted(pa.values(), reverse=True)
            min_margin = min(min_margin, top[0] - top[1])
            flips += alone["answers"]["q"]["choice"] != both["answers"]["q"]["choice"]
            n += 1
    print(f"batch-vs-single: n={n} max|dp|={worst:.3e} min top-2 margin={min_margin:.3e} flips={flips}")
    assert flips == 0
    assert worst <= 2.6e-8 or worst < min_margin / 2
