"""A-PREREG-JEV3-LM: the canonicalised shim (decide.canon_shim, layouts g1v2-canon / g1v2-canon-shuffle), its served
invariance on a tiny lm_label model with the real Qwen3.5-4B tokenizer (SSDM_BPE_TOKENIZER; skipped without it), the
evidence score op (KEEP gate 2) on it, ties, the per-type T, the letter-prior report, and the job params. CPU only."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decide import canon_shim as C  # noqa: E402

TOK = os.environ.get("SSDM_BPE_TOKENIZER")
needs_tok = pytest.mark.skipif(not (TOK and Path(TOK).is_dir()), reason="SSDM_BPE_TOKENIZER (Qwen3.5-4B tokenizer files)")
RUB = ["no impact", "minor impact, workaround exists", "major impact on a core function", "total outage"]


def _choice(crit: dict) -> dict:
    return {"type": "choice", "instructions": "Route the ticket.", "criteria": crit}


# ------------------------------------------------------------------------------------------------ canonicaliser (pure)
def test_canonical_order_is_content_only_and_normalised():
    crit = {"x9": "Billing team", "a1": "  billing\tteam ", "q": "Security", "zz": "Escalate to on-call"}
    rec = C.canonical_question(_choice(crit))
    assert rec["contents"] == sorted(rec["contents"], key=lambda c: C.sort_key(c))
    # every display order and every relabel gives the same contents in the same canonical order
    base = rec["contents"]
    for perm in itertools.permutations(list(crit)):
        r2 = C.canonical_question(_choice({f"n{i}": crit[k] for i, k in enumerate(perm)}))
        assert r2["contents"] == base and r2["question"] == rec["question"]
    # casefold-equal but different contents: ordered by content hash (no id / position); exact duplicates -> tie set
    assert rec["tie_sets"] == []
    dup = C.canonical_question(_choice({"b": "Billing team", "a": "Billing  team", "c": "x"}))
    assert dup["tie_sets"] == [["a", "b"]]


def test_c5_normalisation_nfc_nfd_and_spaces_keep_the_order():
    nfc, nfd = unicodedata.normalize("NFC", "Café refund"), unicodedata.normalize("NFD", "Café refund")
    assert nfc != nfd
    a = C.canonical_question(_choice({"1": nfc, "2": "Other", "3": " Zeta  "}))
    b = C.canonical_question(_choice({"9": "Other", "8": nfd + "  ", "7": "Zeta"}))
    assert a["contents"] == b["contents"] and a["question"] == b["question"]


def test_noul_fixed_and_score_level_order_fail_closed():
    n1 = C.canonical_question({"type": "noul", "instructions": "x", "criteria": {"false": "no", "true": "yes"}})
    assert n1["ids"] == ["true", "false"] and list(n1["question"]["criteria"]) == ["true", "false"]
    s = C.canonical_question({"type": "score", "instructions": "rate", "criteria": RUB})
    assert s["ids"] == ["0", "1", "2", "3"] and s["contents"][2] == "level 2 of 4: " + RUB[2]
    from scripts import ev_score_perturb as SP

    assert all(C.level_content(v, 4, RUB[v]) == SP.render_level(v, 4, RUB[v]) for v in range(4))
    cont = [C.level_content(v, 4, RUB[v]) for v in range(4)]
    ev = {"type": "score", "instructions": "rate", "scale": cont, "options": {"s_3": cont[3], "s_1": cont[1],
                                                                              "s_0": cont[0], "s_2": cont[2]}}
    r = C.canonical_question(ev)
    assert r["ids"] == ["s_0", "s_1", "s_2", "s_3"] and r["question"] == s["question"]
    with pytest.raises(ValueError, match="no level value"):
        C.canonical_question({**ev, "options": {"a": "minor", "b": cont[1], "c": cont[2], "d": cont[3]}, "scale": None})
    with pytest.raises(ValueError, match="scale"):
        C.canonical_question({**ev, "scale": list(reversed(cont))})
    with pytest.raises(ValueError, match="not exactly"):
        C.canonical_question({**ev, "options": {"a": cont[0], "b": cont[0], "c": cont[2], "d": cont[3]}, "scale": None})


def test_training_shuffle_is_seeded_by_content_and_never_by_ids():
    crit = {chr(97 + i): f"destination {i}" for i in range(6)}
    q = _choice(crit)
    req = {"state": "s", "questions": {"q": q}}
    _, a = C.canonical_request(req, "q", shuffle=True)
    _, b = C.canonical_request({"state": "s", "questions": {"q": _choice({f"z{k}": v for k, v in reversed(list(
        crit.items()))})}}, "q", shuffle=True)
    assert a["contents"] == b["contents"]  # same content -> same training order, whatever the ids / display order
    _, c = C.canonical_request(req, "q", shuffle=False)
    assert c["contents"] == sorted(c["contents"], key=C.sort_key)
    # across many rows the shuffle puts the canonical first option on every letter (no letter prior by construction)
    first = []
    for i in range(400):
        rq = {"state": f"row {i}", "questions": {"q": q}}
        _, r = C.canonical_request(rq, "q", shuffle=True)
        first.append(r["contents"].index(c["contents"][0]))
    share = first.count(0) / len(first)
    assert abs(share - 1 / 6) < 0.07


def test_equalize_ties_keeps_mass_and_is_order_free():
    lg = {"a": math.log(0.5), "b": math.log(0.2), "c": math.log(0.3)}
    e = C.equalize_ties(lg, [["a", "b"]])
    assert e["a"] == e["b"] and abs(sum(math.exp(v) for v in e.values()) - 1.0) < 1e-12
    e2 = C.equalize_ties({"a": lg["b"], "b": lg["a"], "c": lg["c"]}, [["a", "b"]])
    assert e2 == e


def test_tie_set_is_the_served_decision():
    from decide.schema import wire_answer

    q = _choice({"b": "same", "a": "same", "c": "other"})
    rec = C.canonical_question(q)
    tie = {"content_key": {o: C.content_key(c) for o, c in zip(rec["ids"], rec["contents"])}, "digits": 12}
    ans = wire_answer(q, {"a": 0.4, "b": 0.4, "c": 0.2}, tie)
    assert ans["tie_set"] == ["a", "b"] and ans["choice"] in ans["tie_set"]


# ------------------------------------------------------------------------------------------------ per-type T, letters
def test_per_type_temperature_and_lookup():
    from jobs.m2_first_cell import screen as S

    rng = random.Random(0)
    logps, labels, cells, noul = [], [], [], []
    for i in range(500):
        k = 3 if i % 2 else 4
        z = [rng.gauss(0, 3) for _ in range(k)]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        logps.append([x - lse for x in z])
        labels.append(rng.randrange(k))
        cells.append(("choice", k))
    pt = S.fit_per_type(logps, labels, cells, noul, 0.25, 4.0, 200)
    assert pt["types"]["choice"]["n"] == 500 and pt["types"]["noul"] == {"T": 1.0, "n": 0}
    assert set(pt["cells"]) == {"choice:3", "choice:4"}
    q3 = {"type": "choice", "criteria": {"a": 1, "b": 2, "c": 3}}
    assert S.t_for(q3, pt) == pt["cells"]["choice:3"]["T"] and S.t_for(q3, 1.5) == 1.5
    assert S.t_for({"type": "score", "criteria": ["a", "b"]}, pt) == 1.0


def test_letter_prior_report_flags_an_a_bias(tmp_path):
    from jobs.m2_first_cell import screen as S

    rows = []
    for i in range(100):
        gold = i % 4
        pred = 0 if i % 2 else gold  # the model picks A half the time
        probs = [0.1] * 4
        probs[pred] = 0.7
        rows.append({"id": f"r{i}/q", "family": "ambiguous" if i < 20 else "policy", "type": "choice",
                     "option_ids": ["a", "b", "c", "d"], "probs": probs, "gold": gold})
    (tmp_path / "calib_probs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = S.letter_prior_report(tmp_path, [], None, {"flag_pp": 5})
    assert rep["slice"]["flag"] and rep["slice"]["A_excess_pp"] > 5 and rep["flag"]
    assert rep["slice"]["choice_acc_without_ambiguous"] is not None


# ------------------------------------------------------------------------------------------------ served system (tokenizer)
def _probs(be, req, qid="q"):
    return {k: math.exp(v) for k, v in be.score(req, lambda: None).logits[qid].items()}


@needs_tok
def test_canonical_shim_rows_and_served_distribution_are_bitwise_invariant():
    from heads.layout import build_row
    from jobs.sym_eval.canon_plug import TEMPLATE, _backend

    be = _backend()
    crit = {"billing": "Billing and refunds", "tech": "Technical problem with the app", "sales": "Buy a larger plan",
            "sec": "Security incident or account takeover"}
    req = {"state": {"ticket": "My card was charged twice"}, "questions": {"q": _choice(crit)}}
    r0 = build_row(req, "q", None, template=TEMPLATE, layout=C.CANON_LAYOUT)
    p0 = _probs(be, req)
    rng = random.Random(3)
    for t in range(6):
        keys = list(crit)
        rng.shuffle(keys)
        ren = {k: f"id_{rng.getrandbits(20):05x}" for k in crit}
        req2 = {**req, "questions": {"q": _choice({ren[k]: crit[k] for k in keys})}}
        r1 = build_row(req2, "q", None, template=TEMPLATE, layout=C.CANON_LAYOUT)
        assert r1.ids == r0.ids and r1.continuations == r0.continuations
        assert [ren[o] for o in r0.option_ids] == r1.option_ids
        p1 = _probs(be, req2)
        assert all(p1[ren[o]] == p0[o] for o in crit)  # bitwise after mapping back
    # the ids never reach the model
    from decide.shim_format import spec_for_template

    text = spec_for_template(TEMPLATE).tok.decode(r0.ids)
    assert not any(k in text for k in ("billing:", "tech:", "sales:", "sec:"))


@needs_tok
def test_noncanonical_shim_is_not_invariant_teeth():
    from jobs.sym_eval.canon_plug import TEMPLATE, _backend
    from heads.decide_backend import DecisionModelBackend

    be = _backend()
    raw = DecisionModelBackend(be.model, None, template=TEMPLATE, layout="g1v2")
    crit = {"a": "Billing and refunds", "b": "Technical problem", "c": "Buy a plan"}
    req = {"state": "charged twice", "questions": {"q": _choice(crit)}}
    rev = {**req, "questions": {"q": _choice(dict(reversed(list(crit.items()))))}}
    pa, pb = _probs(raw, req), _probs(raw, rev)
    assert max(abs(pa[k] - pb[k]) for k in crit) > 0  # the stock shim reads display order: the gate has teeth


@needs_tok
def test_ties_score_and_noul_through_the_backend():
    from jobs.sym_eval.canon_plug import _backend

    be = _backend()
    req = {"state": "s", "questions": {"q": _choice({"x": "Same text", "y": "Same  text", "z": "Other"})}}
    p = _probs(be, req)
    assert p["x"] == p["y"]
    sreq = {"state": "s", "questions": {"q": {"type": "score", "instructions": "rate", "criteria": RUB}}}
    assert set(_probs(be, sreq)) == {"0", "1", "2", "3"}
    nreq = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "is it?"}}}
    assert set(_probs(be, nreq)) == {"true", "false"}


@needs_tok
def test_evidence_score_op_passes_the_canonical_shim(tmp_path):
    from jobs.m2_first_cell.screen import _score_inv_request
    from jobs.sym_eval.canon_plug import _backend
    from scripts import ev_score_flip as SF
    from scripts import ev_score_perturb as SP
    from tests import test_ev_score_inv as T

    be = _backend()

    def head(row):
        lg = be.score(_score_inv_request(row), lambda: None).logits["q"]
        return {"type": "score", "probabilities": {h: math.exp(lg[h]) for h in row["labels"]}}

    items = T._items()
    base, pert = SP.make(items, SP.OPS, (0, 1, 2))
    d = tmp_path / "req"
    d.mkdir()
    SP.write_rows(base, d / "base.jsonl")
    SP.write_rows(pert, d / "perturbed.jsonl")
    for name, rows in (("base", SF.rj(d / "base.jsonl")), ("perturbed", SF.rj(d / "perturbed.jsonl"))):
        (d / f"{name}.preds.jsonl").write_text("".join(json.dumps({"id": r["id"], "answer": head(r), "status": "ok"})
                                                       + "\n" for r in rows), encoding="utf-8")
    res = SF.run_dir(items, d, max_dp=0.0)
    assert res["score_inv_pass"] and res["bitwise"], res["failures"]


@needs_tok
def test_training_rows_use_the_shuffle_and_eval_rows_the_canonical_order():
    from heads.layout import build_row
    from jobs.sym_eval.canon_plug import TEMPLATE

    crit = {chr(97 + i): f"destination number {i}" for i in range(5)}
    req = {"state": "s", "questions": {"q": _choice(crit)}}
    ev = build_row(req, "q", None, template=TEMPLATE, layout=C.CANON_LAYOUT)
    tr = build_row(req, "q", None, template=TEMPLATE, layout=C.CANON_SHUFFLE_LAYOUT)
    assert sorted(ev.option_ids) == sorted(tr.option_ids) and ev.meta["canon_layout"] == C.CANON_LAYOUT
    assert tr.meta["canon_layout"] == C.CANON_SHUFFLE_LAYOUT


# ------------------------------------------------------------------------------------------------ params / cell
def _params():
    return json.loads((ROOT / "jobs/m2_first_cell/job_params_jev3_lm.json").read_text(encoding="utf-8"))


def _cell():
    return json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_lm.json").read_text(encoding="utf-8"))


def test_cell_recipe():
    c = _cell()
    assert c["head"] == "lm_label" and c["layout"] == C.CANON_LAYOUT and c["train_layout"] == C.CANON_SHUFFLE_LAYOUT
    assert c["template"] == "jrt-h2o-shim:qwen3.5-4b" and c["lora_rank"] == 16 and c["cont_chunk"] == 0
    assert c["label_smoothing_noul"] == 0.0 and c["t_ladder"][-1] == 4096 and c["assert_no_truncation"]
    assert c["qwen35"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" and not c["qwen35"].get("host")
    assert not c.get("consistency") and c["max_steps"] == c["n_train_records"] // 8
    ex = {e["file"]: e for e in c["train_set"]["exclude_files"]}
    for f, e in ex.items():
        b = (ROOT / f).read_bytes()
        assert hashlib.sha256(b).hexdigest() == e["sha256"]
        assert len([x for x in b.decode("utf-8").splitlines() if x.strip()]) == e["n"]


def test_params_budget_gates_and_registration():
    from jobs.common import costs
    from jobs.m2_first_cell import screen

    p = _params()
    assert p["mode"] == "jev3_lm" and p["cell_config"] == "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_lm.json"
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    w = costs.worst_case_usd(p)
    assert w <= 18.0 and f"{w:.2f}" in (ROOT / "jobs/m2_first_cell/JOB_SPEC.md").read_text(encoding="utf-8")
    assert "job_params_jev3_lm.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert screen.estimate(p)["ok"]
    jk = p["jevkit"]
    assert jk["paired_primary"] == "native_tfit" and jk["tfit"]["per_type"]["min_cell"] == 200
    assert jk["g1_keep_gate"] and jk["g1_max_dp"] == 0.0 and jk["canon_stock"] and jk["stock_shim"]
    assert jk["score_inv"]["keep_gate"] and jk["letter_prior"]["flag_pp"] == 5 and jk["slice_first"]
    blk = screen.jevkit_block(p, _cell()["template"])
    assert blk["score_inv"]["keep_gate"] and blk["canon_stock"]
