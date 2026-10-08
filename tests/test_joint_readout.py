"""R4 joint readout (decide.joint; G1V3Backend modes joint / joint_seq = rwkv7-hf-g1-v3j / -v3js).

Pure checks run everywhere; the backend checks need the pinned G1k tokenizer and torch (the tiny random RWKV7 of
tests/test_rwkv_g1_v3.py, CPU fp32).
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from decide.joint import JOINT_RULE_ID, candidate_block, cardinality_max_marginals, joint_logits
from decide.prompt_g1v3 import ANSWERS_HEADER, answers_so_far, render_g1v3
from decide.schema import canonical_question_ids

YN = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
STEM = "Mark each candidate tool yes if it should be invoked, or no otherwise.\n\n"


def block_req(names=("alpha", "beta", "gamma")):
    return {"model": "m", "state": "User: get the maintenance config for sandcastle.",
            "questions": {f"tool_{n}": {"type": "choice", "instructions": STEM + f"Candidate tool: {n}", "criteria": dict(YN)}
                          for n in names}}


def mixed_req():
    r = block_req()
    r["questions"]["handling"] = {"type": "choice", "instructions": "How should the assistant handle it?",
                                  "criteria": {"a": "execute now", "b": "ask a clarifying question", "c": "block"}}
    return r


def logp(p_yes):
    return {"yes": math.log(p_yes), "no": math.log(1 - p_yes)}


def test_candidate_block_rule():
    assert candidate_block(block_req()) == ["tool_alpha", "tool_beta", "tool_gamma"]
    assert candidate_block(block_req(("only",))) is None  # n >= 2
    assert candidate_block(mixed_req()) is None  # every question must be a yes/no candidate
    r = block_req()
    r["questions"]["tool_beta"]["instructions"] = "Something else entirely.\nCandidate tool: beta"
    assert candidate_block(r) is None  # one stem
    r = block_req()
    r["questions"]["tool_beta"]["criteria"] = {"k_1": "yes", "k_2": "no"}
    assert candidate_block(r) is None  # option ids must be yes/no
    r = block_req()
    for q in r["questions"].values():
        q["instructions"] = q["instructions"].splitlines()[-1]
    assert candidate_block(r) is None  # no stem (single line)
    n = {"model": "m", "state": "s", "questions": {f"o{i}": {"type": "noul", "instructions": f"Q: which apply?\nCandidate: {i}"}
                                                    for i in range(3)}}
    assert candidate_block(n) == ["o0", "o1", "o2"]


def brute(lp_yes, lp_no, lo, hi):
    n = len(lp_yes)
    best = [[-math.inf, -math.inf] for _ in range(n)]
    for x in itertools.product((0, 1), repeat=n):
        if not lo <= sum(x) <= hi:
            continue
        s = sum(lp_yes[i] if x[i] else lp_no[i] for i in range(n))
        for i in range(n):
            j = 0 if x[i] else 1
            best[i][j] = max(best[i][j], s)
    return [tuple(b) for b in best]


def test_max_marginals_match_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 7)
        lo = rng.randint(0, n)
        hi = rng.randint(lo, n)
        p = [rng.uniform(0.01, 0.99) for _ in range(n)]
        y, z = [math.log(v) for v in p], [math.log(1 - v) for v in p]
        got, exp = cardinality_max_marginals(y, z, lo, hi), brute(y, z, lo, hi)
        for g, e in zip(got, exp):
            for a, b in zip(g, e):
                assert (a == b == -math.inf) or abs(a - b) < 1e-9
    with pytest.raises(ValueError):
        cardinality_max_marginals([0.0], [0.0], 2, 2)


def test_joint_logits_identity_flip_and_distributions():
    lg = {"tool_alpha": logp(0.3), "tool_beta": logp(0.45), "tool_gamma": logp(0.1)}
    # not a block -> the very same dict, nothing recomputed
    m = mixed_req()
    lgm = dict(lg, handling={"a": -1.0, "b": -0.5, "c": -2.0})
    out, d = joint_logits(m, lgm)
    assert out is lgm and d == {"rule": JOINT_RULE_ID, "applied": False}
    # all-no block -> exactly the most likely yes flips; every output is a proper distribution
    out, d = joint_logits(block_req(), lg)
    assert d["applied"] and d["changed"] == ["tool_beta"] and d["bounds"] == [1, 3]
    for qid, v in out.items():
        assert abs(sum(math.exp(x) for x in v.values()) - 1) < 1e-9
    assert out["tool_beta"]["yes"] > out["tool_beta"]["no"]
    assert all(out[q]["no"] > out[q]["yes"] for q in ("tool_alpha", "tool_gamma"))
    # the argmax of the max-marginals is the brute-force joint MAP
    ids = candidate_block(block_req())
    best = max(itertools.product((0, 1), repeat=3), key=lambda x: (sum(x) >= 1) * 1e9 + sum(
        lg[q]["yes"] if b else lg[q]["no"] for q, b in zip(ids, x)))
    assert tuple(int(out[q]["yes"] > out[q]["no"]) for q in ids) == best
    # a slack constraint (>= 2 independent yes answers) leaves every distribution exactly as it was
    lg2 = dict(lg, tool_gamma=logp(0.8), tool_beta=logp(0.7))
    out2, d2 = joint_logits(block_req(), lg2)
    assert d2["changed"] == []
    for q in ids:
        assert all(abs(out2[q][o] - lg2[q][o]) < 1e-9 for o in ("yes", "no"))
    # exactly one yes: the MAP is unchanged, only that question's distribution sharpens (its "no" needs a flip)
    out4, d4 = joint_logits(block_req(), dict(lg, tool_gamma=logp(0.8)))
    assert d4["changed"] == [] and math.exp(out4["tool_gamma"]["yes"]) > 0.8
    assert all(abs(out4[q][o] - lg[q][o]) < 1e-9 for q in ("tool_alpha", "tool_beta") for o in ("yes", "no"))
    # expected-cardinality decode: k = round(sum p) = round(1.65) = 2 -> the two most likely yes
    lg3 = {"tool_alpha": logp(0.6), "tool_beta": logp(0.45), "tool_gamma": logp(0.6)}
    out3, d3 = joint_logits(block_req(), lg3, "card")
    assert d3["bounds"] == [2, 2] and sorted(q for q in ids if out3[q]["yes"] > out3[q]["no"]) == ["tool_alpha", "tool_gamma"]
    with pytest.raises(ValueError):
        joint_logits(block_req(), lg, "beam")


def test_answers_block_rendering():
    from decide.backends.rwkv_g1_v3 import V3_TABLE

    req = mixed_req()
    qids = canonical_question_ids(req["questions"])
    assert answers_so_far(req, []) == ""
    for qid in qids:  # no block / empty block -> the exact v3 prompt
        assert render_g1v3(req, qid, V3_TABLE, answers_block=None) == render_g1v3(req, qid, V3_TABLE)
        assert render_g1v3(req, qid, V3_TABLE, answers_block="") == render_g1v3(req, qid, V3_TABLE)
    blk = answers_so_far(req, [("handling", "b"), ("tool_beta", "no")])
    assert blk.startswith(ANSWERS_HEADER) and "\n\n" not in blk and blk.endswith("\n")
    assert f"FIELD {qids.index('handling') + 1} (handling) = b: ask a clarifying question" in blk
    s0, s1 = render_g1v3(req, "tool_alpha", V3_TABLE), render_g1v3(req, "tool_alpha", V3_TABLE, answers_block=blk)
    cut = s0.index("\nEND STATE\n") + len("\nEND STATE\n")
    assert s1 == s0[:cut] + blk + s0[cut:]  # inserted after END STATE only; the shared prefix is untouched


# ------------------------------------------------------------------------- backend (tiny RWKV7, real tokenizer)
from tests.test_rwkv_g1_v3 import _enc, _tiny_plugin, needs_tok  # noqa: E402


@needs_tok
def test_backend_joint_modes():
    torch = pytest.importorskip("torch")
    from decide import Decider
    from decide.backends.rwkv_g1_v3 import G1V3Backend

    plug, model = _tiny_plugin(_enc())
    cat = G1V3Backend(plug, mode="catalog")
    for req in (block_req(), mixed_req()):
        base = cat.score(req, lambda: None).logits
        bj = G1V3Backend(plug, mode="joint")
        rj = bj.score(req, lambda: None)
        exp, d = joint_logits(req, base)
        assert rj.extras["joint"] == d and rj.inference_calls == cat.score(req, lambda: None).inference_calls
        for qid in base:
            assert all(abs(rj.logits[qid][o] - exp[qid][o]) < 1e-5 for o in base[qid])
        if not d["applied"]:
            assert all(abs(rj.logits[q][o] - base[q][o]) < 1e-6 for q in base for o in base[q])

        bs = G1V3Backend(plug, mode="joint_seq")
        rs = bs.score(req, lambda: None)
        seq = rs.extras["sequential"]
        order = seq["order"]
        assert sorted(order) == sorted(base) and not seq["skipped"]
        assert set(seq["first_pass_logits"]) == set(order[1:])
        # each re-scored question == a full forward over its sequential prompt (own earlier argmax answers only)
        chosen = [(order[0], max(sorted(base[order[0]]), key=lambda o: base[order[0]][o]))]
        indep = dict(base)
        for qid in order[1:]:
            q = next(x for x in bs._questions(req) if x["qid"] == qid)
            _, ids, _ = bs._render(req, qid, block=answers_so_far(req, chosen))
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float(), -1)
            indep[qid] = {o: lp[t].item() for o, t in zip(q["options"], q["tokens"])}
            chosen.append((qid, max(sorted(indep[qid]), key=lambda o: indep[qid][o])))
        exp, _ = joint_logits(req, indep)
        for qid in base:
            assert all(abs(rs.logits[qid][o] - exp[qid][o]) < 1e-4 for o in base[qid]), qid
        plan = bs.plan(req)
        assert plan.inference_calls >= rs.inference_calls and plan.total_input_tokens >= rs.total_input_tokens
        for be in (bj, bs):
            r = Decider(be, enable_v0_1_deltas=True).decide(req)
            assert r["ssdm"]["status"] == "ok", r["ssdm"]
            for qid, a in r["answers"].items():
                pr = r["ssdm"]["raw_probabilities"][qid]
                assert abs(sum(pr.values()) - 1) < 1e-6 and a["choice"] == max(sorted(pr), key=lambda o: pr[o])
    assert G1V3Backend(plug, mode="joint").revision()["prompt_template"]["id"] == "jrt-rwkv-g1-v3j"
    assert G1V3Backend(plug, mode="joint_seq").revision()["prompt_template"]["id"] == "jrt-rwkv-g1-v3js"


def test_engine_and_variants_know_v3j():
    from jobs.common.readout_engine import G1V3_MODES, TEMPLATES
    from jobs.rwkv7.readout_variants import DEFAULT_ORDER, VARIANTS, parse_variants

    assert G1V3_MODES["rwkv7-hf-g1-v3j"] == "joint" and G1V3_MODES["rwkv7-hf-g1-v3js"] == "joint_seq"
    assert TEMPLATES["rwkv7-hf-g1-v3j"] == "jrt-rwkv-g1-v3j" and TEMPLATES["rwkv7-hf-g1-v3js"] == "jrt-rwkv-g1-v3js"
    assert VARIANTS["v3j"] == "rwkv7-hf-g1-v3j" and "v3j" not in parse_variants(DEFAULT_ORDER)
