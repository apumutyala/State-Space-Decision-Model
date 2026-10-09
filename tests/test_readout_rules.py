"""CPU tests for the opt-in readout rules (decide.readout_rules, readout v3w-r) and that readouts without them are unchanged."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest

from decide import Decider
from decide import readout_rules as RR
from decide.backends.base import Plan, ScoreResult

YN = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
SATA_YN = {"no": "No", "yes": "Yes"}


class FixedBackend:
    """Logits given per (qid, option); plan reports ``tokens`` as the longest model input."""

    name = "fixed"

    def __init__(self, logits, tokens=100):
        self.logits, self.tokens = logits, tokens

    def revision(self):
        return {"revision_id": "fixed:v1", "backend": self.name, "model_id": None}

    def plan(self, request):
        return Plan(inference_calls=1, max_call_input_tokens=self.tokens, total_input_tokens=self.tokens)

    def score(self, request, check_deadline):
        check_deadline()
        return ScoreResult(logits={q: dict(self.logits[q]) for q in request["questions"]}, inference_calls=1,
                           total_input_tokens=self.tokens)


def bfcl_request(tools):
    return {"model": "m", "state": {"conversation": [[{"role": "user", "content": "hi"}]], "functions": [],
                                    "task": "BFCL tool-name selection"},
            "questions": {f"tool_{t}": {"type": "choice", "instructions": f"Candidate tool: {t}", "criteria": dict(YN)}
                          for t in tools}}


def sata_request(n):
    return {"model": "m", "state": {"paragraph": "p", "question": "q?", "options": [f"o{j}" for j in range(n)]},
            "questions": {f"option_{j}": {"type": "choice", "instructions": f"Candidate: o{j}",
                                          "criteria": dict(SATA_YN)} for j in range(n)}}


def yes_logits(p_yes):
    import math

    return {k: {"yes": math.log(p), "no": math.log(1 - p)} for k, p in p_yes.items()}


# ----------------------------------------------------------------------------------------------- detection
def test_detect_bfcl_and_sata_from_request_only():
    assert RR.detect(bfcl_request(["a", "b"])) == "bfcl_exactly_one"
    assert RR.detect(sata_request(4)) == "sata_top_m"


@pytest.mark.parametrize("mutate", [
    lambda r: r["state"].update(task="other task"),
    lambda r: r["questions"]["tool_a"].update(criteria={"yes": "y", "no": "n", "maybe": "m"}),
    lambda r: r["questions"]["tool_a"].update(type="noul"),
    lambda r: r.update(state="a string state"),
])
def test_detect_bfcl_negatives(mutate):
    r = bfcl_request(["a", "b"])
    mutate(r)
    assert RR.detect(r) is None


@pytest.mark.parametrize("mutate", [
    lambda r: r["state"].update(extra=1),  # not the SATA state shape
    lambda r: r["state"]["options"].append("o9"),  # option count != question count
    lambda r: r["questions"].update(option_x=r["questions"].pop("option_0")),  # ids not option_0..n-1
    lambda r: r["questions"]["option_1"].update(criteria={"A": "a", "B": "b"}),
])
def test_detect_sata_negatives(mutate):
    r = sata_request(3)
    mutate(r)
    assert RR.detect(r) is None


def test_detect_ignores_row_metadata():
    """Only state + questions are read: an eval row's id / metadata / expected never changes detection."""
    r = sata_request(3)
    row = {**copy.deepcopy(r), "id": "BFCL-tool-selection:simple:x", "metadata": {"category": "parallel"},
           "expected": {"option_0": "yes"}, "_evaluation": {"group_id": "x"}}
    assert RR.detect(row) == RR.detect(r) == "sata_top_m"


def test_detector_matches_only_its_benchmark_on_dev():
    """On the DEV slice the BFCL trigger fires on BFCL rows only and the SATA trigger on SATA rows only."""
    try:
        import sys

        from harness import config as _config

        try:
            import decision_index  # noqa: F401
        except ImportError:
            sys.path.insert(1, str(_config.KIT_CLONE))
        from harness.cli import load_slice

        rows = load_slice("dev", "0.2.1")
    except Exception as e:  # noqa: BLE001 - the kit/data are not in every checkout
        pytest.skip(f"DEV slice unavailable: {type(e).__name__}")
    hits = {}
    for r in rows:
        rule = RR.detect({"state": r["state"], "questions": r["questions"]})
        if rule:
            hits.setdefault(rule, set()).add(r["family"])
    assert hits == {"bfcl_exactly_one": {"BFCL-tool-selection"}, "sata_top_m": {"SATA-Bench"}}


# ----------------------------------------------------------------------------------------------- the rules
def test_bfcl_exactly_one_argmax_and_ties():
    assert RR.selected_yes("bfcl_exactly_one", {"tool_a": 0.9, "tool_b": 0.95, "tool_c": 0.1}) == {"tool_b"}
    assert RR.selected_yes("bfcl_exactly_one", {"tool_a": 0.1, "tool_b": 0.2}) == {"tool_b"}  # all-no -> one yes
    assert RR.selected_yes("bfcl_exactly_one", {"tool_b": 0.5, "tool_a": 0.5}) == {"tool_a"}  # tie: smallest id


@pytest.mark.parametrize("p, want", [
    ({"option_0": 0.9, "option_1": 0.8, "option_2": 0.4, "option_3": 0.1}, {"option_0", "option_1"}),  # sum 2.2 -> 2
    ({"option_0": 0.1, "option_1": 0.05, "option_2": 0.2}, {"option_2"}),  # sum 0.35 -> max(1, 0) = 1
    ({"option_0": 0.9, "option_1": 0.9, "option_2": 0.9}, {"option_0", "option_1", "option_2"}),  # 2.7 -> 3
    ({"option_0": 0.6, "option_1": 0.4, "option_2": 0.4, "option_3": 0.6}, {"option_0", "option_3"}),  # 2.0 -> 2
    ({"option_0": 0.5, "option_1": 0.5, "option_2": 0.5}, {"option_0", "option_1"}),  # 1.5 -> 2 (half-even), tie ids
])
def test_sata_top_m(p, want):
    assert RR.selected_yes("sata_top_m", p) == want


def test_sata_m_uses_model_probabilities_never_gold():
    """m comes from sum p(yes) of the model; the gold count cannot enter: no parameter carries it, and a request
    carrying an ``expected`` field (with a different yes-count) gives the identical answers."""
    params = set(inspect.signature(RR.apply_rules).parameters) | set(inspect.signature(RR.selected_yes).parameters)
    assert not params & {"expected", "gold", "labels", "n_gold", "row"}
    r = sata_request(4)
    p = {"option_0": 0.9, "option_1": 0.7, "option_2": 0.45, "option_3": 0.05}  # sum 2.1 -> m = 2
    probs = {k: {"yes": v, "no": 1 - v} for k, v in p.items()}

    def answers():
        return {k: {"type": "choice", "choice": "yes" if v >= 0.5 else "no", "confidence": max(v, 1 - v),
                    "probabilities": dict(probs[k])} for k, v in p.items()}

    a1, a2 = answers(), answers()
    RR.apply_rules(r, a1, probs)
    RR.apply_rules({**r, "expected": {k: "yes" for k in p}}, a2, probs)  # gold says 4 yes
    assert a1 == a2
    assert {k for k, a in a1.items() if a["choice"] == "yes"} == {"option_0", "option_1"}


def test_apply_rules_changes_choice_only():
    r = bfcl_request(["a", "b", "c"])
    probs = {"tool_a": {"yes": 0.8, "no": 0.2}, "tool_b": {"yes": 0.6, "no": 0.4}, "tool_c": {"yes": 0.1, "no": 0.9}}
    ans = {k: {"type": "choice", "choice": max(v, key=v.get), "confidence": max(v.values()), "probabilities": dict(v)}
           for k, v in probs.items()}
    before = copy.deepcopy(ans)
    d = RR.apply_rules(r, ans)  # offline path: the wire answers' own probabilities
    assert d == {"id": RR.RULES_ID, "rule": "bfcl_exactly_one", "changed": ["tool_b"]}
    assert ans["tool_b"]["choice"] == "no" and ans["tool_b"]["confidence"] == 0.4
    for k in ans:
        assert ans[k]["probabilities"] == before[k]["probabilities"]


def test_non_matching_request_untouched():
    r = {"model": "m", "state": {"x": 1}, "questions": {"q": {"type": "choice", "criteria": {"yes": "", "no": ""}}}}
    ans = {"q": {"type": "choice", "choice": "yes", "confidence": 0.7, "probabilities": {"yes": 0.7, "no": 0.3}}}
    before = copy.deepcopy(ans)
    assert RR.apply_rules(r, ans)["rule"] is None and ans == before


# ----------------------------------------------------------------------------------------------- adapter
def _strip(resp):
    out = copy.deepcopy(resp)
    out["ssdm"]["diagnostics"].pop("latency_ms", None)
    return out


def test_decider_rules_bfcl_end_to_end():
    r = bfcl_request(["a", "b"])
    be = FixedBackend(yes_logits({"tool_a": 0.7, "tool_b": 0.9}))
    old = Decider(be).decide(r)
    new = Decider(be, readout_rules=True).decide(r)
    assert old["ssdm"]["status"] == new["ssdm"]["status"] == "ok"
    assert {k: a["choice"] for k, a in old["answers"].items()} == {"tool_a": "yes", "tool_b": "yes"}
    assert {k: a["choice"] for k, a in new["answers"].items()} == {"tool_a": "no", "tool_b": "yes"}
    for k in r["questions"]:  # probabilities untouched
        assert new["answers"][k]["probabilities"] == old["answers"][k]["probabilities"]
    assert new["ssdm"]["served_probabilities"] == old["ssdm"]["served_probabilities"]
    assert new["ssdm"]["served_config"]["readout_rules"] == {"id": RR.RULES_ID, "rule": "bfcl_exactly_one",
                                                             "changed": ["tool_a"]}
    assert "readout_rules" not in old["ssdm"]["served_config"]
    assert new["ssdm"]["diagnostics"]["revision_id"] == f"fixed:v1+R:{RR.RULES_ID}"


def test_decider_rules_sata_end_to_end():
    r = sata_request(4)
    be = FixedBackend(yes_logits({"option_0": 0.45, "option_1": 0.45, "option_2": 0.45, "option_3": 0.2}))
    new = Decider(be, readout_rules=True).decide(r)  # sum 1.55 -> m = 2 (option_0, option_1 by id tie-break)
    assert {k for k, a in new["answers"].items() if a["choice"] == "yes"} == {"option_0", "option_1"}


@pytest.mark.parametrize("tokens, old_status, new_status", [
    (16_384, "ok", "ok"), (16_385, "exhausted", "ok"), (32_768, "exhausted", "ok"), (32_769, "exhausted", "exhausted")])
def test_rules_readout_serves_under_32k_cap(tokens, old_status, new_status):
    r = sata_request(2)
    be = FixedBackend(yes_logits({"option_0": 0.6, "option_1": 0.3}), tokens=tokens)
    old, new = Decider(be).decide(r), Decider(be, readout_rules=True).decide(r)
    assert old["ssdm"]["status"] == old_status and new["ssdm"]["status"] == new_status
    if new_status == "ok":
        lim = new["ssdm"]["diagnostics"]["limits_applied"]
        assert lim["max_input_tokens"] == RR.READOUT_RULES_CAP == 32_768
        assert lim["_source"]["max_input_tokens"] == "readout_rules_cap"
    if old_status == "ok":
        assert old["ssdm"]["diagnostics"]["limits_applied"]["max_input_tokens"] == 16_384


def test_rules_cap_still_bounded_by_request_and_backend():
    r = sata_request(2)
    be = FixedBackend(yes_logits({"option_0": 0.6, "option_1": 0.3}), tokens=20_000)
    req = {**r, "ssdm": {"limits": {"max_input_tokens": 10_000}}}
    assert Decider(be, readout_rules=True).decide(req)["ssdm"]["status"] == "exhausted"
    assert Decider(be, readout_rules=True, backend_max_input_tokens=8_192).decide(r)["ssdm"]["status"] == "exhausted"


# ----------------------------------------------------------------------------------------------- unchanged
#: sha256 of the canonical response (latency removed) of each request below through ``Decider(FixedBackend)`` as
#: computed on origin/agent-main f1d8b2d (before decide.readout_rules existed): the default readout is byte-identical.
GOLDEN = {
    "bfcl": "4dfec1704a74164f200508e8cff2a0ac38cc19958fa6b530dba84e31d669ce67",
    "sata": "3a2fa583fdd6a3909691486291aa5f0bb88f41a1da9d2ad55fba2782adb56d4f",
    "sata_over_16k": "782e14d38e464ef5474d6838526f8f87fecb448512f1e7b6157573eefe0ed5b2",
}


def _golden_cases():
    return {
        "bfcl": (bfcl_request(["a", "b"]), FixedBackend(yes_logits({"tool_a": 0.7, "tool_b": 0.9}))),
        "sata": (sata_request(4), FixedBackend(yes_logits({"option_0": 0.45, "option_1": 0.45, "option_2": 0.45,
                                                           "option_3": 0.2}))),
        "sata_over_16k": (sata_request(2), FixedBackend(yes_logits({"option_0": 0.6, "option_1": 0.3}),
                                                        tokens=20_000)),
    }


def _digest(resp):
    return hashlib.sha256(json.dumps(_strip(resp), sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_default_readout_byte_identical_to_agent_main(name):
    req, be = _golden_cases()[name]
    assert _digest(Decider(be).decide(req)) == GOLDEN[name]
    assert _digest(Decider(be, readout_rules=False).decide(req)) == GOLDEN[name]


def test_readout_engine_registry():
    from jobs.common import readout_engine as E

    assert E.RULES_FAMILIES == {"rwkv7-hf-g1-v3w-r": "rwkv7-hf-g1-v3w"}
    assert not set(E.RULES_FAMILIES) & set(E.G1V3_MODES)  # existing families keep their construction path
    assert E.G1V3_MODES["rwkv7-hf-g1-v3w"] == "words" and E.TEMPLATES["rwkv7-hf-g1-v3w"] == "jrt-rwkv-g1-v3w"
    assert E.TEMPLATES["rwkv7-hf-g1-v3w-r"] == "jrt-rwkv-g1-v3w"


# ----------------------------------------------------------------------------------------------- untempered rebuild
def _kit_or_skip():
    import sys

    from harness import config as _config

    try:
        import decision_index  # noqa: F401
    except ImportError:
        sys.path.insert(1, str(_config.KIT_CLONE))
    pytest.importorskip("decision_index.engines.base")


def _sata_row():
    return {**sata_request(4), "_evaluation": {"run_id": "33:SATA:test:x", "catalog_id": 33}}


def test_index_predict_untempered_rebuild_reapplies_rules_on_raw():
    """train.m2_gates.index_predict(untempered_from_raw=True) rebuilds answers from raw probabilities: the rules must be
    re-applied there (on raw), giving exactly the identity-calibration rules decider's answers."""
    _kit_or_skip()
    from decide.calibration import OptionCountTemperature
    from train import m2_gates as G

    row = _sata_row()
    be = FixedBackend(yes_logits({"option_0": 0.9, "option_1": 0.1, "option_2": 0.1, "option_3": 0.1}))
    served = Decider(be, calibration=OptionCountTemperature(t0=3.0), readout_rules=True)  # tempered: sum 1.65 -> m 2
    plain = Decider(be, readout_rules=True)  # raw: sum 1.2 -> m 1
    a = G.index_predict(served.decide, [row], untempered_from_raw=True)["33:SATA:test:x"]
    b = G.index_predict(plain.decide, [row])["33:SATA:test:x"]
    assert a["response"] == b["response"]
    assert {k for k, x in a["response"]["answers"].items() if x["choice"] == "yes"} == {"option_0"}
    assert {k for k, x in a["served_response"]["answers"].items() if x["choice"] == "yes"} == {"option_0", "option_1"}
    assert a["readout_rules_untempered"]["rule"] == "sata_top_m"
    # without rules the rebuild is unchanged (no rules record, no re-application)
    c = G.index_predict(Decider(be, calibration=OptionCountTemperature(t0=3.0)).decide, [row],
                        untempered_from_raw=True)["33:SATA:test:x"]
    assert "readout_rules_untempered" not in c


def test_index_predict_fails_closed_when_rules_record_missing():
    _kit_or_skip()
    from train import m2_gates as G

    row = _sata_row()
    dec = Decider(FixedBackend(yes_logits({f"option_{j}": 0.3 for j in range(4)})), readout_rules=True)

    def stripped(req):
        resp = dec.decide(req)
        resp["ssdm"]["served_config"].pop("readout_rules")
        return resp

    with pytest.raises(ValueError, match="readout-rules decider"):
        G.index_predict(stripped, [row], untempered_from_raw=True)


def test_claim_cap_set_per_request_overrides_rules_default():
    """32k is the v3w-r family default, not the claim cap: a claim arm's per-request cap (R3: 65,536, every DEV row in
    full; the longest is 47,509 tokens) overrides it."""
    r = sata_request(2)
    be = FixedBackend(yes_logits({"option_0": 0.6, "option_1": 0.3}), tokens=47_509)
    assert Decider(be, readout_rules=True).decide(r)["ssdm"]["status"] == "exhausted"
    resp = Decider(be, readout_rules=True).decide({**r, "ssdm": {"limits": {"max_input_tokens": 65_536}}})
    assert resp["ssdm"]["status"] == "ok"
    assert resp["ssdm"]["diagnostics"]["limits_applied"]["_source"]["max_input_tokens"] == "request"
