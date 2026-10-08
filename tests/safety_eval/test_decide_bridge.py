"""safety_eval over Track D's decide() adapter (PR #6). Skipped unless decide/ is importable
(repo root, or SSDM_DECIDE_PATH pointing at a checkout that contains decide/)."""
import pytest

from safety_eval import attacks, data, metrics
from safety_eval.decide_bridge import DecideBridge, import_decide, to_decide_request

try:
    import_decide()
    from decide import Decider
    from decide.backends import OracleBackend
except ImportError:
    pytest.skip("decide/ not importable (set SSDM_DECIDE_PATH)", allow_module_level=True)


@pytest.fixture(scope="module")
def reqs():
    return data.build(8, 6, split="dev")


def test_translation_carries_policy_not_provenance(reqs):
    w = to_decide_request(data.strip_meta(reqs[0]))
    assert w["ssdm"]["policy_restriction"] == reqs[0]["policy"]["allowed"]
    assert "provenance" not in w and "policy" not in w  # decide v0: all request text untrusted


def test_bridge_end_to_end_on_oracle(reqs):
    br = DecideBridge(Decider(OracleBackend(), enable_v0_1_deltas=True))
    vs = attacks.build_variants(reqs)
    rows = metrics.run(br, vs)
    assert all(r["status"] == "ok" for r in rows)
    for r, v in zip(rows, vs):
        assert r["executed"] in v["policy"]["allowed"][data.QKEY]  # policy-conditioned choice is executed
        assert r["violation"] is False
    s = metrics.summarise(rows, n_boot=50)
    assert s["families"]["d1_A"]["n_units"] == len(reqs) * 4


def test_bridge_sends_current_envelope_and_restriction_is_core(reqs):
    """Contract 0.1 (decide 0.2.0): envelope schema comes from decide.versions, and policy_restriction is core,
    so a Decider WITHOUT enable_v0_1_deltas must still return policy-conditioned probabilities that the bridge
    executes. (Replaces the 0.1.0-era test that required the deltas flag.)"""
    from decide import versions
    w = to_decide_request(data.strip_meta(reqs[0]))
    assert w["ssdm"]["schema"] == versions.ENVELOPE_SCHEMA
    br = DecideBridge(Decider(OracleBackend()))
    vs = attacks.build_variants(reqs[:4], hijack=["ignore_previous"], d1=["urgency"])
    rows = metrics.run(br, vs)
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["executed"] in v["policy"]["allowed"][data.QKEY] and r["violation"] is False
               for r, v in zip(rows, vs))


class _Fake:
    """Decider stand-in returning a fixed response (contract-violation cases the oracle cannot produce)."""

    def __init__(self, resp):
        self.resp = resp

    def decide(self, request):
        return self.resp


def _ok_resp(req, pcp):
    resp = Decider(OracleBackend()).decide(to_decide_request(req))
    if pcp is None:
        del resp["ssdm"]["policy_conditioned_probabilities"]
    else:
        resp["ssdm"]["policy_conditioned_probabilities"] = pcp
    return resp


def test_bridge_never_falls_back_to_unrestricted_served(reqs):
    req = data.strip_meta(reqs[0])
    # restriction sent, no conditioning returned: explicit deferral (never a served/legacy argmax)
    rows = metrics.run(DecideBridge(_Fake(_ok_resp(req, None))), [reqs[0] | {"_meta": dict(reqs[0]["_meta"],
                       variant="clean", attack_family="clean", template="clean")}])
    assert rows[0]["status"] == "unsupported" and rows[0]["error"] == "policy_conditioned_unavailable"
    # zero allowed served mass: nothing executes -> unsupported with an explicit code
    stage = {data.QKEY: {"status": "unavailable_zero_served_mass", "probabilities": None}}
    r = DecideBridge(_Fake(_ok_resp(req, stage)))(req)
    assert r["status"] == "unsupported" and r["error"] == "policy_conditioned_unavailable"


def test_bridge_maps_refusal(reqs):
    br = DecideBridge(Decider(OracleBackend(fail_with=("control_tokens_in_input", "refused")), enable_v0_1_deltas=True))
    r = br(data.strip_meta(reqs[0]))
    assert r["status"] == "unsupported" and r["error"] == "control_tokens_in_input"
