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


def test_bridge_requires_v0_1_deltas():
    with pytest.raises(ValueError):
        DecideBridge(Decider(OracleBackend()))


def test_bridge_maps_refusal(reqs):
    br = DecideBridge(Decider(OracleBackend(fail_with=("control_tokens_in_input", "refused")), enable_v0_1_deltas=True))
    r = br(data.strip_meta(reqs[0]))
    assert r["status"] == "unsupported" and r["error"] == "control_tokens_in_input"
