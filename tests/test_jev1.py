"""data/jev1.py synthetic generators: deterministic, structurally valid, gold exactly once, rule-derived gold."""
from __future__ import annotations

import collections
import json

from data import augment as A
from data import jev1 as J


def _norm(s):
    return J.norm(s)


def _synth(i):
    return [J._ticket_route(i), J._model_route(i), J._agent_audit(i), J._access_policy(i), J._notice_enum(i),
            J._event_state(i)]


def test_deterministic_and_valid():
    a = [json.dumps(r, sort_keys=True) for i in range(40) for r in _synth(i)]
    b = [json.dumps(r, sort_keys=True) for i in range(40) for r in _synth(i)]
    assert a == b
    for line in a:
        r = json.loads(line)
        A.validate(r)
        q, g = r["questions"]["q"], r["labels"]["q"]
        assert list(r["questions"]) == ["q"] and q["type"] in ("choice", "noul", "score")
        if q["type"] == "choice":
            t = [_norm(v) for v in q["criteria"].values()]
            assert len(set(t)) == len(t) and g in q["criteria"]


def test_long_policy_and_severity_generate():
    rows = list(J.gen_long_policy(30))
    assert len(rows) == 30 and {r["aug"]["jev1"]["kind"] for r in rows} == {"expense", "returns", "leave"}
    for r in rows:
        A.validate(r)
    sev = [r for r in J.gen_severity(25, 0)]
    assert len(sev) == 25 and all(r["questions"]["q"]["type"] == "score" for r in sev)


def test_model_routing_gold_is_cheapest_eligible():
    for i in range(200):
        r = J._model_route(i)
        st = r["state"]
        req = st["request"]
        ok = [m for m in st["model_catalog"] if m["context_tokens"] >= req["input_tokens"]
              and set(req["required_capabilities"]) <= set(m["capabilities"])
              and m["p50_latency_ms"] <= req["max_p50_latency_ms"] and m["quality_tier"] >= req["min_quality_tier"]]
        if q["type"] == "noul":
            assert (r["labels"]["q"] == "true") == bool(ok)
        else:
            want = min(ok, key=lambda m: (m["usd_per_million_tokens"], m["p50_latency_ms"]))["name"] if ok \
                else "no_eligible_model"
            assert r["labels"]["q"] == want


def test_option_order_shuffled_uniformly():
    pos = collections.Counter()
    for i in range(600):
        r = J._ticket_route(i)
        if q["type"] == "choice":
            ids = list(q["criteria"])
            pos[ids.index(r["labels"]["q"]) * 4 // len(ids)] += 1
    assert min(pos.values()) > 0.15 * sum(pos.values())


def test_pool_serialisation_keeps_option_order():
    # the shim letters options in criteria order: the build must not re-sort keys (sorted ids = gold at its id rank)
    r = next(x for x in (J._access_policy(i) for i in range(50)) if x["questions"]["q"]["type"] == "choice")
    line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
    assert list(json.loads(line)["questions"]["q"]["criteria"]) == list(r["questions"]["q"]["criteria"])
    import inspect
    from scripts import jev1_build
    assert "sort_keys" not in inspect.getsource(jev1_build.cmd_gen)


def test_no_item_text_in_aug():
    # harness.heldout reads non-content keys as tags: aug may carry key NAMES only, never source text
    for i in range(20):
        for r in _synth(i):
            assert "item" not in r["aug"]["jev1"]
            ks = r["aug"]["jev1"].get("item_keys")
            assert ks is None or all(k in r["state"] for k in ks)


def test_access_policy_p5_overrides_only_p1_p4():
    for i in range(3000):
        r = J._access_policy(i)
        req = r["state"]["access_request"]
        pol = r["state"]["access_policy"]
        banned = pol[1].split("Purposes ", 1)[1].split(" are never")[0].split(" and ")
        lvl = J.DATA[req["data_class"]]
        if (req["role"] == "auditor" and req["open_audit_ticket"] and req["purpose"] == "annual_audit"
                and lvl >= 2 and req["purpose"] in banned):
            assert r["labels"]["q"] in ("false", "deny_purpose"), r["id"]


def test_leave_blackout_allows_emergency():
    for i in range(3000):
        rr = J.rng("lp-test", i)
        _, case, gold = J._lp_leave(rr)
        if case["blackout"] and case["emergency"] and case["requested_days"] <= 2:
            assert gold["ok"] and gold["why"] == "L7"
        if case["blackout"] and not case["emergency"]:
            assert not gold["ok"] and gold["why"] == "L4"
