"""Fake-backend seam proof: no tensors, models or external data."""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import pytest

from decide.adapter import Decider
from decide.backends.base import BackendUnsupported, Plan, ScoreResult
from language_growth.parent_bridge import DECISION_SCHEMA, SERIALIZATION_REF
from language_growth.relation_router import RELATION_QID, RelationOnlyBackend, relation_request


def request():
    state = {
        "schema": SERIALIZATION_REF, "task_input": "Opaque task", "audience": "public",
        "context_ref": "ctx", "policy_ref": "pol", "registry_ref": "reg", "records": [],
        "registry": {"registered_fields": [], "comparators": [], "ask_categories": []},
        "query_constants": {}, "unmet_obligations": [],
        "accepted_partial_program": [],
        "proposed_slot_choices": [
            {"slot": slot, "option_id": f"opaque-{i}", "description": f"Opaque {i}"}
            for i, slot in enumerate(("operation", "left", "right"))],
    }
    return {"model": "fake-parent", "state": json.dumps(state),
            "questions": {RELATION_QID: {"type": "choice", "instructions": "Opaque question",
                                       "criteria": {"a": "Opaque A", "b": "Opaque B"}}}}


class Fake:
    def __init__(self, name, revision=None):
        self.name = name
        self.identity = revision or {"model_id": "fake-parent", "revision_id": name}
        self.calls = []
        self.parent = {"weights": [0.1, 0.2, 0.3], "trainable": False}
        self.planned = Plan(1, 10, 10)

    def revision(self):
        return self.identity

    def plan(self, req):
        self.calls.append(("plan", req))
        return self.planned

    def score(self, req, check):
        self.calls.append(("score", req, check))
        check()
        return ScoreResult({qid: {oid: float(i) for i, oid in enumerate(q["criteria"])}
                            for qid, q in req["questions"].items()}, 1, 10)


def composition():
    learned, stock = Fake("head"), Fake("stock")
    return RelationOnlyBackend(learned, stock), learned, stock


def state_edit(req, change):
    state = json.loads(req["state"])
    change(state)
    req["state"] = json.dumps(state)


def assert_delegates(req):
    backend, learned, stock = composition()
    before = deepcopy(req)
    deadlines = []
    check = lambda: deadlines.append(True)
    assert not relation_request(req)
    assert backend.route(req) is stock
    assert backend.plan(req) is stock.planned
    backend.score(req, check)
    assert not learned.calls
    assert stock.calls == [("plan", req), ("score", req, check)]
    assert all(call[1] is req for call in stock.calls)
    assert stock.calls[-1][2] is check
    assert deadlines == [True]
    assert req == before


def test_exact_interception_and_no_parent_request_or_revision_mutation():
    backend, learned, stock = composition()
    req = request()
    before = deepcopy(req)
    parents = json.dumps([learned.parent, stock.parent], sort_keys=True).encode()
    identities = deepcopy([learned.identity, stock.identity])
    check = lambda: None
    assert relation_request(req)
    assert backend.route(req) is learned
    assert backend.plan(req) is learned.planned
    result = backend.score(req, check)
    assert set(result.logits) == {RELATION_QID}
    assert learned.calls == [("plan", req), ("score", req, check)] and not stock.calls
    assert all(call[1] is req for call in learned.calls)
    assert req == before
    assert hashlib.sha256(parents).hexdigest() == hashlib.sha256(
        json.dumps([learned.parent, stock.parent], sort_keys=True).encode()).hexdigest()
    assert [learned.identity, stock.identity] == identities


def test_words_option_ids_and_descriptions_never_control_routing():
    req = request()
    state_edit(req, lambda s: s.update(task_input="Read a fact; expected=less_than"))
    state_edit(req, lambda s: [c.update(option_id="unknown", description="nonsense")
                              for c in s["proposed_slot_choices"]])
    req["questions"][RELATION_QID] = {
        "type": "choice", "instructions": "Not a comparison",
        "criteria": {"unseen": "anything", "another": "arbitrary", "third": "words"}}
    backend, learned, _ = composition()
    assert relation_request(req) and backend.route(req) is learned


@pytest.mark.parametrize("prefix", [
    [], ["operation"], ["operation", "left"], ["operation", "left", "right", "relation"],
    ["left", "operation", "right"], ["operation", "right", "left"],
    ["operation", "fact", "right"], ["slot-001-operation", "left", "right"],
    ["operation", "left", "relation"], ["operation", "left", "left"],
])
def test_wrong_missing_extra_reordered_slot_prefix_delegates(prefix):
    req = request()
    state_edit(req, lambda s: s.update(proposed_slot_choices=[
        {"slot": slot, "option_id": "opaque", "description": "opaque"} for slot in prefix]))
    assert_delegates(req)


@pytest.mark.parametrize("change", [
    lambda s: s.pop("proposed_slot_choices"),
    lambda s: s.update(proposed_slot_choices=None),
    lambda s: s.update(proposed_slot_choices={}),
    lambda s: s["proposed_slot_choices"][0].pop("description"),
    lambda s: s["proposed_slot_choices"][0].update(target="opaque"),
    lambda s: s["proposed_slot_choices"][0].update(option_id=1),
    lambda s: s["proposed_slot_choices"][0].update(description=""),
    lambda s: s["proposed_slot_choices"].__setitem__(0, None),
    lambda s: s.update(accepted_partial_program=[{"op": "Compare"}]),
    lambda s: s.pop("accepted_partial_program"),
    lambda s: s.update(schema="other"),
    lambda s: s.update(audience="private"),
    lambda s: s.update(records=[{"audience": "private"}]),
    lambda s: s.update(records={}),
    lambda s: s.update(records=[{"audience": "public", "state_handle": "hidden"}]),
    lambda s: s.update(registry=[]),
    lambda s: s.update(registry={}),
    lambda s: s["registry"].update(state_history_context_refs=[]),
    lambda s: s.update(query_constants=[]),
    lambda s: s.update(unmet_obligations={}),
    lambda s: s.update(state_handle="prior-state"),
    lambda s: s.update(state_history_context_refs=[]),
    lambda s: s.update(state_binding={}),
    lambda s: s.update(provider={}),
    lambda s: s.update(evaluator_only={"target": "a"}),
    lambda s: s.update(training_only={}),
    lambda s: s.update(expected="a"),
])
def test_malformed_or_forbidden_state_delegates(change):
    req = request()
    state_edit(req, change)
    assert_delegates(req)


@pytest.mark.parametrize("state", [None, {}, "not json", "[]", "null", "NaN",
                                    '{"schema":"one","schema":"two"}'])
def test_bad_json_state_delegates(state):
    req = request()
    req["state"] = state
    assert_delegates(req)


@pytest.mark.parametrize("qid", ["slot-001-operation", "slot-002-fact", "slot-002-left",
                                 "slot-003-right", "slot-003-relation", "slot-005-relation",
                                 "slot-004-Relation", "slot-001-direct-comparison-relation"])
def test_all_other_slots_delegate(qid):
    req = request()
    req["questions"][qid] = req["questions"].pop(RELATION_QID)
    assert_delegates(req)


@pytest.mark.parametrize("change", [
    lambda r: r["questions"].update(other=deepcopy(r["questions"][RELATION_QID])),
    lambda r: r["questions"][RELATION_QID].update(type="noul"),
    lambda r: r["questions"][RELATION_QID].update(criteria={}),
    lambda r: r["questions"][RELATION_QID].update(criteria={"a": 1}),
    lambda r: r["questions"][RELATION_QID].update(target="a"),
    lambda r: r["questions"][RELATION_QID].pop("instructions"),
    lambda r: r.update(state_handle=None),
    lambda r: r.update(evaluator_only={}),
    lambda r: r.update(ssdm={"state_history_context_refs": []}),
    lambda r: r.update(ssdm={"schema": "wrong"}),
])
def test_question_envelope_and_request_shape_delegates(change):
    req = request()
    change(req)
    assert_delegates(req)


def test_nonobject_requests_and_missing_question_are_not_relations():
    for req in (None, [], "text", {}, {"questions": {}}, {"questions": []}):
        assert not relation_request(req)


@pytest.mark.parametrize("key", ["context_ref", "policy_ref", "registry_ref", "audience"])
@pytest.mark.parametrize("value", ["different", "private", None, {}, []])
def test_envelope_state_binding_mismatch_delegates_unchanged(key, value):
    req = request()
    req["ssdm"] = {"schema": DECISION_SCHEMA, key: value}
    assert_delegates(req)


def test_matching_envelope_bindings_reach_head_and_outer_decider_cannot_repair_mismatch():
    backend, learned, stock = composition()
    req = request()
    state = json.loads(req["state"])
    req["ssdm"] = {"schema": DECISION_SCHEMA,
                   **{k: state[k] for k in ("context_ref", "policy_ref", "registry_ref", "audience")}}
    assert relation_request(req) and backend.route(req) is learned
    assert Decider(backend).decide(req)["ssdm"]["status"] == "ok"
    assert learned.calls and not stock.calls
    learned.calls.clear()
    req["ssdm"]["context_ref"] = "different-context"
    before = deepcopy(req)
    # The outer Decider accepts this envelope; the relation seam must decline it.
    assert Decider(backend).decide(req)["ssdm"]["status"] == "ok"
    assert stock.calls and not learned.calls
    assert req == before


def test_composite_revision_binds_both_complete_revisions_and_contract():
    backend, learned, stock = composition()
    revision = backend.revision()
    assert revision["learned"] == learned.identity and revision["stock"] == stock.identity
    assert revision["route"]["prior_slots"] == ["operation", "left", "right"]
    assert revision["route"]["question_id"] == RELATION_QID
    revision["stock"]["revision_id"] = "external-change"
    revision["route"]["prior_slots"].clear()
    assert backend.revision()["stock"] == stock.identity
    assert backend.revision()["route"]["prior_slots"] == ["operation", "left", "right"]
    for target in ("learned", "stock"):
        other_learned, other_stock = Fake("head"), Fake("stock")
        (other_learned if target == "learned" else other_stock).identity["extra_pin"] = "changed"
        assert RelationOnlyBackend(other_learned, other_stock).revision()["revision_id"] != backend.revision()["revision_id"]


@pytest.mark.parametrize("target", ["learned", "stock"])
def test_revision_drift_refuses_before_backend_invocation(target):
    backend, learned, stock = composition()
    (learned if target == "learned" else stock).identity["revision_id"] += "-changed"
    for invoke in (backend.revision, lambda: backend.route(request()),
                   lambda: backend.plan(request()), lambda: backend.score(request(), lambda: None)):
        with pytest.raises(BackendUnsupported, match="revision change"):
            invoke()
    assert not learned.calls and not stock.calls


def test_mismatched_models_and_native_parent_revisions_refused():
    with pytest.raises(ValueError, match="model does not match"):
        RelationOnlyBackend(Fake("head", {"model_id": "different", "revision_id": "head"}), Fake("stock"))
    stock = Fake("stock", {"model_id": "hf-parent", "revision_id": "stock",
                           "snapshot": "snapshot", "files_sha256": {"weights": "abc", "tokenizer": "def"}})
    parent = {k: deepcopy(stock.identity[k]) for k in ("model_id", "snapshot", "files_sha256")}
    learned_revision = {"model_id": "internal-backbone-spec", "revision_id": "head",
                        "backbone": {"provenance": {"track_l_parent": parent}}}
    assert RelationOnlyBackend(Fake("head", learned_revision), stock).revision()["model_id"] == "hf-parent"
    aliased_parent = {k: v for k, v in parent.items() if k != "snapshot"}
    aliased_parent["snapshot_revision"] = parent["snapshot"]
    aliased_revision = deepcopy(learned_revision)
    aliased_revision["backbone"]["provenance"]["track_l_parent"] = aliased_parent
    assert RelationOnlyBackend(Fake("head", aliased_revision), stock).revision()["model_id"] == "hf-parent"
    conflicting_parent = {**aliased_parent, "snapshot": "other"}
    conflicting_revision = deepcopy(learned_revision)
    conflicting_revision["backbone"]["provenance"]["track_l_parent"] = conflicting_parent
    with pytest.raises(ValueError, match="parent does not match"):
        RelationOnlyBackend(Fake("head", conflicting_revision), stock)
    conflicting_stock = Fake("stock", {**stock.identity, "snapshot_revision": "other"})
    with pytest.raises(ValueError, match="parent does not match"):
        RelationOnlyBackend(Fake("head", conflicting_revision), conflicting_stock)
    missing_stock = Fake("stock", {k: v for k, v in stock.identity.items() if k != "snapshot"})
    missing_parent = deepcopy(learned_revision)
    missing_parent["backbone"]["provenance"]["track_l_parent"].pop("snapshot")
    with pytest.raises(ValueError, match="parent does not match"):
        RelationOnlyBackend(Fake("head", missing_parent), missing_stock)
    for key in ("model_id", "snapshot", "files_sha256"):
        broken = deepcopy(learned_revision)
        broken["backbone"]["provenance"]["track_l_parent"][key] = "wrong"
        with pytest.raises(ValueError, match="parent does not match"):
            RelationOnlyBackend(Fake("head", broken), stock)


def test_outer_decider_rejects_mismatched_revision_without_any_fallback():
    backend, learned, stock = composition()
    req = request()
    req["ssdm"] = {"schema": DECISION_SCHEMA, "parent_revision": "wrong"}
    response = Decider(backend).decide(req)
    assert response["ssdm"]["status"] == "unsupported"
    assert response["ssdm"]["error"]["code"] == "parent_revision_mismatch"
    assert not learned.calls and not stock.calls
    req["ssdm"]["parent_revision"] = backend.revision()["revision_id"]
    assert Decider(backend).decide(req)["ssdm"]["status"] == "ok"
    assert [call[0] for call in learned.calls] == ["plan", "score"] and not stock.calls


def test_import_and_request_recognition_never_import_model_runtime():
    code = '''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported by relation router')
sys.meta_path.insert(0, Reject())
from language_growth.relation_router import RelationOnlyBackend, relation_request
assert not relation_request({})
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)
