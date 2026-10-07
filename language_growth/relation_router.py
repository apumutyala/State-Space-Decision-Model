"""Protocol-only relation seam. No model loader, tensor code or task interpreter."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from decide.backends.base import BackendUnsupported
from language_growth.parent_bridge import DECISION_SCHEMA, SERIALIZATION_REF


RELATION_QID = "slot-004-relation"
PRIOR_SLOTS = ("operation", "left", "right")
ENVELOPE_STATE_BINDINGS = ("context_ref", "policy_ref", "registry_ref", "audience")
STATE_KEYS = frozenset({
    "schema", "task_input", "audience", "context_ref", "policy_ref", "registry_ref",
    "records", "registry", "query_constants", "unmet_obligations",
    "accepted_partial_program", "proposed_slot_choices",
})
RECORD_KEYS = frozenset({
    "record_ref", "version", "entity_ref", "source_ref", "audience",
    "epistemic_class", "freshness", "conflicts", "fields", "source",
})
ROUTING_CONTRACT = {
    "id": "relation-only-public-exact-prefix-v2",
    "question_id": RELATION_QID, "question_type": "choice", "question_count": 1,
    "serialization": SERIALIZATION_REF, "state_keys": sorted(STATE_KEYS),
    "record_keys": sorted(RECORD_KEYS),
    "envelope_state_bindings_if_present": list(ENVELOPE_STATE_BINDINGS),
    "audience": "public", "accepted_partial_program": [],
    "prior_slots": list(PRIOR_SLOTS),
    "prior_fields": ["description", "option_id", "slot"],
    "provider_state": "forbidden", "fallback": "stock-unchanged",
    "revision_binding": "full-pinned-backends-and-native-parent-provenance-v1",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value)


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate state key")
    return result


def _reject_constant(value):
    raise ValueError("non-JSON state constant")


def _public_record(record):
    # Validate serialized containers, never interpret records or choose a target.
    return (isinstance(record, dict) and set(record) == RECORD_KEYS
            and record["audience"] == "public"
            and all(_text(record[k]) for k in ("record_ref", "entity_ref", "source_ref"))
            and isinstance(record["conflicts"], list)
            and isinstance(record["fields"], dict)
            and isinstance(record["source"], dict)
            and set(record["source"]) <= {"trust", "premise_admission"})


def relation_request(request):
    """Recognize only shape; option IDs/descriptions and task text are opaque.

    The outer Decider validates the decision wire. This narrower seam also works
    when called directly: unknown fields, provider state and malformed prefixes
    fall through without rewriting the caller's request.
    """
    if (not isinstance(request, dict)
            or not {"model", "state", "questions"} <= set(request)
            or not set(request) <= {"model", "state", "questions", "ssdm"}
            or not _text(request["model"])):
        return False
    env = request.get("ssdm", {})
    if (not isinstance(env, dict) or not set(env) <= {
            "schema", "request_id", "parent_revision", "context_ref", "policy_ref",
            "registry_ref", "audience", "limits", "policy_restriction"}
            or ("schema" in env and env["schema"] != DECISION_SCHEMA)):
        return False
    questions = request["questions"]
    if not isinstance(questions, dict) or set(questions) != {RELATION_QID}:
        return False
    question = questions[RELATION_QID]
    if (not isinstance(question, dict) or set(question) != {"type", "instructions", "criteria"}
            or question["type"] != "choice" or not _text(question["instructions"])):
        return False
    criteria = question["criteria"]
    if (not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255
            or any(not _text(k) or not _text(v) for k, v in criteria.items())):
        return False
    try:
        state = json.loads(request["state"], object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (TypeError, ValueError, RecursionError):
        return False
    if (not isinstance(state, dict) or set(state) != STATE_KEYS
            or state["schema"] != SERIALIZATION_REF or state["audience"] != "public"
            or any(not _text(state[k]) for k in ("task_input", "context_ref", "policy_ref", "registry_ref"))
            or state["accepted_partial_program"] != []
            or not isinstance(state["records"], list)
            or not all(_public_record(r) for r in state["records"])
            or not isinstance(state["registry"], dict)
            or not {"registered_fields", "ask_categories"} <= set(state["registry"])
            or not set(state["registry"]) <= {"registered_fields", "comparators", "ask_categories"}
            or any(not isinstance(v, list) or not all(_text(x) for x in v)
                   for v in state["registry"].values())
            or not isinstance(state["query_constants"], dict)
            or not isinstance(state["unmet_obligations"], list)):
        return False
    if any(key in env and env[key] != state[key] for key in ENVELOPE_STATE_BINDINGS):
        return False
    prior = state["proposed_slot_choices"]
    return (isinstance(prior, list) and len(prior) == len(PRIOR_SLOTS)
            and all(isinstance(choice, dict)
                    and set(choice) == {"slot", "option_id", "description"}
                    and choice["slot"] == slot
                    and _text(choice["option_id"]) and _text(choice["description"])
                    for choice, slot in zip(prior, PRIOR_SLOTS)))


def _revision(backend):
    revision = deepcopy(backend.revision())
    if (not isinstance(revision, dict) or not _text(revision.get("model_id"))
            or not _text(revision.get("revision_id"))):
        raise ValueError("backend must identify model and revision")
    _digest(revision)  # require a finite, serializable identity
    return revision


def _snapshot_identity(identity):
    """Normalize the parent bridge's accepted snapshot aliases, rejecting ambiguity."""
    if not isinstance(identity, dict):
        return None
    snapshot = identity.get("snapshot", identity.get("snapshot_revision"))
    if ("snapshot" in identity and "snapshot_revision" in identity
            and identity["snapshot"] != identity["snapshot_revision"]):
        return None
    return snapshot if _text(snapshot) else None


class RelationOnlyBackend:
    """One outer Decider serves a pinned head/stock composition, without edits."""
    name = "relation_only_stock_composition"

    def __init__(self, learned, stock):
        self.learned, self.stock = learned, stock
        learned_revision, stock_revision = _revision(learned), _revision(stock)
        native_parent = (learned_revision.get("backbone", {}).get("provenance", {})
                         .get("track_l_parent"))
        if native_parent is not None:
            expected = {"model_id": stock_revision.get("model_id"),
                        "snapshot": _snapshot_identity(stock_revision),
                        "files_sha256": stock_revision.get("files_sha256")}
            actual = {"model_id": native_parent.get("model_id") if isinstance(native_parent, dict) else None,
                      "snapshot": _snapshot_identity(native_parent),
                      "files_sha256": native_parent.get("files_sha256") if isinstance(native_parent, dict) else None}
            if any(actual[k] != v for k, v in expected.items()):
                raise ValueError("relation head parent does not match stock revision")
        elif learned_revision["model_id"] != stock_revision["model_id"]:
            raise ValueError("relation head model does not match stock revision")
        self._identity = {"backend": self.name, "route": deepcopy(ROUTING_CONTRACT),
                          "learned": learned_revision, "stock": stock_revision}
        self._revision = {**self._identity, "model_id": stock_revision["model_id"],
                          "revision_id": self.name + "@" + _digest(self._identity)}

    def _assert_pinned(self):
        if (_revision(self.learned) != self._identity["learned"]
                or _revision(self.stock) != self._identity["stock"]):
            raise BackendUnsupported("relation_revision_drift", "rebuild the relation composition after a backend revision change")

    def revision(self):
        self._assert_pinned()
        return deepcopy(self._revision)

    def route(self, request):
        self._assert_pinned()
        return self.learned if relation_request(request) else self.stock

    def plan(self, request):
        return self.route(request).plan(request)

    def score(self, request, check_deadline):
        return self.route(request).score(request, check_deadline)
