"""Version identifiers recorded in every response."""

ADAPTER_VERSION = "0.2.0"
INTERFACE_VERSION = "interfaces/DECISION_LANGUAGE_V0.md@v0.1 (PR #4 8109b13)"
#: Optional fields gated by ``enable_v0_1_deltas``: attribution only (policy restriction is
#: core in 0.1). Request provenance (ssdm.provenance.v0) is accepted always (proposed on PR #4).
DELTAS_VERSION = "attribution (accounting only)"
ENVELOPE_SCHEMA = "ssdm.decide.v0.1"
#: Request fingerprint algorithm (pinned for the L bridge):
#: sha256(canonical_json({"state": state, "questions": {qid: {"type", "instructions",
#: "options": canonical_options(q)}}} [+ "provenance": ssdm.provenance as supplied, only if present]))
#: canonical_json = json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False);
#: canonical_options: noul (true, false), choice sorted by id, score by level.
#: Identical to ssdm.request_fingerprint.v0 when no provenance is supplied.
FINGERPRINT_ALGORITHM = "ssdm.request_fingerprint.v0.1"
WIRE_COMPAT = "systemone/v1 (Jev/SystemOne POST /v1/systemone body; Clef joint_schema_model.systemone semantics)"
