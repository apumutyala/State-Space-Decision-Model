"""Version identifiers recorded in every response."""

ADAPTER_VERSION = "0.2.0"
INTERFACE_VERSION = "interfaces/DECISION_LANGUAGE_V0.md@v0.1 (PR #4 787d64f)"
#: Optional fields gated by ``enable_v0_1_deltas``: attribution only (policy restriction is
#: core in 0.1). Request provenance (ssdm.provenance.v0) is a planned 0.2 extension, accepted only
#: with ``enable_experimental_provenance`` (contract 0.1: "D may carry experimental support behind a flag").
DELTAS_VERSION = "attribution (accounting only)"
ENVELOPE_SCHEMA = "ssdm.decide.v0.1"
#: Request fingerprint algorithms. The contract pins ``ssdm.request_fingerprint.v0``:
#: sha256(canonical_json({"state": state, "questions": {qid: {"type", "instructions" or null,
#: "options": canonical_options(q)}}})) with canonical_json = json.dumps(sort_keys=True,
#: separators=(",", ":"), ensure_ascii=False); canonical_options: noul (true, false), choice
#: sorted by id, score by level. Provenance-aware requests (experimental, flag-gated) use a
#: separately versioned id: the same payload plus "provenance": ssdm.provenance as supplied.
FINGERPRINT_ALGORITHM = "ssdm.request_fingerprint.v0"
FINGERPRINT_ALGORITHM_PROVENANCE = "ssdm.request_fingerprint.provenance.experimental-0.2"
WIRE_COMPAT = "systemone/v1 (Jev/SystemOne POST /v1/systemone body; Clef joint_schema_model.systemone semantics)"
