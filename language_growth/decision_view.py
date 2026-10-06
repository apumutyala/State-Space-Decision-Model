"""Build a same-evidence direct decision arm without exposing authored labels."""

from __future__ import annotations

from typing import Any

from .parent_bridge import BridgeError, decision_request, legacy_request_fingerprint


def direct_decision_request(case: dict[str, Any], catalog: dict[str, Any], *,
                            model: str, parent_revision: str,
                            relabel: bool = False) -> dict[str, Any]:
    """Return only model-facing wire data and transport identities, never gold.

    Relabelling changes IDs while preserving descriptions and evidence. Scoring
    that control against its transformed target is the evaluator's separate job.
    This builder neither calls a provider nor scores task correctness.
    """
    host, direct = case["host"], case["decision_only"]
    if (direct["question"] != case["question"] or direct["type"] != "choice"
            or direct["evidence_context_ref"] != host["context_ref"]
            or direct["policy_ref"] != host["policy_ref"]):
        raise BridgeError("decision_view_mismatch")
    entries = direct["options"]
    if not isinstance(entries, list) or not entries:
        raise BridgeError("invalid_options")
    options = {}
    for item in entries:
        if (not isinstance(item, dict) or set(item) != {"id", "description"}
                or not isinstance(item["id"], str) or item["id"] in options
                or not isinstance(item["description"], str)):
            raise BridgeError("invalid_options")
        options[item["id"]] = item["description"]
    if relabel:
        mapping = direct["relabel_control"]["id_map"]
        if (not isinstance(mapping, dict) or set(mapping) != set(options)
                or any(not isinstance(value, str) for value in mapping.values())
                or set(mapping.values()) != set(options)):
            raise BridgeError("invalid_relabel")
        options = {mapping[key]: value for key, value in options.items()}
    request, state_digest = decision_request(
        task_input=case["question"], host=host, catalog=catalog,
        model=model, parent_revision=parent_revision,
        request_id=direct["question_id"] + ("-relabel" if relabel else "-direct"),
        question_id=direct["question_id"], question=case["question"], options=options,
    )
    return {"request": request, "state_sha256": state_digest,
            "request_fingerprint": legacy_request_fingerprint(request)}
