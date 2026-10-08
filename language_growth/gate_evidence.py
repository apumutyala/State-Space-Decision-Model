"""Pinned, reviewed prepared-suite execution. Nothing here generates heldout data.

The review is a separately hash-pinned host attestation, not a model-authored
certificate or reviewer authentication. Raw outputs are always rescored here;
the attestation authorizes datasets/comparator strength/intervention protocol,
and cannot replace any measured gate. This path is never used by dev commands.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from .gate_suites import LEVELS, SUITE_SCHEMA, digest


def read_pinned(path: Path, sha256: str) -> dict:
    from .gates import sha_file, _strict_json
    if not isinstance(sha256, str) or len(sha256) != 64 or sha_file(path) != sha256:
        raise ValueError("review/evidence artifact hash mismatch")
    value = _strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("review/evidence must be a JSON object")
    return value


def load_review(path: Path, sha256: str, *, parent_sha256: str, levels: list[str]) -> dict:
    review = read_pinned(path, sha256)
    if (review.get("schema") != "ssdm.language_gate_review.v1"
            or review.get("candidate_checkpoint_sha256") != parent_sha256
            or review.get("independent_language_holdout") is not True
            or review.get("labels_independently_reviewed") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"]
            or not isinstance(review.get("review_id"), str) or not review["review_id"]):
        raise ValueError("independently reviewed language holdout authorization required")
    if set(review.get("levels", {})) != set(levels):
        raise ValueError("review must bind the exact requested level set")
    for level in levels:
        if level not in LEVELS:
            raise ValueError("unknown language level")
        entry = review["levels"][level]
        required = {"manifest_sha256", "label_audit_sha256", "joint_fixture_audit_sha256",
                    "decision_authorization_sha256", "comparator_artifact_sha256",
                    "comparator_identity_sha256", "comparator_strength_evidence_sha256",
                    "intervention_registration_sha256", "intervention_adapter_source_sha256"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("incomplete reviewed level evidence pins")
        if any(not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin) for pin in entry.values()):
            raise ValueError("every reviewed evidence binding must be a SHA-256")
    return review


def load_prepared_suite(directory: Path, level: str, *, manifest_sha256: str) -> tuple[dict, list[dict], list[dict]]:
    """Verify review pin and both file hashes before reading evaluator labels."""
    from .gates import sha_file, _strict_json, _validate_cohort
    directory = Path(directory)
    manifest = read_pinned(directory / "manifest.json", manifest_sha256)
    if manifest.get("schema") != SUITE_SCHEMA or manifest.get("level") != level or manifest.get("split") != "language_heldout":
        raise ValueError("final evaluation requires an independently reviewed language heldout suite")
    if (sha_file(directory / "inputs.jsonl") != manifest["inputs_sha256"]
            or sha_file(directory / "evaluator.jsonl") != manifest["evaluator_sha256"]):
        raise ValueError("prepared input/evaluator hash mismatch")
    tasks = [_strict_json(line) for line in (directory / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    refs = [_strict_json(line) for line in (directory / "evaluator.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _validate_cohort(tasks, refs)
    if (len(tasks) != manifest["tasks"] or len({t["world_id"] for t in tasks}) != manifest["worlds"]
            or any(t["split"] != "language_heldout" or t["level"] != level for t in tasks)):
        raise ValueError("prepared suite topology mismatch")
    # Independent review may audit generated labels, but training eligibility
    # must remain false and joint adversarial labels must be explicit.
    if manifest.get("training_allowed") is not False:
        raise ValueError("language heldout must never be training-eligible")
    if set(ref.get("stratum") for ref in refs) != {"usefulness", "joint_adversarial"}:
        raise ValueError("reviewed usefulness and joint/adversarial strata both required")
    return manifest, tasks, refs


def run_registered_interventions(tasks: list[dict], refs: list[dict], parent, *,
                                 adapter: str, adapter_source_sha256: str,
                                 registration: Path, registration_sha256: str,
                                 draws: int) -> dict:
    """Execute a source-pinned adapter, then rescore all raw four-arm outputs.

    The reviewed adapter owns native head-preserving capture/edit/replay and
    fitted controls. Its source and experiment artifact are pinned separately.
    Output booleans or a previously calculated pass verdict are never inputs.
    """
    from .gates import _callable, sha_file, intervention_comparison
    prereg = read_pinned(registration, registration_sha256)
    registration_body = {key: value for key, value in prereg.items() if key != "registration_sha256"}
    if (prereg.get("schema") != "ssdm.language_intervention_preregistration.v1"
            or prereg.get("parent_revision") != parent.planner_kwargs()["parent_revision"]
            or prereg.get("registration_sha256") != digest(registration_body)
            or prereg.get("capture_boundary") != "native_fork_prehead.v1"
            or prereg.get("scope") != "final_hidden_semantic_axis_only"
            or prereg.get("output_control") != "fitted_candidate_count_logit_direction.v1"
            or not isinstance(prereg.get("discovery_world_ids"), list)
            or not prereg["discovery_world_ids"]):
        raise ValueError("registered parent and discovery split required for intervention")
    if {task["world_id"] for task in tasks} & set(prereg["discovery_world_ids"]):
        raise ValueError("intervention discovery and evaluation worlds overlap")
    function = _callable(adapter)
    source_path = inspect.getsourcefile(function)
    if source_path is None or sha_file(Path(source_path)) != adapter_source_sha256:
        raise ValueError("intervention adapter source identity mismatch")
    result = function(tasks, parent, prereg)
    if not isinstance(result, dict) or set(result) != {"arms", "provenance"}:
        raise ValueError("intervention adapter must return raw arms and provenance")
    provenance = result["provenance"]
    if not isinstance(provenance, list) or len(provenance) != len(tasks):
        raise ValueError("missing intervention capture provenance")
    for task, record in zip(tasks, provenance):
        if (record.get("task_id") != task["task_id"] or record.get("input_sha256") != digest(task)
                or record.get("parent_revision") != prereg["parent_revision"]
                or record.get("registration_sha256") != prereg.get("registration_sha256")
                or record.get("capture_identity_sha256") != prereg.get("capture_identity_sha256")
                or set(record.get("arms", [])) != {"semantic", "sham", "random", "output_only"}
                or record.get("state_unchanged_sha256_before") != record.get("state_unchanged_sha256_after")):
            raise ValueError("intervention capture/control/parent identity mismatch")
        for key in ("capture_identity_sha256", "state_unchanged_sha256_before", "state_unchanged_sha256_after"):
            pin = record.get(key)
            if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
                raise ValueError("invalid intervention provenance hash")
    scored = intervention_comparison(tasks, refs, result["arms"], draws=draws)
    scored["status"] = "registered_neural_measured"
    scored["passed"] = scored["criterion_observed"]
    scored["reason"] = "raw four-arm outputs rescored under source-pinned reviewed native intervention protocol"
    scored["registration_sha256"] = registration_sha256
    scored["adapter_source_sha256"] = adapter_source_sha256
    scored["provenance"] = provenance
    return scored


def evaluate_prepared_level(directory: Path, level: str, parent, *, review_entry: dict,
                            comparator_artifact: Path, comparator_strength: Path,
                            decision_authorization: Path, decision_parent: Path, decision_candidate: Path,
                            decision_scoring_rows: Path,
                            intervention_adapter: str, intervention_registration: Path,
                            candidate_sha256: str, draws: int = 2000) -> dict:
    from .gates import run_tasks, score_comparator, compare_usefulness
    from .gate_decision_retention import decision_index_retention
    manifest, tasks, refs = load_prepared_suite(directory, level, manifest_sha256=review_entry["manifest_sha256"])
    comparator = read_pinned(comparator_artifact, review_entry["comparator_artifact_sha256"])
    if comparator.get("schema") != "ssdm.language_comparator.v1" or digest(comparator["identity"]) != review_entry["comparator_identity_sha256"]:
        raise ValueError("reviewed strong comparator identity mismatch")
    strength = read_pinned(comparator_strength, review_entry["comparator_strength_evidence_sha256"])
    if (strength.get("schema") != "ssdm.language_comparator_strength.v1"
            or strength.get("identity_sha256") != review_entry["comparator_identity_sha256"]
            or strength.get("open_weight") is not True or strength.get("unrestricted_generation") is not True
            or strength.get("strong_same_task_comparator") is not True
            or not isinstance(strength.get("evidence_sha256"), str) or len(strength["evidence_sha256"]) != 64):
        raise ValueError("reviewed same-task strong open-weight comparator evidence required")
    useful = [i for i, ref in enumerate(refs) if ref["stratum"] == "usefulness"]
    joint = [i for i, ref in enumerate(refs) if ref["stratum"] == "joint_adversarial"]
    useful_tasks, useful_refs = [tasks[i] for i in useful], [refs[i] for i in useful]
    expected_ids = {task["task_id"] for task in tasks}
    if any(row["task_id"] not in expected_ids for row in comparator["observations"]):
        raise ValueError("comparator artifact includes tasks outside reviewed suite")
    predictions = [row for row in comparator["observations"] if row["task_id"] in {task["task_id"] for task in useful_tasks}]
    # All metadata/artifact checks precede any parent inference or labels handed
    # to the planner. Only input tasks enter run_tasks/adapter model calls.
    regression = decision_index_retention(decision_authorization, review_entry["decision_authorization_sha256"],
                    decision_scoring_rows, decision_parent, decision_candidate,
                    qualified_parent_identity=parent.identity(), candidate_identity=parent.identity(), draws=draws)
    rows = run_tasks(tasks, refs, parent, allow_heldout=True)
    comparator_rows = score_comparator(useful_tasks, useful_refs, predictions)
    usefulness = compare_usefulness([rows[i] for i in useful], comparator_rows, draws=draws)
    usefulness.update(status="reviewed_heldout_measured", passed=usefulness["criterion_observed"],
                      reason="independently reviewed heldout with pinned same-task strong open-weight comparator")
    joint_rows = [rows[i] for i in joint]
    joint_gate = {"status": "reviewed_heldout_measured", "passed": bool(joint_rows) and all(row["correct"] and not row["unsupported_release"] for row in joint_rows),
                  "n_items": len(joint_rows), "observations": joint_rows}
    interventions = run_registered_interventions(useful_tasks, useful_refs, parent,
                    adapter=intervention_adapter, adapter_source_sha256=review_entry["intervention_adapter_source_sha256"],
                    registration=intervention_registration, registration_sha256=review_entry["intervention_registration_sha256"], draws=draws)
    gates = {"usefulness": usefulness, "parent_decision": regression,
             "joint_adversarial": joint_gate, "semantic_intervention": interventions}
    return {"scope": manifest["scope"], "passed": all(gate["passed"] for gate in gates.values()),
            "gates": gates, "observations": rows, "comparator_observations": comparator_rows,
            "manifest_sha256": review_entry["manifest_sha256"],
            "limitation": "bounded reviewed task domain and native final-hidden semantic edits; no global no-hallucination or full neural interpretation claim"}
