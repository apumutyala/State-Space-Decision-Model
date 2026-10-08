"""Machine-pinned prepared-suite execution; no human review/approval prerequisite.

The evidence manifest binds exact artifacts, selected comparator and split
inventories. It cannot replace measured outputs or turn authored labels into an
independent audit. Nothing here generates heldout data during evaluation.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from .gate_suites import LEVELS, SUITE_SCHEMA, digest, input_content_digest


def read_pinned(path: Path, sha256: str) -> dict:
    from .gates import sha_file, _strict_json
    if not isinstance(sha256, str) or len(sha256) != 64 or sha_file(path) != sha256:
        raise ValueError("review/evidence artifact hash mismatch")
    value = _strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("review/evidence must be a JSON object")
    return value


def load_evidence_manifest(path: Path, sha256: str, *, parent_sha256: str, levels: list[str]) -> dict:
    review = read_pinned(path, sha256)
    if (review.get("schema") != "ssdm.language_gate_evidence.v1"
            or review.get("candidate_checkpoint_sha256") != parent_sha256
            or review.get("primary_metric") != "useful_complete_answer"):
        raise ValueError("machine-pinned language evidence manifest required")
    if set(review.get("levels", {})) != set(levels):
        raise ValueError("review must bind the exact requested level set")
    for level in levels:
        if level not in LEVELS:
            raise ValueError("unknown language level")
        entry = review["levels"][level]
        required = {"manifest_sha256",
                    "decision_authorization_sha256", "comparator_artifact_sha256",
                    "comparator_identity_sha256",
                    "intervention_registration_sha256", "intervention_adapter_source_sha256"}
        if not isinstance(entry, dict) or set(entry) - required - {"comparator_model_id", "comparator_revision", "comparator_source", "comparator_min_accuracy"} or not required <= set(entry):
            raise ValueError("incomplete level evidence pins")
        if any(not isinstance(entry[key], str) or len(entry[key]) != 64 or any(c not in "0123456789abcdef" for c in entry[key]) for key in required):
            raise ValueError("every reviewed evidence binding must be a SHA-256")
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in ("comparator_model_id", "comparator_revision", "comparator_source")):
            raise ValueError("selected open-weight comparator model/revision/source required")
        if "comparator_min_accuracy" in entry and (type(entry["comparator_min_accuracy"]) not in (int,float) or not 0 <= entry["comparator_min_accuracy"] <= 1):
            raise ValueError("invalid optional preselected comparator objective")
    return review


# Backward-compatible function name only; the old human-approval schema is not accepted.
load_review = load_evidence_manifest


def prepare_evidence_manifest(parent: Path, *, levels: list[str], suite_dir: Path,
                              comparator_artifact: Path, decision_authorization: Path,
                              intervention_adapter: str, intervention_registration: Path,
                              comparator_min_accuracy: float | None = None) -> dict:
    """Capture exact existing artifact/selection pins; no model or approval step."""
    from .gates import sha_file, _strict_json, _callable
    function = _callable(intervention_adapter)
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError("intervention adapter must have a source file")
    evidence = {"schema":"ssdm.language_gate_evidence.v1", "candidate_checkpoint_sha256":sha_file(parent),
                "primary_metric":"useful_complete_answer", "levels":{}}
    for level in levels:
        def location(path): return Path(str(path).replace("{level}",level))
        comparator = _strict_json(location(comparator_artifact).read_text(encoding="utf-8"))
        identity = comparator["identity"]
        entry = {"manifest_sha256":sha_file(suite_dir/level/"manifest.json"),
                 "decision_authorization_sha256":sha_file(location(decision_authorization)),
                 "comparator_artifact_sha256":sha_file(location(comparator_artifact)),
                 "comparator_identity_sha256":digest(identity),
                 "comparator_model_id":identity["model_id"], "comparator_revision":identity["revision"],
                 "comparator_source":identity["open_weight_source"],
                 "intervention_registration_sha256":sha_file(location(intervention_registration)),
                 "intervention_adapter_source_sha256":sha_file(Path(source))}
        if comparator_min_accuracy is not None:
            if not 0 <= comparator_min_accuracy <= 1:
                raise ValueError("invalid optional comparator objective")
            entry["comparator_min_accuracy"] = comparator_min_accuracy
        evidence["levels"][level] = entry
    return evidence


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
    inventories = manifest.get("development_inventories")
    if not isinstance(inventories, dict) or set(inventories) != {"development_train", "development_validation"}:
        raise ValueError("development split inventories required")
    actual_content = {input_content_digest(task) for task in tasks}
    actual_worlds = {task["world_id"] for task in tasks}
    for inventory in inventories.values():
        if (not isinstance(inventory, dict) or set(inventory) != {"world_ids", "input_content_sha256"}
                or not isinstance(inventory["world_ids"], list) or not isinstance(inventory["input_content_sha256"], list)):
            raise ValueError("invalid split inventory")
        if not inventory["world_ids"] or not inventory["input_content_sha256"]:
            raise ValueError("nonempty development inventory required")
        if actual_content & set(inventory["input_content_sha256"]):
            raise ValueError("heldout and development input content overlap")
        if level != "L4" and actual_worlds & set(inventory["world_ids"]):
            raise ValueError("heldout and development evidence worlds overlap")
    return manifest, tasks, refs


def run_registered_interventions(tasks: list[dict], refs: list[dict], parent, *,
                                 adapter: str, adapter_source_sha256: str,
                                 registration: Path, registration_sha256: str,
                                 draws: int) -> dict:
    """Execute a source-pinned adapter, then rescore all raw four-arm outputs.

    The source-pinned adapter owns native head-preserving capture/edit/replay and
    fitted controls. Its source and experiment artifact are pinned separately.
    Output booleans or a previously calculated pass verdict are never inputs.
    """
    from .gates import _callable, sha_file, intervention_comparison
    prereg = read_pinned(registration, registration_sha256)
    registration_body = {key: value for key, value in prereg.items() if key != "registration_sha256"}
    scoped = prereg.get("schema") == "ssdm.language_intervention_preregistration.v2"
    if (prereg.get("schema") not in {"ssdm.language_intervention_preregistration.v1",
                                     "ssdm.language_intervention_preregistration.v2"}
            or prereg.get("parent_revision") != parent.planner_kwargs()["parent_revision"]
            or prereg.get("registration_sha256") != digest(registration_body)
            or prereg.get("capture_boundary") != "native_fork_prehead.v1"
            or prereg.get("scope") != "final_hidden_semantic_axis_only"
            or prereg.get("output_control") != "fitted_candidate_count_logit_direction.v1"
            or not isinstance(prereg.get("discovery_world_ids"), list)
            or not prereg["discovery_world_ids"]):
        raise ValueError("registered parent and discovery split required for intervention")
    target_slots = None
    if scoped:
        from .gate_interventions import validate_target_slots, capture_is_targeted
        target_slots = validate_target_slots(prereg.get("target_slots"))
    if {task["world_id"] for task in tasks} & set(prereg["discovery_world_ids"]):
        raise ValueError("intervention discovery and evaluation worlds overlap")
    function = _callable(adapter)
    source_path = inspect.getsourcefile(function)
    if source_path is None or sha_file(Path(source_path)) != adapter_source_sha256:
        raise ValueError("intervention adapter source identity mismatch")
    result = function(tasks, parent, prereg)
    result_keys = {"arms", "provenance", "instrumentation_valid"}
    if scoped:
        result_keys |= {"targeted_capture_counts", "targeted_cohort_valid"}
    if not isinstance(result, dict) or set(result) != result_keys:
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
    valid = result["instrumentation_valid"] is True
    targeted_counts = {arm: 0 for arm in ("semantic", "sham", "random", "output_only")}
    for record in provenance:
        valid = valid and record.get("instrumentation_valid") is True
        captures = record.get("captures", {})
        valid = valid and set(captures) == {"semantic", "sham", "random", "output_only"}
        valid = valid and all(isinstance(rows, list) and rows and all(
            capture.get("status") == "ok" and capture.get("head_calls") == 1 and capture.get("capture") is not None
            for capture in rows) for rows in captures.values())
        if scoped:
            valid = valid and record.get("target_slots") == target_slots
            for arm, calls in captures.items():
                if arm not in targeted_counts or not isinstance(calls, list):
                    valid = False
                    continue
                for call in calls:
                    try:
                        targeted = capture_is_targeted(call.get("question_id"), target_slots)
                    except ValueError:
                        valid = False
                        continue
                    valid = valid and call.get("targeted") is targeted
                    if targeted:
                        targeted_counts[arm] += 1
                    else:
                        capture = call.get("capture")
                        valid = valid and isinstance(capture, dict)
                        if isinstance(capture, dict):
                            valid = valid and all(capture.get(key) == 0 for key in (
                                "hidden_delta_l2", "matched_delta_l2", "output_delta_l2"))
                            valid = valid and all(capture.get(before) is not None and
                                capture.get(before) == capture.get(after) for before, after in (
                                    ("hidden_before_sha256", "hidden_after_sha256"),
                                    ("logits_before_sha256", "logits_after_sha256")))
    if scoped:
        valid = valid and result["targeted_cohort_valid"] is True
        valid = valid and all(count > 0 for count in targeted_counts.values())
        reported = result["targeted_capture_counts"]
        valid = valid and isinstance(reported, dict) and reported == targeted_counts
        if isinstance(reported, dict):
            valid = valid and all(type(value) is int for value in reported.values())
    valid = valid and all(row.status != "runner_error" and row.reason != "native_intervention_failed"
                          for rows in result["arms"].values() for row in rows)
    scored["status"] = "registered_neural_measured"
    scored["instrumentation_valid"] = bool(valid)
    scored["passed"] = scored["criterion_observed"] and valid
    if not valid:
        scored["status"] = "unmeasured_invalid_instrumentation"
        scored["criterion_observed"] = False
    scored["reason"] = "raw four-arm outputs rescored under source-pinned native intervention protocol"
    scored["registration_sha256"] = registration_sha256
    scored["adapter_source_sha256"] = adapter_source_sha256
    scored["provenance"] = provenance
    if scoped:
        scored["target_slots"] = target_slots
        scored["targeted_capture_counts"] = targeted_counts
    return scored


def evaluate_prepared_level(directory: Path, level: str, parent, *, review_entry: dict,
                            comparator_artifact: Path,
                            decision_authorization: Path, decision_parent: Path, decision_candidate: Path,
                            decision_scoring_rows: Path,
                            intervention_adapter: str, intervention_registration: Path,
                            candidate_sha256: str, draws: int = 2000, evidence_out: Path | None = None) -> dict:
    from .gates import run_tasks, score_comparator, compare_usefulness, _callable, sha_file
    from .gate_decision_retention import authorized_decision_identities, decision_identity, decision_index_retention
    manifest, tasks, refs = load_prepared_suite(directory, level, manifest_sha256=review_entry["manifest_sha256"])
    comparator = read_pinned(comparator_artifact, review_entry["comparator_artifact_sha256"])
    if comparator.get("schema") != "ssdm.language_comparator.v1" or digest(comparator["identity"]) != review_entry["comparator_identity_sha256"]:
        raise ValueError("reviewed strong comparator identity mismatch")
    identity = comparator["identity"]
    if (identity.get("model_id") != review_entry["comparator_model_id"]
            or identity.get("revision") != review_entry["comparator_revision"]
            or identity.get("open_weight_source") != review_entry["comparator_source"]
            or identity.get("unrestricted_generation") is not True
            or identity.get("constrained_decoding") is not False):
        raise ValueError("selected open-weight unrestricted comparator identity mismatch")
    useful = [i for i, ref in enumerate(refs) if ref["stratum"] == "usefulness"]
    joint = [i for i, ref in enumerate(refs) if ref["stratum"] == "joint_adversarial"]
    useful_tasks, useful_refs = [tasks[i] for i in useful], [refs[i] for i in useful]
    expected_ids = {task["task_id"] for task in tasks}
    if any(row["task_id"] not in expected_ids for row in comparator["observations"]):
        raise ValueError("comparator artifact includes tasks outside reviewed suite")
    predictions = [row for row in comparator["observations"] if row["task_id"] in {task["task_id"] for task in useful_tasks}]
    if (len(predictions) != len(useful_tasks)
            or {row["task_id"] for row in predictions} != {task["task_id"] for task in useful_tasks}
            or any(row.get("status") != "ok" for row in predictions)):
        raise ValueError("complete operational comparator responses required; missing/transport failures cannot qualify")
    comparator_rows = score_comparator(useful_tasks, useful_refs, predictions)
    # Cheap file/source checks happen before baseline inference. The adapter's
    # runtime capture validation still runs later; baseline rows are persisted
    # before that potentially failing operation.
    read_pinned(intervention_registration, review_entry["intervention_registration_sha256"])
    adapter_source = inspect.getsourcefile(_callable(intervention_adapter))
    if adapter_source is None or sha_file(Path(adapter_source)) != review_entry["intervention_adapter_source_sha256"]:
        raise ValueError("intervention adapter source identity mismatch")
    # All metadata/artifact checks precede any parent inference or labels handed
    # to the planner. Only input tasks enter run_tasks/adapter model calls.
    qualified_parent_identity, authorized_candidate_identity = authorized_decision_identities(
        decision_authorization, review_entry["decision_authorization_sha256"])
    candidate_identity = parent.identity()
    if (decision_identity(candidate_identity) != authorized_candidate_identity
            or authorized_candidate_identity["artifact_sha256"] != candidate_sha256):
        raise ValueError("loaded language candidate differs from the authorized child identity")
    if qualified_parent_identity == authorized_candidate_identity:
        raise ValueError("parent-retention authorization must name distinct baseline and language-child identities")
    regression = decision_index_retention(decision_authorization, review_entry["decision_authorization_sha256"],
                    decision_scoring_rows, decision_parent, decision_candidate,
                    qualified_parent_identity=qualified_parent_identity,
                    candidate_identity=candidate_identity, draws=draws)
    rows = run_tasks(tasks, refs, parent, allow_heldout=True)
    if evidence_out is not None:
        from .gate_suites import canonical
        evidence_out.mkdir(parents=True, exist_ok=False)
        (evidence_out / "candidate.json").write_bytes(canonical({"schema":"ssdm.language_candidate_outputs.v1",
                        "level":level,"manifest_sha256":review_entry["manifest_sha256"],"observations":rows})+b"\n")
    usefulness = compare_usefulness([rows[i] for i in useful], comparator_rows, draws=draws)
    objective_met = usefulness["comparator_accuracy"] >= review_entry.get("comparator_min_accuracy", 0)
    usefulness.update(status="prepared_heldout_measured", passed=usefulness["criterion_observed"] and objective_met,
                      reason="heldout paired comparison against the selected strong open-weight comparator",
                      comparator_identity=identity, optional_comparator_objective_met=objective_met)
    joint_rows = [rows[i] for i in joint]
    joint_gate = {"status": "prepared_heldout_measured", "passed": bool(joint_rows) and all(row["correct"] and not row["unsupported_release"] for row in joint_rows),
                  "n_items": len(joint_rows), "observations": joint_rows}
    try:
        interventions = run_registered_interventions(useful_tasks, useful_refs, parent,
                        adapter=intervention_adapter, adapter_source_sha256=review_entry["intervention_adapter_source_sha256"],
                        registration=intervention_registration, registration_sha256=review_entry["intervention_registration_sha256"], draws=draws)
    except Exception as exc:
        interventions = {"status":"unmeasured", "passed":False,
                         "reason":"registered intervention execution failed", "error_type":type(exc).__name__}
    gates = {"usefulness": usefulness, "parent_decision": regression,
             "joint_adversarial": joint_gate, "semantic_intervention": interventions}
    return {"scope": manifest["scope"], "passed": all(gate["passed"] for gate in gates.values()),
            "gates": gates, "observations": rows, "comparator_observations": comparator_rows,
            "manifest_sha256": review_entry["manifest_sha256"],
            "label_provenance":manifest.get("label_provenance", "unspecified"),
            "limitation": "bounded reviewed task domain and native final-hidden semantic edits; no global no-hallucination or full neural interpretation claim"}
