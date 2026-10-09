"""Fail-closed data and parent admission for native direction fitting.

This small boundary is kept independent of model runtimes so pair files and
CLI routing can be checked without importing Torch or reconstructing a checkpoint.
"""
from __future__ import annotations

import re

_PAIR_KEYS = {"pair_id", "world_id", "split", "concept", "source", "effect", "source_target", "effect_target"}
_SPEC_KEYS = {"task_input", "host", "catalog", "question_id", "question", "options"}


def preflight_direction_pairs(discovery, validation, *, target_slots=None, known_target_slots=None) -> dict:
    """Validate pair files and their actual D requests without loading a model."""
    from .gate_suites import digest
    from .parent_bridge import decision_request
    from decide.schema import canonical_options

    if not isinstance(discovery, list) or not discovery or not isinstance(validation, list) or not validation:
        raise ValueError("nonempty separate discovery and validation pair lists required")
    if (target_slots is not None and
            (not isinstance(target_slots, list) or not target_slots
             or any(not isinstance(slot, str) or not slot for slot in target_slots)
             or len(target_slots) != len(set(target_slots)))):
        raise ValueError("unique explicit target slots required for scoped direction fitting")
    if target_slots is not None and (
            known_target_slots is None or any(slot not in known_target_slots for slot in target_slots)):
        raise ValueError("preflight target slots must be registered GatePlanner slots")

    def cohort(pairs, split):
        pair_ids, worlds, content, concepts, counts, slots = set(), set(), set(), set(), set(), set()
        for pair in pairs:
            if (not isinstance(pair, dict) or set(pair) != _PAIR_KEYS or pair["split"] != split
                    or any(not isinstance(pair[key], str) or not pair[key]
                           for key in ("pair_id", "world_id", "concept", "source_target", "effect_target"))):
                raise ValueError("only explicitly tagged development direction pairs are accepted")
            requests = {}
            for side in ("source", "effect"):
                spec = pair[side]
                if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
                    raise ValueError("label-free direction request specification required")
                request, _ = decision_request(
                    **spec, model="direction-pair-preflight", parent_revision="direction-pair-preflight",
                    request_id=digest({"pair": pair["pair_id"], "side": side}), proposed_slot_choices=[])
                requests[side] = request
            if any(pair["source"][key] != pair["effect"][key]
                   for key in ("question_id", "question", "options")):
                raise ValueError("matched source/effect question and canonical candidate domain required")
            qid = pair["source"]["question_id"]
            match = re.fullmatch(r"slot-([0-9]{3})-([a-z][a-z-]*)", qid) if isinstance(qid, str) else None
            slot = match.group(2) if match and int(match.group(1)) > 0 else None
            if slot is not None and known_target_slots is not None and slot not in known_target_slots:
                slot = None
            slots.add(slot)
            options = [oid for oid, _ in canonical_options(requests["source"]["questions"][qid])]
            if (pair["source_target"] not in options or pair["effect_target"] not in options
                    or pair["source_target"] == pair["effect_target"]):
                raise ValueError("real source/effect target contrast required within the shared option domain")
            visible = {side: digest({"state": request["state"], "questions": request["questions"]})
                       for side, request in requests.items()}
            if visible["source"] == visible["effect"]:
                raise ValueError("source and effect must change model-visible request content")
            pair_ids.add(pair["pair_id"])
            worlds.add(pair["world_id"])
            content.update(visible.values())
            concepts.add(pair["concept"])
            counts.add(len(options))
        if len(pair_ids) != len(pairs) or len(content) != 2 * len(pairs) or len(concepts) != 1:
            raise ValueError("duplicate pair IDs/content or multiple concepts in a direction cohort")
        return {"pairs": len(pairs), "pair_ids": pair_ids, "worlds": worlds,
                "content": content, "concepts": concepts, "option_counts": counts, "slots": slots}

    train = cohort(discovery, "development_train")
    valid = cohort(validation, "development_validation")
    if (train["pair_ids"] & valid["pair_ids"] or train["worlds"] & valid["worlds"]
            or train["content"] & valid["content"]):
        raise ValueError("discovery and validation worlds/IDs/model-visible content must be disjoint")
    if train["concepts"] != valid["concepts"]:
        raise ValueError("validation must use the preregistered discovery concept")
    if valid["option_counts"] - train["option_counts"]:
        raise ValueError("validation option counts require discovery-fitted controls")
    if target_slots is not None:
        expected_slots = set(target_slots)
        if train["slots"] != expected_slots or valid["slots"] != expected_slots:
            raise ValueError("discovery and validation must cover every declared target slot before model loading")
    return {"discovery_pairs": train["pairs"], "validation_pairs": valid["pairs"],
            "concept": next(iter(train["concepts"])),
            "discovery_option_counts": sorted(train["option_counts"]),
            "validation_option_counts": sorted(valid["option_counts"]),
            "target_slots": sorted(train["slots"]) if target_slots is not None else None}


def resolve_direction_fit_parent(*, candidate_checkpoint, parent_report, parent_report_sha256,
                                 parent_qualification, parent_qualification_sha256,
                                 parent_snapshot, stage1_parent_checkpoint=None,
                                 fit_receipt=None, fit_receipt_sha256=None, device="cpu",
                                 dtype="bfloat16", load_qualified_parent, load_language_child):
    """Route a qualified parent or exact receipt-pinned child to the fitter."""
    child_inputs = (stage1_parent_checkpoint, fit_receipt, fit_receipt_sha256)
    if any(child_inputs) and not all(child_inputs):
        raise ValueError("child direction fitting requires the Stage 1 checkpoint and pinned fit receipt")
    if all(child_inputs):
        if parent_snapshot is None:
            raise ValueError("child direction fitting requires strict local snapshot reconstruction")
        return load_language_child(
            candidate_checkpoint, stage1_checkpoint=stage1_parent_checkpoint,
            receipt=fit_receipt, receipt_sha256=fit_receipt_sha256,
            report=parent_report, report_sha256=parent_report_sha256,
            qualification=parent_qualification,
            qualification_sha256=parent_qualification_sha256,
            snapshot=parent_snapshot, device=device, dtype=dtype)
    return load_qualified_parent(
        candidate_checkpoint, report=parent_report,
        report_sha256=parent_report_sha256, qualification=parent_qualification,
        qualification_sha256=parent_qualification_sha256,
        snapshot=parent_snapshot, device=device, dtype=dtype)
