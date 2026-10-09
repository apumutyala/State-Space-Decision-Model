"""Train-only paired discovery of native final-hidden and fitted logit axes.

``fit_native_directions(parent, discovery_pairs, validation_pairs=..., scale=...,
random_seed=...)`` captures real HeadInput.fork and FinalStateFork logits through
the frozen native provider. Each pair has exactly pair_id/world_id/split/concept,
source/effect request specifications, and source_target/effect_target. Each
specification has task_input/host/catalog/question_id/question/options. Requests
are reconstructed with EMPTY proposal histories; labels only enter diagnostics.

One concept and source->effect orientation are declared before capture. The
fixed estimator is the unit-normalized mean paired feature difference (first
average over candidates), and per-K unit-normalized centered mean paired raw
logit difference. There is no optimization, label-dependent selection, tuning,
random fallback, calibration refit or parent mutation. Zero contrasts fail.
Separate development_validation pairs measure four native edit arms without
changing the fit. This is slot-level development evidence, not a language gate.

Optional ``target_slots=["comparison-relation"]`` binds a finite declared
planner-slot scope to v2 evidence/artifact/registration; omitted scope retains
legacy v1 all-slot application. Both discovery and validation must cover every
declared slot. No tasks or calls are selected based on labels or model behavior.

``write_direction_fit(result, out)`` writes the exact intervention adapter .pt
schema, fitting evidence and machine registration to a fresh directory. The
CLI uses the existing qualified-parent loader and local pinned snapshot only.
No heldout/Index files are read, generated or accepted by the pair API.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .gate_interventions import (
    ARMS, BOUNDARY, DIRECTION_SCHEMA, DIRECTION_SCHEMA_V2, OUTPUT_CONTROL, SCHEMA, SCHEMA_V2, SCOPE, TARGET_SLOTS, _ArmProvider,
    _OWNER, _native, _parent_state_sha, _sha, _tensor_sha, native_capture_identity,
    registration_sha256, question_slot, validate_target_slots,
)
from .parent_bridge import DECISION_SCHEMA, FINGERPRINT_ALGORITHM, decision_request, legacy_request_fingerprint
from .qualified_parent import file_sha256

FIT_SCHEMA = "ssdm.native_fork_direction_fit.v1"
FIT_SCHEMA_V2 = "ssdm.native_fork_direction_fit.v2"
METHOD = "unit_mean_paired_candidate_mean_hidden_and_centered_raw_logits.v1"
_PAIR_KEYS = {"pair_id", "world_id", "split", "concept", "source", "effect", "source_target", "effect_target"}
_SPEC_KEYS = {"task_input", "host", "catalog", "question_id", "question", "options"}


def _request(spec, parent, identity):
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError("label-free request specification required; raw/teacher-forced requests are forbidden")
    pins = parent.planner_kwargs()
    request, _ = decision_request(
        **deepcopy(spec), model=pins["model"], parent_revision=pins["parent_revision"],
        request_id=identity, proposed_slot_choices=[],
    )
    return request


def _validate_pairs(pairs, split, parent):
    from decide.schema import canonical_options
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("nonempty separate train and validation pair lists required")
    checked = []
    for pair in deepcopy(pairs):
        if (not isinstance(pair, dict) or set(pair) != _PAIR_KEYS or pair["split"] != split
                or any(not isinstance(pair[key], str) or not pair[key]
                       for key in ("pair_id", "world_id", "concept", "source_target", "effect_target"))):
            raise ValueError("only explicitly tagged development pairs with fitter-only labels are accepted")
        requests = {side: _request(pair[side], parent, _sha({"pair": pair["pair_id"], "side": side}))
                    for side in ("source", "effect")}
        if any(pair["source"][key] != pair["effect"][key] for key in ("question_id", "question", "options")):
            raise ValueError("matched source/effect question and canonical candidate domain required")
        qid = pair["source"]["question_id"]
        options = [oid for oid, _ in canonical_options(requests["source"]["questions"][qid])]
        if (pair["source_target"] not in options or pair["effect_target"] not in options
                or pair["source_target"] == pair["effect_target"]):
            raise ValueError("a real source/effect target contrast in the unchanged domain is required")
        # Exclude duplicate model-visible content, even if IDs/world tags differ.
        content = {side: _sha({"state": request["state"], "questions": request["questions"]})
                   for side, request in requests.items()}
        if content["source"] == content["effect"]:
            raise ValueError("source and effect must change model-visible inputs")
        checked.append((pair, requests, options, content))
    if len({p[0]["pair_id"] for p in checked}) != len(checked):
        raise ValueError("pair IDs must be unique")
    hashes = [h for row in checked for h in row[3].values()]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate discovery/validation input content")
    if len({row[0]["concept"] for row in checked}) != 1:
        raise ValueError("one preregistered oriented concept per fit is required")
    return checked


class _Capture:
    def __init__(self, k, d):
        self.k, self.d = k, d
        self.features = self.logits = None
        self.calls = 0

    def pre(self, _head, args):
        import torch
        from heads.features import HeadInput
        if (self.calls or not args or not isinstance(args[0], HeadInput)
                or args[0].fork is None or tuple(args[0].fork.shape) != (1, self.k, self.d)
                or args[0].ks != [self.k] or not bool(args[0].option_mask.all())
                or not bool(torch.isfinite(args[0].fork).all())):
            raise ValueError("native capture must traverse exactly one complete candidate domain")
        self.calls += 1
        self.features = args[0].fork.detach().cpu().clone()[0]

    def post(self, _head, _args, result):
        import torch
        if (self.features is None or self.logits is not None
                or tuple(result.logits.shape) != (1, self.k)
                or not bool(torch.isfinite(result.logits).all())):
            raise ValueError("native capture logits mismatch")
        self.logits = result.logits.detach().cpu().clone()[0]


def _response_probabilities(response, request, parent):
    """The ordinary planner's response identity/statelessness and stage checks."""
    from .planner import _RESPONSE_ENVELOPE_KEYS, _check_stages
    if not isinstance(response, dict) or not isinstance(response.get("ssdm"), dict):
        raise ValueError("native fitting response identity mismatch")
    env = response["ssdm"]
    diagnostics = env.get("diagnostics")
    if (set(env) - _RESPONSE_ENVELOPE_KEYS or env.get("schema") != DECISION_SCHEMA
            or env.get("status") != "ok" or env.get("request_id") != request["ssdm"]["request_id"]
            or not isinstance(diagnostics, dict)
            or diagnostics.get("revision_id") != parent.planner_kwargs()["parent_revision"]
            or diagnostics.get("request_fingerprint") != legacy_request_fingerprint(request)
            or diagnostics.get("request_fingerprint_algorithm") != FINGERPRINT_ALGORITHM
            or not isinstance(diagnostics.get("provenance"), dict)
            or diagnostics["provenance"].get("supplied") is not False
            or diagnostics["provenance"].get("applied") is not False
            or diagnostics.get("state_handle_in") is not None or diagnostics.get("state_reuse") is not False
            or env.get("state_handle") is not None):
        raise ValueError("native fitting response identity mismatch")
    qid = next(iter(request["questions"]))
    probabilities, _ = _check_stages(env, request, qid, request["questions"][qid]["criteria"],
                                     parent.planner_kwargs()["expected_served_config"])
    return probabilities


def _capture(parent, request, options, d):
    import torch
    head = _native(parent)._loaded.backend.model.head
    capture = _Capture(len(options), d)
    handles = [head.register_forward_pre_hook(capture.pre), head.register_forward_hook(capture.post)]
    try:
        with torch.inference_mode():
            response = parent.decide(deepcopy(request))
        env = response.get("ssdm", {})
        if env.get("status") != "ok" or capture.calls != 1 or capture.logits is None:
            raise ValueError("native discovery provider failed; no fitted fallback is permitted")
        _response_probabilities(response, request, parent)
        record = {"request_sha256": _sha(request),
                  "request_fingerprint": env.get("diagnostics", {}).get("request_fingerprint"),
                  "option_ids": options, "hidden_shape": list(capture.features.shape),
                  "hidden_sha256": _tensor_sha(capture.features), "logits_sha256": _tensor_sha(capture.logits)}
        if not record["request_fingerprint"]:
            raise ValueError("native discovery request fingerprint missing")
        return capture.features.double(), capture.logits.double(), record
    finally:
        for handle in handles:
            handle.remove()


def _unit(vector, name):
    import torch
    norm = vector.norm()
    if not bool(torch.isfinite(vector).all()) or float(norm) <= 1e-12:
        raise ValueError(f"{name} has a zero/nonfinite paired contrast; no random fallback")
    return (vector / norm).float().contiguous()


def _contrast(row, parent, d):
    pair, requests, options, _content = row
    source_h, source_z, source_record = _capture(parent, requests["source"], options, d)
    effect_h, effect_z, effect_record = _capture(parent, requests["effect"], options, d)
    dh = (effect_h - source_h).mean(0)
    dz = effect_z - source_z
    dz -= dz.mean()
    record = {"pair_id": pair["pair_id"], "world_id": pair["world_id"],
              "source_target": pair["source_target"], "effect_target": pair["effect_target"],
              "source": source_record, "effect": effect_record,
              "hidden_contrast_l2": float(dh.norm()), "logit_contrast_l2": float(dz.norm())}
    return dh, dz, record


def _cosine(left, right):
    denominator = float(left.norm() * right.double().norm())
    return float(left @ right.double() / denominator) if denominator > 1e-12 else None


def _validate(row, parent, capture_identity, direction, outputs, scale, random, protocol_sha, target_slots=None):
    """Use exact native edit hooks and served outputs; labels stay outside provider."""
    import torch
    pair, requests, options, _ = row
    dh, dz, record = _contrast(row, parent, direction.numel())
    k = len(options)
    record["hidden_axis_cosine"] = _cosine(dh, direction)
    record["output_axis_cosine"] = _cosine(dz, outputs[k])
    record["arms"] = {}
    head = _native(parent)._loaded.backend.model.head
    for arm in ARMS:
        registration = {"scale": scale, "registration_sha256": protocol_sha,
                        "capture_identity_sha256": _sha(capture_identity)}
        if target_slots is not None:
            registration["target_slots"] = target_slots
        provider = _ArmProvider(parent, arm, registration,
                                capture_identity, direction, random, outputs)
        handles = [head.register_forward_pre_hook(provider.pre_hook), head.register_forward_hook(provider.post_hook)]
        try:
            with torch.inference_mode():
                response = provider.decide(deepcopy(requests["source"]))
            env = response.get("ssdm", {})
            probabilities = _response_probabilities(response, requests["source"], parent)
            if env.get("status") != "ok" or len(provider.records) != 1 or provider.records[0]["head_calls"] != 1:
                raise ValueError("native development validation arm failed")
            predicted = min(options, key=lambda oid: (-probabilities[oid], oid))
            record["arms"][arm] = {"predicted_option_id": predicted,
                "source_target_selected": predicted == pair["source_target"],
                "effect_target_selected": predicted == pair["effect_target"],
                "effect_minus_source_probability": probabilities[pair["effect_target"]] - probabilities[pair["source_target"]],
                "capture": provider.records[0]}
        finally:
            for handle in handles:
                handle.remove()
    return record


def fit_native_directions(parent, discovery_pairs: list[dict], *, validation_pairs: list[dict],
                          scale: float, random_seed: int, target_slots: list[str] | None = None) -> dict:
    """Fit train-only native axes; separate validation never changes estimator/settings.

    Accept NativeParent for small synthetic CPU tests or an admitted
    QualifiedParent for future champion work. The CLI requires qualification.
    No artifact path is opened here; all input lists are caller-supplied dev data.
    """
    import torch
    if (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
            or type(random_seed) is not int or not 0 <= random_seed < 2**63):
        raise ValueError("fixed positive finite scale and nonnegative 63-bit random seed required")
    native = _native(parent)
    capture_identity = native_capture_identity(parent)
    train = _validate_pairs(discovery_pairs, "development_train", parent)
    validation = _validate_pairs(validation_pairs, "development_validation", parent)
    if target_slots is not None:
        target_slots = validate_target_slots(target_slots)
        for cohort in (train, validation):
            observed_slots = {question_slot(row[0][side]["question_id"]) for row in cohort for side in ("source", "effect")}
            if observed_slots != set(target_slots):
                raise ValueError("discovery and validation must cover exactly the declared target slots")
    train_worlds = sorted({row[0]["world_id"] for row in train})
    validation_worlds = sorted({row[0]["world_id"] for row in validation})
    if (set(train_worlds) & set(validation_worlds)
            or {row[0]["pair_id"] for row in train} & {row[0]["pair_id"] for row in validation}
            or {h for row in train for h in row[3].values()} & {h for row in validation for h in row[3].values()}):
        raise ValueError("discovery and validation worlds/IDs/model-visible content must be disjoint")
    if train[0][0]["concept"] != validation[0][0]["concept"]:
        raise ValueError("validation must use the preregistered oriented discovery concept")
    counts = sorted({len(row[2]) for row in train})
    if {len(row[2]) for row in validation} - set(counts):
        raise ValueError("validation requires train-fitted controls for every candidate count")
    head = native._loaded.backend.model.head
    if head._forward_pre_hooks or head._forward_hooks:
        raise ValueError("native head already has an instrumentation owner")
    if not _OWNER.acquire(blocking=False):
        raise ValueError("another native experiment owns the instrumentation")
    try:
        before = _parent_state_sha(native._loaded.backend.model)
        evidence = {"schema": FIT_SCHEMA, "method": METHOD, "concept": train[0][0]["concept"],
            "fit_split": "development_train", "validation_split": "development_validation",
            "capture_identity": capture_identity, "capture_identity_sha256": _sha(capture_identity),
            "discovery_pairs_sha256": _sha(discovery_pairs), "validation_pairs_sha256": _sha(validation_pairs),
            "discovery_world_ids": train_worlds, "validation_world_ids": validation_worlds,
            "scale": scale, "random_seed": random_seed, "fitter_source_sha256": file_sha256(Path(__file__)),
            "labels_used_for_fit": False, "proposal_history": "empty; no teacher-forced choices",
            "source_to_effect_orientation": "as declared in discovery pairs; no validation selection",
            "validation_protocol": "native source request; semantic/sham/random/output_only edits toward effect concept",
            "scope": "slot-level synthetic development; no semantic efficacy or language qualification claim",
            "state_unchanged_sha256_before": before, "discovery": [], "validation": []}
        if target_slots is not None:
            evidence.update(schema=FIT_SCHEMA_V2, target_slots=target_slots)
        d = capture_identity["hidden_dimension"]
        hidden_sum = torch.zeros(d, dtype=torch.float64)
        output_sums = {k: torch.zeros(k, dtype=torch.float64) for k in counts}
        count_by_k = {k: 0 for k in counts}
        for row in train:
            dh, dz, record = _contrast(row, parent, d)
            hidden_sum += dh
            output_sums[len(row[2])] += dz
            count_by_k[len(row[2])] += 1
            evidence["discovery"].append(record)
        direction = _unit(hidden_sum / len(train), "semantic axis")
        outputs = {k: _unit(output_sums[k] / count_by_k[k], f"K={k} output-only control") for k in counts}
        evidence["semantic_direction_sha256"] = _tensor_sha(direction)
        evidence["output_only_directions_sha256"] = {str(k): _tensor_sha(value) for k, value in outputs.items()}
        evidence["mean_hidden_contrast_l2"] = float((hidden_sum / len(train)).norm())
        evidence["mean_output_contrast_l2"] = {str(k): float((output_sums[k] / count_by_k[k]).norm()) for k in counts}
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        random = torch.randn(direction.shape, dtype=torch.float64, generator=generator)
        random = (random * (direction.double().norm() / random.norm())).float()
        protocol_sha = _sha({"method": METHOD, "pairs": evidence["discovery_pairs_sha256"],
                             "scale": scale, "random_seed": random_seed,
                             "target_slots": target_slots,
                             "direction": _tensor_sha(direction), "outputs": {str(k): _tensor_sha(v) for k, v in outputs.items()}})
        evidence["validation_protocol_sha256"] = protocol_sha
        for row in validation:
            evidence["validation"].append(_validate(row, parent, capture_identity, direction, outputs, scale, random, protocol_sha, target_slots))
        parent.planner_kwargs()
        after = _parent_state_sha(native._loaded.backend.model)
        if before != after:
            raise ValueError("native parent bytes changed during direction discovery")
        evidence["state_unchanged_sha256_after"] = after
        evidence["candidate_count_discovery_pairs"] = {str(k): count_by_k[k] for k in counts}
        evidence["validation_effect_target_selected"] = {
            arm: sum(record["arms"][arm]["effect_target_selected"] for record in evidence["validation"]) for arm in ARMS}
        artifact = {"schema": DIRECTION_SCHEMA, "parent_revision": capture_identity["parent_revision"],
            "capture_identity_sha256": _sha(capture_identity), "capture_boundary": BOUNDARY,
            "source_split": "discovery_train_only", "discovery_world_ids": train_worlds,
            "discovery_pairs_sha256": evidence["discovery_pairs_sha256"],
            "output_only_fit_evidence_sha256": _sha(evidence), "direction": direction,
            "output_only_directions": {str(k): value for k, value in outputs.items()}}
        if target_slots is not None:
            artifact.update(schema=DIRECTION_SCHEMA_V2, target_slots=target_slots)
        return {"artifact": artifact, "evidence": evidence,
                "source_pairs": {"discovery": deepcopy(discovery_pairs), "validation": deepcopy(validation_pairs)}}
    finally:
        _OWNER.release()


def write_direction_fit(result: dict, out: str | Path) -> dict:
    """Write fresh immutable-by-convention artifacts; refuse existing destinations."""
    import torch
    out = Path(out).resolve()
    artifact, evidence, sources = result["artifact"], result["evidence"], result["source_pairs"]
    capture = evidence["capture_identity"]
    discovery_worlds = sorted({pair["world_id"] for pair in sources["discovery"]})
    expected_lineage = {"schema": DIRECTION_SCHEMA, "parent_revision": capture["parent_revision"],
        "capture_identity_sha256": _sha(capture), "capture_boundary": BOUNDARY,
        "source_split": "discovery_train_only", "discovery_world_ids": discovery_worlds,
        "discovery_pairs_sha256": _sha(sources["discovery"]), "output_only_fit_evidence_sha256": _sha(evidence)}
    targeted = artifact.get("schema") == DIRECTION_SCHEMA_V2
    if targeted:
        slots = validate_target_slots(evidence.get("target_slots"))
        if (evidence.get("schema") != FIT_SCHEMA_V2
                or any({question_slot(pair[side]["question_id"]) for pair in cohort for side in ("source", "effect")} != set(slots)
                       for cohort in sources.values())):
            raise ValueError("fitting applicability evidence identity changed")
        expected_lineage.update(schema=DIRECTION_SCHEMA_V2, target_slots=slots)
    elif evidence.get("schema") != FIT_SCHEMA or "target_slots" in evidence:
        raise ValueError("fitting applicability evidence identity changed")
    if (artifact["output_only_fit_evidence_sha256"] != _sha(evidence)
            or set(artifact) != set(expected_lineage) | {"direction", "output_only_directions"}
            or any(artifact[key] != value for key, value in expected_lineage.items())
            or evidence["capture_identity_sha256"] != _sha(capture)
            or evidence["discovery_world_ids"] != discovery_worlds
            or any(pair["split"] != "development_train" for pair in sources["discovery"])
            or any(pair["split"] != "development_validation" for pair in sources["validation"])
            or set(discovery_worlds) & {pair["world_id"] for pair in sources["validation"]}
            or evidence["semantic_direction_sha256"] != _tensor_sha(artifact["direction"])
            or evidence["output_only_directions_sha256"] != {k: _tensor_sha(value) for k, value in artifact["output_only_directions"].items()}
            or evidence["discovery_pairs_sha256"] != _sha(sources["discovery"])
            or evidence["validation_pairs_sha256"] != _sha(sources["validation"])):
        raise ValueError("fitting evidence identity changed")
    out.mkdir(parents=True, exist_ok=False)
    path = out / "directions.pt"
    torch.save(artifact, path)
    p = {"schema": SCHEMA_V2 if targeted else SCHEMA, "parent_revision": artifact["parent_revision"],
         "capture_identity_sha256": artifact["capture_identity_sha256"], "capture_boundary": BOUNDARY,
         "scope": SCOPE, "discovery_world_ids": artifact["discovery_world_ids"],
         "discovery_pairs_sha256": artifact["discovery_pairs_sha256"],
         "directions": {"path": str(path), "sha256": file_sha256(path)},
         "scale": evidence["scale"], "random_seed": evidence["random_seed"], "output_control": OUTPUT_CONTROL}
    if targeted:
        p["target_slots"] = artifact["target_slots"]
    p["registration_sha256"] = registration_sha256(p)
    for name, value in (("fit-evidence.json", evidence), ("registration.json", p), ("source-pairs.json", sources)):
        (out / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p


def development_relation_pairs(examples: list[dict]) -> list[dict]:
    """Fresh empty-prefix less->greater comparison requests from declared dev worlds.

    Equality rows are excluded by the fixed concept definition, never by model
    behavior. This helper never uses model_inputs (which can contain gold prior
    slots). Labels name only the source/effect relation for fitting diagnostics.
    """
    from .development_data import comparison_swap_counterfactual
    pairs = []
    for example in examples:
        if example.get("development_only") is not True or example.get("development_split") not in (
                "development_train", "development_validation"):
            raise ValueError("only synthetic development examples are accepted")
        metadata = example["training_only"]
        if metadata["kind"] != "compare" or metadata["program"]["nodes"][2]["relation"] == "equal":
            continue
        swapped = comparison_swap_counterfactual(example)
        source, effect = ((example, swapped) if metadata["program"]["nodes"][2]["relation"] == "less_than"
                          else (swapped, example))
        def spec(row):
            training = row["training_only"]
            records = {r["record_ref"]: r for r in training["catalog"]["records"]}
            left, right = training["program"]["nodes"][:2]
            # Task intention is normal model-visible input. Do not read the
            # generated model_inputs or any target/proposal history to build it.
            task = f"Compare the October {left['field'].replace('_', ' ')} of {records[left['record_ref']]['entity_ref']} and {records[right['record_ref']]['entity_ref']}."
            return {"task_input": task, "host": deepcopy(training["host"]), "catalog": deepcopy(training["catalog"]),
                    "question_id": "slot-004-comparison-relation", "question": "Select the comparison relation for the original task.",
                    "options": {"r001": "greater_than", "r002": "less_than", "r003": "equal"}}
        pairs.append({"pair_id": example["task_id"] + "-less-to-greater", "world_id": example["task_id"],
                      "split": example["development_split"], "concept": "less_than_to_greater_than",
                      "source": spec(source), "effect": spec(effect), "source_target": "r002", "effect_target": "r001"})
    return pairs


def development_operation_pairs(*, split: str, worlds: int, seed: int,
                                model: str = "gate-operation-fit",
                                parent_revision: str = "operation-fit:v1") -> list[dict]:
    from .gate_operation_pairs import development_operation_pairs as build
    return build(split=split, worlds=worlds, seed=seed, model=model, parent_revision=parent_revision)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-report", "parent-report-sha256", "parent-qualification",
                 "parent-qualification-sha256", "parent-snapshot", "discovery-pairs", "validation-pairs", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--scale", required=True, type=float)
    p.add_argument("--random-seed", required=True, type=int)
    p.add_argument("--target-slots", nargs="+", choices=sorted(TARGET_SLOTS), help="explicit native edit slots; omit for legacy all-slot protocol")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    args = p.parse_args(argv)
    from .gates import _strict_json
    from .qualified_parent import load_qualified_parent
    if Path(args.out).exists():
        raise ValueError("output already exists; preserve previous fitting evidence")
    # Parse explicit pair files before model reconstruction. Invalid split tags
    # cannot be turned into a successful model run by the loader.
    discovery = _strict_json(Path(args.discovery_pairs).read_text(encoding="utf-8"))
    validation = _strict_json(Path(args.validation_pairs).read_text(encoding="utf-8"))
    for rows, split in ((discovery, "development_train"), (validation, "development_validation")):
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or row.get("split") != split for row in rows):
            raise ValueError("explicit separate development train/validation pair files required")
    parent = load_qualified_parent(args.parent, report=args.parent_report, report_sha256=args.parent_report_sha256,
        qualification=args.parent_qualification, qualification_sha256=args.parent_qualification_sha256,
        snapshot=args.parent_snapshot, device=args.device, dtype=args.dtype)
    result = fit_native_directions(parent, discovery, validation_pairs=validation, scale=args.scale,
                                   random_seed=args.random_seed, target_slots=args.target_slots)
    registration = write_direction_fit(result, args.out)
    print(json.dumps({"status": "development_fitted_no_qualification", "registration": registration,
                      "validation_effect_target_selected": result["evidence"]["validation_effect_target_selected"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
