"""CPU-only learned query-target diagnostic on seeded synthetic development data.

This is a new experimental choice slot, not a relevance guarantee or a production
planner. Targets are operation/field/entity/order, never the answer value. Gold
targets come only from authored synthetic programs outside model inputs. The
model projects unmodified task text and all candidate descriptions from a legacy
D request; it does not parse the question or encode IDs, programs or labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .development_data import (
    development_manifest, development_split_spec, generate_development_tasks,
)
from .parent_bridge import decision_request


def target_key(target: dict[str, Any]) -> str:
    return json.dumps(target, sort_keys=True, separators=(",", ":"))


def target_candidates(state: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Enumerate every syntactic target from permitted records, without labels."""
    candidates = []
    records = state["records"]
    for kind in ("read", "explain", "compare"):
        for left in records:
            for field in sorted(left["fields"]):
                scope = left["fields"][field]["temporal_scope"]
                target = {"kind": kind, "field": field, "left": left["record_ref"],
                          "temporal_scope": scope}
                field_text = field.replace("_", " ")
                if kind != "compare":
                    candidates.append((target, f"{kind.title()} {field_text} for {left['entity_ref']} at {scope}."))
                else:
                    for right in records:
                        if right["record_ref"] == left["record_ref"] or field not in right["fields"]:
                            continue
                        # This is syntax enumeration, not a check of truth or comparability.
                        candidates.append((dict(target, right=right["record_ref"]),
                            f"Compare {field_text} of {left['entity_ref']} then {right['entity_ref']} at {scope}."))
    return candidates


def example_target(example: dict[str, Any]) -> dict[str, Any]:
    """Offline supervision only: no natural-language question parsing."""
    training = example["training_only"]
    reads = [node for node in training["program"]["nodes"] if node["op"] == "ReadFact"]
    first = reads[0]
    record = next(r for r in training["catalog"]["records"] if r["record_ref"] == first["record_ref"])
    target = {"kind": training["kind"], "field": first["field"],
              "left": first["record_ref"],
              "temporal_scope": record["fields"][first["field"]]["temporal_scope"]}
    if training["kind"] == "compare":
        target["right"] = reads[1]["record_ref"]
    return target


def make_row(example: dict[str, Any], *, distractors: bool = False) -> dict[str, Any]:
    """Create a D-compatible query-target slot and separate evaluator metadata."""
    original = json.loads(example["model_inputs"][0]["state"])
    training = example["training_only"]
    host, catalog = copy.deepcopy(training["host"]), copy.deepcopy(training["catalog"])
    if distractors:
        # A novel irrelevant field and an additional permitted entity. Their
        # presence cannot remove any original candidate or change the target.
        for record in catalog["records"]:
            record["fields"]["repairable"] = {"value": True, "value_type": "boolean",
                "unit": None, "temporal_scope": "2026-10", "quantity_kind": "repairable"}
        extra = copy.deepcopy(catalog["records"][0])
        extra["record_ref"] = "development-distractor-record"
        names = development_split_spec(example["development_split"])["names"]
        extra["entity_ref"] = next(name for name in names
            if name not in {r["entity_ref"] for r in catalog["records"]})
        catalog["records"].append(extra)
        catalog["snapshots"][host["context_ref"]]["record_refs"].append(extra["record_ref"])
        registry = catalog["registries"][host["registry_ref"]]
        registry["registered_fields"].append("repairable")
        registry["field_predicates"]["repairable"] = "IsRepairable"
        context_ref = host["context_ref"] + ":query-distractors"
        catalog["snapshots"][context_ref] = catalog["snapshots"].pop(host["context_ref"])
        host["context_ref"] = host["provider"]["input_context_ref"] = context_ref
    from .parent_bridge import audience_state
    visible = json.loads(audience_state(original["task_input"], host, catalog))
    candidates = target_candidates(visible)
    rng = random.Random(f"query-target:{example['task_id']}")
    rng.shuffle(candidates)
    binding = {f"qt{i:03d}": target for i, (target, _) in enumerate(candidates, 1)}
    options = {f"qt{i:03d}": description for i, (_, description) in enumerate(candidates, 1)}
    host["limits"]["max_options_per_question"] = len(options)
    request, _ = decision_request(task_input=original["task_input"], host=host, catalog=catalog,
        model="query-target-development-only", parent_revision="query-target-untrained:v1",
        request_id=example["task_id"] + (":distractors" if distractors else ""),
        question_id="query-target", question="Select the target requested by the user's task; do not select an answer value.",
        options=options)
    target = example_target(example)
    gold = next(oid for oid, item in binding.items() if item == target)
    return {"id": example["task_id"], "request": request,
            "evaluation_only": {"target": target, "option_id": gold, "binding": binding}}


def model_text(request: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    state = json.loads(request["state"])
    question, = request["questions"].values()
    # IDs/order are output bookkeeping, never learned features. No candidate is
    # pruned; sorting by its description gives exact ID relabeling invariance.
    choices = sorted(question["criteria"].items(), key=lambda item: item[1])
    return state["task_input"], [oid for oid, _ in choices], [text for _, text in choices]


def byte_batch(texts: list[str]) -> torch.Tensor:
    encoded = [list(text.encode("utf-8")) for text in texts]
    out = torch.zeros((len(texts), max(map(len, encoded))), dtype=torch.long)
    for index, row in enumerate(encoded):
        out[index, :len(row)] = torch.tensor(row) + 1
    return out


class QueryTargetSSM(nn.Module):
    """Shared byte diagonal SSM plus learned candidate interaction scorer.

    h[t] = a*h[t-1] + (1-a)*tanh(B*embedding(byte[t])). Terminal
    and mean recurrent states use the exact closed form, without positional
    attention. This recurrence follows the existing tiny_ssm_slots prototype.
    All weights start random and train here; there is no pretrained parent.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.input = nn.Linear(width, width)
        self.decay_logit = nn.Parameter(torch.logit(torch.linspace(0.5, 0.999, width)))
        self.scorer = nn.Sequential(nn.Linear(width * 6, 96), nn.Tanh(), nn.Linear(96, 1))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0)
        length = mask.sum(1)
        position = torch.arange(ids.shape[1])[None, :]
        lag = (length[:, None] - 1 - position).clamp_min(0)
        remaining = (length[:, None] - position).clamp_min(0)
        a = self.decay_logit.sigmoid()[None, None, :]
        u = torch.tanh(self.input(self.embedding(ids))) * mask[:, :, None]
        terminal = (u * a.pow(lag[:, :, None]) * (1 - a)).sum(1)
        mean = (u * (1 - a.pow(remaining[:, :, None]))).sum(1) / length[:, None].clamp_min(1)
        return torch.cat((terminal, mean), 1)

    def forward(self, queries: torch.Tensor, candidates: torch.Tensor, owners: torch.Tensor) -> torch.Tensor:
        query, candidate = self.encode(queries)[owners], self.encode(candidates)
        return self.scorer(torch.cat((query, candidate, query * candidate), 1)).squeeze(-1)


def tensors(rows: list[dict[str, Any]]):
    projected = [model_text(row["request"]) for row in rows]
    counts = {len(item[1]) for item in projected}
    if len(counts) != 1:
        raise ValueError("batch requires equal candidate counts")
    count = counts.pop()
    queries = byte_batch([item[0] for item in projected])
    candidates = byte_batch([text for item in projected for text in item[2]])
    owners = torch.arange(len(rows)).repeat_interleave(count)
    labels = torch.tensor([item[1].index(row["evaluation_only"]["option_id"])
                           for row, item in zip(rows, projected)])
    return queries, candidates, owners, labels, count


@torch.no_grad()
def evaluate(model: QueryTargetSSM, rows: list[dict[str, Any]], batch_size: int = 32):
    model.eval()
    observations = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        query, candidate, owners, _, count = tensors(batch)
        scores = model(query, candidate, owners).reshape(len(batch), count)
        for row, selected in zip(batch, scores.argmax(1).tolist()):
            _, ids, _ = model_text(row["request"])
            predicted = row["evaluation_only"]["binding"][ids[selected]]
            gold = row["evaluation_only"]["target"]
            observations.append({"id": row["id"], "predicted": predicted, "target": gold,
                                 "correct": predicted == gold})
    fields = ("kind", "field", "left", "right", "temporal_scope")
    summary = {"count": len(rows), "exact_correct": sum(row["correct"] for row in observations),
        "attribute_correct": {field: sum(row["predicted"].get(field) == row["target"].get(field)
                                         for row in observations) for field in fields},
        "by_kind": {kind: {"count": sum(row["target"]["kind"] == kind for row in observations),
            "correct": sum(row["correct"] and row["target"]["kind"] == kind for row in observations)}
            for kind in ("read", "explain", "compare")}}
    return summary, observations


def permute_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(rows)
    for row in result:
        question, = row["request"]["questions"].values()
        old = list(question["criteria"])
        mapping = {oid: f"relabeled-{len(old) - i:03d}" for i, oid in enumerate(old)}
        question["criteria"] = {mapping[oid]: question["criteria"][oid] for oid in reversed(old)}
        ev = row["evaluation_only"]
        ev["binding"] = {mapping[oid]: target for oid, target in ev["binding"].items()}
        ev["option_id"] = mapping[ev["option_id"]]
    return result


# Fixed before training; this is one condition, not a search over templates.
CONFIG = {"model_seeds": [20261006, 20261016, 20261026],
          "train_seed": 20261006, "validation_seed": 20261009,
          "train_count": 600, "validation_count": 180, "steps": 300,
          "width": 32, "batch_size": 32, "lr": 0.003,
          "cpu_threads": 2, "device": "cpu", "max_runtime_seconds": 180,
          "bootstrap_seed": 3197, "bootstrap_replicates": 2000}
TRAIN_TEMPLATES = {
    "read": (
        "What is {entity}'s {field} in October 2026?",
        "Give the October 2026 {field} for {entity}.",
        "For {entity}, report the {field} recorded in October 2026.",
        "In October 2026, what {field} does {entity} have?",
        "Please retrieve {entity}'s October 2026 {field}.",
        "Find the October 2026 fact about {field} for {entity}."),
    "explain": (
        "Explain the checked {field} fact for {entity}.",
        "Show the checked derivation for {entity}'s {field}.",
        "For {entity}, explain how the {field} fact is established.",
        "Give a checked explanation for the {field} of {entity}.",
        "Please justify {entity}'s {field} using the checked derivation.",
        "Walk through the derivation that supports {entity}'s {field}."),
    "compare": (
        "Compare the October {field} of {left} and {right}.",
        "How do {left}'s and {right}'s October {field} compare?",
        "In October 2026, compare {left} and {right} on {field}.",
        "Assess how {left}'s October {field} compares to {right}'s.",
        "Considering October 2026 {field}, compare {left} against {right}.",
        "Please compare {left} with {right} using their October 2026 {field}."),
}
VALIDATION_TEMPLATES = {
    "old_failing": {
        "read": "State {entity}'s {field} for October 2026.",
        "explain": "Provide a checked derivation of {entity}'s {field}.",
        "compare": "For October 2026, compare {left}'s and {right}'s {field}."},
    "new_disjoint": {
        "read": "Tell me the {field} belonging to {entity} as of October 2026.",
        "explain": "Describe the checked reasoning behind the {field} belonging to {entity}.",
        "compare": "With {left} first and {right} second, compare their October 2026 {field}."},
}


def authored_text(example, template):
    """Offline dataset authoring from programs, never a model-side parser."""
    training = example["training_only"]
    reads = [n for n in training["program"]["nodes"] if n["op"] == "ReadFact"]
    records = {r["record_ref"]: r for r in training["catalog"]["records"]}
    left = records[reads[0]["record_ref"]]["entity_ref"]
    right = records[reads[-1]["record_ref"]]["entity_ref"]
    return template.format(entity=left, left=left, right=right,
                           field=reads[0]["field"].replace("_", " "))


def replace_question(row, text):
    row = copy.deepcopy(row)
    state = json.loads(row["request"]["state"])
    state["task_input"] = text
    row["request"]["state"] = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return row


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(out):
    if out.exists():
        raise ValueError("fresh output path required")
    out.mkdir(parents=True)
    train_examples = generate_development_tasks(CONFIG["train_seed"], CONFIG["train_count"], split="development_train")
    val_examples = generate_development_tasks(CONFIG["validation_seed"], CONFIG["validation_count"], split="development_validation")
    baseline = [make_row(e) for e in train_examples]
    diverse = [replace_question(row, authored_text(e, TRAIN_TEMPLATES[e["training_only"]["kind"]][(i // 3) % 6]))
               for i, (e, row) in enumerate(zip(train_examples, baseline))]
    validation = {name: [replace_question(make_row(e), authored_text(e, templates[e["training_only"]["kind"]]))
                         for e in val_examples] for name, templates in VALIDATION_TEMPLATES.items()}
    assert all(a["evaluation_only"] == b["evaluation_only"] for a, b in zip(baseline, diverse))
    assert all(t not in TRAIN_TEMPLATES[k] for family in VALIDATION_TEMPLATES.values() for k, t in family.items())
    data = {"train": {"baseline": baseline, "all_task_diversity": diverse}, "validation": validation}
    registration = {"schema": "ssdm.query_template_transfer_preregistration.v1", "config": CONFIG,
        "base_commit": "4733f785e5284ea959d4d640fff884136487bd05",
        "copied_model_source_commit": "fcf620177d8e9b1575e5e0ed590d4aae1a9cb3b3",
        "train_templates": TRAIN_TEMPLATES, "validation_templates": VALIDATION_TEMPLATES,
        "row_hashes": {group: {name: digest(rows) for name, rows in arms.items()} for group, arms in data.items()},
        "manifests": {"train": development_manifest(train_examples), "validation": development_manifest(val_examples)},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_measure": "Paired exact-target change on old_failing comparison wording, with new_disjoint comparison transfer and read/explain side effects reported.",
        "decision": "One fixed final-step checkpoint per arm/seed. No validation selection, no post-hoc variants. A negative result only folds this exact six-template/all-kind 600-world/300-step recipe if supported across seeds; weak mixed evidence remains inconclusive.",
        "ci": "Percentile 95% paired bootstrap over validation task IDs, keeping both wording families and all three training seeds together in each cluster. Conditional on these three fixed training seeds; not a seed-population CI.",
        "scope": "CPU only, 2 intraop threads/1 interop thread; no GPU/network/spend/Index/test data/teacher API. Synthetic iterative development only. Targets select kind/field/entity/order, not answer values; no production planner or interface changes."}
    write_json(out / "frozen_rows.json", data)
    write_json(out / "preregistration.json", registration)
    print(json.dumps({"prepared": str(out), "row_hashes": registration["row_hashes"]}), flush=True)


def paired_summary(observations, family, kind):
    import numpy as np
    samples = []
    improved = regressed = before_n = after_n = 0
    for seed in CONFIG["model_seeds"]:
        seed_obs = observations[str(seed)]
        families = VALIDATION_TEMPLATES if family == "pooled" else [family]
        for f in families:
            for b, a in zip(seed_obs["baseline"][f], seed_obs["all_task_diversity"][f]):
                if kind != "all" and b["target"]["kind"] != kind:
                    continue
                assert b["id"] == a["id"] and b["target"] == a["target"]
                samples.append((b["id"], int(a["correct"]) - int(b["correct"])))
                before_n += b["correct"]
                after_n += a["correct"]
                improved += not b["correct"] and a["correct"]
                regressed += b["correct"] and not a["correct"]
    clusters = {}
    for task_id, delta in samples:
        clusters.setdefault(task_id, []).append(delta)
    values = np.array([sum(ds) / len(ds) for ds in clusters.values()])
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    draws = values[rng.integers(0, len(values), size=(CONFIG["bootstrap_replicates"], len(values)))].mean(1)
    return {"count": len(samples), "task_clusters": len(clusters), "baseline_correct": int(before_n),
        "augmentation_correct": int(after_n), "improved": int(improved), "regressed": int(regressed),
        "delta_accuracy": float(values.mean()), "task_cluster_ci95": np.quantile(draws, [.025, .975]).tolist()}


def run(out):
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 required")
    registration = json.loads((out / "preregistration.json").read_text())
    assert registration["config"] == CONFIG
    assert registration["source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data = json.loads((out / "frozen_rows.json").read_text())
    assert registration["row_hashes"] == {group: {name: digest(rows) for name, rows in arms.items()} for group, arms in data.items()}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    results, observations = {}, {}
    for seed in CONFIG["model_seeds"]:
        results[str(seed)], observations[str(seed)] = {}, {}
        for arm, rows in data["train"].items():
            torch.manual_seed(seed)
            model = QueryTargetSSM(CONFIG["width"]).cpu()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
            rng = random.Random(seed)
            arm_start, history = time.monotonic(), []
            for step in range(CONFIG["steps"]):
                if time.monotonic() - started >= CONFIG["max_runtime_seconds"]:
                    raise TimeoutError("Total preregistered CPU runtime exceeded")
                indices = rng.sample(range(len(rows)), CONFIG["batch_size"])
                q, c, owners, labels, count = tensors([rows[i] for i in indices])
                model.train()
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(q, c, owners).reshape(len(indices), count), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if (step + 1) % 100 == 0:
                    history.append({"step": step + 1, "loss": loss.item()})
            train_seconds = time.monotonic() - arm_start
            checkpoint = out / f"seed-{seed}-{arm}.pt"
            torch.save({"state_dict": model.state_dict(), "width": CONFIG["width"], "seed": seed}, checkpoint)
            restored = QueryTargetSSM(CONFIG["width"]).cpu()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
            train_summary, _ = evaluate(restored, rows)
            summaries, predictions = {}, {}
            for family, val_rows in data["validation"].items():
                summaries[family], predictions[family] = evaluate(restored, val_rows)
                _, relabeled = evaluate(restored, permute_ids(val_rows))
                assert predictions[family] == relabeled
            results[str(seed)][arm] = {"training_seconds": train_seconds, "history": history,
                "train": train_summary, "validation": summaries,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            observations[str(seed)][arm] = predictions
            print(json.dumps({"seed": seed, "arm": arm, "seconds": train_seconds, "validation": summaries}), flush=True)
    paired = {family: {kind: paired_summary(observations, family, kind) for kind in ("all", "read", "explain", "compare")}
              for family in (*VALIDATION_TEMPLATES, "pooled")}
    receipt = {"schema": "ssdm.query_template_transfer_cpu.v1", "preregistration": registration,
        "parameters": sum(p.numel() for p in model.parameters()), "results": results, "paired": paired,
        "wall_seconds": time.monotonic() - started, "torch": torch.__version__, "python": sys.version,
        "id_permutation_checks": "Exact semantic observations matched for both families in all 6 restored checkpoints.",
        "cost_usd": 0}
    write_json(out / "observations.json", observations)
    write_json(out / "receipt.json", receipt)
    print(json.dumps({"wall_seconds": receipt["wall_seconds"], "paired": paired}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.out)

