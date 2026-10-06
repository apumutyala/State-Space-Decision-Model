"""Matched CPU query-target experiment with comparison-only training paraphrases.

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


CONFIG = {"seed": 20261006, "train_seed": 20261006, "validation_seed": 20261008,
          "train_count": 600, "validation_count": 180, "steps": 300, "width": 32,
          "batch_size": 32, "lr": 0.003, "cpu_threads": 2, "device": "cpu",
          "max_training_seconds_per_arm": 120}
TRAIN_COMPARE = (
    "Compare the October {field} of {left} and {right}.",
    "In October 2026, compare {left} and {right} on {field}.",
    "Assess how {left}'s October {field} compares to {right}'s.",
    "Considering {field} for October 2026, compare {left} against {right}.",
)
VALIDATION_COMPARE = "At October 2026, how does {left}'s {field} compare with {right}'s?"


def comparison_text(example: dict[str, Any], template: str) -> str:
    """Dataset authoring only; the model still receives plain question text."""
    training = example["training_only"]
    reads = [n for n in training["program"]["nodes"] if n["op"] == "ReadFact"]
    records = {r["record_ref"]: r for r in training["catalog"]["records"]}
    return template.format(left=records[reads[0]["record_ref"]]["entity_ref"],
                           right=records[reads[1]["record_ref"]]["entity_ref"],
                           field=reads[0]["field"].replace("_", " "))


def replace_question(row: dict[str, Any], text: str) -> dict[str, Any]:
    row = copy.deepcopy(row)
    state = json.loads(row["request"]["state"])
    state["task_input"] = text
    row["request"]["state"] = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return row


def rows_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def train_arm(rows: list[dict[str, Any]], out: Path, arm: str):
    torch.manual_seed(CONFIG["seed"])
    model = QueryTargetSSM(CONFIG["width"]).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = random.Random(CONFIG["seed"])
    history = []
    started = time.monotonic()
    for step in range(CONFIG["steps"]):
        if time.monotonic() - started > CONFIG["max_training_seconds_per_arm"]:
            raise TimeoutError(f"{arm} exceeded preregistered training bound")
        indices = rng.sample(range(len(rows)), CONFIG["batch_size"])
        query, candidate, owners, labels, count = tensors([rows[i] for i in indices])
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(query, candidate, owners).reshape(len(indices), count), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            history.append({"step": step + 1, "loss": loss.item()})
            print(json.dumps({"arm": arm, **history[-1]}), flush=True)
    elapsed = time.monotonic() - started
    checkpoint = out / (arm + ".pt")
    torch.save({"state_dict": model.state_dict(), "width": CONFIG["width"], "seed": CONFIG["seed"]}, checkpoint)
    restored = QueryTargetSSM(CONFIG["width"]).cpu()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    return restored, {"training_seconds": elapsed, "history": history,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def run(out: Path) -> dict[str, Any]:
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 required before interpreter startup")
    if out.exists():
        raise ValueError("fresh output path required")
    out.mkdir(parents=True)
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train_examples = generate_development_tasks(CONFIG["train_seed"], CONFIG["train_count"], split="development_train")
    validation_examples = generate_development_tasks(CONFIG["validation_seed"], CONFIG["validation_count"], split="development_validation")
    baseline_rows = [make_row(e) for e in train_examples]
    curriculum_rows = [replace_question(row, comparison_text(e, TRAIN_COMPARE[(i // 3) % len(TRAIN_COMPARE)]))
                       if e["training_only"]["kind"] == "compare" else copy.deepcopy(row)
                       for i, (e, row) in enumerate(zip(train_examples, baseline_rows))]
    validation = [replace_question(make_row(e), comparison_text(e, VALIDATION_COMPARE))
                  if e["training_only"]["kind"] == "compare" else make_row(e)
                  for e in validation_examples]
    assert all(before["evaluation_only"] == after["evaluation_only"]
               for before, after in zip(baseline_rows, curriculum_rows))
    assert all(before == after for e, before, after in zip(train_examples, baseline_rows, curriculum_rows)
               if e["training_only"]["kind"] != "compare")
    torch.manual_seed(CONFIG["seed"])
    random_model = QueryTargetSSM(CONFIG["width"]).cpu()
    random_summary, random_observations = evaluate(random_model, validation)
    result, observations = {}, {"random_initialization": random_observations}
    for arm, train in (("baseline", baseline_rows), ("comparison_paraphrases", curriculum_rows)):
        model, metadata = train_arm(train, out, arm)
        train_summary, _ = evaluate(model, train)
        validation_summary, predictions = evaluate(model, validation)
        _, permuted = evaluate(model, permute_ids(validation))
        assert predictions == permuted
        result[arm] = {**metadata, "train": train_summary, "validation": validation_summary,
                       "id_permutation_agreement": len(permuted)}
        observations[arm] = predictions
    before, after = observations["baseline"], observations["comparison_paraphrases"]
    paired = {kind: {"improved": sum(not b["correct"] and a["correct"] for b, a in zip(before, after)
                                      if kind == "all" or b["target"]["kind"] == kind),
                     "regressed": sum(b["correct"] and not a["correct"] for b, a in zip(before, after)
                                      if kind == "all" or b["target"]["kind"] == kind)}
              for kind in ("all", "read", "explain", "compare")}
    receipt = {"schema": "ssdm.comparison_paraphrase_cpu.v1", "config": CONFIG,
        "source_provenance": "Model/target recipe copied from local committed fcf620177d8e9b1575e5e0ed590d4aae1a9cb3b3; no dependency on that branch.",
        "base_commit": "4733f785e5284ea959d4d640fff884136487bd05",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in random_model.parameters()), "torch": torch.__version__,
        "train_comparison_templates": TRAIN_COMPARE, "validation_comparison_template": VALIDATION_COMPARE,
        "manifests": {"train_worlds": development_manifest(train_examples), "validation_worlds": development_manifest(validation_examples)},
        "row_hashes": {"baseline_train": rows_hash(baseline_rows), "curriculum_train": rows_hash(curriculum_rows),
                       "validation": rows_hash(validation)},
        "random_initialization_validation": random_summary, "results": result, "paired_changes": paired,
        "scope": "One final-step comparison curriculum vs matched training baseline; iterative synthetic development, no Index/test/teacher/GPU/network/paid compute; no validation selection. Names/ranges disjoint across train/validation; fresh validation seed and comparison template relative to the source probe. Temporal scope is constant; values are not model inputs; no planner release integration."}
    for name, data in (("receipt.json", receipt), ("observations.json", observations)):
        (out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out)["results"], indent=2))

