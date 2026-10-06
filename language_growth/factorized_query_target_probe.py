"""Single CPU development condition: operation head plus structured target scorer.

PR #75 is read-only data/evidence, never an imported implementation dependency.
The query encoder is its width-32 diagonal byte recurrence. Candidate structure
is enumerated from permitted request records, never from evaluator bindings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

KINDS = ("read", "explain", "compare")
CONFIG = {"model_seeds": [20261006, 20261016, 20261026], "width": 32,
          "steps": 300, "batch_size": 32, "lr": .003, "clip_norm": 1.,
          "cpu_threads": 2, "max_runtime_seconds": 180,
          "bootstrap_seed": 3197, "bootstrap_replicates": 2000,
          "loss": "operation CE plus conditional structured-target CE, equal weight",
          "inference": "argmax operation, then argmax structured score within operation"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project(request):
    """No evaluation metadata or task parsing; match complete enumerated syntax."""
    state = json.loads(request["state"])
    syntax = {}
    for kind in KINDS:
        for left in state["records"]:
            for field in sorted(left["fields"]):
                scope = left["fields"][field]["temporal_scope"]
                field_text, name = field.replace("_", " "), left["entity_ref"]
                if kind != "compare":
                    text = f"{kind.title()} {field_text} for {name} at {scope}."
                    syntax[text] = (KINDS.index(kind), (field_text, name, "", scope))
                else:
                    for right in state["records"]:
                        if right["record_ref"] == left["record_ref"] or field not in right["fields"]:
                            continue
                        right_name = right["entity_ref"]
                        text = f"Compare {field_text} of {name} then {right_name} at {scope}."
                        syntax[text] = (2, (field_text, name, right_name, scope))
    question, = request["questions"].values()
    choices = sorted(question["criteria"].items(), key=lambda item: item[1])
    assert len(syntax) == len(choices) and set(syntax) == {text for _, text in choices}
    return state["task_input"], [oid for oid, _ in choices], [syntax[text] for _, text in choices]


def byte_batch(texts):
    encoded = [list(text.encode("utf-8")) for text in texts]
    out = torch.zeros((len(texts), max(1, max(map(len, encoded)))), dtype=torch.long)
    for index, row in enumerate(encoded):
        if row:
            out[index, :len(row)] = torch.tensor(row) + 1
    return out


class FactorizedQueryTargetSSM(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.input = nn.Linear(width, width)
        self.decay_logit = nn.Parameter(torch.logit(torch.linspace(.5, .999, width)))
        self.operation = nn.Linear(width * 2, 3)
        # q, field, left, right, scope, and q times each component: 9*2w.
        self.scorer = nn.Sequential(nn.Linear(width * 18, 32), nn.Tanh(), nn.Linear(32, 1))

    def encode(self, ids):
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

    def forward(self, queries, components, owners):
        query = self.encode(queries)
        q = query[owners]
        parts = self.encode(components).reshape(len(owners), 4, self.width * 2).unbind(1)
        score = self.scorer(torch.cat((q, *parts, *(q * part for part in parts)), 1)).squeeze(-1)
        return self.operation(query), score


def tensors(rows):
    projections = [project(row["request"]) for row in rows]
    counts = {len(item[1]) for item in projections}
    assert len(counts) == 1
    count = counts.pop()
    query = byte_batch([item[0] for item in projections])
    components = byte_batch([part for item in projections for _, parts in item[2] for part in parts])
    kinds = torch.tensor([[kind for kind, _ in item[2]] for item in projections])
    owners = torch.arange(len(rows)).repeat_interleave(count)
    # Labels are used only for supervised losses or offline evaluation.
    labels = torch.tensor([item[1].index(row["evaluation_only"]["option_id"])
                           for row, item in zip(rows, projections)])
    return query, components, owners, kinds, labels, count


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    observations = []
    for start in range(0, len(rows), CONFIG["batch_size"]):
        batch = rows[start:start + CONFIG["batch_size"]]
        query, components, owners, kinds, _, count = tensors(batch)
        operations, scores = model(query, components, owners)
        selected_operation = operations.argmax(1)
        scores = scores.reshape(len(batch), count).masked_fill(kinds.ne(selected_operation[:, None]), -torch.inf)
        for row, selected, operation in zip(batch, scores.argmax(1).tolist(), selected_operation.tolist()):
            _, ids, _ = project(row["request"])
            predicted = row["evaluation_only"]["binding"][ids[selected]]
            gold = row["evaluation_only"]["target"]
            assert predicted["kind"] == KINDS[operation]
            observations.append({"id": row["id"], "predicted": predicted, "target": gold,
                                 "correct": predicted == gold})
    return summarize(observations), observations


def summarize(observations):
    return {"count": len(observations), "exact_correct": sum(row["correct"] for row in observations),
            "operation_correct": sum(row["predicted"]["kind"] == row["target"]["kind"] for row in observations),
            "by_kind": {kind: {"count": sum(row["target"]["kind"] == kind for row in observations),
                                "correct": sum(row["correct"] and row["target"]["kind"] == kind for row in observations),
                                "operation_correct": sum(row["predicted"]["kind"] == kind and row["target"]["kind"] == kind for row in observations)} for kind in KINDS}}


def permute_ids(rows):
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


def paired(reference, observations, arm, family, kind, measure="exact"):
    clusters = {}
    before = after = improved = regressed = count = 0
    families = ("old_failing", "new_disjoint") if family == "pooled" else (family,)
    for seed in CONFIG["model_seeds"]:
        for name in families:
            b = {row["id"]: row for row in reference[str(seed)][arm][name]}
            a = {row["id"]: row for row in observations[str(seed)][name]}
            assert set(a) == set(b) and len(a) == len(observations[str(seed)][name])
            for ident in sorted(a):
                x, y = b[ident], a[ident]
                assert x["target"] == y["target"]
                if kind != "all" and x["target"]["kind"] != kind:
                    continue
                bx = x["correct"] if measure == "exact" else x["predicted"]["kind"] == x["target"]["kind"]
                ay = y["correct"] if measure == "exact" else y["predicted"]["kind"] == y["target"]["kind"]
                before += bx
                after += ay
                improved += not bx and ay
                regressed += bx and not ay
                count += 1
                clusters.setdefault(ident, []).append(int(ay) - int(bx))
    values = np.array([np.mean(ds) for ds in clusters.values()])
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    draws = rng.choice(values, (CONFIG["bootstrap_replicates"], len(values)), replace=True).mean(1)
    return {"count": count, "task_clusters": len(values), "before_correct": int(before), "after_correct": int(after),
            "improved": int(improved), "regressed": int(regressed), "delta_accuracy": float(values.mean()),
            "task_cluster_ci95": np.quantile(draws, [.025, .975]).tolist()}


def prepare(out, reference_data, reference_observations, reference_registration):
    assert not out.exists()
    out.mkdir(parents=True)
    data = json.loads(reference_data.read_text())
    ref = json.loads(reference_registration.read_text())
    hashes = {group: {name: digest(rows) for name, rows in arms.items()} for group, arms in data.items()}
    assert hashes == ref["row_hashes"]
    assert len(data["train"]["all_task_diversity"]) == 600
    # Validate complete structure without inspecting target correctness.
    for rows in [data["train"]["all_task_diversity"], *data["validation"].values()]:
        for row in rows:
            project(row["request"])
    shutil.copyfile(reference_data, out / "frozen_rows.json")
    shutil.copyfile(reference_observations, out / "reference_observations.json")
    registration = {"schema": "ssdm.factorized_query_target_preregistration.v1", "config": CONFIG,
                    "base_commit": "4733f785e5284ea959d4d640fff884136487bd05", "row_hashes": hashes,
                    "reference_data_sha256": file_hash(reference_data),
                    "reference_observations_sha256": file_hash(reference_observations),
                    "reference_registration": ref, "source_sha256": file_hash(__file__),
                    "parameters": sum(p.numel() for p in FactorizedQueryTargetSSM().parameters()),
                    "reference_parameters": 27937,
                    "decision": "One architecture, three fixed seeds, final step only; no validation tuning. Report both raw baseline and matched all_task_diversity reference pairs.",
                    "scope": "Synthetic iterative development, CPU <=2 threads, <=180 seconds, $0; no GPU/network/Index/held-out/teacher API or production changes.",
                    "ci": "95% percentile bootstrap over task IDs, keeping all seeds and both pooled wording renderings together; conditional on fixed seeds/families."}
    write_json(out / "preregistration.json", registration)
    print(json.dumps({"prepared": str(out), "parameters": registration["parameters"]}), flush=True)


def run(out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    registration = json.loads((out / "preregistration.json").read_text())
    assert registration["config"] == CONFIG and registration["source_sha256"] == file_hash(__file__)
    assert registration["reference_data_sha256"] == file_hash(out / "frozen_rows.json")
    assert registration["reference_observations_sha256"] == file_hash(out / "reference_observations.json")
    data = json.loads((out / "frozen_rows.json").read_text())
    reference = json.loads((out / "reference_observations.json").read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    observations, results = {}, {}
    rows = data["train"]["all_task_diversity"]
    for seed in CONFIG["model_seeds"]:
        torch.manual_seed(seed)
        model = FactorizedQueryTargetSSM().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        rng = random.Random(seed)
        history, seed_start = [], time.monotonic()
        for step in range(CONFIG["steps"]):
            if time.monotonic() - started >= CONFIG["max_runtime_seconds"]:
                raise TimeoutError("Total preregistered CPU runtime exceeded")
            indices = rng.sample(range(len(rows)), CONFIG["batch_size"])
            q, components, owners, kinds, labels, count = tensors([rows[i] for i in indices])
            gold_kind = kinds.gather(1, labels[:, None]).squeeze(1)
            model.train()
            optimizer.zero_grad()
            operation, score = model(q, components, owners)
            conditional = score.reshape(len(indices), count).masked_fill(kinds.ne(gold_kind[:, None]), -torch.inf)
            operation_loss = nn.functional.cross_entropy(operation, gold_kind)
            target_loss = nn.functional.cross_entropy(conditional, labels)
            loss = operation_loss + target_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"])
            optimizer.step()
            if (step + 1) % 100 == 0:
                history.append({"step": step + 1, "operation_loss": operation_loss.item(), "target_loss": target_loss.item()})
        training_seconds = time.monotonic() - seed_start
        checkpoint = out / f"seed-{seed}-factorized.pt"
        torch.save({"state_dict": model.state_dict(), "seed": seed, "config": CONFIG}, checkpoint)
        restored = FactorizedQueryTargetSSM().cpu()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        train_summary, _ = evaluate(restored, rows)
        summaries, predictions = {}, {}
        for family, val_rows in data["validation"].items():
            summaries[family], predictions[family] = evaluate(restored, val_rows)
            _, relabeled = evaluate(restored, permute_ids(val_rows))
            assert predictions[family] == relabeled
        observations[str(seed)] = predictions
        results[str(seed)] = {"training_seconds": training_seconds, "history": history,
                              "train": train_summary, "validation": summaries,
                              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint)}
        print(json.dumps({"seed": seed, "seconds": training_seconds, "validation": summaries}), flush=True)
        write_json(out / "observations.json", observations)
        write_json(out / "partial_results.json", results)
    pairs = {arm: {measure: {family: {kind: paired(reference, observations, arm, family, kind, measure)
                                    for kind in ("all", *KINDS)} for family in ("old_failing", "new_disjoint", "pooled")}
                   for measure in ("exact", "operation")} for arm in ("baseline", "all_task_diversity")}
    elapsed = time.monotonic() - started
    assert elapsed < CONFIG["max_runtime_seconds"]
    receipt = {"schema": "ssdm.factorized_query_target_cpu.v1", "preregistration": registration,
               "parameters": sum(p.numel() for p in model.parameters()), "results": results, "paired": pairs,
               "wall_seconds": elapsed, "torch": torch.__version__, "python": sys.version,
               "id_permutation_checks": "Exact semantic observations matched for both families in all three restored checkpoints.", "cost_usd": 0}
    write_json(out / "receipt.json", receipt)
    print(json.dumps({"wall_seconds": elapsed, "pooled_vs_diversity": pairs["all_task_diversity"]["exact"]["pooled"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--reference-observations", type=Path)
    parser.add_argument("--reference-registration", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out, args.reference_data, args.reference_observations, args.reference_registration)
    else:
        run(args.out)
