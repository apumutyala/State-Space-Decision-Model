"""Single paired-value supervision intervention on fixed six-template training.

Sample original world IDs exactly as the saved baseline. A comparison world
alternates original/swap on successive exposures. Each operation retains 200
uniformly sampled base groups; pairs share their group's probability mass.
No extra samples, steps, loss reweighting, wording or model changes occur.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import torch
from torch import nn

from . import typed_emission_ssm as core
from .development_data import comparison_swap_counterfactual, generate_development_tasks

BASE = "0f4b8ad2ea717c79858ddfbc4deb55c9b71940af"
CORE_SHA = "93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6"
BASELINE_DATA_SHA = "e3cec785d4abcbf796196a1dfc36fc3be2b99ad89b732a56724cc8b495e94505"
BASELINE_PREDICTIONS_SHA = "edbcb8393949cbefbf0d25e49cb1b11f2de2b2d40917ba39df7598decb1489e4"
CONTROL_SEED = 8421


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_batch(train, swaps, indices, visits):
    """Training-only augmentation. No label/value-dependent member selection."""
    batch, members = [], []
    for index in indices:
        original = train[index]
        swap = swaps.get(original["id"])
        member = "swap" if swap is not None and visits[index] % 2 else "original"
        batch.append(swap if member == "swap" else original)
        members.append(member)
        visits[index] += 1
    return batch, members


def schedule_proof(train, swaps, seed):
    rng = random.Random(seed)
    baseline_rng = random.Random(seed)
    visits = [0] * len(train)
    counts = {k: 0 for k in core.KINDS}
    members = {"original_compare": 0, "swap_compare": 0}
    originals = {r["id"]: 0 for r in train if r["id"] in swaps}
    counterfactuals = dict(originals)
    for _ in range(core.CONFIG["steps"]):
        indices = rng.sample(range(len(train)), core.CONFIG["batch_size"])
        control = baseline_rng.sample(train, core.CONFIG["batch_size"])
        assert [train[i]["id"] for i in indices] == [r["id"] for r in control]
        batch, chosen = select_batch(train, swaps, indices, visits)
        for i, row, member in zip(indices, batch, chosen):
            kind = core.gold_target(train[i]["evaluation_only"]["program"])["kind"]
            assert core.gold_target(row["evaluation_only"]["program"])["kind"] == kind
            counts[kind] += 1
            if kind == "compare":
                members[member + "_compare"] += 1
                (counterfactuals if member == "swap" else originals)[train[i]["id"]] += 1
    assert all(abs(originals[k] - counterfactuals[k]) <= 1 for k in originals)
    return {"kind_counts_identical_to_baseline": counts, "comparison_member_counts": members,
            "max_per_world_member_count_difference": max(abs(originals[k] - counterfactuals[k]) for k in originals),
            "base_world_id_draws_identical_to_baseline": True}


def prepare(out, baseline_dir):
    assert not out.exists()
    source = Path(core.__file__)
    assert core.sha(source) == CORE_SHA
    original_bytes = gzip.decompress((baseline_dir / "frozen_rows.json.gz").read_bytes())
    predictions_bytes = gzip.decompress((baseline_dir / "predictions.json.gz").read_bytes())
    assert hashlib.sha256(original_bytes).hexdigest() == BASELINE_DATA_SHA
    assert hashlib.sha256(predictions_bytes).hexdigest() == BASELINE_PREDICTIONS_SHA
    baseline = json.loads(original_bytes)
    previous = json.loads((baseline_dir / "registration.json").read_text())
    assert previous["config"] == core.CONFIG
    train, validation = copy.deepcopy(baseline["train"]), copy.deepcopy(baseline["development"])
    assert len(train) == 600 and len(validation) == 360
    assert all(r["source_split"] == "development_validation" for r in validation)
    kinds = [core.gold_target(r["evaluation_only"]["program"])["kind"] for r in train]
    assert {k: kinds.count(k) for k in core.KINDS} == {k: 200 for k in core.KINDS}
    swaps = {}
    examples = generate_development_tasks(9101, 600, split="development_train")
    for original, example, kind in zip(train, examples, kinds):
        old = core.row(example, example["task_id"], "original")
        assert old["evaluation_only"] == original["evaluation_only"]
        state = copy.deepcopy(original["state"])
        state["task_input"] = old["state"]["task_input"]
        assert state == old["state"]
        if kind == "compare":
            authored = comparison_swap_counterfactual(example)
            swapped = core.row(authored, example["task_id"], "value_swap_train")
            swapped["state"]["task_input"] = original["state"]["task_input"]
            swapped["paired_original_id"] = original["id"]
            assert swapped["state"]["task_input"] == original["state"]["task_input"]
            original_target = core.gold_target(original["evaluation_only"]["program"])
            swapped_target = core.gold_target(swapped["evaluation_only"]["program"])
            assert {k:v for k,v in original_target.items() if k != "relation"} == {k:v for k,v in swapped_target.items() if k != "relation"}
            checked = core.check_program(swapped["evaluation_only"]["program"], swapped["evaluation_only"]["host"], swapped["evaluation_only"]["catalog"])
            assert checked.status == "accepted"
            swaps[original["id"]] = swapped
    assert len(swaps) == 200 and train == baseline["train"] and validation == baseline["development"]
    out.mkdir(parents=True)
    core.write(out / "frozen_rows.json", {"train": train, "paired_comparison_train": swaps, "development": validation})
    (out / "baseline_predictions.json").write_bytes(predictions_bytes)
    shutil.copyfile(baseline_dir / "receipt.json", out / "baseline_receipt.json")
    paths = [source, Path(__file__), *[source.with_name(name) for name in
        ("development_data.py", "kernel.py", "realize.py", "parent_bridge.py", "types.py")]]
    schedule = {str(seed): schedule_proof(train, swaps, seed) for seed in core.CONFIG["seeds"]}
    registration = {"config": core.CONFIG, "source_hashes": {p.name: core.sha(p) for p in paths},
        "source": str(source.resolve()), "base_commit": BASE, "frozen_data_sha256": core.sha(out / "frozen_rows.json"),
        "baseline_data_sha256": BASELINE_DATA_SHA, "baseline_predictions_sha256": BASELINE_PREDICTIONS_SHA,
        "baseline_source_commit": "1a09d189635711bd79319ef8eb334ed587e34c2e", "baseline_registration": previous,
        "baseline_clarification": "Six-template wording-diverse control: 52.13% whole, 23.89% compare, 111/159 relation flips. The folded two-template 24/159 condition is not the control.",
        "unchanged_base_training_rows_sha256": digest(train), "unchanged_validation_rows_sha256": digest(validation),
        "base_training_rows_exactly_reused": True, "validation_rows_exactly_reused": True,
        "only_intervention": "On repeated selections of each comparison base world, alternate original and authored selected-field value-swap member; unchanged wording, model/features/scales, optimizer, sampler base IDs, losses and 300 steps",
        "balance": "200 base groups per operation. Comparison pair shares one group's mass; uniform base sampler is identical to the control, so finite per-kind exposure counts are exactly matched. No extra batch members or kind loss reweighting.",
        "training_schedule_proof": schedule,
        "controls": {"seed": CONTROL_SEED, "static_output": "Reuse learned original complete semantic target for both paired worlds; recompile/check/render against each current state",
                     "random_relation": "Preserve learned operation/structure; choose independent uniform relation for each comparison member; recompile/check/render. Report flips AND both gold relations/complete answers correct."},
        "primary_metrics": "Comparison required relation-flip fidelity and both-answer correctness, with complete-task/acceptance and read/explain side effects",
        "scope": "One frozen train-only paired-value supervision condition on six-template baseline; iterative synthetic development, not official holdout or parent retention; no tuning", "cost_usd": 0}
    core.write(out / "registration.json", registration)
    print(json.dumps({"prepared": str(out), "schedule": schedule}), flush=True)


def run(out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    registration = json.loads((out / "registration.json").read_text())
    assert registration["config"] == core.CONFIG
    for name, expected in registration["source_hashes"].items():
        assert core.sha(Path(__file__).with_name(name)) == expected
    assert core.sha(out / "frozen_rows.json") == registration["frozen_data_sha256"]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    data = json.loads((out / "frozen_rows.json").read_text())
    train, swaps = data["train"], data["paired_comparison_train"]
    assert digest(train) == registration["unchanged_base_training_rows_sha256"]
    assert digest(data["development"]) == registration["unchanged_validation_rows_sha256"]
    results, predictions = {}, {}
    for seed in core.CONFIG["seeds"]:
        torch.manual_seed(seed)
        model = core.TypedEmissionSSM().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=core.CONFIG["lr"])
        rng = random.Random(seed)
        visits = [0] * len(train)
        history = []
        actual_counts = {k: 0 for k in core.KINDS}
        actual_members = {"original_compare": 0, "swap_compare": 0}
        for step in range(core.CONFIG["steps"]):
            if time.monotonic() - started >= core.CONFIG["max_seconds"]:
                raise TimeoutError("Frozen total CPU runtime exceeded")
            indices = rng.sample(range(len(train)), core.CONFIG["batch_size"])
            batch, members = select_batch(train, swaps, indices, visits)
            gold = [core.gold_target(r["evaluation_only"]["program"]) for r in batch]
            for g, member in zip(gold, members):
                actual_counts[g["kind"]] += 1
                if g["kind"] == "compare":
                    actual_members[member + "_compare"] += 1
            q, components, part_indices, numeric, kinds, candidates = core.tensors([r["state"] for r in batch])
            structures = [{k:v for k,v in g.items() if k != "relation"} for g in gold]
            labels = torch.tensor([next(i for i,c in enumerate(cs) if c["target"] == g) for cs,g in zip(candidates,structures)])
            op_labels = torch.tensor([core.KINDS.index(g["kind"]) for g in gold])
            compare = op_labels.eq(2)
            relation_labels = torch.tensor([core.RELATIONS.index(g.get("relation", core.RELATIONS[0])) for g in gold])
            model.train()
            optimizer.zero_grad()
            operation, scores, relations = model(q, components, part_indices, numeric)
            op_loss = nn.functional.cross_entropy(operation, op_labels)
            target_loss = nn.functional.cross_entropy(scores.masked_fill(kinds.ne(op_labels[:,None]), -torch.inf), labels)
            rel_loss = nn.functional.cross_entropy(relations[torch.arange(len(batch)), labels][compare], relation_labels[compare]) if compare.any() else scores.sum() * 0
            loss = op_loss + target_loss + rel_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), core.CONFIG["clip_norm"])
            optimizer.step()
            if (step+1) % 100 == 0:
                history.append({"step":step+1,"operation_loss":op_loss.item(),"target_loss":target_loss.item(),"relation_loss":rel_loss.item()})
        proof = registration["training_schedule_proof"][str(seed)]
        assert actual_counts == proof["kind_counts_identical_to_baseline"]
        assert actual_members == proof["comparison_member_counts"]
        checkpoint = out / f"seed-{seed}.pt"
        torch.save({"state_dict":model.state_dict(),"seed":seed,"config":core.CONFIG}, checkpoint)
        restored = core.TypedEmissionSSM().cpu()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        metrics, observations = core.evaluate(restored, data["development"])
        train_metrics, _ = core.evaluate(restored, train)
        predictions[str(seed)] = observations
        results[str(seed)] = {"metrics":metrics,"paired":core.paired(observations),"train_original":train_metrics,
            "history":history,"actual_kind_counts":actual_counts,"actual_comparison_member_counts":actual_members,
            "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":core.sha(checkpoint)}
        core.write(out / "predictions.json", predictions)
        core.write(out / "partial_results.json", results)
        print(json.dumps({"seed":seed,"metrics":metrics,"elapsed_seconds":time.monotonic()-started}),flush=True)
    qa = core.adversarial(data["development"])
    elapsed = time.monotonic()-started
    assert elapsed <= core.CONFIG["max_seconds"]
    receipt = {"registration":registration,"results":results,
        "pooled":core.summary([r for rows in predictions.values() for r in rows]),"adversarial":qa,
        "wall_seconds":elapsed,"parameters":sum(p.numel() for p in model.parameters()),
        "runtime":{"python":sys.version,"torch":torch.__version__,"device":"cpu","threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads()},
        "raw_predictions_sha256":core.sha(out / "predictions.json"),"data_sha256":core.sha(out / "frozen_rows.json"),
        "scope":registration["scope"],"cost_usd":0}
    core.write(out / "receipt.json",receipt)
    print(json.dumps({"pooled":receipt["pooled"],"wall_seconds":elapsed}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","run"))
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--baseline",type=Path)
    args=parser.parse_args()
    prepare(args.out,args.baseline) if args.mode=="prepare" else run(args.out)
