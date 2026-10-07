"""One frozen train-to-validation development transfer; unchanged PR #80 model.

Only cohort construction changes. The copied core is byte-identical to the
previous committed implementation; it still receives visible states only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import typed_emission_ssm as core
from .development_data import (
    comparison_swap_counterfactual, development_split_spec, generate_development_tasks,
)

BASE = "0f4b8ad2ea717c79858ddfbc4deb55c9b71940af"
CORE_SHA = "93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6"
TRAIN_SPLIT = "development_train"
EVAL_SPLIT = "development_validation"
ALTERNATE_TEMPLATES = {
    "read": "For October 2026, state the {field} of {entity}.",
    "explain": "For {entity}, provide the checked derivation of the {field}.",
}


def paired_rows(tasks):
    rows = []
    for task in tasks:
        original = core.row(task, task["task_id"], "original")
        original["source_split"] = EVAL_SPLIT
        rows.append(original)
        if task["training_only"]["kind"] == "compare":
            partner = core.row(comparison_swap_counterfactual(task), task["task_id"], "value_swap")
            assert partner["state"]["task_input"] == original["state"]["task_input"]
        else:
            partner = copy.deepcopy(original)
            partner["id"] += ":wording"
            partner["member"] = "wording"
            target = core.gold_target(partner["evaluation_only"]["program"])
            record = next(r for r in partner["state"]["records"] if r["record_ref"] == target["left"])
            partner["state"]["task_input"] = ALTERNATE_TEMPLATES[target["kind"]].format(
                entity=record["entity_ref"], field=target["field"].replace("_", " "))
            assert partner["state"]["task_input"] != original["state"]["task_input"]
        partner["source_split"] = EVAL_SPLIT
        rows.append(partner)
    return rows


def prepare(out):
    assert not out.exists()
    source = Path(core.__file__)
    assert core.sha(source) == CORE_SHA
    train_spec = development_split_spec(TRAIN_SPLIT)
    eval_spec = development_split_spec(EVAL_SPLIT)
    assert not set(train_spec["names"]) & set(eval_spec["names"])
    assert train_spec["warranty_range"][1] < eval_spec["warranty_range"][0]
    assert train_spec["price_cents_range"][1] < eval_spec["price_cents_range"][0]
    out.mkdir(parents=True)
    # Identical training seed, rows, ordering and sampler as the original run.
    train = [core.row(t, t["task_id"], "original") for t in generate_development_tasks(
        core.CONFIG["train_seed"], core.CONFIG["train_count"], split=TRAIN_SPLIT)]
    validation = paired_rows(generate_development_tasks(
        core.CONFIG["development_seed"], core.CONFIG["development_clusters"], split=EVAL_SPLIT))
    core.write(out / "frozen_rows.json", {"train": train, "development": validation})
    paths = [source, Path(__file__), *[source.with_name(name) for name in
        ("development_data.py", "kernel.py", "realize.py", "parent_bridge.py", "types.py")]]
    registration = {"config": core.CONFIG, "source_hashes": {p.name: core.sha(p) for p in paths},
        "source": str(source.resolve()), "base_commit": BASE,
        "frozen_data_sha256": core.sha(out / "frozen_rows.json"),
        "paired_task_clusters": core.CONFIG["development_clusters"],
        "implementation_source_commit": "c0db91df4b28fbb3cbd9fe05b210b85ff2a396ee",
        "architecture": "Byte-identical copied PR #80 core; no architecture, scales, seeds, optimizer or training changes",
        "train_split": TRAIN_SPLIT, "evaluation_split": EVAL_SPLIT,
        "split_specs": {TRAIN_SPLIT: train_spec, EVAL_SPLIT: eval_spec},
        "alternate_validation_templates": ALTERNATE_TEMPLATES,
        "pairing": "Original plus selected-field value swap for compare; original plus preauthored disjoint alternate wording for read/explain",
        "candidate_contract": "12 structural candidates; all three relation choices; no truth filtering or target-dependent inference",
        "initialization": "random; no teacher or parent checkpoint", "cost_usd": 0,
        "scope": "Single train-to-disjoint-validation transfer condition; iterative synthetic development, not official holdout; no tuning or parent-retention claim"}
    core.write(out / "registration.json", registration)
    print(json.dumps({"prepared": str(out), "train": len(train), "validation": len(validation),
                      "core_sha256": CORE_SHA}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else core.run)(args.out)
