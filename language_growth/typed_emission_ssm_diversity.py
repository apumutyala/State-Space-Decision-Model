"""One six-template train-only wording condition; unchanged typed-emission SSM."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from . import typed_emission_ssm as core

BASE = "0f4b8ad2ea717c79858ddfbc4deb55c9b71940af"
CORE_SHA = "93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6"
BASELINE_DATA_SHA = "4c63f494a42ec66da89236225d3d9aa6a6c22e4b9844df96c4a8188ee00b0faa"
BASELINE_PREDICTIONS_SHA = "8e164ee8bc163c8eee346cd394c84922d4825ec097c99d817b485dffd0a36678"
# Copied exactly from the earlier PR #75 wording recipe, including assignment.
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diversify(rows):
    """Offline training authoring only; no target-dependent inference changes."""
    diverse = copy.deepcopy(rows)
    for index, (old, new) in enumerate(zip(rows, diverse)):
        target = core.gold_target(old["evaluation_only"]["program"])
        records = {r["record_ref"]: r for r in old["state"]["records"]}
        left = records[target["left"]]["entity_ref"]
        right = records[target["right"]]["entity_ref"] if target["right"] else left
        template = TRAIN_TEMPLATES[target["kind"]][(index // 3) % 6]
        new["state"]["task_input"] = template.format(entity=left, left=left, right=right,
                                                     field=target["field"].replace("_", " "))
        restored = copy.deepcopy(new)
        restored["state"]["task_input"] = old["state"]["task_input"]
        assert restored == old  # The sole change is train task_input.
    return diverse


def prepare(out, baseline_dir):
    assert not out.exists()
    source = Path(core.__file__)
    assert core.sha(source) == CORE_SHA
    baseline_bytes = gzip.decompress((baseline_dir / "frozen_rows.json.gz").read_bytes())
    assert hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_DATA_SHA
    predictions_bytes = gzip.decompress((baseline_dir / "predictions.json.gz").read_bytes())
    assert hashlib.sha256(predictions_bytes).hexdigest() == BASELINE_PREDICTIONS_SHA
    baseline = json.loads(baseline_bytes)
    registration = json.loads((baseline_dir / "registration.json").read_text())
    assert registration["config"] == core.CONFIG
    assert registration["train_split"] == "development_train"
    assert registration["evaluation_split"] == "development_validation"
    assert len(baseline["train"]) == 600 and len(baseline["development"]) == 360
    assert all(r["source_split"] == "development_validation" for r in baseline["development"])
    train = diversify(baseline["train"])
    validation = copy.deepcopy(baseline["development"])
    assert validation == baseline["development"]
    out.mkdir(parents=True)
    core.write(out / "frozen_rows.json", {"train": train, "development": validation})
    (out / "baseline_predictions.json").write_bytes(predictions_bytes)
    shutil.copyfile(baseline_dir / "receipt.json", out / "baseline_receipt.json")
    paths = [source, Path(__file__), *[source.with_name(name) for name in
        ("development_data.py", "kernel.py", "realize.py", "parent_bridge.py", "types.py")]]
    counts = {kind: {str(i): sum(core.gold_target(r["evaluation_only"]["program"])["kind"] == kind and
                                (index // 3) % 6 == i for index, r in enumerate(train)) for i in range(6)}
              for kind in core.KINDS}
    core.write(out / "registration.json", {
        "config": core.CONFIG, "source_hashes": {p.name: core.sha(p) for p in paths},
        "source": str(source.resolve()), "base_commit": BASE,
        "frozen_data_sha256": core.sha(out / "frozen_rows.json"),
        "baseline_data_sha256": BASELINE_DATA_SHA,
        "baseline_predictions_sha256": BASELINE_PREDICTIONS_SHA,
        "baseline_source_commit": "8bca172092dce3eec005fe5bd6a34e6db0bbe4f7",
        "baseline_registration": registration,
        "validation_rows_sha256": digest(validation), "validation_rows_exactly_reused": True,
        "train_templates": TRAIN_TEMPLATES, "template_assignment": "(row_index // 3) % 6; unchanged PR #75 assignment",
        "template_counts_by_kind": counts,
        "template_source": {"commit": "64df83efeafa08f43bebf6e2e5e59a1ff5baf77e",
                            "path": "language_growth/query_template_transfer_probe.py",
                            "sha256": "ba521a134a0cef41ade8e8bf5ff249c9cd3dc7355057ae3502481fcdfa8f61c1"},
        "only_change": "task_input in training rows; model, numeric features/scales, worlds, seed list, optimizer, 300 steps, validation rows and pairs unchanged",
        "initialization": "random; no parent checkpoint or teacher", "cost_usd": 0,
        "scope": "One frozen six-template training-wording condition; paired against saved PR #81 predictions; iterative synthetic development only, not official holdout; no tuning"})
    print(json.dumps({"prepared": str(out), "validation_rows_sha256": digest(validation),
                      "template_counts": counts}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out, args.baseline)
    else:
        core.run(args.out)
