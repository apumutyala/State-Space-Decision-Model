"""Verify frozen artifacts and replay both conditions' checked emissions."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2]))
from language_growth.kernel import check_program
from language_growth.realize import render_checked


manifest = json.loads((root / "raw_artifact_manifest.json").read_text())
decoded = {}
for name, expected in manifest.items():
    archive = (root / name).read_bytes()
    assert hashlib.sha256(archive).hexdigest() == expected["compressed_sha256"]
    original = gzip.decompress(archive)
    assert hashlib.sha256(original).hexdigest() == expected["decompressed_sha256"]
    assert len(original) == expected["decompressed_bytes"]
    decoded[name] = json.loads(original)
registration = json.loads((root / "registration.json").read_text())
data = decoded["frozen_rows.json.gz"]
inputs = {row["id"]: row for row in data["development"]}
assert len(inputs) == 360 and len(data["train"]) == 600
assert all(row["source_split"] == "development_validation" for row in inputs.values())
validation_digest = hashlib.sha256(json.dumps(data["development"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert validation_digest == registration["unchanged_validation_rows_sha256"]
train_digest = hashlib.sha256(json.dumps(data["train"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert train_digest == registration["unchanged_base_training_rows_sha256"]
training = {r["id"]: r for r in data["train"]}
assert len(data["paired_comparison_train"]) == 200
for key, swapped in data["paired_comparison_train"].items():
    original = training[key]
    assert swapped["state"]["task_input"] == original["state"]["task_input"]
    field = original["evaluation_only"]["program"]["nodes"][0]["field"]
    old_records = {r["record_ref"]: r for r in original["state"]["records"]}
    new_records = {r["record_ref"]: r for r in swapped["state"]["records"]}
    refs = sorted(old_records)
    for index, ref in enumerate(refs):
        assert new_records[ref]["fields"][field]["value"] == old_records[refs[1-index]]["fields"][field]["value"]
        for other_field in old_records[ref]["fields"]:
            if other_field != field:
                assert new_records[ref]["fields"][other_field] == old_records[ref]["fields"][other_field]
    ev = swapped["evaluation_only"]
    assert check_program(ev["program"],ev["host"],ev["catalog"]).status == "accepted"
assert manifest["baseline_predictions.json.gz"]["decompressed_sha256"] == registration["baseline_predictions_sha256"]
for artifact, receipt_name in (("predictions.json.gz", "receipt.json"),
                                ("baseline_predictions.json.gz", "baseline_receipt.json")):
    receipt = json.loads((root / receipt_name).read_text())
    assert manifest[artifact]["decompressed_sha256"] == receipt["raw_predictions_sha256"]
    observations = []
    for seed, predictions in decoded[artifact].items():
        assert {r["id"] for r in predictions} == set(inputs)
        for prediction in predictions:
            evidence = inputs[prediction["id"]]["evaluation_only"]
            checked = check_program(prediction["program"], evidence["host"], evidence["catalog"])
            assert (checked.status, checked.reason) == (prediction["kernel_status"], prediction["kernel_reason"])
            rendered = render_checked(checked, evidence["host"], evidence["catalog"]) if checked.status == "accepted" else None
            assert rendered == prediction["rendered"]
            reference = check_program(evidence["program"], evidence["host"], evidence["catalog"])
            correct = checked.status == "accepted" and rendered == render_checked(reference, evidence["host"], evidence["catalog"])
            assert correct == prediction["whole_answer_correct"] == prediction["target_exact"]
            assert prediction["target_exact"] == (prediction["predicted"] == prediction["gold"])
            observations.append(prediction)
    assert len(observations) == 1080
    for kind in ("all", "read", "explain", "compare"):
        rows = observations if kind == "all" else [r for r in observations if r["gold"]["kind"] == kind]
        expected = receipt["pooled"]["all"] if kind == "all" else receipt["pooled"]["by_kind"][kind]
        assert len(rows) == expected["count"]
        for metric in ("target_exact", "kernel_accepted", "whole_answer_correct"):
            assert sum(r[metric] for r in rows) == expected[metric]
controls = decoded["intervention_controls.json.gz"]
control_count = 0
for control, by_seed in controls.items():
    for seed, rows in by_seed.items():
        assert len(rows) == 120
        for prediction in rows:
            evidence = inputs[prediction["id"]]["evaluation_only"]
            checked = check_program(prediction["program"], evidence["host"], evidence["catalog"])
            assert (checked.status,checked.reason) == (prediction["kernel_status"],prediction["kernel_reason"])
            text = render_checked(checked,evidence["host"],evidence["catalog"]) if checked.status == "accepted" else None
            assert text == prediction["rendered"]
            reference = check_program(evidence["program"],evidence["host"],evidence["catalog"])
            correct = checked.status == "accepted" and text == render_checked(reference,evidence["host"],evidence["catalog"])
            assert correct == prediction["whole_answer_correct"] == prediction["target_exact"]
            assert prediction["target_exact"] == (prediction["predicted"] == prediction["gold"])
            control_count += 1
assert control_count == 720
print("Verified all archive hashes, unchanged base/validation rows, 200 authored swaps, and all 2880 model/control kernel-render outcomes.")
