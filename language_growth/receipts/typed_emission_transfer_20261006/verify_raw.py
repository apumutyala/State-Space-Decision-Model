"""Verify committed raw hashes and replay kernel/rendering without training."""
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
for name, record in manifest.items():
    archive = (root / name).read_bytes()
    assert hashlib.sha256(archive).hexdigest() == record["compressed_sha256"]
    raw = gzip.decompress(archive)
    assert hashlib.sha256(raw).hexdigest() == record["decompressed_sha256"]
    assert len(raw) == record["decompressed_bytes"]
    decoded[name] = json.loads(raw)
receipt = json.loads((root / "receipt.json").read_text())
assert manifest["predictions.json.gz"]["decompressed_sha256"] == receipt["raw_predictions_sha256"]
assert manifest["frozen_rows.json.gz"]["decompressed_sha256"] == receipt["data_sha256"]
data = decoded["frozen_rows.json.gz"]
inputs = {r["id"]: r for r in data["development"]}
assert len(data["train"]) == 600 and len(inputs) == 360
assert all(r["source_split"] == "development_validation" for r in inputs.values())
observations = []
for seed, rows in decoded["predictions.json.gz"].items():
    assert {r["id"] for r in rows} == set(inputs)
    for prediction in rows:
        evidence = inputs[prediction["id"]]["evaluation_only"]
        checked = check_program(prediction["program"], evidence["host"], evidence["catalog"])
        assert (checked.status, checked.reason) == (prediction["kernel_status"], prediction["kernel_reason"])
        text = render_checked(checked, evidence["host"], evidence["catalog"]) if checked.status == "accepted" else None
        assert text == prediction["rendered"]
        reference = check_program(evidence["program"], evidence["host"], evidence["catalog"])
        correct = checked.status == "accepted" and text == render_checked(reference, evidence["host"], evidence["catalog"])
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
print("Verified both committed archive hashes, all 1080 kernel/render replays, and every pooled/per-kind result.")
