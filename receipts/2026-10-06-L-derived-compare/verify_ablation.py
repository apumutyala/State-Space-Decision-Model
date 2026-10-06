"""Stdlib-only replay of the preserved ablation using vendored PR #82 inputs."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from language_growth.kernel import check_program
from language_growth.realize import render_checked

sha = lambda raw: hashlib.sha256(raw).hexdigest()
inputs = HERE / "pr82_reference"
provenance = json.loads((inputs / "source_files.json").read_text())
assert provenance["source_commit"] == "1a09d189635711bd79319ef8eb334ed587e34c2e"
for name, expected in provenance["files"].items():
    raw = (inputs / name).read_bytes()
    assert sha(raw) == expected["sha256"] and len(raw) == expected["bytes"]
manifest = json.loads((inputs / "raw_artifact_manifest.json").read_text())
decoded = {}
for name in ("predictions.json.gz", "frozen_rows.json.gz"):
    raw = (inputs / name).read_bytes()
    assert sha(raw) == manifest[name]["compressed_sha256"]
    raw = gzip.decompress(raw)
    assert sha(raw) == manifest[name]["decompressed_sha256"]
    decoded[name] = json.loads(raw)
registration = json.loads((inputs / "registration.json").read_text())
validation = decoded["frozen_rows.json.gz"]["development"]
assert sha(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()) == registration["validation_rows_sha256"]
evidence = {r["id"]: r["evaluation_only"] for r in validation}
saved = {(seed, r["id"]): r for seed, rows in decoded["predictions.json.gz"].items() for r in rows}
report = json.loads((HERE / "ablation.json").read_text())
archive = (HERE / "ablation_observations.json.gz").read_bytes()
assert sha(archive) == report["raw_output"]["compressed_sha256"]
raw = gzip.decompress(archive)
assert sha(raw) == report["raw_output"]["decompressed_sha256"]
rows = json.loads(raw)
assert len(rows) == len(saved) == 1080
for row in rows:
    original = saved[row["seed"], row["id"]]
    assert row["explicit"] == original
    ev = evidence[row["id"]]
    host, catalog, expected = deepcopy((ev["host"], ev["catalog"], original["program"]))
    changed = 0
    for node in expected["nodes"]:
        if node["op"] == "Compare":
            node["op"] = "DerivedCompare"
            del node["relation"]
            changed += 1
    if changed:
        old = host["registry_ref"]
        new = old + ":derived-compare-v0.2"
        catalog["registries"][new] = deepcopy(catalog["registries"][old])
        catalog["registries"][new]["allowed_ops"].append("DerivedCompare")
        host["registry_ref"] = expected["registry_ref"] = new
        expected["schema"] = "ssdm.semantic_program.v0.2"
    assert expected == row["derived"]["program"]
    assert changed == row["transformed_compare_nodes"]
    checked = check_program(expected, host, catalog)
    assert (checked.status, checked.reason) == (row["derived"]["kernel_status"], row["derived"]["kernel_reason"])
    rendered = render_checked(checked, host, catalog) if checked.status == "accepted" else None
    assert rendered == row["derived"]["rendered"]
    reference = check_program(ev["program"], ev["host"], ev["catalog"])
    correct = checked.status == "accepted" and rendered == render_checked(reference, ev["host"], ev["catalog"])
    assert correct == row["derived"]["whole_answer_correct"]
for kind in ("all", "compare", "read", "explain"):
    group = [r for r in rows if kind == "all" or r["kind"] == kind]
    for metric in ("kernel_accepted", "whole_answer_correct"):
        for arm, key in (("explicit", "explicit_count"), ("derived", "derived_count")):
            assert sum(r[arm][metric] for r in group) == report["statistics"][kind][metric][key]
assert sum(r["transformed_compare_nodes"] for r in rows) == 369
print("Verified 5 exact vendored PR #82 files, both compressed archives, and all 1080 preserved ablation outputs; no Git objects or NumPy required.")
