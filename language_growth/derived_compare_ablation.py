"""Inference-free module ablation of the fixed PR #82 development predictions.

Reads only committed local Git objects. No model, optimizer, task parser or
candidate selection runs here. Frozen gold is used only after kernel execution.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .kernel import check_program
from .realize import render_checked
from .types import Comparison


REFERENCE = "1a09d189635711bd79319ef8eb334ed587e34c2e"
PREFIX = "language_growth/receipts/typed_emission_diversity_20261006/"
OUT = Path("receipts/2026-10-06-L-derived-compare")
BOOTSTRAP = {"seed": 3197, "replicates": 2000,
             "unit": "unique task cluster, preserving both arms, all three fixed model seeds and both members",
             "interval": "95% paired percentile, conditional on fixed model seeds and synthetic development families"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_reference(name):
    return subprocess.check_output(["git", "show", REFERENCE + ":" + PREFIX + name])


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_program(program, host, catalog):
    """Only change relation computation, plus explicit host/schema enablement."""
    program, host, catalog = deepcopy((program, host, catalog))
    changed = 0
    for node in program["nodes"]:
        if node["op"] == "Compare":
            node["op"] = "DerivedCompare"
            del node["relation"]
            changed += 1
    if changed:
        old_registry = host["registry_ref"]
        new_registry = old_registry + ":derived-compare-v0.2"
        registry = deepcopy(catalog["registries"][old_registry])
        registry["allowed_ops"].append("DerivedCompare")
        catalog["registries"][new_registry] = registry
        host["registry_ref"] = program["registry_ref"] = new_registry
        program["schema"] = "ssdm.semantic_program.v0.2"
    return program, host, catalog, changed


def ci(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(BOOTSTRAP["seed"]).choice(
        values, (BOOTSTRAP["replicates"], len(values)), replace=True).mean(1)
    return {"estimate": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    manifest = json.loads(read_reference("raw_artifact_manifest.json"))
    source_receipt = json.loads(read_reference("receipt.json"))
    artifacts, decoded = {}, {}
    for name in ("predictions.json.gz", "frozen_rows.json.gz"):
        archive = read_reference(name)
        raw = gzip.decompress(archive)
        expected = manifest[name]
        assert sha(archive) == expected["compressed_sha256"]
        assert sha(raw) == expected["decompressed_sha256"]
        assert len(raw) == expected["decompressed_bytes"]
        artifacts[name] = expected
        decoded[name] = json.loads(raw)
    assert artifacts["predictions.json.gz"]["decompressed_sha256"] == source_receipt["raw_predictions_sha256"]
    inputs = {row["id"]: row for row in decoded["frozen_rows.json.gz"]["development"]}
    assert len(inputs) == 360
    assert all(row["source_split"] == "development_validation" for row in inputs.values())
    observations = []
    for seed, rows in decoded["predictions.json.gz"].items():
        assert {r["id"] for r in rows} == inputs.keys()
        for saved in rows:
            evidence = inputs[saved["id"]]["evaluation_only"]
            program, host, catalog = saved["program"], evidence["host"], evidence["catalog"]
            explicit = check_program(program, host, catalog)
            explicit_text = render_checked(explicit, host, catalog) if explicit.status == "accepted" else None
            assert (explicit.status, explicit.reason, explicit_text) == (saved["kernel_status"], saved["kernel_reason"], saved["rendered"])
            transformed, enabled_host, enabled_catalog, changed = derive_program(program, host, catalog)
            derived = check_program(transformed, enabled_host, enabled_catalog)
            derived_text = render_checked(derived, enabled_host, enabled_catalog) if derived.status == "accepted" else None
            selected = deepcopy(saved["predicted"])
            if changed and derived.status == "accepted":
                comparison = derived.outputs[0].content
                assert isinstance(comparison, Comparison)
                selected["relation"] = comparison.relation
            # Gold/reference scoring happens after both arms have run. Neither
            # transformation nor checker receives a task-derived target.
            reference = check_program(evidence["program"], host, catalog)
            assert reference.status == "accepted"
            reference_text = render_checked(reference, host, catalog)
            assert saved["whole_answer_correct"] == (explicit.status == "accepted" and explicit_text == reference_text)
            target_exact = selected == saved["gold"]
            correct = derived.status == "accepted" and derived_text == reference_text
            assert target_exact == correct
            current = {**saved, "program": transformed, "predicted": selected,
                       "kernel_status": derived.status, "kernel_reason": derived.reason,
                       "kernel_accepted": derived.status == "accepted", "rendered": derived_text,
                       "target_exact": target_exact, "whole_answer_correct": correct,
                       "relation_correct": saved["gold"]["kind"] == "compare" and selected.get("relation") == saved["gold"]["relation"]}
            # Structural predictions, including the chosen operation family,
            # are frozen. Only its relation-computation implementation changes.
            assert {k: v for k, v in selected.items() if k != "relation"} == {k: v for k, v in saved["predicted"].items() if k != "relation"}
            observations.append({"seed": seed, "id": saved["id"], "cluster": saved["cluster"],
                                 "member": saved["member"], "kind": saved["gold"]["kind"],
                                 "transformed_compare_nodes": changed,
                                 "explicit": saved, "derived": current})
    assert len(observations) == 1080
    metrics = ("kernel_accepted", "whole_answer_correct", "accepted_wrong_target", "false_relation_reject")
    def metric(row, arm, name):
        value = row[arm]
        if name == "accepted_wrong_target":
            return value["kernel_accepted"] and not value["whole_answer_correct"]
        if name == "false_relation_reject":
            return value["kernel_reason"] == "unsupported_assertion" and any(n["op"] in ("Compare", "DerivedCompare") for n in value["program"]["nodes"])
        return value[name]
    statistics = {}
    for kind in ("all", "read", "explain", "compare"):
        rows = [r for r in observations if kind == "all" or r["kind"] == kind]
        clusters = defaultdict(list)
        for row in rows:
            clusters[row["cluster"]].append(row)
        assert all(len(rows) == 6 for rows in clusters.values())
        stats = {"observations": len(rows), "unique_clusters": len(clusters)}
        for name in metrics:
            before = [np.mean([metric(r, "explicit", name) for r in group]) for group in clusters.values()]
            after = [np.mean([metric(r, "derived", name) for r in group]) for group in clusters.values()]
            stats[name] = {"explicit_count": sum(metric(r, "explicit", name) for r in rows),
                           "derived_count": sum(metric(r, "derived", name) for r in rows),
                           "explicit": ci(before), "derived": ci(after),
                           "paired_delta": ci(np.asarray(after) - np.asarray(before))}
        for arm in ("explicit", "derived"):
            stats[arm + "_reject_reasons"] = dict(Counter(r[arm]["kernel_reason"] for r in rows if not r[arm]["kernel_accepted"]))
        stats["improved"] = sum(not r["explicit"]["whole_answer_correct"] and r["derived"]["whole_answer_correct"] for r in rows)
        stats["regressed"] = sum(r["explicit"]["whole_answer_correct"] and not r["derived"]["whole_answer_correct"] for r in rows)
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["seed"], row["cluster"]].append(row)
        assert all(len(pair) == 2 for pair in pairs.values())
        paired_clusters = defaultdict(list)
        for (_, cluster), pair in pairs.items():
            paired_clusters[cluster].append(tuple(all(r[arm]["whole_answer_correct"] for r in pair) for arm in ("explicit", "derived")))
        before = [np.mean([p[0] for p in group]) for group in paired_clusters.values()]
        after = [np.mean([p[1] for p in group]) for group in paired_clusters.values()]
        stats["both_members_correct"] = {"seed_task_pairs": len(pairs),
             "explicit_count": sum(all(r["explicit"]["whole_answer_correct"] for r in pair) for pair in pairs.values()),
             "derived_count": sum(all(r["derived"]["whole_answer_correct"] for r in pair) for pair in pairs.values()),
             "explicit": ci(before), "derived": ci(after), "paired_delta": ci(np.asarray(after) - np.asarray(before))}
        statistics[kind] = stats
    swaps = defaultdict(list)
    for row in observations:
        if row["kind"] == "compare":
            swaps[row["seed"], row["cluster"]].append(row)
    assert len(swaps) == 180 and all(len(pair) == 2 for pair in swaps.values())
    swap_statistics = {"seed_task_pairs": len(swaps), "unique_task_pairs": len({r["cluster"] for r in observations if r["kind"] == "compare"})}
    for arm in ("explicit", "derived"):
        swap_statistics[arm] = {
            "required_relation_changes": sum(pair[0][arm]["gold"]["relation"] != pair[1][arm]["gold"]["relation"] for pair in swaps.values()),
            "predicted_relation_changes": sum(pair[0][arm]["predicted"].get("relation") != pair[1][arm]["predicted"].get("relation") for pair in swaps.values()),
            "required_changes_met": sum(pair[0][arm]["gold"]["relation"] != pair[1][arm]["gold"]["relation"] and pair[0][arm]["predicted"].get("relation") != pair[1][arm]["predicted"].get("relation") for pair in swaps.values()),
            "both_structures_exact": sum(all(r[arm]["structure_exact"] for r in pair) for pair in swaps.values()),
            "both_relation_labels_correct": sum(all(r[arm]["relation_correct"] for r in pair) for pair in swaps.values()),
            "both_answers_correct": sum(all(r[arm]["whole_answer_correct"] for r in pair) for pair in swaps.values()),
            "both_inequality_answers_correct": sum(pair[0][arm]["gold"]["relation"] != "equal" and all(r[arm]["whole_answer_correct"] for r in pair) for pair in swaps.values()),
            "both_equality_answers_correct": sum(pair[0][arm]["gold"]["relation"] == "equal" and all(r[arm]["whole_answer_correct"] for r in pair) for pair in swaps.values())}
    # Reproduce PR #82's headline before crediting any change.
    assert statistics["all"]["whole_answer_correct"]["explicit_count"] == source_receipt["pooled"]["all"]["whole_answer_correct"] == 563
    assert statistics["compare"]["whole_answer_correct"]["explicit_count"] == 86
    assert sum(r["explicit"]["structure_exact"] for r in observations if r["kind"] == "compare") == 142
    raw = encoded(observations)
    archive = gzip.compress(raw, compresslevel=9, mtime=0)
    (OUT / "ablation_observations.json.gz").write_bytes(archive)
    report = {"schema": "ssdm.derived_compare_ablation.v0", "scope": "inference-free deterministic module ablation of fixed PR #82 synthetic development outputs; not a new model result",
              "source_commit": REFERENCE, "source_prefix": PREFIX, "source_artifacts": artifacts,
              "bootstrap": BOOTSTRAP, "statistics": statistics, "value_swap": swap_statistics,
              "transformed_compare_nodes": sum(r["transformed_compare_nodes"] for r in observations),
              "observations": len(observations), "cost_usd": 0, "model_calls": 0,
              "raw_output": {"compressed_sha256": sha(archive), "decompressed_sha256": sha(raw)},
              "source_hashes": {str(path): sha(path.read_bytes()) for path in (Path("language_growth/derived_compare_ablation.py"), Path("language_growth/kernel.py"), Path("language_growth/realize.py"), Path("language_growth/types.py"))}}
    write("ablation.json", report)
    print(json.dumps({"statistics": statistics, "value_swap": swap_statistics, "transformed_compare_nodes": report["transformed_compare_nodes"]}, indent=2))


if __name__ == "__main__":
    main()
