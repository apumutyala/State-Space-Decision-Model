"""Decompose saved semantic-slot endpoint probabilities; no model is loaded."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SOURCE_RECEIPT = HERE / "source_receipt.json.gz"
SEEDS = ("20261006", "20261016", "20261026")
BETAS = ("0", "005")
FAMILIES = ("old_failing", "new_disjoint")
KINDS = ("read", "explain", "compare")
BOOTSTRAP_SEED = 9203
BOOTSTRAP_DRAWS = 2000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def binding_exact(binding: dict, target: dict) -> bool:
    if binding.get("kind") != target.get("kind"):
        return False
    for key in ("field", "left", "right", "temporal_scope"):
        if key in target and binding.get(key) != target.get(key):
            return False
    return True


def read_arm(seed: str, beta: str) -> list[dict]:
    suffix = "0" if beta == "0" else "005"
    name = f"seed-{seed}-beta-{suffix}-endpoints.jsonl"
    path = RAW / name
    receipt_bytes = gzip.decompress(SOURCE_RECEIPT.read_bytes())
    receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
    expected = receipt["results"][f"{seed}-{suffix}"]["raw_sha256"][name]
    raw = path.read_bytes()
    if sha256(raw) != expected:
        raise ValueError(f"source raw hash mismatch: {name}")
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        probs, bindings = row["probabilities"], row["bindings"]
        if len(probs) != len(bindings) or len(probs) != 12:
            raise ValueError(f"candidate shape mismatch: {row['id']}")
        if abs(sum(probs) - 1.0) > 1e-5:
            raise ValueError(f"probabilities do not sum to one: {row['id']}")
        target = row["target"]
        if binding_exact(bindings[max(range(len(probs)), key=probs.__getitem__)], target) != row["correct"]:
            raise ValueError(f"saved exact flag differs from endpoint MAP: {row['id']}")
    return rows


def summarize_row(row: dict) -> dict[str, float]:
    probs, bindings, target = row["probabilities"], row["bindings"], row["target"]
    operation_mass = {kind: sum(p for p, b in zip(probs, bindings) if b["kind"] == kind)
                      for kind in KINDS}
    predicted_kind = max(KINDS, key=operation_mass.__getitem__)
    gold_indices = [i for i, b in enumerate(bindings) if b["kind"] == target["kind"]]
    if len(gold_indices) != 4:
        raise ValueError(f"expected four candidates per operation: {row['id']}")
    gold_mass = sum(probs[i] for i in gold_indices)
    if gold_mass <= 0:
        raise ValueError(f"zero gold-operation mass: {row['id']}")
    cond_idx = max(gold_indices, key=probs.__getitem__)
    hier_indices = [i for i, b in enumerate(bindings) if b["kind"] == predicted_kind]
    hier_idx = max(hier_indices, key=probs.__getitem__)
    joint_idx = max(range(len(probs)), key=probs.__getitem__)
    exact = lambda i: float(binding_exact(bindings[i], target))
    gold_conditional_probability = sum(probs[i] for i in gold_indices if binding_exact(bindings[i], target)) / gold_mass
    return {
        "operation_marginal_correct": float(predicted_kind == target["kind"]),
        "gold_operation_mass": operation_mass[target["kind"]],
        "oracle_gold_operation_pointer_exact": exact(cond_idx),
        "gold_operation_conditional_exact_probability": gold_conditional_probability,
        "hierarchical_exact": exact(hier_idx),
        "joint_map_exact": exact(joint_idx),
        "saved_map_exact": float(row["correct"]),
        "field_conditional_correct": float(bindings[cond_idx].get("field") == target.get("field")),
        "left_conditional_correct": float(bindings[cond_idx].get("left") == target.get("left")),
        "right_conditional_correct": float(bindings[cond_idx].get("right") == target.get("right")),
    }


def means(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty group")
    keys = rows[0]["metrics"].keys()
    return {key: sum(r["metrics"][key] for r in rows) / len(rows) for key in keys}


def paired_cluster_ci(base: dict[str, float], treatment: dict[str, float], *, seed: int) -> dict:
    ids = sorted(set(base) & set(treatment))
    if len(ids) != len(base) or len(ids) != len(treatment) or not ids:
        raise ValueError("paired world clusters differ")
    delta = np.asarray([treatment[i] - base[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))
    draws = np.sort(delta[indices].mean(axis=1))
    return {"n_world_clusters": len(ids), "estimate": float(delta.mean()),
            "ci95": [float(draws[int(.025 * BOOTSTRAP_DRAWS)]),
                     float(draws[int(.975 * BOOTSTRAP_DRAWS)])],
            "method": "paired percentile bootstrap over world_id; seed/family observations averaged within world",
            "seed": seed, "draws": BOOTSTRAP_DRAWS}


def mean_cluster_ci(rows: list[dict], metric: str, *, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["world_id"]].append(row["metrics"][metric])
    ids = sorted(grouped)
    if not ids:
        raise ValueError("empty world cluster set")
    values = np.asarray([sum(grouped[i]) / len(grouped[i]) for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))
    draws = np.sort(values[indices].mean(axis=1))
    return {"n_world_clusters": len(ids), "mean": float(values.mean()),
            "ci95": [float(draws[int(.025 * BOOTSTRAP_DRAWS)]),
                     float(draws[int(.975 * BOOTSTRAP_DRAWS)])],
            "method": "world-cluster percentile bootstrap; seed replicates averaged within world",
            "seed": seed, "draws": BOOTSTRAP_DRAWS}


def main(output_path: Path | None = None) -> None:
    raw_rows = []
    raw_hashes = {}
    for seed in SEEDS:
        for beta in BETAS:
            suffix = "0" if beta == "0" else "005"
            name = f"seed-{seed}-beta-{suffix}-endpoints.jsonl"
            raw_hashes[name] = sha256((RAW / name).read_bytes())
            for row in read_arm(seed, beta):
                row["seed"] = seed
                row["beta"] = beta
                row["metrics"] = summarize_row(row)
                raw_rows.append(row)

    if len(raw_rows) != 6 * 360:
        raise ValueError(f"expected 2160 rows, got {len(raw_rows)}")
    world_keys = {(r["family"], r["world_id"]) for r in raw_rows}
    world_ids = {r["world_id"] for r in raw_rows}
    if len(world_keys) != 360 or len(world_ids) != 180:
        raise ValueError(f"expected 360 family/world pairs on 180 underlying worlds, got {len(world_keys)}/{len(world_ids)}")
    for family in FAMILIES:
        for beta in BETAS:
            subset = [r for r in raw_rows if r["family"] == family and r["beta"] == beta]
            by_seed = defaultdict(list)
            for row in subset:
                by_seed[row["seed"]].append(row)
            if set(by_seed) != set(SEEDS) or any(len(v) != 180 for v in by_seed.values()):
                raise ValueError(f"seed/family count mismatch: {family}/{beta}")
    output = {"schema": "ssdm.semantic_slot_conditional_binding_diagnostic.v1",
              "source_receipt_sha256": sha256(gzip.decompress(SOURCE_RECEIPT.read_bytes())),
              "source_receipt_gzip_sha256": sha256(SOURCE_RECEIPT.read_bytes()),
              "source_model_run_commit": "bd0c591",
              "source_freeze_sha256": json.loads(gzip.decompress(SOURCE_RECEIPT.read_bytes()))["freeze_sha256"],
              "command": "python -X utf8 analyze.py",
              "runtime": {"python": sys.version, "numpy": np.__version__, "device": "CPU-only",
                          "cuda_visible_devices": "-1", "model_loaded": False, "model_inference": False,
                          "training": False, "cost_usd": 0},
              "raw_sha256": raw_hashes, "rows": len(raw_rows),
              "family_world_pairs": 360, "underlying_world_clusters": 180,
              "numpy_version": np.__version__, "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_draws": BOOTSTRAP_DRAWS,
              "interpretation": "saved-posterior offline oracle diagnostic; no model inference or training",
              "definitions": {
                  "operation_marginal_correct": "argmax of candidate probability summed over all bindings of each allowed operation",
                  "oracle_gold_operation_pointer_exact": "top candidate after restricting to the gold operation; not served performance",
                  "hierarchical_exact": "operation marginal argmax, then top binding candidate within that selected operation",
                  "gold_operation_conditional_exact_probability": "posterior probability of the exact target binding conditional on the gold operation",
                  "cluster": "world_id; training seed and repeated family rows are averaged within world before bootstrap"},
              "results": {}}

    metric_keys = list(raw_rows[0]["metrics"])
    for family in FAMILIES + ("pooled",):
        family_rows = raw_rows if family == "pooled" else [r for r in raw_rows if r["family"] == family]
        output["results"][family] = {}
        for beta in BETAS:
            subset = [r for r in family_rows if r["beta"] == beta]
            output["results"][family]["beta" + beta] = {
                "n_rows": len(subset), "overall": means(subset),
                "overall_ci95": {metric: mean_cluster_ci(subset, metric,
                                                          seed=BOOTSTRAP_SEED + index)
                                 for index, metric in enumerate(metric_keys)},
                "by_gold_operation": {kind: means([r for r in subset if r["target"]["kind"] == kind]) for kind in KINDS},
                "by_gold_operation_ci95": {
                    kind: {metric: mean_cluster_ci([r for r in subset if r["target"]["kind"] == kind], metric,
                                                   seed=BOOTSTRAP_SEED + 100 + index)
                           for index, metric in enumerate(metric_keys)}
                    for kind in KINDS},
                "by_seed": {seed: means([r for r in subset if r["seed"] == seed]) for seed in SEEDS}}
        base_rows = [r for r in family_rows if r["beta"] == "0"]
        treat_rows = [r for r in family_rows if r["beta"] == "005"]
        base_worlds, treat_worlds = defaultdict(list), defaultdict(list)
        for row in base_rows:
            base_worlds[row["world_id"]].append(row)
        for row in treat_rows:
            treat_worlds[row["world_id"]].append(row)
        base_means = {wid: means(rows) for wid, rows in base_worlds.items()}
        treat_means = {wid: means(rows) for wid, rows in treat_worlds.items()}
        output["results"][family]["beta005_minus_beta0"] = {
            metric: paired_cluster_ci({wid: m[metric] for wid, m in base_means.items()},
                                      {wid: m[metric] for wid, m in treat_means.items()},
                                      seed=BOOTSTRAP_SEED + index)
            for index, metric in enumerate(metric_keys)}
    target_path = output_path or (HERE / "conditional_result.json")
    target_path.write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({family: output["results"][family]["beta005_minus_beta0"]
                      for family in FAMILIES + ("pooled",)}, sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the analysis JSON here; defaults to conditional_result.json beside this script")
    args = parser.parse_args()
    main(args.output)
