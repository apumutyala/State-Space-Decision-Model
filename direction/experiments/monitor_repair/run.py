"""CPU-only, static monitor-corpus repair experiment; run as a module from repo root."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy

from direction.experiments.artifact_audit import audit
from keel.paths import private_root
from .repairs import B2_SEED, canonical, repair_candidates, structure

OUT = Path(__file__).resolve().parent
SOURCE_SHA256 = "bdff0e9aec67ca848bdd763e3821d09799f34e1f3aeb94250a44428a6b5b6106"
FAMILIES = {
    "size": ["characters", "lines"],
    "layout": ["blank_fraction", "comment_fraction", "comment_lines_fraction", "indent_mean"],
    "identifiers": ["identifier_mean_length", "identifier_underscore_fraction"],
    "ast_counts": ["ast_" + k for k in audit.AST_KINDS],
    "branch_equalities": audit.STRUCTURE,
    "tells": ["explicit_tells", "comment_tells", "identifier_tells"],
}
PRIVILEGED = "honest_anchor_distance_privileged"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def corpus(path, data):
    digest = save(path, data)
    save(path.with_name("data_report.json"), {"dataset_sha256": digest})
    return digest


def build(data, root):
    normalized, records, buckets = [], [], defaultdict(list)
    for index, row in enumerate(data["submissions"]):
        code, tree, notes = canonical(row["code"])
        normalized.append({**{k: v for k, v in row.items() if k != "comments"}, "code": code})
        key = audit.digest(structure(tree))
        records.append({"submission_id": row["submission_id"], "problem_id": row["problem_id"],
                        "source_sha256": audit.digest(row["code"]), "code_sha256": audit.digest(code),
                        "ast_sha256": key, **notes})
        buckets[key].append(index)
    kept, exclusions = set(), []
    for key, indices in sorted(buckets.items()):
        mixed = len({normalized[i]["sabotage"] for i in indices}) > 1
        representative = None if mixed else min(indices, key=lambda i: (
            records[i]["source_sha256"], normalized[i]["submission_id"]))
        if representative is not None:
            kept.add(representative)
        for index in indices:
            if index != representative:
                row = normalized[index]
                exclusions.append({"source_index": index, "submission_id": row["submission_id"],
                                   "problem_id": row["problem_id"], "sabotage": row["sabotage"],
                                   "ast_sha256": key, "source_sha256": records[index]["source_sha256"],
                                   "reason": "mixed_label_ast_group" if mixed else "ast_duplicate",
                                   "retained_submission_id": None if mixed else
                                       normalized[representative]["submission_id"]})
    exclusions.sort(key=lambda r: r["source_index"])
    manifest_hash = save(root / "exclusions.json", exclusions)
    normalisation_hash = save(root / "normalisation.json", records)
    retained = [normalized[i] for i in sorted(kept)]
    v2 = {"submissions": retained, "problems": [{**p, "reference": canonical(p["reference"])[0]}
                                               for p in data["problems"]]}
    original_kept = {"problems": data["problems"],
                     "submissions": [data["submissions"][i] for i in sorted(kept)]}
    v2_hash = corpus(root / "v2" / "dataset.json", v2)
    matched_hash = corpus(root / "matched_original" / "dataset.json", original_kept)
    if len({r["ast_sha256"] for i, r in enumerate(records) if i in kept}) != len(retained):
        raise ValueError("AST deduplication failed")
    return v2, original_kept, {"source_sha256": SOURCE_SHA256, "v2_sha256": v2_hash,
        "matched_original_sha256": matched_hash, "exclusions_sha256": manifest_hash,
        "normalisation_sha256": normalisation_hash, "exclusions": dict(Counter(r["reason"] for r in exclusions)),
        "excluded_labels": dict(Counter("backdoor" if r["sabotage"] else "honest" for r in exclusions)),
        "normalized_programs": len(records), "roundtrip_equal": len(records),
        "programs_with_comments": sum(r["comments_removed"] > 0 for r in records),
        "programs_with_docstrings": sum(r["docstrings_removed"] > 0 for r in records),
        "programs_with_reflection_flags": sum(bool(r["reflection_flags"]) for r in records),
        "reflection_flags": dict(Counter(f for r in records for f in r["reflection_flags"])),
        "identifier_renaming": False}


def evaluate(path, data, destination, assignments=None):
    destination.mkdir(parents=True, exist_ok=True)
    summary = audit.audit_monitor(destination, dataset_path=path, assignments=assignments)
    rows = load(destination / "monitor_rows.json")
    labels = np.asarray([r["sabotage"] for r in data["submissions"]], dtype=bool)
    groups = np.asarray([r["problem_id"] for r in data["submissions"]])
    split = {part: np.flatnonzero([rows["split"][g] == part for g in groups])
             for part in ("train", "calibration", "evaluation")}
    for family, columns in FAMILIES.items():
        matrix = np.asarray([[r[k] for k in columns] for r in rows["features"]], dtype=float)
        name = "family_" + family
        scores, fit = audit.fit_logistic(matrix, labels, split["train"], columns)
        summary["fits"][name] = fit
        summary["metrics"][name] = audit.monitor_summary(scores, labels, groups,
                                                       split["calibration"], split["evaluation"])
        rows["scores"][name] = scores.tolist()
    rows["scores"][PRIVILEGED] = rows["privileged_scores"]
    save(destination / "monitor_rows.json", rows)
    save(destination / "summary.json", summary)
    return summary, rows


def paired_changes(before, after, original):
    """Same original problem draws for both corpora, preserving all row multiplicities."""
    ids = [r["submission_id"] for r in original["submissions"]]
    labels = np.asarray([r["sabotage"] for r in original["submissions"]], dtype=bool)
    groups = np.asarray([r["problem_id"] for r in original["submissions"]])
    split = {part: np.flatnonzero([before["split"][g] == part for g in groups])
             for part in ("calibration", "evaluation")}
    rng = np.random.default_rng(audit.SEED)
    cal, ev = split["calibration"], split["evaluation"]
    draws = list(zip(audit.clustered_draws(groups, cal, rng), audit.clustered_draws(groups, ev, rng)))
    results = {}
    for method in before["scores"]:
        aligned = []
        for row in (before, after):
            index = {sid: i for i, sid in enumerate(row["submission_ids"])}
            score = np.asarray([row["scores"][method][index[sid]] if sid in index else 0 for sid in ids])
            eligible = np.asarray([sid in index and (method != PRIVILEGED or
                                   row["privileged_eligible"][index[sid]]) for sid in ids])
            aligned.append((score, eligible))

        def difference(cb, eb):
            values = [audit.monitor_point(s, labels, cb[ok[cb]], eb[ok[eb]]) for s, ok in aligned]
            return values[1]["auroc"] - values[0]["auroc"] if all(values) else None

        results[method] = audit.interval(difference(cal, ev), [difference(cb, eb) for cb, eb in draws])
    return results


def static_sample(data, root):
    positives = sorted((r for r in data["submissions"] if r["sabotage"]), key=lambda r: r["submission_id"])
    selected = np.random.default_rng(B2_SEED).choice(len(positives), size=20, replace=False)
    results = []
    for index in selected:
        row = positives[index]
        honest = [r for r in data["submissions"] if r["problem_id"] == row["problem_id"] and not r["sabotage"]]
        results.append(repair_candidates(row, honest))
    sample_hash = save(root / "repair_candidates.json", results)
    return {"seed": B2_SEED, "draw": "numpy default_rng choice without replacement on sorted original backdoor IDs",
            "population": len(positives), "programs": len(results),
            "problems": len({r["problem_id"] for r in results}), "candidates_sha256": sample_hash,
            "programs_with_parseable_candidates": sum(bool(r["candidates"]) for r in results),
            "programs_with_exact_honest_match": sum(any(c["exact_honest_matches"] for c in r["candidates"])
                                                   for r in results),
            "source_patch_rejections": sum(r["rejected_source_patches"] for r in results),
            "dynamic_validation": False}


def summarize_review(root):
    candidates = load(root / "repair_candidates.json")
    review_path = root / "feasibility_review.json"
    review = load(review_path)
    by_id = {r["submission_id"]: r for r in review}
    if len(by_id) != len(review) or set(by_id) != {r["submission_id"] for r in candidates}:
        raise ValueError("Review must cover every sampled program exactly once")
    for item in candidates:
        decision = by_id[item["submission_id"]]
        if decision["plausible"]:
            selected = [c for c in item["candidates"] if c["candidate_id"] == decision["candidate_id"]]
            if len(selected) != 1 or not decision["rationale"]:
                raise ValueError("Plausible repairs require a checked candidate and rationale")
    labels = np.asarray([by_id[r["submission_id"]]["plausible"] for r in candidates], dtype=bool)
    groups = np.asarray([r["problem_id"] for r in candidates])
    draws = audit.clustered_draws(groups, np.arange(len(candidates)), np.random.default_rng(audit.SEED))
    result = load(OUT / "results.json")
    result["static_repair"].update({"plausibly_minimal": int(labels.sum()),
        "plausibly_minimal_fraction": audit.interval(labels.mean(), [labels[ix].mean() for ix in draws]),
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "reviewer": "one coding assistant, no independent human adjudication",
        "reason_counts": dict(Counter(r["reason"] for r in review))})
    audit.dump(OUT / "results.json", result)
    print(json.dumps(result["static_repair"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-review", action="store_true")
    args = parser.parse_args()
    root = private_root() / "research-monitor-repair"
    if root.is_relative_to(OUT.parents[2]):
        raise ValueError("Per-item files must stay outside the repository")
    if args.summarize_review:
        summarize_review(root)
        return
    started = time.monotonic()
    source = private_root() / "monitor_v0" / "dataset.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Original corpus changed")
    data = load(source)
    v2, matched, construction = build(data, root)
    feasibility = static_sample(data, root)
    summaries, rows = {}, {}
    summaries["original"], rows["original"] = evaluate(source, data, root / "original_audit")
    assignments = rows["original"]["split"]
    for name, corpus_data in (("matched_original", matched), ("v2", v2)):
        summaries[name], rows[name] = evaluate(root / name / "dataset.json", corpus_data,
                                             root / (name + "_audit"), assignments)
    # Fail rather than silently ship a non-replication of the inherited baselines.
    frozen = load(OUT.parent / "artifact_audit" / "monitor.json")
    for method, values in frozen["metrics"].items():
        if abs(summaries["original"]["metrics"][method]["auroc"]["estimate"] -
               values["auroc"]["estimate"]) > 1e-10:
            raise ValueError("Frozen baseline differs: " + method)
    differences = {"v2_minus_original": paired_changes(rows["original"], rows["v2"], data),
                   "v2_minus_matched_original": paired_changes(rows["matched_original"], rows["v2"], data)}
    compact = {name: {"n": s["n"], "problems": s["problems"], "counts": s["counts"],
                     "split": s["split"],
                     "metrics": {method: {k: v for k, v in values.items() if k == "auroc" or
                                         method in {"style_logistic", "all_logistic"}}
                                 for method, values in s["metrics"].items()},
                     "feature_diagnostics": s["feature_diagnostics"], "duplicates": s["duplicates"],
                     "within_problem_auc": s["within_problem_auc"]} for name, s in summaries.items()}
    result = {"construction": construction, "corpora": compact, "paired_auc_changes": differences,
              "feature_families": FAMILIES, "static_repair": feasibility,
              "run": {"seed": audit.SEED, "bootstrap": audit.BOOT, "python": platform.python_version(),
                      "numpy": np.__version__, "scipy": scipy.__version__,
                      "elapsed_seconds": time.monotonic() - started,
                      "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "paid_usd": 0, "dataset_code_executed": False,
                      "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (Path(__file__), OUT / "repairs.py", Path(audit.__file__))}}}
    audit.dump(OUT / "results.json", result)
    print(json.dumps({"construction": construction, "static_repair": feasibility, "run": result["run"]}))
    for method in ("style_logistic", "all_logistic", *("family_" + f for f in FAMILIES)):
        print(method, json.dumps({name: summary["metrics"][method]["auroc"] for name, summary in summaries.items()}))


if __name__ == "__main__":
    main()
