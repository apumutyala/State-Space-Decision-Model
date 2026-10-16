"""Run real, pinned CPU verifiers; raw records and source snapshots stay private."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

from reference import ENDPOINTS, DEV, HELD, mutations, problems

GSM_SHA256 = "c22b81bccaaebfd1123be296e8b3c3dca3ddc704c46b70bf4a50603bb627e197"
GSM_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
PINS = {"reasoning-gym": "0.1.25", "math-verify": "0.9.0",
        "latex2sympy2-extended": "1.11.0", "antlr4-python3-runtime": "4.13.2"}


def load_verifiers(private):
    for name, pinned in PINS.items():
        assert importlib.metadata.version(name) == pinned, (name, pinned)
    import reasoning_gym
    from math_verify import parse, verify

    scores = {name: reasoning_gym.get_score_answer_fn(name) for name in ENDPOINTS[:-2]}
    scores["math_verify"] = lambda answer, entry: float(verify(parse(entry["answer"]), parse(answer)))
    source = (private / "gsm_dataset.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == GSM_SHA256
    tree = ast.parse(source)
    # Execute only the original library's two constants and two grading functions;
    # avoid importing its unrelated Torch training class. No dataset code executes.
    names = {"ANS_RE", "INVALID_ANS", "extract_answer", "is_correct"}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in names) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    assert len(nodes) == 4
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_gsm_library", "exec"), namespace)
    scores["gsm8k"] = namespace["is_correct"]
    source_map = {}
    for name, scorer in scores.items():
        if name not in ("math_verify", "gsm8k"):
            path = Path(inspect.getsourcefile(scorer))
            source_map[name] = {"function": scorer.__qualname__, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    source_map["gsm8k"] = {"commit": GSM_COMMIT, "sha256": GSM_SHA256}
    from math_verify import grader, parser
    source_map["math_verify"] = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                 for m in (grader, parser)}
    return scores, source_map


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z*z/n
    mid = (p + z*z/(2*n)) / d
    width = z*((p*(1-p)/n + z*z/(4*n*n))**0.5) / d
    return [max(0, mid-width), min(1, mid+width)]


def rate(rows, mode):
    import numpy as np
    eligible = [r for r in rows if r["valid"] == (mode == "false_rejection")]
    if not eligible:
        return None
    clusters = sorted({r["problem"] for r in eligible})
    lookup = {c: i for i, c in enumerate(clusters)}
    numer, denom = np.zeros(len(clusters)), np.zeros(len(clusters))
    for r in eligible:
        j = lookup[r["problem"]]
        error = (not r["accepted"]) if mode == "false_rejection" else (r["positive_reward"] if mode == "positive_wrong_reward" else r["accepted"])
        numer[j] += error
        denom[j] += 1
    indices = np.random.default_rng(77219).integers(0, len(clusters), size=(2000, len(clusters)))
    replicates = numer[indices].sum(axis=1) / denom[indices].sum(axis=1)
    affected = int((numer > 0).sum())
    return {"errors": int(numer.sum()), "n": int(denom.sum()),
            "rate": float(numer.sum()/denom.sum()), "ci95": np.quantile(replicates, [.025, .975]).tolist(),
            "clusters": len(clusters), "any_error_clusters": affected,
            "any_error_wilson95": wilson(affected, len(clusters))}


def summarize(rows):
    return {"false_acceptance": rate(rows, "false_acceptance"),
            "false_rejection": rate(rows, "false_rejection"),
            "positive_wrong_reward": rate(rows, "positive_wrong_reward"),
            "exceptions": dict(Counter(r["exception"] for r in rows if r["exception"])),
            "incomplete": sum(not r["complete"] for r in rows),
            "false_rejection_native": rate([r for r in rows if r["presentation"] == "native"], "false_rejection")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("dev", "held"))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    private = Path(os.environ.get("SSDM_PRIVATE", Path.home()/"ssdm/private")) / "research-verifier-adequacy"
    private.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.monotonic()
    if args.split == "held":
        # An interrupted evaluation remains consumed: do not silently rerun it.
        with (private / "HELD_EVALUATION_STARTED.json").open("x") as f:
            json.dump({"commit": commit, "run_name": args.run_name}, f)
    scores, source_map = load_verifiers(private)
    rows = []
    with (private / f"{args.run_name}.jsonl").open("x") as raw:
        for endpoint in ENDPOINTS:
            for problem in problems(endpoint):
                assert problem.correct(problem.tokens), "Canonical reference disagreement"
                for row in mutations(problem, args.split):
                    error = None
                    try:
                        score = float(scores[endpoint](row["answer"], problem.entry))
                    except Exception as exc:
                        score, error = 0.0, type(exc).__name__
                    row.update(score=score, accepted=score == 1.0, positive_reward=score > 0.0, exception=error,
                               gold=problem.entry["answer"])
                    raw.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rows.append(row)
    output = {"split": args.split, "run_name": args.run_name, "commit": commit, "pins": PINS,
              "versions": {name: importlib.metadata.version(name) for name in ("sympy", "numpy")},
              "source_map": source_map, "seed": 901016, "base_problems_per_endpoint": 64,
              "bootstrap_resamples": 2000, "bootstrap_seed": 77219, "rows": len(rows),
              "endpoints": {}, "cells": {}}
    for endpoint in ENDPOINTS:
        selected = [r for r in rows if r["endpoint"] == endpoint]
        controls = [r for r in selected if r["family"] == "canonical"]
        output["endpoints"][endpoint] = summarize([r for r in selected if r["family"] != "canonical"])
        output["endpoints"][endpoint]["canonical_accepted"] = sum(r["accepted"] for r in controls)
        output["cells"][endpoint] = {family: summarize([r for r in selected if r["family"] == family])
                                     for family in (DEV if args.split == "dev" else HELD)}
    output["wall_seconds"] = time.monotonic() - started
    output["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    output["raw_sha256"] = hashlib.sha256((private / f"{args.run_name}.jsonl").read_bytes()).hexdigest()
    with (private / f"{args.run_name}_aggregate.json").open("x") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({"rows": len(rows), "wall_seconds": output["wall_seconds"], "peak_rss_mib": output["peak_rss_mib"],
                      "canonical_accepted": {e: output["endpoints"][e]["canonical_accepted"] for e in ENDPOINTS}}))


if __name__ == "__main__":
    main()
