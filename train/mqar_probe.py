"""MQAR-style recall probe records for the nuPI recall constraint (docs/NUPI.md, train/constraints.py).

Multi-query associative recall (Arora et al., Zoology 2023) posed in the decision-record format: the state lists
``n_pairs`` key=value bindings, and each of ``n_queries`` choice questions asks for one key's value among the
``n_values`` value symbols. A model that keeps the bindings in its state answers every query; the probe loss (the
decision CE on these rows) rises when a conversion cell (attention -> SSM) loses recall while its KL to the parent
stays low (CONTROL_RESEARCH.md #4). Synthetic, seeded, disjoint from every Index / JevBench item by construction.

    python -X utf8 -m train.mqar_probe --n 512 --pairs 16 --queries 4 --seed 0 --out probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any


def mqar_records(n: int, n_pairs: int = 16, n_queries: int = 4, n_values: int = 8, seed: int = 0,
                 n_keys: int = 64) -> list[dict[str, Any]]:
    if not (1 <= n_queries <= n_pairs <= n_keys and n_values >= 2):
        raise ValueError("need 1 <= n_queries <= n_pairs <= n_keys and n_values >= 2")
    rng = random.Random(f"mqar:{seed}")
    values = [f"v{j}" for j in range(n_values)]
    recs = []
    for i in range(n):
        keys = rng.sample([f"k{j}" for j in range(n_keys)], n_pairs)
        bind = {k: rng.choice(values) for k in keys}
        asked = rng.sample(keys, n_queries)
        recs.append({
            "id": f"mqar{seed}-{i}",
            "request": {
                "model": "mqar-probe",
                "state": "; ".join(f"{k}={bind[k]}" for k in keys),
                "questions": {f"q{q}": {"type": "choice", "instructions": f"value of {k}?",
                                        "criteria": {v: None for v in values}} for q, k in enumerate(asked)},
            },
            "labels": {f"q{q}": bind[k] for q, k in enumerate(asked)},
        })
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--values", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in mqar_records(a.n, a.pairs, a.queries, a.values, a.seed):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
