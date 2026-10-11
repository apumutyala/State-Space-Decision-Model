"""Write jevkit perturbations WITHOUT sort_keys (B-PREREG-FLIP amendment 1).

`python -m jevkit perturb make` (jevkit/cli.py line 133) serializes with json.dumps(..., sort_keys=True). Every dict is
written back in sorted key order, so option_reverse / option_shuffle rows reach the model in the base order (the public
items load with sorted criteria) and markup_noise's dict-state key reversal is undone. Those operators were no-ops on
disk in sha 655d028a. key_relabel's opt_1..opt_k keys were sorted too, which made it relabel + an implicit reorder.
This writes the same Perturbed objects (jevkit.perturb.make_perturbations, frozen, unchanged) with dict order kept.

    python -X utf8 receipts/2026-10-10-ev-flip-rate/gen_pert_nosort.py --out pert_public_seed0_nosort.jsonl \
        [--ops option_reverse,option_shuffle,key_relabel --out-subset sub.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jevkit import perturb as P  # noqa: E402
from jevkit.items import load_public  # noqa: E402


def write(ps, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for p in ps:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", default=None, help="subset operators for --out-subset")
    ap.add_argument("--out-subset", default=None)
    a = ap.parse_args(argv)
    items = load_public()
    ps = P.make_perturbations(items, None, (0,))
    write(ps, a.out)
    # self-check: a reorder row must now differ in criteria order from its base, and round-trip must keep it
    base = {it.id: it for it in items}
    bad = 0
    for line, p in zip(Path(a.out).read_text(encoding="utf-8").splitlines(), ps):
        d = json.loads(line)
        if p.op in ("option_reverse", "option_shuffle"):
            bad += list(d["question"]["criteria"]) == list(base[p.base_id].question["criteria"])
    print(json.dumps({"n": len(ps), "per_operator": Counter(p.op for p in ps), "reorder_rows_unchanged": bad}))
    if a.out_subset:
        keep = set(a.ops.split(","))
        write([p for p in ps if p.op in keep], a.out_subset)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
