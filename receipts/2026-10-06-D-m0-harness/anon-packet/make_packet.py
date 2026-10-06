"""Build the anonymous DEV packet (author side; needs the suite and DEV rows).

    python -X utf8 -I receipts/2026-10-06-D-m0-harness/anon-packet/make_packet.py

Systems are the receipt's fixtures (same seeds as reproduce.py). Writes packet.json.gz.
The shuffling seed comes from os.urandom and is not recorded, so a rebuilt packet is a
different (equally valid) shuffle; verify against the committed packet, not a rebuild."""
import random
from pathlib import Path

from harness import packet, predictors
from harness.cases import case_of
from harness.cli import load_slice

rows = load_slice("dev")
perfect = predictors.predict(rows, "perfect", seed=1)
base = predictors.predict(rows, "noisy", seed=6, acc=0.55)
cid = case_of(rows)
case_by_rid = {r["_evaluation"]["run_id"]: cid(r) for r in rows}


def improve(frac, seed):
    rng = random.Random(seed)
    return {k: (perfect[k] if rng.random() < frac else v) for k, v in base.items()}


def improve_by_case(frac, seed):
    rng = random.Random(seed)
    chosen = {c for c in sorted(set(case_by_rid.values())) if rng.random() < frac}
    return {k: (perfect[k] if case_by_rid[k] in chosen else v) for k, v in base.items()}


systems = {
    "perfect": perfect,
    "uniform": predictors.predict(rows, "uniform", seed=2),
    "noisy-0.7": predictors.predict(rows, "noisy", seed=3, acc=0.7, status_drop=0.02),
    "base-0.55": base,
    "independent-0.60": predictors.predict(rows, "noisy", seed=7, acc=0.60),
    "base+11pct_rows_fixed": improve(0.11, 21),
    "base+4pct_rows_fixed": improve(0.04, 22),
    "base+11pct_cases_fixed": improve_by_case(0.11, 23),
}
p = packet.build(rows, systems)
out = Path(__file__).with_name("packet.json.gz")
packet.save(p, out)
print(out, out.stat().st_size)
