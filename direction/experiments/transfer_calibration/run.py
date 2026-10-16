"""Can a readout bias estimated on synthetic, non-public items fix the one-pass "yes" prior on public items?

Free (CPU): uses cached trust-mode one-pass probabilities only, no model calls.
- **Fit:** label bias is estimated on the 192 held-out *synthetic* clean items of the distillation pilot
  (`$SSDM_PRIVATE/distill_v0`: repo generators, never public). Two corrections, each estimated per model:
  - yes/no: one log-odds offset that maximises balanced accuracy on the synthetic yes/no items;
  - choice: per-letter-position mean log-probability ("calibrate before use" style), subtracted at read time.
- **Evaluate:** the frozen corrections are applied once to the public-231 clean items (EVAL-ONLY) read from the Card
  score caches (`$SSDM_PRIVATE/card` for stock, `$SSDM_PRIVATE/card-student` for the student). Accuracy before and
  after is reported with a family-cluster bootstrap.

    python -X utf8 direction/experiments/transfer_calibration/run.py   # writes results.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from jevkit.items import load_public  # noqa: E402
from keel.paths import private_root  # noqa: E402
from keel.protocol import DecisionRequest, from_jevbench_item, render  # noqa: E402

MODELS = {"stock": ("qwen3.5-4b", "stock_4b.json", "card"),
          "student": ("/results/students/distill-v0-20261015/merged", "student_4b.json", "card-student")}
CARD_SEED, CARD_MAX_THINK = 20261015, 3072


def synthetic(name: str) -> list[dict]:
    root = private_root() / "distill_v0"
    units = {u["id"]: u for u in map(json.loads, gzip.open(root / "teacher-pilot-screened/units.jsonl.gz", "rt"))
             if u["kind"] == "clean"}
    outputs = json.loads((root / "pilot" / MODELS[name][1]).read_text())["outputs"]
    rows = []
    for uid, u in units.items():
        req = DecisionRequest.from_json(u["request"])
        r = render(req, mode="trust", think=False)
        probs = outputs[uid]["probs"]
        rows.append({"type": req.questions[0].type, "ids": list(r.option_ids),
                     "lp": [math.log(max(probs[k], 1e-12)) for k in r.option_ids], "gold": str(u["gold"]),
                     "cluster": u["group"]})
    return rows


def public(name: str) -> list[dict]:
    model, _, cache = MODELS[name]
    rows = []
    for it in load_public():
        if it.type == "score":
            continue
        req = from_jevbench_item(it.to_dict())
        r = render(req, mode="trust", think=False)
        key = hashlib.sha256(json.dumps({"version": 1, "model": model, "arm": "one_pass_trust", "prompt": r.prompt,
                                         "labels": r.labels, "seed": CARD_SEED, "max_think": CARD_MAX_THINK,
                                         "n_samples": 1}, sort_keys=True).encode()).hexdigest()
        packed = json.loads((private_root() / cache / "scores" / f"{key}.json").read_text())
        gold = {"yes": "true", "no": "false"}.get(str(it.expected), str(it.expected)) if it.type == "noul" else str(it.expected)
        rows.append({"type": it.type, "ids": list(r.option_ids), "tier": it.tier, "cluster": it.family,
                     "lp": [math.log(max(p, 1e-12)) for p in packed["probs"]], "gold": gold})
    return rows


def yes_logodds(row: dict) -> float:
    lp = dict(zip(row["ids"], row["lp"]))
    return lp["true"] - lp["false"]


def fit_offset(rows: list[dict], objective: str = "balanced") -> float:
    """Offset t (decide yes iff log-odds > t) maximising balanced (or plain) accuracy on labelled yes/no rows."""
    yn = [(yes_logodds(r), r["gold"] in ("true", "yes")) for r in rows if r["type"] == "noul"]
    cands = sorted({z for z, _ in yn})
    cands = [cands[0] - 1] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1]

    def score(t: float) -> float:
        pos = [z > t for z, y in yn if y]
        neg = [z <= t for z, y in yn if not y]
        if objective == "accuracy":
            return (sum(pos) + sum(neg)) / len(yn)
        return (sum(pos) / max(len(pos), 1) + sum(neg) / max(len(neg), 1)) / 2

    return max(cands, key=lambda t: (score(t), -abs(t)))


def training_yes_rates() -> dict[str, dict[str, float]]:
    """Gold yes rate of the screened yes/no training labels the student was distilled on, per variant."""
    labels = gzip.open(private_root() / "distill_v0/teacher-train-screened/labels.jsonl.gz", "rt")
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for line in labels:
        row = json.loads(line)
        if row["request"]["questions"][0]["type"] == "noul":
            by_kind[row["kind"]].append(str(row["gold"]) == "true")
    return {kind: {"n": len(v), "gold_yes": sum(v) / len(v)} for kind, v in sorted(by_kind.items())}


def fit_position_bias(rows: list[dict]) -> dict[int, float]:
    """Mean log-prob per letter position on synthetic choice rows (renormalised within each row first)."""
    sums, counts = defaultdict(float), defaultdict(int)
    for r in rows:
        if r["type"] != "choice":
            continue
        m = max(r["lp"])
        z = m + math.log(sum(math.exp(x - m) for x in r["lp"]))
        for i, x in enumerate(r["lp"]):
            sums[i] += x - z
            counts[i] += 1
    mean = {i: sums[i] / counts[i] for i in sums}
    centre = sum(mean.values()) / len(mean)
    return {i: v - centre for i, v in mean.items()}


def decide(row: dict, offset: float | None, bias: dict[int, float] | None) -> str:
    if row["type"] == "noul":
        return "true" if yes_logodds(row) > (offset or 0.0) else "false"
    lp = [x - (bias.get(i, 0.0) if bias else 0.0) for i, x in enumerate(row["lp"])]
    return row["ids"][max(range(len(lp)), key=lp.__getitem__)]


def cluster_ci(correct: dict[str, list[bool]], draws: int = 4000, seed: int = 0) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    keys = sorted(correct)
    point = sum(sum(v) for v in correct.values()) / sum(len(v) for v in correct.values())
    stats = []
    for _ in range(draws):
        pick = [rng.choice(keys) for _ in keys]
        n = sum(len(correct[k]) for k in pick)
        stats.append(sum(sum(correct[k]) for k in pick) / n)
    stats.sort()
    return point, [stats[int(0.025 * draws)], stats[int(0.975 * draws)]]


def paired_ci(before: dict[str, list[bool]], after: dict[str, list[bool]], draws: int = 4000, seed: int = 1):
    rng = random.Random(seed)
    keys = sorted(before)
    diffs = []
    for _ in range(draws):
        pick = [rng.choice(keys) for _ in keys]
        n = sum(len(before[k]) for k in pick)
        diffs.append((sum(sum(after[k]) for k in pick) - sum(sum(before[k]) for k in pick)) / n)
    diffs.sort()
    point = (sum(sum(v) for v in after.values()) - sum(sum(v) for v in before.values())) / sum(len(v) for v in before.values())
    return point, [diffs[int(0.025 * draws)], diffs[int(0.975 * draws)]]


def main() -> None:
    out = {"protocol": __doc__.split("\n\n")[0], "training_yes_rates": training_yes_rates(), "models": {}}
    for name in MODELS:
        syn, pub = synthetic(name), public(name)
        offset, bias = fit_offset(syn), fit_position_bias(syn)
        syn_yn = [r for r in syn if r["type"] == "noul"]
        res = {"fit": {"synthetic_yes_no_items": len(syn_yn), "offset": offset,
                       "synthetic_pred_yes_rate": sum(yes_logodds(r) > 0 for r in syn_yn) / len(syn_yn),
                       "synthetic_gold_yes_rate": sum(r["gold"] in ("true", "yes") for r in syn_yn) / len(syn_yn),
                       "position_bias": {str(k): round(v, 4) for k, v in sorted(bias.items())}}}
        for typ in ("noul", "choice"):
            rows = [r for r in pub if r["type"] == typ]
            before, after = defaultdict(list), defaultdict(list)
            for r in rows:
                before[r["cluster"]].append(decide(r, None, None) == r["gold"])
                after[r["cluster"]].append(decide(r, offset if typ == "noul" else None,
                                                  bias if typ == "choice" else None) == r["gold"])
            b, bci = cluster_ci(before)
            a, aci = cluster_ci(after)
            d, dci = paired_ci(before, after)
            entry = {"n": len(rows), "acc_before": b, "ci_before": bci, "acc_after": a, "ci_after": aci,
                     "delta": d, "delta_ci": dci}
            if typ == "noul":
                entry["pred_yes_before"] = sum(yes_logodds(r) > 0 for r in rows) / len(rows)
                entry["pred_yes_after"] = sum(yes_logodds(r) > offset for r in rows) / len(rows)
                entry["gold_yes"] = sum(r["gold"] == "true" for r in rows) / len(rows)
                hard = [r for r in rows if r["tier"] == "hard"]
                entry["hard_acc_before"] = sum(decide(r, None, None) == r["gold"] for r in hard) / len(hard)
                entry["hard_acc_after"] = sum(decide(r, offset, None) == r["gold"] for r in hard) / len(hard)
                # Descriptive ceiling only: the best single offset chosen WITH public labels. Not a method.
                ceiling = fit_offset(rows, objective="accuracy")
                entry["ceiling_offset_descriptive"] = ceiling
                entry["ceiling_acc_descriptive"] = sum(decide(r, ceiling, None) == r["gold"] for r in rows) / len(rows)
            res[f"public_{typ}"] = entry
        out["models"][name] = res
        for typ in ("noul", "choice"):
            e = res[f"public_{typ}"]
            print(f"{name:8s} {typ:6s} acc {e['acc_before']:.3f} -> {e['acc_after']:.3f} "
                  f"(delta {e['delta']:+.3f}, 95% CI {e['delta_ci'][0]:+.3f}..{e['delta_ci'][1]:+.3f})"
                  + (f" ceiling {e['ceiling_acc_descriptive']:.3f} at t={e['ceiling_offset_descriptive']:.2f}"
                     if typ == "noul" else ""))
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
