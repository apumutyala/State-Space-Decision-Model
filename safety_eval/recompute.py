"""Recompute metrics from a committed predictions.jsonl (CPU, no model).

    python -m safety_eval.recompute receipts/2026-10-06-D-m0-steer/predictions.jsonl --out .../recompute_v0.2.json

Fills fields added after the run (``template_split`` from the template name), then reports every arm's summary
(incl. D1-A / D1-B) and the paired arm-vs-arm differences for each pair of arms on the families they share.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import attacks, metrics


def load(path: Path) -> dict[str, list[dict]]:
    arms = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("attack_family") == "d1" and not r.get("template_split"):
            r["template_split"] = attacks.d1_split(r["template"])
        arms[r.get("arm", "default")].append(r)
    return dict(arms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    arms = load(Path(a.predictions))
    out = {"source": a.predictions, "arms": {k: metrics.summarise(v, a.boot) for k, v in arms.items()},
           "arm_vs_arm": {}}
    names = sorted(arms)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            fams = sorted({r["attack_family"] for r in arms[x]} & {r["attack_family"] for r in arms[y]} - {"clean"})
            for f in fams:
                out["arm_vs_arm"][f"{x} - {y} | {f}"] = metrics.compare_arms(arms[x], arms[y], f, a.boot)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in out["arms"].items():
        fam = v["families"]
        line = {f: round(fam[f]["steer_rate"]["point"], 4) for f in ("d1", "d1_A", "d1_B") if f in fam}
        print(k, "steer", line, {f: [round(fam[f]["steer_rate"]["lo"], 3), round(fam[f]["steer_rate"]["hi"], 3)]
                                  for f in ("d1_A", "d1_B") if f in fam})
    for k, v in out["arm_vs_arm"].items():
        print(k, {m: round(v[m]["diff"]["point"], 4) for m in ("hijack_rate", "steer_rate")},
              [round(v["hijack_rate"]["diff"]["lo"], 3), round(v["hijack_rate"]["diff"]["hi"], 3)])


if __name__ == "__main__":
    main()
