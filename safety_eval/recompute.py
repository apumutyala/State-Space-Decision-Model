"""Recompute metrics from a committed predictions.jsonl (CPU, no model).

    python -m safety_eval.recompute receipts/2026-10-06-D-m0-steer/predictions.jsonl --out .../recompute_v0.3.json

Fills fields added after the run:
  * ``template_split`` from the template name;
  * ``unit_sha`` by REBUILDING every variant deterministically from the generator (data.build + build_variants,
    final=True) and the committed gcg_attacks.jsonl suffixes, then hashing it (metrics.unit_fingerprint). A row whose
    request_id cannot be rebuilt gets no hash, so any arm comparison on it fails loudly.
Reports every arm's summary (incl. D1-A / D1-B, completeness) and paired arm-vs-arm differences. Control arms
were run on a predeclared subset (clean + role-spoof hijacks), so their comparisons use that subset as
``expected_ids`` (declared, not an inner join).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import attacks, data, metrics


def rebuild_variants(manifest_n: tuple[int, int], gcg_path: Path | None) -> list[dict]:
    reqs = data.build(*manifest_n, split="dev")
    vs = attacks.build_variants(reqs, final=True)
    if gcg_path and gcg_path.exists():
        by_id = {r["request_id"]: r for r in reqs}
        for line in open(gcg_path, encoding="utf-8"):
            g = json.loads(line)
            vs.append(attacks.suffix_variant(by_id[g["request_id"]], g["suffix"], f"gcg_{g['objective']}", "gcg"))
    return vs


def expected_for(arm_rows: list[dict], variants: list[dict], canonical: bool) -> list[dict]:
    """Predeclared expected unit set: the canonical arm was run on every variant (incl. GCG suffixes); control
    arms on clean + role-spoof hijack variants only (run_smoke --controls)."""
    if canonical:
        return variants
    return [v for v in variants if v["_meta"]["attack_family"] == "clean" or v["_meta"].get("role_spoof")]


def load(path: Path, n_route: int = 120, n_pay: int = 80, return_variants: bool = False):
    variants = rebuild_variants((n_route, n_pay), path.parent / "gcg_attacks.jsonl")
    fp = {v["request_id"]: metrics.unit_fingerprint(v) for v in variants}
    arms = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("attack_family") == "d1" and not r.get("template_split"):
            r["template_split"] = attacks.d1_split(r["template"])
        if not r.get("unit_sha") and r["request_id"] in fp:
            r["unit_sha"] = fp[r["request_id"]]
        arms[r.get("arm", "default")].append(r)
    return (dict(arms), variants) if return_variants else dict(arms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n-route", type=int, default=120)
    ap.add_argument("--n-pay", type=int, default=80)
    a = ap.parse_args(argv)
    arms, variants = load(Path(a.predictions), a.n_route, a.n_pay, return_variants=True)
    names = sorted(arms, key=lambda k: -len(arms[k]))  # largest arm = canonical/reference
    ref = names[0]
    out = {"source": a.predictions,
           "arms": {k: metrics.summarise(v, a.boot, expected=expected_for(v, variants, k == ref))
                    for k, v in arms.items()},
           "arm_vs_arm": {}}
    for other in names[1:]:
        fams = sorted({r["attack_family"] for r in arms[other]} - {"clean"})
        for f in fams:
            expected = {r["request_id"] for r in arms[other] if metrics._select(r, f)}  # declared control subset
            out["arm_vs_arm"][f"{other} - {ref} | {f} | eligibility=reference_y"] = metrics.compare_arms(
                arms[other], arms[ref], f, a.boot, eligibility="reference_y", expected_ids=expected)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in out["arms"].items():
        fam = v["families"]
        print(k, {f: (round(fam[f]["steer_rate"]["point"], 4), fam[f]["complete"], fam[f]["outcomes"])
                  for f in ("d1", "d1_A", "d1_B", "hijack") if f in fam})
    for k, v in out["arm_vs_arm"].items():
        print(k, "complete", v["complete"], {m: round(v[m]["diff"]["point"], 4) for m in ("hijack_rate", "steer_rate")})


if __name__ == "__main__":
    main()
