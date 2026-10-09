"""The DIAGNOSTIC report of jobs/q9af_v2 mode clefflash (README_clefflash.md, ``q9af-v2-clefflash-diag-v1``).

    python -m jobs.q9af_v2.diag <results.json> <index dir> <export.json> <decision.json>

NO keep / discard / promote rule: the job measures what removing attention costs at Clef-flash's level. This module only
gathers the pre-registered numbers into decision.json and says whether the measurement is complete (verdict
DIAGNOSTIC) or not (NO_MEASUREMENT, exit 9). Pre-registered metrics (proxy600, 644 rows, ids sha 7d42c398...,
B 2000, seed 0; edition 0.2.1 primary, 0.3 (--omit-gsm8k03) alongside):

  pair A (primary): Clef-flash's OFFICIAL joint head (baselines.clef_index_engine, gdn_path=reference):
      stock Clef-flash  vs  converted Clef-flash (all 8 attention layers -> GDN, the head unchanged)
      -> delta_A = converted - stock, paired (index/compare_clef_conv_vs_stock[_03].json)
  pair B: the LM-logit v3w readout (make_v3w qwen3.5-9b; next-token logits over each option's single token):
      stock Clef-flash backbone  vs  converted backbone -> delta_B (index/compare_v3w_conv_vs_stock[_03].json).
      Only B's converted arm is a fully attention-free system (the joint head cross-attends over positions).
  plus the four arm scores, the held-out mixture KL of the conversion, and the export identity.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RULE_VERSION = "q9af-v2-clefflash-diag-v1"
ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
PROXY_IDS_SHA256 = "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6"
BOOTSTRAP_B, BOOTSTRAP_SEED = 2000, 0
VERDICTS = ("NO_MEASUREMENT", "DIAGNOSTIC")
ARMS = ("clef_stock", "clef_conv", "v3w_stock", "v3w_conv")
PAIRS = {"A_joint_head": ("clef_conv", "clef_stock"), "B_v3w_lm_logit": ("v3w_conv", "v3w_stock")}
EDITIONS = {"0.2.1": "", "0.3": "_03"}
BINDING = [f"delta_{p}_{e}_{k}" for p in PAIRS for e in EDITIONS for k in ("point", "lo", "hi")] + \
          [f"score_{a}_{e}" for a in ARMS for e in EDITIONS] + ["heldout_mix_kl"]
ACTION = "none: diagnostic only (no keep/discard rule); report both paired deltas and the four scores"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def decide(res: dict | None, idx: Path, exp: dict | None) -> dict:
    problems, numbers = [], {}
    if not (exp and exp.get("ok") and exp.get("mode") == "final" and exp.get("attention_free")
            and exp.get("converted") == ATTENTION_LAYERS):
        problems.append("export is not a final attention-free export of all 8 attention layers")
    kl = (((res or {}).get("final_heldout") or {}).get("mix") or {}).get("kl")
    numbers["heldout_mix_kl"] = kl
    for ed, sfx in EDITIONS.items():
        for pair, (cand, base) in PAIRS.items():
            c = _load(idx / f"compare_{cand}_vs_stock{sfx}.json")  # compare_clef_conv_vs_stock, compare_v3w_conv_vs_stock
            if c is None:
                problems.append(f"{pair} {ed}: no compare file")
                continue
            if c.get("subset_ids_sha256") != PROXY_IDS_SHA256:
                problems.append(f"{pair} {ed}: compare not on proxy600")
            if c.get("B") != BOOTSTRAP_B or c.get("seed") != BOOTSTRAP_SEED:
                problems.append(f"{pair} {ed}: bootstrap is not B {BOOTSTRAP_B} / seed {BOOTSTRAP_SEED}")
            d = (c.get("diff") or {}).get("index") or {}
            for k in ("point", "lo", "hi"):
                numbers[f"delta_{pair}_{ed}_{k}"] = d.get(k)
        for arm in ARMS:
            s = _load(idx / f"score_{arm}{sfx}.json")
            numbers[f"score_{arm}_{ed}"] = (((s or {}).get("slice_index") or {}).get("index") or {}).get("point")
    missing = [k for k in BINDING if not _num(numbers.get(k))]
    if missing:
        problems.append(f"missing numbers: {missing}")
    verdict = "NO_MEASUREMENT" if problems else "DIAGNOSTIC"
    return {"rule_version": RULE_VERSION, "verdict": verdict, "action": ACTION if not problems else "diagnose the run",
            "problems": problems, "numbers": numbers, "attention_free": bool(exp and exp.get("attention_free")),
            "attention_free_system": {"A_joint_head": False, "B_v3w_lm_logit": True},
            "report": {"binding": BINDING, "pairs": {p: {"candidate": c, "baseline": b} for p, (c, b) in PAIRS.items()},
                       "primary": "delta_A_joint_head_0.2.1", "keep_discard_rule": None,
                       "proxy_ids_sha256": PROXY_IDS_SHA256, "B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED}}


def valid_decision(d: dict) -> bool:
    """finalize.py validator: a complete diagnostic (every pre-registered number present)."""
    return (isinstance(d, dict) and d.get("rule_version") == RULE_VERSION and d.get("verdict") == "DIAGNOSTIC"
            and all(_num(d["numbers"].get(k)) for k in d["report"]["binding"]))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        print("usage: python -m jobs.q9af_v2.diag <results.json> <index dir> <export.json> <decision.json>")
        return 2
    d = decide(_load(Path(argv[0])), Path(argv[1]), _load(Path(argv[2])))
    Path(argv[3]).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({k: d[k] for k in ("verdict", "problems", "numbers")}))
    return 0 if d["verdict"] == "DIAGNOSTIC" else 9


if __name__ == "__main__":
    sys.exit(main())
