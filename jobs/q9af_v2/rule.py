"""Pre-registered decision rule of jobs/q9af_v2 (rule q9af-v2-rule-v1). Committed with the job, before any v2 run or data.

    python -m jobs.q9af_v2.rule <results.json> <compare_proxy600.json | -> <export.json> <decision.json>

Replaces full8's per-layer 0.95 hidden-cosine gate (job-2xe5w: it rejected L3 on cosine alone while L3's step dNLL
was NEGATIVE) with an END-TO-END gate on the whole attention-free model:

1. **Attention-free by construction.** The export must cover exactly the 8 attention layers (else NO_MEASUREMENT).
2. **Held-out generic KL** (``results.final_heldout.mix.kl``: KL(teacher || converted) in nats/token on the held-out
   mixture, token-weighted over web / chat / math) <= MAX_HELDOUT_KL. A model this far from its teacher is broken,
   whatever a 644-row screen says. (Context, not a constant source: job-2xe5w's 6-layer hybrid had KL 0.094 on
   wikitext and still lost 26 Index points, so KL alone can never ADVANCE anything.)
3. **Stock Index proxy600 screen** (paired, converted minus stock, both arms in this job through the same readout
   code path, ``jobs.common.dev_subset compare --subset jobs/common/dev_proxy600.json``, edition 0.2.1, B 2000 seed 0).

Why these thresholds (all from receipts that predate this rule; none from a v2 number):
  stock Qwen3.5-9B DEV subset 38.27; current champion 38.67 (RWKV7 D1 + v3w, job-39gnn); measured TRAINING lifts:
  our cell recipe on RWKV7 +7.60 over its best stock readout (31.07 -> 38.67) and Clef-flash +17.51 over stock Q9
  (38.27 -> 55.78). The midpoint lift is ~+12.6.
  ADVANCE_POINT -10: a converted model at stock - 10 (~28.3) needs a +10.4 lift to beat the champion: between our
     measured +7.6 and Clef's +17.5, i.e. a cell is a fair bet.
  MARGINAL_POINT -17: at stock - 17 (~21.3) only a Clef-level lift (+17.5) reaches the champion; decision-needed.
  STRONG: point >= -4 and CI low >= -8 (near-lossless conversion): a cell is an expected win.
  Q9-H2 (job-2xe5w, 6 layers) was at -26.0: FOLD under every threshold here.

Verdicts (first match): NO_MEASUREMENT, FOLD_KL, STRONG, ADVANCE, MARGINAL, FOLD. Each maps to one action (ACTIONS).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RULE_VERSION = "q9af-v2-rule-v1"
ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
MAX_HELDOUT_KL = 0.25
STRONG_POINT = -4.0
STRONG_LO = -8.0
ADVANCE_POINT = -10.0
MARGINAL_POINT = -17.0
PROXY_IDS_SHA256 = "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6"
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
VERDICTS = ("NO_MEASUREMENT", "FOLD_KL", "STRONG", "ADVANCE", "MARGINAL", "FOLD")
ACTIONS = {
    "NO_MEASUREMENT": "no verdict: diagnose the run; nothing is pinned",
    "FOLD_KL": "fold v2 as converted; the export is archived, not pinned; next lever = more KD tokens / trainable MLP",
    "STRONG": "pin gdn_converted.pt for jobs/q9af_cell (jobs.q9af_cell.pin fill) in its own commit; run the full "
              "2,008-row subset stock arm (binding replication) and the r64 cell",
    "ADVANCE": "pin gdn_converted.pt for jobs/q9af_cell in its own commit; run the r64 cell (fork head, v3w-equivalent "
               "readout, anti-collapse settings) and its proxy600 screen",
    "MARGINAL": "decision-needed: a cell needs a Clef-level lift (+17.5) to beat the champion; report and ask",
    "FOLD": "fold v2 as converted; the export is archived, not pinned; log in DEAD_ENDS with the numbers",
}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def numbers(results: dict, cmp: dict | None) -> dict:
    fh = (results.get("final_heldout") or {}).get("mix") or {}
    d = ((cmp or {}).get("diff") or {}).get("index") or {}
    st, cv = results.get("probe_stock") or {}, results.get("probe_conv8") or {}
    mf = results.get("mmlu_full") or {}
    acc_s, acc_c = (mf.get("stock") or {}).get("acc"), (mf.get("final") or {}).get("acc")
    num = {"heldout_kl": fh.get("kl"), "heldout_delta_nll": fh.get("delta_nll"), "heldout_hidden_cos": fh.get("hidden_cos"),
           "proxy_diff_point": d.get("point"), "proxy_diff_lo": d.get("lo"), "proxy_diff_hi": d.get("hi"),
           "mmlu_full_drop_pts": 100 * (acc_s - acc_c) if _num(acc_s) and _num(acc_c) else None,
           "mmlu285_drop_pts": (100 * (st["mmlu"]["acc"] - cv["mmlu"]["acc"])
                                if _num((st.get("mmlu") or {}).get("acc")) and _num((cv.get("mmlu") or {}).get("acc"))
                                else None)}
    for ctx in ("ctx1024", "ctx4096"):
        a, b = (cv.get("ppl") or {}).get(ctx, {}).get("ppl"), (st.get("ppl") or {}).get(ctx, {}).get("ppl")
        num[f"wikitext_ppl_ratio_{ctx}"] = a / b if _num(a) and _num(b) and b > 0 else None
    for k in ("mqar", "niah"):
        a, b = (cv.get(k) or {}).get("mean_acc"), (st.get(k) or {}).get("mean_acc")
        num[f"{k}_drop"] = b - a if _num(a) and _num(b) else None
    return num


def decide(results: dict, cmp: dict | None, export: dict | None) -> dict:
    num = numbers(results, cmp)
    problems = []
    conv = sorted((export or {}).get("converted") or [])
    if not (export or {}).get("ok") or (export or {}).get("mode") != "final" or conv != ATTENTION_LAYERS \
            or not (export or {}).get("attention_free"):
        problems.append(f"export is not the full attention-free final export (converted {conv})")
    if cmp is not None:
        if cmp.get("subset_ids_sha256") != PROXY_IDS_SHA256:
            problems.append(f"compare is not on proxy600 ({cmp.get('subset_ids_sha256')})")
        if cmp.get("B") != BOOTSTRAP_B or cmp.get("seed") != BOOTSTRAP_SEED:
            problems.append(f"compare bootstrap B={cmp.get('B')} seed={cmp.get('seed')} != {BOOTSTRAP_B}/{BOOTSTRAP_SEED}")
    for k in ("heldout_kl", "proxy_diff_point", "proxy_diff_lo"):
        if not _num(num.get(k)):
            problems.append(f"{k} missing")
    if problems:
        v = "NO_MEASUREMENT"
    elif num["heldout_kl"] > MAX_HELDOUT_KL:
        v = "FOLD_KL"
    elif num["proxy_diff_point"] >= STRONG_POINT and num["proxy_diff_lo"] >= STRONG_LO:
        v = "STRONG"
    elif num["proxy_diff_point"] >= ADVANCE_POINT:
        v = "ADVANCE"
    elif num["proxy_diff_point"] >= MARGINAL_POINT:
        v = "MARGINAL"
    else:
        v = "FOLD"
    s2 = results.get("stage2") or {}
    return {"rule_version": RULE_VERSION, "verdict": v, "action": ACTIONS[v], "problems": problems, "numbers": num,
            "attention_free": conv == ATTENTION_LAYERS and bool((export or {}).get("attention_free")), "converted": conv,
            "report": {"binding": ["heldout_kl", "proxy_diff_point", "proxy_diff_lo"],
                       "init": (results.get("init_cmp") or {}).get("chosen"),
                       "stage1_tokens": (results.get("stage1") or {}).get("tokens"),
                       "stage2_tokens": s2.get("tokens"), "stage2_curve": s2.get("curve"),
                       "heldout_per_source": {k: v_ for k, v_ in (results.get("final_heldout") or {}).items()},
                       "wikitext_gate": results.get("wikitext_gate"),
                       "proxy_candidate": (cmp or {}).get("candidate"), "proxy_baseline": (cmp or {}).get("baseline")}}


def valid_decision(d: dict) -> bool:
    """finalize.py validator: a real verdict (NO_MEASUREMENT is not one) with its binding numbers present."""
    return (isinstance(d, dict) and d.get("rule_version") == RULE_VERSION and d.get("verdict") in VERDICTS
            and d["verdict"] != "NO_MEASUREMENT" and all(_num(d["numbers"].get(k)) for k in d["report"]["binding"]))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        print("usage: python -m jobs.q9af_v2.rule <results.json> <compare.json|-> <export.json> <decision.json>")
        return 2
    res = json.loads(Path(argv[0]).read_text())
    cmp = None if argv[1] == "-" else json.loads(Path(argv[1]).read_text())
    exp = json.loads(Path(argv[2]).read_text()) if Path(argv[2]).exists() else None
    d = decide(res, cmp, exp)
    Path(argv[3]).write_text(json.dumps(d, indent=1))
    print(json.dumps({k: d[k] for k in ("verdict", "action", "problems", "numbers")}))
    return 0 if d["verdict"] != "NO_MEASUREMENT" else 9


if __name__ == "__main__":
    sys.exit(main())
