"""Lever classification table for GAP_BREAKDOWN_B (reads out/gap_b.json; writes out/levers.json + out/table.md).
Measured columns come from gap_b.json (frozen scorer). Lever class, analog training family and the expected-points
rule are judgement (X) and are written out here so they can be argued with.

Expected-points rules (X):
  readout = clip(v3w_pts - v1_pts, 0, gap_D1)      (stock readout gain assumed to compose with D1 training)
  data    = 0.4 * (gap_D1 - readout)               family added / multiplied in B1 100k-v2 with a direct analog
            (0.4 = conservative vs the 0.55 median transfer ratio of in-draw families in D1)
          = max(revert, 0.4*gap) for regression families (D1 below stock): revert = -train_lift
  capacity / frozen / closed = 0
Run: python -X utf8 -I analysis/gap_b/levers.py
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import FROZEN, SOURCES, index_of, load_results, load_rows  # noqa: E402

d = json.load(open(HERE / "out" / "gap_b.json", encoding="utf-8"))
Bn, W, pr, qs = d["bench"], d["weights"], d["prior_xfit"], d["qstats"]

# analog training family: (D1 24k draw trained items, 45k D4 draw, 100k-v2 draw) -- counts from
# analysis-a/out/d1-draw-families.json and b1-draw receipts build-report.json by_family. "0*" = drawn but dropped by max_k 16.
ANALOG = {
    1: ("glaive_fc+toolace tool_select", "3,312", "10,741", "20,132"),
    2: ("tool catalogs (glaive_catalog16/long)", "0", "5,000", "15,602"),
    3: ("ext:api_bank (<=2,048 tok only)", "0", "604", "604"),
    4: ("banking77 (k16 / full / member)", "0* (1,127 drawn)", "5,000", "6,438"),
    5: ("clinc150 (k16 / member)", "0* (1,451 drawn)", "4,500", "10,000"),
    11: ("anli (sentence NLI; no doc-level NLI)", "1,617", "1,500", "6,000"),
    12: ("anli", "1,617", "1,500", "6,000"),
    21: ("humicroedit s1+s2", "1,521", "1,000", "3,000"),
    23: ("none", "0", "0", "0"),
    25: ("mmlu_aux", "5,315", "2,000", "11,416"),
    28: ("winogrande", "1,097", "1,200", "3,500"),
    29: ("hellaswag", "1,455", "1,100", "4,000"),
    30: ("gsm8k", "877", "4,238", "4,238"),
    36: ("nq retrieval", "1,616", "817", "817"),
    37: ("nq retrieval (loose)", "1,616", "817", "817"),
    38: ("none -> clinc/banking member (ACOS-style low-rate yes/no)", "0", "0", "3,275"),
    39: ("sst5 (loose)", "604", "700", "2,000"),
    40: ("none", "0", "0", "0"),
    41: ("none (stance)", "0", "0", "0"),
    42: ("none -> anli_claim (minority support)", "0", "0", "3,000"),
    43: ("ext:cruxeval synthetic", "0", "2,400", "2,400"),
    45: ("mmlu_aux (loose)", "5,315", "2,000", "11,416"),
    48: ("none", "0", "0", "0"),
    50: ("none", "0", "0", "0"),
    56: ("none", "0", "0", "0"),
    57: ("mmlu_aux / mmlu_verify", "5,315", "2,000", "11,416"),
    59: ("boolq / anli_claim (loose)", "1,227", "1,700", "7,000"),
    61: ("anli_claim (HoVer-style)", "0", "0", "3,000"),
    62: ("when2call", "895", "2,000", "3,867"),
}
# lever class + next action (X). R = readout, D = data, C = capacity/knowledge, L = context length, F = frozen, - = closed
CLS = {
    1: ("D (near-closed)", "keep BFCL in mix; do NOT adopt v3w word ids here (v3w-v1 = -0.34)"),
    2: ("L + D", "3 rows 13.7k-14.6k tok return no result; tool-catalog views are in 45k/100k-v2"),
    3: ("R + D", "v3w word ids on the trained cell (ckpt-readout job); api_bank rows >2,048 tok need the T=4096 rung"),
    4: ("D + R", "train the k>16 catalogs D1 dropped (100k-v2 has 6.4k banking rows, max_k 255)"),
    5: ("D", "clinc150_k16 7k + member 3k in 100k-v2; full 150-option catalogs still unsourced"),
    11: ("D (+train-T)", "doc-level NLI views at T=4096 (median row 3.9k tok; 72% of D1 misses are gold-rank-2)"),
    12: ("- (D1 > Clef)", "none"),
    21: ("C", "none: 1,521 in-draw records moved skill 0; every D1 miss is near-uniform (p_gold>=0.4)"),
    23: ("C", "none (per-user colour preference; Clef-flash only 12.7)"),
    25: ("- (D1 = Clef)", "none (14 questions)"),
    28: ("D + C", "3.2x winogrande in 100k-v2 + more steps; Clef-flash is 100"),
    29: ("D (regression guard)", "4,000 hellaswag in 100k-v2; -9.3 skill is within noise"),
    30: ("- (D1 > Clef)", "none (do not upweight)"),
    36: ("L", "5 rows 11.8k-21k tok give no result (Clef-flash also fails 4)"),
    37: ("R", "v3w word ids (stock v3w 25.9 vs D1 21.7, Clef-flash 28.7)"),
    38: ("D (regression) + R", "low-rate member views in 100k-v2; stock v3w 25.9 already beats Clef-flash 19.8"),
    39: ("D", "sst5 x3.3 in 100k-v2; keep v1 ids (v3w -0.07)"),
    40: ("- (D1 > Clef)", "none"),
    41: ("- (D1 > Clef)", "keep v1 ids: v3w -0.76 on stock is the largest readout risk"),
    42: ("D (regression) + calib", "anli_claim minority-support views (100k-v2); keep v1 ids (v3w -0.51)"),
    43: ("D + R", "cruxeval synthetic 2,400 (45k/100k-v2); GSM8K precedent closed >100% from 877 records"),
    45: ("C", "none (Clef-flash 6.2)"),
    48: ("C (drops in 0.3)", "none: ForecastBench leaves the 0.3 index"),
    50: ("D (regression) + C", "absent family; revert the -0.19 regression"),
    56: ("D (regression) + calib", "D1 over-calls (cross-fit prior +0.96); stock v1/v3w 52.6 > Clef-flash 50.0"),
    57: ("R + C", "v3w ids on trained cell (+0.83 on stock); mmlu_aux x1.6 + mmlu_verify (ratio was 0.13)"),
    59: ("- (D1 > Clef)", "none"),
    61: ("- (D1 = Clef)", "none"),
    62: ("D + R", "when2call x4.3 in 100k-v2; v3c +0.72 / v3w +0.28 on stock"),
}
REGRESSION = {n for n in (29, 38, 42, 50, 56)}

rows = []
in_draw_ratios = []
for n, b in Bn.items():
    n_i = int(n)
    s = b["skill"]
    if not b["frozen"] and ANALOG.get(n_i, ("", "0"))[1] not in ("0",) and not ANALOG[n_i][1].startswith("0*") and s["clef"] > s["v1"] + 5:
        in_draw_ratios.append((s["d1"] - s["v1"]) / (s["clef"] - s["v1"]))
med_ratio = statistics.median(in_draw_ratios)

tot = {"frozen": 0.0, "readout": 0.0, "data": 0.0, "capacity": 0.0, "closed": 0.0, "context": 0.0}
for n, b in sorted(Bn.items(), key=lambda kv: -kv[1]["gap_d1"]):
    n_i = int(n)
    g = b["gap_d1"]
    ro_delta = b["readout"]["v3w"]
    rec = dict(n=n_i, name=b["name"], area=b["area"], frozen=b["frozen"], gap_d1=g, se_d1=b["gap_d1_se"],
               gap_v3w=b["gap_v3w"], se_v3w=b["gap_v3w_se"], lift=b["train_lift"], ro_v3w=ro_delta,
               ro_best=b["readout_best"], prior_d1=max(pr[f"d1|label|{n}"], pr[f"d1|pos|{n}"]),
               skill=b["skill"])
    if b["frozen"]:
        rec.update(cls="F", action="measure only (frozen catalog)", exp_r=0.0, exp_d=0.0)
        tot["frozen"] += g
    else:
        cls, action = CLS[n_i]
        exp_r = min(max(g, 0), max(ro_delta, 0)) if "R" in cls else 0.0
        if cls.startswith("D") or " D" in cls or "+ D" in cls:
            rest = max(g - exp_r, 0)
            exp_d = 0.4 * rest
            if n_i in REGRESSION:
                exp_d = min(rest, max(-b["train_lift"], 0.4 * rest))
        else:
            exp_d = 0.0
        if cls.startswith("L"):
            exp_d = 0.0
        rec.update(cls=cls, action=action, exp_r=exp_r, exp_d=exp_d)
        tot["readout"] += exp_r
        tot["data"] += exp_d
        if cls.startswith("-"):
            tot["closed"] += g
        elif cls.startswith("L"):
            tot["context"] += max(g, 0)
        else:
            tot["capacity"] += max(g, 0) - exp_r - exp_d
    a = ANALOG.get(n_i)
    rec["analog"] = a
    rows.append(rec)

# context: what the no-result rows are worth (Clef-flash transplant of exactly those rows)
R_rows = load_rows()
res = {k: load_results(SOURCES[k]) for k in ("d1", "v3w", "clef")}
res["clef"] = {r["_evaluation"]["run_id"]: res["clef"][r["_evaluation"]["run_id"]] for r in R_rows}
ctx = {}
for cand in ("d1", "v3w"):
    base = index_of(R_rows, res[cand])["index"]
    t = dict(res[cand])
    for rid, x in res[cand].items():
        if x["status"] != "ok":
            t[rid] = res["clef"][rid]
    ctx[cand] = index_of(R_rows, t)["index"] - base

out = dict(rows=rows, totals=tot, median_in_draw_transfer_ratio=med_ratio, in_draw_ratios=in_draw_ratios, no_result_transplant=ctx)
json.dump(out, open(HERE / "out" / "levers.json", "w", encoding="utf-8"), indent=1, default=float)

L = ["| # | Family | Area | Frozen | D1 gap (pts ± SE) | v3w gap (pts ± SE) | D1 train lift vs v1 | v3w − v1 readout | best readout | D1 xfit prior | D1 / 45k / 100k-v2 analog rows | Lever | Exp. pts readout / data | Next action |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    a = r["analog"]
    an = f"{a[0]}: {a[1]} / {a[2]} / {a[3]}" if a else "—"
    L.append(f"| {r['n']} | {r['name']} | {r['area']} | {'**F**' if r['frozen'] else ''} | {r['gap_d1']:+.2f} ± {r['se_d1']:.2f} | "
             f"{r['gap_v3w']:+.2f} ± {r['se_v3w']:.2f} | {r['lift']:+.2f} | {r['ro_v3w']:+.2f} | {r['ro_best'][0]} {r['ro_best'][1]:+.2f} | "
             f"{r['prior_d1']:+.2f} | {an} | {r['cls']} | {r['exp_r']:.2f} / {r['exp_d']:.2f} | {r['action']} |")
(HERE / "out" / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps(dict(totals=tot, med=med_ratio, ratios=[round(x, 2) for x in in_draw_ratios], ctx=ctx), indent=1))
top = sorted([r for r in rows if not r["frozen"]], key=lambda r: -(r["exp_r"] + r["exp_d"]))[:12]
for r in top:
    print(r["n"], r["name"], r["cls"], round(r["exp_r"], 2), round(r["exp_d"], 2), round(r["gap_d1"], 2))
