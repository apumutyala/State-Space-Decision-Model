"""EVIDENCE_REPORT_V2 charts. Every number is transcribed from a receipt cited in docs/EVIDENCE_REPORT_V2.md.

Run: python docs/evidence_report_v2/charts_v2.py docs/evidence_report_v2
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
OURS, EXT, BASE, HOST = "#0969da", "#8c959f", "#bf3989", "#1a7f37"
I_C, C_C, S_C, K_C = "#0969da", "#54aeff", "#bf8700", "#8250df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


# 1. Target status board (status per target, from the V2 summary table)
targets = [
    ("0 Kit parity (public-231 cards)", "PROVEN"),
    ("1 Composite vs H2O (paired local)", "FALSIFIED"),
    ("2 Capability vs best downloadable", "FALSIFIED"),
    ("3 MD1 (reported numbers)", "NOT MET / NOT COMPARABLE"),
    ("4 Clef DI full DEV", "FALSIFIED"),
    ("5a Reorder+relabel invariance", "PROVEN"),
    ("5b Conformal machinery", "PROVEN"),
    ("5b 'D19 more actionable than Clef'", "FALSIFIED"),
    ("5b FS-CRR on unseen split", "OPEN"),
    ("5c Injection-robust sets (CPU)", "PROVEN"),
    ("5c Trained G16 + GPU G15", "OPEN"),
    ("5d State stability", "PARTLY PROVEN"),
    ("5e nuPI / consistency", "OPEN"),
    ("6 Leak-free data", "PROVEN"),
    ("7 Independent rescores", "PROVEN"),
]
col = {"PROVEN": "#1a7f37", "PARTLY PROVEN": "#4ac26b", "FALSIFIED": "#cf222e",
       "NOT MET / NOT COMPARABLE": "#fb8f44", "OPEN": "#8c959f"}
fig, ax = plt.subplots(figsize=(7.6, 5.4))
for i, (t, s) in enumerate(targets):
    ax.barh(i, 1, color=col[s], height=0.7)
    ax.text(0.03, i, s, va="center", color="white", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(targets)))
ax.set_yticklabels([t for t, _ in targets])
ax.invert_yaxis()
ax.set_xticks([])
ax.set_xlim(0, 1)
ax.spines["bottom"].set_visible(False)
ax.set_title("15 verdicts: 6 proven, 1 partly, 5 falsified / not met, 3 open", loc="left", fontsize=11, color=INK)
save(fig, "target_status.png")

# 2. Composite A paired deltas. Left: vs H2O (scorecard e2fcd82b, J1 a8ca3e7e, 6zs79 f93eccfe target gap).
#    Right: H2O-host training arms vs the in-job H2O control (x7c42 bf706854, q668q a100c3b9).
vs_h2o = [  # label, d, lo, hi, ours
    ("torchcast-12b (external)", -3.6, -6.6, -0.8, False),
    ("stock Qwen3.5-4B + H2O shim", -9.4, -23.1, -4.2, True),
    ("JEV1-4B r2 (trained)", -13.21, -37.82, -7.59, True),
    ("D19 RWKV7-13.3B (trained)", -14.4, -31.7, -10.6, True),
    ("JEV3-SYM seed 0 (trained, sym-fork)", -46.80, -59.36, -30.87, True),
]
host = [
    ("JEV2-H2O x7c42 (2,664 rows)", -0.41, -1.75, 1.39),
    ("JEV2-H2O-L q668q (17,959 rows)", -0.97, -3.38, 1.12),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
for i, (lab, d, lo, hi, ours) in enumerate(vs_h2o):
    c = OURS if ours else EXT
    a1.plot([lo, hi], [i, i], color=c, lw=2)
    a1.plot(d, i, "o", color=c, ms=6)
a1.axvline(0, color=BASE, lw=1.2)
a1.set_yticks(range(len(vs_h2o)))
a1.set_yticklabels([a[0] for a in vs_h2o])
a1.invert_yaxis()
a1.set_xlabel("Δ Composite A vs H2O-Lightning-4B v1.1 (95% CI)")
a1.set_title("Our stacks vs H2O: all CIs < 0", loc="left", fontsize=11, color=INK)
a1.grid(axis="x", color=GRID, lw=0.6)
for i, (lab, d, lo, hi) in enumerate(host):
    a2.plot([lo, hi], [i, i], color=HOST, lw=2)
    a2.plot(d, i, "o", color=HOST, ms=6)
a2.axvline(0, color=BASE, lw=1.2)
a2.set_yticks(range(len(host)))
a2.set_yticklabels([h[0] for h in host])
a2.set_ylim(1.6, -0.6)
a2.set_xlabel("Δ Composite A vs in-job H2O control (95% CI)")
a2.set_title("H2O + our data: flat (Holm: none)", loc="left", fontsize=11, color=INK)
a2.grid(axis="x", color=GRID, lw=0.6)
save(fig, "composite_delta_v2.png")

# 3. Shapley axis attribution of the Composite A gap vs H2O (lead-gap-attribution, review shap_rank.out f2a04059)
cands = [("stock 4B + shim\n(gap 9.38)", 6.283, 3.142, -0.044, 0.0),
         ("JEV1-4B r2\n(gap 13.21)", 8.058, 4.231, 0.953, -0.033),
         ("D19\n(gap 14.39)", 7.256, 1.827, 2.122, 3.182)]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
for i, (lab, I, C, S, K) in enumerate(cands):
    left = 0.0
    for v, c, n in ((I, I_C, "I"), (C, C_C, "C"), (S, S_C, "S"), (K, K_C, "K")):
        if v > 0:
            ax.barh(i, v, left=left, color=c, height=0.55, label=n if i == 2 else None)
            if v > 1.0:
                ax.text(left + v / 2, i, f"{n} {v:.1f}", ha="center", va="center", color="white", fontsize=9)
            left += v
ax.set_yticks(range(len(cands)))
ax.set_yticklabels([c[0] for c in cands])
ax.invert_yaxis()
ax.set_xlabel("Composite-A points of the gap to H2O (exact Shapley over I, C, S, K)")
ax.set_title("The gap is mostly Intelligence; D19 also pays for Speed and Cost", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, ncol=4, loc="lower right", fontsize=9)
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "gap_shapley.png")

# 4. Slice gain vs public result (x7c42 bf706854, q668q a100c3b9, 6zs79 f93eccfe)
runs = [("JEV2-H2O\nx7c42", 4.44, -0.41, "+0.15"), ("JEV2-H2O-L\nq668q", 8.97, -0.97, "-0.32"),
        ("JEV3-SYM s0\n6zs79", 1.08, -36.66, "-0.50 (top-1)")]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
x = range(len(runs))
ax.bar([i - 0.18 for i in x], [r[1] for r in runs], width=0.36, color=HOST, label="held-out slice top-1 gain, pp")
ax.bar([i + 0.18 for i in x], [r[2] for r in runs], width=0.36, color="#cf222e", label="public-231 Δ Composite A")
for i, r in enumerate(runs):
    ax.text(i, max(r[1], 0) + 1.5, f"transfer corr {r[3]}", ha="center", fontsize=8.5, color=MUTED)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([r[0] for r in runs])
ax.set_ylim(-40, 16)
ax.set_title("Gains on our own held-out data do not reach the public items", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(axis="y", color=GRID, lw=0.6)
save(fig, "slice_vs_public.png")

# 5. Flip rate (ev-flip-rate 4a65dd18 run2; JEV3-SYM 6zs79 f93eccfe)
sys_ = [
    ("MD1 (reported; own items/ops)", 1.3, None, None, EXT),
    ("H2O-Lightning-4B v1.1 (8-op)", 2.44, 1.26, 3.75, EXT),
    ("D19 (3 reorder ops only)", 2.88, 1.36, 4.44, OURS),
    ("JEV3-SYM seed 0 (8-op; reorder/relabel 0 bitwise)", 4.14, 2.94, 5.35, OURS),
    ("RWKV7-13.3B stock v3w (8-op)", 5.37, 3.95, 6.99, OURS),
    ("Qwen3.5-4B + H2O shim (8-op)", 7.55, 5.71, 9.67, OURS),
]
fig, ax = plt.subplots(figsize=(7.6, 3.5))
for i, (lab, v, lo, hi, c) in enumerate(sys_):
    ax.barh(i, v, color=c, height=0.55, alpha=0.9 if lo is not None else 0.45)
    if lo is not None:
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
    ax.text((hi if lo is not None else v) + 0.15, i, f"{v:.2f}%", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(sys_)))
ax.set_yticklabels([s[0] for s in sys_])
ax.invert_yaxis()
ax.set_xlabel("Average flip rate, % (95% CI; MD1 = context only, different estimand)")
ax.set_title("sym-fork removes reorder/relabel flips; paraphrase/noise flips remain", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "flip_rate_v2.png")

# 6. Clef per-area accuracy gap (ev-guar-area ca32b61f AREA_ACCURACY.json) + headline Index
areas = [("knowledge", -9.1, -10.6, -7.8), ("arts", -3.9, -6.4, -1.5), ("tools", -2.4, -3.4, -1.5),
         ("retrieval", -0.9, -3.5, 1.8), ("language", 0.9, 0.0, 1.7), ("overall", -1.4, -2.2, -0.6)]
fig, ax = plt.subplots(figsize=(7.6, 3.2))
for i, (a, d, lo, hi) in enumerate(areas):
    c = "#cf222e" if hi < 0 else EXT
    ax.plot([lo, hi], [i, i], color=c, lw=2)
    ax.plot(d, i, "o", color=c, ms=6)
ax.axvline(0, color=BASE, lw=1.2)
ax.set_yticks(range(len(areas)))
ax.set_yticklabels([a[0] for a in areas])
ax.invert_yaxis()
ax.set_xlabel("D19 − Clef-flash decision accuracy, pp (95% CI)  |  Index: D19 49.49 vs 57.29 / 61.89")
ax.set_title("D19 loses the Index mainly in knowledge, arts and tools", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "clef_area_gap.png")

# 7. Guarantee act-fraction progression (unchanged from v1; receipts ca32b61f / b4835e05 / 01738528)
stages = ["per-decision", "per-area", "one-per-row", "CRR c=8", "FS-CRR\nunseen (OPEN)"]
models = {"Clef 27B": ([81.1, 61.6, 59.3, 0.0], "#8250df"), "Clef-flash": ([72.2, 67.3, 51.6, 0.0], "#bf8700"),
          "D19": ([75.1, 60.1, 0.0, 0.0], OURS), "D13": ([36.1, 40.3, 0.0, 0.0], "#54aeff")}
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for name, (vals, c) in models.items():
    ax.plot(range(4), vals, "-o", color=c, lw=2, ms=5, label=name)
ax.axvspan(3.5, 4.5, color=GRID, alpha=0.5)
ax.text(4, 45, "not run:\n~$35", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(range(5))
ax.set_xticklabels(stages)
ax.set_ylabel("Certified TEST act fraction, %")
ax.set_ylim(-3, 90)
ax.set_title("Cluster-valid tests certify nothing on DEV; the unseen-split test is unfunded", loc="left",
             fontsize=11, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
save(fig, "guarantee_progression_v2.png")

# 8. Budget: open items vs balance (ledger c783be49: balance 19.10 after job-6zs79)
items = [("FS-CRR predictions (D19 + 2 Clefs)", 35.0), ("+ D13 arm (optional)", 22.0),
         ("JEV3-SYM seed 1 (worst-case hold)", 21.67), ("JEV3-SYM-NU / G16 cell (worst-case hold, est.)", 21.67),
         ("JEV2-4B full cell (worst-case hold)", 79.88)]
fig, ax = plt.subplots(figsize=(7.6, 3.2))
for i, (lab, v) in enumerate(items):
    ax.barh(i, v, color=EXT, height=0.55)
    ax.text(v + 1, i, f"${v:.2f}", va="center", fontsize=9, color=INK)
ax.axvline(19.10, color=BASE, lw=1.5)
ax.text(19.6, len(items) - 0.4, "balance $19.10", color=BASE, fontsize=9)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([i[0] for i in items])
ax.invert_yaxis()
ax.set_xlabel("USD")
ax.set_title("Every open item costs more than the remaining balance", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "open_item_costs.png")
print("ok")
