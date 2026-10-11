"""Evidence-report charts. Numbers are transcribed from the cited receipts (see docs/EVIDENCE_REPORT.md)."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
OURS, EXT, BASE = "#0969da", "#8c959f", "#bf3989"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})

# 1. Composite A paired delta vs H2O-Lightning-4B v1.1 (public-231, B1000 seed16)
# claude/ev-scorecard @e2fcd82b SCORECARD.md; JEV1-4B r2 from claude/ev-j1-rescore @a8ca3e7e RESCORE_XT3AX.md
arms = [  # label, delta, lo, hi, ours?
    ("torchcast-decision-12b (ext)", -3.6, -6.6, -0.8, False),
    ("decider-12b-v1 (ext)", -4.3, -7.8, -1.1, False),
    ("decider-12b v2 (ext)", -5.6, -9.3, -1.9, False),
    ("stock Qwen3.5-4B + H2O shim", -9.4, -23.1, -4.2, True),
    ("gemma-4-E4B + v3w", -11.1, -33.5, -4.4, True),
    ("JEV1-4B r2 (trained)", -13.21, -37.82, -7.59, True),
    ("D19 RWKV7-13.3B + joint (trained)", -14.4, -31.7, -10.6, True),
    ("sizecliff-4B native (trained)", -14.6, -34.2, -10.1, True),
    ("Qwen3.5-4B + v3w", -23.0, -50.0, -8.5, True),
    ("RWKV7-13.3B stock + v3w", -31.0, -42.5, -14.9, True),
]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for i, (lab, d, lo, hi, ours) in enumerate(arms):
    c = OURS if ours else EXT
    ax.plot([lo, hi], [i, i], color=c, lw=2)
    ax.plot(d, i, "o", color=c, ms=6)
ax.axvline(0, color=BASE, lw=1.2)
ax.text(0.5, -0.9, "H2O = 0", color=BASE, fontsize=9)
ax.set_yticks(range(len(arms)))
ax.set_yticklabels([a[0] for a in arms])
ax.invert_yaxis()
ax.set_xlabel("Paired Δ Composite A vs H2O-Lightning-4B v1.1 (95% CI)")
ax.set_title("No arm beats H2O on public-231: every CI is below 0", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(out, "composite_delta_vs_h2o.png"), dpi=150)
plt.close(fig)

# 2. Flip rate per system (claude/ev-flip-rate @4a65dd18 FLIP_RATE.md, run 2)
sys_ = [  # label, value, lo, hi, color
    ("MD1 (reported, own items/ops)", 1.3, None, None, EXT),
    ("H2O-Lightning-4B v1.1 (8-op)", 2.44, 1.26, 3.75, EXT),
    ("D19 (3 reorder-family ops only)", 2.88, 1.36, 4.44, OURS),
    ("RWKV7-13.3B stock v3w (8-op)", 5.37, 3.95, 6.99, OURS),
    ("Qwen3.5-4B + H2O shim (8-op)", 7.55, 5.71, 9.67, OURS),
]
fig, ax = plt.subplots(figsize=(7.6, 3.2))
for i, (lab, v, lo, hi, c) in enumerate(sys_):
    ax.barh(i, v, color=c, height=0.55, alpha=0.9 if lo is not None else 0.45)
    if lo is not None:
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
    ax.text(v + 0.15 if lo is None else hi + 0.15, i, f"{v:.2f}%", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(sys_)))
ax.set_yticklabels([s[0] for s in sys_])
ax.invert_yaxis()
ax.set_xlabel("Average flip rate, % (95% CI; MD1 = context only, different estimand)")
ax.set_title("Flip rate: H2O is the most robust of the systems we ran", loc="left", fontsize=11, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(out, "flip_rate.png"), dpi=150)
plt.close(fig)

# 3. Guarantee act-fraction progression, alpha 0.10, primary (run_id) config, marginal
# FULLDEV (claude/ev-guar-area @ca32b61f receipts/2026-10-10-B-guarantees-fulldev), AREA (same branch),
# CLUSTER + CRR (claude/ev-guarantee-proofs @01738528)
stages = ["per-decision\n(marginal)", "per-area\n(Mondrian)", "one-per-row\n(cluster, marginal)", "CRR c=8\n(marginal)"]
models = {
    "Clef 27B": ([81.1, 61.6, 59.3, 0.0], "#8250df"),
    "Clef-flash": ([72.2, 67.3, 51.6, 0.0], "#bf8700"),
    "D19": ([75.1, 60.1, 0.0, 0.0], OURS),
    "D13": ([36.1, 40.3, 0.0, 0.0], "#54aeff"),
}
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for name, (vals, c) in models.items():
    ax.plot(range(len(stages)), vals, "-o", color=c, lw=2, ms=5, label=name)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages)
ax.set_ylabel("Certified TEST act fraction, % (α 0.10, ε 0.05, δ 0.10)")
ax.set_ylim(-3, 90)
ax.set_title("As the test gets more honest, no model keeps a certificate on DEV", loc="left", fontsize=11, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, "guarantee_act_fraction.png"), dpi=150)
plt.close(fig)
print("ok")
