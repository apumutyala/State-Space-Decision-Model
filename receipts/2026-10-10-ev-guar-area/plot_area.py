"""Figure for GUARANTEES_AREA.md from GUARANTEES_AREA.json (primary split, alpha 0.10).

    python -I plot_area.py GUARANTEES_AREA.json out.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

o = json.load(open(sys.argv[1], encoding="utf-8"))["run_id"]["models"]
models = ["Clef27B", "D19", "ClefFlash", "D13"]
areas = ["arts", "knowledge", "language", "retrieval", "tools"]
c_m, c_a = "#9aa5b1", "#2f6fb0"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
x = range(len(models))
mar = [o[m]["alpha"]["0.1"]["marginal_reference"]["test_act"] * 100 for m in models]
are = [o[m]["alpha"]["0.1"]["overall"]["act"] * 100 for m in models]
ax[0].bar([i - 0.2 for i in x], mar, 0.4, color=c_m, label="marginal (one threshold for all areas)")
ax[0].bar([i + 0.2 for i in x], are, 0.4, color=c_a, label="per-area (each area certified separately)")
for i in x:
    ax[0].text(i - 0.2, mar[i] + 1, f"{mar[i]:.1f}", ha="center", fontsize=8)
    ax[0].text(i + 0.2, are[i] + 1, f"{are[i]:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(list(x), ["Clef 27B", "D19", "Clef-flash", "D13"])
ax[0].set_ylabel("% of TEST decisions acted on (error <= 5%, delta 0.1)")
ax[0].set_ylim(0, 100)
ax[0].legend(fontsize=8, frameon=False, loc="upper right")
ax[0].set_title("Actionable fraction, alpha 0.10", fontsize=10)
for s in ("top", "right"):
    ax[0].spines[s].set_visible(False)
w = 0.2
for j, m in enumerate(models):
    pa = o[m]["alpha"]["0.1"]["per_area"]
    xs = [i + (j - 1.5) * w for i in range(len(areas))]
    ax[1].scatter(xs, [pa[a]["cov_m"] for a in areas], marker="o", facecolors="none", edgecolors=c_m, s=22)
    ax[1].scatter(xs, [pa[a]["cov"] for a in areas], marker="o", color=c_a, s=22)
ax[1].axhline(0.90, color="black", lw=0.8, ls="--")
ax[1].set_xticks(range(len(areas)), areas)
ax[1].set_ylabel("TEST coverage")
ax[1].set_title("Per-area coverage (hollow = marginal, filled = per-area); 4 models per area", fontsize=10)
for s in ("top", "right"):
    ax[1].spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
