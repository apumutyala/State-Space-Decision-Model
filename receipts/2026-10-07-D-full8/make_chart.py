"""Render curve.png from result.json (CPU, matplotlib Agg). python receipts/2026-10-07-D-full8/make_chart.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "result.json").read_text(encoding="utf-8"))["result"]["decision"]
curve, steps = d["report"]["curve"], d["report"]["steps"]
gate = d["rule"]["gate"]["min_hidden_cos"]

BLUE, INK, MUTED, SURF, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# left: MMLU (285-item probe subset, the per-step curve) vs layers converted
n = [8 - c["attention_layers_kept"] for c in curve]
mmlu = [100 * c["mmlu"] for c in curve]
a.plot(n, mmlu, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
for x, y, c in zip(n, mmlu, curve):
    if c["added_layer"] is not None:
        a.annotate(f"+L{c['added_layer']}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                   color=MUTED, fontsize=8)
a.annotate(f"{mmlu[0]:.1f}", (n[0], mmlu[0]), textcoords="offset points", xytext=(-14, -4), ha="right", color=INK)
a.annotate(f"{mmlu[-1]:.1f}", (n[-1], mmlu[-1]), textcoords="offset points", xytext=(10, -4), ha="left", color=INK)
a.set_xticks(n)
a.set_xlim(-0.5, 6.8)
a.set_ylim(64, 83.5)
a.set_xlabel("attention layers converted to GDN (stock = 0)")
a.set_ylabel("MMLU accuracy, % (probe subset)")
a.set_title("MMLU falls as layers convert (full MMLU: 78.5 -> 66.3)", loc="left", color=INK, fontsize=11)

# right: per-step gate hidden_cos, in conversion order, with the 0.95 floor
xs = list(range(1, len(steps) + 1))
hc = [s["hidden_cos"] for s in steps]
b.plot(xs, hc, color=BLUE, lw=2, zorder=2)
for x, s in zip(xs, steps):
    ok = s["accepted"]
    b.scatter([x], [s["hidden_cos"]], s=64, zorder=3, color=BLUE if ok else SURF, edgecolor=BLUE, linewidth=2)
    b.annotate(f"L{s['layer']}" + ("" if ok else "\nrejected"), (x, s["hidden_cos"]), textcoords="offset points",
               xytext=(0, 9) if ok else (0, -26), ha="center", color=INK if not ok else MUTED, fontsize=8)
b.axhline(gate, color=MUTED, lw=1, ls="--")
b.text(0.6, gate + 0.0015, f"gate floor {gate}", color=MUTED, fontsize=8)
b.set_xticks(xs)
b.set_xlim(0.4, 8.6)
b.set_ylim(0.91, 1.0)
b.set_xlabel("conversion step (KL order); filled = accepted, hollow = rolled back")
b.set_ylabel("hidden-state cosine vs teacher")
b.set_title("Each step costs fidelity; L3 and L27 miss the floor", loc="left", color=INK, fontsize=11)

fig.suptitle("job-2xe5w full-8 Q9-AF conversion -> partial (Q9-H2: 6 of 8 converted)", x=0.01, ha="left",
             color=INK, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "curve.png", dpi=150, facecolor=SURF)
print("wrote", HERE / "curve.png")
