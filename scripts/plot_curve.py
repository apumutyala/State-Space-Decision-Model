"""Plot the 'cost of attention-freedom' curve from a pilot results.json.

python scripts/plot_curve.py receipts/<dir>
Writes <dir>/curve.png, <dir>/curve.csv, <dir>/stage_losses.png
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"


def metrics(p):
    return {"ppl@1k": p["ppl"]["ctx1024"]["ppl"], "ppl@4k": p["ppl"]["ctx4096"]["ppl"],
            "NIAH (mean over lengths)": p["niah"]["mean_acc"], "MQAR (mean over N)": p["mqar"]["mean_acc"],
            "MMLU 0-shot (5/subject)": p["mmlu"]["acc"],
            **{f"NIAH L{k[1:]}": v["acc"] for k, v in p["niah"].items() if k.startswith("L")},
            **{f"MQAR {k}": v["acc"] for k, v in p["mqar"].items() if k.startswith("N")}}


def main(d):
    R = json.load(open(os.path.join(d, "results.json")))
    n_attn = 6
    pts = [(n_attn, metrics(R["probe_stock"]), "stock")]
    for st in R.get("progressive", {}).get("steps", []):
        tag = st.get("probe_tag")
        if tag and f"probe_{tag}" in R:
            pts.append((n_attn - len(st["converted_after"]), metrics(R[f"probe_{tag}"]), tag))
    swa = [(k, metrics(v)) for k, v in R.items() if k.startswith("probe_swa")]
    zero = metrics(R["probe_attn_deleted"]) if "probe_attn_deleted" in R else None

    keys = ["ppl@1k", "ppl@4k", "NIAH (mean over lengths)", "MQAR (mean over N)", "MMLU 0-shot (5/subject)"]
    with open(os.path.join(d, "curve.csv"), "w", newline="") as f:
        allk = list(pts[0][1].keys())
        w = csv.writer(f)
        w.writerow(["config", "attention_layers_kept"] + allk)
        for k, m, tag in pts:
            w.writerow([tag, k] + [m.get(x) for x in allk])
        for tag, m in swa:
            w.writerow([tag[6:], "6 (windowed)"] + [m.get(x) for x in allk])
        if zero:
            w.writerow(["attn_deleted_no_training", 0] + [zero.get(x) for x in allk])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), facecolor=SURF)
    xs = [p[0] for p in pts]
    for ax, key in zip(axes, keys):
        ax.set_facecolor(SURF)
        ys = [p[1][key] for p in pts]
        ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=6, label="converted (GDN, KL-distilled)", zorder=3)
        ax.axhline(pts[0][1][key], color=MUTED, lw=1, ls="--", label="stock hybrid", zorder=1)
        for tag, m in swa:
            ax.scatter([6], [m[key]], marker="s", s=60, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=4,
                       label=f"SWA+sinks, no training ({tag[10:]})")
        if zero:
            ax.scatter([0], [zero[key]], marker="X", s=70, color=AQUA, edgecolor=SURF, linewidth=1.5, zorder=4,
                       label="attention deleted, no training")
        ax.set_title(key, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("attention layers kept (of 6)")
        ax.invert_xaxis()
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        if key.startswith("ppl"):
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.annotate(f"{ys[-1]:.3g}" if key.startswith("ppl") else f"{ys[-1]:.2f}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(4, 6), color=INK, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Qwen3.5-0.8B attention->GDN conversion: cost of attention-freedom (pipeline qualification, "
                 "~10M distillation tokens; not a 9B claim)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(d, "curve.png"), dpi=140, facecolor=SURF)

    # stage losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=SURF)
    s1 = R.get("stage1", {}).get("hist", [])
    if s1:
        layers = [k for k in s1[0] if k.startswith("rel_mse")]
        pal = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300"]
        for c, k in zip(pal, layers):
            axes[0].plot([r["tokens"] / 1e6 for r in s1], [r[k] for r in s1], color=c, lw=2, label=k[8:])
        axes[0].set_yscale("log")
        axes[0].set_title("Stage 1: per-layer relative MSE to attention output", loc="left", fontsize=10)
        axes[0].set_xlabel("tokens (M)")
        axes[0].legend(frameon=False, ncol=3, fontsize=8)
    off = 0
    for st in R.get("progressive", {}).get("steps", []):
        hx = [off + r["tokens"] / 1e6 for r in st["hist"]]
        axes[1].plot(hx, [r["kl"] for r in st["hist"]], color=BLUE, lw=2)
        axes[1].annotate(f"+L{st['layer']}", (hx[0], st["hist"][0]["kl"]), fontsize=8, color=INK)
        off = hx[-1]
    axes[1].set_title("Stage 2: end-to-end KL(teacher||student), progressive swaps", loc="left", fontsize=10)
    axes[1].set_xlabel("cumulative stage-2 tokens (M)")
    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "stage_losses.png"), dpi=140, facecolor=SURF)
    print(open(os.path.join(d, "curve.csv")).read())


if __name__ == "__main__":
    main(sys.argv[1])
