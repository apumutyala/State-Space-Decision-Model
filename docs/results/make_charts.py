"""Render the README result charts from the 2026-10-15 benchmark card.

Every value below is copied from docs/benchmark_card/ssdm-benchmark-card.html
(branch claude/benchmark-card), which cites the receipts for each number.
Run: python docs/results/make_charts.py  ->  docs/results/*.svg (light and dark).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

THEMES = {
    "light": dict(ours="#2a78d6", them="#85847e", ink="#0b0b0b", muted="#52514e", grid="#e4e3df", whisker="#3a3936"),
    "dark":  dict(ours="#3987e5", them="#6f6e69", ink="#f0efec", muted="#c3c2b7", grid="#34332f", whisker="#d8d7d0"),
}

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
})

# name, sublabel, value, lo, hi, ours
CHARTS = {
 "decision_index": dict(
    title="Decision Index 0.2.1, full dev set (12,104 rows) · higher is better",
    note="Same harness and scorer for every model. Whiskers: 95% CI. One seed per trained model.",
    fmt="{:.1f}", xmax=70, ticks=[0, 10, 20, 30, 40, 50, 60, 70],
    rows=[("Clef 27B", None, 61.89, 60.28, 63.28, False),
          ("Clef-flash", None, 57.29, 55.88, 58.67, False),
          ("Order-Locked RWKV 13B", "proof: reorder-invariant", 49.49, 47.77, 50.80, True),
          ("Injection-Hardened RWKV 13B", "measured: unseen-attack robust", 43.21, None, None, True)]),
 "jevbench": dict(
    title="JevBench Composite, 231 public items · higher is better",
    note="Local jevkit run (reproduces published scorecards within 1 item). *torchcast failed the reproduction check by 5 items.",
    fmt="{:.1f}", xmax=90, ticks=[0, 15, 30, 45, 60, 75, 90],
    rows=[("H2O-Lightning-4B", None, 76.6, 72.8, 78.3, False),
          ("torchcast-12b*", None, 73.1, 69.4, 75.1, False),
          ("decider-12b-v1", None, 72.4, 68.6, 74.4, False),
          ("decider-12b v2", None, 71.0, 66.9, 73.7, False),
          ("Symmetry-Locked 4B", "proof: reorder + relabel invariant", 66.97, None, None, True),
          ("Order-Locked RWKV 13B", "proof: reorder-invariant", 62.3, 44.1, 64.9, True)]),
 "answer_flips": dict(
    title="Answer flips under 8 harmless edits · lower is better",
    note="Reorder, shuffle and relabel flips are exactly 0 for Symmetry-Locked 4B by construction; the rest come from paraphrase-type edits.\nMD1 is self-reported on its own items and not comparable.",
    fmt="{:.2f}%", xmax=10, ticks=[0, 2, 4, 6, 8, 10], tickfmt="{:g}%",
    rows=[("MD1 (self-reported)", None, 1.3, None, None, False),
          ("H2O-Lightning-4B", None, 2.44, 1.26, 3.75, False),
          ("Symmetry-Locked 4B", "proof: 0 reorder/relabel flips", 3.64, None, None, True),
          ("Stock Qwen3.5-4B", None, 7.55, 5.71, 9.67, False)]),
 "keel_steering": dict(
    title="Keel: targeted steering on unseen attack templates · lower is better",
    note="Static attacks. Searched (adaptive) attacks still steer 90–100% of every 4B judge, so Keel enforces authority in host-side structure.",
    fmt="{:.1f}%", xmax=30, ticks=[0, 5, 10, 15, 20, 25, 30], tickfmt="{:g}%",
    rows=[("Stock 4B, one pass", None, 25.0, None, None, False),
          ("Stock 4B, thinking", None, 13.8, None, None, False),
          ("Trust-Distilled 4B", "one token · host-enforced authority", 9.8, None, None, True)]),
 "serving_cost": dict(
    title="Serving cost per 1,000 Index decisions (projection) · lower is better",
    note="Ours is a max-batch H100 projection from token counts with exact prefix sharing, not a measured invoice. Clef figures use its API list prices.",
    fmt="${:.3f}", xmax=0.25, ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], tickfmt="${:.2f}",
    rows=[("Clef-flash (9B)", None, 0.061, None, None, False),
          ("Clef 27B", None, 0.167, None, None, False),
          ("Order-Locked RWKV 13B", "projected, with prefix sharing", 0.183, None, None, True)]),
}

ATTACKS = dict(
    title="Unseen injection attacks: 21 templates, 7,920 attacked decisions per model · lower is better",
    note="Hijack: planted text overrides the answer. Steer: planted text pushes toward a chosen wrong option. Whiskers: 95% CI.\nOurs beats Clef-flash significantly after multiple-comparison correction; against Clef 27B it is a tie.",
    rows=[("Injection-Hardened RWKV 13B", True, (2.5, 0, 8.4), (0.3, 0, 1.4)),
          ("Clef 27B", False, (1.3, 0, 4.0), (1.8, 0, 5.1)),
          ("Clef-flash", False, (11.5, 1.9, 27.2), (8.9, 1.0, 20.9))])


def style_axis(ax, t, xmax, ticks, tickfmt):
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xticks(ticks)
    ax.set_xticklabels([(tickfmt or "{:g}").format(v) for v in ticks], color=t["muted"], fontsize=9)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0)


def bar_row(ax, y, v, lo, hi, ours, t, fmt, xmax, h=0.56):
    col = t["ours"] if ours else t["them"]
    ax.barh(y, v, height=h, color=col, linewidth=0)
    end = v
    if lo is not None:
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=t["whisker"], elinewidth=1.1, capsize=3, alpha=0.7)
        end = hi
    ax.text(end + xmax * 0.015, y, fmt.format(v), va="center", ha="left", color=t["ink"], fontsize=9.5)


def labels(ax, rows_meta, t):
    ax.set_yticks([])
    for y, name, sub, ours in rows_meta:
        if sub:
            ax.text(-0.02, y + 0.13, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color=t["ink"], fontsize=10, fontweight="bold" if ours else "normal")
            ax.text(-0.02, y - 0.2, sub, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color=t["muted"], fontsize=8.5)
        else:
            ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color=t["ink"], fontsize=10, fontweight="bold" if ours else "normal")


def header(fig, title, note, t, top_pad):
    fig.text(0.01, 1 - 0.1 / top_pad, title, ha="left", va="top", color=t["ink"], fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.012, note, ha="left", va="bottom", color=t["muted"], fontsize=8.5, linespacing=1.4)


def simple(name, spec, theme):
    t = THEMES[theme]
    rows = spec["rows"]
    n = len(rows)
    note_lines = spec["note"].count("\n") + 1
    hgt = 0.95 + 0.52 * n + 0.2 * note_lines
    fig = plt.figure(figsize=(8.2, hgt))
    bottom = (0.42 + 0.2 * note_lines) / hgt
    ax = fig.add_axes([0.33, bottom, 0.64, 1 - bottom - 0.5 / hgt])
    meta = []
    for i, (lab, sub, v, lo, hi, ours) in enumerate(rows):
        y = n - 1 - i
        bar_row(ax, y, v, lo, hi, ours, t, spec["fmt"], spec["xmax"])
        meta.append((y, lab, sub, ours))
    ax.set_ylim(-0.6, n - 0.4)
    style_axis(ax, t, spec["xmax"], spec["ticks"], spec.get("tickfmt"))
    labels(ax, meta, t)
    header(fig, spec["title"], spec["note"], t, hgt)
    fig.savefig(OUT / f"{name}-{theme}.svg", transparent=True)
    plt.close(fig)


def attacks(theme):
    t = THEMES[theme]
    rows = ATTACKS["rows"]
    n = len(rows)
    hgt = 0.95 + 0.52 * n + 0.4
    fig = plt.figure(figsize=(8.2, hgt))
    bottom = 0.82 / hgt
    top = 1 - 0.72 / hgt
    axes = [fig.add_axes([0.33, bottom, 0.3, top - bottom]), fig.add_axes([0.67, bottom, 0.3, top - bottom])]
    for k, (ax, panel) in enumerate(zip(axes, ["Hijack rate", "Steer rate"])):
        for i, (lab, ours, hij, ste) in enumerate(rows):
            y = n - 1 - i
            v, lo, hi = hij if k == 0 else ste
            bar_row(ax, y, v, lo, hi, ours, t, "{:.1f}%", 30)
        ax.set_ylim(-0.6, n - 0.4)
        style_axis(ax, t, 30, [0, 10, 20, 30], "{:g}%")
        ax.set_yticks([])
        ax.set_title(panel, loc="left", color=t["ink"], fontsize=10, pad=4)
    labels(axes[0], [(n - 1 - i, lab, None, ours) for i, (lab, ours, _, _) in enumerate(rows)], t)
    header(fig, ATTACKS["title"], ATTACKS["note"], t, hgt)
    fig.savefig(OUT / f"injection_attacks-{theme}.svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    for theme in THEMES:
        for name, spec in CHARTS.items():
            simple(name, spec, theme)
        attacks(theme)
    print(sorted(p.name for p in OUT.glob("*.svg")))
