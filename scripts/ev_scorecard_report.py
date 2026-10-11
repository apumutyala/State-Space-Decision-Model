"""Render SCORECARD.md + scorecard.png from SCORECARD.json (ev_scorecard.py output). CPU, no network.

  python -I -X utf8 scripts/ev_scorecard_report.py --card SCORECARD.json --md SCORECARD.md --png scorecard.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def ci(c, d=1):
    return "-" if not c else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--png", required=True)
    a = ap.parse_args(argv)
    card = json.loads(Path(a.card).read_text(encoding="utf-8"))
    arms, pairs, base = card["arms"], card["pairs_vs_baseline"], card["baseline"]
    order = sorted(arms, key=lambda n: -arms[n]["composite"]["A"])

    rows = []
    hdr = ("| # | arm (job) | I | C | S | K | Capability [CI] | Composite A [CI] | ΔComp A vs H2O [CI] | "
           "ΔCap vs H2O [CI] | ΔComp A, S=H2O's [CI] | price class ($/M in) | tok/dec |")
    rows += [hdr, "|" + "---|" * 13]
    for r, n in enumerate(order, 1):
        x = arms[n]
        c95 = x.get("ci95") or {}
        p = pairs.get(n)
        def dcell(k):
            if p is None:
                return "baseline"
            d = p[k]
            mark = " **CI>0**" if d["ci_gt_0"] else (" (CI<0)" if d["ci_lt_0"] else "")
            return f"{d['point']:+.1f} {ci(d['ci95'])}{mark}"
        rows.append(f"| {r} | {n} ({x['job']}) | {f(x['I'])} | {f(x['C'])} | {f(x['S'])} | {f(x['K'])} | "
                    f"{f(x['capability'])} {ci(c95.get('capability'))} | {f(x['composite']['A'])} "
                    f"{ci((c95.get('composite') or {}).get('A'))} | {dcell('composite_A')} | {dcell('capability')} | "
                    f"{dcell('composite_A_commonS')} | {x['price_class']} ({x['price']['usd_per_m_input']}, "
                    f"{x['price']['kind']}) | {f(x['tokens_per_decision']['mean'], 0)} |")

    best_c = max((n for n in pairs), key=lambda n: pairs[n]["composite_A"]["point"])
    best_k = max((n for n in pairs), key=lambda n: pairs[n]["capability"]["point"])
    any_c = [n for n in pairs if pairs[n]["composite_A"]["ci_gt_0"]]
    any_k = [n for n in pairs if pairs[n]["capability"]["ci_gt_0"]]
    ext = ("h2o", "decider", "torchcast", "d1-3b")
    ours = [n for n in pairs if not n.startswith(ext)]
    trained = [n for n in ours if arms[n]["job"] in ("job-9j3uu", "job-27anq", "job-haxmm", "job-reknq")
               and "(9j3uu)" not in n]

    def best_line(group, what):
        bc = max(group, key=lambda n: pairs[n]["composite_A"]["point"])
        bk = max(group, key=lambda n: pairs[n]["capability"]["point"])
        return (f"Best of {what}: Δ Composite A **{bc}** {pairs[bc]['composite_A']['point']:+.1f} "
                f"{ci(pairs[bc]['composite_A']['ci95'])}; Δ Capability **{bk}** "
                f"{pairs[bk]['capability']['point']:+.1f} {ci(pairs[bk]['capability']['ci95'])}.")
    lat = sorted({(arms[n]["job"], arms[n]["lane"], arms[n]["latency"]["n"]) for n in arms})
    md = [
        "# Local JevBench scorecard (EV-PREREG-COMPOSITE-LOCAL)",
        "",
        f"**Caveat (every number): {card['caveat']}.** Local Composite is never compared with the board's 72.5.",
        "",
        f"**Verdict:** {'no arm beats' if not any_c else f'{len(any_c)} arm(s) beat'} H2O-Lightning-4B v1.1 on "
        f"Composite A under the rule. CI>0 on Composite A: "
        f"{', '.join(any_c) or 'none'}; on Capability: {', '.join(any_k) or 'none'}. Best paired Δ Composite A: "
        f"**{best_c}** {pairs[best_c]['composite_A']['point']:+.1f} {ci(pairs[best_c]['composite_A']['ci95'])}; best "
        f"paired Δ Capability: **{best_k}** {pairs[best_k]['capability']['point']:+.1f} "
        f"{ci(pairs[best_k]['capability']['ci95'])}.",
        "",
        "- " + best_line(ours, "OUR candidates (our trained cells + our serving configs on stock hosts; excludes the "
                         "external systems H2O, decider, torchcast, d1)"),
        "- " + best_line(trained, "OUR TRAINED cells (D19, sizecliff 4B/2B/0.8B arms)"),
        "",
        f"Baseline: `{base}` (job-zbtbj, our run, pinned 193ad740; kit parity 205/231 = card). Protocol: "
        f"{card['protocol']}. {len(arms)} arms, {card['n_items']} public items, same ids in every arm. "
        f"Flags: {len(card['flags'])}; max |diff| vs each in-job report: {card['max_abs_diff']}.",
        "",
        *rows,
        "",
        "Δ columns: paired candidate − H2O on the same 1,000 stratified resamples; **CI>0** = the 95% CI excludes 0 "
        "in our favour (the rule's success condition); (CI<0) = H2O significantly better. Capability = f(I, C) only, "
        "so it is independent of latency and price.",
        "",
        "## Caveats per row",
        "- **Speed comparability:** every arm's S comes from `pred latency_s` on the 72 open standard+judge items, "
        "in-process wall time on 1x NVIDIA H100 80GB HBM3 with the self-hosted x2 + 0.15 s adjustment, but measured in "
        "DIFFERENT jobs (zbtbj, 9j3uu, 27anq, haxmm, reknq, r3uc4: different processes, readouts and batch paths). "
        "Same hardware class, not the same measurement session, so S differences of a few points may be measurement "
        "path. The `ΔComp A, S=H2O's` column holds S at H2O's measured S for every arm (isolates I, C, K); where it "
        "and the own-S Δ disagree in sign of significance, the latency path decides the row.",
        "- **Price class** is each system's own declaration (column): zbtbj host-scan price.json per candidate; "
        "sizecliff jobs use the A-PREREG-SIZECLIFF declared $/M input with output 0; 9j3uu and r3uc4 their own "
        "price.json. size_class_estimate rows are estimates, not list tariffs.",
        "- **Kit parity:** torchcast-decision-12b failed kit parity (198 vs 203); its rows are reported, not "
        "comparable as that system's true score.",
        "- **qwen3.5-4b+h2o_shim** is stock Qwen3.5-4B served with H2O's vendored shim readout (no training of "
        "ours); it is the ledger's 'stock 4B + shim, local Composite 67.3' arm.",
        "- **RWKV7-13.3B stock** appears twice (host scan zbtbj and D19's in-job parent 9j3uu): I identical 43.4, "
        "C/S differ by run (82.0/86.1 vs 82.8/83.4), a direct measure of cross-job S noise (~2.7 points).",
        "- **Open public items only**; the board's sealed split may rank differently (open→board I gap on H2O −17.6).",
        "",
        "Latency sources (job, lane, n timed): " + "; ".join(f"{j} {l} n={k}" for j, l, k in lat),
        "",
        "Files: SCORECARD.json (all axes, CIs, pairs, sha256 of every preds/price/report), arms.json (manifest), "
        "reports/ (per-arm jevkit reports), prices/ (sizecliff price classes from each screen.json), "
        "scorecard.png, SHA256SUMS. Code: scripts/ev_scorecard.py, scripts/ev_scorecard_report.py.",
    ]
    Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#d8dee4"
    base_c, win_c, other_c = "#0b6bcb", "#1a7f37", "#8c959f"
    n = len(order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.32 * n + 1.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1]})
    for i, nm in enumerate(order):
        y = n - 1 - i
        x = arms[nm]
        lo, hi = (x.get("ci95") or {}).get("composite", {}).get("A") or (None, None)
        col = base_c if nm == base else (win_c if pairs.get(nm, {}).get("composite_A", {}).get("ci_gt_0") else other_c)
        ax1.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax1.plot([x["composite"]["A"]], [y], "o", color=col, ms=6, mec="white", mew=1.5)
        if nm != base:
            d = pairs[nm]["composite_A"]
            ax2.plot(d["ci95"], [y, y], color=col, lw=2, solid_capstyle="round")
            ax2.plot([d["point"]], [y], "o", color=col, ms=6, mec="white", mew=1.5)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(list(reversed(order)), fontsize=8, color=ink)
    ax2.axvline(0, color=muted, lw=1)
    ax1.set_xlabel("Composite A (local, 231 public items) with 95% CI", color=ink)
    ax2.set_xlabel("Paired Δ Composite A vs H2O-Lightning-4B (95% CI)", color=ink)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
    head = "no arm beats H2O-Lightning-4B" if not any_c else f"{len(any_c)} arm(s) beat H2O-Lightning-4B (green)"
    fig.suptitle(f"Local JevBench Composite A: {head} (blue = H2O; grey = paired CI not > 0). "
                 "Open public items only.", fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.png, dpi=130)
    print(f"best dComp {best_c} {pairs[best_c]['composite_A']}; best dCap {best_k} {pairs[best_k]['capability']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
