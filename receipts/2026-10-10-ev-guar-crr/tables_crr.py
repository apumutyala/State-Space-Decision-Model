"""Print the GUARANTEES_CRR.md tables from GUARANTEES_CRR.json:  python -X utf8 -I tables_crr.py GUARANTEES_CRR.json"""
import json
import sys

J = json.load(open(sys.argv[1], encoding="utf-8"))
NAMES = {"D19": "D19", "D13": "D13", "ClefFlash": "Clef-flash", "Clef27B": "Clef 27B"}
V = ("published", "cluster", "crr8", "crr1", "crr_max")


def lam_txt(rep, an):
    if an == "marginal":
        lam = rep["marginal_ltt"]["lambda"]
        return "" if lam is None else f" (λ {lam:.2f})"
    got = [f"{g} {r['lambda']:.2f}" for g, r in sorted(rep["area_ltt"].items()) if r.get("lambda") is not None]
    return f" ({', '.join(got)})" if got else ""


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


for mode in ("run_id", "case"):
    print(f"\n## {mode}  caps {J[mode]['split']['caps']}")
    for a in ("0.1", "0.05"):
        print(f"\n### alpha {a}: TEST act fraction  published / one-per-row / CRR8 [CRR1, CRRmax]")
        print("| Model | Marginal: published | one-per-row | **CRR c=8** | c=1 | uncapped | Per-area: published | one-per-row | **CRR c=8** | c=1 | uncapped |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for m, nm in NAMES.items():
            r = J[mode]["models"][m]["alpha"][a]
            cells = []
            for an in ("marginal", "area"):
                for v in V:
                    t = r[v][an + "_test"]["act"]
                    s = f"{t:.3f}" + (lam_txt(r[v], an) if v.startswith("crr8") else "")
                    cells.append(f"**{s}**" if v == "crr8" else s)
            print(f"| {nm} | " + " | ".join(cells) + " |")
        print("\nTEST acted error at CRR8: decision-weighted / cluster-weighted / cap8-weighted; LTT p, clusters")
        for m, nm in NAMES.items():
            r = J[mode]["models"][m]["alpha"][a]["crr8"]
            for an in ("marginal", "area"):
                t = r[an + "_test"]
                print(f"  {nm} {an}: act {t['act']:.3f} n_act {t['n_act']} err {pct(t['acted_error'])} / "
                      f"{pct(t['acted_error_cluster_weighted'])} / {pct(t['acted_error_cap8_weighted'])}")
            ml = r["marginal_ltt"]
            print(f"    marginal LTT: {ml}")
            for g, rep in sorted(r["area_ltt"].items()):
                print(f"    area {g}: lam {rep.get('lambda')} p {rep.get('p_value')} nclu {rep.get('n_clusters')} "
                      f"ndec {rep.get('n_decisions')} werr {rep.get('accepted_error_cal_weighted')}")
    print("\n### paired contrasts")
    for pair, d in J[mode]["bootstrap"]["pairs"].items():
        for a, dv in d.items():
            for an in ("marginal", "area"):
                c, k = dv["cluster"][an], dv["crr8"][an]
                p = dv["published"][an]
                print(f"| {pair} | {a} | {an} | {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] {p['verdict']} | "
                      f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {c['verdict']} | "
                      f"**{k['diff']:+.3f} [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}] {k['verdict']}** |")
    print("\n### within model crr8 - cluster / crr8 - published")
    for m, d in J[mode]["bootstrap"]["within_model"].items():
        for a, dv in d.items():
            for k, x in dv.items():
                for an, y in x.items():
                    print(f"  {m} {a} {k} {an}: {y['diff']:+.3f} [{y['ci95'][0]:+.3f}, {y['ci95'][1]:+.3f}]")
print("\nreproduction exact:", all(r["exact"] for r in J["reproduction"]), len(J["reproduction"]))
