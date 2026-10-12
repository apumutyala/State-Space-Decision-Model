"""Markdown tables from guarantees_jev3.json (B-PREREG-GUAR-JEV3). python -I -X utf8 tables_jev3.py <json>"""
import json
import sys


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main(p):
    R = json.load(open(p, encoding="utf-8"))
    for name, src in R["sources"].items():
        print(f"\n### {name}  (n {src['n']}, top-1 {src['native_top1']:.4f}, dev-card {src['dev_card_n']})")
        if "shim_parity" in src:
            print(f"shim parity max |d| {src['shim_parity']['max_abs_diff_vs_vendored_format_answer']:.2e}")
        for arm, subs in src["arms"].items():
            for sub, res in subs.items():
                sp = res["split"]
                print(f"\n**{name} / {arm} / {sub}** FIT {sp['n_fit']} LTT {sp['n_ltt']} TEST {sp['n_test']}; "
                      f"types {sp['type_counts']}")
                print("| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err "
                      "| FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |")
                print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for r in res["results"]:
                    s = r["sets_test"]["all"]
                    sel = r["selective_test_primary"]["all"]
                    c1 = r["secondary"]["fs_crr_c1"]
                    fb = r["secondary"]["fs_binomial"]
                    pt = r["secondary"]["per_type_ltt"]["test"]["all"]
                    print(f"| {r['alpha']} | {r['variant']} | {f(s['coverage'])} | {'FLAG' if s['flagged'] else 'ok'} | "
                          f"{f(s['mean_set_size'], 2)} | {f(s['singleton_rate'])} | {f(s['singleton_accuracy'])} | "
                          f"{f(r['ltt_primary']['lambda'], 2)} | {f(sel['act_fraction'])} | {f(sel['acted_error'])} | "
                          f"{f(c1['ltt']['lambda'], 2)} / {f(c1['test']['act_fraction'])} / {f(c1['test']['acted_error'])} | "
                          f"{f(fb['ltt']['lambda'], 2)} / {f(fb['test']['act_fraction'])} / {f(fb['test']['acted_error'])} | "
                          f"{f(pt['act_fraction'])} / {f(pt['acted_error'])} |")
                # per-type / per-family for variant T
                for r in res["results"]:
                    if r["variant"] != "T":
                        continue
                    print(f"\nper type / family, variant T, alpha {r['alpha']} (groups {json.dumps(r['groups'])}; "
                          f"pooled-small {r['pooled_small_members']} full-set {r['pooled_small_full_set']})")
                    print("| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |")
                    print("|---|---|---|---|---|---|---|---|---|---|")
                    for key in ("per_type", "per_family"):
                        for k, s in r["sets_test"][key].items():
                            sel = r["selective_test_primary"][key][k]
                            pt = r["secondary"]["per_type_ltt"]["test"][key][k]
                            print(f"| {k} | {s['n']} | {f(s['coverage'])} | {f(s['mean_set_size'], 2)} | "
                                  f"{f(s['singleton_rate'])} | {f(s['singleton_accuracy'])} | {f(s['top1_accuracy'])} | "
                                  f"{f(sel['act_fraction'])} | {f(sel['acted_error'])} | {f(pt['act_fraction'])} / {f(pt['acted_error'])} |")
                    a = r["ltt_half_attribution"]
                    print(f"LTT-half attribution: best lambda by p {json.dumps(a['best_lambda_by_p'])}; "
                          f"Bonferroni level {a['bonferroni_level']:.2e}; zero-error acts needed {a['zero_error_acts_needed_bonferroni']} "
                          f"(single test {a['zero_error_acts_needed_single_test']}); LTT singleton rate {a['ltt_singleton_rate']:.3f}; "
                          f"per-type lambdas { {t: v['lambda'] for t, v in r['secondary']['per_type_ltt']['ltt'].items()} }; "
                          f"FS-CRR c1 stop {json.dumps(r['secondary']['fs_crr_c1']['ltt'].get('fs_stop'))}; "
                          f"FS-CRR c8 lam {r['secondary']['fs_crr_c8']['ltt']['lambda']} stop {json.dumps(r['secondary']['fs_crr_c8']['ltt'].get('fs_stop'))}")


if __name__ == "__main__":
    main(sys.argv[1])
