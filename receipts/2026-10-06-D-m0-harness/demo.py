import json, sys, collections
from harness.cli import load_slice
from harness import predictors, score, bootstrap, calibration, mde
from harness.suite import load_suite
out = {}
rows = load_slice("dev")
base = predictors.predict(rows, "noisy", seed=6, acc=0.55)
cand = predictors.predict(rows, "noisy", seed=7, acc=0.60)
cmp_ = bootstrap.compare(score.decompose(rows, cand), score.decompose(rows, base), B=2000, seed=0)
nclu = sum(len(b.clusters) for b in score.decompose(rows, base).values())
out["compare_noisy_0.60_vs_0.55"] = {"diff_index": cmp_["diff"]["index"], "diff_areas": cmp_["diff"]["areas"], "candidate_index": cmp_["candidate"]["index"], "baseline_index": cmp_["baseline"]["index"], "clusters": nclu, "mde_table": mde.table(cmp_["diff"]["index"]["se"], nclu)}
out["calibration_noisy_0.60_onehot"] = calibration.report(calibration.fields(rows, cand))
# board ECE set check: fields of the inferred 32 benchmarks over the whole local suite
ex = {2, 36, 6, 48, 34}
n_fields = collections.Counter()
for r in load_suite().rows(apply_exclusions=True):
    c = r["_evaluation"]["catalog_id"]
    if c < 56 and c not in ex:
        n_fields[c] += sum(1 for k, g in r["expected"].items() if g is not None)
tot = sum(n_fields.values()) + 501  # + HLE (1 field per row, not built locally)
out["board_ece_set_check"] = {"benchmarks": len(n_fields) + 1, "fields_total_incl_hle": tot, "one_sixth": tot / 6, "board_n": 72594}
json.dump(out, sys.stdout, indent=1)
