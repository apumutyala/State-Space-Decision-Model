"""GAP_AND_FUNDING (agent-b, 2026-10-09): D13 vs Clef-flash and Clef 27B per Index family on proxy600. CPU only.

Run (repo root = this worktree; prediction files are data, read with json.loads, never executed):
    CUDA_VISIBLE_DEVICES= python -X utf8 -I analysis/gap_funding/gap_funding.py <repo_root> <out.json>

Only the frozen scorer is called (harness.score.decompose / weights / point; harness.bootstrap.compare, B=2000, seed 0,
case clusters). Nothing frozen is edited. Each family's Index gap = 100 * w_n * (skill_D13 - skill_ref): exact and additive
(each benchmark's skill depends only on its own rows, weights are fixed), asserted against the scorer's Index difference.
"""
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from harness import config as _config  # noqa: E402

sys.path.insert(1, str(_config.KIT_CLONE))
from harness import bootstrap, score  # noqa: E402
from harness.cli import load_slice  # noqa: E402
from harness.suite import spec  # noqa: E402
from jobs.common.dev_proxy import load_proxy  # noqa: E402

EDITION = "0.2.1"
U = Path(r"C:\Users\apumu")
FILES = {
    "d13": U / "d13_land/dl/x/final/out/screen/child.results.jsonl",
    "d1v3w": U / "rab_d1_repro/final/out/readout_ab/child.v3w.results.jsonl",
    "clef27": U / "clef_zfexa_land/x/clef-index-full/index/decide_32k_substituted/results.jsonl",
    "flash": U / "clef_t5yg2_land/dl/clef-flash-index-full/index/decide_32k_substituted/results.jsonl",
}
FROZEN = {9, 20, 22, 31, 32, 33, 44, 58, 64}  # heldout-draw-v1: no data / generator / distillation may target these


def read(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["run_id"]] = r
    return out


prox = load_proxy("proxy600")
ids = set(prox["run_ids"])
rows = [r for r in load_slice("dev", EDITION) if r["_evaluation"]["run_id"] in ids]
assert len(rows) == 644
res = {k: {i: v for i, v in read(p).items() if i in ids} for k, p in FILES.items()}
for k, r in res.items():
    assert set(r) == ids, k
benches = {k: score.decompose(rows, r, edition=EDITION) for k, r in res.items()}
pt = {k: score.point(b, edition=EDITION)["index"] for k, b in benches.items()}
S = spec(EDITION)
area = {n: a["id"] for a in S["areas"] for n in a["benchmarks"]}
w = score.weights(set(benches["d13"]), True, EDITION)

# per-benchmark point skill from the frozen scorer's point()
skill = {k: {n: v["skill"] for n, v in score.point(b, edition=EDITION)["per_benchmark"].items()} for k, b in benches.items()}

shape = {}
for r in rows:
    n = r["_evaluation"]["catalog_id"]
    s = shape.setdefault(n, {"family": r["family"], "rows": 0, "qs": 0, "K": [], "tok": [], "case_q": []})
    qs = r["questions"]
    s["rows"] += 1
    s["qs"] += len(qs)
    s["case_q"].append(len(qs))
    for q in qs.values():
        s["K"].append(len(q.get("criteria") or {}) if q.get("type") == "choice" else 1)
    s["tok"].append(res["flash"][r["_evaluation"]["run_id"]].get("proxy_tokens") or 0)

# context: D13 rows with no result (input cap) -> transplant Clef-flash's result for exactly those rows (M)
bad = [i for i, v in res["d13"].items() if v.get("status") != "ok"]
ctx_gain = {}
for ref in ("flash", "clef27"):
    trans = dict(res["d13"])
    for i in bad:
        trans[i] = res[ref][i]
    ctx_gain[ref] = score.point(score.decompose(rows, trans, edition=EDITION), edition=EDITION)["index"] - pt["d13"]

fam = []
for n in sorted(skill["d13"], key=int):
    n_i = int(n)
    s = shape[n_i]
    fam.append({
        "n": n_i, "family": s["family"], "area": area.get(n_i), "frozen": n_i in FROZEN, "w": w[n_i],
        "rows": s["rows"], "questions": s["qs"], "max_K": max(s["K"] or [0]),
        "median_K": statistics.median(s["K"]) if s["K"] else 0, "q_per_row": round(s["qs"] / s["rows"], 2),
        "median_tok": statistics.median(s["tok"]), "frac_tok_gt2048": round(sum(t > 2048 for t in s["tok"]) / len(s["tok"]), 2),
        "d13_not_ok": sum(1 for r in rows if r["_evaluation"]["catalog_id"] == n_i
                          and res["d13"][r["_evaluation"]["run_id"]].get("status") != "ok"),
        **{f"skill_{k}": round(100 * skill[k][n], 2) for k in FILES},
        "gap_flash_pts": round(100 * w[n_i] * (skill["d13"][n] - skill["flash"][n]), 3),
        "gap_clef27_pts": round(100 * w[n_i] * (skill["d13"][n] - skill["clef27"][n]), 3),
        "d13_minus_d1v3w_pts": round(100 * w[n_i] * (skill["d13"][n] - skill["d1v3w"][n]), 3),
    })
for ref in ("flash", "clef27"):
    tot = sum(f[f"gap_{ref}_pts"] for f in fam)
    assert abs(tot - (pt["d13"] - pt[ref])) < 0.02, (ref, tot, pt["d13"] - pt[ref])

cmp = {}
for ref in ("flash", "clef27", "d1v3w"):
    c = bootstrap.compare(benches["d13"], benches[ref], B=2000, seed=0)
    cmp[ref] = {"d13": c["candidate"]["index"], "ref": c["baseline"]["index"], "diff": c["diff"]["index"],
                "areas": c["diff"]["areas"]}
out = {"rows": 644, "proxy_ids_sha256": prox["ids_sha256"], "files": {k: str(p) for k, p in FILES.items()},
       "index_point": pt, "paired": cmp, "context_transplant": {"d13_not_ok_rows": bad, "gain_pts": ctx_gain},
       "families": fam}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print({k: round(v, 2) for k, v in pt.items()}, "ctx transplant", {k: round(v, 3) for k, v in ctx_gain.items()}, "bad rows", len(bad))
for f in sorted(fam, key=lambda f: f["gap_flash_pts"]):
    print(f"{f['n']:>3} {f['family'][:28]:28} {str(f['area'])[:9]:9} {'F' if f['frozen'] else ' '} w={f['w']:.3f} "
          f"K={f['max_K']:>3} q/r={f['q_per_row']:>5} tok={f['median_tok']:>6} >2k={f['frac_tok_gt2048']:.2f} "
          f"d13={f['skill_d13']:6.1f} fl={f['skill_flash']:6.1f} c27={f['skill_clef27']:6.1f} d1={f['skill_d1v3w']:6.1f} "
          f"gapF={f['gap_flash_pts']:+6.2f} gap27={f['gap_clef27_pts']:+6.2f}")
