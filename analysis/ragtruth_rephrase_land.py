"""B-PREREG-RAGTRUTH-REPHRASE stage 2 landing (CPU): the Index delta of RT1 on full DEV minus proxy600.

Run: python -X utf8 -I analysis/ragtruth_rephrase_land.py <repo_root> <out.json> --arm <d13|d19>
         --landed <the arm's landed full-DEV child.results.jsonl> --rt1 <job's <arm>-dev-rt1.results.jsonl>
         --orig <job's <arm>-dev-original.results.jsonl>

PRIMARY (the pre-registered number): candidate = the landed full-DEV predictions with the DEV catalog-59 rows outside
proxy600 replaced by the job's RT1 rows; baseline = the landed predictions unchanged (original wording). Scored with
the frozen harness.score.decompose and paired with harness.bootstrap.compare (case clusters, B 2000, seed 0) on DEV
minus proxy600 (n 11,460) = the B-PREREG-RR claim mode (analysis/readout_rules_rescore.py). SUCCESS iff the 95% CI
lower bound of (RT1 - original) > 0. Also: the excl_overlength51 variant, (d19) the 4-row lineage-exclusion variant,
and a sensitivity pairing against the job's own original-wording rows (same splice). Nothing is fitted.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from harness import config as _config  # noqa: E402

try:
    import decision_index  # noqa: F401
except ImportError:
    sys.path.insert(1, str(_config.KIT_CLONE))
from harness import bootstrap, score  # noqa: E402
from harness.cli import load_slice  # noqa: E402

EDITION = "0.2.1"
B = 2000
CATALOG = 59
#: B-PREREG-CLAIM addendum 2 lineage exclusions (D17/D18/D19 lineage), matched by run_id substring
LINEAGE4 = ("SATA-Bench test:1427", "BRIGHT earth_science:54", "Amazon-ESCI test:2163000", "HoVer candidates-v3:61:461ff029")


def read_pred(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            out[j["run_id"]] = j
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def ci(d):
    return {k: round(d[k], 4) for k in ("point", "lo", "hi") if k in d} | {"excludes_zero": d.get("excludes_zero")}


def pair(rows, cand, base):
    ids = {r["_evaluation"]["run_id"] for r in rows}
    dc = score.decompose(rows, {k: v for k, v in cand.items() if k in ids}, edition=EDITION)
    db = score.decompose(rows, {k: v for k, v in base.items() if k in ids}, edition=EDITION)
    c = bootstrap.compare(dc, db, B=B, seed=0)
    return {"n_rows": len(rows), "ids_sha256": ids_sha256(ids), "index_candidate": round(score.point(dc)["index"], 4),
            "index_baseline": round(score.point(db)["index"], 4), "delta": ci(c["diff"]["index"]),
            "benchmarks_changed": {n: ci(v) for n, v in c["diff"]["benchmarks"].items() if abs(v["point"]) > 1e-12}}


def main():
    from jobs.m2_first_cell.fulldev_eval import overlength_ids
    from jobs.m2_first_cell.ragtruth_rephrase import proxy600_ids

    a = sys.argv[3:]
    opt = {a[i]: a[i + 1] for i in range(0, len(a) - 1, 2)}
    arm = opt["--arm"]
    landed, rt1, orig = read_pred(opt["--landed"]), read_pred(opt["--rt1"]), read_pred(opt["--orig"])
    dev = load_slice("dev", EDITION)
    assert len(dev) == 12_104, len(dev)
    pids = proxy600_ids()
    rows = [r for r in dev if r["_evaluation"]["run_id"] not in pids]
    assert len(rows) == 11_460, len(rows)
    swap = {r["_evaluation"]["run_id"] for r in rows if r["_evaluation"]["catalog_id"] == CATALOG}
    assert set(rt1) == swap == set(orig), (len(swap), len(rt1), len(orig))
    assert set(landed) >= {r["_evaluation"]["run_id"] for r in dev}
    cand = {**landed, **{k: rt1[k] for k in swap}}
    base_job = {**landed, **{k: orig[k] for k in swap}}
    dp = [abs(float(orig[k]["response"]["answers"]["q"]["noul"]) - float(landed[k]["response"]["answers"]["q"]["noul"]))
          for k in swap if orig[k].get("status") == "ok" and landed[k].get("status") == "ok"]
    ol = overlength_ids()
    rows_x = [r for r in rows if r["_evaluation"]["run_id"] not in ol]
    out = {"prereg": "B-PREREG-RAGTRUTH-REPHRASE (state PREREG.md @0683fa7c)", "arm": arm, "edition": EDITION,
           "scorer": f"harness.score.decompose + bootstrap.compare (B={B}, seed 0)",
           "files": {k: {"path": opt[k], "sha256": sha(opt[k])} for k in ("--landed", "--rt1", "--orig")},
           "n_swapped_rows": len(swap), "swapped_ids_sha256": ids_sha256(swap),
           "reproduction_original_vs_landed": {"n": len(dp), "max_abs_dp": max(dp) if dp else None},
           "primary_dev_minus_proxy600": pair(rows, cand, landed),
           "dev_minus_proxy600_excl_overlength51": pair(rows_x, cand, landed),
           "sensitivity_vs_job_original": pair(rows, cand, base_job)}
    if arm == "d19":
        rows_l = [r for r in rows if not any(t in r["_evaluation"]["run_id"] for t in LINEAGE4)]
        out["dev_minus_proxy600_minus_lineage4"] = pair(rows_l, cand, landed)
    lo = out["primary_dev_minus_proxy600"]["delta"]["lo"]
    out["success"] = bool(lo > 0)
    out["verdict"] = ("SUCCESS: RT1 Index delta CI lower bound > 0 (still needs the independent re-measurement before "
                      "it becomes primary)") if out["success"] else "FAIL: RT1 retired for this arm (DEAD_ENDS)"
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("arm", "n_swapped_rows", "reproduction_original_vs_landed", "verdict")}))
    print(json.dumps(out["primary_dev_minus_proxy600"]["delta"]))


if __name__ == "__main__":
    main()
