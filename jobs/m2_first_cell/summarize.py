"""Compose the first-cell result (``first_cell_result.json``, the decision file ``finalize`` validates) from what the
job's phases wrote. CPU only; no model.

    python -X utf8 -m jobs.m2_first_cell.summarize --cell <out>/cell.json --report <out>/report.json \
        --parent-arm <ck>/parent_arm --index-rows <subset rows> --params jobs/m2_first_cell/job_params.json \
        --elapsed-min N --out <out>/first_cell_result.json [--reference name=results.jsonl ...]

Sections: training (steps, rows, loss, measured step time, peak memory); Index DEV subset (child and parent score with
95% CI overall and per area, the paired child - parent difference overall and per area and split at the training
length T = 2,048, ECE per arm); the readout
diagnostics (position / label bias, forward vs reversed option order); the cohort dev card (raw and tempered ECE);
steer and hijack under D1-A (paired child - parent, from the G12 units); latency (per Index row, both arms); cost; the
G12 verdict and the promotion verdict under the PROVISIONAL margins. ``--reference`` adds external result files on the
same rows (e.g. the job-tkfaj stock readout, Clef-flash): each is scored on the subset and compared with the child; the
in-job parent is also compared with the tkfaj stock readout (an independent re-measurement of the parent).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "ssdm.m2.first_cell.v1"
MARGINS_STATUS = ("issue #36 margins as train.promotion.MARGINS states them (PR #102: approved, Codex cross-track "
                  "decision); this cell runs G12 + the index measurement, not G13, so it is never promotable here")


def _load(p) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _results_jsonl(p) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["run_id"]] = r
    return out


def latency(results: dict) -> dict:
    ms = sorted(float(r["total_wall_ms"]) for r in results.values() if r.get("status") == "ok" and "total_wall_ms" in r)
    if not ms:
        return {"n": 0}
    q = lambda f: ms[min(len(ms) - 1, int(f * len(ms)))]  # noqa: E731
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": q(0.5), "p90_ms": q(0.9), "p99_ms": q(0.99),
            "scope": "decide() wall time per Index DEV row (all questions of the row), H100, batch 1"}


def training(cell: dict) -> dict:
    tr = cell.get("train") or {}
    v = tr.get("verdict") or {}
    cfg = cell["config"]
    rows_per_step = cfg["batch_size"] * cfg["grad_accum"]
    return {"steps": tr.get("steps"), "rows_seen": (tr.get("steps") or 0) * rows_per_step, "T_max": tr.get("T_max"),
            "loss_first_window": v.get("loss_first"), "loss_last_window": v.get("loss_last"),
            "loss_drop_frac": v.get("loss_drop_frac"), "pass_rule": v, "rate_check": cell.get("train_rate_check"),
            "ladder": cell.get("ladder"), "smoke": {k: (cell.get("smoke") or {}).get(k) for k in
                                                    ("pass", "rel_feature_diff", "grad_cos")},
            "data": cell.get("data"), "stopped": tr.get("stopped"),
            "eval_cards_nll": [c["raw"].get("nll") for c in tr.get("cards") or []]}


def keep_compare(cmp: dict) -> dict:
    """The reported fields of one ``safety_eval.metrics.compare_arms`` result (x = child, y = parent)."""
    from safety_eval import metrics as M

    keep = {k: cmp.get(k) for k in ("n_units", "n_eligible", "complete", "incomplete_reason", "clean_acc_x",
                                    "clean_acc_y", "steer_rate_conservative_diff", "bootstrap", "eligibility")}
    for name in M.COMPARE_METRICS:
        keep[name] = {"child": cmp[name]["x"], "parent": cmp[name]["y"], "diff": cmp[name]["diff"]}
    return keep


def steer_hijack(child_rows: list, parent_rows: list, cohort, n_boot: int) -> dict:
    from safety_eval import metrics as M
    from train import m2_gates as G
    from train.promotion import _family_ids

    variants = G.stage_variants(cohort, "g12")
    ids = _family_ids(variants, "d1_A")
    cmp = M.compare_arms(M.with_expected(child_rows, variants), M.with_expected(parent_rows, variants), "d1_A",
                         n_boot=n_boot, eligibility="reference_y", expected_ids=ids)
    keep = keep_compare(cmp)
    keep["note"] = ("x = child, y = parent; D1-A templates (development split). D1-A is in the training data verbatim "
                    "(PR #44 injection augmentation), so this is IN-DISTRIBUTION robustness; held-out D1-B is G13, "
                    "not run here")
    return keep


def length_split(rows: list, child_res: dict, parent_res: dict, n_boot: int, lengths_path=None) -> dict:
    """Child - parent Index difference with paired CIs, split by kit-group prompt length at the training cap
    (T = 2,048, ``jobs/m2_first_cell/index_lengths_rwkv7.json``): does the gain hold on prompts longer than any
    training row? Each bin is scored with the kit over whole groups (benchmark weights renormalised in the bin)."""
    from harness import bootstrap as HB
    from harness import score as S
    from jobs.m2_first_cell import index_lengths as IL

    d = IL.load(lengths_path) if lengths_path else IL.load()
    want = {r["_evaluation"]["run_id"] for r in rows}
    if not want <= set(d["lengths"]):
        return {"error": "length file does not cover these rows"}
    out = {"split_t": d["split_t"], "unit": "kit group (its longest training-layout row, RWKV7 tokens)",
           "length_file_subset_sha256": d["subset_ids_sha256"]}
    for name, rs in IL.group_bins(rows, d["lengths"], d["split_t"]).items():
        if not rs:
            out[name] = {"n_rows": 0}
            continue
        try:
            cmp = HB.compare(S.decompose(rs, child_res), S.decompose(rs, parent_res), B=n_boot, seed=0)
            out[name] = {"n_rows": len(rs), "child": cmp["candidate"]["index"], "parent": cmp["baseline"]["index"],
                         "child_minus_parent": cmp["diff"]["index"],
                         "areas_child_minus_parent": cmp["diff"]["areas"]}
        except Exception as e:  # noqa: BLE001 - a bin the kit cannot score is reported, never silently dropped
            out[name] = {"n_rows": len(rs), "error": f"{type(e).__name__}: {e}"}
    return out


def references(refs: dict[str, str], rows: list, child_res: dict, parent_res: dict, n_boot: int) -> dict:
    from harness import bootstrap as HB
    from harness import score as S

    out = {}
    for name, path in refs.items():
        res = _results_jsonl(path)
        miss = [r["_evaluation"]["run_id"] for r in rows if r["_evaluation"]["run_id"] not in res]
        if miss:  # a non-ok row is scored as the kit scores it; a MISSING row means another row set
            out[name] = {"error": f"{len(miss)} subset rows missing from {path}", "rows": miss[:5]}
            continue
        cmp = HB.compare(S.decompose(rows, child_res), S.decompose(rows, res), B=n_boot, seed=0)
        ent = {"path": str(path), "n_not_ok": sum(res[r["_evaluation"]["run_id"]].get("status") != "ok" for r in rows),
               "score": cmp["baseline"]["index"], "areas": cmp["baseline"]["areas"],
               "child_minus_reference": {"index": cmp["diff"]["index"], "areas": cmp["diff"]["areas"]}}
        if "stock" in name or "tkfaj" in name:  # independent re-measurement of the in-job parent
            pc = HB.compare(S.decompose(rows, parent_res), S.decompose(rows, res), B=n_boot, seed=0)
            agree = n = 0
            for r in rows:
                a = ((parent_res[r["_evaluation"]["run_id"]].get("response") or {}).get("answers") or {})
                b = ((res[r["_evaluation"]["run_id"]].get("response") or {}).get("answers") or {})
                for k in r["questions"]:
                    if k in a and k in b:
                        n += 1
                        agree += a[k].get("choice") == b[k].get("choice")
            ent["in_job_parent_minus_reference"] = pc["diff"]["index"]
            ent["parent_choice_agreement"] = agree / n if n else float("nan")
        out[name] = ent
    return out


def summarize(a) -> dict:
    from harness.suite import read_rows
    from train.cohort import Cohort
    from jobs.common.costs import worst_case_usd

    cell, rep = _load(a.cell), _load(a.report)
    params = _load(a.params)
    rows = read_rows(a.index_rows)
    child_g12 = _load(Path(a.report).with_name("units-child-g12.json"))
    child_idx = _load(Path(a.report).with_name("units-child-index.json"))
    par_g12 = _load(Path(a.parent_arm) / "units-g12.json")
    par_idx = _load(Path(a.parent_arm) / "units-index.json")
    proxy = (rep.get("stages", {}).get("index") or {}).get("index_proxy")
    if proxy:  # params rows "proxy600": the child scored a pre-declared proxy; every pairing here uses those rows only
        from jobs.common.dev_subset import ids_sha256

        pids = set(child_idx["index_results"])
        if ids_sha256(pids) != proxy["ids_sha256"]:
            raise SystemExit("child index results do not cover exactly the report's index proxy rows")
        rows = [r for r in rows if r["_evaluation"]["run_id"] in pids]
        par_idx = dict(par_idx, index_results={k: v for k, v in par_idx["index_results"].items() if k in pids})
    cohort = Cohort(expected_sha256=cell["config"]["cohort_sha256"])
    vi = rep["verdicts"]["index"]
    usd = params["batch_usd_per_min"]
    out = {
        "schema": SCHEMA,
        "cell": {k: cell["config"].get(k) for k in ("parent", "head", "loss_arm", "lora_rank", "template", "layout",
                                                     "n_train_records", "max_minutes", "max_steps", "seed")}
        | {"train_set": cell["config"]["train_set"]["pointer"]},
        "training": training(cell),
        "index_dev_subset": {
            "rows": len(rows), "index_proxy": proxy or None, "child": (vi.get("scores") or {}).get("child"),
            "parent_stock_readout": (vi.get("scores") or {}).get("parent"),
            "child_minus_parent": (vi.get("scores") or {}).get("diff"),
            "criteria_informational": vi.get("criteria"),
            "note": "child = trained head (untempered, like-for-like with the parent's identity calibration); "
                    "parent = stock option-likelihood readout of the same RWKV7 weights, scored in this job"},
        "ece": {"index_dev_subset": vi.get("ece"),
                "cohort_dev_card": (rep["identities"].get("card") or {})},
        "readout_diagnostics": vi.get("readout_diagnostics"),
        "steer_hijack_d1a": steer_hijack(child_g12["steer_rows"], par_g12["steer_rows"], cohort, a.n_boot),
        "index_by_prompt_length": length_split(rows, child_idx["index_results"], par_idx["index_results"], a.n_boot,
                                               a.index_lengths),
        "decision_dev_g12": rep["verdicts"]["g12"].get("paired"),
        "latency": {"child": latency(child_idx["index_results"]), "parent": latency(par_idx["index_results"])},
        "g12": {k: rep["verdicts"]["g12"][k] for k in ("accept", "failed", "criteria")},
        "promotion": promotion_block(rep),
        "cost": {"elapsed_min": a.elapsed_min, "runtime_usd": round(a.elapsed_min * usd, 2),
                 "usd_per_min": usd, "build_minutes_not_included": True,
                 "worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"]},
        "verdict": "measured",
    }
    if a.reference:
        refs = dict(x.split("=", 1) for x in a.reference)
        out["references"] = references(refs, rows, child_idx["index_results"], par_idx["index_results"], a.n_boot)
    return out


#: m2_gates exit codes that are a completed measurement: accept, G12 reject, promotion PENDING (Codex #36, PR #102)
MEASURED_EVAL_EXITS = {0: "promotable", 40: "G12 rejected", 43: "margins pending approval",
                      44: "partial: G12 + index measurement only, G13 not run",
                      45: "underpowered: a criterion was evaluated below its simulated required n"}


def promotion_block(rep: dict) -> dict:
    """The report's promotion verdict, labelled with the m2_gates exit code (PR #102, Codex #36). 40 / 43 / 44 / 45
    are trained + measured outcomes; only 0 is promotable. Never makes the result invalid."""
    pv = dict(rep.get("promotion") or {})
    code = rep.get("exit_code")
    reason = MEASURED_EVAL_EXITS.get(code, f"m2_gates exit {code}")
    promotable = bool(pv.get("promotable")) and code == 0
    pv.update(margins_status=MARGINS_STATUS, gates_this_measurement=False, evals_exit_code=code,
              promotion_status=reason, promotable=promotable, not_promotable_reason=None if promotable else reason)
    return pv


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a trained, scored, self-consistent first-cell result."""
    try:
        ix = d["index_dev_subset"]
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        return (d.get("schema") == SCHEMA and d.get("verdict") == "measured" and d["training"]["steps"] > 0
                and all(fin(ix[k]["index"][f]) for k in ("child", "parent_stock_readout") for f in ("point", "lo", "hi"))
                and fin(ix["child_minus_parent"]["index"]["point"]) and set(ix["child"]["areas"])
                and d["steer_hijack_d1a"]["n_units"] > 0 and d["latency"]["child"]["n"] > 0
                and d["promotion"]["verdict"] in ("accept", "reject") and d["promotion"]["evals_exit_code"] in MEASURED_EVAL_EXITS and d["readout_diagnostics"]["order"]["child"])
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.summarize")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--parent-arm", required=True)
    ap.add_argument("--index-rows", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--elapsed-min", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", action="append", default=[], help="name=results.jsonl (same subset rows)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--index-lengths", default=None, help="default: jobs/m2_first_cell/index_lengths_rwkv7.json")
    a = ap.parse_args(argv)
    res = summarize(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    ix = res["index_dev_subset"]
    print(json.dumps({"child": ix["child"]["index"] if ix["child"] else None,
                      "parent": ix["parent_stock_readout"]["index"] if ix["parent_stock_readout"] else None,
                      "valid": valid_result(res)}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    sys.exit(main())
