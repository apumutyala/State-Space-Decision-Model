"""Mode ``ragtruth_rephrase`` (B-PREREG-RAGTRUTH-REPHRASE, state PREREG.md @0683fa7c): the train-split check of the ONE
pre-registered RAGTruth statement rephrase (RT1, ``decide.ragtruth_rephrase``) on D13 and D19.

Trigger: ``ragtruth_polarity`` (job-3zmj2, art-srefc) found D13's RAGTruth ranking INVERTED on 300 RAGTruth TRAIN items
(AUROC 0.418 [0.353, 0.481]). Same items (``ragtruth_polarity`` build, ids 794bcce3..., items dbbca4ff...), same
readout v3w untempered, same AUROC + percentile bootstrap (B 2000, seed 20261009, ``ragtruth_polarity.bootstrap_auroc``).

Scored in the job: D13 (art-cr9a2 best1600) under RT1; D19 (art-p3gpi best2800) under the original wording AND RT1.
D13's original-wording predictions are job-3zmj2's ``d13.results.jsonl`` (shipped, sha-pinned; same checkpoint, readout
and items). Pre-registered rule (``decide_arm``):

    PASS for an arm iff RT1's AUROC 95 % lower bound > 0.5 on the 300 items;
    D19 additionally: RT1's point AUROC on the items OUTSIDE its training draw > 0.5 (175 of the 300 are in D19's
    bschema-draw-v3b draw with the ORIGINAL wording and labels; ``ragtruth_rephrase_d19_indraw.json``), else
    "in-draw driven" and no DEV stage.

Reported, never deciding: paired dAUROC (RT1 - original, the same bootstrap draws), per-track AUROC, the in-draw /
out-of-draw split, mean p, fraction predicted hallucinated, F1 / accuracy at 0.5.

    python -X utf8 -m jobs.m2_first_cell.ragtruth_rephrase indraw --manifest <v3b manifest> --items <items> --out <json>
    python -X utf8 -m jobs.m2_first_cell.ragtruth_rephrase estimate --params <job_params>
    # GPU, once per arm (the model loads once; every wording of the arm is scored from it):
    python -X utf8 -m jobs.m2_first_cell.ragtruth_rephrase score --arm d19 --wordings original,rt1 ...
    python -X utf8 -m jobs.m2_first_cell.ragtruth_rephrase report --out-dir <out> --params <p> --items <items> ...
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from jobs.m2_first_cell import ragtruth_polarity as RP

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ssdm.m2.ragtruth_rephrase.v1"
PREREG = "B-PREREG-RAGTRUTH-REPHRASE (state PREREG.md @0683fa7c)"
CANDIDATE = "rt1"
WORDINGS = ("original", CANDIDATE)
ARM_NAMES = ("d13", "d19")
#: arms whose training draw holds some of the items (with the original wording + labels): the contamination guard
INDRAW_GUARD_ARMS = ("d19",)
NULL_AUROC = 0.5
N_BOOT = RP.N_BOOT
BOOT_SEED = RP.BOOT_SEED
INDRAW_FILE = "jobs/m2_first_cell/ragtruth_rephrase_d19_indraw.json"
DECISION_RULE = ("per arm: PASS iff RT1's AUROC 95% percentile-bootstrap lower bound (2000 item resamples, seed "
                 "20261009) > 0.5 on the 300 RAGTruth TRAIN items; D19 additionally needs RT1's point AUROC > 0.5 on "
                 "the items outside its training draw (else 'in-draw driven', no DEV stage). Only a passing arm gets "
                 "the DEV-minus-proxy600 stage.")
SLOW_BOX = 1.25


def read_jsonl(path: str | Path) -> list[dict]:
    return RP.read_jsonl(path)


def _sha(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ D19 in-draw ids
def indraw_ids(manifest: str | Path, item_ids: set[str]) -> list[str]:
    """Item response ids that appear in a bschema draw manifest as a RAGTruth row (``m2:ragtruth:<id>~view...``)."""
    hit = set()
    with gzip.open(manifest, "rt", encoding="utf-8") as f:
        for line in f:
            if "ragtruth" not in line:
                continue
            rid = str(json.loads(line).get("id", ""))
            if rid.startswith("m2:ragtruth:"):
                tail = rid[len("m2:ragtruth:"):].split("~", 1)[0]
                if tail in item_ids:
                    hit.add(tail)
    return sorted(hit)


def items_response_ids(items: list[dict]) -> list[str]:
    return [str(it["metadata"]["source_id"]) for it in items]


# ------------------------------------------------------------------------------------------------ metrics
def paired_bootstrap(sa: list[float], sb: list[float], labels: list[bool], b: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict[str, Any]:
    """AUROC(b) - AUROC(a) on the SAME item resamples (percentile 95 % CI); single-class resamples skipped."""
    rng = random.Random(seed)
    n = len(labels)
    reps, skipped = [], 0
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        lab = [labels[i] for i in idx]
        x, y = RP.auroc([sa[i] for i in idx], lab), RP.auroc([sb[i] for i in idx], lab)
        if math.isnan(x) or math.isnan(y):
            skipped += 1
            continue
        reps.append(y - x)
    reps.sort()

    def q(p: float) -> float:
        if not reps:
            return float("nan")
        xx = p * (len(reps) - 1)
        f = math.floor(xx)
        c = min(f + 1, len(reps) - 1)
        return reps[f] + (reps[c] - reps[f]) * (xx - f)

    return {"point": RP.auroc(sb, labels) - RP.auroc(sa, labels), "lo": q(0.025), "hi": q(0.975), "B": b,
            "seed": seed, "skipped_single_class": skipped, "n": n}


def scores_of(items: list[dict], results: dict[str, dict]) -> tuple[list[float | None], list[bool]]:
    sc, lab = [], []
    for it in items:
        r = results.get(it["_evaluation"]["run_id"]) or {"status": "missing"}
        sc.append(float(r["response"]["answers"]["q"]["noul"]) if r.get("status") == "ok" else None)
        lab.append(bool(it["expected"]["q"]))
    return sc, lab


def subset_auroc(items, sc, lab, keep) -> dict[str, Any]:
    s = [x for it, x in zip(items, sc) if keep(it) and x is not None]
    y = [l for it, x, l in zip(items, sc, lab) if keep(it) and x is not None]
    return {"n": len(s), "n_pos": sum(y), "auroc": RP.auroc(s, y) if s else float("nan")}


def decide_arm(arm: str, rt1_ci: dict[str, Any], rt1_out_of_draw_point: float | None) -> str:
    """The pre-registered rule on one arm (``DECISION_RULE``)."""
    lo = rt1_ci.get("lo")
    if not (isinstance(lo, (int, float)) and math.isfinite(lo) and lo > NULL_AUROC):
        return "fail: RT1 lower bound <= 0.5 (no DEV stage)"
    if arm in INDRAW_GUARD_ARMS:
        if not (isinstance(rt1_out_of_draw_point, (int, float)) and math.isfinite(rt1_out_of_draw_point)
                and rt1_out_of_draw_point > NULL_AUROC):
            return "fail: in-draw driven (out-of-draw RT1 AUROC <= 0.5; no DEV stage)"
    return "pass: run the DEV-minus-proxy600 stage"


def arm_report(arm: str, items: list[dict], by_wording: dict[str, dict[str, dict]], indraw: set[str]) -> dict[str, Any]:
    """Every wording's metrics (``ragtruth_polarity.arm_metrics`` + in/out-of-draw), the paired dAUROC and the rule."""
    out: dict[str, Any] = {"wordings": {}}
    in_d = lambda it: str(it["metadata"]["source_id"]) in indraw  # noqa: E731
    sc = {}
    for w, res in by_wording.items():
        m = RP.arm_metrics(items, res)
        m.pop("decision", None)  # ragtruth_polarity's own rule is not this mode's rule
        per = m.pop("per_item")
        s, lab = scores_of(items, res)
        sc[w] = s
        m["in_draw"] = subset_auroc(items, s, lab, in_d)
        m["out_of_draw"] = subset_auroc(items, s, lab, lambda it: not in_d(it))
        m["per_item"] = per
        out["wordings"][w] = m
    lab = [bool(it["expected"]["q"]) for it in items]
    if all(w in sc for w in WORDINGS) and all(x is not None for w in WORDINGS for x in sc[w]):
        out["paired_dauroc_rt1_minus_original"] = paired_bootstrap(sc["original"], sc[CANDIDATE], lab)
    rt1 = out["wordings"].get(CANDIDATE)
    out["decision"] = (decide_arm(arm, rt1["auroc"], rt1["out_of_draw"]["auroc"]) if rt1 else "fail: RT1 not scored")
    out["n_in_draw"] = sum(1 for it in items if in_d(it))
    return out


# ------------------------------------------------------------------------------------------------ stage 2 rows
PROXY600 = "jobs/common/dev_proxy600.json"
PROXY600_SHA_PREFIX = "7d42c398"


def proxy600_ids() -> set[str]:
    d = json.loads((ROOT / PROXY600).read_text(encoding="utf-8"))
    ids = set(d["run_ids"])
    if len(ids) != 644 or RP.ids_sha256(ids) != d["ids_sha256"] or not d["ids_sha256"].startswith(PROXY600_SHA_PREFIX):
        raise ValueError("proxy600 identity")
    return ids


def dev_stage_rows(dev_rows: str | Path) -> list[dict]:
    """Stage 2 rows: Index DEV catalog-59 (RAGTruth) rows NOT in proxy600, in file order."""
    pids = proxy600_ids()
    return [r for r in read_jsonl(dev_rows)
            if r["_evaluation"]["catalog_id"] == RP.CATALOG_ID and r["_evaluation"]["run_id"] not in pids]


def dev_ids_sha256(rows: list[dict]) -> str:
    return RP.ids_sha256(r["_evaluation"]["run_id"] for r in rows)


def report_dev(a) -> dict[str, Any]:
    """Stage 2 in-job report: per arm x wording on the DEV-minus-proxy600 RAGTruth rows (AUROC, mean p, status), the
    paired dAUROC, and the reproduction of the original-wording rows against the landed full-DEV rows (max |dp|, if
    shipped). The Index delta itself is computed at landing (analysis/ragtruth_rephrase_land.py) with the frozen
    scorer on full DEV minus proxy600 with only these rows swapped."""
    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    rows = dev_stage_rows(a.dev_rows)
    items = [{**r, "expected": {"q": bool(r["expected"]["q"])}, "metadata": {**r["metadata"],
              "track": r["metadata"].get("track", "?")}} for r in rows]
    out: dict[str, Any] = {"schema": SCHEMA + ".dev", "prereg": PREREG, "split": "Index DEV catalog 59 minus proxy600",
                           "n_rows": len(rows), "ids_sha256": dev_ids_sha256(rows), "arms": {}}
    for arm in params["arms"]:
        by_w = {}
        for w in WORDINGS:
            rp = d / f"{arm}-dev-{w}.results.jsonl"
            if rp.is_file():
                by_w[w] = {r["run_id"]: r for r in read_jsonl(rp)}
        if not by_w:
            continue
        m = arm_report(arm, items, by_w, set())
        for w, mw in m["wordings"].items():
            mw.pop("per_item")
            mw["results_sha256"] = _sha(d / f"{arm}-dev-{w}.results.jsonl")
        m.pop("decision", None)
        ref = (params.get("dev_stage") or {}).get("landed", {}).get(arm)
        lp = Path(a.landed_dir) / f"{arm}.ragtruth.results.jsonl" if a.landed_dir else None
        if ref and lp and lp.is_file() and "original" in by_w:
            if _sha(lp) != ref["ragtruth_rows_sha256"]:
                raise ValueError(f"{arm}: shipped landed rows sha256 != pin")
            land = {r["run_id"]: r for r in read_jsonl(lp)}
            dp = [abs(float(by_w["original"][k]["response"]["answers"]["q"]["noul"])
                      - float(land[k]["response"]["answers"]["q"]["noul"]))
                  for k in by_w["original"] if by_w["original"][k].get("status") == "ok" and land.get(k, {}).get("status") == "ok"]
            m["reproduction_vs_landed"] = {"n": len(dp), "max_abs_dp": max(dp) if dp else None,
                                           "mean_abs_dp": (sum(dp) / len(dp)) if dp else None}
        out["arms"][arm] = m
    out["verdict"] = "measured"
    return out


# ------------------------------------------------------------------------------------------------ GPU: score
def score(a) -> int:
    from decide.ragtruth_rephrase import rephrased, wording_sha256
    from jobs.m2_first_cell.fulldev_eval import checkpoint_declarer, limited, load_child, readout_spec
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train import m2_gates as G
    from train.cohort import CohortLeak

    wordings = [w for w in a.wordings.split(",") if w]
    if not wordings or any(w not in WORDINGS for w in wordings):
        print(f"--wordings {a.wordings!r}: from {WORDINGS}", file=sys.stderr)
        return 5
    if a.dev_rows:  # stage 2: the Index DEV catalog-59 rows outside proxy600, pinned by params dev_stage.ids_sha256
        items = dev_stage_rows(a.dev_rows)
        pin = json.loads(Path(a.params).read_text(encoding="utf-8"))["dev_stage"]
        if dev_ids_sha256(items) != pin["ids_sha256"] or len(items) != pin["n"]:
            print("DEV stage rows: ids sha256 / n != params dev_stage pin", file=sys.stderr)
            return 10
    else:
        items = read_jsonl(a.items)
        want = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        if RP.ids_sha256(items_response_ids(items)) != want["ids_sha256"] or len(items) != want["n"]:
            print("items: ids sha256 / n != the selection pin", file=sys.stderr)
            return 10
    tag = f"{a.arm}-dev" if a.dev_rows else a.arm
    try:
        cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
        ro = readout_spec(RP.READOUT, cfg)
        decider, _, ident = load_child(a, temperature=False, readout=(ro["template"], ro["layout"]))
        ident = {**ident, "readout": ro, "arm_name": a.arm,
                 "wordings": {w: wording_sha256(w) for w in wordings}}
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"identity-{tag}.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    rdir = Path(a.resume_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, dict]] = {}
    for w in wordings:
        done[w] = {}
        p = rdir / f"{tag}-{w}.partial.jsonl"
        if p.is_file():
            for x in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(x)
                    done[w][r["run_id"]] = r
                except (ValueError, KeyError):
                    break  # a torn tail: rescored
    todo = {w: [it for it in items if it["_evaluation"]["run_id"] not in done[w]] for w in wordings}
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(it) for w in wordings for it in todo[w]])  # ONE total: the setup cap ends here
    base = G.beat_wrap(prog.wrap(limited(decider.decide, a.max_input_tokens)))
    declare = checkpoint_declarer()
    timing = {}
    for w in wordings:
        decide = rephrased(base, w)
        t0 = time.perf_counter()
        p = rdir / f"{tag}-{w}.partial.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:  # rewritten from the parsed rows: drops a torn tail
            for r in done[w].values():
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
            for it in todo[w]:
                res = G.index_predict(decide, [it])
                f.write(json.dumps(res[it["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
                f.flush()
                declare()
        declare(force=True)
        got = {json.loads(x)["run_id"]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
        rids = [it["_evaluation"]["run_id"] for it in items]
        if set(got) != set(rids):
            print(f"{tag}/{w}: {len(set(rids) - set(got))} items have no result", file=sys.stderr)
            return 9
        with open(out / f"{tag}-{w}.results.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for rid in rids:
                f.write(json.dumps(got[rid], separators=(",", ":")) + "\n")
        timing[w] = {"seconds_this_attempt": time.perf_counter() - t0, "rows_this_attempt": len(todo[w]),
                     "sum_row_wall_s": sum(float(got[i].get("total_wall_ms") or 0) for i in rids) / 1000}
    (out / f"timing-{tag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def load_indraw(items: list[dict], manifest: str | Path | None) -> dict[str, Any]:
    """The committed D19 in-draw ids; if the draw manifest is given, it must rebuild them exactly."""
    rec = json.loads((ROOT / INDRAW_FILE).read_text(encoding="utf-8"))
    ids = set(rec["ids"])
    if RP.ids_sha256(ids) != rec["ids_sha256"] or rec["items_ids_sha256"] != RP.ids_sha256(items_response_ids(items)):
        raise ValueError("in-draw file: ids sha256 / items identity mismatch")
    if manifest is not None:
        if _sha(manifest) != rec["manifest_sha256"]:
            raise ValueError("draw manifest sha256 != the in-draw file's manifest_sha256")
        if set(indraw_ids(manifest, set(items_response_ids(items)))) != ids:
            raise ValueError("draw manifest does not rebuild the committed in-draw ids")
    return {"ids": ids, "n": len(ids), "ids_sha256": rec["ids_sha256"], "manifest_sha256": rec["manifest_sha256"],
            "rebuilt_from_manifest": manifest is not None}


def report(a) -> dict[str, Any]:
    from jobs.common.costs import worst_case_usd

    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    items = read_jsonl(a.items)
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    ind = load_indraw(items, a.manifest)
    d13o = Path(a.d13_original) if a.d13_original else None
    if d13o is not None and _sha(d13o) != params["reference"]["d13_original_results_sha256"]:
        raise ValueError("D13 original-wording results sha256 != params reference pin")
    out: dict[str, Any] = {"schema": SCHEMA, "prereg": PREREG, "split": "RAGTruth TRAIN (never Index DEV)",
                           "readout": RP.READOUT, "candidate": CANDIDATE, "n_items": len(items),
                           "ids_sha256": sel["ids_sha256"], "items_sha256": _sha(a.items), "decision_rule": DECISION_RULE,
                           "d19_in_draw": {k: v for k, v in ind.items() if k != "ids"},
                           "arms_expected": list(params["arms"]), "arms": {}}
    from decide.ragtruth_rephrase import wording_sha256

    out["wording_sha256"] = {w: wording_sha256(w) for w in WORDINGS}
    for arm in params["arms"]:
        by_w: dict[str, dict[str, dict]] = {}
        for w in WORDINGS:
            rp = d / f"{arm}-{w}.results.jsonl"
            if arm == "d13" and w == "original" and d13o is not None:
                rp = d13o
            if rp.is_file():
                by_w[w] = {r["run_id"]: r for r in read_jsonl(rp)}
        if not by_w:
            continue
        m = arm_report(arm, items, by_w, ind["ids"] if arm in INDRAW_GUARD_ARMS else set())
        for w, mw in m["wordings"].items():
            (d / f"{arm}-{w}.per_item.jsonl").write_text("".join(json.dumps(x) + "\n" for x in mw.pop("per_item")),
                                                        encoding="utf-8")
            ext = arm == "d13" and w == "original" and d13o is not None
            src = d13o if ext else d / f"{arm}-{w}.results.jsonl"
            mw["results_sha256"] = _sha(src)
            mw["source"] = "job-3zmj2 art-srefc (pinned)" if ext else "this job"
        idf = d / f"identity-{arm}.json"
        if idf.is_file():
            idc = json.loads(idf.read_text(encoding="utf-8"))
            m["checkpoint_loaded"] = idc.get("checkpoint")
            m["readout_loaded"] = idc.get("readout")
            m["wordings_loaded"] = idc.get("wordings")
        tm = d / f"timing-{arm}.json"
        if tm.is_file():
            m["timing"] = json.loads(tm.read_text(encoding="utf-8"))
        out["arms"][arm] = m
    usd = params["batch_usd_per_min"]
    out["cost"] = {"elapsed_min": a.elapsed_min, "runtime_usd_batch": round(a.elapsed_min * usd, 2),
                   "worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"]}
    out["checkpoints"] = {arm: {kk: vv for kk, vv in (params.get(params["arms"][arm]["block"]) or {}).items()
                                if not kk.startswith("_")} for arm in params["arms"]}
    out["verdict"] = "measured"
    return out


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: both arms, both wordings, every item scored, finite AUROC CIs, the
    loaded checkpoints equal the pins."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        ok = d.get("schema") == SCHEMA and d.get("verdict") == "measured" and set(d["arms"]) == set(d["arms_expected"])
        pins = {arm: d["checkpoints"][arm]["ckpt_sha256"] for arm in d["arms"]}
        for arm, m in d["arms"].items():
            ok = ok and set(m["wordings"]) == set(WORDINGS) and isinstance(m.get("decision"), str)
            for mw in m["wordings"].values():
                ok = ok and sum(mw["status"].values()) == d["n_items"] and mw["n_scored"] == d["n_items"]
                ok = ok and all(fin(mw["auroc"][f]) for f in ("point", "lo", "hi"))
            ok = ok and all(fin(m["paired_dauroc_rt1_minus_original"][f]) for f in ("point", "lo", "hi"))
            ok = ok and (m.get("checkpoint_loaded") or {}).get("sha256") == pins[arm]
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def valid_result_dev(d: Any) -> bool:
    """Stage 2 validator: every expected arm with both wordings on every row, finite AUROCs."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        ok = d.get("schema") == SCHEMA + ".dev" and d.get("verdict") == "measured" and bool(d["arms"])
        for m in d["arms"].values():
            ok = ok and set(m["wordings"]) == set(WORDINGS)
            for mw in m["wordings"].values():
                ok = ok and mw["n_scored"] == d["n_rows"] and fin(mw["auroc"]["point"])
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def valid_any(d: Any) -> bool:
    return valid_result_dev(d) if str(d.get("schema", "")).endswith(".dev") else valid_result(d)


# ------------------------------------------------------------------------------------------------ CPU: pricing
def estimate(params: dict) -> dict[str, Any]:
    """Setup preflight (exit 15 BEFORE any checkpoint is fetched). Per arm: load = ``evals_fixed_minutes[arm]``
    (MEASURED: D13 6.75 min, job-3zmj2 rate-score_d13.json scoring_started_min; D19 21.75 min, job-fi5xf
    rate-screen.json) and scoring = n wordings x ``score_minutes_per_wording`` (MEASURED: D13 196.5 s for the 300 items,
    job-3zmj2 timing-d13.json). Pass = on a box SLOW_BOX (1.25x) slower, per arm: load <= the rate guard's setup cap,
    setup cap + scoring <= 95 % of the phase cap, load + scoring <= 95 % of the phase cap; the whole expected schedule
    x 1.25 fits max_duration minus reserve."""
    caps, scap = params["phase_caps_minutes"], params["score_setup_cap_minutes"]
    per_w = float(params["score_minutes_per_wording"])
    checks, minutes = {}, {}
    for arm in params["arms"]:
        n = len(params["arms"][arm]["wordings"])
        load, sc = float(params["evals_fixed_minutes"][arm]), n * per_w
        minutes[f"score_{arm}"] = round(load + sc, 2)
        checks[f"load_{arm}_slow_le_setup_cap"] = load * SLOW_BOX <= scap[arm]
        checks[f"setup_cap_{arm}_leaves_scoring_room"] = scap[arm] + sc * SLOW_BOX <= 0.95 * caps[f"score_{arm}"]
        checks[f"score_{arm}_slow_le_95pct_cap"] = (load + sc) * SLOW_BOX <= 0.95 * caps[f"score_{arm}"]
    sched = sum(v for k, v in params["expected_minutes"].items() if k != "build")
    checks["schedule_slow_fits"] = sched * SLOW_BOX <= params["max_duration_minutes"] - params["reserve_minutes"]
    return {"minutes": minutes, "slow_box_factor": SLOW_BOX, "expected_run_minutes": sched, "checks": checks,
            "ok": all(checks.values())}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.ragtruth_rephrase")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("indraw", help="(local, CPU) the items inside a bschema draw manifest -> the committed file")
    i.add_argument("--manifest", required=True)
    i.add_argument("--items", required=True)
    i.add_argument("--out", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARM_NAMES, required=True)
    s.add_argument("--wordings", required=True, help="comma list from " + ",".join(WORDINGS))
    s.add_argument("--items", default=None)
    s.add_argument("--selection", default=None)
    s.add_argument("--dev-rows", default=None, help="stage 2: the DEV rows file (catalog 59 minus proxy600 is used)")
    s.add_argument("--params", default=None, help="stage 2: job params (dev_stage pin)")
    s.add_argument("--cell", required=True)
    s.add_argument("--ckpt-dir", required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records", required=True)
    s.add_argument("--train-dir", required=True)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness", required=True)
    s.add_argument("--max-input-tokens", type=int, default=65_536)
    s.add_argument("--resume-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    p = sub.add_parser("report")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--d13-original", default=None)
    p.add_argument("--manifest", default=None, help="D19's draw manifest: rebuilds the committed in-draw ids")
    p.add_argument("--elapsed-min", type=float, required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("report-dev")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--params", required=True)
    q.add_argument("--dev-rows", required=True)
    q.add_argument("--landed-dir", default=None)
    q.add_argument("--out", required=True)
    r = sub.add_parser("dev-pin", help="(local, CPU) n + ids sha256 of the stage-2 rows")
    r.add_argument("--dev-rows", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "dev-pin":
        rows = dev_stage_rows(a.dev_rows)
        print(json.dumps({"n": len(rows), "ids_sha256": dev_ids_sha256(rows)}))
        return 0
    if a.cmd == "report-dev":
        res = report_dev(a)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        ok = res["arms"] and all(set(m["wordings"]) == set(WORDINGS) for m in res["arms"].values())
        print(json.dumps({arm: {w: m["wordings"][w]["auroc"] for w in m["wordings"]} for arm, m in res["arms"].items()},
                         default=float))
        return 0 if ok else 9
    if a.cmd == "indraw":
        items = read_jsonl(a.items)
        rids = items_response_ids(items)
        ids = indraw_ids(a.manifest, set(rids))
        by = {}
        for it in items:
            if str(it["metadata"]["source_id"]) in set(ids):
                k = f"{it['metadata']['track']}|{'hallucinated' if it['expected']['q'] else 'supported'}"
                by[k] = by.get(k, 0) + 1
        rec = {"_comment": "RAGTruth TRAIN polarity items (ragtruth_polarity) whose response id is a RAGTruth row of "
                           "D19's training draw (bschema-draw-v3b manifest), i.e. D19 trained on them with the ORIGINAL "
                           "wording and labels (B-PREREG-RAGTRUTH-REPHRASE contamination guard)",
               "manifest": "m2/build/bschema-draw-v3b/bschema-draw-v3b.manifest.jsonl.gz",
               "manifest_sha256": _sha(a.manifest), "items_sha256": _sha(a.items),
               "items_ids_sha256": RP.ids_sha256(rids), "n_items": len(rids), "n": len(ids),
               "ids_sha256": RP.ids_sha256(ids), "by_track_label": dict(sorted(by.items())), "ids": ids}
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "ids"}, indent=1))
        return 0
    if a.cmd == "estimate":
        est = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")))
        print(json.dumps(est, indent=1))
        return 0 if est["ok"] else 15
    if a.cmd == "score":
        return score(a)
    res = report(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({arm: {"decision": m["decision"],
                            **{w: m["wordings"][w]["auroc"] for w in m["wordings"]},
                            "dauroc": m.get("paired_dauroc_rt1_minus_original")}
                      for arm, m in res["arms"].items()}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
