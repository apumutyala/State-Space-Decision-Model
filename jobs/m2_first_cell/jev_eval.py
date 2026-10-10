"""Mode ``d19_jevbench``: a SAVED cell checkpoint (D19, job-fi5xf) and its stock parent scored on the 231 public JevBench
v1.2 items (pinned b6b8fff7, EVAL-ONLY) through jevkit, PAIRED: the same items, the same declared price, the same
bootstrap resamples, so every trained-minus-stock delta (I, C, Capability, Composite A/B/C; Speed and Cost as points)
is paired.

    # GPU (in the job), one arm per process (first_cell_job.sh, eval_only "jevbench")
    python -X utf8 -m jobs.m2_first_cell.jev_eval score --arm child --cell <out>/cell.json --ckpt-dir <ck>/cell \
        --ckpt <best> --model-dir <snap> --records <records> --train-dir <train set> --guard-readiness <readiness.json> \
        --out-dir <out>/jevbench
    python -X utf8 -m jobs.m2_first_cell.jev_eval score --arm parent --cell-config <cfg> --model-dir <snap> --out-dir ...
    # CPU: jevkit report per arm + the paired deltas -> jev_eval_result.json (finalize validator: valid_result)
    python -X utf8 -m jobs.m2_first_cell.jev_eval report --params <job_params> --out-dir <out>/jevbench --out <result>

Child arm = EXACTLY the D19 screen's child path (``jobs.m2_first_cell.fulldev_eval.load_child(temperature=False)``,
as ``jobs.m2_first_cell.screen`` scores): ``Trainer.load`` of the BEST checkpoint (config hash + lineage: backbone
digest, tokenizer, layout, training items), the cell's NATIVE readout (template jrt-rwkv-g1-v3w, layout g1v2: v3w
exactly as trained, train == serve), the joint_record head scoring a whole record per forward (a JevBench item is one
record with one question), the UNTEMPERED arm decider (identity calibration; nothing fitted on these items).
Parent arm = ``fulldev_eval.load_parent``: the stock readout of the same weights in the same format (G1V3Backend
words = stock v3w, the screen's parent arm), identity calibration.

Per item, one request at a time (JevBench timing), two warm-up requests not recorded. Usage per record:
input_tokens = the decision's longest single model input (Plan.max_call_input_tokens), output_tokens = 1 (the board's
price basis), compute_input_tokens = every position the backend ran. Cost uses the job params' declared ``price``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("child", "parent")
WARMUP = 2
SCHEMA = "ssdm.m2.jev_eval.v1"


class BilledDecider:
    """A decide.Decider whose ok responses carry the scan's usage (as jobs/jev_host_scan): input_tokens = longest
    single model input, output_tokens = 1, compute_input_tokens = every position the backend ran."""

    def __init__(self, decider):
        self.d = decider

    def revision(self):
        return self.d.revision()

    def decide(self, request):
        resp = self.d.decide(request)
        ss = resp.get("ssdm") or {}
        if ss.get("status") == "ok":
            ic = (ss.get("diagnostics") or {}).get("input_counts") or {}
            resp["usage"] = {"input_tokens": ic.get("max_call_input_tokens"), "output_tokens": 1,
                             "compute_input_tokens": ic.get("total_input_tokens")}
        return resp


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            pass  # a cut last line is re-scored
    return out


def run_items(decider, items, out_dir: Path, warmup: int = WARMUP) -> dict[str, Any]:
    """Serial, one request at a time; preds appended + flushed per item (a cap kill keeps every finished item)."""
    from jevkit.systemone import SystemOneAdapter

    ad = SystemOneAdapter(BilledDecider(decider), kind="decider", model_name="ssdm-m2")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / "preds.jsonl"
    done = _done(preds)
    todo = [it for it in items if it.id not in done]
    t0 = time.time()
    for it in items[:warmup]:
        ad.answer_item(it)
    st: dict[str, int] = {}
    with open(preds, "a", encoding="utf-8", newline="\n") as fh:
        for it in todo:
            rec = ad.answer_item(it)
            rec["tier"] = it.tier
            st[rec["status"]] = st.get(rec["status"], 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    return {"items": len(todo), "resumed_from": len(done), "status": st, "wall_s": round(time.time() - t0, 1)}


def score(a) -> int:
    from jevkit.items import load_public

    from jobs.m2_first_cell.fulldev_eval import load_child, load_parent

    items = load_public()
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out_dir) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train.cohort import CohortLeak

    t0 = time.time()
    try:
        decider, _, ident = load_child(a, temperature=False) if a.arm == "child" else load_parent(a)
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    ident = {**ident, "load_s": round(time.time() - t0, 1), "revision": decider.revision().get("revision_id")}
    (out / "identity.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    res = run_items(decider, items, out)
    if str(a.device).startswith("cuda"):
        import torch

        res["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "run.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return 0 if len(_done(out / "preds.jsonl")) >= len(items) else 9


# ------------------------------------------------------------------------------------------------ CPU: report
def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def paired(items, preds: dict[str, list[dict]], reports: dict[str, dict], b: int, seed: int) -> dict[str, Any]:
    """child - parent per axis on the SAME stratified resamples (jevkit's strata; B / seed of the board). Speed and
    Cost are per-system measurements held fixed per resample (as jevkit.scoring.bootstrap), so their deltas are points."""
    import numpy as np

    from jevkit import scoring as SC
    from jevkit.systemone import preds_to_answers

    tabs = {k: SC.build_table(items, preds_to_answers(v)) for k, v in preds.items()}
    if tabs["child"].ids != tabs["parent"].ids:
        raise ValueError("paired: the arms are not scored on the same item list")
    sk = {k: (reports[k]["axes"]["speed"], reports[k]["axes"]["cost"]) for k in tabs}
    rng = np.random.default_rng(seed)
    st = SC.strata(tabs["child"])
    keys = ("I", "C", "capability", "A", "B", "C_opt")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(b):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        v = {}
        for nm, t in tabs.items():
            r = SC.aggregate(t, idx, detail=False)
            i, c = r["intelligence"]["I"], r["calibration"]["score"]
            s, k = sk[nm]
            v[nm] = {"I": i, "C": c, "capability": SC.capability(i, c), "A": SC.composite(i, c, s, k, "A"),
                     "B": SC.composite(i, c, s, k, "B"), "C_opt": SC.composite(i, c, s, k, "C")}
        for k in keys:
            if v["child"][k] is not None and v["parent"][k] is not None:
                draws[k].append(v["child"][k] - v["parent"][k])

    def pt(k, rk):
        x, y = rk(reports["child"]), rk(reports["parent"])
        return None if x is None or y is None else x - y

    getters = {"I": lambda r: r["axes"]["intelligence"], "C": lambda r: r["axes"]["calibration"],
               "capability": lambda r: r["capability"], "A": lambda r: r["scores"]["A"],
               "B": lambda r: r["scores"]["B"], "C_opt": lambda r: r["scores"]["C"]}
    out = {}
    for k in keys:
        d = draws[k]
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else None
        out[{"C_opt": "composite_C", "A": "composite_A", "B": "composite_B", "C": "calibration",
             "I": "intelligence"}.get(k, k)] = {
            "point": pt(k, getters[k]), "ci95": ci, "p_child_better": float(np.mean([x > 0 for x in d])) if d else None}
    for ax in ("speed", "cost"):
        x, y = reports["child"]["axes"][ax], reports["parent"]["axes"][ax]
        out[ax] = {"point": None if x is None or y is None else x - y, "ci95": None,
                   "note": "a measured per-system value, held fixed per resample (jevkit.scoring.bootstrap)"}
    out["_resamples"] = {"B": b, "seed": seed, "strata": "jevkit.scoring.strata (side x type x tier)"}
    return out


def report(a) -> dict[str, Any]:
    from jevkit.items import load_public

    p = json.loads(Path(a.params).read_text(encoding="utf-8"))
    items = load_public()
    base = Path(a.out_dir)
    price = base / "price.json"
    price.write_text(json.dumps({"price": p["price"]}, indent=1), encoding="utf-8")
    reps, preds, arms = {}, {}, {}
    for arm in ARMS:
        d = base / arm
        preds[arm] = _read(d / "preds.jsonl")
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "jevkit", "score", "--public", "--pred",
                            str(d / "preds.jsonl"), "--price", str(price), "--out", str(d / "report.json")],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"jevkit score {arm}: {r.stderr[-1500:]}")
        reps[arm] = json.loads((d / "report.json").read_text(encoding="utf-8"))
        ident = json.loads((d / "identity.json").read_text(encoding="utf-8")) if (d / "identity.json").exists() else {}
        rj = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").exists() else {}
        rep = reps[arm]
        arms[arm] = {"n_preds": len(preds[arm]), "complete": len(preds[arm]) >= len(items),
                     "ok": sum(1 for x in preds[arm] if x.get("status") == "ok"),
                     "axes": rep["axes"], "composite": rep["scores"], "capability": rep["capability"],
                     "ci95": rep.get("ci95"), "speed": rep.get("speed"),
                     "cost": {k: (rep.get("cost") or {}).get(k) for k in ("usd_per_1000", "mean_input_tokens",
                                                                          "mean_output_tokens", "kind", "price")},
                     "intelligence": {k: rep["intelligence"].get(k) for k in ("I_open", "basis", "per_type_split")
                                      if k in rep["intelligence"]},
                     "calibration": rep.get("calibration"), "validity": rep.get("validity"),
                     "identity": {k: ident.get(k) for k in ("arm", "revision", "checkpoint", "readout", "index_arm",
                                                            "load_s", "parent", "calibration")},
                     "run": rj}
    pair = paired(items, preds, reps, int(p.get("bootstrap_B", 1000)), int(p.get("bootstrap_seed", 16)))
    leader = {"key": "h2o-lightning-4b", "composite_A": 72.5232271521435, "I": 60.03839906196252,
              "C": 89.95399367673842, "S": 92.61084474720658, "K": 60.29882443070202, "board": "JevBench v1.6.1"}
    return {"schema": SCHEMA, "verdict": None, "measurement_only": True, "mode": p.get("mode"), "items": len(items),
            "checkpoint": p.get("checkpoint", {}).get("artifact"), "job": p.get("checkpoint", {}).get("job"),
            "basis": "open-only, the 231 public JevBench v1.2 items (b6b8fff7); not the v1.6.1 open set P300 and no "
                     "sealed side (jevkit/README.md)",
            "price": p["price"], "arms": arms, "child_minus_parent": pair, "board_leader": leader,
            "elapsed_min": a.elapsed_min}


def valid_result(d: Any) -> bool:
    try:
        return (d.get("schema") == SCHEMA and all(d["arms"][x]["complete"] and d["arms"][x]["composite"]["A"]
                                                  is not None for x in ARMS)
                and d["child_minus_parent"]["composite_A"]["ci95"] is not None)
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.jev_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--cell", default=None)
    s.add_argument("--cell-config", default=None)
    s.add_argument("--ckpt-dir", default=None)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records", default=None)
    s.add_argument("--train-dir", default=None)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness", default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--limit", type=int, default=0, help="CPU tests only: the first N items")
    r = sub.add_parser("report")
    r.add_argument("--params", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--elapsed-min", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.arm == "child" and not all([a.cell, a.ckpt_dir, a.records, a.train_dir, a.guard_readiness]):
            ap.error("child needs --cell --ckpt-dir --records --train-dir --guard-readiness")
        if a.arm == "parent" and not a.cell_config:
            ap.error("parent needs --cell-config")
        return score(a)
    try:
        d = report(a)
    except Exception as e:  # noqa: BLE001
        print(f"report: {type(e).__name__}: {e}", file=sys.stderr)
        return 9
    Path(a.out).write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: d["child_minus_parent"][k] for k in ("composite_A", "intelligence", "calibration")}))
    return 0 if valid_result(d) else 9


if __name__ == "__main__":
    sys.exit(main())
