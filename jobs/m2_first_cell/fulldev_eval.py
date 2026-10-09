"""Mode ``fulldev_eval``: score a SAVED m2 cell checkpoint on the FULL Index DEV slice (12,104 rows, edition 0.2.1),
the same rows file, edition and scorer as the Clef comparators (job-zfexa art-p93d4, job-t5yg2 art-7pyk2).

    # GPU (in the job): the child arm, then (optional) the stock-readout parent arm on the same rows
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval score --arm child --cell <out>/cell.json --ckpt-dir <ck>/cell \
        --model-dir <snap> --records <records> --train-dir <150k> --guard-readiness <readiness.json> \
        --rows <data>/slices/dev-rows.jsonl.gz --resume-dir <ck>/fulldev --out-dir <out>/fulldev [--progress <p.jsonl>]
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval score --arm parent --cell-config <cfg> --model-dir <snap> ...
    # CPU: pricing from the committed row lengths (the runner's setup preflight; exit 15 past a cap)
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval estimate --params jobs/m2_first_cell/job_params_fulldev_eval.json
    # CPU: the report (frozen scorer via `python -I -m harness score|ece|compare`; JSONs; finalize validator)
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval report --out-dir <out>/fulldev --params <job_params> \
        --elapsed-min N --out <out>/fulldev_eval_result.json

Child arm = EXACTLY the in-job Index arm of ``train.m2_gates`` (stage ``index``): the checkpoint restored through
``Trainer.load`` (config hash + lineage), the cohort temperature fit, the SERVED (tempered) decider, and
``train.m2_gates.index_predict(..., untempered_from_raw=True)``: ``response`` = the untempered answers (T = 1, the arm
D1's 35.12 subset number is computed on), ``served_response`` = the tempered answers (served ECE).

Over-length rows (the comparators' 51, ``jobs/clef_baseline/overlength_rows.json``). Clef's shipped max_length is
16,384; the comparators re-read those 51 rows at max_length 32,768 with NO truncation (all fit) and substituted them
(``decide_32k_substituted``). RWKV7 is attention-free and has no positional context limit, so the candidate reads
EVERY row in full too: the adapter's 16,384 default is raised through the request's ``ssdm.limits.max_input_tokens``
(exactly how ``baselines.clef_index_engine`` raises it for the Clef 32k arm) to ``max_input_tokens`` (job params,
default 65,536, above the longest full-DEV row: ``jobs/m2_first_cell/fulldev_lengths_rwkv7.json``). A $0 plan-only
pass over all 12,104 rows (tokenisation, no forward) runs first and is exported (``plan.json``): any row the decider
would still refuse is listed with its code and scores as the kit scores it (not ok = no credit). The report gives BOTH
  * ``full``: all 12,104 rows (the headline; pairs with the comparators' 32k-substituted files), and
  * ``excl_overlength51``: the same results with the 51 comparator over-length rows left out of the slice (12,053
    rows), so the gap cannot come from rows only one side was built to read.
No truncation anywhere; nothing is excluded after scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OVERLENGTH = ROOT / "jobs" / "clef_baseline" / "overlength_rows.json"
SCHEMA = "ssdm.m2.fulldev_eval.v1"
EDITION = "0.2.1"
N_DEV = 12_104
N_OVERLENGTH = 51
DEFAULT_MAX_INPUT_TOKENS = 65_536
WARMUP_ROWS = 10  # the Clef latency row (jobs/clef_baseline/summarize.latency) drops the first 10 ok rows
ARMS = ("child", "parent")


# ------------------------------------------------------------------------------------------------ shared helpers
def overlength_ids() -> set[str]:
    return {r["run_id"] for r in json.loads(OVERLENGTH.read_text(encoding="utf-8"))["rows"]}


SMOKE = ROOT / "jobs" / "m2_first_cell" / "fulldev_smoke_rows.json"
SMOKE_SEED = "fulldev-smoke-v1"
ROW_MODES = ("full", "smoke", "proxy600")  # proxy600: jobs/common/dev_proxy600.json (jobs.common.dev_proxy, pinned)


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def selected_ids(rows_mode: str) -> set[str] | None:
    """None = every DEV row; ``smoke`` = the committed, pre-declared smoke slice (sha256 re-checked); ``proxy600`` =
    the pre-declared ~600-row screening proxy of the 2,008 m1 subset (``jobs.common.dev_proxy``: ids sha pinned there,
    DEV manifest + base subset re-checked)."""
    if rows_mode == "full":
        return None
    if rows_mode == "proxy600":
        from jobs.common.dev_proxy import proxy_ids

        return proxy_ids("proxy600")
    if rows_mode != "smoke":
        raise ValueError(f"rows mode {rows_mode!r}: one of {ROW_MODES}")
    d = json.loads(SMOKE.read_text(encoding="utf-8"))
    if ids_sha256(d["run_ids"]) != d["ids_sha256"] or len(set(d["run_ids"])) != d["n"]:
        raise ValueError("fulldev_smoke_rows.json run_ids do not match their ids_sha256 / n")
    return set(d["run_ids"])


def build_smoke(manifest: list[dict], lengths: dict[str, int], n_target: int = 250, n_long: int = 12,
                seed: str = SMOKE_SEED) -> dict[str, Any]:
    """The smoke slice, from the DEV manifest + committed RWKV7 row lengths only (no model output). Whole kit groups.
      1. long rows: of the rows over 16,384 RWKV7 tokens, sorted by length (desc, then run_id), the longest plus
         ``n_long - 1`` more at evenly spaced ranks (so it spans 16k-47.5k and mixes the 51 with RWKV7-only ones);
      2. families: rounds over the tracks (sorted); each round adds, per track, its next whole group in
         sha256(f"{seed}:{catalog_id}:{group_id}") order whose rows are all <= 16,384 tokens, until ``n_target``.
    """
    by_group: dict[tuple, list[dict]] = {}
    for e in manifest:
        by_group.setdefault((e["catalog_id"], str(e["group_id"])), []).append(e)
    gkey = {e["run_id"]: (e["catalog_id"], str(e["group_id"])) for e in manifest}
    long = sorted((rid for rid, n in lengths.items() if n > 16_384), key=lambda r: (-lengths[r], r))
    picks = sorted({0} | {round(i * (len(long) - 1) / (n_long - 1)) for i in range(1, n_long)})
    chosen: list[tuple] = []
    for i in picks:
        g = gkey[long[i]]
        if g not in chosen:
            chosen.append(g)

    def n_rows():
        return sum(len(by_group[g]) for g in chosen)

    queues: dict[str, list[tuple]] = {}
    for g, es in by_group.items():
        if g in chosen or any(lengths[e["run_id"]] > 16_384 for e in es):
            continue
        queues.setdefault(es[0]["track"], []).append(g)
    for t in queues:
        queues[t].sort(key=lambda g: hashlib.sha256(f"{seed}:{g[0]}:{g[1]}".encode()).hexdigest())
    while n_rows() < n_target and any(queues.values()):
        for t in sorted(queues):
            if queues[t] and n_rows() < n_target:
                chosen.append(queues[t].pop(0))
    ids = sorted(e["run_id"] for g in chosen for e in by_group[g])
    ol = overlength_ids()
    return {"_comment": "Pre-declared smoke slice of the Index DEV for mode fulldev_smoke (jobs.m2_first_cell."
                        "fulldev_eval smoke-build; DEV manifest + RWKV7 row lengths only, never model output). Proves "
                        "the fulldev_eval pipeline end to end; NOT a capability estimate.",
            "seed": seed, "n_target": n_target, "n_long_target": n_long, "n": len(ids), "ids_sha256": ids_sha256(ids),
            "n_tracks": len({by_group[g][0]["track"] for g in chosen}),
            "n_over_16384": sum(lengths[i] > 16_384 for i in ids), "n_overlength51": len(ol & set(ids)),
            "max_tokens": max(lengths[i] for i in ids), "total_tokens": sum(lengths[i] for i in ids),
            "run_ids": ids}


def limited(decide, max_input_tokens: int):
    """``decide`` with the adapter's input cap raised to ``max_input_tokens`` through the request envelope (the
    Clef 32k arm's mechanism, ``baselines.clef_index_engine``); every other default limit is unchanged."""
    def call(req: dict) -> dict:
        return decide({**req, "ssdm": {"limits": {"max_input_tokens": int(max_input_tokens)}}})
    return call


def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else None


def latency(results: list[dict]) -> dict[str, Any]:
    """Per-row decide() wall time, the Clef latency row's definition (ok rows in scoring order, first 10 dropped as
    warm-up, median + nearest-rank p95)."""
    ok = [r for r in results if r.get("status") == "ok"][WARMUP_ROWS:]
    ms = [float(r["total_wall_ms"]) for r in ok if r.get("total_wall_ms") is not None]
    return {"n": len(ms), "median_ms": statistics.median(ms) if ms else None, "p95_ms": pct(ms, 0.95),
            "mean_ms": statistics.fmean(ms) if ms else None, "p99_ms": pct(ms, 0.99), "warmup_rows_dropped": WARMUP_ROWS,
            "scope": "decide() wall time per full-DEV row (all questions of the row), H100 80GB, batch 1, one "
                     "request at a time (same definition as the Clef latency row)"}


def plan_rows(decider, rows: list[dict], max_input_tokens: int) -> dict[str, Any]:
    """$0 plan-only pass (tokenisation, no forward) through the real decider with the raised cap: which rows reach
    ``score()``, and the longest model input per row (``backend.plan``)."""
    from decide import Decider
    from jobs.common.dev_full import PlanOnly

    class _Recording(PlanOnly):  # one tokenisation per row: the adapter's own plan() call, recorded
        def plan(self, request):
            self.last = None
            self.last = super().plan(request)
            return self.last

    po = _Recording(decider.backend)
    d = limited(Decider(po, enable_v0_1_deltas=getattr(decider, "enable_v0_1_deltas", False),
                        backend_max_input_tokens=getattr(decider, "backend_max_input_tokens", None)).decide,
                max_input_tokens)
    ol = overlength_ids()
    refused, lengths = [], {}
    for r in rows:
        rid = r["_evaluation"]["run_id"]
        req = {"model": "ssdm-m2", "state": r["state"], "questions": r["questions"]}
        po.last = None
        po.reached = False
        resp = d(req)
        lengths[rid] = getattr(po.last, "max_call_input_tokens", None)
        if not po.reached:
            ss = resp.get("ssdm") or {}
            refused.append({"run_id": rid, "status": ss.get("status"), "code": (ss.get("error") or {}).get("code"),
                            "message": (ss.get("error") or {}).get("message"), "overlength51": rid in ol})
    known = [v for v in lengths.values() if v is not None]
    return {"n_rows": len(rows), "max_input_tokens": max_input_tokens, "n_refused": len(refused), "refused": refused,
            "n_over_16384": sum(1 for v in known if v > 16_384),
            "n_over_32768": sum(1 for v in known if v > 32_768), "max_row_tokens": max(known) if known else None,
            "overlength51_max_tokens": max((lengths.get(i) or 0) for i in ol) if ol else None,
            "revision": decider.revision().get("revision_id"),
            "rule": "plan-only through decide.Decider with ssdm.limits.max_input_tokens raised; refused rows are "
                    "scored as the kit scores them (not ok = no credit), never dropped"}


# ------------------------------------------------------------------------------------------------ GPU: arms
def load_child(a, temperature: bool = True, readout: tuple[str, str] | None = None):
    """The served + arm deciders of a saved checkpoint, built exactly as ``train.m2_gates.main`` builds them
    (cohort, guard, training items for the lineage check, ``Trainer.load``, the cohort temperature fit).
    ``temperature=False`` (the proxy screen, ``jobs.m2_first_cell.screen``): no cohort role records, no temperature
    fit; the decider is the UNTEMPERED arm decider ``train.m2_gates`` scores the Index stage with (identity
    calibration, like the parent arm), returned with tempered=False. ``readout`` = (template, layout) to score under
    instead of the cell's own (screen ``screen_readout``; same model object, another prompt, as readout_ab)."""
    if readout is not None and temperature:
        raise ValueError("load_child: a readout override is for the untempered screen arm only (temperature=False)")
    import torch

    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_pilot.guard_ready import load_ready_guard
    from train import m2_gates as G
    from train.calibration import item_content
    from train.cohort import Cohort
    from train.run_cell import (attach_order_views, attach_teacher, cell_items, load_cell_data, load_parent_backbone,
                                parent_encoder, train_config)
    from train.trainer import Trainer, with_execution

    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cfg = cell["config"]
    cohort = Cohort(expected_sha256=cfg["cohort_sha256"])
    guard = load_ready_guard(a.guard_readiness)
    parent = cfg.get("parent", "rwkv7")
    enc = parent_encoder(parent, a.model_dir)
    spec = G.unit_spec(cfg)
    pools = G.pool_dirs_arg(a.pool_dir)
    train_recs, recs, tr_ids, _, _ = load_cell_data(cfg, a.records, a.train_dir, cohort)
    cohort.assert_training_ids(tr_ids)
    role_recs = ({role: G.role_records(cohort, a.records, role, recs, pools) for role in ("dev", "calibration", "retention")}
                 if temperature else None)
    t_max = cell["ladder"]["T_max"]
    train_items = [x for x in cell_items(train_recs, tr_ids, cfg, enc, guard, max(cfg["t_ladder"]))
                   if len(x.row.ids) <= t_max]
    attach_order_views(cfg, train_items, train_recs, enc, guard, t_max)
    attach_teacher(cfg, train_items, a.teacher_dir, tr_ids)
    bb = load_parent_backbone(parent, a.model_dir, cfg, None, device=a.device)
    cuda = torch.cuda.is_available() and str(a.device).startswith("cuda")
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    model.to(bb.device)
    tr = Trainer(model, with_execution(train_config(cfg, cfg["max_steps"], a.ckpt_dir, cuda), cfg), train_items,
                 guard=guard)
    ckpt = Path(a.ckpt) if a.ckpt else Trainer.latest(a.ckpt_dir)
    if ckpt is None:
        raise ValueError("no checkpoint")
    tr.load(ckpt)  # config hash + lineage (backbone digest/provenance, tokenizer, layout, training items)
    if not temperature:
        tpl, lay = readout or (cfg["template"], cfg["layout"])
        backend = DecisionModelBackend(model, enc, template=tpl, layout=lay,
                                       checkpoint={"path": ckpt.name, "step": tr.step,
                                                   "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
        decider = Decider(backend)
        return decider, False, {"arm": "child", "arm_decider": decider.revision(), "checkpoint": backend.checkpoint,
                                "lineage": tr.lineage, "cell_config": cfg,
                                "readout": {"template": tpl, "layout": lay,
                                            "native": (tpl, lay) == (cfg["template"], cfg["layout"])},
                                "index_arm": "untempered arm decider (identity calibration), as train.m2_gates "
                                             "scores the Index stage; no temperature fit"}
    items_by_role = {r: G.decision_items(role_recs[r], enc, spec, guard) for r in role_recs}
    ft = G.fit_child_temperature(model, items_by_role["calibration"],
                                 {"eval": items_by_role["dev"], "retention": items_by_role["retention"],
                                  "train": train_items})
    backend = DecisionModelBackend(model, enc, template=cfg["template"], layout=cfg["layout"],
                                   checkpoint={"path": ckpt.name, "step": tr.step,
                                               "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
    served = ft.to_serving()
    decider = Decider(backend, calibration=served)
    for role, name in (("dev", "eval"), ("retention", "retention")):
        served.verify_disjoint(name, item_content(items_by_role[role]))
    ident = {"arm": "child", "served": decider.revision(), "checkpoint": backend.checkpoint, "lineage": tr.lineage,
             "temperature": {**ft.identity(), "proof_sha256": ft.proof_sha256}, "cell_config": cfg,
             "index_arm": "served decider; response = untempered answers rebuilt from raw_probabilities (T = 1), "
                          "served_response = tempered (train.m2_gates.index_predict untempered_from_raw)"}
    return decider, True, ident


def load_parent(a):
    """The stock readout (identity calibration) as the parent arm is served (``score_parent.parent_decider``), with
    its trained-context cap lifted to the same request cap as the child (both arms read every row in full)."""
    from decide import Decider
    from jobs.m2_parent_arm.score_parent import parent_decider

    cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
    dec, ident = parent_decider(cfg.get("parent", "rwkv7"), a.model_dir, None, cfg, device=a.device)
    lifted = Decider(dec.backend, calibration=dec.calibration,
                     enable_v0_1_deltas=getattr(dec, "enable_v0_1_deltas", False))
    return lifted, False, {"arm": "parent", **ident, "revision": lifted.revision()["revision_id"],
                           "trained_ctx_cap": dec.backend_max_input_tokens, "cap_lifted_to_request_limit": True}


def _read_partial(path: Path) -> dict[str, dict]:
    from harness.jsonl import drop_truncated_tail, iter_jsonl_strict

    if not path.is_file():
        return {}
    drop_truncated_tail(path)
    return {r["run_id"]: r for _, r in iter_jsonl_strict(path)}


def score(a) -> int:
    from harness.suite import read_rows
    from train import m2_gates as G

    rows = read_rows(a.rows)
    ids = [r["_evaluation"]["run_id"] for r in rows]
    if len(ids) != N_DEV or len(set(ids)) != N_DEV:
        print(f"rows: {len(set(ids))} unique of {len(ids)}, want the full DEV slice ({N_DEV})", file=sys.stderr)
        return 10
    try:
        keep = selected_ids(a.rows_select)
    except ValueError as e:
        print(f"rows: {e}", file=sys.stderr)
        return 10
    if keep is not None:
        if not keep <= set(ids):
            print(f"rows: {len(keep - set(ids))} {a.rows_select} ids are not DEV rows", file=sys.stderr)
            return 10
        rows = [r for r in rows if r["_evaluation"]["run_id"] in keep]
        ids = [r["_evaluation"]["run_id"] for r in rows]
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train.cohort import CohortLeak

    try:
        decider, tempered, ident = (load_child if a.arm == "child" else load_parent)(a)
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = plan_rows(decider, rows, a.max_input_tokens)
    (out / f"plan-{a.arm}.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    (out / f"identity-{a.arm}.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ("n_rows", "n_refused", "n_over_16384", "max_row_tokens")}), flush=True)
    resume = Path(a.resume_dir) / f"{a.arm}.partial.jsonl"
    resume.parent.mkdir(parents=True, exist_ok=True)
    done = _read_partial(resume)
    todo = [r for r in rows if r["_evaluation"]["run_id"] not in done]
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(r) for r in todo])
    decide = G.beat_wrap(prog.wrap(limited(decider.decide, a.max_input_tokens)))
    t0 = time.perf_counter()
    with open(resume, "a", encoding="utf-8", newline="\n") as f:
        for r in todo:
            res = G.index_predict(decide, [r], untempered_from_raw=tempered)
            f.write(json.dumps(res[r["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
            f.flush()
    secs = time.perf_counter() - t0
    done = _read_partial(resume)
    if set(done) != set(ids):
        print(f"{len(set(ids) - set(done))} rows have no result", file=sys.stderr)
        return 9
    for name, key in (("results", "response"), ("results_served", "served_response")):
        if name == "results_served" and not tempered:
            continue
        with open(out / f"{a.arm}.{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for rid in ids:  # DEV row order
                r = dict(done[rid])
                served = r.pop("served_response", None)
                if key == "served_response" and r.get("status") == "ok":
                    r["response"] = served
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    (out / f"timing-{a.arm}.json").write_text(json.dumps(
        {"seconds_this_attempt": secs, "rows_this_attempt": len(todo), "rows_resumed": len(ids) - len(todo),
         "sum_row_wall_s": sum(float(done[i].get("total_wall_ms") or 0) for i in ids) / 1000}, indent=1),
        encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def _harness(cmd: str, *args: str) -> dict:
    """The frozen scorer as the comparators ran it (``python -X utf8 -I -m harness <cmd> ... --slice dev --edition
    0.2.1``; DEV rows = ``$SSDM_DATA/slices/dev-rows.jsonl.gz``). Not ``-I`` here: the rwkv7 image does not pip-install
    the harness, so it runs from the code root (same files, same pins)."""
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "harness", cmd, *args], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"harness {cmd} rc={p.returncode}: {p.stderr[-800:]}")
    return json.loads(p.stdout)


def _subset(results_path: Path, keep: set[str] | None, drop: set[str]):
    from harness import cli, score as S

    rows = [r for r in cli.load_slice("dev", EDITION) if (keep is None or r["_evaluation"]["run_id"] in keep)
            and r["_evaluation"]["run_id"] not in drop]
    want = set(cli._ids(rows))
    res = {k: v for k, v in cli.load_results(str(results_path), None, False).items() if k in want}
    S.require_results(rows, res, False)
    return rows, res


def subset_score(results_path: Path, keep: set[str] | None = None, drop: set[str] = frozenset(), B: int = 2000,
                 seed: int = 0) -> dict:
    """Index over DEV rows restricted to ``keep`` minus ``drop`` with the same functions ``harness score`` calls
    (decompose + case-cluster bootstrap); every kept row must have a result."""
    from harness import bootstrap, score as S

    rows, res = _subset(results_path, keep, drop)
    rep = bootstrap.replicate_skills([S.decompose(rows, res, edition=EDITION)], B=B, seed=seed)[0]
    return {"edition": EDITION, "rows": len(rows), "slice_index": bootstrap.summarize_one(rep, edition=EDITION)}


def subset_ece(results_path: Path, keep: set[str] | None = None) -> dict:
    from harness import calibration

    rows, res = _subset(results_path, keep, set())
    return calibration.report(calibration.fields(rows, res))


def subset_compare(cand: Path, base: Path, keep: set[str] | None = None, B: int = 2000, seed: int = 0) -> dict:
    from harness import bootstrap, score as S

    rows, rc = _subset(cand, keep, set())
    _, rb = _subset(base, keep, set())
    c = bootstrap.compare(S.decompose(rows, rc, edition=EDITION), S.decompose(rows, rb, edition=EDITION), B=B,
                          seed=seed)
    return {"rows": len(rows), "candidate": c["candidate"], "baseline": c["baseline"], "diff": c["diff"]}


def excl_overlength_score(results_path: Path, B: int = 2000, seed: int = 0, keep: set[str] | None = None) -> dict:
    """Index over the (selected) DEV rows minus the 51 comparator over-length rows."""
    out = subset_score(results_path, keep, overlength_ids(), B, seed)
    out["excluded_run_ids"] = "jobs/clef_baseline/overlength_rows.json (51)"
    return out


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def report(a) -> dict:
    from jobs.common.costs import quoted_max_cost_usd, worst_case_usd

    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    arms = [x for x in ARMS if (d / f"{x}.results.jsonl").is_file()]
    if "child" not in arms:
        raise SystemExit("no child.results.jsonl")
    ed = ("--slice", "dev", "--edition", EDITION)
    mode = params.get("rows", "full")
    keep = selected_ids(mode)
    n = N_DEV if keep is None else len(keep)
    out: dict[str, Any] = {"schema": SCHEMA, "edition": EDITION, "rows_mode": mode, "rows": n, "arms": {},
                           "scorer": ("harness CLI (score / ece / compare --slice dev --edition 0.2.1)" if keep is None
                                      else f"{'SMOKE' if mode == 'smoke' else 'PROXY'} SUBSET ({mode}): harness.score."
                                           "decompose + harness.bootstrap + harness.calibration in process on the "
                                           "pre-declared rows (the CLI scores whole slices only); "
                                           + ("a pipeline proof, not a capability estimate" if mode == "smoke" else
                                              "a cell SCREEN (jobs/common/dev_proxy.py); winners get the full subset "
                                              "+ full DEV"))}
    for arm in arms:
        rp = d / f"{arm}.results.jsonl"
        res = [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if keep is None:
            sc = _harness("score", "--results", str(rp), *ed)
            ece = _harness("ece", "--results", str(rp), *ed)
        else:
            sc = {**subset_score(rp, keep), "official_index_missing_as_zero": None}
            ece = subset_ece(rp, keep)
        (d / f"{arm}.score.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
        (d / f"{arm}.ece.json").write_text(json.dumps(ece, indent=1), encoding="utf-8")
        ex = excl_overlength_score(rp, keep=keep)
        (d / f"{arm}.score_excl_overlength51.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
        ent = {"results_sha256": _sha(rp), "status": _count(r.get("status") for r in res),
               "full": sc["slice_index"], "official_index_missing_as_zero": sc["official_index_missing_as_zero"],
               "excl_overlength51": ex["slice_index"],
               "ece": {"board_protocol": (ece.get("board_protocol") or {}).get("ece"),
                       "overall_10bin": (ece.get("overall") or {}).get("ece_10bin")},
               "latency": latency(res)}
        sp = d / f"{arm}.results_served.jsonl"
        if sp.is_file():  # the tempered (served) answers: same argmax, served calibration
            se = _harness("ece", "--results", str(sp), *ed) if keep is None else subset_ece(sp, keep)
            (d / f"{arm}.ece_served.json").write_text(json.dumps(se, indent=1), encoding="utf-8")
            ent["ece_served"] = {"board_protocol": (se.get("board_protocol") or {}).get("ece"),
                                 "overall_10bin": (se.get("overall") or {}).get("ece_10bin")}
        pl = d / f"plan-{arm}.json"
        if pl.is_file():
            p = json.loads(pl.read_text(encoding="utf-8"))
            ent["plan"] = {k: p.get(k) for k in ("max_input_tokens", "n_refused", "n_over_16384", "n_over_32768",
                                                 "max_row_tokens", "overlength51_max_tokens")}
        tm = d / f"timing-{arm}.json"
        if tm.is_file():
            ent["timing"] = json.loads(tm.read_text(encoding="utf-8"))
        out["arms"][arm] = ent
    if "parent" in arms:  # paired child - parent on identical rows (frozen bootstrap.compare)
        cmp = (_harness("compare", "--candidate", str(d / "child.results.jsonl"),
                        "--baseline", str(d / "parent.results.jsonl"), *ed) if keep is None
               else subset_compare(d / "child.results.jsonl", d / "parent.results.jsonl", keep))
        (d / "compare_child_vs_parent.json").write_text(json.dumps(cmp, indent=1), encoding="utf-8")
        out["child_minus_parent"] = cmp.get("diff", cmp)
    usd = params["batch_usd_per_min"]
    out["cost"] = {"elapsed_min": a.elapsed_min, "runtime_usd_batch": round(a.elapsed_min * usd, 2),
                   "usd_per_min_batch": usd, "runtime_usd_quoted": round(a.elapsed_min * 0.063, 2),
                   "usd_per_min_quoted": 0.063, "build_minutes_not_included": True,
                   "worst_case_usd": worst_case_usd(params), "quoted_max_cost_usd": quoted_max_cost_usd(params),
                   "hard_cap_usd": params["hard_cap_usd"]}
    out["checkpoint"] = {k: v for k, v in (params.get("checkpoint") or {}).items() if not k.startswith("_")}
    idf = d / "identity-child.json"
    if idf.is_file():
        out["checkpoint"]["loaded"] = json.loads(idf.read_text(encoding="utf-8")).get("checkpoint")
    out["overlength_treatment"] = ("every row read in full (no truncation; adapter cap raised to "
                                   f"{params.get('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)} via ssdm.limits, the "
                                   "Clef 32k arm's mechanism); 'full' = every selected row (12,104 in mode full; "
                                   "pairs with the comparators' 32k-substituted files), 'excl_overlength51' = the same "
                                   "minus the 51 comparator over-length rows")
    out["verdict"] = "measured"
    return out


def _count(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a complete full-DEV measurement of the child arm."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        c = d["arms"]["child"]
        sel = selected_ids(d.get("rows_mode", "full"))
        n = N_DEV if sel is None else len(sel)
        return (d.get("schema") == SCHEMA and d.get("verdict") == "measured" and d.get("rows") == n
                and sum(c["status"].values()) == n
                and all(fin(c[k]["index"][f]) for k in ("full", "excl_overlength51") for f in ("point", "lo", "hi"))
                and fin(c["latency"]["median_ms"]) and fin(c["latency"]["p95_ms"])
                and bool(d["checkpoint"].get("ckpt_sha256"))
                and (d["checkpoint"].get("loaded") or {}).get("sha256") == d["checkpoint"]["ckpt_sha256"])
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ CPU: pricing
LENGTHS = ROOT / "jobs" / "m2_first_cell" / "fulldev_lengths_rwkv7.json"


def estimate(params: dict, lengths: dict | None = None) -> dict[str, Any]:
    """Projected minutes per scoring phase = committed full-DEV row tokens x the MEASURED per-token decide() wall of
    the arm (``measured_rates_reference``: D1 job-c7pq2 units-child-index.json / the job-uhxb5 parent arm, ok rows of
    the 2,008 subset) + the phase's fixed minutes (model load, lineage, temperature fit, plan pass). Pass = every
    phase <= 95 % of its cap (the rate guards' rule). Also checks the readout template the params name against the
    mode's cell config."""
    lengths = lengths or json.loads(LENGTHS.read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / params["cell_config"]).read_text(encoding="utf-8"))
    if params.get("readout_template") != cfg["template"]:
        raise ValueError(f"readout_template {params.get('readout_template')!r} != cell config template {cfg['template']!r}")
    if params["max_input_tokens"] <= lengths["max_tokens"]:
        raise ValueError(f"max_input_tokens {params['max_input_tokens']} <= the longest DEV row {lengths['max_tokens']}")
    m = params["measured_rates_reference"]
    keep = selected_ids(params.get("rows", "full"))
    toks = lengths["total_tokens"] if keep is None else sum(lengths["lengths"][i] for i in keep)
    phases = {"fulldev": toks / 1000 * m["child_s_per_1k_tokens"] / 60 + params["evals_fixed_minutes"]["child"]}
    if params.get("fulldev_parent_arm"):
        phases["parent_full"] = toks / 1000 * m["parent_s_per_1k_tokens"] / 60 + params["evals_fixed_minutes"]["parent"]
    caps = params["phase_caps_minutes"]
    out = {"rows": params.get("rows", "full"), "tokens": toks, "template": cfg["template"], "lengths_template": lengths["template"],
           "minutes": {k: round(v, 1) for k, v in phases.items()},
           "caps": {k: caps[k] for k in phases}, "ok": all(v <= 0.95 * caps[k] for k, v in phases.items())}
    out["usd_batch_scoring"] = round(sum(phases.values()) * params["batch_usd_per_min"], 2)
    return out


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.fulldev_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--cell", help="child: the checkpoint's cell.json (config + ladder)")
    s.add_argument("--cell-config", help="parent: the mode's cell config (template, WKV)")
    s.add_argument("--ckpt-dir")
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records")
    s.add_argument("--train-dir")
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness")
    s.add_argument("--rows", required=True, help="the full DEV rows file (data/slices/dev-rows.jsonl.gz)")
    s.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    s.add_argument("--rows-select", choices=ROW_MODES, default="full", help="full DEV, the committed smoke slice, or the proxy600 screen")
    s.add_argument("--resume-dir", required=True, help="per-row append log (checkpoint dir): a restart resumes")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    b = sub.add_parser("smoke-build", help="(re)build fulldev_smoke_rows.json from the DEV manifest + row lengths")
    b.add_argument("--out", default=str(SMOKE))
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--params", required=True)
    r.add_argument("--elapsed-min", type=float, required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        need = ("cell", "ckpt_dir", "records", "train_dir", "guard_readiness") if a.arm == "child" else ("cell_config",)
        miss = [k for k in need if not getattr(a, k)]
        if miss:
            ap.error(f"--arm {a.arm} needs {miss}")
        return score(a)
    if a.cmd == "smoke-build":
        from jobs.common.dev_subset import read_manifest

        d = build_smoke(read_manifest(), json.loads(LENGTHS.read_text(encoding="utf-8"))["lengths"])
        Path(a.out).write_text(json.dumps(d, indent=0) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in d.items() if k not in ("run_ids", "_comment")}))
        return 0
    if a.cmd == "estimate":
        try:
            est = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")))
        except (ValueError, KeyError) as e:
            print(f"estimate: {e}", file=sys.stderr)
            return 5
        print(json.dumps(est, indent=1))
        return 0 if est["ok"] else 15
    res = report(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    c = res["arms"]["child"]
    print(json.dumps({"full": c["full"]["index"], "excl_overlength51": c["excl_overlength51"]["index"],
                      "latency_ms": [c["latency"]["median_ms"], c["latency"]["p95_ms"]], "cost": res["cost"],
                      "valid": valid_result(res)}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
