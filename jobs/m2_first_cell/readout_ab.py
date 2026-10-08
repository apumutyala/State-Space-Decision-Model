"""Mode ``readout_ab``: score a SAVED layout-g1v2 cell checkpoint (D3) on the 2,008-row Index DEV subset under TWO OR
MORE readouts in ONE process (one model load, one lineage check, one temperature fit), each written and scored as its
own results file. The first readout is the cell's own training template (the matched control): ``v1`` for D1
(layout ordered), ``v2f`` for D3; then ``v3w`` (the R3 winner) and optionally more. ONE generic readout over every row
per results file (never a per-family pick, which would be tuning on DEV). Each readout also reports decode throughput
(plan tokens / decide wall s over ok rows) and per-row latency (the Clef latency row definition).

    # GPU (in the job): the child arm over every readout, then (optional) the stock parent under the same readouts
    python -X utf8 -m jobs.m2_first_cell.readout_ab score --arm child --readouts v2f,v3p --cell <out>/cell.json \
        --ckpt-dir <ck>/cell --ckpt <ck>/cell/ckpt-stepN.pt --model-dir <snap> --records <records> --train-dir <150k> \
        --guard-readiness <readiness.json> --rows <work>/index-rows.jsonl.gz --resume-dir <ck>/readout_ab \
        --out-dir <out>/readout_ab [--progress <p.jsonl>]
    python -X utf8 -m jobs.m2_first_cell.readout_ab score --arm parent --readouts v2f,v3p --cell-config <cfg> ...
    # CPU: pricing (the runner's setup preflight; exit 15 past a cap)
    python -X utf8 -m jobs.m2_first_cell.readout_ab estimate --params jobs/m2_first_cell/job_params_readout_ab_d3.json
    # CPU: the report (jobs.common.dev_subset score|ece|compare = the frozen harness on the subset rows)
    python -X utf8 -m jobs.m2_first_cell.readout_ab report --out-dir <out>/readout_ab --params <p> --elapsed-min N \
        --out <out>/readout_ab_result.json

Readouts (``heads.layout`` layout ``g1v2``; each is the stock R3 family's own renderer, asserted byte-equal per row):
  v2f   jrt-rwkv-g1-v2f   the D3 training template (stock family rwkv7-hf-g1-v2f; parent 29.10 on job-fgu2c)
  v3w   jrt-rwkv-g1-v3w   v3 + word option ids scored by the first answer token of " <id>" (when non-opaque and unique)
                          (stock family rwkv7-hf-g1-v3w; parent 31.07 on job-fgu2c, the R3 winner)
  v3p   jrt-rwkv-g1-v3p   v3 codes + code key (k >= 9) + seeded per-question presentation permutation
                          (stock family rwkv7-hf-g1-v3p; parent 30.60 on job-fgu2c)
  v2fp  jrt-rwkv-g1-v2fp  v2f + the same permutation only (eval-only, no stock family; no parent arm)

Validity of a non-training readout on the trained head. The cache-fork head (``heads.heads.FinalStateFork``) scores
option k from the backbone's hidden state after ``prompt + " <code_k>"`` through ONE shared MLP plus a symmetric
(mean-over-options) summary: it has no per-code or per-position parameters, so any readout whose answer slot is
"ANSWER:" at the last prompt position followed by one code token per option is well defined. Every readout here keeps
that slot. Under a permuted readout, canonical option j is forked with the code PRESENTED for it
(``decide.prompt_g1v3.question_codes``; the stock backend uses the same function), and the head's logits come back
keyed by option id in canonical order, so ``wire_answer`` maps them exactly. What changes for the LoRA is the prompt
distribution (v3p: the code-key block for k >= 9, codes past Z, codes no longer aligned with sorted ids), which is the
measured effect, not an ill-definedness. v3w: the head never reads the LM's next-token logits, so a word id is NOT
scored as a vocabulary logit the LoRA never learned; option j is forked with the first answer token of its own id (one
token, unique among the options by ``word_ids_ok``) and the SAME MLP scores that hidden state. Well defined, but the
furthest from training: the head was only ever fit on states after " <CODE>" letter tokens, so on word-id questions
(mostly tools: ToolRet / BFCL / BRIGHT) it reads a state class it never saw; v3w on the trained cell measures whether
the head transfers, and its stock-parent gain (logit of the id's own first token) need not carry over. Ladders and budgets are the stock backend's (code-key labels 64 / 24 / none
under 16,383 tokens; the adapter's default 16,384-token input cap, as the in-job index stage and job-fgu2c).

Child arm = the in-job Index arm of ``train.m2_gates`` stage ``index`` (``jobs.m2_first_cell.fulldev_eval.load_child``:
``Trainer.load`` config hash + lineage, the cohort temperature fit, served decider), with the readout's backend swapped
in over the SAME model object: ``response`` = untempered answers (T = 1, rebuilt from ``raw_probabilities``),
``served_response`` = the served answers under the ONE temperature fitted on the training template's calibration items
(reported as ``ece_served``; for a non-training readout it is "the deployed calibrator applied to another prompt", not
a per-readout fit). The v2f child results must reproduce the D3 job's own in-job child index arm (same rows, same
code path) up to GPU non-determinism: the report carries that cross-check when ``reference_results_sha256.d3_child_index``
is pinned and the file is shipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ssdm.m2.readout_ab.v1"
N_SUBSET = 2_008
ARMS = ("child", "parent")
#: readout name -> layout-g1v2 template id (heads.layout G1_TRAIN_TEMPLATES / G1_EVAL_TEMPLATES)
READOUTS = {"v1": "jrt-rwkv-g1-v1", "v2f": "jrt-rwkv-g1-v2f", "v3w": "jrt-rwkv-g1-v3w", "v3p": "jrt-rwkv-g1-v3p",
            "v2fp": "jrt-rwkv-g1-v2fp"}
#: readout name -> heads.layout layout (v1 = D1's own "ordered" JRT layout; the R3 readouts are layout g1v2)
READOUT_LAYOUT = {"v1": "ordered", "v2f": "g1v2", "v3w": "g1v2", "v3p": "g1v2", "v2fp": "g1v2"}
#: readout name -> stock decide.backends.rwkv_g1_v3.G1V3Backend mode (parent arm); v2fp has no stock family
PARENT_MODES = {"v2f": "v2f", "v3w": "words", "v3p": "permute"}


def parse_readouts(spec: str | list[str]) -> list[str]:
    out = [v.strip() for v in (spec.split(",") if isinstance(spec, str) else spec) if v.strip()]
    bad = [v for v in out if v not in READOUTS]
    if len(out) < 2 or bad or len(set(out)) != len(out):
        raise ValueError(f"readouts must be >= 2 distinct names from {sorted(READOUTS)}; got {spec!r}")
    return out


def is_native(cfg: dict, r: str) -> bool:
    """Readout ``r`` is the cell's own training template AND layout (the control)."""
    return READOUTS[r] == cfg.get("template") and READOUT_LAYOUT[r] == cfg.get("layout")


def check_readouts(cfg: dict, readouts: list[str], allow_unmatched_control: bool = False) -> dict[str, Any]:
    """The first readout must be the cell's own training template + layout (the matched control).
    ``allow_unmatched_control`` (plumbing dry runs only) records the mismatch instead."""
    matched = is_native(cfg, readouts[0])
    if not matched and not allow_unmatched_control:
        raise ValueError(f"readout_ab needs the cell's training template as the first readout "
                         f"({readouts[0]} = {READOUTS[readouts[0]]} / {READOUT_LAYOUT[readouts[0]]}); cell: layout "
                         f"{cfg.get('layout')!r} template {cfg.get('template')!r}")
    return {"control": readouts[0], "control_template": READOUTS[readouts[0]], "cell_template": cfg.get("template"),
            "cell_layout": cfg.get("layout"), "matched_control": matched,
            "note": None if matched else "PLUMBING ONLY: the readouts are not this cell's training template"}


def _read_partial(path: Path) -> dict[str, dict]:
    from harness.jsonl import drop_truncated_tail, iter_jsonl_strict

    if not path.is_file():
        return {}
    drop_truncated_tail(path)
    return {r["run_id"]: r for _, r in iter_jsonl_strict(path)}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ GPU: deciders
def child_deciders(a, readouts: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """{readout: served Decider} over ONE loaded checkpoint (``fulldev_eval.load_child``), plus the identity block."""
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from jobs.m2_first_cell.fulldev_eval import load_child

    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
    ctl = check_readouts(cfg, readouts, a.allow_unmatched_control)
    base, _, ident = load_child(a)  # served decider on the cell's own template (Trainer.load + temperature fit)
    b0 = base.backend
    out = {}
    for r in readouts:
        if is_native(cfg, r):
            out[r] = base
        else:  # same model object, same encoder, same checkpoint record, same served calibration; another prompt
            be = DecisionModelBackend(b0.model, b0.encode, template=READOUTS[r], layout=READOUT_LAYOUT[r],
                                      checkpoint=b0.checkpoint, shared_prefix=b0.shared_prefix)
            out[r] = Decider(be, calibration=base.calibration)
    ident = {**ident, "readout_control": ctl,
             "readouts": {r: {"template": READOUTS[r], "layout": READOUT_LAYOUT[r], "served": d.revision()}
                          for r, d in out.items()}}
    return out, ident


def parent_deciders(a, readouts: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """{readout: identity-calibration Decider} of the STOCK readout families over ONE loaded plugin (same WKV as the
    child, ``score_parent.parent_decider``'s loader)."""
    from decide import Decider
    from decide.backends.hf_readout import load_rwkv7_hf
    from decide.backends.rwkv_g1_v3 import G1V3Backend

    import third_party.rwkv7_hf as V

    miss = [r for r in readouts if r not in PARENT_MODES]
    if miss:
        raise ValueError(f"no stock parent readout for {miss} (stock families: {sorted(PARENT_MODES)})")
    cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
    plug = load_rwkv7_hf(a.model_dir, device=a.device, wkv=cfg.get("rwkv7_wkv"))
    out = {r: Decider(G1V3Backend(plug, mode=PARENT_MODES[r])) for r in readouts}
    ident = {"arm": "parent", "parent": "rwkv7", "repo": V.REPO, "revision": V.REVISION,
             "wkv": str(plug.kernel_stack.get("wkv_implementation")),
             "readouts": {r: {"family_mode": PARENT_MODES[r], "revision": d.revision()["revision_id"]}
                          for r, d in out.items()}}
    return out, ident


# ------------------------------------------------------------------------------------------------ GPU: scoring
def plan_pass(decider, rows: list[dict], model: str = "ssdm-freeze") -> dict[str, dict[str, Any]]:
    """``jobs.common.dev_full.classify`` (same PlanOnly decider, same excludable statuses) that also records each
    row's planned input tokens (``Plan.total_input_tokens``, the throughput numerator)."""
    from decide import Decider
    from jobs.common.dev_full import EXCLUDABLE_STATUSES, FrozenMismatch, PlanOnly

    class _Recording(PlanOnly):
        def plan(self, request):
            self.last = super().plan(request)
            return self.last

    po = _Recording(decider.backend)
    d = Decider(po, enable_v0_1_deltas=getattr(decider, "enable_v0_1_deltas", False),
                backend_max_input_tokens=getattr(decider, "backend_max_input_tokens", None))
    out = {}
    for r in rows:
        po.reached, po.last = False, None
        resp = d.decide({"model": model, "state": r["state"], "questions": r["questions"]})
        ss = resp.get("ssdm") or {}
        err = ss.get("error") or {}
        rid = r["_evaluation"]["run_id"]
        toks = getattr(po.last, "total_input_tokens", None)
        if po.reached:
            out[rid] = {"supported": True, "total_input_tokens": toks}
        elif ss.get("status") not in EXCLUDABLE_STATUSES:
            raise FrozenMismatch(f"row {rid}: pre-forward outcome {ss.get('status')}/{err.get('code')} is not an "
                                 f"excludable structural refusal (fix the bug)")
        else:
            out[rid] = {"supported": False, "status": ss.get("status"), "code": err.get("code"),
                        "message": err.get("message"), "total_input_tokens": toks}
    return out


def throughput(results: list[dict], tokens: dict[str, Any]) -> dict[str, Any]:
    """Per-readout speed: per-row decide() latency (``fulldev_eval.latency``: ok rows, first 10 dropped, median /
    nearest-rank p95) and decode throughput = planned input tokens / decide wall seconds over the same ok rows."""
    from jobs.m2_first_cell.fulldev_eval import WARMUP_ROWS, latency

    ok = [r for r in results if r.get("status") == "ok" and r.get("total_wall_ms") is not None][WARMUP_ROWS:]
    tok = [tokens.get(r["run_id"]) for r in ok]
    secs = sum(float(r["total_wall_ms"]) for r in ok) / 1000
    known = sum(t for t in tok if isinstance(t, (int, float)))
    return {"latency": latency(results), "rows": len(ok), "tokens": known, "wall_s": round(secs, 3),
            "tokens_per_s": round(known / secs, 1) if secs > 0 and None not in tok else None,
            "rows_per_min": round(len(ok) / secs * 60, 2) if secs > 0 else None,
            "scope": "ok rows after the 10 warm-up rows; tokens = the readout's own plan (all question prompts)"}


def score(a) -> int:
    from harness.suite import read_rows
    from jobs.common.dev_full import FrozenMismatch
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train import m2_gates as G
    from train.cohort import CohortLeak

    try:
        readouts = parse_readouts(a.readouts)
    except ValueError as e:
        print(f"readouts: {e}", file=sys.stderr)
        return 5
    rows = read_rows(a.rows)
    ids = [r["_evaluation"]["run_id"] for r in rows]
    if len(ids) != N_SUBSET or len(set(ids)) != N_SUBSET:
        print(f"rows: {len(set(ids))} unique of {len(ids)}, want the DEV subset ({N_SUBSET})", file=sys.stderr)
        return 10
    try:
        deciders, ident = (child_deciders if a.arm == "child" else parent_deciders)(a, readouts)
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    tempered = a.arm == "child"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"identity-{a.arm}.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    resume = Path(a.resume_dir)
    resume.mkdir(parents=True, exist_ok=True)
    todo: dict[str, list[dict]] = {}
    for r in readouts:  # $0 plan-only pass per readout (tokenisation, no forward): which rows each readout refuses
        try:
            cls = plan_pass(deciders[r], rows)
        except FrozenMismatch as e:
            print(f"plan {r}: {e}", file=sys.stderr)
            return 10
        refused = sorted(i for i, c in cls.items() if not c["supported"])
        (out / f"plan-{a.arm}-{r}.json").write_text(json.dumps(
            {"readout": r, "n_rows": len(cls), "n_refused": len(refused),
             "refused": [{"run_id": i, **cls[i]} for i in refused],
             "tokens": {i: c.get("total_input_tokens") for i, c in cls.items()},
             "rule": "decide.Decider default limits (16,384 input tokens, as the in-job index stage and job-fgu2c); "
                     "refused rows score as the kit scores them (not ok = no credit)"}, indent=1), encoding="utf-8")
        print(json.dumps({"arm": a.arm, "readout": r, "n_refused": len(refused)}), flush=True)
        done = _read_partial(resume / f"{a.arm}.{r}.partial.jsonl")
        todo[r] = [x for x in rows if x["_evaluation"]["run_id"] not in done]
    prog = G.Progress(a.progress)  # one plan over every readout: the live rate guard sees the whole phase
    prog.plan([G.unit_weight(x) for r in readouts for x in todo[r]])
    timing = {}
    for r in readouts:
        decide = G.beat_wrap(prog.wrap(deciders[r].decide))
        part = resume / f"{a.arm}.{r}.partial.jsonl"
        t0 = time.perf_counter()
        with open(part, "a", encoding="utf-8", newline="\n") as f:
            for x in todo[r]:
                res = G.index_predict(decide, [x], untempered_from_raw=tempered)
                f.write(json.dumps(res[x["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
                f.flush()
        secs = time.perf_counter() - t0
        done = _read_partial(part)
        if set(done) != set(ids):
            print(f"{r}: {len(set(ids) - set(done))} rows have no result", file=sys.stderr)
            return 9
        for name, key in (("results", "response"), ("results_served", "served_response")):
            if name == "results_served" and not tempered:
                continue
            with open(out / f"{a.arm}.{r}.{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for rid in ids:  # subset row order
                    row = dict(done[rid])
                    served = row.pop("served_response", None)
                    if key == "served_response" and row.get("status") == "ok":
                        row["response"] = served
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
        timing[r] = {"seconds_this_attempt": secs, "rows_this_attempt": len(todo[r]),
                     "rows_resumed": N_SUBSET - len(todo[r]),
                     "sum_row_wall_s": sum(float(done[i].get("total_wall_ms") or 0) for i in ids) / 1000}
        (out / f"timing-{a.arm}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def _subset(cmd: str, *args: str) -> dict:
    """The frozen scorer on the subset rows (``python -X utf8 -m jobs.common.dev_subset <cmd>``: harness.score /
    bootstrap / calibration over ``$SSDM_DATA/slices/dev-rows.jsonl.gz`` restricted to dev_subset_m1.json)."""
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "jobs.common.dev_subset", cmd, *args], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"dev_subset {cmd} rc={p.returncode}: {p.stderr[-800:]}")
    return json.loads(p.stdout)


def _count(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def report(a) -> dict:
    from jobs.common.costs import quoted_max_cost_usd, worst_case_usd

    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    readouts = parse_readouts(params["readouts"])
    out: dict[str, Any] = {"schema": SCHEMA, "rows": N_SUBSET, "readouts": readouts, "control": readouts[0],
                           "arms": {}, "paired": {}}
    for arm in ARMS:
        have = [r for r in readouts if (d / f"{arm}.{r}.results.jsonl").is_file()]
        if not have:
            continue
        out["arms"][arm] = {}
        for r in have:
            rp = d / f"{arm}.{r}.results.jsonl"
            res = [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines() if x.strip()]
            sc = _subset("score", "--results", str(rp))
            ece = _subset("ece", "--results", str(rp))
            (d / f"{arm}.{r}.score.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
            (d / f"{arm}.{r}.ece.json").write_text(json.dumps(ece, indent=1), encoding="utf-8")
            ent = {"template": READOUTS[r], "layout": READOUT_LAYOUT[r], "results_sha256": _sha(rp), "status": _count(x.get("status") for x in res),
                   "index": sc["slice_index"], "ece": {"board_protocol": (ece.get("board_protocol") or {}).get("ece"),
                                                       "overall_10bin": (ece.get("overall") or {}).get("ece_10bin")}}
            sp = d / f"{arm}.{r}.results_served.jsonl"
            if sp.is_file():
                se = _subset("ece", "--results", str(sp))
                (d / f"{arm}.{r}.ece_served.json").write_text(json.dumps(se, indent=1), encoding="utf-8")
                ent["ece_served"] = {"board_protocol": (se.get("board_protocol") or {}).get("ece"),
                                     "overall_10bin": (se.get("overall") or {}).get("ece_10bin"),
                                     "calibrator": "the one temperature fitted on the training template"}
            pl = d / f"plan-{arm}-{r}.json"
            plan = json.loads(pl.read_text(encoding="utf-8")) if pl.is_file() else {}
            ent["n_refused_plan"] = plan.get("n_refused")
            ent["speed"] = throughput(res, plan.get("tokens") or {})
            out["arms"][arm][r] = ent
        tm = d / f"timing-{arm}.json"
        if tm.is_file():
            out["arms"][arm]["_timing"] = json.loads(tm.read_text(encoding="utf-8"))
        ctl = readouts[0]
        for r in have:  # paired readout - control on identical rows (frozen bootstrap.compare)
            if r == ctl or ctl not in have:
                continue
            cmp = _subset("compare", "--candidate", str(d / f"{arm}.{r}.results.jsonl"),
                          "--baseline", str(d / f"{arm}.{ctl}.results.jsonl"))
            (d / f"compare_{arm}_{r}_vs_{ctl}.json").write_text(json.dumps(cmp, indent=1), encoding="utf-8")
            out["paired"][f"{arm}:{r}-{ctl}"] = cmp.get("diff", cmp)
    if "parent" in out["arms"]:  # child - parent per readout (the trained cell's gain under each prompt)
        for r in readouts:
            c, p = d / f"child.{r}.results.jsonl", d / f"parent.{r}.results.jsonl"
            if c.is_file() and p.is_file():
                cmp = _subset("compare", "--candidate", str(c), "--baseline", str(p))
                (d / f"compare_child_vs_parent_{r}.json").write_text(json.dumps(cmp, indent=1), encoding="utf-8")
                out["paired"][f"child-parent:{r}"] = cmp.get("diff", cmp)
    ref_pin = (params.get("reference_results_sha256") or {}).get("d3_child_index")
    ref = Path(a.reference) if a.reference else None
    if ref_pin and ref is not None and ref.is_file() and (d / f"child.{readouts[0]}.results.jsonl").is_file():
        if _sha(ref) != ref_pin:
            raise SystemExit(f"reference {ref} sha256 != reference_results_sha256.d3_child_index")
        mine = {json.loads(x)["run_id"]: json.loads(x) for x in
                (d / f"child.{readouts[0]}.results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        theirs = {json.loads(x)["run_id"]: json.loads(x) for x in ref.read_text(encoding="utf-8").splitlines() if x.strip()}
        same = sum(1 for k, v in mine.items() if (theirs.get(k) or {}).get("response") == v.get("response"))
        out["control_reproduces_in_job_index"] = {"rows": len(mine), "identical_responses": same,
                                                  "reference_sha256": ref_pin}
    usd = params["batch_usd_per_min"]
    out["cost"] = {"elapsed_min": a.elapsed_min, "runtime_usd_batch": round(a.elapsed_min * usd, 2),
                   "usd_per_min_batch": usd, "runtime_usd_quoted": round(a.elapsed_min * 0.063, 2),
                   "usd_per_min_quoted": 0.063, "build_minutes_not_included": True,
                   "worst_case_usd": worst_case_usd(params), "quoted_max_cost_usd": quoted_max_cost_usd(params),
                   "hard_cap_usd": params["hard_cap_usd"]}
    out["checkpoint"] = {k: v for k, v in (params.get("checkpoint") or {}).items() if not k.startswith("_")}
    idf = d / "identity-child.json"
    if idf.is_file():
        idc = json.loads(idf.read_text(encoding="utf-8"))
        out["checkpoint"]["loaded"] = idc.get("checkpoint")
        out["readout_control"] = idc.get("readout_control")
    out["verdict"] = "measured"
    return out


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: every readout of the child arm measured on all subset rows, from the
    pinned checkpoint."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        c = d["arms"]["child"]
        rs = d["readouts"]
        return (d.get("schema") == SCHEMA and d.get("verdict") == "measured" and d.get("rows") == N_SUBSET
                and len(rs) >= 2 and all(sum(c[r]["status"].values()) == N_SUBSET for r in rs)
                and all(fin(c[r]["index"]["index"][f]) for r in rs for f in ("point", "lo", "hi"))
                and all(f"child:{r}-{rs[0]}" in d["paired"] for r in rs[1:])
                and bool(d["checkpoint"].get("ckpt_sha256"))
                and (d["checkpoint"].get("loaded") or {}).get("sha256") == d["checkpoint"]["ckpt_sha256"])
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ CPU: pricing
def estimate(params: dict) -> dict[str, Any]:
    """Projected minutes of the scoring phase(s) = rows x readouts at the MEASURED per-row child rate of D1 on this
    subset (``measured_rates_reference.child_rows_per_min``, job-c7pq2 units-child-index.json), each readout scaled by
    its committed prompt-length ratio to the control (``readout_length_ratio``: CPU char counts of the rendered
    prompts over the 2,008 rows, ``python -m jobs.m2_first_cell.readout_ab ratios``), + the phase's fixed minutes
    (model load, lineage rebuild, temperature fit, plan passes). Pass = every phase <= 95 % of its cap."""
    readouts = parse_readouts(params["readouts"])
    cfg = json.loads((ROOT / params["cell_config"]).read_text(encoding="utf-8"))
    ctl = check_readouts(cfg, readouts, bool(params.get("allow_unmatched_control")))
    m = params["measured_rates_reference"]
    ratio = params.get("readout_length_ratio") or {}
    per = {r: N_SUBSET / m["child_rows_per_min"] * float(ratio.get(r, 1.0)) for r in readouts}
    phases = {"readout_ab": sum(per.values()) + params["evals_fixed_minutes"]["child"]}
    if params.get("readout_ab_parent_arm"):
        phases["parent_ab"] = (sum(N_SUBSET / m["parent_rows_per_min"] * float(ratio.get(r, 1.0)) for r in readouts)
                               + params["evals_fixed_minutes"]["parent"])
    caps = params["phase_caps_minutes"]
    out = {"readouts": readouts, "control": ctl, "minutes_per_readout": {r: round(v, 1) for r, v in per.items()},
           "minutes": {k: round(v, 1) for k, v in phases.items()}, "caps": {k: caps[k] for k in phases},
           "ok": all(v <= 0.95 * caps[k] for k, v in phases.items())}
    fixed = sum(params["expected_minutes"].get(k, 0) for k in ("setup", "guard", "checkpoint", "report"))
    run = fixed + sum(phases.values())
    out["expected_run_minutes"] = round(run, 1)
    out["usd_batch_expected"] = round(run * params["batch_usd_per_min"], 2)
    out["usd_batch_expected_with_build"] = round((run + params["expected_minutes"].get("build", 0))
                                                 * params["batch_usd_per_min"], 2)
    return out


def ratios(rows_path: str, readouts: list[str], tokenizer_dir: str | None = None) -> dict[str, Any]:
    """CPU: total rendered prompt characters per readout over the subset rows (every question prompt of every row,
    the layout-g1v2 render with the label ladder at its first rung), as a ratio to the first readout's. v3w's word-id
    questions need the tokenizer (``tokenizer_dir``: the RWKV7-G1k tokenizer.json dir); without it v3w renders as v3."""
    import gzip

    from decide.schema import canonical_question_ids
    from heads.layout import _g1_prompt, _g1_spec, _g1_table, g1_word_qids, resolve_template, tagged_segments

    enc = None
    if tokenizer_dir:
        from train.run_cell import parent_encoder

        enc = parent_encoder("rwkv7", tokenizer_dir)

    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    tot = {}
    for r in readouts:
        tid = READOUTS[r]
        n = 0
        if READOUT_LAYOUT[r] != "g1v2":  # v1: the ordered layout's own segments
            tpl = resolve_template(tid)
            for x in rows:
                req = {"state": x["state"], "questions": x["questions"]}
                for qid in canonical_question_ids(x["questions"]):
                    n += sum(len(t) for t, _, _ in tagged_segments(tpl, req, qid, READOUT_LAYOUT[r]))
            tot[r] = n
            continue
        table = _g1_table(_g1_spec(tid)["table"])[0]
        for x in rows:
            req = {"state": x["state"], "questions": x["questions"]}
            words = g1_word_qids(tid, req, enc) if enc is not None else frozenset()
            for qid in canonical_question_ids(x["questions"]):
                n += len(_g1_prompt(tid, req, qid, table, 64, words)[1])
        tot[r] = n
    return {"rows": len(rows), "chars": tot, "word_ids_tokenizer": bool(enc), "ratio": {r: round(tot[r] / tot[readouts[0]], 4) for r in readouts}}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.readout_ab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--readouts", required=True, help="comma list, control first (e.g. v2f,v3p)")
    s.add_argument("--cell", help="child: the checkpoint's cell.json (config + ladder)")
    s.add_argument("--cell-config", help="parent: the mode's cell config (WKV)")
    s.add_argument("--ckpt-dir")
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records")
    s.add_argument("--train-dir")
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness")
    s.add_argument("--rows", required=True, help="the 2,008-row subset (jobs.common.dev_subset rows)")
    s.add_argument("--resume-dir", required=True, help="per-row append logs (checkpoint dir): a restart resumes")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--allow-unmatched-control", action="store_true", help="plumbing dry runs on a non-g1v2 cell only")
    s.add_argument("--device", default="cuda")
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    q = sub.add_parser("ratios")
    q.add_argument("--rows", required=True)
    q.add_argument("--readouts", default="v1,v2f,v3w,v3p,v2fp")
    q.add_argument("--tokenizer-dir", default=None)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--params", required=True)
    r.add_argument("--elapsed-min", type=float, required=True)
    r.add_argument("--reference", default=None, help="the D3 job's child index results (pinned in params)")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        need = ("cell", "ckpt_dir", "records", "train_dir", "guard_readiness") if a.arm == "child" else ("cell_config",)
        miss = [k for k in need if not getattr(a, k)]
        if miss:
            ap.error(f"--arm {a.arm} needs {miss}")
        a.max_input_tokens = None  # load_child does not read it; the subset keeps the adapter's default cap
        return score(a)
    if a.cmd == "estimate":
        try:
            est = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")))
        except (ValueError, KeyError) as e:
            print(f"estimate: {e}", file=sys.stderr)
            return 5
        print(json.dumps(est, indent=1))
        return 0 if est["ok"] else 15
    if a.cmd == "ratios":
        print(json.dumps(ratios(a.rows, [x for x in a.readouts.split(",") if x], a.tokenizer_dir), indent=1))
        return 0
    res = report(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    c = res["arms"]["child"]
    print(json.dumps({"index": {r: c[r]["index"]["index"] for r in res["readouts"]}, "paired": res["paired"],
                      "cost": res["cost"], "valid": valid_result(res)}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
