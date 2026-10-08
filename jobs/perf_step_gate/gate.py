"""jobs/perf_step_gate: the m2 cell's train step under four pre-registered execution variants (rule.py, RULE v2):
greedy (job-tvuat's ADOPT config, the numerics reference) / fast (= d1_fast: min_cost + single cont + compile_blocks,
the speed reference) / fast_fla (fast with rwkv7_wkv "fla") / fast_fla_nocompile (fla without compile_blocks).
fla variants select the unverified fla WKV (``allow_unverified=True``: this job IS a gate) and record
``heads.rwkv7_wkv.provenance()`` (fla calls, fallbacks to "chunked") in the variant record.

    python -m jobs.perf_step_gate.gate variant --name V --out DIR --grads GDIR --model-dir SNAP --deadline-s S
    python -m jobs.perf_step_gate.gate decide  --out DIR --grads GDIR      # rule.decide -> DIR/decision.json

``variant`` (one process per variant, clean peaks): load, then ONE un-applied optimiser step on the rows of mix step
``rule.GRAD_STEP`` (lora_B set from ``rule.GRAD_B_SEED`` first) -> its loss + the flattened LoRA gradient (fp32, to
GDIR, not exported), then the 23 steps of ``jobs/fla_rwkv7/train_mix_v2.json`` (CUDA-synchronised seconds, peak GB reset
per step, OOM recorded), then the ``rule.LADDER`` rungs (row_batch copies of a synthetic K = 10 row). The record is
rewritten after every step. No step starts that cannot finish before ``--deadline-s``, judged from the slowest so far.
``decide``: the cosine of each variant's LoRA gradient with the baseline's (float64, chunked), then ``rule.decide``.

``PERF_STEP_DRYRUN=1``: the tiny vendored RWKV-7 on CPU, same code path (the runner sets SSDM_COMPILE_BACKEND=aot_eager).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from pathlib import Path

from jobs.perf_step_gate import rule as R

DRYRUN = os.environ.get("PERF_STEP_DRYRUN") == "1"
DEVICE = "cpu" if DRYRUN else "cuda"


class Recorder:
    def __init__(self, path: Path, **d):
        self.path, self.d = path, d
        self.flush()

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def ladder_shape(T: int) -> dict:
    k = R.LADDER_K
    a = T - 2 * k - 2
    return {"T": T, "conts": [R.LADDER_CONT_LEN] * k, "qtype": "choice", "state_span": [16, a], "field_span": [2, 12],
            "option_spans": [[a + 2 * j, a + 2 * j + 1] for j in range(k)]}


def load_backbone(model_dir: str, wkv: str):
    from heads.rwkv7_wkv import select_wkv

    if DRYRUN:
        from heads.backbone import tiny_hf_backbone

        bb = tiny_hf_backbone("rwkv7", fork_mode="cache")
    else:
        from heads.backbone import load_rwkv7_hf_backbone

        bb = load_rwkv7_hf_backbone(model_dir, device=DEVICE, fork_mode="cache", wkv=None)
    select_wkv(bb.model, wkv, allow_unverified=True)  # fla is GPU-unverified for training: this job is its gate
    return bb


def lora_named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.endswith(("lora_A", "lora_B"))]


def grad_step(tr, rows_from: int, gfile: Path, cuda: bool) -> dict:
    """One optimiser step's loss and LoRA gradient on items[rows_from : +8], NOT applied (opt.step skipped, step counter
    and LR schedule restored), with every lora_B set from GRAD_B_SEED first and restored after."""
    import torch

    named = lora_named(tr.model)
    bs = [p for n, p in named if n.endswith("lora_B")]
    saved_b = [p.detach().clone() for p in bs]
    g = torch.Generator().manual_seed(R.GRAD_B_SEED)
    with torch.no_grad():
        for p in bs:
            p.copy_(torch.randn(p.shape, generator=g) * R.GRAD_B_SCALE)
    real_step, sched_state, step0 = tr.opt.step, tr.sched.state_dict(), tr.step
    grads: dict = {}

    def capture(*a, **k):
        for n, p in named:
            grads[n] = None if p.grad is None else p.grad.detach().float().flatten().cpu()

    tr.opt.step = capture
    tr.epoch, tr.cursor = 0, rows_from
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    try:
        loss = tr.train_step()
        if cuda:
            torch.cuda.synchronize()
        out = {"loss": loss, "finite": math.isfinite(loss), "s": time.perf_counter() - t}
        missing = [n for n, v in grads.items() if v is None]
        vec = torch.cat([grads[n] if grads[n] is not None else torch.zeros(p.numel()) for n, p in named])
        torch.save({"names": [n for n, _ in named], "vec": vec}, gfile)
        out.update(file=gfile.name, numel=int(vec.numel()), norm=float(vec.double().norm()), missing=missing[:5],
                   n_missing=len(missing))
    except Exception as e:  # noqa: BLE001  (OOM included): recorded; the variant then cannot qualify
        out = {"error": f"{type(e).__name__}: {e}"[:300], "s": time.perf_counter() - t}
    finally:
        tr.opt.step = real_step
        tr.opt.zero_grad(set_to_none=True)
        tr.sched.load_state_dict(sched_state)
        tr.step = step0
        tr.history = tr.history[:step0]
        with torch.no_grad():
            for p, b in zip(bs, saved_b):
                p.copy_(b)
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    out["rows"] = list(range(rows_from, rows_from + R.ROWS_PER_STEP))
    return out


def dynamo_stats() -> dict:
    try:
        from torch._dynamo.utils import counters

        return {"stats": dict(counters.get("stats", {})), "recompiles": sum(counters.get("recompiles", {}).values()),
                "graph_breaks": sum(counters.get("graph_break", {}).values())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def phase_variant(out: Path, gdir: Path, name: str, model_dir: str, deadline_s: float) -> None:
    t_start = time.perf_counter()
    import torch

    from heads import act_ckpt
    from heads.model import DecisionModel
    from jobs.fla_rwkv7 import train_mix
    from train.guard import SplitGuard
    from train.run_cell import train_config
    from train.trainer import Trainer, with_execution

    if name not in R.VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; pre-registered: {R.VARIANTS}")
    flags = dict(R.FLAGS[name])
    cuda = DEVICE == "cuda"
    mix = train_mix.load()
    cell = {**mix["cell"], "ckpt_every": 0, "eval_every": 0, **flags}
    per = cell["batch_size"] * cell["grad_accum"]
    if per != R.ROWS_PER_STEP or len(mix["rows"]) < R.N_STEPS * per:
        raise SystemExit(f"train_mix_v2.json: {len(mix['rows'])} rows, {per} per step; need {R.N_STEPS} x 8")
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "dryrun": DRYRUN,
           "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "compile_backend": os.environ.get("SSDM_COMPILE_BACKEND", "inductor")}
    rec = Recorder(out / f"variant_{name}.json", variant=name, flags=flags, env=env,
                   alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), mix_sha256=mix["sha256"], cell=mix["cell"],
                   deadline_s=deadline_s, steps=[], ladder={}, complete=False)
    bb = load_backbone(model_dir, flags["rwkv7_wkv"])
    rec.d["load_s"] = round(time.perf_counter() - t_start, 1)
    rec.d["weights_gb"] = torch.cuda.memory_allocated() / 1e9 if cuda else None
    items = train_mix.rows_from_shapes(mix["rows"][: R.N_STEPS * per], vocab=bb.model.config.vocab_size, seed=0)
    model = DecisionModel(bb, cell["head"], cell.get("head_kwargs") or {}, pad_id=0, seed=cell["seed"])
    model.to(bb.device)
    tcfg = with_execution(train_config(cell, R.N_STEPS, None, cuda), cell)
    tr = Trainer(model, tcfg, items, [], guard=SplitGuard.synthetic_only(
        "perf_step_gate: synthetic ids at the m2 cell's row shapes"))
    tr._order = lambda epoch: list(range(len(items)))  # the mix is already in the cell's epoch-0 order
    from heads.rwkv7_wkv import select_wkv

    active = {"rwkv7_wkv": str(select_wkv(bb.model, None)), "act_ckpt": act_ckpt.is_enabled(bb.model),
              "row_batch": tcfg.row_batch, "row_tokens": tcfg.row_tokens, "row_plan": tcfg.row_plan,
              "cont_plan": tcfg.cont_plan,
              "compile_blocks": act_ckpt.compile_backend_of(bb.model) is not None}
    rec.d["active"] = active
    if active != flags:
        raise SystemExit(f"flags did not take effect: {active} vs {flags}")
    rec.flush()
    rec.d["grad"] = grad_step(tr, R.GRAD_STEP * per, gdir / f"grad_{name}.pt", cuda)
    rec.d["dynamo_after_grad"] = dynamo_stats()
    rec.flush()

    def left() -> float:
        return deadline_s - (time.perf_counter() - t_start)

    def run_step(rows_from: int, label: dict) -> dict:
        r = dict(label)
        tr.epoch, tr.cursor = 0, rows_from
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            loss = tr.train_step()
            if cuda:
                torch.cuda.synchronize()
            r.update(s=time.perf_counter() - t, loss=loss, finite=math.isfinite(loss))
        except torch.OutOfMemoryError as e:
            r.update(oom=True, s=time.perf_counter() - t, error=str(e)[:200])
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        if r.get("oom"):
            tr.opt.zero_grad(set_to_none=True)
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        return r

    done: list[float] = []
    for i in range(R.N_STEPS):
        rows = items[i * per:(i + 1) * per]
        lab = {"i": i, "timed": i >= R.WARMUP_STEPS, "tokens": sum(len(x.row.ids) for x in rows),
               "max_T": max(len(x.row.ids) for x in rows)}
        if left() <= 0 or (done and left() < 1.5 * max(done[-3:])):
            rec.d["steps"].append({**lab, "skipped": "deadline"})
            rec.flush()
            continue
        r = run_step(i * per, lab)
        if "loss" in r:
            done.append(r["s"])
        rec.d["steps"].append(r)
        rec.flush()
    rec.d["dynamo_after_steps"] = dynamo_stats()
    from heads import rwkv7_wkv as W

    rec.d["fla"] = W.provenance()  # rule v2: an fla variant needs fla calls > 0 and no fallback to "chunked"
    rec.d["steps_s"] = round(time.perf_counter() - t_start, 1)
    rec.flush()

    unit = max(1, flags["row_batch"])
    tr.cfg = dataclasses.replace(tr.cfg, grad_accum=unit, batch_size=1)
    last = max(done[-3:]) if done else 60.0
    for T in R.LADDER:
        if left() < 3 * last + 20:
            rec.d["ladder"][str(T)] = {"skipped": "deadline"}
            rec.flush()
            break
        litems = train_mix.rows_from_shapes([ladder_shape(T)] * unit, vocab=bb.model.config.vocab_size, seed=T)
        tr.train_items = litems
        tr._order = lambda epoch, n=len(litems): list(range(n))
        r = run_step(0, {"rows": unit, "K": R.LADDER_K, "cont_len": R.LADDER_CONT_LEN})
        r["ok"] = bool(r.get("finite")) and not r.get("oom") and r["peak_gb"] <= R.MAX_PEAK_GB
        rec.d["ladder"][str(T)] = r
        rec.flush()
        last = max(last, r["s"])
        if not r["ok"]:
            break
    rec.d["wall_s"] = round(time.perf_counter() - t_start, 1)
    rec.d["complete"] = True
    rec.flush()


def grad_cosine(a: Path, b: Path) -> float | None:
    import torch

    if not (a.is_file() and b.is_file()):
        return None
    x, y = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    if x["names"] != y["names"] or x["vec"].numel() != y["vec"].numel():
        return None
    xv, yv = x["vec"], y["vec"]
    dot = nx = ny = 0.0
    for s in range(0, xv.numel(), 1 << 24):
        u, v = xv[s:s + (1 << 24)].double(), yv[s:s + (1 << 24)].double()
        dot += float(u @ v)
        nx += float(u @ u)
        ny += float(v @ v)
    return dot / math.sqrt(nx * ny) if nx > 0 and ny > 0 else None


def phase_decide(out: Path, gdir: Path) -> int:
    recs = {}
    for v in R.VARIANTS:
        p = out / f"variant_{v}.json"
        recs[v] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    base = gdir / f"grad_{R.BASELINE}.pt"
    cos = {v: grad_cosine(gdir / f"grad_{v}.pt", base) for v in R.VARIANTS if v != R.BASELINE}
    d = R.decide(recs, cos)
    (out / "decision.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": d["verdict"], "chosen": d.get("chosen"),
                      "s_per_step": {v: d["variants"][v].get("matched_median_step_s") for v in R.VARIANTS},
                      "speedup_vs_fast": {v: d["variants"][v].get("speedup_vs_fast") for v in R.VARIANTS},
                      "rows_190min": {v: d["variants"][v].get("rows_190min") for v in R.VARIANTS},
                      "fla_speeds_training": d.get("fla_speeds_training"), "grad_cos": cos}))
    return 0 if R.valid_decision(d) else 9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.perf_step_gate.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variant")
    v.add_argument("--name", required=True)
    v.add_argument("--out", required=True, type=Path)
    v.add_argument("--grads", required=True, type=Path)
    v.add_argument("--model-dir", required=True)
    v.add_argument("--deadline-s", required=True, type=float)
    d = sub.add_parser("decide")
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--grads", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "variant":
        a.grads.mkdir(parents=True, exist_ok=True)
        phase_variant(a.out, a.grads, a.name, a.model_dir, a.deadline_s)
        return 0
    return phase_decide(a.out, a.grads)


if __name__ == "__main__":
    raise SystemExit(main())
