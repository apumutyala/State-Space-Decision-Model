"""jobs/cell_step_profile: the m2 first cell's train step under the four pre-registered execution variants (rule.py).

    python -m jobs.cell_step_profile.cell_step variant --name V --out DIR --model-dir SNAP --deadline-s S
    python -m jobs.cell_step_profile.cell_step decide  --out DIR          # rule.decide -> DIR/decision.json
    python -m jobs.cell_step_profile.cell_step profile --out DIR --model-dir SNAP --deadline-s S

``variant`` (one process per variant, so every peak is clean): the cell's own step (``train.trainer.Trainer.train_step``
with ``train.run_cell.train_config`` + ``train.trainer.with_execution``: fork head, cache fork, LoRA r64, Clef loss,
batch 1 x accumulation 8, bf16 autocast). First the A2 probe (``probe_loss``: forward loss, no grad, eval mode, on the
rows of mix step ``rule.PROBE_STEP``, before any training), then the 23 steps of ``jobs/fla_rwkv7/train_mix_v2.json`` (rows rebuilt from
their shapes, exactly as the fla_rwkv7 ``train_v2`` phase does). Per step: CUDA-synchronised seconds, tok/s (prefix
tokens / s), peak GB (reset before every step), OOM (recorded; the run goes on), loss. Then the K = 10 ladder.
``DIR/variant_<V>.json`` is rewritten after every step, so a timeout keeps what was measured. No step or ladder rung
starts that cannot finish before ``--deadline-s`` (seconds from process start), judged from the slowest step so far.

``profile`` (after ``decide``, one process): ``torch.profiler`` (CPU + CUDA) over one re-run of the mid-length step
(``profile_step_index``: the step with the median prefix-token count among those the baseline completed) for the decided variant (the fastest present one
if nothing qualified) and for ``baseline``: top-20 ops by self device time and by call count, the kernel-launch count
and the aten call count -> ``DIR/profile.json`` (+ gzipped Chrome traces when time allows). It runs last, so profiler
overhead never touches the timed steps; informational (not part of the rule).

``CELL_STEP_DRYRUN=1`` (CPU dry run, tests and the shell-level dry run): the tiny 2-layer vendored RWKV-7 on CPU
instead of the 13.3B, same rows, same code path; the profiler records CPU activity only.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

from jobs.cell_step_profile import rule as R

DRYRUN = os.environ.get("CELL_STEP_DRYRUN") == "1"
DEVICE = "cpu" if DRYRUN else "cuda"
#: profile tables: top-N ops by self device time and by call count
TOP = 20


def _torch():
    import torch

    return torch


def profile_step_index(mix_rows: list[dict], completed: list[int] | None = None) -> int:
    """The step with the median prefix-token count among ``completed`` (the steps the baseline finished, so its profile
    does not OOM), or among all 23 when the baseline finished none."""
    per = R.ROWS_PER_STEP
    tokens = [sum(r["T"] for r in mix_rows[i * per:(i + 1) * per]) for i in range(R.N_STEPS)]
    pool = sorted(completed) if completed else list(range(R.N_STEPS))
    return sorted(pool, key=lambda i: (tokens[i], i))[len(pool) // 2]


def ladder_shape(T: int) -> dict:
    """A synthetic row of T prefix tokens with K = LADDER_K continuations of LADDER_CONT_LEN tokens."""
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
    select_wkv(bb.model, wkv)
    return bb


class Recorder:
    def __init__(self, path: Path, **d):
        self.path, self.d = path, d
        self.flush()

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def _env() -> dict:
    torch = _torch()
    d = {"torch": torch.__version__, "cuda": torch.version.cuda, "dryrun": DRYRUN,
         "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    if torch.cuda.is_available() and not DRYRUN:
        d["gpu"] = torch.cuda.get_device_name(0)
    return d


def _top(prof) -> dict:
    """Top-20 ops by self device time and by call count, and the kernel-launch count, from a finished profiler."""
    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0.0)
        rows.append({"name": e.key, "count": int(e.count), "self_device_us": float(dev or 0.0),
                     "self_cpu_us": float(e.self_cpu_time_total)})
    launches = sum(r["count"] for r in rows if r["name"].startswith(("cudaLaunchKernel", "cuLaunchKernel")))
    aten = [r for r in rows if r["name"].startswith("aten::")]
    return {"by_device_time": sorted(rows, key=lambda r: -r["self_device_us"])[:TOP],
            "by_count": sorted(aten, key=lambda r: -r["count"])[:TOP],
            "kernel_launches": launches, "aten_calls": sum(r["count"] for r in aten),
            "self_device_us_total": sum(r["self_device_us"] for r in rows)}


def probe_loss(tr, rows, cuda: bool) -> dict:
    """A2: the step loss of ``rows`` without training: eval mode, no grad, bf16 autocast as in training, through the
    variant's own execution path (row groups when ``row_batch`` > 1, else one row per call), weighted exactly as
    ``Trainer.train_step`` weights it (the mean over the step's rows)."""
    torch = _torch()
    from train.losses import decision_loss
    from train.trainer import row_groups

    n = len(rows)
    groups = (row_groups([len(it.row.ids) for it in rows], tr.cfg.row_batch, tr.cfg.row_tokens)
              if tr.cfg.row_batch > 1 else [[j] for j in range(n)])
    tr.model.eval()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    try:
        total = 0.0
        with torch.no_grad():
            for g in groups:
                batch = [rows[j] for j in g]
                with tr._autocast():
                    lp = tr.model([it.row for it in batch]).log_probs.float()
                y = torch.tensor([it.label for it in batch], device=lp.device)
                total += float(decision_loss(lp, y, [it.row.qtype for it in batch], tr.cfg.loss)) * len(g) / n
        out = {"loss": total, "finite": math.isfinite(total), "calls": len(groups)}
    except Exception as e:  # noqa: BLE001  (OOM included): the clause then fails (variant) / is inconclusive (baseline)
        out = {"error": f"{type(e).__name__}: {e}"[:300]}
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    finally:
        tr.model.train()
    out.update(step=R.PROBE_STEP, rows=n, tokens=sum(len(it.row.ids) for it in rows), s=time.perf_counter() - t,
               peak_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0)
    return out


def phase_variant(out: Path, name: str, model_dir: str, deadline_s: float) -> None:
    t_start = time.perf_counter()
    torch = _torch()
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
    rec = Recorder(out / f"variant_{name}.json", variant=name, flags=flags, env=_env(),
                   alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), mix_sha256=mix["sha256"], cell=mix["cell"],
                   deadline_s=deadline_s, steps=[], ladder={}, complete=False)
    bb = load_backbone(model_dir, flags["rwkv7_wkv"])
    rec.d["load_s"] = round(time.perf_counter() - t_start, 1)
    rec.d["weights_gb"] = torch.cuda.memory_allocated() / 1e9 if cuda else None
    items = train_mix.rows_from_shapes(mix["rows"][: R.N_STEPS * per], vocab=bb.model.config.vocab_size, seed=0)
    model = DecisionModel(bb, cell["head"], cell.get("head_kwargs") or {}, pad_id=0, seed=cell["seed"])
    model.to(bb.device)
    tcfg = with_execution(train_config(cell, R.N_STEPS, None, cuda), cell)
    # the cell's own warmup (PR #74's train_config reads it; agent-main's does not): only the LR moves, not the cost
    tcfg = dataclasses.replace(tcfg, warmup_steps=int(cell.get("warmup_steps") or 0))
    tr = Trainer(model, tcfg, items, [], guard=SplitGuard.synthetic_only(
        "cell_step_profile: synthetic ids at the m2 cell's row shapes"))
    tr._order = lambda epoch: list(range(len(items)))  # the mix is already in the cell's epoch-0 order
    from heads.rwkv7_wkv import select_wkv

    rec.d["active"] = {"wkv": select_wkv(bb.model, None), "act_ckpt": act_ckpt.is_enabled(bb.model),
                       "batch_rows": bool(bb.batch_rows), "row_batch": tcfg.row_batch, "row_tokens": tcfg.row_tokens}
    if (rec.d["active"]["wkv"], rec.d["active"]["act_ckpt"], tcfg.row_batch, tcfg.row_tokens) != (
            flags["rwkv7_wkv"], flags["act_ckpt"], flags["row_batch"], flags["row_tokens"]):
        raise SystemExit(f"flags did not take effect: {rec.d['active']} vs {flags}")
    rec.d["setup_s"] = round(time.perf_counter() - t_start, 1)
    rec.flush()
    # A2 GPU loss clause: forward loss on the probe step's rows BEFORE any training (same seeds in every process)
    rec.d["probe"] = probe_loss(tr, items[R.PROBE_STEP * per:(R.PROBE_STEP + 1) * per], cuda)
    rec.flush()

    def left() -> float:
        return deadline_s - (time.perf_counter() - t_start)

    def run_step(rows_from: int, timed_label: dict) -> dict:
        """One optimiser step on items[rows_from : rows_from + 8]; fills seconds / peak / loss or OOM."""
        r = dict(timed_label)
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
        r["reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9 if cuda else 0.0
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
               "cont_tokens": sum(sum(map(len, x.row.continuations)) for x in rows),
               "max_T": max(len(x.row.ids) for x in rows), "K": [x.row.n_options for x in rows]}
        if left() <= 0 or (done and left() < 1.5 * max(done)):
            rec.d["steps"].append({**lab, "skipped": "deadline"})
            rec.flush()
            continue
        r = run_step(i * per, lab)
        if "loss" in r:
            done.append(r["s"])
            r["tok_s"] = r["tokens"] / r["s"]
        rec.d["steps"].append(r)
        rec.flush()
    timed = [s for s in rec.d["steps"] if s["timed"] and "tok_s" in s]
    rec.d["median_tok_s"] = statistics.median([s["tok_s"] for s in timed]) if timed else None
    rec.d["steps_s"] = round(time.perf_counter() - t_start, 1)
    rec.flush()

    # ---- K = 10 ladder: one optimiser step over row_batch copies of a synthetic row, ascending, stop at first failure
    unit = max(1, flags["row_batch"])
    tr.cfg = dataclasses.replace(tr.cfg, grad_accum=unit, batch_size=1)
    last = max(done) if done else 60.0
    for T in R.LADDER:
        if left() < 2 * last + 20:
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


def _profile_one(out: Path, bb, name: str, items, pstep: int, cell: dict, left, warm: bool) -> dict:
    """Profile one re-run of step ``pstep`` under ``name``'s flags on the already-loaded backbone."""
    torch = _torch()
    from torch.profiler import ProfilerActivity, profile

    from heads.model import DecisionModel
    from heads.rwkv7_wkv import select_wkv
    from train.guard import SplitGuard
    from train.lora import remove_lora
    from train.run_cell import train_config
    from train.trainer import Trainer, with_execution

    flags = R.FLAGS[name]
    cuda = DEVICE == "cuda"
    select_wkv(bb.model, flags["rwkv7_wkv"])
    c = {**cell, **flags}
    model = DecisionModel(bb, c["head"], c.get("head_kwargs") or {}, pad_id=0, seed=c["seed"])
    model.to(bb.device)
    tr = Trainer(model, with_execution(train_config(c, R.N_STEPS, None, cuda), c), items, [],
                 guard=SplitGuard.synthetic_only("cell_step_profile: profile"))
    tr._order = lambda epoch: list(range(len(items)))
    per = R.ROWS_PER_STEP
    rec = {"variant": name, "flags": flags, "step": pstep}
    try:
        if warm:  # one un-profiled run first (first profile of the process): allocator and caches warm
            tr.epoch, tr.cursor = 0, pstep * per
            tr.train_step()
        tr.epoch, tr.cursor = 0, pstep * per
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        with profile(activities=acts) as prof:
            loss = tr.train_step()
            if cuda:
                torch.cuda.synchronize()
        rec.update(step_s_under_profiler=round(time.perf_counter() - t, 2), loss=loss)
        t = time.perf_counter()
        rec.update(_top(prof))
        rec["tables_s"] = round(time.perf_counter() - t, 1)
        if left() > 4 * (rec["step_s_under_profiler"] + rec["tables_s"]) + 30:
            import gzip
            import shutil

            raw = out / f"trace_{name}.json"
            t = time.perf_counter()
            prof.export_chrome_trace(str(raw))
            with open(raw, "rb") as fi, gzip.open(str(raw) + ".gz", "wb", compresslevel=5) as fo:
                shutil.copyfileobj(fi, fo)
            raw.unlink()
            rec.update(trace=raw.name + ".gz", trace_s=round(time.perf_counter() - t, 1))
        del prof
    except Exception as e:  # noqa: BLE001  (OOM included: recorded, the other profile still runs)
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        tr.opt.zero_grad(set_to_none=True)
        del tr, model
        remove_lora(bb.model)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return rec


def phase_profile(out: Path, model_dir: str, deadline_s: float) -> None:
    t_start = time.perf_counter()
    from jobs.fla_rwkv7 import train_mix

    def left() -> float:
        return deadline_s - (time.perf_counter() - t_start)

    dec_p = out / "decision.json"
    dec = json.loads(dec_p.read_text(encoding="utf-8")) if dec_p.exists() else {}
    best = dec.get("chosen")
    if best in (None, "baseline"):  # nothing (else) qualified: the fastest present non-baseline variant
        speed = {v: ((dec.get("variants") or {}).get(v) or {}).get("median_tok_s") or 0.0 for v in R.VARIANTS[1:]}
        best = max(R.VARIANTS[1:], key=lambda v: (speed[v], -R.VARIANTS.index(v)))
    mix = train_mix.load()
    bp = out / "variant_baseline.json"
    base_steps = json.loads(bp.read_text(encoding="utf-8")).get("steps", []) if bp.exists() else []
    pstep = profile_step_index(mix["rows"], [s["i"] for s in base_steps if "tok_s" in s])
    rec = Recorder(out / "profile.json", env=_env(), profile_step=pstep, decided=dec.get("chosen"), profiled={},
                   complete=False)
    bb = load_backbone(model_dir, "chunked")
    items = train_mix.rows_from_shapes(mix["rows"][: R.N_STEPS * R.ROWS_PER_STEP], vocab=bb.model.config.vocab_size,
                                       seed=0)
    cell = {**mix["cell"], "ckpt_every": 0, "eval_every": 0}
    for j, name in enumerate((best, "baseline")):  # the decided variant first: it is the short one
        if left() < 60:
            rec.d["profiled"][name] = {"skipped": "deadline"}
            rec.flush()
            continue
        rec.d["profiled"][name] = _profile_one(out, bb, name, items, pstep, cell, left, warm=j == 0)
        rec.flush()
    rec.d["complete"] = True
    rec.flush()


def phase_decide(out: Path) -> int:
    recs = {}
    for v in R.VARIANTS:
        p = out / f"variant_{v}.json"
        recs[v] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    d = R.decide(recs)
    (out / "decision.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({k: d.get(k) for k in ("verdict", "chosen", "cell_config", "projection")}))
    return 0 if R.valid_decision(d) else 9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("variant", "decide", "profile"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", choices=R.VARIANTS)
    ap.add_argument("--model-dir")
    ap.add_argument("--deadline-s", type=float, default=math.inf)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "decide":
        return phase_decide(out)
    if a.phase == "profile":
        if not a.model_dir:
            ap.error("profile needs --model-dir")
        phase_profile(out, a.model_dir, a.deadline_s)
        return 0
    if not a.name or not a.model_dir:
        ap.error("variant needs --name and --model-dir")
    if not DRYRUN and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != R.ALLOC_CONF:
        raise SystemExit(f"PYTORCH_CUDA_ALLOC_CONF must be {R.ALLOC_CONF!r} (pre-registered)")
    phase_variant(out, a.name, a.model_dir, a.deadline_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
