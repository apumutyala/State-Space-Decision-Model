"""EV-PREREG-H2O-HOST eval-only arms: H2O-Lightning-4B (pinned revision) on the 231 public JevBench items, every arm
paired in ONE job (same items, same bootstrap draws B=1000 seed 16, same hardware).

    python -m jobs.jev_host_scan.h2o_eval slice   --records R.jsonl.gz --out slice.json   # deterministic held-out slice
    python -m jobs.jev_host_scan.h2o_eval verify  --model-dir SNAP --out verify.json      # revision + file pins (exit 10)
    python -m jobs.jev_host_scan.h2o_eval run     --model-dir SNAP --slice slice.json --records R --out DIR
    python -m jobs.jev_host_scan.h2o_eval derive  --out DIR                               # score-only arms (no GPU)
    python -m jobs.jev_host_scan.h2o_eval summarize --root DIR --out decision.json

Arms (all on public-231; ``DIR/arms/<arm>/preds.jsonl``, scored by ``jevkit score --public`` under the H2O price class):

  h2o            CONTROL. The pinned revision's OWN shim (``h2o_lightning_shim.py`` + ``serve_config.json`` from the
                 snapshot, both pinned by git blob id): its prompt, labels, per-type temperature and yes/no floor; the
                 vLLM call replaced by one transformers forward reading the label tokens' log-probabilities at the
                 answer slot (lm head fp32, config head_dtype). The raw label log-probabilities are kept (raw.jsonl).
  a_T            (a) one scalar T_J fitted by NLL on the FIXED held-out jev2a slice (never the public items), replacing
                 the shim's temperatures; the shim's yes/no floor kept.                       [prereg arm (a)]
  a_Ttype        (a, secondary) one T per question type (choice / noul / score), fitted on the slice only.
  a_Tfam         (a, secondary; EV-PREREG-AREA-TEMP/a's optional T_f) T_f per jev2a family with >= 200 slice decisions,
                 T_J elsewhere.
  b_native       the same forward read raw: T = 1, no floor (the uncalibrated "native" reading).
  b_shimcal      (b) jobs.m2_first_cell.screen.shim_calibrate semantics on b_native: the shim's own post-processing
                 (its temperatures + floor) applied to the native probabilities. On H2O itself this is ALGEBRAICALLY the
                 control (softmax(log softmax(l)/T) = softmax(l/T)); reported as that consistency check.
  c_symfork      (c) heads.symmetric ``sym-fork`` rows (jrt-h2o-shim:qwen3.5-4b format: options out of the prefix, one
                 forked continuation per option, content-canonical axis) scored by the LM's own continuation
                 log-likelihood (no head, no training), softmax over options. G1 invariance probe; its Composite is
                 reported but it is NOT a contender.

Derived arms reuse the control's forward (same latency and usage records by construction: they differ only in the
post-processing of the same logits). G1 probe: jevkit perturb ops option_reverse / option_shuffle / key_relabel (seed 0,
in-memory objects: no sort_keys round trip) for c_symfork and for the control.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / os.environ.get("JEV_SCAN_PARAMS", "jobs/jev_host_scan/job_params_h2o_eval.json")
WARMUP = 2
T_RANGE = (0.25, 8.0)  # EV-PREREG-AREA-TEMP's T range
T_GRID = 1401
MIN_FAMILY_FIT = 200  # EV-PREREG-AREA-TEMP/a: >= 200 decisions per family, else T_J
TYPES = ("choice", "noul", "score")


def cfg() -> dict:
    return json.loads(PARAMS.read_text(encoding="utf-8"))["h2o_eval"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob(p: Path) -> str:
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------ slice
def select_slice(records_gz: Path, salt: str, per_family: int) -> dict:
    """Deterministic held-out slice: per jev1 family8, the ``per_family`` ids with the smallest
    sha256(salt + "\\n" + id); streamed (one record in memory at a time + a bounded heap per family)."""
    import heapq

    heaps: dict[str, list] = {}
    n, h_all = 0, hashlib.sha256()
    with gzip.open(records_gz, "rb") as f:
        for line in f:
            h_all.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            fam = r["aug"]["jev1"]["family8"]
            k = hashlib.sha256((salt + "\n" + r["id"]).encode()).hexdigest()
            hp = heaps.setdefault(fam, [])
            heapq.heappush(hp, (-int(k[:16], 16), k, r["id"]))
            if len(hp) > per_family:
                heapq.heappop(hp)
    ids, fams = [], {}
    for fam in sorted(heaps):
        sel = sorted((k, i) for _, k, i in heaps[fam])
        fams[fam] = [i for _, i in sel]
        ids += fams[fam]
    blob = ("\n".join(sorted(ids)) + "\n").encode()
    return {"salt": salt, "per_family": per_family, "n_records": n, "records_uncompressed_sha256": h_all.hexdigest(),
            "ids_sha256": hashlib.sha256(blob).hexdigest(), "n": len(ids), "families": fams}


def slice_records(records_gz: Path, ids: set[str]) -> list[dict]:
    out = []
    with gzip.open(records_gz, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in ids:
                out.append(r)
    out.sort(key=lambda r: r["id"])
    return out


# ------------------------------------------------------------------------------------------------ identities
def verify(model_dir: Path) -> dict:
    """Fail closed unless the snapshot IS the pinned revision: the snapshot directory name (huggingface_hub names it by
    the commit sha) == revision, every pinned file present with its pinned sha256 (LFS) / git blob id (non-LFS)."""
    c = cfg()
    rev = c["revision"]
    d = Path(model_dir)
    got_rev = d.resolve().name
    if got_rev != rev:
        raise RuntimeError(f"snapshot commit {got_rev} != pinned revision {rev}")
    bad, out = {}, {"revision": got_rev, "files": {}}
    for name, pin in c["files"].items():
        p = d / name
        if not p.exists():
            bad[name] = "missing"
            continue
        if "sha256" in pin:
            g = _sha256(p.resolve())
            ok = g == pin["sha256"]
            out["files"][name] = {"sha256": g}
        else:
            g = _git_blob(p.resolve())
            ok = g == pin["git_blob"]
            out["files"][name] = {"git_blob": g, "sha256": _sha256(p.resolve())}
        if not ok:
            bad[name] = g
    if bad:
        raise RuntimeError(f"pinned files differ at {rev}: {bad}")
    return out


# ------------------------------------------------------------------------------------------------ the revision's shim
def load_rev_shim(model_dir: Path):
    """(module, Contract) of the snapshot's own shim (pinned by verify()), env {} (no SHIM_* override)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("h2o_lightning_shim_rev", Path(model_dir) / "h2o_lightning_shim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    conf = json.loads((Path(model_dir) / "serve_config.json").read_text(encoding="utf-8"))
    return mod, mod.Contract(conf, env={})


class RevShimEngine:
    """The revision's shim readout over a transformers plug-in; ``post`` decides the post-processing:
    ``post(type, family) -> (T, floor)``. ``last_raw`` holds the raw label log-probabilities of the last request."""

    def __init__(self, plugin, model_dir: Path, model_name: str):
        self.mod, self.c = load_rev_shim(model_dir)
        self.plugin, self.model_name, self.model_dir = plugin, model_name, Path(model_dir)
        self._tok, self.truncated = None, 0
        self.label_ids = self._label_ids()
        self.c.label_ids = self.label_ids
        self.post = lambda typ, fam=None: (self.c.temperature_by_type[typ], self.c.noul_floor)
        self.last_raw: list[dict] = []

    def tokenize(self, text: str) -> list[int]:
        return list(self.plugin.encode(text, True))

    def _label_ids(self) -> dict[str, int]:
        n = len(self.c.labels)
        probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(n)}}
        text = self.c.question_prompt(self.mod.render_state({"x": 1}, self.c.state_compact), probe)[0]
        base = self.tokenize(text)
        ids, bad = {}, []
        for label in self.c.labels:
            full = self.tokenize(text + " " + label)
            if len(full) != len(base) + 1 or full[:len(base)] != base:
                bad.append(label)
            else:
                ids[label] = full[-1]
        if bad or len(set(ids.values())) != len(ids):
            raise RuntimeError(f"labels not one distinct token at the answer slot: {bad[:8]}")
        return ids

    def truncate_state(self, text: str) -> str:
        """The revision's Shim.truncate_state: over max_state_tokens keep head + tail around a "..." line (vLLM's
        tokenize / detokenize replaced by the snapshot's own tokenizer)."""
        mx = self.c.max_state_tokens
        if not mx or len(text.encode("utf-8", "ignore")) <= mx:
            return text
        ids = self.tokenize(text)
        if len(ids) <= mx:
            return text
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        half = mx // 2
        self.truncated += 1
        return self._tok.decode(ids[:half]) + "\n...\n" + self._tok.decode(ids[-half:])

    def raw(self, request: dict) -> list[dict]:
        import torch

        state_txt = self.truncate_state(self.mod.render_state(request["state"], self.c.state_compact))
        out = []
        for qid, q in request["questions"].items():
            text, names, descs, labels = self.c.question_prompt(state_txt, q, qid)
            ids = self.tokenize(text)
            with torch.inference_mode():
                lg = self.plugin.forward(torch.tensor([ids], dtype=torch.long, device=self.plugin.device), 1)
                lp = torch.log_softmax(lg[0, -1].float(), dim=-1)
            lab = [self.label_ids[l] for l in labels]
            vals = lp.index_select(0, torch.tensor(lab, device=lp.device)).tolist()
            thr = self.mod.noul_threshold(q) if hasattr(self.mod, "noul_threshold") else None
            out.append({"qid": qid, "type": q.get("type", "choice"), "names": list(names), "descs": list(descs),
                        "lp": vals, "threshold": thr, "ntok": len(ids)})
        return out

    def answers(self, raw: list[dict], family: str | None = None) -> dict:
        ans = {}
        for r in raw:
            t, floor = self.post(r["type"], family)
            ans[r["qid"]] = self.mod.format_answer(r["type"], r["names"], r["descs"], r["lp"], t, floor, r["threshold"])
        return ans

    def __call__(self, request: dict) -> dict:
        raw = self.raw(request)
        self.last_raw = raw
        ntok = sum(r["ntok"] for r in raw)
        return {"model": self.model_name, "answers": self.answers(raw, request.get("_family")),
                "usage": {"input_tokens": ntok, "output_tokens": 1, "compute_input_tokens": ntok}}


class SymForkLMEngine:
    """sym-fork rows (heads.symmetric, jrt-h2o-shim format) scored by the LM's continuation log-likelihood: per option
    one forward of prefix + continuation (independent of every other option: reorder / relabel give the same per-option
    inputs), sum of log p over the continuation tokens, log-softmax over options, heads.typed.to_wire (12-digit sym wire,
    content-based ties)."""

    def __init__(self, plugin, template_id: str, model_name: str):
        from decide.shim_format import template_obj

        self.plugin, self.model_name = plugin, model_name
        self.tpl = template_obj(template_id)

    def row(self, request: dict, qid: str):
        from heads.symmetric import SYM_LAYOUT, build_row_sym

        return build_row_sym(request, qid, self.plugin.encode, self.tpl, SYM_LAYOUT)

    def __call__(self, request: dict) -> dict:
        import torch

        from heads.symmetric import row_input_tokens, sym_tie_info
        from heads.typed import to_wire

        answers, ntok, ncomp = {}, 0, 0
        for qid, q in request["questions"].items():
            row = self.row(request, qid)
            scores = []
            for cont in row.continuations:
                ids = list(row.ids) + list(cont)
                with torch.inference_mode():
                    lg = self.plugin.forward(torch.tensor([ids], dtype=torch.long, device=self.plugin.device),
                                             len(cont) + 1)
                    lp = torch.log_softmax(lg[0, :-1].float(), dim=-1)
                    idx = torch.tensor(list(cont), device=lp.device)
                    scores.append(float(lp.gather(1, idx[:, None]).sum()))
                ncomp += len(ids)
            lps = torch.log_softmax(torch.tensor(scores, dtype=torch.float64), dim=-1)
            answers[qid] = to_wire(q, row.option_ids, lps, sym_tie_info(row))
            ntok += row_input_tokens(row)
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": ntok, "output_tokens": 1, "compute_input_tokens": ncomp}}


# ------------------------------------------------------------------------------------------------ temperature fit
def gold_index(rec: dict, raw: dict) -> int | None:
    """Index of the slice record's gold label among the shim's option names (noul: true/false; choice: key; score:
    level index), else None (counted, not fitted)."""
    g = (rec.get("labels") or {}).get(raw["qid"])
    names = raw["names"]
    if raw["type"] == "noul":
        m = {True: "true", False: "false", "yes": "true", "no": "false", "true": "true", "false": "false"}
        g = m.get(g if not isinstance(g, str) else g.lower(), g)
    if isinstance(g, bool):
        return None
    s = str(g)
    if s in names:
        return names.index(s)
    if raw["type"] == "score" and s in [str(d) for d in raw["descs"]]:
        return [str(d) for d in raw["descs"]].index(s)
    return None


def _nll(pairs: list[tuple[list[float], int]], t: float) -> float:
    tot = 0.0
    for lp, g in pairs:
        v = [x / t for x in lp]
        m = max(v)
        lse = m + math.log(sum(math.exp(x - m) for x in v))
        tot += lse - v[g]
    return tot / len(pairs)


def fit_t(pairs: list[tuple[list[float], int]]) -> dict:
    """argmin over a log-uniform grid of T_GRID points in T_RANGE of the mean NLL (pre-floor softmax(lp / T))."""
    lo, hi = math.log(T_RANGE[0]), math.log(T_RANGE[1])
    grid = [math.exp(lo + (hi - lo) * i / (T_GRID - 1)) for i in range(T_GRID)]
    vals = [_nll(pairs, t) for t in grid]
    k = min(range(len(grid)), key=vals.__getitem__)
    return {"T": grid[k], "nll": vals[k], "nll_T1": _nll(pairs, 1.0), "n": len(pairs), "at_bound": k in (0, len(grid) - 1)}


def fit_temperatures(slice_raw: list[dict], slice_recs: dict[str, dict]) -> dict:
    pairs_all, by_type, by_fam, skipped = [], {}, {}, 0
    for row in slice_raw:
        rec = slice_recs[row["id"]]
        fam = rec["aug"]["jev1"]["family8"]
        for r in row["raw"]:
            g = gold_index(rec, r)
            if g is None:
                skipped += 1
                continue
            pairs_all.append((r["lp"], g))
            by_type.setdefault(r["type"], []).append((r["lp"], g))
            by_fam.setdefault(fam, []).append((r["lp"], g))
    out = {"global": fit_t(pairs_all), "skipped_no_gold": skipped, "by_type": {}, "by_family": {}}
    for t in TYPES:
        p = by_type.get(t, [])
        out["by_type"][t] = fit_t(p) if p else {"n": 0, "T": None, "fallback": "global"}
    for f, p in sorted(by_fam.items()):
        out["by_family"][f] = fit_t(p) if len(p) >= MIN_FAMILY_FIT else {"n": len(p), "T": None, "fallback": "global"}
    return out


# ------------------------------------------------------------------------------------------------ run (GPU)
PROGRESS: list[Path] = []  # one line per finished unit (slice row / public item): jobs.common.rate_guard --results


def _tick(tag: str) -> None:
    if PROGRESS:
        with open(PROGRESS[0], "a", encoding="utf-8", newline="\n") as f:
            f.write(tag + "\n")


def _run_items(adapter, items, out: Path, engine=None, raw_out: Path | None = None) -> dict:
    """Serial, one request at a time, warm-up not recorded; the control also keeps each item's raw log-probs."""
    out.mkdir(parents=True, exist_ok=True)
    for it in items[:WARMUP]:
        adapter.answer_item(it)
    recs, raws, t0 = [], [], time.time()
    for it in items:
        rec = adapter.answer_item(it)
        _tick(out.name)
        rec["tier"] = it.tier
        recs.append(rec)
        if engine is not None:
            raws.append({"id": it.id, "family": it.family, "raw": engine.last_raw if rec["status"] == "ok" else None})
    _jsonl(out / "preds.jsonl", recs)
    if raw_out is not None:
        _jsonl(raw_out, raws)
    st: dict[str, int] = {}
    for r in recs:
        st[r["status"]] = st.get(r["status"], 0) + 1
    return {"items": len(recs), "status": st, "wall_s": round(time.time() - t0, 1)}


def cmd_run(a) -> int:
    from decide.backends import hf_readout as HR
    from jevkit.items import load_public
    from jevkit.systemone import SystemOneAdapter

    c = cfg()
    out = Path(a.out)
    (out / "arms").mkdir(parents=True, exist_ok=True)
    PROGRESS[:] = [out / "progress.txt"]
    items = load_public()
    if a.limit:
        items = items[: a.limit]
    meta: dict[str, Any] = {"revision": c["revision"], "items": len(items), "phases": {}}
    t0 = time.time()
    md = Path(a.model_dir)
    conf = json.loads((md / "config.json").read_text(encoding="utf-8"))
    fp32 = (conf.get("head_dtype") or (conf.get("text_config") or {}).get("head_dtype")) == "float32"
    plug = HR.load_hf_causal(md, family="h2o-lightning-4b", device=a.device, dtype="bfloat16", model_id=c["repo"],
                             fp32_head=fp32)
    meta["load_s"] = round(time.time() - t0, 1)
    meta["plugin"] = {"model_id": plug.model_id, "files": plug.files, "notes": plug.notes,
                      "max_context": plug.max_context, "kernel_stack": getattr(plug, "kernel_stack", None)}
    rc = 0

    def save():
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    # 1) the held-out slice: raw label log-probs under the revision's shim (no answers scored, nothing public)
    eng = RevShimEngine(plug, md, c["repo"])
    meta["shim"] = {"temperature_by_type": dict(eng.c.temperature_by_type), "noul_floor": eng.c.noul_floor,
                    "labels": len(eng.c.labels), "system": eng.c.system,
                    "label_ids_sha256": hashlib.sha256(json.dumps(eng.label_ids, sort_keys=True).encode()).hexdigest()}
    sl = json.loads(Path(a.slice).read_text(encoding="utf-8"))
    ids = {i for v in sl["families"].values() for i in v}
    recs = slice_records(Path(a.records), ids)
    t1 = time.time()
    rows, fail = [], 0
    for r in recs:
        try:
            rows.append({"id": r["id"], "raw": eng.raw({"state": r["state"], "questions": r["questions"]})})
        except Exception as e:  # noqa: BLE001 - recorded; a failed slice row is not fitted
            fail += 1
            rows.append({"id": r["id"], "raw": [], "error": f"{type(e).__name__}: {e}"})
        _tick("slice")
    _jsonl(out / "slice_raw.jsonl", rows)
    fits = fit_temperatures(rows, {r["id"]: r for r in recs})
    fits["slice_ids_sha256"] = sl["ids_sha256"]
    fits["slice_failed_rows"] = fail
    (out / "temperature_fit.json").write_text(json.dumps(fits, indent=1), encoding="utf-8")
    meta["phases"]["slice"] = {"rows": len(rows), "failed": fail, "wall_s": round(time.time() - t1, 1)}
    save()

    # 2) the control on public-231 (raw kept for the score-only arms)
    try:
        ad = SystemOneAdapter(eng, kind="request", model_name=c["repo"])
        meta["phases"]["h2o"] = _run_items(ad, items, out / "arms" / "h2o", eng, out / "raw_public.jsonl")
    except Exception as e:  # noqa: BLE001
        meta["phases"]["h2o"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        rc = 4
    save()

    # 3) c_symfork on public-231
    sym = None
    try:
        from decide.shim_format import bind_template, make_spec, _SPECS

        tid = c["symfork_template"]
        try:
            spec = bind_template(tid, md)
            meta["symfork_spec"] = {"id": tid, "pinned_identity": True, "sha256": spec.identity()["sha256"]}
        except ValueError as e:  # the revision's tokenizer files differ from the pinned spec: unpinned probe, recorded
            spec = make_spec(tid, md)
            _SPECS[tid] = spec
            meta["symfork_spec"] = {"id": tid, "pinned_identity": False, "why": str(e),
                                    "sha256": spec.identity()["sha256"]}
        sym = SymForkLMEngine(plug, tid, c["repo"])
        ad = SystemOneAdapter(sym, kind="request", model_name=c["repo"])
        meta["phases"]["c_symfork"] = _run_items(ad, items, out / "arms" / "c_symfork")
    except Exception as e:  # noqa: BLE001
        meta["phases"]["c_symfork"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        rc = 4
    save()

    # 4) G1 probe: reorder / relabel perturbations for c_symfork and the control
    from jobs.sym_eval.perturb_eval import REORDER_RELABEL_OPS, run as perturb_run

    for name, engine, row_fn in (("c_symfork", sym, (lambda req: sym.row(req, "q")) if sym else None),
                                 ("h2o", eng, None)):
        if engine is None:
            continue
        try:
            t2 = time.time()
            rep = perturb_run(engine, items, out / "g1" / name, REORDER_RELABEL_OPS, (0,), row_fn=row_fn,
                              kind="request")
            meta["phases"][f"g1_{name}"] = {"g1_pass": rep["g1_pass"], "rows_all_identical": rep.get("rows_all_identical"),
                                            "wall_s": round(time.time() - t2, 1)}
        except Exception as e:  # noqa: BLE001
            meta["phases"][f"g1_{name}"] = {"error": f"{type(e).__name__}: {e}",
                                            "traceback": traceback.format_exc()[-4000:]}
            rc = 4
        save()
    if a.device.startswith("cuda"):
        import torch

        meta["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        meta["gpu"] = torch.cuda.get_device_name(0)
    meta["wall_s"] = round(time.time() - t0, 1)
    save()
    return rc


# ------------------------------------------------------------------------------------------------ derive (CPU)
def derive(out: Path, model_dir: Path) -> dict:
    """Score-only arms from the control's raw log-probs: a_T, a_Ttype, a_Tfam, b_native, b_shimcal."""
    from jevkit.items import load_public
    from jevkit.systemone import normalize_answer

    mod, c = load_rev_shim(model_dir)
    fits = json.loads((out / "temperature_fit.json").read_text(encoding="utf-8"))
    tj = fits["global"]["T"]
    ttype = {t: (fits["by_type"][t]["T"] or tj) for t in TYPES}
    tfam = {f: v["T"] for f, v in fits["by_family"].items() if v.get("T")}
    floor = c.noul_floor
    posts: dict[str, Callable[[str, str], tuple[float, float]]] = {
        "a_T": lambda typ, fam: (tj, floor),
        "a_Ttype": lambda typ, fam: (ttype[typ], floor),
        "a_Tfam": lambda typ, fam: (tfam.get(fam, tj), floor),
        "b_native": lambda typ, fam: (1.0, 0.0),
    }
    items = {it.id: it for it in load_public()}
    ctrl = _read_jsonl(out / "arms" / "h2o" / "preds.jsonl")
    raw = {r["id"]: r for r in _read_jsonl(out / "raw_public.jsonl")}
    made = {}
    for arm, post in posts.items():
        recs = []
        for rec in ctrl:
            r = dict(rec)
            rw = raw.get(rec["id"])
            if rec.get("status") == "ok" and rw and rw.get("raw"):
                a = {}
                for q in rw["raw"]:
                    t, fl = post(q["type"], rw.get("family"))
                    a = mod.format_answer(q["type"], q["names"], q["descs"], q["lp"], t, fl, q["threshold"])
                r["answer"] = normalize_answer(items[rec["id"]].question, a)
            r["derived_from"] = "h2o"
            recs.append(r)
        (out / "arms" / arm).mkdir(parents=True, exist_ok=True)
        _jsonl(out / "arms" / arm / "preds.jsonl", recs)
        made[arm] = len(recs)
    # b_shimcal: shim_calibrate (jobs.m2_first_cell.screen) semantics with this revision's shim: log of the native
    # probabilities, then format_answer with the shim's per-type temperature and floor
    lg = lambda p: math.log(max(float(p), 1e-12))  # noqa: E731
    recs = []
    for rec in _read_jsonl(out / "arms" / "b_native" / "preds.jsonl"):
        r = dict(rec)
        a, it = r.get("answer"), items.get(r["id"])
        if r.get("status") == "ok" and isinstance(a, dict) and it is not None:
            q = it.question
            t = c.temperature_by_type[q["type"]]
            if q["type"] == "noul":
                py = float(a["noul"])
                r["answer"] = mod.format_answer("noul", ["true", "false"], ["", ""], [lg(py), lg(1 - py)], t, floor)
            else:
                names = list(a["probabilities"])
                descs = [str(x) for x in (q.get("criteria") or [])] if q["type"] == "score" else names
                r["answer"] = normalize_answer(q, mod.format_answer(q["type"], names, descs,
                                                                    [lg(a["probabilities"][n]) for n in names], t, floor))
        r["derived_from"] = "b_native"
        recs.append(r)
    (out / "arms" / "b_shimcal").mkdir(parents=True, exist_ok=True)
    _jsonl(out / "arms" / "b_shimcal" / "preds.jsonl", recs)
    made["b_shimcal"] = len(recs)
    doc = {"T_J": tj, "T_by_type": ttype, "T_by_family": tfam, "noul_floor": floor, "arms": made,
           "shim_temperature_by_type": dict(c.temperature_by_type)}
    (out / "derive.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------------------ summarize (CPU)
ARMS = ("h2o", "a_T", "a_Ttype", "a_Tfam", "b_native", "b_shimcal", "c_symfork")
CONTENDERS = ("a_T", "a_Ttype", "a_Tfam", "b_shimcal")  # c_symfork: G1 probe, not a contender; b_native: informative


def _binom_two_sided(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _top1(t, items) -> list[bool]:
    """host_scan.top1 semantics per item: choice / score argmax == gold; noul P(yes) >= 0.5 <-> yes; invalid wrong."""
    out = []
    for i, it in enumerate([it for it in items if it.expected is not None]):
        p = t.parsed[i]
        if not p.valid or not p.has_probs and it.type != "noul":
            out.append(False)
        elif it.type == "noul":
            out.append(bool((float(t.pyes[i]) >= 0.5) == bool(t.yes[i])))
        else:
            out.append(bool(t.top_correct[i]))
    return out


def _choice_nll(t, items) -> list[float | None]:
    out = []
    for i, it in enumerate([it for it in items if it.expected is not None]):
        p = t.parsed[i]
        if it.type != "choice":
            out.append(None)
        elif p.valid and p.has_probs:
            out.append(-math.log(max(float(p.probs.get(str(it.expected), 0.0)), 1e-12)))
        else:
            out.append(None)
    return out


def summarize(root: Path) -> dict:
    import numpy as np

    from jevkit import scoring as JS
    from jevkit import spec as JSPEC
    from jevkit.items import load_public
    from jevkit.systemone import preds_to_answers

    items = load_public()
    scored = [it for it in items if it.expected is not None]
    tables, sc, reports, rows = {}, {}, {}, {}
    for arm in ARMS:
        pp, rp = root / "arms" / arm / "preds.jsonl", root / "arms" / arm / "report.json"
        if not pp.exists() or not rp.exists():
            rows[arm] = {"state": "missing"}
            continue
        recs = _read_jsonl(pp)
        rep = json.loads(rp.read_text(encoding="utf-8"))
        reports[arm] = rep
        tables[arm] = JS.build_table(items, preds_to_answers(recs))
        sc[arm] = (rep["axes"]["speed"], rep["axes"]["cost"])
        rows[arm] = {"n_preds": len(recs), "complete": len(recs) >= len(items), "axes": rep["axes"],
                     "composite": rep["scores"], "capability": rep["capability"], "ci95": rep.get("ci95"),
                     "speed": {k: rep["speed"].get(k) for k in ("p50_s_raw", "p95_s_raw", "p50_s_adjusted",
                                                               "p95_s_adjusted")},
                     "cost": {k: rep["cost"].get(k) for k in ("usd_per_1000", "mean_input_tokens")}}
    if "h2o" not in tables:
        return {"job": "jev_host_scan_h2o_eval", "verdict": None, "error": "control arm missing", "rows": rows}
    names = list(tables)
    ref = tables["h2o"]
    b, seed = JSPEC.BOOTSTRAP_B, JSPEC.BOOTSTRAP_SEED
    rng = np.random.default_rng(seed)  # the SAME draws for every arm (as jevkit.scoring.bootstrap)
    st = JS.strata(ref)
    tix = {t: np.flatnonzero(ref.typ == JS.TYPE_IX[t]) for t in TYPES}
    keys = ("I", "C", "A")
    draws = {nm: {k: [] for k in keys} for nm in names}
    tdraws = {nm: {t: {"I": [], "C": []} for t in TYPES} for nm in names}
    for _ in range(b):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        for nm in names:
            r = JS.aggregate(tables[nm], idx, detail=False)
            i, c = r["intelligence"]["I"], r["calibration"]["score"]
            s, k = sc[nm]
            draws[nm]["I"].append(i)
            draws[nm]["C"].append(c)
            draws[nm]["A"].append(JS.composite(i, c, s, k, "A"))
            for t in TYPES:
                sub = idx[np.isin(idx, tix[t])]
                rt = JS.aggregate(tables[nm], sub, detail=False) if len(sub) else None
                tdraws[nm][t]["I"].append(rt["intelligence"]["I"] if rt else None)
                tdraws[nm][t]["C"].append(rt["calibration"]["score"] if rt else None)

    def ci(v):
        v = [x for x in v if x is not None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

    def dif(a, bb):
        return [x - y for x, y in zip(a, bb) if x is not None and y is not None]

    qs = (1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)
    top = {nm: _top1(tables[nm], items) for nm in names}
    nll = {nm: _choice_nll(tables[nm], items) for nm in names}
    for nm in names:
        a_ = [x for x in draws[nm]["A"] if x is not None]
        rows[nm]["composite_A_quantiles"] = {str(q): float(np.percentile(a_, q)) for q in qs} if a_ else None
        rows[nm]["frac_resamples_I_lt_50"] = float(np.mean([x < 50 for x in draws[nm]["I"] if x is not None]))
        rows[nm]["top1"] = {"correct": int(sum(top[nm])), "n": len(top[nm])}
        v = [x for x in nll[nm] if x is not None]
        rows[nm]["mean_nll_choice"] = float(np.mean(v)) if v else None
        # per-TYPE breakdown (point values on the full item set + bootstrap CIs on the shared draws)
        bt = {}
        for t in TYPES:
            m = tix[t]
            r = JS.aggregate(tables[nm], m, detail=False)
            bt[t] = {"n": int(len(m)), "top1": int(sum(top[nm][j] for j in m)), "I": r["intelligence"]["I"],
                     "C": r["calibration"]["score"], "I_ci95": ci(tdraws[nm][t]["I"]), "C_ci95": ci(tdraws[nm][t]["C"])}
        rows[nm]["by_type"] = bt
    paired = {}
    for nm in names:
        if nm == "h2o":
            continue
        d_a = dif(draws[nm]["A"], draws["h2o"]["A"])
        bb = sum(1 for x, y in zip(top["h2o"], top[nm]) if x and not y)
        cc = sum(1 for x, y in zip(top["h2o"], top[nm]) if y and not x)
        pa, pc = [x for x in nll["h2o"]], [x for x in nll[nm]]
        both = [(x, y) for x, y in zip(pa, pc) if x is not None and y is not None]
        lo_hi = ci(d_a)
        pt = {}
        for t in TYPES:
            pt[t] = {"dI_ci95": ci(dif(tdraws[nm][t]["I"], tdraws["h2o"][t]["I"])),
                     "dC_ci95": ci(dif(tdraws[nm][t]["C"], tdraws["h2o"][t]["C"])),
                     "d_top1": int(sum(top[nm][j] for j in tix[t]) - sum(top["h2o"][j] for j in tix[t]))}
        paired[f"{nm} - h2o"] = {
            "contender": nm in CONTENDERS,
            "composite_A_diff": (rows[nm]["composite"]["A"] - rows["h2o"]["composite"]["A"])
            if rows[nm]["composite"].get("A") is not None and rows["h2o"]["composite"].get("A") is not None else None,
            "composite_A_diff_ci95": lo_hi,
            "primary_pass": bool(lo_hi and lo_hi[0] > 0) if nm in CONTENDERS else None,
            "dI_ci95": ci(dif(draws[nm]["I"], draws["h2o"]["I"])), "dC_ci95": ci(dif(draws[nm]["C"], draws["h2o"]["C"])),
            "dS": sc[nm][0] - sc["h2o"][0] if sc[nm][0] is not None and sc["h2o"][0] is not None else None,
            "dK": sc[nm][1] - sc["h2o"][1] if sc[nm][1] is not None and sc["h2o"][1] is not None else None,
            "d_top1": {"count": cc - bb, "acc": (cc - bb) / len(top[nm]), "h2o_only_right": bb, "arm_only_right": cc,
                       "mcnemar_exact_p": _binom_two_sided(bb, cc)},
            "d_mean_nll_choice": float(np.mean([y - x for x, y in both])) if both else None,
            "by_type": pt}
    # cross-check: jevkit's own paired bootstrap on the same tables (identical draws -> identical composite CIs)
    jk = JS.bootstrap(tables, sc, b=b, seed=seed)
    g1 = {}
    for nm in ("c_symfork", "h2o"):
        p = root / "g1" / nm / "perturb_report.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            g1[nm] = {"g1_pass": r["g1_pass"], "rows_all_identical": r.get("rows_all_identical"),
                      "per_operator": {k: {x: v.get(x) for x in ("n", "flips", "max_abs_dp", "invalid")}
                                       for k, v in r["flip_report"]["per_operator"].items()},
                      "duplicate_tie_items": r.get("duplicate_tie_items")}
    return {"job": "jev_host_scan_h2o_eval", "verdict": None, "measurement_only": False,
            "prereg": "EV-PREREG-H2O-HOST (+ EV-PREREG-COMPOSITE-LOCAL, EV-PREREG-SECONDARY-AXES, EV-PREREG-AREA-TEMP/a)",
            "label": "H2O-Lightning-4B (rev %s) + SSDM (<arm>)" % cfg()["revision"][:8],
            "caveat": "open public items only; the board's sealed split may rank differently (open->board I gap "
                      "measured on H2O: -17.6)",
            "bootstrap": {"B": b, "seed": seed, "paired": True}, "rows": rows, "paired_vs_h2o": paired,
            "jevkit_bootstrap_pairs": jk.get("_pairs"), "g1": g1,
            "temperature_fit": json.loads((root / "temperature_fit.json").read_text(encoding="utf-8"))
            if (root / "temperature_fit.json").exists() else None,
            "derive": json.loads((root / "derive.json").read_text(encoding="utf-8"))
            if (root / "derive.json").exists() else None,
            "notes": {"b_shimcal": "algebraically identical to the control on H2O (shim post-processing of the shim's "
                                   "own raw reading); a consistency check, not an independent arm",
                      "c_symfork": "G1 invariance probe (LM continuation likelihood over sym-fork rows, no head); "
                                   "Composite reported, NOT a contender"}}


def valid(decision: dict) -> bool:
    """finalize.py validator: the control and at least one contender scored and paired."""
    try:
        return bool(decision.get("job") == "jev_host_scan_h2o_eval" and decision["rows"]["h2o"].get("composite")
                    and any(k.split(" - ")[0] in CONTENDERS for k in decision["paired_vs_h2o"]))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.jev_host_scan.h2o_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice")
    s.add_argument("--records", required=True)
    s.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--model-dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--slice", required=True)
    r.add_argument("--records", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=0, help="CPU tests only: the first N public items")
    d = sub.add_parser("derive")
    d.add_argument("--out", required=True)
    d.add_argument("--model-dir", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "slice":
        c = cfg()["slice"]
        sl = select_slice(Path(a.records), c["salt"], c["per_family"])
        Path(a.out).write_text(json.dumps(sl, indent=1), encoding="utf-8")
        ok = sl["ids_sha256"] == c["ids_sha256"] and sl["records_uncompressed_sha256"] == c["records_uncompressed_sha256"]
        print(json.dumps({"ids_sha256": sl["ids_sha256"], "n": sl["n"], "matches_config": ok}))
        return 0 if ok else 10
    if a.cmd == "verify":
        try:
            print(json.dumps(verify(Path(a.model_dir))))
        except Exception as e:  # noqa: BLE001
            print(f"VERIFY FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return 10
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "derive":
        print(json.dumps(derive(Path(a.out), Path(a.model_dir))))
        return 0
    if a.cmd == "summarize":
        dec = summarize(Path(a.root))
        Path(a.out).write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: {x: v.get(x) for x in ("composite_A_diff", "composite_A_diff_ci95", "primary_pass")}
                          for k, v in (dec.get("paired_vs_h2o") or {}).items()}, default=str))
        return 0
    return 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
