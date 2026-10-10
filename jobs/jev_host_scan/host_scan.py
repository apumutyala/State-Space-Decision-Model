"""jobs/jev_host_scan: score candidate decision-model hosts on the 231 public JevBench items through jevkit.

    python -m jobs.jev_host_scan.host_scan plan                                 # candidates, one line each (shell)
    python -m jobs.jev_host_scan.host_scan verify --cand ID --model-dir SNAP    # pinned sha256 + code pins (exit 10)
    python -m jobs.jev_host_scan.host_scan run --cand ID --model-dir SNAP --out DIR [--device cuda] [--limit N]
    python -m jobs.jev_host_scan.host_scan price --cand ID --out price.json
    python -m jobs.jev_host_scan.host_scan summarize --root DIR --out decision.json

One model load per candidate (one process; the shell wraps it in jobs.common.rate_guard with the candidate's cap);
its readout arms run one after another, SERIALLY, one request at a time (the JevBench timing protocol), each arm's
prediction records appended and flushed per item to DIR/<arm>/preds.jsonl, so a cap kill keeps every finished item.
Readouts (candidates.json "arms"):

  h2o_shim  H2O-Lightning-4B v1.1's own shim (third_party/h2o_lightning_shim, vendored unmodified): its prompt, label
            table, temperature 0.8 and yes/no floor 0.801; the vLLM call replaced by one transformers forward that reads
            the label tokens' log-probabilities at the answer slot (lm head in fp32 when config.json says
            head_dtype float32, as vLLM does). On another checkpoint (stock Qwen3.5-4B) = that model under H2O's readout.
  m0        decide.Decider over HFReadoutBackend(jrt-chatml-v1) WITHOUT the prefix fork: sum log p(' ' + id + '\\n')
            per option, one forward per (question, option), <= 4 same-length rows per call (job-5cmat's stock readout).
  v3w       decide.Decider over decide.backends.v3w_nofork (make_v3w(backbone_id): the model's own chat template, its
            single-token code table, word ids; RWKV7: the legacy G1 v3w byte for byte), one forward per question.
  d1        LiquidAI d1-3B's documented ``system_one(state, questions)`` (third_party/d1_3b, vendored unmodified).

Every arm is uncalibrated: no temperature or rule is fitted on these items (EVAL-ONLY). Usage per record:
input_tokens = the decision's longest single model input (the prompt read once, as a prefix-cached server bills it),
output_tokens = 1 (the board's price basis: one output token per decision), compute_input_tokens = all positions run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "jobs/jev_host_scan/candidates.json"
#: H2O-Lightning-4B v1.1 model card ("Identity check for an evaluator", H100 SXM, JevBench commit bb05a33, the same
#: 231 public items): 205 / 231 top-1 correct, easy 48 / standard 71 / hard 86; near-ties GPU numerics can flip
H2O_IDENTITY = {"correct": 205, "by_tier": {"easy": 48, "standard": 71, "hard": 86},
                "near_ties": ["hard-opus-a-temporal_numeric-07", "hard-opus-a-temporal_numeric-12",
                              "hard-opus-c-temporal_numeric-04", "hard-sol-b-temporal_numeric-03",
                              "hard-opus-b-multi_hop-03", "hard-sol-c-judge_hard-13"],
                "wrong_on_card": ["hard-opus-a-temporal_numeric-07", "hard-opus-a-temporal_numeric-12",
                                  "hard-opus-c-temporal_numeric-04", "hard-sol-b-temporal_numeric-03"],
                "tolerance": 3}
WARMUP = 2


def load_candidates() -> dict:
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def candidate(cid: str) -> dict:
    for c in load_candidates()["candidates"]:
        if c["id"] == cid:
            return c
    raise SystemExit(f"unknown candidate {cid!r}")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ identities
def verify(c: dict, model_dir: Path) -> dict:
    """sha256 of every pinned file (snapshot_pin) + the vendored-code pins of the candidate's loader."""
    lines = [ln.split(None, 1) for ln in (ROOT / c["snapshot_pin"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad = {}
    for sha, name in lines:
        p = model_dir / name.strip()
        got = _sha(p.resolve()) if p.exists() else "missing"
        if got != sha:
            bad[name.strip()] = got
    if bad:
        raise RuntimeError(f"{c['id']}: snapshot files differ from {c['snapshot_pin']}: {bad}")
    out = {"files": len(lines)}
    if c["loader"] == "rwkv7_hf":
        import third_party.rwkv7_hf as V

        out["code"] = V.verify_snapshot_code(model_dir, c["repo"])
    elif c["loader"] == "d1":
        import third_party.d1_3b as D

        out["code"] = D.verify_snapshot_code(model_dir)
    if any(a["readout"] == "h2o_shim" for a in c["arms"]) and c["repo"] == "h2oai/h2o-lightning-4b":
        import third_party.h2o_lightning_shim as H

        out["shim"] = H.verify_snapshot(model_dir)
    return out


def tokenizers_root(c: dict, model_dir: Path) -> Path:
    """A tokenizers root holding the snapshot under the v3w registry's directory name, so make_v3w reads the
    snapshot's own (sha-verified) tokenizer files and its identity equals the CPU-pinned one."""
    from decide.v3w_generic import BACKBONES

    root = Path(tempfile.mkdtemp(prefix="v3w-tok-"))
    (root / BACKBONES[c["backbone_id"]].tokenizer_dir).symlink_to(Path(model_dir).resolve(), target_is_directory=True)
    return root


# ------------------------------------------------------------------------------------------------ engines
class BilledDecider:
    """A decide.Decider whose ok responses carry the scan's usage: input_tokens = longest single model input
    (Plan.max_call_input_tokens), output_tokens = 1, compute_input_tokens = every position the backend ran."""

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


class ShimEngine:
    """H2O-Lightning-4B v1.1's shim readout over a transformers plug-in (``plugin.encode`` / ``plugin.forward``)."""

    def __init__(self, plugin, model_name: str = "h2oai/h2o-lightning-4b"):
        import third_party.h2o_lightning_shim as H

        self.mod, self.c = H.load_contract()
        self.plugin, self.model_name = plugin, model_name
        self.label_ids = self._label_ids()
        self.c.label_ids = self.label_ids

    def tokenize(self, text: str) -> list[int]:
        return list(self.plugin.encode(text, True))  # = vLLM /tokenize with add_special_tokens false

    def _label_ids(self) -> dict[str, int]:
        """Shim.check_backend's probe: every label is ONE distinct token after " " at the answer slot."""
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
        mx = self.c.max_state_tokens
        if not mx or len(text.encode("utf-8", "ignore")) <= mx:
            return text
        ids = self.tokenize(text)
        if len(ids) <= mx:
            return text
        raise RuntimeError("state over max_state_tokens: the shim's vLLM detokenize path is not replicated")

    def __call__(self, request: dict) -> dict:
        import torch

        state_txt = self.truncate_state(self.mod.render_state(request["state"], self.c.state_compact))
        answers, ntok = {}, 0
        for qid, q in request["questions"].items():
            text, names, descs, labels = self.c.question_prompt(state_txt, q, qid)
            ids = self.tokenize(text)
            with torch.inference_mode():
                lg = self.plugin.forward(torch.tensor([ids], dtype=torch.long, device=self.plugin.device), 1)
                lp = torch.log_softmax(lg[0, -1].float(), dim=-1)
            lab = [self.label_ids[l] for l in labels]
            vals = lp.index_select(0, torch.tensor(lab, device=lp.device)).tolist()
            thr = None
            if q.get("type", "choice") == "noul":
                t = q.get("threshold", q.get("noul_threshold"))
                if isinstance(t, (int, float)) and 0.0 <= float(t) <= 1.0:
                    thr = float(t)
            answers[qid] = self.mod.format_answer(q.get("type", "choice"), names, descs, vals, self.c.temperature,
                                                  self.c.noul_floor, thr)
            ntok += len(ids)
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": ntok, "output_tokens": 1, "compute_input_tokens": ntok}}


class D1Engine:
    """d1-3B's documented readout: ``model.system_one(state, questions)``."""

    def __init__(self, model):
        self.model = model

    def __call__(self, request: dict) -> dict:
        out = self.model.system_one(request["state"], request["questions"])
        n = int((out.get("usage") or {}).get("input_tokens") or 0)
        return {"model": "LiquidAI/d1-3B", "answers": out["answers"],
                "usage": {"input_tokens": n, "output_tokens": 1, "compute_input_tokens": n}}


def load_model(c: dict, model_dir: Path, device: str, dtype: str = "bfloat16"):
    """(kind, handle): one load per candidate, shared by its arms."""
    if c["loader"] == "d1":
        import third_party.d1_3b as D

        return "d1", D.load_d1(model_dir, device=device, dtype=dtype)
    from decide.backends import hf_readout as HR

    if c["loader"] == "rwkv7_hf":
        import third_party.rwkv7_hf as V
        from decide.v3w_generic import BACKBONES

        codeless = c["repo"] in V.CODELESS_CONFIG_SHA256
        return "plugin", HR.load_rwkv7_hf(model_dir, device=device, dtype=dtype, repo=c["repo"],
                                         wkv="chunked" if codeless else None,
                                         trained_ctx=BACKBONES[c["backbone_id"]].max_context)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    fp32 = (cfg.get("head_dtype") or (cfg.get("text_config") or {}).get("head_dtype")) == "float32"
    return "plugin", HR.load_hf_causal(model_dir, family=c["id"], device=device, dtype=dtype, model_id=c["repo"],
                                       fp32_head=fp32)


def build_arm(c: dict, arm: dict, kind: str, handle, model_dir: Path, tok_root: Path | None = None):
    """(SystemOneAdapter, identity dict) for one readout arm."""
    from jevkit.systemone import SystemOneAdapter

    r = arm["readout"]
    if r == "d1":
        if kind != "d1":
            raise ValueError("d1 readout needs the d1 loader")
        return SystemOneAdapter(D1Engine(handle), kind="request", model_name=c["repo"]), \
            {"readout": "d1.system_one", "vendored": "third_party/d1_3b"}
    plug = handle
    if r == "h2o_shim":
        eng = ShimEngine(plug, model_name=c["repo"])
        import third_party.h2o_lightning_shim as H

        return SystemOneAdapter(eng, kind="request", model_name=c["repo"]), \
            {"readout": "h2o_shim", "shim": f"{H.REPO}@{H.TAG}", "fp32_head": plug.notes.get("fp32_head"),
             "temperature": eng.c.temperature, "noul_floor": eng.c.noul_floor,
             "label_ids_sha256": hashlib.sha256(json.dumps(eng.label_ids, sort_keys=True).encode()).hexdigest()}
    from decide import Decider

    if r == "m0":
        from decide.backends.hf_readout import HFReadoutBackend

        backend = HFReadoutBackend(plug, "jrt-chatml-v1", max_batch_rows=4, fork=False)
    elif r == "v3w":
        from decide.backends.v3w_nofork import V3WNoForkBackend
        from decide.v3w_generic import make_v3w

        v3w = make_v3w(c["backbone_id"], tokenizers_root=tok_root or tokenizers_root(c, model_dir))
        got = v3w.identity()["sha256"]
        if c.get("v3w_sha256") and got != c["v3w_sha256"]:
            raise RuntimeError(f"v3w identity {got} != pinned {c['v3w_sha256']}")
        backend = V3WNoForkBackend(plug, v3w)
    else:
        raise ValueError(f"unknown readout {r!r}")
    dec = BilledDecider(Decider(backend, enable_v0_1_deltas=True, backend_max_input_tokens=plug.max_context))
    rev = dec.revision()
    return SystemOneAdapter(dec, kind="decider", model_name=c["repo"]), \
        {"readout": r, "revision_id": rev["revision_id"], "prompt_template": rev.get("prompt_template"),
         "scoring_path": (rev.get("inference_config") or {}).get("scoring_path")}


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            pass
    return out


def run_arm(adapter, items, out_dir: Path, warmup: int = WARMUP) -> dict:
    """Serial, one request at a time; warm-up requests (the first ``warmup`` items, re-sent) are not recorded."""
    import time as _t

    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / "preds.jsonl"
    # B-PREREG-FLIP: perturbed items (ids "<base>~<op>~<seed>") go to preds_pert.jsonl, so preds.jsonl stays the
    # 231 public items that `jevkit score --public` reads, unchanged
    preds_pert = out_dir / "preds_pert.jsonl"
    done = _done_ids(preds) | _done_ids(preds_pert)
    todo = [it for it in items if it.id not in done]
    t0 = _t.time()
    for it in items[:warmup]:
        adapter.answer_item(it)
    st = {"ok": 0, "error": 0, "unsupported": 0, "exhausted": 0}
    with open(preds, "a", encoding="utf-8", newline="\n") as fh, \
            open(preds_pert, "a", encoding="utf-8", newline="\n") as fp:
        for it in todo:
            rec = adapter.answer_item(it)
            rec["tier"] = it.tier
            st[rec["status"]] = st.get(rec["status"], 0) + 1
            f = fp if "~" in it.id else fh
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return {"items": len(todo), "resumed_from": len(done), "status": st, "wall_s": round(_t.time() - t0, 1)}


def load_items(items_file: str | None):
    """The 231 public items, then (B-PREREG-FLIP) the perturbed rows of ``items_file`` in file order."""
    from jevkit.items import Item, load_public

    items = load_public()
    if items_file:
        base = {it.id for it in items}
        for line in Path(items_file).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            meta = d.pop("perturbation")
            if meta["base_id"] not in base or "~" not in d["id"]:
                raise ValueError(f"perturbed row {d.get('id')!r} does not derive from a public item")
            items.append(Item.from_dict(d))
    return items


def cmd_run(a) -> int:
    c = candidate(a.cand)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(a.items_file)
    if a.limit:
        items = items[: a.limit]
    meta: dict[str, Any] = {"candidate": c, "device": a.device, "items": len(items), "arms": {}}
    t0 = time.time()
    try:
        kind, handle = load_model(c, Path(a.model_dir), a.device)
    except Exception as e:  # noqa: BLE001 - recorded; the shell moves on to the next candidate
        meta["load_error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()[-4000:]
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        print(f"LOAD FAILED {c['id']}: {meta['load_error']}", flush=True)
        return 3
    meta["load_s"] = round(time.time() - t0, 1)
    if kind == "plugin":
        meta["plugin"] = {"model_id": handle.model_id, "files": handle.files, "notes": handle.notes,
                          "max_context": handle.max_context, "kernel_stack": handle.kernel_stack}
    rc = 0
    for arm in c["arms"]:
        rec: dict[str, Any] = {}
        try:
            adapter, ident = build_arm(c, arm, kind, handle, Path(a.model_dir),
                                       Path(a.tokenizers_root) if a.tokenizers_root else None)
            rec["identity"] = ident
            rec.update(run_arm(adapter, items, out / arm["arm"]))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            rc = 4
            print(f"ARM FAILED {c['id']}/{arm['arm']}: {rec['error']}", flush=True)
        meta["arms"][arm["arm"]] = rec
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if a.device.startswith("cuda"):
        import torch

        meta["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        meta["gpu"] = torch.cuda.get_device_name(0)
    meta["wall_s"] = round(time.time() - t0, 1)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return rc


# ------------------------------------------------------------------------------------------------ scoring
def price_doc(c: dict) -> dict:
    """The declared price (jevkit --price JSON): the candidate's price class, ``model`` = the priced model."""
    pc = dict(load_candidates()["price_classes"][c["price_class"]])
    pc.setdefault("model", c["repo"])
    return {"price": pc, "price_class": c["price_class"]}


def top1(records: list[dict], items) -> dict:
    """Top-1 correctness per tier (the model cards' "correct" count): choice / score argmax == gold, noul P(yes)
    >= 0.5 <-> yes; a missing / invalid answer is wrong."""
    from jevkit.scoring import build_table

    by = {r["id"]: (r.get("answer") if r.get("status", "ok") == "ok" else None) for r in records}
    t = build_table(items, by)
    out = {"correct": 0, "n": 0, "by_tier": {}, "wrong": []}
    for i, it in enumerate([it for it in items if it.expected is not None]):
        ok = bool(t.correct[i])
        out["n"] += 1
        out["correct"] += ok
        bt = out["by_tier"].setdefault(it.tier, [0, 0])
        bt[0] += ok
        bt[1] += 1
        if not ok:
            out["wrong"].append(it.id)
    return out


def h2o_parity(acc: dict) -> dict:
    ref = H2O_IDENTITY
    wrong = set(acc["wrong"])
    extra_wrong = sorted(wrong - set(ref["wrong_on_card"]) - set(ref["near_ties"]))
    fixed = sorted(set(ref["wrong_on_card"]) - wrong)
    return {"reference": ref, "correct": acc["correct"], "by_tier": {k: v[0] for k, v in acc["by_tier"].items()},
            "diff": acc["correct"] - ref["correct"], "wrong_outside_card_and_near_ties": extra_wrong,
            "card_wrong_now_right": fixed,
            "pass": abs(acc["correct"] - ref["correct"]) <= ref["tolerance"]}


def summarize(root: Path) -> dict:
    """decision.json: per candidate x arm, the jevkit report's axes / composite / capability (+ CIs), latency, tokens,
    cost, top-1 counts; the H2O identity check; rankings. Measurement only (no verdict)."""
    from jevkit.items import load_public

    items = load_public()
    cands = load_candidates()
    rows, status = [], {}
    for c in cands["candidates"]:
        d = root / "cand" / c["id"]
        st = json.loads((d / "status.json").read_text(encoding="utf-8")) if (d / "status.json").exists() else \
            {"state": "not_run"}
        status[c["id"]] = st
        for arm in c["arms"]:
            rp = d / arm["arm"] / "report.json"
            pp = d / arm["arm"] / "preds.jsonl"
            if not pp.exists():
                continue
            recs = [json.loads(x) for x in pp.read_text(encoding="utf-8").splitlines() if x.strip()]
            acc = top1(recs, items)
            row = {"candidate": c["id"], "repo": c["repo"], "revision": c["revision"], "arm": arm["arm"],
                   "group": c["group"], "params_b": c["params_b"], "price_class": c["price_class"],
                   "n_preds": len(recs), "complete": len(recs) >= len(items),
                   "ok": sum(1 for r in recs if r.get("status") == "ok"),
                   "top1": {"correct": acc["correct"], "n": acc["n"],
                            "by_tier": {k: f"{v[0]}/{v[1]}" for k, v in acc["by_tier"].items()}}}
            if rp.exists():
                rep = json.loads(rp.read_text(encoding="utf-8"))
                ci = rep.get("ci95") or {}
                row.update({
                    "axes": rep["axes"], "composite": rep["scores"], "capability": rep["capability"],
                    "ci95": {k: ci.get(k) for k in ("composite", "capability", "I", "C")} if ci else None,
                    "speed": {k: (rep.get("speed") or {}).get(k) for k in ("p50_s_raw", "p95_s_raw", "p50_s_adjusted",
                                                                            "p95_s_adjusted", "n", "source")},
                    "cost": {k: (rep.get("cost") or {}).get(k) for k in ("usd_per_1000", "mean_input_tokens",
                                                                          "kind", "price_rule")},
                    "eligibility": rep.get("eligibility"), "intelligence_basis": (rep.get("intelligence") or {}).get("basis")})
            if c["id"] == "h2o-lightning-4b" and arm["readout"] == "h2o_shim":
                row["h2o_identity_check"] = h2o_parity(acc)
            rows.append(row)
    scored = [r for r in rows if r.get("composite") and r["complete"]]

    def key(m):
        return lambda r: (r["composite"].get("A") if m == "A" else r.get("capability")) or -1

    leader = {"key": "h2o-lightning-4b", "composite_A": 72.5232271521435, "I": 60.03839906196252,
              "C": 89.95399367673842, "S": 92.61084474720658, "K": 60.29882443070202, "board": "JevBench v1.6.1"}
    return {"job": "jev_host_scan", "verdict": None, "measurement_only": True, "items": len(items),
            "basis": "open-only, the 231 public JevBench v1.2 items (b6b8fff7); not the v1.6.1 open set P300, no "
                     "sealed side: local numbers measure the board's constructs on a different draw (jevkit/README.md)",
            "board_leader": leader, "status": status, "rows": rows,
            "ranking_composite_A": [(r["candidate"], r["arm"], round(r["composite"]["A"], 2)) for r in
                                    sorted(scored, key=key("A"), reverse=True)],
            "ranking_capability": [(r["candidate"], r["arm"], round(r["capability"], 2)) for r in
                                   sorted(scored, key=key("cap"), reverse=True) if r.get("capability") is not None]}


def valid_scan(decision: dict) -> bool:
    """finalize.py validator: a decision with every candidate accounted for and at least one scored, complete arm."""
    try:
        cands = {c["id"] for c in load_candidates()["candidates"]}
        return (decision.get("job") == "jev_host_scan" and set(decision["status"]) == cands
                and any(r.get("composite") and r.get("complete") for r in decision["rows"]))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.jev_host_scan.host_scan", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    v = sub.add_parser("verify")
    v.add_argument("--cand", required=True)
    v.add_argument("--model-dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cand", required=True)
    r.add_argument("--model-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=0, help="CPU tests only: the first N items")
    r.add_argument("--tokenizers-root", default=None, help="CPU tests only: v3w tokenizers root")
    r.add_argument("--items-file", default=None, help="B-PREREG-FLIP: jevkit perturb make output, run after the "
                                                      "public items (predictions -> <arm>/preds_pert.jsonl)")
    p = sub.add_parser("price")
    p.add_argument("--cand", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        for c in load_candidates()["candidates"]:
            print(c["id"], c["repo"], c["revision"], c["cap_min"], c["download_cap_min"], len(c["arms"]),
                  c["snapshot_pin"], ",".join(c["allow_patterns"]))
        return 0
    if a.cmd == "verify":
        try:
            print(json.dumps(verify(candidate(a.cand), Path(a.model_dir)), default=str))
        except Exception as e:  # noqa: BLE001
            print(f"VERIFY FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return 10
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "price":
        Path(a.out).write_text(json.dumps(price_doc(candidate(a.cand)), indent=1), encoding="utf-8")
        return 0
    if a.cmd == "summarize":
        dec = summarize(Path(a.root))
        Path(a.out).write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
        print(json.dumps({"ranking_composite_A": dec["ranking_composite_A"][:12]}, default=str))
        return 0
    return 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
