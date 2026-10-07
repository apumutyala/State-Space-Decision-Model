"""Feature-cache driver for one arm (jobs/q9_feature_cache/cache_job.sh runs it once per arm under the live rate guard).

    python -X utf8 -m jobs.q9_feature_cache.cache run --arm stock --config jobs/q9_feature_cache/cache_config.json \
        --model-dir <snapshot> --dev-rows /data/slices/dev-rows.jsonl.gz --out <dir> \
        (--train-dir /data/m2-150k --guard-readiness <readiness.json> | --toy-train-records <jsonl.gz>) \
        [--gdn-ckpt gdn_converted.pt] [--device cuda] [--tok-s X --call-s Y --cap-min M]

Order: load -> build EVERY row of every split (tokenisation only) -> exact volume -> in-job preflight from the MEASURED
rates (exit 15 if the projection is past 95% of the cap) -> numerics self-check (exit 2) -> select, holdout, train.

Training rows come ONLY from ``train.train_set.draw`` (pointer-verified 150k set, cohort-v2 exclusion by id and
content, pinned D1-B exclusions, drawn-record heldout scan) and pass ``SplitGuard.check`` with the verified
content-index guard (``jobs.m2_pilot.guard_ready.load_ready_guard``) BEFORE tokenisation; every drawn id is also checked
against the DEV/TEST manifests by the guard. ``--toy-train-records`` exists for the CPU dry run only (synthetic
records, synthetic-only guard) and is recorded as such in summary.json.

Exit codes: 0 ok; 2 numerics self-check; 5 setup/usage; 10 data identity (train set, DEV rows, gdn sha256);
15 projected too slow (in-job preflight); 27 guard not ready; 28 cohort / train-set leak.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXIT = {"ok": 0, "numerics": 2, "setup": 5, "data": 10, "too_slow": 15, "guard": 27, "leak": 28}
#: the in-job preflight margin (as jobs.common.rate_guard.MARGIN)
MARGIN = 0.95


def _jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dev_requests(dev_rows: str, subset_path: str, arm_e5: bool) -> list[dict]:
    from jobs.common import dev_subset as D

    from .extract import dev_request

    sub = D.load_subset(ROOT / subset_path)
    ids = set(sub["run_ids"])
    out = []
    for r in _jsonl_gz(dev_rows):
        rid = r["_evaluation"]["run_id"]
        if rid in ids:
            req, back = dev_request(r, "e5" if arm_e5 else "stock")
            out.append({"rid": rid, "catalog_id": r["_evaluation"]["catalog_id"], "request": req, "e5_back": back})
    if len(out) != len(ids):
        raise LookupError(f"{subset_path}: matched {len(out)} of {len(ids)} DEV rows")
    return out


def train_records(cfg: dict, a, guard) -> tuple[list[dict], dict]:
    """Drawn training records (draw order) + identity. Real: train.train_set.draw. Dry run: toy records."""
    if a.toy_train_records:
        recs = list(_jsonl_gz(a.toy_train_records))
        return recs, {"train_set": "DRY RUN: synthetic toy records (not the 150k set)", "n_drawn": len(recs)}
    from train.train_set import draw

    by_id, order, ident = draw(cfg, a.train_dir)
    return [by_id[i] for i in order], ident


def train_rows(recs: list[dict], cfg: dict, arm: dict, enc, guard, n_rows: int) -> tuple[list[dict], dict]:
    """Requests of guarded, keep_item-filtered training rows, in draw order, until ``n_rows`` rows."""
    from heads.option_keys import e5_labels, e5_recode
    from train.data import items_from_records
    from train.run_cell import keep_item

    out, kept, dropped = [], 0, {"filter": 0}
    for rec in recs:
        if kept >= n_rows:
            break
        guard.check(rec)  # on the ORIGINAL record (the E5 rendering changes content hashes)
        use = rec
        if arm.get("e5"):
            req, back = e5_recode(rec["request"])
            use = {"id": rec["id"], "request": req, "labels": e5_labels(rec["labels"], back)}
        items = items_from_records([use], enc, layout=cfg["layout"], template=cfg["template"], guard=None)
        keep = [it for it in items if keep_item(it.row, cfg["max_k"], cfg["max_t"])]
        dropped["filter"] += len(items) - len(keep)
        keep = keep[: n_rows - kept]
        if keep:
            out.append({"rid": rec["id"], "rows": [it.row for it in keep], "labels": [it.label for it in keep]})
            kept += len(keep)
    return out, {"rows": kept, "requests": len(out), "dropped": dropped, "short": kept < n_rows}


def plan(units: list[dict], opt_chunk: int, tok_s: float, call_s: float) -> float:
    """Adds ``positions`` / ``calls`` / ``w`` (projected seconds) to each unit; returns the total weight."""
    from .extract import n_chunks_calls

    tot = 0.0
    for u in units:
        if not u["rows"]:
            u.update(positions=0, calls=0, w=0.0)
            continue
        p, c = n_chunks_calls(u["rows"], opt_chunk)
        u.update(positions=p, calls=c, w=p / tok_s + c * call_s)
        tot += u["w"]
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9_feature_cache.cache")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--model-dir", required=True)
    r.add_argument("--dev-rows", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--train-dir", default=None)
    r.add_argument("--guard-readiness", default=None)
    r.add_argument("--toy-train-records", default=None, help="CPU dry run only")
    r.add_argument("--gdn-ckpt", default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--tok-s", type=float, default=None, help="MEASURED prefill rate (job_params measured)")
    r.add_argument("--call-s", type=float, default=None, help="MEASURED per-call cost (job_params measured)")
    r.add_argument("--cap-min", type=float, default=None, help="this arm's phase cap (minutes)")
    r.add_argument("--elapsed-before-min", type=float, default=0.0)
    r.add_argument("--progress", default=None, help="weighted progress file (jobs.common.rate_guard run-weighted)")
    r.add_argument("--plan-only", action="store_true", help="tokenise + volume + preflight, no model")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.arm not in cfg["arms"]:
        print(f"unknown arm {a.arm}", file=sys.stderr)
        return EXIT["setup"]
    arm = cfg["arms"][a.arm]
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"arm": a.arm, "arm_cfg": arm, "feature_version": cfg["feature_version"],
                     "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "dry_run": bool(a.toy_train_records)}

    def done(code: int, why: str = "") -> int:
        summary.update(exit=code, reason=why, minutes=(time.time() - t0) / 60)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        print(json.dumps({"exit": code, "reason": why}), flush=True)
        return code

    # ---------------------------------------------------------------- guard + encoder (no model yet)
    from transformers import AutoTokenizer

    from heads.layout import hf_segment_encoder

    enc = hf_segment_encoder(AutoTokenizer.from_pretrained(a.model_dir))
    if a.toy_train_records:
        from train.guard import SplitGuard

        guard = SplitGuard.synthetic_only("q9_feature_cache dry run: toy records")
    else:
        from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard

        try:
            guard = load_ready_guard(a.guard_readiness)
        except GuardNotReady as e:
            return done(EXIT["guard"], f"guard not ready: {e}")
    summary["guard"] = guard.identity()
    # ---------------------------------------------------------------- rows of every split
    from train.cohort import CohortLeak
    from train.guard import ForbiddenRow
    from train.train_set import TrainSetMismatch

    from .extract import build_rows

    try:
        units: dict[str, list[dict]] = {}
        for split, path in cfg["dev_splits"].items():
            reqs = dev_requests(a.dev_rows, path, bool(arm.get("e5")))
            for u in reqs:
                rows, bad = build_rows(u["request"], enc)
                long = [x.qid for x in rows if len(x.ids) > cfg["dev_max_t"]]
                u["rows"] = [x for x in rows if len(x.ids) <= cfg["dev_max_t"]]
                u["skipped"] = {**bad, **{q: f"T > dev_max_t {cfg['dev_max_t']}" for q in long}}
                u["labels"] = None
            units[split] = reqs
        recs, ident = train_records(cfg, a, guard)
        summary["train_identity"] = ident
        units["train"], summary["train_rows"] = train_rows(recs, cfg, arm, enc, guard, arm["train_rows"])
    except (TrainSetMismatch, LookupError) as e:
        return done(EXIT["data"], f"data identity: {e}")
    except (CohortLeak, ForbiddenRow) as e:
        return done(EXIT["leak"], f"leak: {e}")
    if summary["train_rows"]["short"]:
        return done(EXIT["data"], f"only {summary['train_rows']['rows']} training rows after the filter")
    # ---------------------------------------------------------------- exact volume + in-job preflight
    tok_s, call_s = a.tok_s or 1.0, a.call_s or 0.0
    allu = [u for s in ("select", "holdout", "train") for u in units[s]]
    total_w = plan(allu, cfg["opt_chunk"], tok_s, call_s)
    vol = {s: {"requests": len(units[s]), "questions": sum(len(u["rows"]) for u in units[s]),
               "skipped_questions": sum(len(u.get("skipped") or {}) for u in units[s]),
               "positions": sum(u["positions"] for u in units[s]), "calls": sum(u["calls"] for u in units[s]),
               "projected_min": sum(u["w"] for u in units[s]) / 60} for s in units}
    d, nb = cfg["d_model"], cfg["bank_chunks"]
    for s in units:  # upper-bound output size: (fork, e, lex) per option + 2 question vectors, bf16; one bank/request
        q_bytes = sum((3 * r.n_options + 2) * d * 2 for u in units[s] for r in u["rows"])
        banks, seen = 0, 0
        for u in units[s]:
            banks += s != "train" or seen < cfg["rich_rows"]
            seen += len(u["rows"])
        vol[s]["est_gb"] = round((q_bytes + banks * nb * d * 2) / 1e9, 2)
    summary["est_output_gb"] = round(sum(v["est_gb"] for v in vol.values()), 2)
    summary["volume"] = vol
    summary["preflight"] = {"tok_s": a.tok_s, "call_s": a.call_s, "cap_min": a.cap_min,
                            "elapsed_before_min": a.elapsed_before_min, "projected_min": total_w / 60}
    if a.cap_min is not None and a.tok_s:
        proj = a.elapsed_before_min + (time.time() - t0) / 60 + total_w / 60
        summary["preflight"].update(projected_total_min=proj, ok=proj <= MARGIN * a.cap_min)
        if proj > MARGIN * a.cap_min:
            return done(EXIT["too_slow"], f"projected {proj:.1f} min > {MARGIN} x cap {a.cap_min}")
    if a.plan_only:
        return done(EXIT["ok"], "plan only")
    if a.progress:
        Path(a.progress + ".total.json").write_text(json.dumps({"total_weight": total_w}), encoding="utf-8")
    # ---------------------------------------------------------------- model + numerics self-check
    import torch

    from .extract import ShardWriter, extract_request, load, recompute_check

    try:
        text, w_out, prov = load(a.model_dir, a.device, a.dtype, a.gdn_ckpt, arm.get("gdn_sha256"), arm["layers"])
    except ValueError as e:
        return done(EXIT["data"], f"model/gdn identity: {e}")
    summary["provenance"] = prov
    torch.manual_seed(cfg["seed"])
    sc = cfg["selfcheck"]
    checks = []
    for u in units["select"][: sc["rows"]]:
        if not u["rows"]:
            continue
        feats, _ = extract_request(text, w_out, u["rows"], cfg["opt_chunk"])
        checks.append(recompute_check(text, w_out, u["rows"][0], feats[0], 0))
    summary["selfcheck"] = checks
    if not checks or any(c["fork_rel_diff"] > sc["max_fork_rel_diff"] or c["lm_abs_diff"] > sc["max_lm_abs_diff"]
                         for c in checks):
        return done(EXIT["numerics"], f"self-check failed: {checks}")
    # ---------------------------------------------------------------- extraction
    rich_left = cfg["rich_rows"]
    timing = {}
    for split in ("select", "holdout", "train"):
        ts = time.time()
        wr = ShardWriter(out / split, cfg["shard_requests"], out / f"progress_{split}.jsonl")
        for u in units[split]:
            rec = {"rid": u["rid"], "split": split, "catalog_id": u.get("catalog_id"), "questions": [], "banks": {},
                   "skipped": u.get("skipped") or {}, "e5_back": u.get("e5_back") or {}}
            if u["rows"]:
                bank = cfg["bank_chunks"] if (split != "train" or rich_left > 0) else 0
                qs, banks = extract_request(text, w_out, u["rows"], cfg["opt_chunk"], bank)
                if split == "train":
                    rich_left -= len(qs) if bank else 0
                    for q, y in zip(qs, u["labels"]):
                        q["label"] = int(y)
                back = u.get("e5_back") or {}
                for q in qs:
                    if q["qid"] in back:  # store ORIGINAL option ids (answers are written in them)
                        q["e5_codes"] = list(q["option_ids"])
                        q["option_ids"] = [back[q["qid"]][c] for c in q["option_ids"]]
                rec["questions"], rec["banks"] = qs, banks
            wr.add(rec)
            if a.progress:
                with open(a.progress, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"w": u["w"]}) + "\n")
        wr.flush()
        timing[split] = {"minutes": (time.time() - ts) / 60, "requests": wr.n_requests, "questions": wr.n_questions,
                         "shards": wr.n_shards}
    summary["timing"] = timing
    if torch.cuda.is_available():
        summary["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return done(EXIT["ok"], "ok")


if __name__ == "__main__":
    sys.exit(main())
