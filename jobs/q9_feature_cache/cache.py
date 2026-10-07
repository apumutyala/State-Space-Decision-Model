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
15 projected too slow (in-job preflight); 16 output over max_output_gib (estimate or bytes written);
27 guard not ready; 28 cohort / train-set leak or a record that targets a frozen held-out benchmark.
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
EXIT = {"ok": 0, "numerics": 2, "setup": 5, "data": 10, "too_slow": 15, "too_big": 16, "guard": 27, "leak": 28}
#: the platform packages a job's WHOLE output directory as ONE tar artifact, bounded by max_artifact_bytes = 5 GiB
#: (list_limits: "the object store's single-PUT bound; cannot be raised past it today"). One arm per job, and its output
#: must stay <= cfg["max_output_gib"] (4.5 GiB, 10% under the bound): checked on the exact estimate before the model
#: loads and on the bytes actually written before the job may succeed (exit 16 either way).
GIB = 1 << 30
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


def frozen_family_check(dry_run: bool):
    """``harness.heldout.check_record`` (heldout-draw-v1: refuses a record that targets one of the 9 FROZEN held-out
    benchmarks). Fail-closed on the paid path; only the CPU dry run, whose test env has no Decision Index kit (the harness
    imports it), may run without it, and summary.json records that."""
    try:
        from harness.heldout import check_record
    except ImportError:
        if not dry_run:
            raise
        return None
    return check_record


def train_rows(recs: list[dict], cfg: dict, arm: dict, enc, guard, n_rows: int,
               frozen_check=None) -> tuple[list[dict], dict]:
    """Requests of guarded, keep_item-filtered training rows, in draw order, until ``n_rows`` rows. Every record passes
    ``guard.check`` and (when given) the frozen-family check on its ORIGINAL form before tokenisation."""
    from heads.option_keys import e5_labels, e5_recode
    from train.data import items_from_records
    from train.run_cell import keep_item

    out, kept, dropped = [], 0, {"filter": 0}
    for rec in recs:
        if kept >= n_rows:
            break
        guard.check(rec)  # on the ORIGINAL record (the E5 rendering changes content hashes)
        if frozen_check is not None:
            frozen_check(rec)
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


def estimate_bytes(units: dict, cfg: dict) -> dict:
    """Upper bound of what the arm writes: per question (fork, e) [K, D] + (q_pre, q_end) [D] in bf16 + option token
    ids (int32) + lm_lp / lm_ntok; one bank [<= bank_chunks, D] bf16 per DEV request and per rich training request;
    the lex_vocab table (lm_head rows of every option token, bf16); pickle overhead 2%."""
    from .extract import option_tokens

    d, nb = cfg["d_model"], cfg["bank_chunks"]
    out, union = {}, set()
    for s, us in units.items():
        toks = set()
        q_bytes, banks, seen = 0, 0, 0
        for u in us:
            for r in u["rows"]:
                q_bytes += (2 * r.n_options + 2) * d * 2 + 4 * sum(b - a for a, b in r.option_spans) + 6 * r.n_options
            banks += bool(u["rows"]) and (s != "train" or seen < cfg["rich_rows"])
            seen += len(u["rows"])
            toks |= option_tokens(u["rows"])
        out[s] = int(1.02 * (q_bytes + banks * nb * d * 2))
        out[f"lex_{s}"] = len(toks) * (d * 2 + 8)  # this split alone (an upper bound for any part holding it)
        union |= toks
    out["lex_vocab"] = len(union) * (d * 2 + 8)  # ONE table per part: the union over the part's splits
    out["lex_vocab_tokens"] = len(union)
    out["total"] = sum(out[s] for s in units) + out["lex_vocab"]
    return out


def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())


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
    r.add_argument("--splits", default="select,holdout,train",
                   help="the splits this PART caches (one job = one part = one <= max_output_gib artifact)")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.arm not in cfg["arms"]:
        print(f"unknown arm {a.arm}", file=sys.stderr)
        return EXIT["setup"]
    arm = cfg["arms"][a.arm]
    splits = [s for s in ("select", "holdout", "train") if s in a.splits.split(",")]
    if not splits or set(a.splits.split(",")) - set(splits):
        print(f"--splits must name select / holdout / train, got {a.splits}", file=sys.stderr)
        return EXIT["setup"]
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    part = "+".join(splits)
    summary: dict = {"arm": a.arm, "splits": splits, "part": f"{a.arm}:{part}", "arm_cfg": arm,
                     "feature_version": cfg["feature_version"],
                     "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "dry_run": bool(a.toy_train_records)}

    def done(code: int, why: str = "") -> int:
        summary.update(exit=code, reason=why, minutes=(time.time() - t0) / 60)
        (out / f"summary_{part}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        print(json.dumps({"exit": code, "reason": why}), flush=True)
        return code

    # ---------------------------------------------------------------- guard + encoder (no model yet)
    from transformers import AutoTokenizer

    from heads.layout import hf_segment_encoder

    enc = hf_segment_encoder(AutoTokenizer.from_pretrained(a.model_dir))
    if "train" not in splits:
        guard = None  # a DEV-only part draws no training records
    elif a.toy_train_records:
        from train.guard import SplitGuard

        guard = SplitGuard.synthetic_only("q9_feature_cache dry run: toy records")
    else:
        from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard

        try:
            guard = load_ready_guard(a.guard_readiness)
        except GuardNotReady as e:
            return done(EXIT["guard"], f"guard not ready: {e}")
    summary["guard"] = guard.identity() if guard is not None else "no train split in this part"
    # ---------------------------------------------------------------- rows of every split
    from train.cohort import CohortLeak
    from train.guard import ForbiddenRow
    from train.train_set import TrainSetMismatch

    from .extract import build_rows

    try:
        units: dict[str, list[dict]] = {}
        for split, path in cfg["dev_splits"].items():
            if split not in splits:
                continue
            reqs = dev_requests(a.dev_rows, path, bool(arm.get("e5")))
            for u in reqs:
                rows, bad = build_rows(u["request"], enc)
                long = [x.qid for x in rows if len(x.ids) > cfg["dev_max_t"]]
                u["rows"] = [x for x in rows if len(x.ids) <= cfg["dev_max_t"]]
                u["skipped"] = {**bad, **{q: f"T > dev_max_t {cfg['dev_max_t']}" for q in long}}
                u["labels"] = None
            units[split] = reqs
        if "train" in splits:
            recs, ident = train_records(cfg, a, guard)
            summary["train_identity"] = ident
            fz = frozen_family_check(bool(a.toy_train_records))
            summary["frozen_family_check"] = ("harness.heldout.check_record (heldout-draw-v1)" if fz else
                                              "SKIPPED: dry run without the kit (paid path is fail-closed)")
            units["train"], summary["train_rows"] = train_rows(recs, cfg, arm, enc, guard, arm["train_rows"], fz)
    except (TrainSetMismatch, LookupError) as e:
        return done(EXIT["data"], f"data identity: {e}")
    except (CohortLeak, ForbiddenRow) as e:
        return done(EXIT["leak"], f"leak: {e}")
    except ValueError as e:  # harness.heldout.FrozenFamilyError subclasses ValueError
        if type(e).__name__ != "FrozenFamilyError":
            raise
        return done(EXIT["leak"], f"frozen held-out family: {e}")
    if "train" in splits and summary["train_rows"]["short"]:
        return done(EXIT["data"], f"only {summary['train_rows']['rows']} training rows after the filter")
    # ---------------------------------------------------------------- exact volume + in-job preflight
    tok_s, call_s = a.tok_s or 1.0, a.call_s or 0.0
    allu = [u for s in splits for u in units[s]]
    total_w = plan(allu, cfg["opt_chunk"], tok_s, call_s)
    vol = {s: {"requests": len(units[s]), "questions": sum(len(u["rows"]) for u in units[s]),
               "skipped_questions": sum(len(u.get("skipped") or {}) for u in units[s]),
               "positions": sum(u["positions"] for u in units[s]), "calls": sum(u["calls"] for u in units[s]),
               "projected_min": sum(u["w"] for u in units[s]) / 60} for s in units}
    est = estimate_bytes(units, cfg)
    for s in units:
        vol[s]["est_gib"] = round(est[s] / GIB, 3)
    summary["est_output_gib"] = {"splits": {s: vol[s]["est_gib"] for s in units}, "lex_vocab": round(est["lex_vocab"] / GIB, 3),
                                 "part": f"{a.arm}:{part}",
                                 "lex_vocab_splits": {s: round(est[f"lex_{s}"] / GIB, 3) for s in units},
                                 "lex_vocab_tokens": est["lex_vocab_tokens"], "total": round(est["total"] / GIB, 3),
                                 "max_output_gib": cfg["max_output_gib"]}
    summary["volume"] = vol
    if est["total"] > cfg["max_output_gib"] * GIB:
        return done(EXIT["too_big"], f"estimated output {est['total'] / GIB:.2f} GiB > max_output_gib {cfg['max_output_gib']}")
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

    from .extract import ShardWriter, extract_request, lex_vocab, load, option_tokens, recompute_check, sha256_file

    try:
        text, w_out, prov = load(a.model_dir, a.device, a.dtype, a.gdn_ckpt, arm.get("gdn_sha256"), arm["layers"])
    except ValueError as e:
        return done(EXIT["data"], f"model/gdn identity: {e}")
    summary["provenance"] = prov
    torch.manual_seed(cfg["seed"])
    sc = cfg["selfcheck"]
    checks = []
    for u in [u for u in units[splits[0]] if u["rows"]][: sc["rows"]]:
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
    for split in splits:
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
    toks = set()
    for s in splits:
        for u in units[s]:
            toks |= option_tokens(u["rows"])
    lv = out / f"lex_vocab_{part}.pt"  # ONE lexical table per part (parts of one arm unpack side by side)
    torch.save(lex_vocab(w_out, toks), lv)
    summary["lex_vocab"] = {"file": lv.name, "splits": splits, "sha256": sha256_file(lv), "tokens": len(toks),
                            "bytes": lv.stat().st_size}
    written = sum(dir_bytes(out / s) for s in splits) + lv.stat().st_size
    summary["output_bytes"] = written
    if written > cfg["max_output_gib"] * GIB:
        return done(EXIT["too_big"], f"wrote {written / GIB:.2f} GiB > max_output_gib {cfg['max_output_gib']}")
    if torch.cuda.is_available():
        summary["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return done(EXIT["ok"], "ok")


if __name__ == "__main__":
    sys.exit(main())
