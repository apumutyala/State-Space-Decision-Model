"""q9af_corpus_check phase projections from the measured / estimated rates in job_params and the margin rule.

    python -m jobs.q9af_corpus_check.preflight [--params jobs/q9af_corpus_check/job_params.json]
                                               [--config configs/q9af_v2_h100.json] [--out f]

fetch = pinned parquet bytes / fetch rate. C (rebuild + scan, one streaming pass) = train + held-out tokens / the
cpu-8 tokenise+DEV-filter rate + words / (guard words/s per core x workers x efficiency) + fixed overhead (DEV
13-gram set, worker index load, the analysis). Rules (exit 5 if any fails; the runner calls this at setup): every cap
>= (1 + margin) x its projection; caps + reserve == max_duration; worst case <= hard cap; the config is w3c2e's
(dev13-v1 filter, the 280M-token mix).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project(params: dict, cfg: dict) -> dict:
    e = params["estimates_not_measured"]
    mc = cfg["mix"]
    tokens = mc["train_tokens"] + sum(mc["heldout_tokens"].values())
    words = tokens * e["words_per_token"]
    scan_rate = params["measured"]["laptop_guard_words_s_per_core"] * e["workers"] * e["scan_efficiency"]
    c = tokens / e["cpu8_tokenize_dev_filter_tok_s"] / 60 + words / scan_rate / 60 + e["fixed_overhead_min"]
    return {"phases": {"setup": e["setup_min"], "fetch": e["source_gb"] * 1024 / e["fetch_mb_s"] / 60, "C": c},
            "tokens": tokens, "words": words, "scan_words_s": scan_rate}


def check(params: dict, cfg: dict) -> dict:
    pr = project(params, cfg)
    mf = params["margin_min_frac"]
    caps = params["phase_caps_minutes"]
    rows = {ph: {"projected_min": round(p, 2), "cap_min": caps[ph], "ok": caps[ph] >= (1 + mf) * p}
            for ph, p in pr["phases"].items()}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    from jobs.common.costs import worst_case_usd

    rows["hard_cap"] = {"worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"],
                        "ok": worst_case_usd(params) <= params["hard_cap_usd"]}
    rows["w3c2e_config"] = {"decontam": cfg["mix"].get("decontam"), "train_tokens": cfg["mix"]["train_tokens"],
                            "ok": cfg["mix"].get("decontam") == "dev13-v1" and cfg["mix"]["train_tokens"] == 280_000_000}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, **pr}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_corpus_check.preflight")
    ap.add_argument("--params", default=str(ROOT / "jobs/q9af_corpus_check/job_params.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/q9af_v2_h100.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = check(json.loads(Path(a.params).read_text(encoding="utf-8")),
                json.loads(Path(a.config).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
