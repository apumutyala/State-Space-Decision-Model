"""Receipt script (cohort v3): MEASURED usable retention items and the per-arm G13 retention evaluation volume / cost.

For every retention record of the cohort (v2's PR #16 records + the new m2-150k records; each line re-hashed against
its cohort entry):
* items: ``train.data.items_from_records`` with each parent's real template + tokenizer (``train.run_cell.parent_encoder``
  on the pinned tokenizer files) kept by ``train.run_cell.keep_item`` at the pre-registered unit spec (K <= 16,
  T <= 2,048): exactly what ``train.m2_gates.decision_items`` scores;
* volume: ``jobs.common.readout_cost.row_volumes`` (the decide HF readout's prefix-fork plan; pure tokenisation) of
  every record with >= 1 kept item (``arm_decision_fields`` calls ``decide`` only for those), plus the upper-bound
  convention of ``score_parent.VOLUME_TOKENS`` (prompt tokens of every question row; reproduces v2's 5,724,231).
Cost = positions / tok_s + calls x call_s at MEASURED rates (job-tkfaj RWKV7-13.3B; job-5cmat Qwen3.5-9B), H100 batch
$0.0495/min.

    python -X utf8 receipts/2026-10-07-D-cohort-v3/measure.py train/cohorts/m2-dev-cohort-v3.jsonl.gz \\
        receipts/2026-10-07-D-cohort-v3/measure.json --workers 20
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
PR16 = DATA / "m2/build/sample-20000-seed20261006/records.jsonl.gz"
P150 = DATA / "m2/build/sample-150000-seed20261006/records.jsonl.gz"
TOK = {"rwkv7": DATA / "tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
       "qwen3.5": DATA / "tokenizers/Qwen--Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
TEMPLATE = {"rwkv7": "jrt-rwkv-g1-v1", "qwen3.5": "jrt-chatml-v1"}
MAX_K, MAX_T = 16, 2048

_W = {}


def _init(family):
    sys.path.insert(0, REPO)
    from jobs.common import readout_cost as RC
    from train.run_cell import parent_encoder

    enc = parent_encoder("rwkv7" if family == "rwkv7" else "q9af", str(TOK[family]))
    rec, fk = RC.backends(RC.encoder(family), TEMPLATE[family])
    _W.update(family=family, enc=enc, rec=rec, fk=fk)


def _one(r):
    from jobs.common.readout_cost import row_volumes
    from train.data import items_from_records
    from train.run_cell import keep_item

    fam = _W["family"]
    out = {"id": r["id"]}
    try:
        its = items_from_records([r], _W["enc"], layout="ordered", template=TEMPLATE[fam])
    except Exception as e:  # noqa: BLE001
        out["refused"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out
    kept = [x for x in its if keep_item(x.row, MAX_K, MAX_T)]
    out.update(n_rows=len(its), n_kept=len(kept), prompt_tokens_all=sum(len(x.row.ids) for x in its))
    if kept:
        _W["rec"]._enc_memo.clear()
        try:
            v = row_volumes({"model": "cost", "state": r["state"], "questions": r["questions"]}, _W["rec"], _W["fk"])
            out["fork"] = v["fork"]
        except Exception as e:  # noqa: BLE001
            out["vol_refused"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


def load_records(cohort_path, limit=None):
    op = gzip.open if str(cohort_path).endswith(".gz") else open
    with op(cohort_path, "rt", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().split("\n") if x.strip()]
    ents = [e for e in lines[1:] if e["role"] == "retention"]
    want = {"pr16": {}, "m2-150k": {}}
    for e in ents:
        want[e.get("pool", "pr16")][e["id"]] = e
    if limit:
        keep = set(sorted(want["m2-150k"])[:limit]) | set(sorted(want["pr16"])[: max(1, limit // 10)])
        want = {k: {i: e for i, e in v.items() if i in keep} for k, v in want.items()}
    recs = []
    for pool, path in (("pr16", PR16), ("m2-150k", P150)):
        if not want[pool]:
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="\n") as fh:
            for line in fh:
                line = line.rstrip("\n")
                r = json.loads(line)
                e = want[pool].get(r["id"])
                if e is None:
                    continue
                assert hashlib.sha256(line.encode()).hexdigest() == e["sha256"], r["id"]
                r["_pool"] = pool
                recs.append(r)
    assert len(recs) == sum(len(v) for v in want.values())
    return recs


#: MEASURED rates. RWKV7-G1k-13.3B: the job-tkfaj probe (prefill tok/s at 2,048 tokens; small_call_s 0.117, pack_call_s
#: 0.60), the rates the m2 first cell prices its scoring phases with (PR #74). Qwen3.5-9B (the Q9 / Q9-AF backbone):
#: job-5cmat (min stock prefill over 4k-32k; call_s fitted end to end on the real stock readout, as
#: jobs/q9_feature_cache prices it; pack_call_s 0.2369 from jobs/full8).
RATES = {"rwkv7": {"job": "job-tkfaj", "tok_s": 3757.0, "call_s": 0.117, "pack_call_s": 0.60},
         "qwen3.5": {"job": "job-5cmat", "tok_s": 22144.0, "call_s": 0.0595, "pack_call_s": 0.2369}}
USD_PER_MIN = 0.0495


def cost(res):
    out = {}
    for fam, r in res.items():
        rt = RATES[fam]
        for part in ("all", "m2-150k", "pr16"):
            t = r["totals"][part]
            m = (t["fork_positions_scored"] / rt["tok_s"] + t["fork_calls_scored"] * rt["call_s"]) / 60
            cons = (t["fork_positions_scored"] / rt["tok_s"] + t["fork_calls_scored"] * rt["pack_call_s"]) / 60
            ub = t["prompt_tokens_all_rows"] / rt["tok_s"] / 60
            out.setdefault(fam, {})[part] = {
                "rates": rt, "positions": t["fork_positions_scored"], "calls": t["fork_calls_scored"],
                "items": t["items_kept"], "minutes": round(m, 1), "usd": round(m * USD_PER_MIN, 2),
                "minutes_conservative_pack_call": round(cons, 1), "usd_conservative": round(cons * USD_PER_MIN, 2),
                "upper_bound_all_rows_minutes": round(ub, 1), "upper_bound_usd": round(ub * USD_PER_MIN, 2)}
    return out


def composition(recs):
    from train.cohort_v3 import _record_stats
    from train.cohort_v3 import composition as comp

    stats = {r["id"]: _record_stats(r) for r in recs}
    return {"all": comp(list(stats), stats),
            **{p: comp([r["id"] for r in recs if r["_pool"] == p], stats) for p in ("pr16", "m2-150k")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--family", default="both")
    a = ap.parse_args()
    recs = load_records(a.cohort, a.limit)
    pools = {r["id"]: r["_pool"] for r in recs}
    res = {}
    for fam in (["rwkv7", "qwen3.5"] if a.family == "both" else [a.family]):
        t = time.time()
        with mp.Pool(a.workers, initializer=_init, initargs=(fam,)) as pool:
            per = list(pool.imap(_one, recs, chunksize=64))
        tot = {}
        for p in ("pr16", "m2-150k", "all"):
            rows = [x for x in per if p == "all" or pools[x["id"]] == p]
            ok = [x for x in rows if "refused" not in x]
            tot[p] = {"records": len(rows), "refused": len(rows) - len(ok),
                      "rows": sum(x["n_rows"] for x in ok), "items_kept": sum(x["n_kept"] for x in ok),
                      "records_with_items": sum(1 for x in ok if x["n_kept"]),
                      "prompt_tokens_all_rows": sum(x["prompt_tokens_all"] for x in ok),
                      "fork_positions_scored": sum(x["fork"]["positions"] for x in ok if "fork" in x),
                      "fork_tokens_scored": sum(x["fork"]["tokens"] for x in ok if "fork" in x),
                      "fork_calls_scored": sum(x["fork"]["calls"] for x in ok if "fork" in x),
                      "vol_refused": sum(1 for x in ok if "vol_refused" in x)}
        res[fam] = {"totals": tot, "secs": round(time.time() - t, 1),
                    "refused_examples": [x for x in per if "refused" in x or "vol_refused" in x][:10]}
        print(fam, json.dumps(tot["all"]), res[fam]["secs"], flush=True)
    op = gzip.open if a.cohort.endswith(".gz") else open
    with op(a.cohort, "rt", encoding="utf-8") as f:
        text = f.read()
    doc = {"cohort_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "unit_spec": {"max_k": MAX_K, "max_t": MAX_T, "layout": "ordered", "templates": TEMPLATE,
                         "tokenizers": {k: v.name for k, v in TOK.items()}},
           "families": res, "composition_retention_role": composition(recs), "cost": cost(res)}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(doc["cost"], indent=1))


if __name__ == "__main__":
    main()
