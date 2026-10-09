"""Mode ``fulldev_eval``: score a SAVED m2 cell checkpoint on the FULL Index DEV slice (12,104 rows, edition 0.2.1),
the same rows file, edition and scorer as the Clef comparators (job-zfexa art-p93d4, job-t5yg2 art-7pyk2).

    # GPU (in the job): the child arm, then (optional) the stock-readout parent arm on the same rows
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval score --arm child --cell <out>/cell.json --ckpt-dir <ck>/cell \
        --model-dir <snap> --records <records> --train-dir <150k> --guard-readiness <readiness.json> \
        --rows <data>/slices/dev-rows.jsonl.gz --resume-dir <ck>/fulldev --out-dir <out>/fulldev [--progress <p.jsonl>]
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval score --arm parent --cell-config <cfg> --model-dir <snap> ...
    # GPU, sharded (mode fulldev_eval_d13_v3w_s<K>of2): the same with --rows-select shard<K>of<N> (or --shard K/N):
    #   the committed fulldev_shards_<N>.json, rebuilt from the rows in-job (case clusters never split); optional
    #   --seed-partial <a stopped run's child.partial.jsonl> (+ .identity.json) resumes its rows, identity-checked
    # CPU: merge completed shards into the unsharded child.results.jsonl (refuses on any identity / hash mismatch)
    python -X utf8 -m jobs.m2_first_cell.fulldev_eval merge <s1 fulldev dir> <s2 fulldev dir> --rows <dev rows>         --out <dir>/fulldev/child.results.jsonl
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
import errno
import hashlib
import json
import math
import os
import shutil
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
    DEV manifest + base subset re-checked); ``shard<K>of<N>`` = shard K of the committed N-way full-DEV shard plan
    (``fulldev_shards_<N>.json``: every shard's ids sha256 re-checked, shards disjoint and covering the DEV ids)."""
    if rows_mode == "full":
        return None
    if parse_shard(rows_mode) is not None:
        k, n = parse_shard(rows_mode)
        return set(load_shard_plan(n)["shards"][k - 1]["run_ids"])
    if rows_mode == "proxy600":
        from jobs.common.dev_proxy import proxy_ids

        return proxy_ids("proxy600")
    if rows_mode != "smoke":
        raise ValueError(f"rows mode {rows_mode!r}: one of {ROW_MODES} or shard<K>of<N>")
    d = json.loads(SMOKE.read_text(encoding="utf-8"))
    if ids_sha256(d["run_ids"]) != d["ids_sha256"] or len(set(d["run_ids"])) != d["n"]:
        raise ValueError("fulldev_smoke_rows.json run_ids do not match their ids_sha256 / n")
    return set(d["run_ids"])


# ------------------------------------------------------------------------------------------------ full-DEV shards
#: One ~500-min full-DEV job tolerates a box only ~1.8 % slower than the screen under its 570-min cap, and a stopped run
#: loses every scored row. Sharded full DEV = N pre-declared, disjoint, case-closed row sets (one job each) merged on CPU
#: (``merge``) into the exact child.results.jsonl an unsharded job writes (same rows, DEV row order, same bytes per row).
SHARD_SEED = "fulldev-shard-v1"
SHARD_TOL = 0.002  # rebalance until the heaviest - lightest shard is <= 0.2 % of the DEV tokens (or no move helps)


def shard_file(n: int) -> Path:
    return ROOT / "jobs" / "m2_first_cell" / f"fulldev_shards_{int(n)}.json"


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """``"K/N"`` or ``"shard<K>of<N>"`` -> (K, N) with 1 <= K <= N, N >= 2; anything else -> None."""
    import re

    m = re.fullmatch(r"(?:shard)?(\d+)(?:/|of)(\d+)", str(spec or ""))
    if not m:
        return None
    k, n = int(m.group(1)), int(m.group(2))
    return (k, n) if 1 <= k <= n and n >= 2 else None


def shard_plan(rows: list[dict], lengths: dict[str, int], n: int, seed: str = SHARD_SEED,
               tol: float = SHARD_TOL) -> dict[str, Any]:
    """The N-way shard plan of the DEV rows (manifest + committed RWKV7 lengths only; never model output).
      1. clusters = the bootstrap's cluster unit: ``harness.cases.case_ids`` over ALL the rows (the frozen case-cluster
         bootstrap of ``harness.score.decompose(clusters="case")``); a cluster never straddles two shards;
      2. initial shard of a cluster = int(sha256(f"{seed}:{case_id}")) mod N;
      3. balance by RWKV token count (expected minutes = tokens x the per-token rate): while the heaviest - lightest
         shard gap > ``tol`` x total tokens, move the heaviest shard's cluster whose tokens are nearest gap / 2 (only
         clusters lighter than the gap, so the sum of squared loads strictly falls: it terminates; ties by sha) to the
         lightest. Deterministic: same rows + lengths + seed -> same plan."""
    from harness.cases import case_ids

    cid = case_ids(rows)
    members: dict[str, list[str]] = {}
    for r in rows:
        e = r["_evaluation"]
        members.setdefault(cid[(e["catalog_id"], str(e["group_id"]))], []).append(e["run_id"])
    tok = {c: sum(int(lengths[i]) for i in ids) for c, ids in members.items()}
    h = {c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest() for c in members}
    where = {c: int(h[c], 16) % n for c in members}
    load = [0] * n
    for c, k in where.items():
        load[k] += tok[c]
    total = sum(load)
    moves = 0
    while True:
        hi = max(range(n), key=lambda k: (load[k], -k))
        lo = min(range(n), key=lambda k: (load[k], k))
        gap = load[hi] - load[lo]
        if gap <= tol * total:
            break
        cands = [c for c in members if where[c] == hi and 0 < tok[c] < gap]
        if not cands:
            break
        c = min(cands, key=lambda c: (abs(gap / 2 - tok[c]), h[c]))
        where[c] = lo
        load[hi] -= tok[c]
        load[lo] += tok[c]
        moves += 1
    shards = []
    for k in range(n):
        ids = sorted(i for c in members if where[c] == k for i in members[c])
        shards.append({"k": k + 1, "n_rows": len(ids), "n_clusters": sum(1 for c in where if where[c] == k),
                       "tokens": load[k], "token_share": round(load[k] / total, 6), "ids_sha256": ids_sha256(ids),
                       "run_ids": ids})
    all_ids = [r["_evaluation"]["run_id"] for r in rows]
    return {"n": n, "seed": seed, "tol": tol, "rebalance_moves": moves, "cluster_unit": "harness.cases.case_ids",
            "n_rows": len(all_ids), "n_clusters": len(members), "total_tokens": total,
            "dev_ids_sha256": ids_sha256(all_ids), "max_imbalance_tokens": max(load) - min(load), "shards": shards}


def check_shard_plan(plan: dict, n: int | None = None) -> dict:
    """ValueError unless the plan's shards reproduce their ids hashes, are disjoint and cover its DEV ids hash."""
    if n is not None and plan.get("n") != n:
        raise ValueError(f"shard plan n {plan.get('n')} != {n}")
    seen: set[str] = set()
    for i, s in enumerate(plan["shards"]):
        ids = s["run_ids"]
        if s["k"] != i + 1 or len(set(ids)) != len(ids) or len(ids) != s["n_rows"] or ids_sha256(ids) != s["ids_sha256"]:
            raise ValueError(f"shard {i + 1}: run_ids do not match its k / n_rows / ids_sha256")
        if seen & set(ids):
            raise ValueError(f"shard {i + 1} overlaps an earlier shard")
        seen |= set(ids)
    if len(plan["shards"]) != plan["n"] or ids_sha256(seen) != plan["dev_ids_sha256"] or len(seen) != plan["n_rows"]:
        raise ValueError("shards do not cover exactly the plan's DEV ids")
    return plan


def load_shard_plan(n: int) -> dict:
    p = shard_file(n)
    if not p.is_file():
        raise ValueError(f"no committed shard plan {p.name}")
    return check_shard_plan(json.loads(p.read_text(encoding="utf-8")), n)


def verify_shard_rows(rows: list[dict], lengths: dict[str, int], rows_mode: str) -> dict:
    """In-job: the committed plan must be exactly what the DEV rows + lengths rebuild (cluster integrity re-proved on
    the rows the job scores; ~3 s CPU). Returns the shard entry (no run_ids)."""
    k, n = parse_shard(rows_mode)
    plan = load_shard_plan(n)
    got = shard_plan(rows, lengths, n, plan["seed"], plan["tol"])
    if [s["ids_sha256"] for s in got["shards"]] != [s["ids_sha256"] for s in plan["shards"]] \
            or got["dev_ids_sha256"] != plan["dev_ids_sha256"]:
        raise ValueError(f"fulldev_shards_{n}.json does not rebuild from the DEV rows + lengths (clusters / balance)")
    return {**{x: y for x, y in plan["shards"][k - 1].items() if x != "run_ids"}, "n": n,
            "dev_ids_sha256": plan["dev_ids_sha256"], "seed": plan["seed"]}


def readout_fingerprint(ident: dict) -> str:
    return hashlib.sha256(json.dumps(ident.get("readout"), sort_keys=True, default=str).encode()).hexdigest()


def run_identity(ident: dict, arm: str, rows_mode: str, keep_sha: str, code_sha: str | None) -> dict:
    """What a partial / shard result must share with another to be combined: checkpoint bytes, readout, code, rows."""
    return {"arm": arm, "checkpoint_sha256": (ident.get("checkpoint") or {}).get("sha256") or ident.get("revision"),
            "readout_fingerprint": readout_fingerprint(ident), "code_sha": code_sha, "rows_mode": rows_mode,
            "rows_ids_sha256": keep_sha}


SHARD_SCHEMA = "ssdm.m2.fulldev_shard.v1"
SEED_KEYS = ("arm", "checkpoint_sha256", "readout_fingerprint", "code_sha", "rows_mode", "rows_ids_sha256")


def seed_partial(seed: Path, resume: Path, want: dict, keep: set[str]) -> int:
    """Append a stopped job's partial rows (``seed`` + its ``<stem>.identity.json``) to this job's resume log, after
    checking the identity (checkpoint sha, readout fingerprint, code sha, arm, rows mode + ids hash; code sha must be
    known) and that every seed row belongs to this run's rows. Rows already in the resume log are kept as they are.
    Returns the number of rows added; ValueError on any mismatch (nothing is written then)."""
    idf = seed.with_name(seed.name.replace(".jsonl", "") + ".identity.json")
    if not idf.is_file():
        raise ValueError(f"seed partial {seed.name}: no identity file {idf.name}")
    have = json.loads(idf.read_text(encoding="utf-8"))
    if not want.get("code_sha"):
        raise ValueError("seed partial: this run's code sha is unknown (refusing an unverifiable seed)")
    bad = [k for k in SEED_KEYS if have.get(k) != want.get(k)]
    if bad:
        raise ValueError(f"seed partial identity differs on {bad}: {[(have.get(k), want.get(k)) for k in bad]}")
    rows = _read_partial(seed)  # a torn tail (killed mid-write) is dropped, as on an in-job resume
    stray = set(rows) - keep
    if stray:
        raise ValueError(f"seed partial: {len(stray)} rows are not in this run's rows (e.g. {sorted(stray)[:3]})")
    done = _read_partial(resume)
    add = [rid for rid in sorted(rows) if rid not in done]
    with open(resume, "a", encoding="utf-8", newline="\n") as f:
        for rid in add:
            f.write(json.dumps(rows[rid], separators=(",", ":")) + "\n")
    return len(add)


def seed_fetch(url: str, sha256: str, out_dir: Path, arm: str = "child") -> Path:
    """Fetch a stopped job's phase tgz (any phase exported after scoring began), check its sha256, and extract ONLY
    ``fulldev/<arm>.partial.jsonl`` + ``fulldev/<arm>.partial.identity.json`` (by member name, written under fixed
    names; untrusted archive: nothing else is read). Returns the partial's path. ValueError on any mismatch."""
    import tarfile
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz = out_dir / "seed.tgz"
    with urllib.request.urlopen(url, timeout=300) as r, open(tgz, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if _sha(tgz) != sha256:
        raise ValueError(f"seed archive sha256 {_sha(tgz)} != {sha256}")
    want = {f"{arm}.partial.jsonl": None, f"{arm}.partial.identity.json": None}
    with tarfile.open(tgz, "r:*") as tf:
        for m in tf.getmembers():
            parts = m.name.replace("\\", "/").split("/")
            if m.isfile() and len(parts) >= 2 and parts[-2] == "fulldev" and parts[-1] in want:
                if want[parts[-1]] is not None:
                    raise ValueError(f"seed archive has more than one {parts[-1]}")
                want[parts[-1]] = tf.extractfile(m).read()
        miss = [k for k, v in want.items() if v is None]
        if miss:
            raise ValueError(f"seed archive lacks fulldev/{miss}")
    for k, v in want.items():
        (out_dir / k).write_bytes(v)
    return out_dir / f"{arm}.partial.jsonl"


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


#: job-afpc2: the worker's SHARED disk (it holds $GMN_CHECKPOINT_DIR, $GMN_OUTPUT_DIR and the platform's spool) filled
#: mid-scoring (201,875,456 B free at the end): the child died (rc 1), the phase tgz could not be written (export ok=false,
#: no sha) and the platform refused the artifact (a transfer must leave 4,294,998,016 B free). Below this many free bytes
#: on the resume / output dirs the child stops cleanly: declare the slot, yield, exit DISK_LOW_RC.
DISK_LOW_BYTES = 6 << 30
DISK_LOW_RC = 16
MAX_YIELDS = 2
CHECKPOINT_EVERY_S = 600.0  # the platform floor is one accepted declare per 300 s
#: --readout: "native" = the cell's own template + layout, served with the cohort temperature fit (the tempered
#: decider; Index arm = untempered answers rebuilt from raw); any other name = jobs.m2_first_cell.screen SCREEN_READOUTS
#: through load_child's readout override (the same model object under another prompt), UNTEMPERED (identity
#: calibration, as the screen scores it): no temperature fit exists for a non-training readout in this code.
READOUT_NAMES = ("native", "v1", "v2f", "v3w")


def progress_beat(every_s: float = 30.0):
    """A throttled toucher of ``$SSDM_PROGRESS_BEAT`` (the runner's progress-gated watchdog withholds the heartbeat
    after STALL_S = 1200 s without a file write; the platform kills after two missed 600 s polls). For CPU loops that
    write nothing until they end (the plan pass). Unset (tests): a no-op. Outputs are unchanged either way; a hung
    loop stops touching it."""
    path = os.environ.get("SSDM_PROGRESS_BEAT")
    last = [float("-inf")]

    def beat() -> None:
        if not path:
            return
        now = time.monotonic()
        if now - last[0] >= every_s:
            last[0] = now
            try:
                Path(path).touch()
            except OSError:
                pass

    return beat


def _meta_post(door: str) -> bool:
    """POST ``$GMN_METADATA_URL/v1/<door>`` (checkpoint | yield). Unset (tests, local): False, nothing sent."""
    import urllib.request

    url = os.environ.get("GMN_METADATA_URL")
    if not url:
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(f"{url}/v1/{door}", data=b"{}", method="POST",
                                                      headers={"Content-Type": "application/json"}), timeout=10)
        return True
    except Exception as e:  # noqa: BLE001  a 429 (rate floor) / 503 / 409 is not a failure: the last slot stands
        print(f"{door} declare: {e!r}", file=sys.stderr, flush=True)
        return False


def checkpoint_declarer(every_s: float = CHECKPOINT_EVERY_S, post=None):
    """A throttled declare of the platform checkpoint slot ($GMN_CHECKPOINT_DIR = the resume dir: the per-row append
    log): a preempted / yielded attempt resumes from the last committed capture instead of from zero (job-afpc2's
    three preempted attempts restarted from nothing: no declare). ``declare(force=True)`` ignores the throttle."""
    post = post or (lambda: _meta_post("checkpoint"))
    last = [float("-inf")]

    def declare(force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - last[0] < every_s:
            return False
        last[0] = now
        return post()

    return declare


def disk_watch(paths, low_bytes: int = DISK_LOW_BYTES, every_s: float = 30.0, usage=None):
    """Throttled free-space check of ``paths``: ``None`` while every path's filesystem has >= ``low_bytes`` free, else
    ``{path: free_bytes}`` of the low ones. Read-only; changes no output."""
    usage = usage or (lambda q: shutil.disk_usage(q).free)
    last = [float("-inf")]

    def low():
        now = time.monotonic()
        if now - last[0] < every_s:
            return None
        last[0] = now
        free = {}
        for q in paths:
            try:
                free[str(q)] = int(usage(q))
            except OSError:
                continue
        bad = {q: f for q, f in free.items() if f < low_bytes}
        return bad or None

    return low


def stop_for_disk(resume_dir: Path, declare, why: str, yield_post=None) -> int:
    """The worker disk is (nearly) full: declare the slot (best effort: rows already in the append log), then yield
    (the rerun re-queues, possibly on another machine, and resumes from the slot) at most MAX_YIELDS times per job
    (counted in the slot), and return DISK_LOW_RC either way so the runner fails loudly if no yield takes effect."""
    yield_post = yield_post or (lambda: _meta_post("yield"))
    print(f"WORKER DISK LOW: {why}", file=sys.stderr, flush=True)
    n_file = resume_dir / "disk_yields.json"
    try:
        n = int(json.loads(n_file.read_text(encoding="utf-8")).get("yields", 0))
    except (OSError, ValueError):
        n = 0
    declared = declare(force=True)
    if n < MAX_YIELDS:
        try:
            n_file.write_text(json.dumps({"yields": n + 1, "why": why}), encoding="utf-8")
        except OSError:
            pass
        ok = yield_post()
        print(f"yield {n + 1}/{MAX_YIELDS}: {'sent' if ok else 'not accepted'} (checkpoint declared: {declared})",
              file=sys.stderr, flush=True)
    else:
        print(f"no yield: {n} of {MAX_YIELDS} already used", file=sys.stderr, flush=True)
    return DISK_LOW_RC


def readout_spec(name: str | None, cfg: dict) -> dict[str, Any]:
    """{name, template, layout, native} of ``--readout`` against the cell config (ValueError on an unknown name)."""
    name = name or "native"
    if name not in READOUT_NAMES:
        raise ValueError(f"readout {name!r}: one of {READOUT_NAMES}")
    if name == "native":
        tpl, lay = cfg["template"], cfg.get("layout", "ordered")
    else:
        from jobs.m2_first_cell.screen import SCREEN_READOUTS

        tpl, lay = SCREEN_READOUTS[name]
    return {"name": name, "template": tpl, "layout": lay,
            "native": (tpl, lay) == (cfg["template"], cfg.get("layout", "ordered"))}


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
    beat = progress_beat()
    for r in rows:
        beat()  # job-4ha7t: a full-DEV plan pass is ~40 min of tokenisation with no file write (watchdog STALL_S 1200)
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
    ckpt = Path(a.ckpt) if a.ckpt else Trainer.scored(a.ckpt_dir)  # best-by-dev when the run kept one
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
    if keep is not None and not keep <= set(ids):
        print(f"rows: {len(keep - set(ids))} {a.rows_select} ids are not DEV rows", file=sys.stderr)
        return 10
    shard = None
    if parse_shard(a.rows_select) is not None:  # the committed plan must rebuild from these rows (cluster integrity)
        try:
            shard = verify_shard_rows(rows, json.loads(LENGTHS.read_text(encoding="utf-8"))["lengths"], a.rows_select)
        except (ValueError, KeyError) as e:
            print(f"rows: {e}", file=sys.stderr)
            return 10
    if keep is not None:
        rows = [r for r in rows if r["_evaluation"]["run_id"] in keep]
        ids = [r["_evaluation"]["run_id"] for r in rows]
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train.cohort import CohortLeak

    try:
        if a.arm == "child":
            ro = readout_spec(getattr(a, "readout", None),
                              json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"])
            if ro["name"] == "native":  # the served decider + the cohort temperature fit (ECE served)
                decider, tempered, ident = load_child(a)
            else:  # another readout of the same loaded model: untempered (load_child's override)
                decider, tempered, ident = load_child(a, temperature=False, readout=(ro["template"], ro["layout"]))
            ident = {**ident, "readout": ro}
        else:
            decider, tempered, ident = load_parent(a)
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
    # the partial's identity (a stopped run's rows are exported with it; --seed-partial checks it before reuse)
    rid_ = run_identity(ident, a.arm, a.rows_select, ids_sha256(ids), getattr(a, "code_sha", None))
    idp = resume.with_name(f"{a.arm}.partial.identity.json")
    if idp.is_file():
        old = json.loads(idp.read_text(encoding="utf-8"))
        bad = [k for k in SEED_KEYS if k != "code_sha" and old.get(k) != rid_.get(k)]
        if bad:  # the slot holds another checkpoint's / readout's / row set's rows: never mix them
            print(f"identity: resume log {resume} belongs to another run (differs on {bad})", file=sys.stderr)
            return 10
    idp.write_text(json.dumps(rid_, indent=1), encoding="utf-8")
    if getattr(a, "seed_partial", None):
        try:
            n_seed = seed_partial(Path(a.seed_partial), resume, rid_, set(ids))
        except (ValueError, KeyError, OSError) as e:
            print(f"identity: seed partial: {e}", file=sys.stderr)
            return 10
        print(json.dumps({"seed_partial_rows_added": n_seed}), flush=True)
    done = _read_partial(resume)
    todo = [r for r in rows if r["_evaluation"]["run_id"] not in done]
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(r) for r in todo])
    decide = G.beat_wrap(prog.wrap(limited(decider.decide, a.max_input_tokens)))
    declare = checkpoint_declarer()
    low = disk_watch([resume.parent, out])
    declare(force=True)  # the slot now holds the rows so far
    t0 = time.perf_counter()
    n_done = len(ids) - len(todo)
    try:
        with open(resume, "a", encoding="utf-8", newline="\n") as f:
            for r in todo:
                bad = low()
                if bad:
                    f.flush()
                    return stop_for_disk(resume.parent, declare, f"free bytes {bad} < {DISK_LOW_BYTES} before row "
                                         f"{r['_evaluation']['run_id']} ({n_done} of {len(ids)} rows done)")
                res = G.index_predict(decide, [r], untempered_from_raw=tempered)
                f.write(json.dumps(res[r["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
                f.flush()
                n_done += 1
                declare()
    except OSError as e:  # ENOSPC only (a torn tail is dropped by _read_partial on resume); anything else propagates
        if e.errno != errno.ENOSPC:
            raise
        return stop_for_disk(resume.parent, declare, f"{type(e).__name__}: {e} ({n_done} of {len(ids)} rows done)")
    declare(force=True)
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
    if shard is not None:  # what merge checks: shard id hash, identity, results bytes
        (out / f"shard-{a.arm}.json").write_text(json.dumps(
            {"schema": SHARD_SCHEMA, **shard, **rid_, "results_sha256": _sha(out / f"{a.arm}.results.jsonl"),
             "results_served_sha256": (_sha(out / f"{a.arm}.results_served.jsonl") if tempered else None)},
            indent=1), encoding="utf-8")
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
                                      else f"{'SMOKE' if mode == 'smoke' else 'SHARD' if parse_shard(mode) else 'PROXY'} "
                                           f"SUBSET ({mode}): harness.score."
                                           "decompose + harness.bootstrap + harness.calibration in process on the "
                                           "pre-declared rows (the CLI scores whole slices only); "
                                           + ("a pipeline proof, not a capability estimate" if mode == "smoke" else
                                              "ONE SHARD of the full DEV: the headline is the merged file "
                                              "(fulldev_eval merge -> analysis/land_fulldev.py)" if parse_shard(mode) else
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
        nested = {}
        if keep is None:  # the 2,008 m1 subset + the proxy600 screen rows, from the SAME full-DEV predictions
            from jobs.common.dev_subset import load_subset

            for name, ids_ in (("subset2008", set(load_subset()["run_ids"])), ("proxy600", selected_ids("proxy600"))):
                ns = subset_score(rp, ids_)
                (d / f"{arm}.score_{name}.json").write_text(json.dumps(ns, indent=1), encoding="utf-8")
                nested[name] = {"rows": len(ids_), "index": ns["slice_index"]["index"]}
        ent = {"results_sha256": _sha(rp), "status": _count(r.get("status") for r in res),
               "full": sc["slice_index"], "official_index_missing_as_zero": sc["official_index_missing_as_zero"],
               "excl_overlength51": ex["slice_index"],
               "ece": {"board_protocol": (ece.get("board_protocol") or {}).get("ece"),
                       "overall_10bin": (ece.get("overall") or {}).get("ece_10bin")},
               "latency": latency(res), **nested}
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
        idc = json.loads(idf.read_text(encoding="utf-8"))
        out["checkpoint"]["loaded"] = idc.get("checkpoint")
        out["readout"] = idc.get("readout")
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


# ------------------------------------------------------------------------------------------------ CPU: shard merge
class MergeError(ValueError):
    pass


def _shard_dir(p: Path, arm: str) -> Path:
    """A shard's fulldev out dir: the dir itself, the dir of a given results file, or the one fulldev/ under it."""
    p = Path(p)
    if p.is_file():
        return p.parent
    if (p / f"shard-{arm}.json").is_file() or (p / f"{arm}.results.jsonl").is_file():
        return p
    hits = sorted(x.parent for x in p.rglob(f"fulldev/shard-{arm}.json"))
    if len(hits) != 1:
        raise MergeError(f"{p}: want exactly one fulldev/shard-{arm}.json under it, found {len(hits)}")
    return hits[0]


def merge(shards: list[Path], rows_path: Path, out: Path, arm: str = "child") -> dict[str, Any]:
    """Merge N shard results into the exact ``<arm>.results.jsonl`` an unsharded job writes: every row's line as the
    shard wrote it (score() writes the same json.dumps per row either way), in DEV row order. Refuses (MergeError,
    nothing written) unless: one shard per K = 1..N of one committed N-way plan; each shard's recorded AND recomputed
    ids hash equals the plan's; results bytes match their recorded sha; the shards share the checkpoint sha, readout
    fingerprint and (known) code sha; they are disjoint and cover exactly the DEV rows (merged ids hash == ``ids_sha256``
    over all 12,104 DEV run_ids, the unsharded convention)."""
    from harness.suite import read_rows

    metas, lines = [], []
    for sp in shards:
        d = _shard_dir(sp, arm)
        mf = d / f"shard-{arm}.json"
        if not mf.is_file():
            raise MergeError(f"{d}: no shard-{arm}.json (not a completed shard)")
        m = json.loads(mf.read_text(encoding="utf-8"))
        rp = d / f"{arm}.results.jsonl"
        if m.get("schema") != SHARD_SCHEMA or m.get("arm") != arm:
            raise MergeError(f"{mf}: schema / arm {m.get('schema')!r} / {m.get('arm')!r}")
        if not rp.is_file() or _sha(rp) != m.get("results_sha256"):
            raise MergeError(f"{rp}: missing or sha256 != the shard's recorded results_sha256")
        metas.append(m)
        lines.append(rp.read_bytes().decode("utf-8").splitlines(keepends=True))
    ns = {m.get("n") for m in metas}
    if len(ns) != 1:
        raise MergeError(f"shards come from different plans (n = {sorted(map(str, ns))})")
    n = ns.pop()
    try:
        plan = load_shard_plan(n)
    except (ValueError, TypeError) as e:
        raise MergeError(str(e)) from e
    ks = sorted(m.get("k") for m in metas)
    if ks != list(range(1, n + 1)):
        raise MergeError(f"want one shard per k = 1..{n}, got {ks}")
    for key in ("checkpoint_sha256", "readout_fingerprint", "code_sha", "seed", "dev_ids_sha256"):
        vals = {json.dumps(m.get(key)) for m in metas}
        if len(vals) != 1 or any(m.get(key) is None for m in metas):
            raise MergeError(f"shards disagree on (or lack) {key}: {sorted(vals)}")
    if metas[0]["dev_ids_sha256"] != plan["dev_ids_sha256"] or metas[0]["seed"] != plan["seed"]:
        raise MergeError("shards were cut from another plan (seed / DEV ids hash) than the committed one")
    by_id: dict[str, str] = {}
    for m, ls in zip(metas, lines):
        want = plan["shards"][m["k"] - 1]
        ids = []
        for x in ls:
            if not x.strip():
                continue
            rid = json.loads(x)["run_id"]
            if rid in by_id:
                raise MergeError(f"run_id {rid} is in more than one shard (not disjoint)")
            by_id[rid] = x if x.endswith("\n") else x + "\n"
            ids.append(rid)
        if not (m.get("ids_sha256") == ids_sha256(ids) == want["ids_sha256"] == m.get("rows_ids_sha256")) \
                or len(ids) != want["n_rows"] or m.get("rows_mode") != f"shard{m['k']}of{n}":
            raise MergeError(f"shard {m['k']}: ids do not reproduce the pre-declared shard ids hash "
                             f"{want['ids_sha256'][:12]} (got {ids_sha256(ids)[:12]}, recorded "
                             f"{str(m.get('ids_sha256'))[:12]})")
    dev = [r["_evaluation"]["run_id"] for r in read_rows(rows_path)]
    if len(set(dev)) != len(dev) or ids_sha256(dev) != plan["dev_ids_sha256"]:
        raise MergeError(f"{rows_path}: not the plan's DEV slice")
    if set(by_id) != set(dev):
        raise MergeError(f"shards cover {len(by_id)} ids, {len(set(dev) - set(by_id))} DEV ids missing, "
                         f"{len(set(by_id) - set(dev))} extra")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        for rid in dev:  # DEV row order, as score() writes the unsharded file
            f.write(by_id[rid])
    srt = sorted(metas, key=lambda m: m["k"])
    rec = {"schema": "ssdm.m2.fulldev_merge.v1", "arm": arm, "n": n, "rows": len(dev), "ids_sha256": ids_sha256(by_id),
           "dev_ids_sha256": plan["dev_ids_sha256"], "shard_ids_sha256": [s["ids_sha256"] for s in plan["shards"]],
           "checkpoint_sha256": srt[0]["checkpoint_sha256"], "readout_fingerprint": srt[0]["readout_fingerprint"],
           "code_sha": srt[0]["code_sha"], "shard_results_sha256": [m["results_sha256"] for m in srt],
           "results": str(out), "results_sha256": _sha(out)}
    out.with_name(out.name.replace(".jsonl", "") + ".merge.json").write_text(json.dumps(rec, indent=1),
                                                                             encoding="utf-8")
    return rec


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
    ro = readout_spec(params.get("readout"), cfg)
    if params.get("readout_template") != ro["template"]:
        raise ValueError(f"readout_template {params.get('readout_template')!r} != readout {ro['name']} template "
                         f"{ro['template']!r}")
    ratio = 1.0 if ro["native"] else float(params["readout_length_ratio"][ro["name"]])
    if params["max_input_tokens"] <= lengths["max_tokens"]:
        raise ValueError(f"max_input_tokens {params['max_input_tokens']} <= the longest DEV row {lengths['max_tokens']}")
    m = params["measured_rates_reference"]
    keep = selected_ids(params.get("rows", "full"))
    toks = lengths["total_tokens"] if keep is None else sum(lengths["lengths"][i] for i in keep)
    toks = toks * ratio  # the lengths are v1/ordered tokens; another readout's prompts are this much longer / shorter
    fixed = params["evals_fixed_minutes"]["child"]
    if params.get("plan_minutes_full") is not None:  # the plan pass tokenises only the selected rows
        fixed += params["plan_minutes_full"] * (toks / ratio) / lengths["total_tokens"]
    phases = {"fulldev": toks / 1000 * m["child_s_per_1k_tokens"] / 60 + fixed}
    if params.get("fulldev_parent_arm"):
        phases["parent_full"] = toks / 1000 * m["parent_s_per_1k_tokens"] / 60 + params["evals_fixed_minutes"]["parent"]
    caps = params["phase_caps_minutes"]
    out = {"rows": params.get("rows", "full"), "tokens": toks, "template": ro["template"], "readout": ro["name"],
           "length_ratio": ratio, "lengths_template": lengths["template"],
           "minutes": {k: round(v, 1) for k, v in phases.items()},
           "caps": {k: caps[k] for k in phases}, "ok": all(v <= 0.95 * caps[k] for k, v in phases.items())}
    out["usd_batch_scoring"] = round(sum(phases.values()) * params["batch_usd_per_min"], 2)
    out["headroom_pct"] = {k: round(100 * (caps[k] / v - 1), 1) for k, v in phases.items()}
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
    s.add_argument("--rows-select", default="full",
                   help=f"one of {ROW_MODES} (full DEV, the committed smoke slice, the proxy600 screen) or shard<K>of<N>")
    s.add_argument("--shard", default=None, help="K/N: shard K of the committed N-way full-DEV plan (= --rows-select "
                                                 "shard<K>of<N>)")
    s.add_argument("--seed-partial", default=None, help="a stopped run's <arm>.partial.jsonl (+ .identity.json next to "
                                                        "it): its rows are reused after the identity check")
    s.add_argument("--code-sha", default=None, help="the code sha the job runs (CONTEXT_MANIFEST m1_sha)")
    s.add_argument("--resume-dir", required=True, help="per-row append log (checkpoint dir): a restart resumes")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--readout", choices=READOUT_NAMES, default="native",
                   help="child: native (cell template + temperature fit) or another readout, untempered")
    b = sub.add_parser("smoke-build", help="(re)build fulldev_smoke_rows.json from the DEV manifest + row lengths")
    b.add_argument("--out", default=str(SMOKE))
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--params", required=True)
    r.add_argument("--elapsed-min", type=float, required=True)
    r.add_argument("--out", required=True)
    mg = sub.add_parser("merge", help="merge completed shard out dirs into one full-DEV <arm>.results.jsonl")
    mg.add_argument("shards", nargs="+", help="each shard's fulldev out dir (or its results file / extracted artifact)")
    mg.add_argument("--rows", default=None, help="DEV rows (default $SSDM_DATA/slices/dev-rows.jsonl.gz)")
    mg.add_argument("--arm", choices=ARMS, default="child")
    mg.add_argument("--out", required=True, help="e.g. <dir>/fulldev/child.results.jsonl (land_fulldev.py --art <dir>)")
    sf = sub.add_parser("seed-fetch", help="fetch + check a stopped job's phase tgz; extract its partial rows")
    sf.add_argument("--url", required=True)
    sf.add_argument("--sha256", required=True)
    sf.add_argument("--arm", choices=ARMS, default="child")
    sf.add_argument("--out-dir", required=True)
    sb = sub.add_parser("shard-build", help="(re)build fulldev_shards_<N>.json from the DEV rows + RWKV7 row lengths")
    sb.add_argument("--n", type=int, required=True)
    sb.add_argument("--rows", required=True)
    sb.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        rows = a.rows or str(Path(os.environ.get("SSDM_DATA", "")) / "slices" / "dev-rows.jsonl.gz")
        try:
            rec = merge([Path(x) for x in a.shards], Path(rows), Path(a.out), a.arm)
        except MergeError as e:
            print(f"merge refused: {e}", file=sys.stderr)
            return 11
        print(json.dumps(rec, indent=1))
        return 0
    if a.cmd == "seed-fetch":
        try:
            p = seed_fetch(a.url, a.sha256, Path(a.out_dir), a.arm)
        except (ValueError, OSError) as e:
            print(f"seed-fetch: {e}", file=sys.stderr)
            return 10
        print(json.dumps({"partial": str(p), "rows": len(_read_partial(p))}))
        return 0
    if a.cmd == "shard-build":
        from harness.suite import read_rows

        d = shard_plan(read_rows(a.rows), json.loads(LENGTHS.read_text(encoding="utf-8"))["lengths"], a.n)
        d = {"_comment": f"Pre-declared {a.n}-way shard plan of the full Index DEV for the sharded fulldev_eval modes "
                         "(jobs.m2_first_cell.fulldev_eval shard-build; DEV rows + RWKV7 row lengths only, never model "
                         "output). Clusters = harness.cases.case_ids (the bootstrap's cluster unit), never split; "
                         "initial shard = sha256(seed:case_id) mod N; rebalanced by token count. The job rebuilds it "
                         "from its rows and refuses on any difference; merge re-checks every hash.", **d}
        Path(a.out or shard_file(a.n)).write_text(json.dumps(d, indent=0) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in d.items() if k not in ("shards", "_comment")}))
        print(json.dumps([{k: v for k, v in x.items() if k != "run_ids"} for x in d["shards"]]))
        return 0
    if a.cmd == "score":
        need = ("cell", "ckpt_dir", "records", "train_dir", "guard_readiness") if a.arm == "child" else ("cell_config",)
        miss = [k for k in need if not getattr(a, k)]
        if miss:
            ap.error(f"--arm {a.arm} needs {miss}")
        if a.shard is not None:
            ks = parse_shard(a.shard)
            if ks is None or a.rows_select not in ("full", f"shard{ks[0]}of{ks[1]}"):
                ap.error(f"--shard {a.shard!r}: K/N with 1 <= K <= N, N >= 2 (and --rows-select full or the same shard)")
            a.rows_select = f"shard{ks[0]}of{ks[1]}"
        if a.rows_select not in ROW_MODES and parse_shard(a.rows_select) is None:
            ap.error(f"--rows-select {a.rows_select!r}: one of {ROW_MODES} or shard<K>of<N>")
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
