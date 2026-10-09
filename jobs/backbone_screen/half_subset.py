"""A deterministic, stratified half of proxy600 for the no-fork qwen36_a3b screen (its exact m0 readout re-reads every
prompt once per option: ~84M positions on all 644 rows vs ~2.4M with the prefix fork).

    python -X utf8 -m jobs.backbone_screen.half_subset build --out jobs/backbone_screen/proxy600_half_q36.json
    python -X utf8 -m jobs.backbone_screen.half_subset costs --rows <dev-rows.jsonl.gz> \
        --out jobs/backbone_screen/qwen36_m0_cost.json            (CPU, the Qwen3.6 tokenizer only)
    python -X utf8 -m jobs.backbone_screen.half_subset order --rows proxy-rows.jsonl.gz \
        --first jobs/backbone_screen/proxy600_half_q36.json --out ordered.jsonl.gz        (in the job)

Per benchmark (catalog_id), the proxy600 GROUPS (bootstrap clusters, so a group is kept or dropped whole) are ranked by
sha256(SEED + group_id) and the first ceil(n_groups / 2) are kept, with all their rows. No row content, length, option
count or label enters the choice: it is a fixed random half, and every benchmark keeps >= 3 rows (proxy600 has >= 6
rows per benchmark). Same file format as jobs/common/dev_proxy600.json (``dev_subset score|compare --subset``).

``order`` gives the row order the job runs in: the half first, then the other half, each in sha256(SEED + run_id) order,
so the rows of each part come in a cost-shuffled order and the half is complete before the rest starts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SEED = "q36-half-v1:"
PROXY = "jobs/common/dev_proxy600.json"
#: cost of one model call in positions (fixed per-forward overhead of a 40-layer MoE), for the rate guard's weights
CALL_POSITIONS = 2000
INPUT_CAP = 16384


def _h(s: str) -> str:
    return hashlib.sha256((SEED + s).encode()).hexdigest()


def choose(entries: list[dict], ids: list[str]) -> list[str]:
    """entries: DEV manifest rows (run_id, catalog_id, group_id); ids: proxy600 run_ids -> sorted kept run_ids."""
    want = set(ids)
    groups: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e["run_id"] in want:
            groups[e["catalog_id"]][str(e["group_id"])].append(e["run_id"])
    if sum(len(r) for g in groups.values() for r in g.values()) != len(want):
        raise SystemExit("proxy600 ids not all in the DEV manifest")
    kept = []
    for cat in sorted(groups):
        g = groups[cat]
        for gid in sorted(g, key=_h)[: math.ceil(len(g) / 2)]:
            kept += g[gid]
    return sorted(kept)


def order(all_ids: list[str], first: list[str]) -> list[str]:
    f = set(first)
    return sorted(f, key=_h) + sorted(set(all_ids) - f, key=_h)


def build() -> dict:
    from jobs.common import dev_subset as D

    proxy = D.load_subset(PROXY)
    ids = choose(D.read_manifest(), proxy["run_ids"])
    return {"_comment": ("Deterministic stratified half of proxy600 (jobs/backbone_screen/half_subset.py, seed "
                         f"{SEED!r}): per benchmark, ceil(n_groups/2) groups by sha256(seed + group_id), all their rows. "
                         "The qwen36_a3b no-fork m0 screen scores this half first (its primary result) and the rest "
                         "if time allows. IDs only."),
            "seed": SEED, "base_subset": PROXY, "base_subset_ids_sha256": proxy["ids_sha256"],
            "dev_manifest_sha256_uncompressed": proxy["dev_manifest_sha256_uncompressed"],
            "ids_sha256": D.ids_sha256(ids), "summary": D.summary(D.read_manifest(), ids), "run_ids": ids}


def costs(rows_path: str) -> dict:
    """Exact no-fork m0 volume per proxy600 row with the Qwen3.6 tokenizer (HFReadoutBackend(fork=False).plan, the
    batching the job uses): positions, calls, longest call; weight = positions + CALL_POSITIONS x calls, 0 for a row
    over the input cap (the adapter refuses it before any forward)."""
    from decide.backends.hf_readout import HFReadoutBackend, ModelPlugin
    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS, HFEncoder
    from jobs.common import dev_subset as D
    from jobs.common.readout_engine import QWEN36_MAX_BATCH_ROWS

    enc = HFEncoder(DEFAULT_TOKENIZERS / BACKBONES["qwen3.6-35b-a3b"].tokenizer_dir)
    plug = ModelPlugin("qwen3.6-moe", "tokenizer-only", None, lambda t, trusted=True: enc(t, trusted), None,
                       special_ids=enc.control_ids, device="cpu", dtype="bfloat16", max_context=262144)
    be = HFReadoutBackend(plug, "jrt-chatml-v1", max_batch_rows=QWEN36_MAX_BATCH_ROWS, fork=False)
    ids = set(D.load_subset(PROXY)["run_ids"])
    rows = {}
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rid = r["_evaluation"]["run_id"]
            if rid in ids:
                p = be.plan({"state": r["state"], "questions": r["questions"]})
                over = p.max_call_input_tokens > INPUT_CAP
                rows[rid] = {"positions": p.total_input_tokens, "calls": p.inference_calls,
                             "max_call": p.max_call_input_tokens, "over_cap": over,
                             "weight": 0 if over else p.total_input_tokens + CALL_POSITIONS * p.inference_calls}
    if set(rows) != ids:
        raise SystemExit(f"rows file has {len(rows)} of {len(ids)} proxy600 rows")
    return {"_comment": ("per-row no-fork m0 cost of stock Qwen3.6-35B-A3B on proxy600 (jobs/backbone_screen/"
                         "half_subset.py costs); the rate guard projects the half by these weights"),
            "tokenizer_sha256": enc.tokenizer_sha256, "max_batch_rows": QWEN36_MAX_BATCH_ROWS,
            "call_positions": CALL_POSITIONS, "input_cap": INPUT_CAP, "rows": dict(sorted(rows.items()))}


def weights_for(cost: dict, subset_ids: list[str]) -> dict[str, float]:
    """{run_id: weight} of the rows that must finish (the rate guard's --weights file)."""
    return {i: float(cost["rows"][i]["weight"]) for i in sorted(subset_ids)}


def write_order(rows_path: str, first_path: str, out: str, cost_path: str | None = None,
                weights_out: str | None = None) -> dict:
    first = json.loads(Path(first_path).read_text(encoding="utf-8"))["run_ids"]
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        lines = {json.loads(ln)["_evaluation"]["run_id"]: ln for ln in fh if ln.strip()}
    if not set(first) <= set(lines):
        raise SystemExit("first subset is not inside the rows file")
    ids = order(list(lines), first)
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as fh:
        for i in ids:
            ln = lines[i]
            fh.write(ln if ln.endswith("\n") else ln + "\n")
    rep = {"rows": len(ids), "first": len(first), "order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    if cost_path and weights_out:
        w = weights_for(json.loads(Path(cost_path).read_text(encoding="utf-8")), first)
        Path(weights_out).write_text(json.dumps(w), encoding="utf-8")
        rep["first_weight"] = sum(w.values())
    return rep


def parts(results: str, out: str, primary: str, full: str) -> dict:
    """Copy the complete, parseable, first-seen lines of a results file (a cap kill may cut the last line) and say
    whether the primary and the full subset are complete in it."""
    ids, keep = set(), []
    try:
        with open(results, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.endswith("\n"):
                    continue
                try:
                    rid = json.loads(ln)["run_id"]
                except (ValueError, KeyError, TypeError):
                    continue
                if rid not in ids:
                    ids.add(rid)
                    keep.append(ln)
    except FileNotFoundError:
        pass
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(keep)
    p = set(json.loads(Path(primary).read_text(encoding="utf-8"))["run_ids"])
    f = set(json.loads(Path(full).read_text(encoding="utf-8"))["run_ids"])
    return {"rows_written": len(ids), "primary_rows": len(p), "primary_done": p <= ids, "primary_missing": len(p - ids),
            "full_rows": len(f), "full_done": f <= ids, "full_missing": len(f - ids)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="half_subset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    c = sub.add_parser("costs")
    c.add_argument("--rows", required=True)
    c.add_argument("--out", required=True)
    o = sub.add_parser("order")
    o.add_argument("--rows", required=True)
    o.add_argument("--first", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--cost", default=None, help="qwen36_m0_cost.json: also write the first part's rate-guard weights")
    o.add_argument("--weights-out", default=None)
    q = sub.add_parser("parts")
    q.add_argument("--results", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--primary", required=True)
    q.add_argument("--full", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "parts":
        print(json.dumps(parts(a.results, a.out, a.primary, a.full)))
        return 0
    if a.cmd == "order":
        print(json.dumps(write_order(a.rows, a.first, a.out, a.cost, a.weights_out)))
        return 0
    d = build() if a.cmd == "build" else costs(a.rows)
    Path(a.out).write_bytes((json.dumps(d, indent=1) + "\n").encode())
    print(json.dumps({"rows": len(d["run_ids"]), "ids_sha256": d["ids_sha256"]} if a.cmd == "build"
                     else {"rows": len(d["rows"]), "weight": sum(r["weight"] for r in d["rows"].values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
