"""Paired comparison of a candidate against Clef (27B) and Clef-flash (COORDINATION.md "Full verification").

    python -X utf8 -m analysis.paired_vs_clef --clef <Clef results.jsonl> --clef-flash <Clef-flash results.jsonl> \
        [--candidate <cand results.jsonl>] [--rows full|subset] [--context 16k|32k] [--out report.json]

Inputs are per-row decide results files (the kit's results.jsonl).
  * --rows full   (default): the 12,104-row DEV slice (``harness.cli.load_slice("dev")``).
  * --rows subset: the pre-registered 2,008-row m1 DEV subset (``jobs/common/dev_subset_m1.json``), so a candidate
    scored only on the subset (stock RWKV7, the R1 rescore) can be paired with Clef now. A full-DEV results file
    (e.g. job-sgihj) is accepted on the subset and restricted to the subset ids; the dropped rows are reported.
With no --candidate, the report is Clef 27B vs Clef-flash (the R2 comparator row).

Nothing is re-implemented here: rows and results go through ``harness.cli`` (strict reader), points and CIs through
``harness.score.decompose`` + ``harness.bootstrap.compare`` (case-cluster bootstrap; every pairing uses the same B and
seed, so the diffs share replicate weights), ECE through ``harness.calibration``.

Pairing is by run_id: after loading, every file must answer EXACTLY the slice's run_id set (asserted; extra rows are
an error on --rows full). ``--allow-missing`` is the only way past that and is recorded in the output.

Over-length context: the 51 DEV rows over Clef's shipped max_length 16,384 (jobs/clef_baseline/overlength_rows.json)
are ``unsupported`` in a 16k file and scored (status ok, input_tokens > 16,384) in a 32k-substituted file. The context
of every input is detected from those rows and must be the same for Clef and Clef-flash (pair 32k with 32k; the
candidate's is reported, not enforced: a non-Clef encoder has its own length limit); --context, when given, must
match what was detected. Reference files:
  * Clef-flash 16k: job-sgihj decide results (full DEV 56.79). 32k: job-ftpnf decide_32k_substituted (57.35).
  * Clef 27B: job index/decide/results.jsonl (16k) or index/decide_32k_substituted/results.jsonl (32k).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # runnable as `python -I analysis/paired_vs_clef.py` (no cwd on sys.path)
    sys.path.insert(0, str(ROOT))

from harness import config as _config  # noqa: E402

try:  # the kit (pinned clone under SSDM_DATA) when it is not installed in this interpreter
    import decision_index  # noqa: F401
except ImportError:
    if (_config.KIT_CLONE / "decision_index").is_dir():
        sys.path.insert(1, str(_config.KIT_CLONE))

from harness import bootstrap, calibration, cli, config, score  # noqa: E402
from harness.jsonl import read_results_strict  # noqa: E402

OVERLENGTH = ROOT / "jobs" / "clef_baseline" / "overlength_rows.json"
SHIPPED_MAX = 16384
SYSTEMS = ("candidate", "clef", "clef_flash")


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slice_rows(mode: str, edition: str | None, gsm8k03: bool):
    """(rows, meta) for the DEV slice or the m1 DEV subset; both via the repo's own loaders."""
    if mode == "full":
        return cli.load_slice("dev", edition, gsm8k03), {"slice": "dev"}
    if mode == "subset":
        from jobs.common import dev_subset

        sub = dev_subset.load_subset()
        return dev_subset._rows(sub, edition, gsm8k03), {"slice": "dev_subset_m1", "subset_ids_sha256": sub["ids_sha256"]}
    raise ValueError(f"rows mode {mode!r}: full or subset")


def overlength_ids() -> set[str]:
    return {r["run_id"] for r in json.loads(OVERLENGTH.read_text(encoding="utf-8"))["rows"]}


def detect_context(raw: dict, ids: set[str]) -> dict:
    """16k / 32k from the over-length rows: 16k files have them unsupported (input_tokens_exceeded), 32k files ok with
    usage.input_tokens > 16,384. Anything else is 'mixed' (refused unless --allow-mixed-context)."""
    ok = unsup = other = 0
    for rid in sorted(ids):
        r = raw.get(rid)
        if r is None:
            other += 1
            continue
        tok = ((r.get("response") or {}).get("usage") or {}).get("input_tokens")
        if r.get("status") == "ok" and tok is not None and tok > SHIPPED_MAX:
            ok += 1
        elif r.get("status") != "ok" and "input_tokens_exceeded" in str(r.get("error", "")):
            unsup += 1
        else:
            other += 1
    n = len(ids)
    ctx = "none" if n == 0 else "32k" if ok == n else "16k" if unsup == n else "mixed"
    return {"context": ctx, "overlength_rows_in_slice": n, "scored_over_16k": ok, "unsupported_16k": unsup, "other": other}


def _load(rows, path, mode, allow_missing):
    """Strict load, restricted to the slice ids, with the identical-row-set assertion."""
    ids = cli._ids(rows)
    want = set(ids)
    raw = read_results_strict(path, ids, allow_partial=allow_missing)
    extra = sorted(set(raw) - want)
    if extra and mode == "full":
        raise ValueError(f"{path}: {len(extra)} rows are not in the DEV slice (e.g. {extra[:3]}): wrong slice/edition?")
    res = cli.load_results(path, ids, allow_missing)
    res = {k: v for k, v in res.items() if k in want}
    if not allow_missing:
        assert set(res) == want, f"{path}: row set differs from the slice ({len(want - set(res))} missing)"
    score.require_results(rows, res, allow_missing, path)
    return raw, res, len(extra)


def _ece(rows, res) -> dict:
    r = calibration.report(calibration.fields(rows, res))
    return {"board_protocol_ece": r["board_protocol"].get("ece"), "overall_ece_10bin": r["overall"].get("ece_10bin"),
            "overall_smece": r["overall"].get("smece"), "full": r}


def run(clef: str, clef_flash: str, candidate: str | None = None, rows_mode: str = "full", edition: str | None = None,
        B: int = 2000, seed: int = 0, allow_missing: bool = False, omit_gsm8k03: bool = False,
        context: str | None = None, allow_mixed_context: bool = False) -> dict:
    rows, meta = slice_rows(rows_mode, edition, not omit_gsm8k03)
    ids = set(cli._ids(rows))
    files = {k: p for k, p in zip(SYSTEMS, (candidate, clef, clef_flash)) if p}
    loaded = {k: _load(rows, p, rows_mode, allow_missing) for k, p in files.items()}
    res = {k: v[1] for k, v in loaded.items()}
    # identical row sets on every side (pair by run_id; the strict reader already refused duplicates)
    sets = {k: frozenset(r) for k, r in res.items()}
    if not allow_missing:
        assert len(set(sets.values())) == 1, {k: len(s) for k, s in sets.items()}
    ol = overlength_ids() & ids
    ctx = {k: detect_context(v[0], ol) for k, v in loaded.items()}
    # the context rule binds the two Clef references (same tokenizer, same 16,384 limit); a non-Clef candidate's
    # over-length rows are reported for information only (its own encoder decides what it can read)
    seen = {ctx[k]["context"] for k in ("clef", "clef_flash")}
    if not allow_mixed_context:
        if "mixed" in seen or len(seen) > 1:
            raise ValueError(f"over-length context differs between the Clef references (pair 32k with 32k): "
                             f"{ {k: c['context'] for k, c in ctx.items()} }")
        if context and ol and seen != {context}:
            raise ValueError(f"--context {context} but the inputs are {seen}")
    benches = {k: score.decompose(rows, r, edition=edition) for k, r in res.items()}
    out = {"edition": config.edition_id(edition), "rows_mode": rows_mode, **meta, "rows": len(rows), "B": B,
           "seed": seed, "allow_missing": allow_missing, "context": ctx,
           "files": {k: {"path": str(Path(p).resolve()), "sha256": _sha(p), "rows_outside_slice_dropped": loaded[k][2]}
                     for k, p in files.items()},
           "index": {}, "vs": {}, "ece": {}}
    pairs = [("candidate", "clef"), ("candidate", "clef_flash")] if candidate else [("clef", "clef_flash")]
    for cand, ref in pairs:
        c = bootstrap.compare(benches[cand], benches[ref], B=B, seed=seed, allow_unaligned=allow_missing)
        key = ref if candidate else f"{cand}_vs_{ref}"
        out["vs"][key] = {"diff_index": c["diff"]["index"], "diff_areas": c["diff"]["areas"],
                          "diff_benchmarks": c["diff"]["benchmarks"]}
        out["index"].setdefault(cand, {"index": c["candidate"]["index"], "areas": c["candidate"]["areas"]})
        out["index"].setdefault(ref, {"index": c["baseline"]["index"], "areas": c["baseline"]["areas"]})
    for k, r in res.items():
        out["ece"][k] = _ece(rows, r)
    if candidate:
        out["beats_both"] = all(out["vs"][r]["diff_index"]["lo"] > 0 for r in ("clef", "clef_flash"))
    return out


def _fmt(ci: dict) -> str:
    return f"{ci['point']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


def table(out: dict) -> str:
    """Markdown paired table: per-system Index + board ECE, then each paired diff."""
    lines = [f"rows: {out['rows_mode']} ({out['rows']} rows, edition {out['edition']}, B={out['B']}, seed={out['seed']}); "
             f"context: { {k: c['context'] for k, c in out['context'].items()} }",
             "", "| system | Index [95% CI] | board ECE | ECE 10-bin |", "|---|---|---|---|"]
    for k, v in out["index"].items():
        e = out["ece"][k]
        lines.append(f"| {k} | {_fmt(v['index'])} | {e['board_protocol_ece']:.4f} | {e['overall_ece_10bin']:.4f} |")
    lines += ["", "| pairing | paired diff Index [95% CI] | p(diff<=0) |", "|---|---|---|"]
    for k, v in out["vs"].items():
        d = v["diff_index"]
        lines.append(f"| {k if '_vs_' in k else 'candidate_vs_' + k} | {d['point']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] "
                     f"| {d.get('p_le_zero', float('nan')):.4f} |")
    return "\n".join(lines)


def brief(out: dict) -> dict:
    b = {"index": {k: v["index"] for k, v in out["index"].items()},
         "diff_vs": {k: v["diff_index"] for k, v in out["vs"].items()},
         "board_ece": {k: v["board_protocol_ece"] for k, v in out["ece"].items()},
         "context": {k: c["context"] for k, c in out["context"].items()}}
    if "beats_both" in out:
        b["beats_both"] = out["beats_both"]
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.paired_vs_clef", description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", help="omit for Clef 27B vs Clef-flash")
    ap.add_argument("--clef", required=True)
    ap.add_argument("--clef-flash", required=True)
    ap.add_argument("--rows", choices=("full", "subset"), default="full")
    ap.add_argument("--context", choices=("16k", "32k"))
    ap.add_argument("--allow-mixed-context", action="store_true")
    ap.add_argument("--edition", default=config.EDITION, choices=config.EDITIONS)
    ap.add_argument("--omit-gsm8k03", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true", help="score missing rows as wrong (unaligned pairing)")
    ap.add_argument("--out")
    ap.add_argument("--table", action="store_true", help="also print the markdown paired table")
    a = ap.parse_args(argv)
    out = run(a.clef, a.clef_flash, a.candidate, a.rows, a.edition, a.B, a.seed, a.allow_missing, a.omit_gsm8k03,
              a.context, a.allow_mixed_context)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(brief(out), indent=1))
    if a.table:
        print(table(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
