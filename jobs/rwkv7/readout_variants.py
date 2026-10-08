"""jobs/rwkv7 JOB=readout_g1v3 (R3): several readout variants over the same rows in ONE process (one model load).

    python -X utf8 -m jobs.rwkv7.readout_variants --model-dir SNAP --rows ROWS.jsonl.gz --out-root DIR \
        --variants v2,v2f,v3,v3w,v3r,v3p,v1,v3c

Each variant is an unmodified Decision Index kit run (``decision_index.runner.run``, the code behind
``python -m decision_index run``) of ``jobs.common.readout_engine:ReadoutIndexEngine`` with its own readout family,
written to ``DIR/<variant>/`` (results.jsonl, environment.json, status.json), so every variant is scored by the frozen
scorer exactly like a separate job. The RWKV7 plugin is cached per process (readout_engine._RWKV7_PLUGINS), so the
26 GB model is loaded once and every variant reads the same weights. A finished variant is not re-run (kit resume).
Exit 0 only when every variant completed; 12 when a variant raised.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ENGINE = "jobs.common.readout_engine:ReadoutIndexEngine"
VARIANTS = {"v2": "rwkv7-hf-g1-v2", "v2f": "rwkv7-hf-g1-v2f", "v3": "rwkv7-hf-g1-v3", "v3w": "rwkv7-hf-g1-v3w",
            "v3r": "rwkv7-hf-g1-v3r", "v3p": "rwkv7-hf-g1-v3p", "v1": "rwkv7-hf", "v3c": "rwkv7-hf-g1-v3c",
            "v3j": "rwkv7-hf-g1-v3j", "v3js": "rwkv7-hf-g1-v3js"}  # R4 joint arms: opt-in, not in DEFAULT_ORDER
#: run order = priority (coordinator 2026-10-08): v2 control (byte-identical to job-zhyvg), v2f (v2 + dict-instruction
#: fix), v3 code key, v3w word ids, v3r re-read, v3p permutation, v1 (jrt-rwkv-g1-v1, for a later CPU-side
#: pre-declared arithmetic-mean ensemble v1 + v3 on identical rows), v3c calibration
DEFAULT_ORDER = "v2,v2f,v3,v3w,v3r,v3p,v1,v3c"


def parse_variants(spec: str) -> list[str]:
    out = [v.strip() for v in spec.split(",") if v.strip()]
    bad = [v for v in out if v not in VARIANTS]
    if not out or bad or len(set(out)) != len(out):
        raise SystemExit(f"--variants must be distinct names from {sorted(VARIANTS)}; got {spec!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="readout_variants")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--variants", default=DEFAULT_ORDER)
    a = ap.parse_args(argv)
    variants = parse_variants(a.variants)
    from decision_index.runner import run

    summary = {}
    for v in variants:
        t0 = time.time()
        try:
            res = run(ENGINE, {"family": VARIANTS[v], "model_dir": a.model_dir}, Path(a.rows), a.out_root / v)
        except Exception as exc:  # noqa: BLE001
            summary[v] = {"family": VARIANTS[v], "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"variant": v, **summary[v]}), flush=True)
            (a.out_root / "variants.json").write_text(json.dumps(summary, indent=1))
            return 12
        summary[v] = {"family": VARIANTS[v], "result": res, "minutes": round((time.time() - t0) / 60, 2)}
        print(json.dumps({"variant": v, **summary[v]}), flush=True)
        (a.out_root / "variants.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
