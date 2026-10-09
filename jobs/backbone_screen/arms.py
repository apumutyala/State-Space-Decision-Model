"""jobs/backbone_screen one-load modes (params ``screen.one_load_arms``, MODE qwen36_a3b): both readout arms over the
same proxy600 rows in ONE process, so the checkpoint is loaded once.

    python -X utf8 -m jobs.backbone_screen.arms <job_params.json> --model-dir SNAP --repo REPO --rows ROWS.jsonl.gz \
        --out-root DIR --selfcheck-dir DIR

Arm a (``screen.family``) writes DIR/stock/, arm b (``screen.arm_b.family``) writes DIR/v3w/: each is an unmodified
Decision Index kit run (``decision_index.runner.run``, the code behind ``python -m decision_index run``) of
``jobs.common.readout_engine:ReadoutIndexEngine``, so every arm is scored by the frozen scorer exactly like a separate
job (as jobs/rwkv7/readout_variants.py). The plug-in is cached per process (readout_engine._HF_PLUGINS). Each arm runs
its canary fork == recompute self-check before row one (selfcheck.json / selfcheck_b.json). A finished arm is not
re-run (kit resume). Exit 0 only when both arms completed; 12 when an arm raised.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ENGINE = "jobs.common.readout_engine:ReadoutIndexEngine"


def arm_options(params: dict, model_dir: str, repo: str, selfcheck_dir: Path) -> list[tuple[str, dict]]:
    """[(out subdir, engine options)] for arm a then arm b, from the params' screen block (pure; tested on CPU)."""
    s = params["screen"]
    common = {"model_dir": model_dir, "repo": repo, "input_cap": str(s["input_cap"])}
    common.update({k: str(v) for k, v in (s.get("engine_options") or {}).items()})
    a = {"family": s["family"], **common, "selfcheck_out": str(selfcheck_dir / "selfcheck.json")}
    b = s["arm_b"]
    bo = {"family": b["family"], **common, "backbone_id": b["backbone_id"], "v3w_sha256": b["v3w_sha256"],
          "selfcheck_out": str(selfcheck_dir / "selfcheck_b.json")}
    return [("stock", a), ("v3w", bo)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="backbone_screen.arms")
    ap.add_argument("params")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--selfcheck-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    from decision_index.runner import run

    summary = {}
    for name, opts in arm_options(params, a.model_dir, a.repo, a.selfcheck_dir):
        t0 = time.time()
        try:
            res = run(ENGINE, opts, Path(a.rows), a.out_root / name)
        except Exception as exc:  # noqa: BLE001
            summary[name] = {"family": opts["family"], "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"arm": name, **summary[name]}), flush=True)
            (a.out_root / "arms.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            return 12
        summary[name] = {"family": opts["family"], "result": res, "minutes": round((time.time() - t0) / 60, 2)}
        print(json.dumps({"arm": name, **summary[name]}, default=str), flush=True)
        (a.out_root / "arms.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
