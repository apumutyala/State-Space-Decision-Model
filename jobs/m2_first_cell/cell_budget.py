"""Cell-phase budget (job-wm8b5 exit 7, 2026-10-12): the cell cap must cover the training clock AND everything run_cell
does outside it, or the phase timeout kills the run after the clock with the selected weights unexported.

  cell minutes needed = pre-train (snapshot load + tokenisation + the T ladder + smoke + the step-0 eval / dev cards)
                      + the training clock (max_minutes; the in-loop dev cards run INSIDE it and cost steps, not wall)
                      + one dev card past the clock (a card that starts just before the clock ends runs to its end)
                      + the final eval card + the final save + the inference export
                      + margin

The constants are MEASURED (``MEASURED``; source job-wm8b5: stock Qwen3.5-4B, lm_label, g1v2-canon, H100):
pre-train 11.6 min (cell start 20:06:57 -> first step 20:18:33), a 360-row dev card 41 s (88 s at step 0, which is
inside the pre-train figure), 5.07 s/step mean. ``check`` is run by scripts/cell_preflight.py and the tests BEFORE any
GPU job; it fails if the cap is too small, and reports the steps the clock buys after the in-loop dev cards.
"""
from __future__ import annotations

import math

MEASURED = {"pre_train_min": 11.6, "dev_card_s": 41.0, "final_card_min": 1.5, "save_export_min": 1.5,
            "margin_min": 2.0, "s_per_step": 5.07,
            "source": "job-wm8b5 (stock Qwen3.5-4B + LoRA r16 + lm_label, g1v2-canon, jev3, H100)"}


def required_cell_minutes(cfg: dict, m: dict = MEASURED) -> dict:
    clock = float(cfg["max_minutes"])
    one_card = m["dev_card_s"] / 60.0 if cfg.get("dev_card_every") else 0.0
    need = m["pre_train_min"] + clock + one_card + m["final_card_min"] + m["save_export_min"] + m["margin_min"]
    every = int(cfg.get("dev_card_every") or 0)
    steps_no_cards = clock * 60.0 / m["s_per_step"]
    n_cards = math.floor(steps_no_cards / every) if every else 0
    steps = max(0.0, (clock * 60.0 - n_cards * m["dev_card_s"]) / m["s_per_step"])
    return {"need_min": round(need, 2), "clock_min": clock, "pre_train_min": m["pre_train_min"],
            "after_clock_min": round(one_card + m["final_card_min"] + m["save_export_min"], 2),
            "margin_min": m["margin_min"], "in_clock_dev_cards": n_cards,
            "expected_steps": int(min(steps, float(cfg.get("max_steps") or steps))), "measured": m}


def check(params: dict, cfg: dict, m: dict = MEASURED) -> dict:
    r = required_cell_minutes(cfg, m)
    cap = float(params["phase_caps_minutes"]["cell"])
    r.update({"cap_min": cap, "pass": cap >= r["need_min"], "max_clock_for_cap": round(cap - (r["need_min"] - r["clock_min"]), 2)})
    return r


def main(argv=None) -> int:
    """``python -X utf8 -m jobs.m2_first_cell.cell_budget --params P --cell-config C``: the job's SETUP gate (exit 15
    before any training spend when the cell cap is below the measured budget); prints the check."""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--cell-config", required=True)
    a = ap.parse_args(argv)
    r = check(json.loads(Path(a.params).read_text(encoding="utf-8")),
              json.loads(Path(a.cell_config).read_text(encoding="utf-8")))
    print(json.dumps(r))
    return 0 if r["pass"] else 15


if __name__ == "__main__":
    raise SystemExit(main())
