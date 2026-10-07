"""PRE-REGISTERED decision rule of the first Q9-AF decision cell (jobs/q9af_cell; Clef-flash-recipe analogue on the
attention-free Qwen3.5-9B: LoRA + fork head + LS-CE/Brier, the m2 150k draw).

Committed in its OWN commit before any q9af_cell run, before the full-8 export (job-2xe5w) existed and before any
Q9-AF cell number was seen. Every literal constant below is pinned by tests/test_q9af_cell_rule.py; an amendment is a
new dated commit made before the run it governs (never after its numbers are seen). Applied ONLY by
``jobs/q9af_cell/summarize.py`` (``decide`` per rank variant, ``compare_ranks`` across the r64 / r256 pair).

Sources of the constants (all written before this rule):
* Clef-flash on the same rows: 55.78 (2,008-row DEV subset, 0.2.1), 56.79 (full DEV, 0.2.1, job-sgihj), 57.35 (full
  DEV with the 51 over-length rows re-run at 32k, job-ftpnf; ``tests/clef/fixtures/job_ftpnf_result_full.json``).
* Stock Qwen3.5-9B readout 38.27 on the subset (INDEX_0.3.md section 3). PREMORTEM S8 go/no-go: an attention-free
  head below ~50 cannot reach Clef-flash; techniques/10 E2 anchor target >= 50 on the subset.
* G12 / G13 and their margins are ``train.m2_gates`` / ``train.promotion.MARGINS`` (PR #102, Codex #36 contract): this
  rule never re-derives or loosens them; it reads their exit code.

Binding numbers: Decision Index **0.2.1** (the edition every receipt and Clef-flash reference above is on). The 0.3
public-DEV index is REPORTED beside it, never decisive (INDEX_0.3.md: no 0.3 binding until the PREREG says so).

Verdicts of ONE variant (``decide``), first match wins:
  NO_MEASUREMENT  a required input is missing / not finite, the training stage did not finish (run_cell exits 21-29,
                  except 26), the gates did not run both G12 and G13, or the export pin is not filled.
  THROUGHPUT      training stopped on its measured rate (run_cell exit 26): re-price, not a recipe verdict.
  HYBRID_REPORT   the parent is a hybrid (partial conversion, attention kept): reported, NEVER promoted, never
                  compared with an attention-free claim; escalated to the user (decision-needed on shipping a hybrid).
  FOLD            the cell trained (pass_rule pass) but the paired full-DEV gain over its own parent is below
                  MIN_GAIN_PTS at the upper CI end, or the child's full-DEV index is below FOLD_BELOW_INDEX.
  ESCALATE        the child's full-DEV index (0.2.1) point estimate is >= CLEF_FLASH_FULL_DEV_021 (a Clef-flash-level
                  result: decision-needed before any PREREG / full Index / TEST unseal), or G13 rejected while the
                  capability gain is positive (a safety-vs-capability trade the user decides).
  PROMOTE         G12 and G13 accepted (m2_gates exit 0, approved margins), paired gain CI lower bound > 0, and the
                  full-DEV index >= PROMOTE_MIN_INDEX: the variant becomes a CANDIDATES.md row and the scale-up base.
  CONTINUE        everything else with a positive paired gain (CI lower > 0) and index >= CONTINUE_MIN_INDEX: a
                  scale-up cell may build on it, but it is not a candidate.
  FOLD            otherwise.

Rank pair (``compare_ranks``, E9 of techniques/10): paired child(r256) - child(r64) on full DEV (0.2.1):
  R256  CI lower > RANK_MIN_GAIN_PTS (r256 is better beyond noise: keep the 4x adapter)
  R64   otherwise (r64 is cheaper and transfers better per techniques/10 E9; ties go to r64)
  NO_MEASUREMENT if either variant is NO_MEASUREMENT / THROUGHPUT / HYBRID_REPORT (A1: or INCONCLUSIVE).

Amendment A1 (2026-10-07, after independent review, before any data). Committed alone (rule.py, its tests and the
JOB_SPEC rule text only) after the independent review of PR #121 (FIX-THEN-LAUNCH, rule edge cases); no q9af_cell
run, export or Index number existed. Nothing above is loosened; A1 only removes positive outcomes and pins inputs:
  (a) A G12 reject (gates rc 40, or 42 = both rejected) can never yield CONTINUE or PROMOTE: such a run is FOLD unless
      an ESCALATE condition holds first (the user decides a Clef-flash-level or safety-vs-capability result).
  (b) An underpowered G13 (gates rc 45) yields INCONCLUSIVE, never PROMOTE or CONTINUE. Action: expand the frozen
      evaluation (retention cohort) and re-gate the SAME checkpoint; the variant is no scale-up base meanwhile.
  (c) A run that failed its training-pass rule (run_cell pass_rule: finite, loss drop, eval NLL improves, memory)
      cannot PROMOTE (it may still be CONTINUE / FOLD / ESCALATE by the other conditions).
  (d) The bootstrap is part of the rule: B = BOOTSTRAP_B, seed = BOOTSTRAP_SEED for every CI the rule reads.
      ``decide`` requires the inputs to state them (``m["bootstrap"]``) and returns NO_MEASUREMENT otherwise.
  (e) Every verdict maps to an action (``ACTIONS``, pinned by the tests).
"""
from __future__ import annotations

import math
from typing import Any

RULE_ID = "q9af_cell/rule-v1"
EDITION_BINDING = "0.2.1"
EDITION_REPORTED = "0.3"
CLEF_FLASH_SUBSET_021 = 55.78
CLEF_FLASH_FULL_DEV_021 = 56.79
CLEF_FLASH_FULL_DEV_32K = 57.35
STOCK_Q9_SUBSET_021 = 38.27
PROMOTE_MIN_INDEX = 50.0
CONTINUE_MIN_INDEX = 44.0
FOLD_BELOW_INDEX = 40.0
MIN_GAIN_PTS = 3.0
RANK_MIN_GAIN_PTS = 0.0
#: run_cell exit codes (train/run_cell.py EXIT): 26 = measured too slow; the others are no-measurement stops
TRAIN_STOP_CODES = (21, 22, 23, 24, 25, 27, 28, 29, 35)
TRAIN_TOO_SLOW = 26
GATES_ACCEPT = 0
GATES_G13_REJECT = (41, 42)
VERDICTS = ("NO_MEASUREMENT", "THROUGHPUT", "HYBRID_REPORT", "FOLD", "ESCALATE", "PROMOTE", "CONTINUE")
#: --- Amendment A1 (2026-10-07, after independent review, before any data) ---
AMENDMENTS = ("A1",)
GATES_G12_REJECT = (40, 42)
GATES_UNDERPOWERED = 45
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
VERDICTS_A1 = VERDICTS + ("INCONCLUSIVE",)
ACTIONS = {
    "NO_MEASUREMENT": "fix the missing input / stop cause and re-run; no recipe conclusion",
    "THROUGHPUT": "re-price from the measured step time (run_cell exit 26); no recipe conclusion",
    "HYBRID_REPORT": "report as a hybrid (NOT attention-free); decision-needed issue to the user; never promoted",
    "FOLD": "stop this variant; log the negative result in CANDIDATES.md",
    "ESCALATE": "decision-needed issue to the user before any further spend, PREREG, full Index or TEST unseal",
    "PROMOTE": "CANDIDATES.md row; the variant is the scale-up base",
    "CONTINUE": "a scale-up cell may build on it; not a candidate",
    "INCONCLUSIVE": "expand the frozen retention evaluation and re-gate the same checkpoint; not a scale-up base",
}
RANK_VERDICTS = ("R256", "R64", "NO_MEASUREMENT")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def decide(m: dict[str, Any]) -> dict[str, Any]:
    """``m`` (built by summarize.py from the job outputs):
      export_pinned (bool), attention_free (bool), train_rc (int), train_pass (bool), gates_rc (int | None),
      gates_stages (list[str]), full_dev_child (float, 0.2.1 index point), full_dev_gain (dict point/lo/hi: paired
      child - parent, 0.2.1), capability_gain_positive (bool, optional; defaults to full_dev_gain lo > 0),
      bootstrap ({"B", "seed"}: A1 d, must equal BOOTSTRAP_B / BOOTSTRAP_SEED)."""
    why: list[str] = []
    if not m.get("export_pinned"):
        return {"verdict": "NO_MEASUREMENT", "why": ["export pin not filled"], "rule": RULE_ID}
    rc = m.get("train_rc")
    if rc == TRAIN_TOO_SLOW:
        return {"verdict": "THROUGHPUT", "why": ["run_cell exit 26 (measured too slow)"], "rule": RULE_ID}
    if rc in TRAIN_STOP_CODES or rc is None:
        return {"verdict": "NO_MEASUREMENT", "why": [f"training stopped (run_cell rc {rc})"], "rule": RULE_ID}
    gain = m.get("full_dev_gain") or {}
    idx = m.get("full_dev_child")
    if not (_num(idx) and all(_num(gain.get(k)) for k in ("point", "lo", "hi"))):
        return {"verdict": "NO_MEASUREMENT", "why": ["full-DEV index or paired gain missing / not finite"],
                "rule": RULE_ID}
    if sorted(m.get("gates_stages") or []) != ["g12", "g13"] or m.get("gates_rc") is None:
        return {"verdict": "NO_MEASUREMENT", "why": ["G12 and G13 were not both run"], "rule": RULE_ID}
    if m.get("bootstrap") != {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED}:  # A1 (d)
        return {"verdict": "NO_MEASUREMENT", "why": [f"bootstrap {m.get('bootstrap')} != the pinned "
                                                      f"B {BOOTSTRAP_B} / seed {BOOTSTRAP_SEED} (A1 d)"],
                "rule": RULE_ID}
    if not m.get("attention_free"):
        return {"verdict": "HYBRID_REPORT", "why": ["parent is a hybrid: attention layers kept; never promoted"],
                "rule": RULE_ID}
    trained = bool(m.get("train_pass"))
    if trained and (gain["hi"] < MIN_GAIN_PTS or idx < FOLD_BELOW_INDEX):
        why.append(f"trained, but gain CI upper {gain['hi']:.2f} < {MIN_GAIN_PTS} or index {idx:.2f} < "
                   f"{FOLD_BELOW_INDEX}")
        return {"verdict": "FOLD", "why": why, "rule": RULE_ID}
    cap_pos = m.get("capability_gain_positive", gain["lo"] > 0)
    if idx >= CLEF_FLASH_FULL_DEV_021:
        return {"verdict": "ESCALATE", "why": [f"index {idx:.2f} >= Clef-flash {CLEF_FLASH_FULL_DEV_021}: "
                                               "decision-needed before any PREREG / full Index"], "rule": RULE_ID}
    if m["gates_rc"] in GATES_G13_REJECT and cap_pos:
        return {"verdict": "ESCALATE", "why": ["G13 rejected with a positive capability gain"], "rule": RULE_ID}
    if m["gates_rc"] in GATES_G12_REJECT:  # A1 (a)
        return {"verdict": "FOLD", "why": [f"G12 rejected (gates rc {m['gates_rc']}): never CONTINUE / PROMOTE (A1 a)"],
                "rule": RULE_ID}
    if m["gates_rc"] == GATES_UNDERPOWERED:  # A1 (b)
        return {"verdict": "INCONCLUSIVE", "why": ["G13 underpowered (gates rc 45): never PROMOTE / CONTINUE (A1 b)"],
                "rule": RULE_ID}
    if m["gates_rc"] == GATES_ACCEPT and gain["lo"] > 0 and idx >= PROMOTE_MIN_INDEX and trained:  # A1 (c): trained
        return {"verdict": "PROMOTE", "why": ["G12+G13 accepted, gain CI > 0, index >= promote floor"],
                "rule": RULE_ID}
    if gain["lo"] > 0 and idx >= CONTINUE_MIN_INDEX:
        return {"verdict": "CONTINUE", "why": [f"gain CI > 0, index {idx:.2f} >= {CONTINUE_MIN_INDEX}; gates rc "
                                               f"{m['gates_rc']}"], "rule": RULE_ID}
    return {"verdict": "FOLD", "why": [f"gain CI lower {gain['lo']:.2f}, index {idx:.2f}"], "rule": RULE_ID}


def compare_ranks(v64: dict, v256: dict, paired_256_minus_64: dict | None) -> dict[str, Any]:
    """E9: ``paired_256_minus_64`` = paired full-DEV (0.2.1) child(r256) - child(r64): point / lo / hi."""
    bad = {"NO_MEASUREMENT", "THROUGHPUT", "HYBRID_REPORT", "INCONCLUSIVE"}  # A1: INCONCLUSIVE added
    if v64.get("verdict") in bad or v256.get("verdict") in bad or not paired_256_minus_64 \
            or not all(_num(paired_256_minus_64.get(k)) for k in ("point", "lo", "hi")):
        return {"verdict": "NO_MEASUREMENT", "rule": RULE_ID}
    if paired_256_minus_64["lo"] > RANK_MIN_GAIN_PTS:
        return {"verdict": "R256", "why": "r256 - r64 CI lower > 0", "rule": RULE_ID}
    return {"verdict": "R64", "why": "no r256 gain beyond noise: the cheaper adapter (ties go to r64)", "rule": RULE_ID}
