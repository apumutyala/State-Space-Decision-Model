# Which of the 373 branches matter for Keel? (triage, 2026-10-15)

Method: `git branch -r --no-merged origin/agent-main`, commits ahead, last commit, files touched; lineage checks with
`git merge-base --is-ancestor`; dry-run merges with `git merge-tree` against `stack/qwen35/jev3-lm-r`.

**Bottom line:**
- 156 branches are already merged into `agent-main`, so their code is there.
- Of the 214 unmerged code branches, 25 matter for Keel:
  - 8 to build on;
  - 5 more already contained in the top one;
  - 12 as tools or references.
  The other 189 are history.
- Nothing is deleted; branch deletion needs the owner's approval.

## Build on (Tier 1)

| Branch | What it gives Keel | Status |
|---|---|---|
| `stack/qwen35/jev3-lm-r` | Tip of the Qwen3.5-4B lineage. It already contains `sizecliff` (the four 4B memory/fork fixes), `sym-invariant` (exact reorder/relabel invariance), `consistency` (view generator), `jev3-sym` and `jev3-lm` (answer-slot readout, content-canonical inference via `decide/canon_shim.py`, the fixed runner). It is packaged for the never-completed full-epoch run | base for Keel-4B |
| `stack/qwen35/nupi` | νPI (PID-Lagrangian) constraint controller: `train/pi_control.py`, `train/constraints.py`, `docs/NUPI.md`; CPU-tested, never trained | merge; conflicts in `train/trainer.py`, `train/run_cell.py` |
| `stack/qwen35/jev3-sym-nu` | Work in progress wiring νPI into the sym lineage (a constraint on yes/no rows) | reference for the merge above |
| `stack/qwen35/g16-sym` | Order-symmetric trusted-stream head: `controlsafe/g16_sym.py`, `g16_sym_plug.py` (bet B) | merge; 1 test conflict |
| `claude/ev-guarantee-proofs` | Cluster-valid LTT/CRR in `jevkit/guarantees.py` (`cluster_of`, `cluster_test`), plus `docs/guarantees/` (conformal, injection-robust conformal, reorder invariance, review). `agent-main` still has the invalid per-decision test | merge (clean) |
| `claude/ev-jev3-draw` | Builder for human-labelled TRAIN rows (judge_hard, adequacy, policy), with provenance re-checks; the best data foundation for triplet construction. The data files live in `ssdm-data` on the owner's machine | merge (clean) |
| `claude/ev-jev-heldout` | Builder and gates for the sealed 2,199-item confirmation set | merge (clean); the data stays sealed |
| `stack/clef/safety-heldout` | 21 blind held-out attack templates (`analysis/safety_heldout/heldout_v1_templates.json`), training-draw attack audit, design doc | merge; conflicts only in old SF job specs |

## Tools and references (Tier 2)

| Branch | Use |
|---|---|
| `claude/ev-score-inv` (includes `claude/ev-cell-kit`) | One-command rescore and guarantee report per landing; score-item invariance operator (clean merge) |
| `claude/ev-sym-check` | Independent verification harness for invariance and the trust gate (real-tokenizer fuzzing) |
| `stack/qwen35/sym-fast` | Faster invariant serving (2 calls or 1 packed call per decision); merge when latency matters |
| `claude/ev-flip-rate` | Flip-rate battery report wiring in `jevkit/cli.py` |
| `claude/safety-spec` | `SAFETY_SPEC.md` guarantee table; to be rewritten to the scoped claims |
| `claude/ev-review-stability`, `claude/lead-state-stability`, `stack/rwkv7/sym-contractive` | State-stability theorems, their review, and the contractive RWKV-7 variant; relevant only to bet E (long-horizon state-space monitors) |
| `claude/ev-evidence-report-v2`, `claude/lead-tech-report`, `claude/lead-gap-attribution` | The most complete written record of what was measured and why it failed |

## Leave as history (Tier 3, 189 branches)

| Group | Count | Why not |
|---|---|---|
| `stack/rwkv7/*` Decision-Index cells (D1–D20, readouts, KD) | 53 | the attention-free 13.3B line; results are in the ledger |
| `codex/*` Track L language growth | 45 | frozen; no qualified level |
| `codex/*` runner, lint and launch-policy fixes | 20 | SF Compute job runner, replaced by per-second GPUs |
| `stack/q9*` attention→GDN conversion | 5 | recall collapse; closed |
| `claude/b-*` Index-era agent-b work | 16 | Index-targeted draws, readouts and receipts |
| `claude/ev-*` rescore and receipt branches | 24 | landed numbers, recorded on `state` and in the evidence report |
| Older 4B cells (`jev1-4b`, `jev1-diag`, `jev2-4b`, `jev3-typed`, H2O-host `jev2-h2o`, `jev3-h2o`) | 6 | superseded by `jev3-lm-r`; H2O-host only if a control cell is approved |
| Misc (`artifact-size-cap-*`, CPU-test and runner branches, scouts, onboarding, backup, two `stack/clef` baselines) | 20 | infrastructure or one-off |

## Proposed consolidation

Create one integration branch (e.g. `collab/mottopanikeiku/keel-base`) from `stack/qwen35/jev3-lm-r`. Merge, in order:

1. `claude/ev-guarantee-proofs`, `claude/ev-score-inv`, `claude/ev-jev3-draw`, `claude/ev-jev-heldout` — all clean.
2. `stack/clef/safety-heldout` — take ours for the old SF job specs.
3. `stack/qwen35/g16-sym` — one test conflict.
4. `stack/qwen35/nupi` — the only real code conflict, in the trainer.
5. These docs.

Then run the CPU test suite once. Expected effort: about half a day of agent time and $0.
