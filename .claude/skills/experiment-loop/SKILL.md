---
name: experiment-loop
description: The iterate-to-beat-Clef loop for RWKV7: pick the best next hypothesis, run several in parallel, keep what wins, compound. Use every session while Stage 1 is open.
---

# Experiment loop

**Each loop:**
1. Reread GOAL (the Clef numbers) and the BOARD top. Pick hypotheses by expected Index gain ÷ cost, and favour ones that attack the largest measured gap. Per-area gap vs Clef-flash: knowledge, arts, retrieval, tools, language. Read the latest ledger row for the current numbers.
2. Keep 2–4 GPU jobs in flight. Each job tests ONE change relative to the current kept parent, so results attribute cleanly. While they run, package the next ones.
3. When a job lands (`land-result`):
   - **keep:** it becomes the parent for the next variants;
   - **discard:** add it to DEAD_ENDS;
   - **near-miss:** combine two near-misses into one run.
4. When a cell is within about 5 pts of 55.78 on the DEV subset, run full DEV on it plus the paired comparison (`analysis/paired_vs_clef.py`, currently only on branch `stack/clef/full-dev`, not agent-main). That is the only heavy verification.

**Lever menu, roughly in order of expected value:**
- **Readout format fixed to RWKV's documented chat format** (single-token answer codes, `\n\n` collapsed, token 0, G1k no-think suffix). Applies to every later cell.
- **Distillation** from the Clef-flash soft labels (alpha and T sweeps).
- **More training rows:** 24k → 45k → 100k+ from the allowed pools. Weight the draw toward the weak areas. Add the high-k intent views (BANKING77/CLINC150).
- **LoRA rank 64 → 256.** Also consider full-layer unfreezing of the top blocks if memory allows. Tune LR and epochs.
- **Loss mix:** LS-CE + Brier for calibration. Keep ECE in check: the final claim needs calibration.
- **Option-order shuffle** as cheap augmentation only. Order bias was shown to be mostly label bias (PR #151).

**Stall rule:** if two consecutive cells fail to beat their parent, or the gap is still above 15 pts after format + KD, write it on BOARD and tell the operator. The parked fallback is issue #140 option (b), re-converting Qwen layers 3 and 27.
