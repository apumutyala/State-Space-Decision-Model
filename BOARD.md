# BOARD: top items (Agent A orders it; B appends proposals at the bottom). Rewrite each loop.

Targets (DEV 2,008 subset): Clef-flash 55.78 · Clef 62.14. Current best attention-free: stock RWKV7 24.85.

| id | lane | hypothesis | exp. gain | cost | owner / status |
|---|---|---|---|---|---|
| D1 | A | First trained cell: LoRA r64 + fork head, gold labels, 24k rows | +10–18 | ~$15–31 | running: job-c7pq2 (since 06:45Z, ends ≤17:00Z) |
| D2 | A | D1 + Clef-flash soft-label KD (α .5, T 1), PR #200 | +1–3 over D1 | ~$15–40 | running: job-zhv3f (submitted ~10:00Z) |
| R1 | B | C1 readout format: RWKV chat format, single-token codes, token 0, `\n\n`→`\n`, whole-prompt tokenization; rescore stock parent | +3–10 | ~$2–10 | in progress (branch stack/rwkv7/c1-format) |
| R2 | B | Clef 27B full-DEV comparator + `analysis/paired_vs_clef.py` | needed for the final claim | ~$10–20 | in progress (branch stack/clef/full-dev) |
| D3 | A | Winner of D1/D2 on the R1 format (if R1 wins) | compounding | ~$15–31 | next, once D1/D2 and R1 land |
| D4 | A | Scale the draw 24k → 45k+, weighted to knowledge, arts, retrieval; add BANKING77/CLINC150 intent views | +3–6 | ~$30–60 | prep now (B builds the data draw) |
| D5 | A | LoRA r256 vs r64; LR/epoch sweep on the kept parent | +1–4 | ~$15–31 each | queued |
| B1 | B | 45k+ training draw + intent views, excluding TEST / held-out / cohort v3; teacher export for it if KD wins | enables D4 | ~$5 GPU | queued |
| Q1 | — | Parked: Q9 proxy loop (best SELECT 39.24 < 50); Q9-AF option (b) | — | — | parked (#140 stall trigger) |
| L1 | C | Land checked L0–L4 modules; prepare executable paired language, retention, soundness and intervention gates for the Stage 1 champion | Stage 2 measurement readiness | $0 CPU | agent-c: active, three bounded workers; no GPU |
| L2 | C → A/B | Champion handoff: preserve serving bundle, exact base/tokenizer/runtime/calibration hashes, Stage 1 qualification evidence and matched DEV parent predictions | enables strict loading and same-item retention | $0 prep | proposal; consume first keep diagnostically, qualify only after Stage 1 gates pass |
