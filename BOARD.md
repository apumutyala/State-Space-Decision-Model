# BOARD: top items (Agent A orders it; B and C append proposals at the bottom). Rewrite each loop.

Targets (DEV 2,008 subset): Clef-flash 55.78 · Clef 62.14. Current best attention-free: stock RWKV7 24.85.
Loss map (research-notes/STOCK_RWKV7_LOSS_BREAKDOWN.md, measured): Clef-flash-answer headroom is tools +9.3, knowledge +9.5 (7.1 procedural, 2.4 facts), retrieval +4.4, language +4.3, arts +3.4. Parse failures 0%. Main failures: label-prior collapse, and long catalogs (>16 options: 31% vs 59%). Gold is 2nd choice on 74% of the questions Clef-flash gets right and RWKV misses.

| id | lane | hypothesis | exp. gain | cost | owner / status |
|---|---|---|---|---|---|
| D1 | A | First trained cell: LoRA r64 + fork head, gold labels, 24k rows | +10–18 | ≤$31.5 | running: job-c7pq2 (cell since 06:45Z) |
| D2 | A | D1 + Clef-flash KD (α .5, T 1), PR #200 | +1–3 over D1 | ≤$40 | running: job-zhv3f (09:41Z). Caveat: teacher follows injected instructions (d1a 23%); read its steer score accordingly |
| R1 | B | C1 readout format v2 (single-token codes etc.); rescore stock parent | +2–6 | ~$2 | B: job-zhyvg |
| R3 | B | Long-catalog binding + label-prior readout: single-token codes, catalog repeated next to the answer slot for k>16, fit-free contextual calibration (basis: >16 options 31% vs 59%) | +3–7 (B est.) | ~$3–6 | B: designing; rescore after R1 |
| D3 | A | D1/D2 winner on the R1/R3 format; KD masked on injected rows (stack/rwkv7/kd-injmask) | compounding | ≤$31.5 | next, once D1/D2 and R1 land |
| D6 | A | D1 + option-order permutation (kills position/label prior) | +2–4 | ≤$31.5 | packaging (stack/rwkv7/d6-order-perm) |
| D4 | A | 45k draw re-aimed: tools > procedural knowledge > BANKING77/CLINC > retrieval; uniform gold positions | +3–6 | ~$30–60 | mode on stack/rwkv7/d4-draw45k; waiting on B1 |
| B1 | B | 45k/100k draw (re-aimed as in D4), exclusions vs TEST/held-out/cohort v3 | enables D4 | ~$0–5 | B: stack/rwkv7/b1-draw |
| D5 | A | LoRA r256; LR 2x (measured lever estimate only ≤ +1–2) | ≤ +1–2 | ≤$31.5 each | packaged; low priority, fills idle GPU only |
| R2 | B | Clef 27B full-DEV comparator + analysis/paired_vs_clef.py | final claim | ~$10–20 | B: job-jxevr |
| Q1 | — | Parked: Q9 proxy loop; Q9-AF option (b) | — | — | parked (#140 stall trigger) |

## Proposals (B/C append below)
