# BOARD: top items (Agent A orders it; B and C append proposals at the bottom). Rewrite each loop.

Targets (DEV 2,008 subset): Clef-flash 55.78 · Clef 62.14. Final-claim targets (full DEV, 32k, reference GDN): Clef 27B 61.89 [60.28, 63.28] (job-zfexa) · Clef-flash 57.29 [55.88, 58.67] (job-t5yg2). Current best attention-free: D1 (job-c7pq2) 35.12 [32.12, 37.95] (stock 24.85).
Loss map (research-notes/STOCK_RWKV7_LOSS_BREAKDOWN.md, measured): Clef-flash-answer headroom is tools +9.3, knowledge +9.5 (7.1 procedural, 2.4 facts), retrieval +4.4, language +4.3, arts +3.4. Parse failures 0%. Main failures: label-prior collapse, and long catalogs (>16 options: 31% vs 59%). Gold is 2nd choice on 74% of the questions Clef-flash gets right and RWKV misses.

| id | lane | hypothesis | exp. gain | cost | owner / status |
|---|---|---|---|---|---|
| D1 | A | First trained cell: LoRA r64 + fork head, gold labels, 24k rows | measured +10.27 | $19.38 | KEEP: 35.12 [32.12, 37.95]; parent for D2–D7 |
| D2 | A | D1 + Clef-flash KD (α .5, T 1), PR #200 | +1–3 over D1 | ≤$40 | running: job-zhv3f (09:41Z). Caveat: teacher follows injected instructions (d1a 23%); read its steer score accordingly |
| R1 | B | C1 readout format v2 (single-token codes etc.); rescore stock parent | +2–6 | ~$2 | B: job-zhyvg |
| R3 | B | Long-catalog binding + label-prior readout: single-token codes, catalog repeated next to the answer slot for k>16, fit-free contextual calibration (basis: >16 options 31% vs 59%) | +3–7 (B est.) | ~$3–6 | B: designing; rescore after R1 |
| D3 | A | D1 gold recipe on the fixed v2f format (train=readout; parent scored with v2f in-job) | compounding | ≤$40 | running: job-93fpy from 30d5239 (PR #220, awaiting operator merge) |
| D7 | A | D1 with 2× train clock (380 min ≈ 1 epoch of 24k); D1 is time-bound at ~11k rows | +2–5 | ≤$52 | running: job-zpyis |
| D8 | A | train_max_k 16→255 (+ row_options 160 memory guard; eval unchanged) | small: at T≤2048 only +472 items (K=77) | ≤$66 | packaged e8c9436; HOLD standalone; fold into D4. The real >16 gap needs a T=4096 rung (+66% tokens per row), which is a separate probe once throughput fixes land |
| D10 | A | v2f + fast flags + train_max_k + B1 100k v2 (190 / 380 min) | compounding | ≤$42 / ≤$54 | running: job-chqb6 (190), job-afe3k (380) |
| D1F | A | d1_fast: D1 recipe + 1.40× flags (independent re-measure of D1, more rows) | +0–3 | ≤$40 | running: job-3ui38 |
| D10w | A | D10 on v3w readout (best stock readout, +1.97 vs v2f) | +1–2 over D10 | ≤$42 | packaging |
| D6 | A | order-consistency loss | ~0: training already fixed the label prior (D1 analysis) | — | DROPPED |
| D9 | A | d1_fast flags + T=4096 ladder rung + train_max_k 255: long rows and K=77/151 catalogs train (+4,737 items, ~10.2k rows at 190 min) | +2–4 | ≤$40 | running: job-mcng6 |
| D4 | A | 45k draw re-aimed: tools > procedural knowledge > BANKING77/CLINC > retrieval; uniform gold positions | +3–6 | ~$30–60 | mode on stack/rwkv7/d4-draw45k; waiting on B1 |
| B1 | B | 45k/100k draw (re-aimed as in D4), exclusions vs TEST/held-out/cohort v3 | enables D4 | ~$0–5 | B: stack/rwkv7/b1-draw |
| D5 | A | D5b LR 2x: running job-i3buq. D5a r256 (90be840): packaged, on hold | ≤ +1–2 | ≤$40 each | D5b running |
| R2 | B | Clef 27B full-DEV comparator + analysis/paired_vs_clef.py | needed for the final claim | $7.66 | DONE: job-zfexa 61.89 [60.28,63.28] full DEV 32k (ref GDN); vs Clef-flash +4.54 [3.20,5.90]; R2b Clef-flash ref-GDN job-t5yg2 running |
| H1 | B | Hijack measurement for the champion. G12 builds no hijack variants, so hijack is unmeasured; Done needs it | required for the final claim | ~$2–5 | B: proposed |
| Q1 | — | Parked: Q9 proxy loop; Q9-AF option (b) | — | — | parked (#140 stall trigger) |

## Queue order (A, 2026-10-08 ~22:40Z): highest expected Index gain per GPU-hour first; build, train and eval all stacks in parallel
Rule: rank by (expected Index gain × confidence) ÷ (GPU hours to a verdict). New cells use the screen stage (proxy600, ~41-min eval). Full gates only on promoted cells. Never idle a slot; no item waits on operator input.
| # | item | expected gain | hours to verdict | state |
|---|---|---|---|---|
| 1 | D14 RLCD stage from D1 (screen) | +1–4 | ~3.5 | repackaging on screen, submits on balance |
| 2 | D13 attention-free joint record head (screen) | +2–5 (case-exact families) | ~5 | repackaging on screen |
| 3 | Big-data cell on B-1 synthetic 300k (+ C multirecord), v3w + fla + joint head if D13 wins | +5–10 | ~6 | waiting on B-1 / C data |
| 4 | Combine winners of D10/D10w/D11/D12f/D2/D13/D14 into one champion cell, then full DEV | compounding | ~6 + 7 | after landings |
| 5 | D2 KD (running b7xax), D1-fla re-measure (gi58k), D5a r256 (igcu4), D10/D10L/D10w/D11/D12f (running) | +1–6 each | landing 23:00–04:00Z | running |

## Proposals (B/C append below)

- **A (D4 training side, 2026-10-08): what `d4_draw45k` needs from B1** — interface only; B1's content/weights spec on `stack/rwkv7/b1-draw` is authoritative. Mode is on `stack/rwkv7/d4-draw45k` @ba38550 (cut from c7pq2's 11fb140). To make it launchable, B1 commits to that branch (or tells A the names): (1) `data/manifests/b1-d4-draw45k.pointer.json` in the 150k pointer format (`records_uncompressed_sha256`, `manifest`, `manifest_uncompressed_sha256`, `content_map`, `content_map_uncompressed_sha256`, `n_records` = 45000, `build`, `receipt`); (2) the three raw-gz sha256 lines in `jobs/m2_first_cell/data_d4_draw45k.sha256` (they are placeholders now and fail closed); (3) files at `ssdm-data/m2/build/b1-d4-draw45k/` (`records.jsonl.gz`, `b1-d4-draw45k.manifest.jsonl.gz`, `b1-d4-draw45k.content.jsonl.gz`). Records must be plain `ssdm-train-v0`: intent views are ordinary records with ≤16 options, with no sidecar the D1 loader can't read. The manifest needs `source` + `family` per row, and the content hash is `train.cohort.record_content_hash`. The set must contain 0 cohort-v2/v3 records (by id, content, source item and key-free text) and 0 `train.heldout` hits, so drop `m2:mmlu_aux:21118/39640/39643`. It must contain no TEST or held-out Index rows. The trainer draws the whole set in sha256(seed:id) order and does not reweight, so B1's counts are the mix.

- **C (Stage 2 implementation merged, 2026-10-08):** #215 @21ae4d4 supplies `python -m language_growth.gates --parent CHECKPOINT`, strict qualified M2 loading, same-task open-weight comparator, frozen Index retention, and native-head semantic/sham/random/output-only interventions. 104 CPU tests passed; receipt `receipts/2026-10-08-L-stage2-ready/`. No GPU order before Stage 1 passes.
- **C → A/B (champion inputs):** preserve the keep's M2 report/serving bundle, complete saved TrainConfig/lineage, original snapshot/tokenizer/runtime hashes, and complete matched DEV scoring/prediction exports. Eventual full Stage 1 Clef/safety qualification is required; keep alone does not qualify. Later measurement also needs comparator responses and discovery-fitted semantic/output directions. Current L4 historical table is one source-world and cannot pass the uncertainty floor; all learned gates remain unqualified.
- **A (D4 update):** `stack/rwkv7/d4-draw45k` @86508de has three clock modes on the same B1 set. `d4_draw45k` uses D1's 190 min, so only the mix changes (worst $67.02, expected ~$26.58). `_t380` trains ~22k rows (worst $85.83, expected ~$36.04). `_t780` is one pass over the 45k (worst $126.42, expected ~$56.33). The package step refuses only on the missing B1 files and pins.
| G1 | B | (operator goal 2026-10-08) Own final-claim evidence: champion on full DEV, paired CI vs both Clefs + ECE/steer/hijack/latency/cost + B rescore. Stretch: 2x Clef on cost/latency and/or capability | the claim | per champion ~$25 + $4 hijack | B: fulldev smoke PASS (job-46yeb); H1 hijack on D1 0.0%% vs parent 34.9%% (job-j8wgy); champion run ready |
| L1 | B | Cost/latency (PARKED 2026-10-08 per operator: capability first). Built: memo/packed/tree exact fixes; fla inference NOT_EQUIVALENT; no fair 2x | — | — | PARKED |
| B2 | B | 100k v3: official train splits ACOS/NLI4CT/VAST/RAGTruth/HoVer/Habermas/MNLI at natural base rates (operator-approved download) | regressed families +5-15 skill | $0 GPU | DONE: 100k v3 @1a023e8 (b1_check 0 fail); ACOS/NLI4CT/VAST have no stated license (private-research only, tagged) |

- **C (native learning prerequisites, 2026-10-08):** #223 @3358ab4,106CPUtests passing. Train-only1124slots/322tasks ready at ssdm-data/agent-c-gate-training-20261008-w32. Native fitter/explicit target-slot v2 controls and checkpoint header CLI are onagent-main. Next C work: independently grounded L4 contexts; existing IPCC table is one source-world and cannot qualify by replication. No GPU order beforeStage1.
- **C (L4 source audit, 2026-10-08):** public IPCC/JRC/NETL/NREL/EDF documents yield no verified set of 20 independent, same-boundary source groups. Exact source scopes and correlation traps: plans/codex/L4_INDEPENDENT_SOURCE_AUDIT_20261008.md. Do not multiply rows, regions, years or scenarios into worlds. Next source candidate must support within-boundary, independent study groups or L4's current scope stays explicitly unqualified.
| B-1 | B | Large synthetic schema-permutation draw (Clef recipe: multi-question/case-exact records, field-order/prompt/schema permutations) 300k-1M rows from allowed train material; never frozen catalogs | data scale 10x+ (capability) | $0 GPU | B: in progress (stack/rwkv7/b-schema-synth) |
| B-2 | B | ~600-row stratified proxy of the 2,008 subset (frozen scorer), validated leave-one-system-out, for 45-min cell screening | iteration speed 4x | $0 GPU | DONE: proxy600 @d2944c0 (Spearman 0.94; 91%% sign agree on 1-5pt pairs) + screen stage @2b9fd92 (evals ~41 vs 178 min; DROP if child-champion < -2.5) |
