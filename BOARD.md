# BOARD (rewritten each loop; top ~15; history lives in archive/, PREREG.md, PROGRESS.md, DEAD_ENDS.md)

**Objective (operator 2026-10-09):** beat the JevBench v1.6.1 open-weights leader on the official 4-axis **Composite** first, then **Capability**, then **MD1's reported numbers**, AND **beat Clef-flash + Clef 27B** on the Decision Index (PREREG.md rules). Hybrid state-space main line + strict attention-free research line. Structure must CONTROL learning with stated guarantees.

| Target | Current best | Our best so far |
|---|---|---|
| JevBench Composite (H2O-Lightning-4B v1.1) | 72.5 board (API: Sage 1.3.0 74.0). LOCAL public-231 scale differs (H2O I 77.6 local vs 60.0 board): target = paired Δ Composite vs H2O on public-231, CI>0 (EV-PREREG-COMPOSITE-LOCAL) | Local scorecard (claude/ev-scorecard @e2fcd82b, 28 arms, paired vs H2O B1000 seed16, open public items only): NO arm CI>0. Best ours: stock Qwen3.5-4B+H2O shim ΔComp A −9.4 [−23.1,−4.2] (67.3 vs 76.6); best trained cell D19 −14.4 [−31.7,−10.6] (62.3). Best external: torchcast-12b −3.6 [−6.6,−0.8]. Kit parity DONE: H2O 205, decider-12b v2 197 (pub 198), decider-12b-v1 200 (pub 200) |
| JevBench Capability (Quyet-1.0-Large) | 81.7 | Quyet not downloadable -> paired vs best downloadable Capability row (H2O local 80.5) per EV-PREREG-COMPOSITE-LOCAL: best ours D19 ΔCap −17.2 [−26.6,−10.7] (63.3); stock 4B+shim −17.7 [−26.5,−8.2]; external torchcast −1.0 [−7.7,5.2] (ns). claude/ev-scorecard @e2fcd82b |
| MD1 reported | 83.5% avg / cal 92.2 / p50 85 ms (Foundry) / 1.3% flips | not measured yet |
| Decision Index proxy600: Clef-flash / Clef 27B | 59.03 / 64.56 | D19 48.61 [44.10,53.08] (RWKV7 attention-free + joint head + v3b; +10.9 vs D1+v3w); D13 41.07 |
| Decision Index FULL DEV: Clef-flash / Clef 27B | 57.29 / 61.89 | D19 49.49 [47.77,50.80] (paired −7.80 / −12.40; ECE 0.041; p50 569 ms); D13 43.21 (−14.08 / −18.68). Held-out attacks: D13 MORE ROBUST than Clef-flash (hijack 0.025 vs 0.115; steer 0.003 vs 0.089, Holm-significant) |
| Guaranteed actionability on full DEV (LTT acted-on error ≤ 5%, δ 0.1) | CLUSTER-LEVEL LTT (B-PREREG-GUAR-CLUSTER, one decision per row; the valid test for multi-decision rows): Clef-flash 51.6% marginal / 39.9% per-area; Clef 27B 59.3% / 7.9% | D19 0.0% / 0.0% and D13 0.0% / 0.0%: certify NOTHING once correlated decisions are counted as correlated (D19's confident decisions come from few rows; it never acts above λ≈0.96; 11.5% error on single-decision rows). All earlier 'D19 more actionable than Clef-flash' verdicts WITHDRAWN. Receipts claude/ev-guarantee-proofs @b4835e05. Lever: row-level confidence (calibrated high-λ decisions on hard single-decision rows) |

| # | Item | Lane | Owner | Status |
|---|---|---|---|---|
| 1 | Merge harness PRs: #241 pricing, #242 chained cells, #243 jevkit, #244 harness-v3 (goal/AGENTS/SessionStart hook) | both | B reviews | open |
| 2 | Kit parity: reproduce ≥3 published open-weights rows (H2O-Lightning-4B anchor) with jevkit before any of our numbers count | evidence | B → new evidence agent | DONE 2026-10-10 (EV-PREREG-KIT-PARITY): H2O 205/205 (job-zbtbj), decider-12b 197/198, decider-12b-v1 200/200 PASS (job-r3uc4, receipts claude/ev-kit-parity@5d674b2d); torchcast-12b 198/203 FAIL = raw-vs-masked logprob gap (5 items). Basis: author-published public-231 counts; board P300+sealed cells not downloadable |
| 3 | Host scan through jevkit: H2O-4B anchor + newest small backbones (Qwen3.8/3.6/3.5 small, gemma-4 small, LFM, Granite-H, Nemotron Nano, SmolLM3) + RWKV7 small/13.3B | lead | A | packaging (claude/jev-host-scan) |
| 4 | Backbone shortlist (research done 2026-10-09): 1 Qwen3.5-4B main (75% GDN; H2O 72.5); 2 LiquidAI LFM2.5-2.6B / d1-3B (API d1 73.0; LFM1.0 licence); 3 gemma-4-E4B (cost class, 0% SSM); 4 Nemotron-3-Nano-4B (most SSM). AVOID Qwen3.5-9B / Qwen3.6-35B-A3B (Cost gated). No small official Qwen3.8 exists. Build-our-own from scratch: no (40-2,000x over budget). Next: A-PREREG-SIZECLIFF probe (~$45) | lead | A → new lead | probe packaging next |
| 5 | Port best recipe (v3w + LS-CE/Brier/RPS + joint head + anti-collapse) to the chosen host; chained training | lead | new lead agent | after 3+4 |
| 6 | Guarantees first: exact reorder/relabel invariance, conformal + abstention, injection-robust conformal (CONTROL_RESEARCH.md) | evidence+lead | both | kit metrics done (#243); model side next |
| 7 | Data: bschema v4 re-aimed at JevBench task families (TRAIN-only, all gates) | evidence | B | building |
| 8 | Q9-AF v3 recall-preserving conversion (research line) | lead | lead | S job-irdef LANDED: GO 8/8 layers (weak: NIAH at ceiling, MQAR Δ ≤ 0.005, single-layer only). A ($49–54) / B packaged on claude/b-q9af-v3 @aedc25e9, PARKED by the lead 2026-10-10: the 9B is Cost-gated for the Composite; credit goes to D20 l2/l3 + the 4B line first. Revisit if credit frees up. |
| 9 | D13 full DEV vs Clef (shards job-3z2m7 + job-hc6wp) → land_fulldev, B rescore | evidence | B | running |
| 10 | D18 job-i8mcp / D19 job-fi5xf screens (joint head + v2e/v3b data) | lead | A | running |
| 11 | D13 safety battery job-4yedh; heldout_v1 Clef arms | evidence | B | running / partly done |
| 12 | HANDOFF.md finalised → operator starts the two new agents | both | A+B | DONE (all gates merged) |
| 13 | D20 chained 3-link run (D19 recipe on v4 minus phish, 1,694 training min): link 1 job-7d3u9 running → package links 2 and 3 from JOB_SPEC "D20" | lead | new lead | running |

Proposals: append below this line (one line each); only the lead reorders.
