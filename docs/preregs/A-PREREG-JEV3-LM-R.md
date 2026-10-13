# A-PREREG-JEV3-LM-R (DRAFT on stack/qwen35/jev3-lm-r; NOT on state, NOT submitted)

- **A-PREREG-JEV3-LM-R (lead, 2026-10-12, DRAFT; to be registered on state BEFORE any JEV3-LM-R run; package
  stack/qwen35/jev3-lm-r, modes jev3_lm_r_s0 / jev3_lm_r_s1).** The JEV3-LM claim family (A-PREREG-JEV3-LM + Amendments 1-2),
  re-run at the full 1-epoch budget, with coverage rows added and a different primary readout, on 2 seeds. Worst case
  **$23.06 per seed** = (60 + 2 x 153) x 0.063 (hard cap $23.10); expected $6.36 + the build (<= $3.78) per seed; both seeds:
  worst $46.12. **Label: "design and readout chosen after seeing JEV3-LM-S (job-zbmhc) public results; optimistically
  biased; confirmation via the slice readout comparison by the evidence agent and the sealed jev-heldout secondary".**
  - MODEL: unchanged from JEV3-LM-S: STOCK Qwen3.5-4B @851bf6e8 + LoRA r16 (lr_lora 4e-5, lr_head 3e-4) + the lm_label head
    (train == score: the shim's label token at the Answer: slot, tied LM head); LS 0.05 (0 on yes/no); max_len 4,096 with 0
    truncation; the canonicalised shim (inference g1v2-canon, training rows g1v2-canon-shuffle; decide/canon_shim.py); the
    head-aware lm_label collapse guard; best checkpoint by dev-card NLL on the 360-row dev-card sample of the jev3 slice,
    exported on every save.
  - DATA: bschema-draw-jev3-lmr (pointer data/manifests/bschema-draw-jev3-lmr.pointer.json; ids 52523672..., 7,760 rows;
    builder scripts/jev3_lm_r_draw.py, deterministic, reproduced byte for byte locally) = the JEV3-TYPED draw
    (bschema-draw-jev3-typed, stack/qwen35/jev3-typed @a21049a9, scripts/jev3_typed_draw.py @ff06d4c0, built by job-stbvu:
    jev3's 5,960 rows verbatim + 600 each of the gated jev1 routing / intent / severity rows, ranked by sha256(20261012:id),
    skipping all-empty-rubric score rows and the jev2-mix exclusion lists (c0c175f5..., 23 ids; 5cebf60b..., 104 ids;
    sha + count asserted), and DISJOINT from the jev3 slice, the jev3 train rows, the 231 public JevBench items and each other
    by id, (source, source_id), content hash and normalised state; draw-report dropped {}) with each ADDED CHOICE row's option
    count subsampled to the public JevBench K of its family (routing 6 = max over public routing / routing_hard; intent 5),
    keeping the gold option + the distractors ranked first by sha256("jev3-lm-r-k:<id>:<key>"); 814 rows changed (routing 331,
    intent 483); yes/no rows and severity score rows (K 4-5 levels; a scale is not subsampled) unchanged. Sampler: the JEV3-LM
    exclude file (be643f63..., 124 ids: aegis_reply 109, hs2_bar_coherence 14 of 27, the 1 over-length canonical row) and the
    re-gate list (01ba4719, empty) -> pool 7,636; 1 epoch = N = 954 steps (8 rows / step), warmup 50 + cosine to 0.1 x over
    exactly 954, fixed before step 1. WHY the coverage rows: the evidence rescore of zbmhc (claude/ev-zbmhc-rescore @e03b1d42)
    puts the residual -0.78 vs stock mainly in SCORE items (-13.5) and ROUTING (0/3), and jev3 has no routing / intent /
    ordinal family.
  - SEEDS: s0 and s1 (cell "seed" = the draw order and the init RNG; the per-row training shuffle is content-seeded and
    identical across seeds). Everything else identical.
  - PRIMARY: paired delta Composite A of **native_shimcal** (the vendored H2O shim's post-processing applied to the trained
    lm_label distribution: temperature 0.8 + the yes/no commit floor 0.801, the shim's unchanged constants; nothing is fitted)
    minus the NON-canonical in-job stock + shim control, public-231, jevkit B 1000 seed 16. CLAIM (per seed) iff the 95% CI > 0
    AND both KEEP gates pass AND the slice floor passes.
  - WHY this readout (chosen AFTER seeing results; optimistically biased): (a) JEV3-LM-S public secondary (biased: the same
    231 items the claim is scored on): shimcal +3.48 [-1.09, +12.56] vs the pre-registered native_tfit -0.78 [-16.36, +3.47];
    (b) NON-public support (claude/ev-guar-jev3 @36141c14, the evidence agent's slice-only readout diagnostic on zbmhc's
    1,200-row jev3 slice): shimcal vs tfit slice NLL -0.281 [-0.352, -0.213], yes/no band 0; Composite-like slice proxy
    +4.83 [-0.76, +11.82] (n.s.); shimcal vs raw +40.3. (b) is the justification; (a) is reported and labelled biased.
  - T_noul: the A-PREREG-JEV3-LM per-type rule ("T_noul minimises slice NLL subject to slice yes/no band rate <= stock's +
    5pp") is REMOVED. It was infeasible by construction (the stock + shim control's commit floor puts 0 / 741 slice yes/no
    decisions in the 0.2-0.8 band, so the bound was band <= 0.05, which no T in [0.25, 4] met on zbmhc), and with shimcal as
    the primary it is moot: shimcal fits no temperature. No per-type T is fitted in this cell.
  - KEEP 1: G1 bitwise (option_reverse / option_shuffle / key_relabel; 0 flips, 0 invalid, max|dp| 0, rows bit-identical) on
    the primary arm, as in A-PREREG-JEV3-LM; the non-canonical stock control on the same files = teeth. KEEP 2: the evidence
    score op (claude/ev-score-inv @c47c0a46, vendored ev_score_perturb.py 803eeb90 / ev_score_flip.py b59f2165), --max-dp 0,
    on the trained arm; stock control for teeth; the evidence re-run disagreeing = NOT passed.
  - REPLICATION: the cell is REPLICATED iff both seeds' primary point estimates are > 0 AND seed 0's 95% CI > 0 (with both
    seeds' KEEP gates and slice floors passing). Seed 0 alone with CI > 0 = "claimed, single seed, replication pending" until
    s1 lands; s1's point <= 0 = "not replicated" (reported with both numbers; the claim is withdrawn). s1's CI is reported,
    not required.
  - SECONDARIES: native_tfit (ONE scalar T fitted on the slice by NLL, then moved toward 1 if it would raise the slice yes/no
    band rate above its T = 1 value: screen.constrain_t_by_band; feasible by construction, T = 1 is always allowed); raw
    native; canon_stock (canonical layout on the untrained host) vs stock; 8-op flips vs 7.55%; H2O target gap (e00d5ac7);
    slice floor + per-family slice deltas; per-family public deltas (routing / intent / ordinal named); the stock slice
    yes/no band rate (tfit.stock_band).
  - SECONDARY CONFIRMATION (pre-registered): each seed's exported best checkpoint is scored ONCE on jev-heldout
    (EV-PREREG-JEV-HELDOUT: the evidence agent's sealed ~2,000-item confirmation set, being built; records sha
    PLACEHOLDER, ids sha PLACEHOLDER; filled in params jev_heldout BEFORE that scoring), readout native_shimcal vs the stock +
    shim control, the rule set by EV-PREREG-JEV-HELDOUT. It is not run by this job's runner and does not gate the primary.
  - GOLD-LETTER SHORTCUT / A-SHARE: as A-PREREG-JEV3-LM (gold-letter distributions; the model's per-letter prior on the slice;
    slice choice accuracy with / without ambiguous; predicted-A share on public choice vs gold-A); FLAG if predicted-A exceeds
    gold-A by > 5 pp on the slice or public. A flag is reported beside the claim; it is not a gate.
  - FALLBACK RULES (all of them, in prose; the code implements exactly these):
    1. Setup gates (exit 15, before any training spend): the screen projection must fit its cap (no proxy600: 22.0 <= 0.95
       x 25) and the cell cap must cover cell_budget.py's need (113.28 <= 114).
    2. Step-10 rate check: the planned final step is min(954, the clock-affordable steps at the measured s/step). If short of
       954 the run CONTINUES with the cosine horizon unchanged (954) and records train_budget_shortened; below min_steps 800
       it stops (exit 26) and there is no result. The 96-min clock covers 954 steps at the slowest measured rate.
    3. Collapse guard (lm_label; train/trainer.py): a 50-step window mean loss above 1.5 x the FIRST full window's mean,
       or a dev card's accuracy below 0.6 x the best card's, is a collapse event: reload the best checkpoint's trainable
       weights and optimiser moments, halve both LRs, clear the loss window (step counter and data cursor go on). After 3
       reloads the next event stops training; scoring then uses the best checkpoint. Events are reported.
    4. Checkpoint selection: the lowest dev-card NLL (tie -> higher accuracy) among the dev cards run (step 0, then every
       50 steps); it is exported whenever it changes, so a timeout keeps the selected weights.
    5. Slice floor: the selected checkpoint's slice top-1 must beat step 0 (the untrained host) on the jev3 slice; otherwise
       "step 0 = no change", the run is NEGATIVE and no primary claim is made.
    6. Yes/no answers: the primary's yes/no probabilities are those of the shim (T 0.8, then max(P, 1-P) below 0.801 moves
       to 0.801 on its own side); no other band rule applies to the primary.
    7. Exact duplicate options form a tie set (equal served probability); empty-description options render by id (0 public
       / train / slice choice items affected).
    8. Any data, pin, guard, truncation or KEEP-gate failure fails closed (non-zero exit, no claim). A crash is fixed and
       relaunched under the same prereg (crash -> fix -> relaunch); the fix is recorded with the receipt.
    9. If the primary CI straddles 0: "not claimed" + the point estimate; no follow-up tuning on public-231; any successor
       selects / fits only on the slice or on non-public data.
  - SELECTION: only the jev3 slice (dev-card NLL); slice_first timestamps precede any public prediction. LIMITS: the readout
    and the coverage rows were chosen after seeing zbmhc's public numbers (bias stated above); one training shuffle per row;
    the coverage rows are subsampled to the public K (a format choice informed by the public items' option counts, not their
    content: disjointness from the public items is asserted by id, source, content and state).
