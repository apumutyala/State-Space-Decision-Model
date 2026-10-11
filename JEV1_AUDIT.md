# JEV1 train-side audit: why JEV1-4B r2 got worse on the families jev1 targets

Evidence lane, 2026-10-11. Subject: `bschema-draw-jev1` (pointer `data/manifests/bschema-draw-jev1.pointer.json`,
records sha256 `45735109…`, branch `claude/b-jev1-draw` @60a6dd7) as trained by `stack/qwen35/jev1-4b` @5e159437
(job-xt3ax, art-mv9cr). Public JevBench items were read for **shape only** (type, K, state type, lengths, noul
criteria presence). No content was selected, copied or tuned on. Scripts and outputs: `receipts/2026-10-11-jev1-audit/`.

## The result being explained (job-xt3ax `result.decision.jevkit`, ledger row 2026-10-10T23:40Z)
- Composite A 63.43 vs stock+shim 67.50 (Δ −4.07, CI [−25.6, 3.2]); I 51.2 vs 57.0; C 64.7 vs 70.2; S 85.3 vs 89.6.
- Top-1 is almost unchanged: **172 vs 173 / 231**. Covered families 94 vs 100 / 115, of which **long_policy is 4 vs 10 / 19**.
  So the other six covered families are **90 vs 90 / 96**. At the argmax level, all of the covered-family loss is long_policy.
  The rest of the Composite loss is probability quality (C, the proper-score part of I) plus speed (S, a readout-path cost, not data).
- The `native_shimcal` arm (the shim's T 0.8 and yes/no floor applied to the native probabilities) gets I to 56.6 vs 57.0
  (Δ −0.4). Most of the I gap is calibration, not wrong answers.

## Findings ranked by likely impact

### 1. long_policy is out of the training distribution in length and in request format (HIGH: explains the −6 top-1)
| | jev1 long_policy (12,000 rows) | public long_policy (19 items) |
|---|---|---|
| shim-prompt tokens (Qwen3.5-4B) | min 658 / p50 934 / p90 1,088 / max 1,262 | min 2,350 / p50 2,657 / p90 3,655 / max 3,952 |
| rows > 2,048 tokens | 0 | **19 / 19** |
| state type | dict `{case, policy_document}` (12,000 / 12,000) | plain string (19 / 19), 8.2k–15.2k chars |
| case facts | typed JSON fields (booleans and ints, e.g. `final_sale`, `notice_days`, `items[]`) | inside the prose record |
| policy text as the model sees it | a JSON string value: newlines show up as literal `\n` escapes inside compact JSON | raw prose with real newlines |
| noul criteria | **empty on all 4,627 noul rows**, so the shim fills in "the statement holds / it does not" | explicit true/false descriptions on 5 / 5 |

- No jev1 row of any family is longer than 1,975 tokens (the build drops rows over 2,048 and the ladder caps T_max at 2,048).
  Every long_policy eval prompt therefore runs the LoRA (r64 on every GDN, attention and MLP projection) in a length regime
  it was never trained in, while the stock arm runs plain Qwen3.5-4B, which handles 4k tokens fine.
- **Disambiguating check (cheap, CPU, no GPU):** public multi_hop is also 17 / 18 over 2,048 tokens but is uncovered.
  In art-mv9cr's `native.preds.jsonl` vs `stock_shim.preds.jsonl`:
  - if multi_hop also fell, length is the cause;
  - if multi_hop held or rose, the long_policy template and its priors (finding 2) are the cause.
  I did not download the 506 MB artifact. This is a 1-minute job for whoever holds it (jev1-diag / the rescore).

### 2. long_policy is a three-template monoculture with strong "restrictive" label priors (HIGH)
- 100% synthetic, from 3 generators (`lp_expense` 4,005, `lp_returns` 4,008, `lp_leave` 3,987) and **11 instruction strings** in total.
  Every policy document is unique once digits are masked, but each is the same rule skeleton plus boilerplate filler sections.
- Per (sub, type) gold skew:
  - noul false 65.5% (expense), 63.6% (leave), 72.0% (returns); 66.2% false overall;
  - returns choice `not_returnable` 68.3%;
  - leave choice `L6` 41.0%;
  - expense choice `approve_partial` 52.2%;
  - expense score level 1: 44.2%, level 3: 8.4%.
- The leave choice instruction gives away the rule order (it lists the clauses to check, in order). Public items carry no such hint.
- The roughly 1,097 long_policy rows r2 saw (below) are enough to learn these priors, which are learned first, but nowhere near enough
  to learn rule application on unseen document structure. That is the classic recipe for "confidently negative on new policies".

### 3. Every jev1 state is a JSON object; every covered public item's state is a short plain string (MEDIUM: likely the C / I calibration loss)
- jev1: 0 / 96,000 string states. Public covered families: 115 / 115 string states, 0 dicts.
- The covered original/easy items are short (shim tokens p50: policy 129, adequacy 117, intent 133, extraction 118, ordinal 145, routing 157).
  The matching jev1 families are longer and structured:
  - policy p50 269, with a `content_policy` dict;
  - adequacy p50 354, up to 1,974 (HelpSteer2 responses, agent traces);
  - routing p50 509, with rule catalogs.
- Top-1 on these families did not move (90 / 90), but the probability mass did. Together with the shimcal result, this points to
  a sharpness and calibration shift under a train/eval request-shape gap, not lost knowledge.

### 4. Half of the noul rows train against the shim's default option text (MEDIUM)
- 14,540 / 30,046 noul rows (48.4%) have `criteria: {}`: all long_policy (4,627), intent (2,378), enum (1,378), routing (2,821)
  and state_tracking (3,336). The shim renders these as `A) true: the statement holds / B) false: it does not`.
- Every public noul item (adequacy, fact, policy, judge_hard, long_policy and the rest) carries explicit descriptions.
- Only policy and adequacy noul rows in jev1 (15,506) match the eval shape.

### 5. Option-count and type mismatches with the eval shape (LOW to MEDIUM)
- intent: jev1 choice K is 6–16 (no K ≤ 5; 2,378 noul yes/no rows on top). Public intent is K = 5 on 24 / 24.
- routing: jev1 K is 4–12 with long rule states. Public routing is K = 6 on 12 / 12, with 18–79-char states.
- extraction and ordinal have no family of their own in jev1. enum (K 4–10, `{document_type, text}` states) and severity
  (score K 4–5, `{report, rubric_rule}`) stand in for them. Public ordinal is score K = 4 with a plain-string state.
- adequacy: public is 12 / 12 noul with one fixed instruction. jev1 adequacy mixes noul (4,689), score (4,070) and choice (3,241),
  and its noul gold is 62% true (`hs2_bar` alone is 81% true).

### 6. Things that are fine (checked, not the cause)
- **Sampler:** shuffled. `Trainer._order` = `torch.randperm(n, Generator(seed·100003 + epoch))` over the draw order
  (`split_ids`: sha256(`0:`+id)). I reproduced it (`seen_rows.py`):
  - the 1,121 × 8 = 8,968 rows seen (9.3% of an epoch) have 1,058–1,163 rows per family;
  - the family mix is flat across thirds of the run, with no same-family run longer than 5;
  - 5,541 synthetic / 3,427 dataset rows;
  - long_policy 1,097 rows: 381 expense (choice 166, noul 123, score 92), 376 leave, 340 returns; noul gold 294 false / 166 true.
  - The best checkpoint (step 1000, 8,000 rows) has the same mix.
  - **Latent risk:** `records.jsonl.gz` itself is family-sorted (aegis policy 7,118, then intent 8,530, …). Any loader that streamed
    it in file order would have trained 0.093 epoch on policy and intent only.
- **Order contract:** holds end to end.
  - `_cell_records` uses `json.loads(object_pairs_hook=…)`, which keeps insertion order. `to_train_record` copies by reference.
    The shim's `render_state` calls `json.dumps` without `sort_keys`. `options_of` uses `list(criteria)`.
  - Option axis canonical, displayed label = request position: `tests/test_jev1_4b.py`, plus the 140-row train == serve receipt
    `receipts/2026-10-10-jev1-4b/train_serve.json` (0 mismatches).
  - Presented gold position chi2 p ≥ 0.13 in every family (`build-report.json`). My per-K position tables are flat.
  - noul is always `A) true / B) false`. That is the shim's own convention, identical at eval.
- **Truncation:** none. The shim path never truncates.
  - The pool filter dropped 6 rows over 2,048 tokens (3 adequacy, 3 policy).
  - Training kept all 96,000 (max 1,975 ≤ T_max 2,048).
  - My token counts reproduce the build report (long_policy p50 934 vs 935).
- **Duplicates:** 0 exact (state + question). Near-duplicates (digits masked, criteria sorted): 1,835 rows (1.9%): adequacy 754,
  enum 645, severity 310, routing 66, state_tracking 60.
- **Mix:** 59,444 synthetic (61.9%) / 36,556 dataset rows.
  - Synthetic share per family: long_policy 100%, severity 82%, enum 82%, state_tracking 77%, routing 70%, adequacy 45%,
    policy 41%, intent 0%.
  - Label sources in policy: aegis human 6,344, llm_jury 774.

## Fixes

### For jev2 (new draw)
1. **long_policy rebuilt for the eval regime:**
   - single plain-text records, with policy and case narrated in prose (no typed case dict), at 2,000–4,500 shim tokens;
   - ≥ 10 policy domains with overrides, exceptions and amendments;
   - ≥ 200 paraphrased instructions with no rule-order hints;
   - explicit noul criteria on every row;
   - per (sub, type) gold balance: noul within 45–55%, choice near uniform over outcomes.
2. **Train at eval length:**
   - raise `t_ladder` to include 4,096 (sized by the ladder probe; row_batch 1 already);
   - make the build's token cap 4,096 for long families;
   - add a gate: for each covered family, the train p90 token length must be ≥ the public p90 (schema statistic only).
3. **State-format mix in every family:** render 50–70% of states as plain text (a dict flattened into natural sentences) and keep
   some short records (20–160 chars) alongside the long ones, so the model sees both `record: {json}` and `record: <prose>`.
4. **Always populate noul criteria** (with paraphrased descriptions). Keep at most ~10% default-text rows for robustness.
5. **K coverage:** intent and routing choice rows at K 2–6 as well as larger catalogs. Add an extraction-style family (choice K 4–5 over
   short records) and an ordinal-style score family (K = 4, plain-text record).
6. **Calibration:** keep the `native_shimcal` arm. Also fit a temperature on a jev2 calibration slice (never on public items) and
   report both. Consider label smoothing 0.05–0.1 on the synthetic rule families, whose hard labels are noise-free and drive sharpness.
7. **Build gate (shape support, no content):** compare draw-side schema statistics against a committed public-items schema
   fingerprint (per family: state-type ratio, token quantiles, K range, noul-criteria presence). Fail when a covered family's
   eval shape lies outside train support. Finding 1 would have been caught before spend.

### For a jev1 rebuild (if jev1 is rerun as a control)
- Drop or replace `long_policy`. Even as-is, regenerating it with balanced labels, explicit noul criteria and no rule-order
  hint isolates finding 2 from finding 1.
- Fill the 14,540 empty noul criteria. Re-render about 50% of states as plain text. Keep the family-sorted file, but add a test
  that the trainer's first-N rows are family-balanced (today this holds only because `Trainer._order` shuffles).
- Pre-register the multi_hop length check (finding 1) on the r2 preds before any new spend.

## Receipts (`receipts/2026-10-11-jev1-audit/`)
- `jev1_stats.py` → `jev1_stats.out.txt`: per-family gold, K, position, instruction, state-key and duplicate statistics.
- `pub_schema.py`: public items, schema only.
- `tok_lens.py`: shim-prompt token lengths, public (lengths only) vs jev1. Run with `.venv-gpu`; it needs `third_party/` from the
  stack branch on `sys.path`.
- `lp_detail.py`: long_policy per (sub, type) gold, instructions, case keys, doc reuse, noul criteria presence.
- `seen_rows.py`: exact reproduction of the rows r2 trained on.
