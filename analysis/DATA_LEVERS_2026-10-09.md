# Data levers for D16 / D17 (2026-10-09): which collapsing or trailing families the next training draws miss

Agent B analysis. CPU only, $0. This is diagnosis that steers TRAINING DATA. No DEV row is used to fit or tune anything.
Labels: **M** = measured, **X** = judgement.

## 0. Takeaway first

1. **The biggest data lever is a format gap that neither draw closes.**
   - Every scored API-Bank question is a **K = 53** API catalog. The largest API-Bank K in D16 or D17 is **10**, and the
     largest tool catalog of any kind is K = 27 (glaive). (M)
   - MMLU-Pro is mostly **K = 10** (35 of 43 questions). All `mmlu_aux` training items are K <= 4. (M)
   - These two together are about **+0.97 pts** expected (X).
2. **Three collapses are label-prior imbalances that training data created.** (M)
   - **CLINC150 "out of scope"** (option_80) is 21% of gold. It is 0.8% of CLINC training items in both draws (49 / 95
     `oos_train` rows), and D10w never predicts it.
   - **ForecastBench "yes"** is 28% of gold. Across all K = 2 {yes, no} training items the yes rate is **14.1% in D16**
     (61k items, 41% of all D16 items) and **18.7% in D17**. The champion never answers yes.
   - **PhishNChips** has no training analog at all. D10w answers "not phishing" on 98.7% of rows.
3. **D16 is missing families that D17 already has.** RAGTruth (D10w collapse), Habermas, VAST and ACOS are absent from
   D16 (the 100k-v2 draw). D17 has train-split views of all four, with base rates close to the eval. **RAGTruth in D17:
   41.9% "hallucinated" vs 42% gold** (M).
4. **D17 dilutes some families that are present.** Both cells run on the same 380-min clock (~24-32k rows seen). D17 has
   2.5x the records, so it sees only about 12% of its draw, against about 30% for D16.
   - BANKING77's full 77-way catalog: about 150 records seen in D17 vs about 620 in D16.
   - CRUXEval goes the other way: about 4,300 seen in D17 vs about 700 in D16. (X: exposure = records x rows-seen /
     draw size.)
5. **Not data levers (present, base rate matched, still wrong):** WinoGrande (+0.54, capacity), When2Call, HellaSwag,
   Humicroedit. Humicroedit's headline_1 collapse happens with 50/50 position-balanced pairs in both draws.

### Top data levers (expected Index pts = 0.4 x Index weight x headroom; headroom = best known arm skill - D10w skill)

| Rank | Family | Class | D16 has | D17 has | Proposed allowed source | Exp. pts |
|---|---|---|---|---|---|---|
| 1 | API-Bank | FORMAT-MISMATCH (K 53 vs <= 10) | 604 rec, K 4-10 | 1,194 rec, K 2-10 | On disk: `b1fam_apibank/lv{1,2,3}-api-train.json`. Generate K 40-60 catalog views by adding distractor APIs from the train-split API inventory. Use compact one-line API descriptions so the rows fit T 2,048 | **0.61** |
| 2 | CRUXEval | PRESENT, under-volumed in D16 | 2,400 (1.6% of items) | 35,997 (9.8%) | Already in D17 (synthetic). D16: raise the synthetic CRUXEval share (generatable) | **0.58** |
| 3 | ForecastBench | ABSENT + IMBALANCED yes-prior | 0; yes/no prior 14.1% yes | 0; 18.7% yes | Generatable: rebalance K=2 {yes, no} views to ~30-50% yes (BoolQ train as a yes/no **choice** view: 62% yes; HoVer SUPPORTED->yes). **It leaves the 0.3 index**, so this only helps 0.2.1 | 0.52 |
| 4 | PhishNChips | ABSENT | 0 | 0 | Not on disk. Generatable synthetic emails with verdict, click / do_not_click and is_phishing labels at ~46% phishing. Or a public phishing-email train split behind the doc-overlap guard | **0.45** |
| 5 | BANKING77 (K 77) | PRESENT, diluted in D17 | 2,089 K77 items (1.4%) | 1,112 K77 (0.3%) | On disk `banking77/train.csv`: restore the full-catalog K 77 share in D17 to at least the D16 level | 0.44 |
| 6 | MMLU-Pro | FORMAT-MISMATCH (K 10 vs 4) | mmlu_aux 7,648, K 4 | 7,019, K 2-4 | On disk `mmlu_aux` (auxiliary_train). Generate K 10 views with same-subject distractors. GSM8K K 10 already exists (2,134 / 2,069) | **0.36** |
| 7 | ContractNLI | FORMAT-MISMATCH (doc length; NotMentioned 42%) | anli 3-way, short, balanced | anli 3-way + mnli 2-way, short | Generatable at <= 2,048 tok: multi-paragraph "document NLI" from anli / mnli train with a NotMentioned class (hypothesis taken from another document), ~45/40/15. Not on disk: ContractNLI official train split (needs the T 4,096 rung) | 0.24 |
| 8 | NLI4CT | ABSENT (clinical domain) | anli claim 2-way (33% SUPPORTED) | + mnli 2-way, HoVer | On disk `dl-2026-10-08/nli4ct/training_data.zip`, **only after** the NGRAM audit's filter (drop records whose CTR covers >= 50% of an Index premise). Otherwise none | 0.20 (0.43 vs champion) |
| 9 | Habermas | ABSENT in D16 | 0 | 2,027 (K 2-4) | On disk `dl-2026-10-08/habermas`; already built in D17 | 0.19 |
| 10 | CLINC150 OOS | IMBALANCED (0.8% vs 21%) + K 151 vs <= 16 / 79 | 49 oos of 6,420; max K 16 | 95 oos of 12,322; max K 79 | On disk `clinc150/data_full.json` `oos_train` (100 rows). Generatable OOS: K 151 full-catalog views whose utterance is an off-domain train question (NQ, glaive user turns, BoolQ questions; not banking77) at ~15-25% OOS | 0.16 |
| 11 | VAST | ABSENT in D16 | 0 | 3,670 | On disk `vast/vast_train.csv`; already in D17 | 0.15 |
| 12 | RAGTruth | ABSENT in D16 (D10w collapse) | 0 (boolq noul 50%) | 16,964; 41.9% true | On disk `ragtruth`; already in D17 | 0.09 (vs Clef 0) |
| 13 | Amazon ESCI | ABSENT (no 4-way E/S/C/I) | nq relevance 743 | nq 33 | Not on disk: Amazon Shopping Queries train split (Apache-2.0) behind the doc-overlap guard | 0.08 (0.17 vs champion) |
| 14 | FinEntity | FORMAT-MISMATCH (3-way choice vs sst5 score) | sst5 score 2,000 | sst5 score 3,327; VAST 3-way | On disk sst5: add 3-way neg / neutral / pos choice views | 0.06 |

**Sum of the data-lever rows, excluding PRESENT-BUT-STILL-WRONG and ForecastBench: ~3.6 pts (X).** Of that, ~1.0 is
already in D17 and missing from D16 (CRUXEval volume 0.58, Habermas 0.19, VAST 0.15, RAGTruth 0.09). About 2.6 is
missing from both draws: API-Bank K 53, PhishNChips, BANKING77 K 77 exposure, MMLU-Pro K 10, ContractNLI doc-NLI,
NLI4CT, CLINC OOS, ESCI and FinEntity.

## 1. Inputs and receipts (M)

| Arm | File | Index (subset, 2,008 rows) |
|---|---|---|
| Champion D1 + v3w | `rab_d1_repro\dl\art-zskmx.tar` -> `./phases/final.tgz` -> `out/readout_ab/child.v3w.results.jsonl` | **38.669** |
| D10w | `art\job-hgwsq\x\final\out\units-child-index.json` | **40.143** |
| Clef-flash | `review-artifacts\rwkv7-baseline-preserve-20261007\clef-flash-job-sgihj-results.jsonl` | **55.780** |

- `analysis/data_levers_eval.py` writes `analysis/data_levers_out/eval.json`. It reports per-family skill, gaps,
  label-collapse flags and eval shape (K, gold base rate). It only calls the frozen scorer.
  - It re-implements the rule in `jobs/common/label_collapse.py` (origin/claude/b-collapse-warn) read-only.
  - It reproduces the tool's flags: champion Humicroedit headline_1 (0% vs 36% gold) and ForecastBench yes (0% vs 28%);
    D10w RAGTruth hallucinated (0% vs 42%) and CLINC option_80 (0% vs 21%); D10w PhishNChips is_phishing near-collapse
    (1.3% vs 46%). POP909 chord_119 (both arms) is in a frozen catalog and is not considered.
- `analysis/data_levers_train.py` writes `analysis/data_levers_out/train_shape.json`. Records are read as JSON data with
  `python -I`. It reports, per (source, family): records, items, item share, K histogram, gold position rate, semantic
  label rates and noul true rate.
  - D16 = `b1-draw100k-v2` minus the 5,201 `aug.injection.family == "d1a"` rows (`train_set.attack_clean`, matches the
    spec), giving 94,799 records / 149,789 items. The 14-id doc_clean pin is not applied (<0.02%).
  - D17 = `bschema-draw230k-v2e`, 234,521 records / 366,741 items (matches the JOB_SPEC pointer).
- Selection: every non-frozen family with a collapse or near-collapse flag in either arm, or one that trails Clef-flash
  by >= 10 skill points in either arm. That gives 19 families. ACOS is selected but has no data lever: D10w 29.8 already
  beats Clef-flash 19.8, D17 carries 1,314 ACOS train records at 1.4% yes vs 0.6% gold, and D16 has only the low-rate
  member analog.
- Expected points: 0.4 x weight x (best of {champion, D10w, Clef-flash} skill - D10w skill).
  - 0.4 is the gap-breakdown transfer rule (`GAP_BREAKDOWN_B_2026-10-08.md`; median in-draw transfer 0.51).
  - The base is D10w because D16 is D10w's data plus safety changes, on the same v3w readout.
  - Columns against the champion are in `eval.json`.

## 2. Per-family classification (eval vs D16 / D17 training)

Eval skill: champion / D10w / Clef-flash. "Seen" is exposure under the 380-min clock: ~28k rows, so x0.30 for D16 and
x0.12 for D17 (X).

| Family (w) | Eval shape (gold base rate) | Skill C / D / Clef | D16 analog | D17 analog | Class |
|---|---|---|---|---|---|
| API-Bank (.041) | choice K 53 (all 38) | 51.7 / 59.8 / 97.3 | b1fam_apibank 604, K 4-10 (pos uniform); glaive K <= 27 | 1,194, K 2-10; glaive K <= 27 | **FORMAT-MISMATCH** (K) |
| CRUXEval (.024) | K 2-10, 21/43 K 2 | 12.9 / 24.2 / 84.8 | synth 2,400 (seen ~700) | synth 35,997 (seen ~4,300) | PRESENT; D16 under-volumed |
| WinoGrande (.029) | K 2, 65% pos 0 | 53.5 / 53.5 / 100 | 3,186, pos 48/52 | 4,299, 51/49 | PRESENT-BUT-STILL-WRONG |
| ForecastBench (.017) | K 2 yes/no, 28% yes | 11.2 / 0.0 / 78.6 | none; global yes/no prior 14.1% yes | none; 18.7% yes | **ABSENT + IMBALANCED** (champion collapse to "no") |
| BANKING77 (.036) | K 77 (all) | 51.0 / 55.4 / 85.3 | 5,765: K 16 3,676 + K 77 2,089 (seen ~620) | 9,201: K 77 only 1,112 (seen ~130) | PRESENT; diluted in D17 |
| PhishNChips (.030) | verdict K 2 (54% phishing), click K 2, is_phishing noul 46% | 34.2 / 13.2 / 50.0 | none | none | **ABSENT** (D10w near-collapse) |
| NLI4CT (.024) | K 2 Entailment / Contradiction, 64% Contradiction | 18.8 / 41.6 / 62.4 | anli claim 2-way 67% NOT_SUPPORTED (format OK, domain absent) | + mnli 2-way 17,548, HoVer 20,240 | **ABSENT** (clinical CTR domain; nli4ct train held out for premise overlap) |
| MMLU-Pro (.029) | K 10 (35/43) | 39.8 / 32.0 / 63.4 | mmlu_aux K 4 only | K 2-4 | **FORMAT-MISMATCH** (K) |
| ContractNLI (.024) | K 3: Entailment 46 / NotMentioned 42 / Contradiction 12; median 3.9k tok | 38.8 / 50.1 / 75.1 | anli 3-way balanced, short (0.4% of D10 items > 2,048 tok) | anli + mnli, max T 2,048 | **FORMAT-MISMATCH** (length, NotMentioned) |
| When2Call (.034) | K 4 | 47.7 / 41.7 / 58.3 | 3,555, positions uniform | 6,702 | PRESENT-BUT-STILL-WRONG |
| Habermas (.014) | K 4 (51/60) | 8.0 / 8.0 / 42.5 | none | 2,027, K 2-4 | **ABSENT in D16**; PRESENT in D17 |
| HellaSwag (.029) | K 4 | 81.4 / 90.7 / 96.9 | 3,581 | 4,267 | PRESENT-BUT-STILL-WRONG (small) |
| Amazon ESCI (.030) | K 4 E/S/I/C 64/25/8/3 | 15.0 / 22.4 / 28.7 | nq binary relevance 743 | nq 33 | **ABSENT** (4-way query-product relevance) |
| CLINC150 (.036) | K 151; OOS (option_80) 21% | 57.4 / 57.9 / 68.8 | 6,420 K 16; oos 0.8% | 12,322 K <= 79; oos 0.8% | **IMBALANCED** (OOS) + K mismatch (D10w collapse) |
| ACOS (.024) | K 2 yes/no, 0.6% yes | 5.6 / 29.8 / 19.8 | member analog (5-6% yes) | acos 1,314 rec / 12,800 items, 1.4% yes | ABSENT in D16; PRESENT in D17 (D10w already > Clef) |
| FinEntity (.024) | K 3 Neutral 49 / Pos 27 / Neg 24 | 86.1 / 94.0 / 100 | sst5 score (regression) | sst5 score; VAST 3-way stance | **FORMAT-MISMATCH** (score vs 3-way) |
| VAST (.024) | K 3 | 32.7 / 17.2 / 27.3 | none | 3,670 | **ABSENT in D16**; PRESENT in D17 |
| Humicroedit (.014) | K 2, headline_2 64% | 28.8 / 39.0 / 42.4 | s2 pair 1,368, pos 50/50 | 2,156, 50/50 | PRESENT-BUT-STILL-WRONG (champion collapse is not a training-prior effect; D10w does not collapse) |
| RAGTruth (.024) | noul, 42% true | 9.1 / 0.0 / 0.0 | none (boolq noul 50%) | 16,964; noul 41.9% true | **ABSENT in D16** (D10w collapse); PRESENT in D17 |

## 3. Which draw is missing what

- **D16 (100k v2, attack/doc-clean)** is missing:
  - RAGTruth: the collapse family.
  - Habermas, VAST and ACOS: no train-split views.
  - Enough CRUXEval volume: 2,400 synthetic rows.
  - Its yes/no items are 41% of all items at 14% yes. That is the most no-skewed binary prior of any draw, which fits
    the ForecastBench "no" collapse and the PhishNChips near-collapse.
- **D17 (bschema 230k v2e)** fixes RAGTruth, Habermas, VAST, ACOS and CRUXEval. It still lacks:
  - API-Bank K 53, MMLU-Pro K 10 and CLINC OOS / K 151.
  - Doc-length NLI (ContractNLI) and PhishNChips.
  - Clinical NLI and ESCI.
  - It also cuts BANKING77 full-catalog exposure by ~4.5x.
- **Both draws:** CLINC OOS is 0.8% vs 21% gold, and no item is longer than 2,048 tokens.

## 4. Caveats

- All per-family expected points are X: 0.4 transfer, additive. Families with 43-76 questions have skill SE of 7-15.
- Exposure under the clock is an estimate (24-32k rows seen, JOB_SPEC). The draw order is `sha256(seed:id)`, assumed
  uniform over families.
- Gold base rates are measured on the 2,008 subset only, to judge whether a TRAINING prior is far off. No eval row is
  proposed as training data, and no readout or threshold is fit to them.
- Proposed sources that are not on disk (PhishNChips corpora, ContractNLI train, ESCI train) must pass the doc-overlap
  and attack guards before use. NLI4CT train is allowed only with the audit's CTR-overlap filter.
