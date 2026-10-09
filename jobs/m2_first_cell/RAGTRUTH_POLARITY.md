# ragtruth_polarity: is D13's RAGTruth readout polarity inverted? (DRAFT, NOT SUBMITTED)

## Pre-registered decision rule (fixed before any GPU result exists)

For each arm separately (D13 and D1), on the RAGTruth **TRAIN** items only:

- **Polarity is inverted** if the AUROC 95 % bootstrap **upper** bound is below 0.5. Then a statement rephrase of the
  RAGTruth question is proposed.
- **No change** if the CI includes 0.5, or lies above it.

The AUROC compares p(hallucinated) (the `noul` answer, p_true) against the gold label (true = the response has a
hallucination). The CI is a percentile bootstrap: 2,000 item resamples, seed 20261009, 95 %. Items are independent
because there is one response per source document. The rule is coded as `ragtruth_polarity.decide_polarity`. Its
constants (`NULL_AUROC`, `CI_LEVEL`, `N_BOOT`, `BOOT_SEED`) are pinned by `tests/test_m2_ragtruth_polarity.py`.
The report also gives, per arm, the AUROC of the flipped score (1 - p) with its own CI, plus F1 and accuracy at 0.5,
mean p, the fraction predicted hallucinated, and the AUROC per task type. These are descriptive. Only the rule above
decides.

## Why

D13 is job-hjwpq, artifact art-cr9a2, best checkpoint at step 1600, readout v3w. Its proxy600 RAGTruth AUROC is 0.18.
The D1 reference predicts nearly every response as positive. One of two things is happening: the model ranks with
inverted polarity, or the proxy holds too few RAGTruth rows to tell. This decision uses only the RAGTruth
TRAIN split (15,090 responses, prevalence 0.445). Index DEV is never used for the decision.

## Items (CPU, built locally; shipped sha-pinned)

`python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity build` makes the items from the release files
`ssdm-data/m2/sources/dl-2026-10-08/ragtruth/{response,source_info}.jsonl`. Both files are sha-pinned in
`ragtruth_polarity_selection.json`.

- **Format.** Each item is presented exactly as the Index presents RAGTruth (catalog 59):
  `state = {prompt, response}`, the Index's single `noul` question `q`, and the kit's metadata keys. Before any item
  is written, all 270 DEV RAGTruth rows are rebuilt from the same files and must be equal. The fields checked are id,
  benchmark, family, split, state, questions, expected, metadata and scoring. Scoring goes through
  `train.m2_gates.index_predict` with the v3w decider, the same path the Index arm uses.
- **Exclusions.** Two kinds of response are excluded:
  - (a) its source document is the source of a DEV RAGTruth row (0 responses);
  - (b) it fails `train.doc_overlap_guard` (rule doc_overlap_guard/v1, index af096960, allowlist 556296cc). This
    gate checks 13-gram document overlap with every Index eval row: DEV, TEST and held-out. It caught 24 responses,
    all attributed to held-out RAGTruth documents and none to DEV. Excluding them also keeps held-out rows out of the
    decision.
  The guard ran on all 15,090 TRAIN responses. Its report is saved as `ssdm-data/m2/ragtruth-polarity/guard-scan.json`.
- **Sample.** One response is kept per source document: the first in sha256(`ragtruth-polarity-v1:<id>`) order. From
  those, 300 are drawn with proportional allocation over task_type x label. Prevalence is 0.453. The ids sha256 is
  `794bcce30da94f7da3696b522a1a20d1e8719b51286970dc2a47ac7fc8ab9295` and the items file sha256 is
  `dbbca4ff03f141b8b0f5d17c17fe0cfbb2593df226f4f80793dff84366349fd4`.

## Job

The mode is `first_cell_job.sh ragtruth_polarity`, with params in `job_params_ragtruth_polarity.json`. It runs
eval-only, with no training. The phases are:

1. setup and guard;
2. `checkpoint`: D13 from `$CKPT_URL`;
3. `score_d13`;
4. `checkpoint_d1`: D1 from `$CKPT_URL_D1`;
5. `score_d1`;
6. `report`, which writes `ragtruth_polarity_result.json`.

Both checkpoints are installed by `checkpoint_loader install --block` and checked byte for byte against the pins.
Both arms are scored untempered under v3w through `fulldev_eval.load_child`'s readout override, which is how the
screen scored them. `sf_submit.py` fills the `<...>` URL placeholders from the shell environment.

**Cost.** Expected run time is about 73 min, or $3.61 at the batch rate (about $4.36 including a 15-min build).
Scoring takes about 6 min per arm. The rest is fixed cost per arm (model load plus the 150k lineage check: D13 25
min, D1 12 min) and setup. The worst case is (60 + 2 x 177) x 0.0495 = **$20.49**, against a $21 cap.
