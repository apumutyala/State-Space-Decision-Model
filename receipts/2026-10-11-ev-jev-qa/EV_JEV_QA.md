# EV_JEV_QA: five JevBench questions, with receipts (evidence lane, 2026-10-11)

Base: origin/agent-main `7c131f19` (PR #243 jevkit merged). Nothing frozen was edited. Public items were read at the
format level only (types, families, K, field names, lengths). No item text, option text or gold distribution is
reported here. Every fetch was a plain GET of public pages. The raw bytes are in `raw_hf/` and `raw_github/`, and
each folder has a `SHA256SUMS`. The scripts are in `scripts/` and their outputs in `outputs/`. Downloaded files were
read with `python -I` from a separate directory.

## 1. Licence: h2oai/h2o-lightning-4b and Qwen/Qwen3.5-4B (verified; the lead's claim holds)

| | h2oai/h2o-lightning-4b | Qwen/Qwen3.5-4B |
|---|---|---|
| HF API `sha` (revision) | **`0940cbd7f3d2c9ee618e5dfaa3c74bfb765b12de`** (lastModified 2026-10-10T23:26:05Z) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (2026-03-02) |
| API tags | `license:apache-2.0`, `base_model:finetune:Qwen/Qwen3.5-4B` | `license:apache-2.0`, `base_model:finetune:Qwen/Qwen3.5-4B-Base` |
| `cardData.license` | `apache-2.0` (`base_model: Qwen/Qwen3.5-4B`) | `apache-2.0`, `license_link: …/Qwen3.5-4B/blob/main/LICENSE` |
| gated | false | false |
| LICENSE file | 11,544 B, sha256 `bbedc3fd…f1e57a` | 11,544 B, sha256 `bbedc3fd…f1e57a` (**byte-identical**) |

- **The LICENSE file is stock Apache License 2.0.** Its first lines are "Apache License / Version 2.0, January 2004".
  The appendix boilerplate reads "Copyright 2026 Alibaba Cloud". H2O ships Qwen's LICENSE file unchanged.
- **The model card says the same.** The front matter has `license: apache-2.0`. The closing `## License` section of
  README.md (sha256 `087a1447…7727`) says:
  > Apache-2.0. Built on Qwen/Qwen3.5-4B by the Qwen team (Alibaba Cloud), Apache-2.0; served with unmodified vLLM (Apache-2.0).
- **No extra clauses.** There is no acceptable-use policy, no usage restriction, no research-only or non-commercial
  term, and no gating. The card's other "licence" mentions concern its own training data ("licence-clean data").
- **The card's Disclosures section says** no JevBench or ImageJevBench data was used for training, and that the
  evaluator price basis is Qwen3.5-4B at bf16 with one output token per decision.
- **Derivatives are permitted.** Apache-2.0 §4 allows derivative works and redistribution, provided we keep the
  LICENSE and NOTICE text, mark changed files, and keep attribution. The patent grant (§3) and the trademark exclusion
  (§6) apply.
- **Pin the revision.** The repo was edited on 2026-10-10. Pin `0940cbd7…` for any parity or derivative work. The
  repo also ships `adapter/adapter_model.safetensors`, `h2o_lightning_shim.py` and `serve_config.json` beside the
  merged `model.safetensors` (4,539,265,536 BF16 parameters).

## 2. Long items: 23 items of about 77–82k tokens exist, but our local JSON has no per-item data

**The file.** The local results JSON is `ssdm-wt/b-landscape/raw/agg/v1.6.1.json`: 1,351,153 B, sha256
`5cd8c133…c83c1`, the same bytes as the board receipt's `api_v161.json`. Copies sit in two scratchpads. A walk of
all 655 key paths (`outputs/walk_agg.txt`) finds:

- **No per-item data.** There are no item ids, no per-item token or length fields, and no numbers in the
  60k–100k range. Per-system data is aggregate only: per split × type × tier CCs, calibration parts, speed, cost and
  validity rates.
- **The 23 items appear only in prose.** `$.v16.long_items_note` says 23 items of this draw are very long, about
  77,000 to 81,000 input tokens. It says systems with shorter context windows refuse them, Jev-Omni runs out of
  memory, decider-4b v2 truncates them to about 32.8k tokens, and Plumb-4B reads them in full. Each outcome is scored
  as the system's own result.
- **The 2026-10-05 v1.6.1 cost-rule amendment** measures cost on all items except these 23. Hence
  `cost.common_basis.n_items = 1477` (1500 − 23). Refusals still count against Intelligence.

**Which split.** "This draw" is the v1.6 S ∪ P pool: 1,500 items, P = 300 public, S = 1,200 sealed. The JSON does not
say how the 23 split between P and S, or which families and types they are. One saved board page narrows it
(`b-landscape/raw/page/jev-models.html`, sha256 `470d2711…dffb`; extract in `outputs/nine.txt`):

- **Exact token range.** Per-row notes say that on this pool exactly 23 of 1,500 items are 77,857–82,318 tokens,
  and that the 24th largest is below 2,048 tokens.
- **At most 9 can be public.** The v1.7.7 API note says nine items in A4 ∪ P are about 77–82k tokens. A4 is a
  separate set of 300 fresh sealed items. So at most 9 of the 23 are in the public P300, and at least 14 are in the
  sealed S1200.
- **Not stated anywhere:** the family or type of any long item. Long-document families such as long_policy or
  multi_hop are plausible guesses, but nothing confirms them.

**Our pinned 231 public items have none.** No state exceeds 40,000 characters. State JSON length is 18 / 130 / 15,161
characters (min / median / max).

**Consequence.** Any system with a context window under about 83k tokens loses up to 23 of 1,500 items: at least 14
sealed and at most 9 public. We cannot see them or test on them. Nothing sealed was downloaded, and none is public.

## 3. Why 231 and not 300: the 69 "missing" items do not exist in the repo; P300 is a different draw

**Our 231 items are v1.2's whole public release.** The upstream `datasets/manifest.json` (`protocol: jevbench::v1.2`,
frozen 2026-09-19) lists 534 decisions:

| split | n | in the repo |
|---|---|---|
| original | 72 | yes, `datasets/public/original.jsonl` |
| easy | 48 | yes, `datasets/public/easy.jsonl` |
| hard | 111 | yes, `datasets/public/hard.jsonl` ("public half; 109 more held out") |
| heldout | 24 | no, held out |
| easy-heldout | 24 | no, held out |
| hard held out | 109 | no, held out |
| router | 78 | no; the judge tier, "imported decisions from v1.0" |
| judge | 68 | no; the judge tier, "imported decisions from v1.0" |

72 + 48 + 111 = **231**. That is every item file the repo has ever published.

**v1.6.1's open set is a new draw, not a superset of the 231.**

- The board JSON's `v16.item_sets` lists P "Public set" = 300 items (choice 150 / noul 75 / score 75), S = 1,200
  sealed (600 / 300 / 300), and A = 300 (a subset of S).
- P's set-manifest sha256 is `6b2321f8…16ce5ba`.
- Its type mix differs from the 231 (139 choice / 74 noul / **18 score**).
- 300 − 231 = 69 is a coincidence of arithmetic. P300 is not "the 231 plus 69 more", and nothing in the repo maps one
  onto the other.

**No newer commit adds items.**

- The commit list shows HEAD = `7625ecb1` (2026-10-10), 7 commits after the pin `b6b8fff7` (2026-10-09). They add CI,
  tests, a validate-data CLI, the adapter guide and packaging.
- The HEAD tree, with `truncated: false`, lists exactly the same `datasets/` blobs at the same sizes as the pin:
  original 57,237 B, easy 37,220 B, hard 651,848 B, manifest 2,891 B.
- `manifest.json` at HEAD has sha256 `0983ad3e…`. Its split shas equal our pins.
- No v1.6 or P300 file exists anywhere in the HEAD tree (`scripts/tree2.txt`). The README points to
  benchmarkheaven.com for v1.6.1 and says "sealed inputs prevent complete independent reruns".

**So** cell-level parity (`open|*` CCs) needs the P300 file from Benchmark Heaven, which is not published in the
repo. jevkit/README.md already records this.

## 4. What a "score" question is, and how jevkit scores it

**There are three question types, not four.** `jevkit/spec.py` defines `TYPES = ("choice", "noul", "score")`, each
weighted 1/3. **"extraction" is a family, not a type**: 24 public items, all choice. There is no "noul / extraction"
type.

**What a score question is.** An ordinal rating on a K-level rubric. It is not a free 0–100 number.

- **Labels** must be `"0".."K-1"` (`items.py` validation).
- **Criteria** is a list of K level descriptions.
- **`expected`** is an **int** level index.
- **The answer** is a distribution over the levels: `probabilities` keyed `"0".."K-1"`. A label-only answer uses a
  numeric `score` in [0, K-1].
- **The prediction** is the **expected level** E[level] = Σ k·p_k, not the argmax (`answers.py`).

**How it is scored (`scoring.py`).**

| quantity | formula |
|---|---|
| per-item nMAE | \|E[level] − gold\| / (K−1); an invalid answer gets nMAE = 1 |
| per-item chance nMAE | mean over l of \|l − gold\| / (K−1), i.e. a uniform guess |
| Intelligence CC | **100 · (1 − mean nMAE / mean chance-nMAE)** per tier; tier-weighted, then clipped at 0 per split × type |

- **It is chance-corrected**, but by distance, not by hit rate. Argmax accuracy is computed as a diagnostic only.
- **Calibration part** = 50·(1 − 2·topECE) + 50·(1 − nRPS), each half clipped at 0. nRPS is the ranked probability
  score / (K−1).

**Is the ordinal family the score type?** Partly. All 12 `ordinal` items are score, but score appears in 6 more
families.

| tier × type | choice | noul | score |
|---|---|---|---|
| easy (48) | 36 | 12 | 0 |
| standard (72) | 36 | 24 | 12 |
| hard (111) | 67 | 38 | 6 |
| **total (231)** | **139** | **74** | **18** |

Score items by family: ordinal 12 (standard, K = 4); long_policy 2; adversarial 1; multi_hop 1 (K = 5); temporal_numeric
1; trap 1 (all hard, K = 4 unless noted). Score K overall: 17 items with K = 4 and 1 with K = 5.

Full family × type table (`scripts/items.py`):

| family | choice | noul | score |
|---|---|---|---|
| adequacy | | 12 | |
| adversarial | 3 | 2 | 1 |
| ambiguous | 7 | | |
| extraction | 24 | | |
| fact | | 12 | |
| intent | 24 | | |
| judge_hard | | 17 | |
| long_policy | 12 | 5 | 2 |
| multi_hop | 16 | 1 | 1 |
| ordinal | | | 12 |
| policy | | 12 | |
| probability | 7 | 3 | |
| routing | 12 | | |
| routing_hard | 5 | | |
| temporal_numeric | 10 | 4 | 1 |
| tool_selection | 12 | | |
| tradeoff | 4 | 2 | |
| trap | 3 | 4 | 1 |

**Why score matters.** Score is only 18/231 = 7.8 % of the public items, but it is 1/3 of Intelligence. The v1.6.1
board draw has 75/300 open and 300/1,200 sealed score items (25 %).

## 5. Which jev1 / jev2 families emit score-type rows

**bschema-draw-jev1 (`claude/b-jev1-draw` @ `60a6dd7b`; 96,000 rows).** 18,331 score rows (19.1 %, type_mix 0.1909),
from 5 of its 8 families:

| family | score rows | what the score question is |
|---|---|---|
| severity | 12,000 (all of it) | incident-rubric level (synthetic) + SST-5 sentiment intensity |
| adequacy | 4,070 | HelpSteer2 attribute grade (`hs2_score`) + agent-audit violation count |
| long_policy | 990 | reimbursement-share level |
| enum | 792 | `notice_priority_level` |
| state_tracking | 479 | ticket priority after events |
| routing / policy / intent | 0 | |

**bschema-draw-jev2a (`claude/ev-jev2-draw` tip `77065b1c`, stage 1; 30,000 rows).** 1,100 score rows (3.67 %):

| family | score rows | what the score question is |
|---|---|---|
| temporal_numeric | 698 | `temporal_days_late` lateness bucket |
| multi_hop | 402 | `ref_chase_count`, "how many effective transfers" |
| judge_hard / fact / tool_selection | 0 | |

**Not built yet.** jev2 stage b (probability, trap, ambiguous, tradeoff, adversarial) has not been built. Its
FAMILY_SPEC plans score rows for trap and adversarial.

**Coverage against the public score families.** We emit score rows for long_policy, multi_hop and temporal_numeric,
plus severity as an ordinal analogue of `ordinal`. There are none yet for adversarial or trap (stage b), and both are
hard-tier score families in the public set.

Receipts: `outputs/q5out.txt`, from each branch's `build-report.json` (`types` per family), plus `data/jev1.py`
`score_q` call sites.
