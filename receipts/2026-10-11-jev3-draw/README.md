# bschema-draw-jev3: human-labelled TRAIN-split rows for six JevBench judgement families

Evidence lane (data worker), 2026-10-11. Branch `claude/ev-jev3-draw`, cut from `claude/ev-jev2-draw` @282a4867.

- **Spec:** `FAMILY_SPEC_JEV3.md`, committed at f34e8413 before any build. Amendments A0–A5 were each written before the
  gate output that would use them, with the reason stated.
- **Code:**
  - generator: `data/jev3.py`;
  - builder: `scripts/jev3_build.py`, where `check` is jev1's `cmd_check` unchanged;
  - gates: `scripts/jev3_prov_gate.py` (split provenance + MATH train/test), `scripts/jev3_strict_fp.py` (strict full-row
    Index fp), `scripts/regate_exact_leaves.py` + `regate_jev2a.py` (copied from ev-jev2a-regate @91b66644),
    `scripts/jev3_hs3_dedupe.py` and `scripts/jev3_gates_summary.py`;
  - streams and fetch: `scripts/jev3_prm_stream.py`, `scripts/jev3_hs3_stream.py`, `scripts/jev3_fetch.py`.
- **Run:** `pipeline.sh` with PHASE=pool, then pre, final, post. Per-family pools come first from
  `jev3_build.py gen --families f`.
- **Why:** x7c42 and q668q gained on our held-out jev slice but not on the public items. The diagnosis is rule-engine
  conventions, saturated families and shape gaps (`RESCORE_X7C42.md` section 7).
  - jev3 trains the judgement families on HUMAN labels: judge_hard, adequacy, policy and ambiguous.
  - It also trains the families the slice cannot teach on real-text human answers: temporal_numeric and long_policy.

## Pointers
| pointer | role | rows | ids sha256 (sorted, "\n"-joined) |
|---|---|---|---|
| `data/manifests/bschema-draw-jev3.pointer.json` | TRAIN | 5,960 (1,000 × 5 + long_policy 960) | `feb681f3517c69863d7180dbbf3307dd7b6ded4b4a6411b2ae8f360af33f3120` |
| `data/manifests/bschema-draw-jev3-slice.pointer.json` | held-out selection slice (checkpoint selection only, never trained) | 1,200 (200 × 6) | `844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078` |

- **Files:** each draw dir `ssdm-data/m2/build/<name>/` holds records, manifest, content map and
  `<name>.lens.jsonl.gz`. The last one is the **per-row shim token length**, `jrt-h2o-shim:qwen3.5-4b`.
- **Slice vs train:** 0 shared ids, source items, groups (document for TimeQA / PrivacyQA, context for HelpSteer3),
  content hashes and **normalised state texts**.
- **State dedupe:** no two rows of train + slice share a normalised state. Rows dropped by family:
  - adequacy 11;
  - long_policy 157 (full-policy windows of the same query across views);
  - all other families 0.

## Per family (TRAIN)
| family | rows | sources (licence) | types | K | string states | shim tokens p50 / p90 / max | noul true |
|---|---|---|---|---|---|---|---|
| judge_hard | 1,000 | PRM800K phase2 train, MATH-TRAIN problems only (MIT) | noul | – | 0 % (dict `{request, response}`, as public) | 476 / 896 / 1,374 | 44.7 % |
| adequacy | 1,000 | HelpSteer2 train 430 + HelpSteer3 feedback train 570 (CC-BY-4.0) | noul | – | 100 % | 437 / 723 / 1,188 | 54.1 % |
| policy | 1,000 | Aegis 2.0 train, human labels only (CC-BY-4.0) | noul | – | 100 % | 175 / 301 / 527 | 49.5 % |
| ambiguous | 1,000 | AmbigNQ light train (CC-BY-SA-3.0) | choice | 3 (863), 4 (137) | 83.7 % | 604 / 846 / 932 | – |
| temporal_numeric | 1,000 | TAT-QA train 730 (CC-BY-4.0) + TimeQA human-annotated train 270 (BSD-3) | choice 591 / noul 324 / score 85 | 3–6 | 80.6 % | 721 / 1,767 / 2,532 | 50.6 % |
| long_policy | 960 | PrivacyQA train (MIT) | choice 503 / noul 346 / score 111 | 4–6 | 100 % | 3,008 / 3,704 / 4,091 (window 2,000–4,096) | 45.4 % |

Label rates, subfamilies, position χ² and the full token histograms are in `build-report.json`.

## Sources (revision / sha; raw files under ssdm-data/raw/jev3, fetch log `fetch-log.jsonl`)
| source | split used | licence | revision / sha256 |
|---|---|---|---|
| PRM800K `phase2_train.jsonl` | train. Problem unique_id `train/*` (MATH TRAIN), and the problem text is in the official MATH train file and NOT in MATH test | MIT | openai/prm800k @7ecc7947; streamed and never stored; stream sha256 = LFS oid `1110237f…1089`; `math_splits/train.jsonl` `90d96dae…` |
| MATH reference (for the gate only) | train + test | MIT | EleutherAI/hendrycks_math @21a56338 (14 parquet files, shas in `prov-gate.json`) |
| HelpSteer2 `train.jsonl.gz` | train; no prompt used by any jev1 pool row (helpsteer2 / hs2pref) | CC-BY-4.0 | the dl-2026-10-10 copy (as jev1) |
| HelpSteer3 `feedback/train.jsonl.gz` | train; English general / stem; consensus verdicts; no user turn equal to a HelpSteer2 prompt turn | CC-BY-4.0 | nvidia/HelpSteer3 @f6d14577; streamed and never stored; gz sha256 `6f51cf5b…519c` |
| Aegis 2.0 `train.json` | train; ids not used by jev1; human prompt / response labels only | CC-BY-4.0 | the dl-2026-10-10 copy (as jev1) |
| AmbigNQ light `train_light.json` | train (the 2,002 dev ids are checked absent) | CC-BY-SA-3.0 | ambignq_light.zip `3f5dada6…20db` |
| TAT-QA `tatqa_dataset_train.json` | train | CC-BY-4.0 (data) | NExTplusplus/TAT-QA @870accc4, `2df6e722…7c69` |
| TimeQA `human_annotated_train.json` | train | BSD-3-Clause | wenhuchen/Time-Sensitive-QA @38b05989, `99eec051…39b5` |
| PrivacyQA `policy_train_data.csv` | train (27 policies) | MIT | AbhilashaRavichander/PrivacyQA_EMNLP @5178ff3e, `81aa6744…004a` |

Every row carries `aug.jev1.prov`: dataset, revision, licence, split `train`, the source row key and the sha256 of the
canonical source record. `scripts/jev3_prov_gate.py` resolves each key against the TRAIN file and finds 0 failures.

**Not used:**
- ContractNLI: it overlaps the Index fingerprint, per the brief.
- OPP-115 and PolicyQA: non-commercial terms / partial context.
- FinQA: TAT-QA covers numeric QA.
- TempReason: synthetic answers.
- ASQA: it overlaps AmbigNQ.

## PRM800K / MATH provenance (coordinator 2026-10-11)
- The stream read 97,782 phase2 lines. It skipped 31,468 lines whose problem is a PRM800K `test/*` problem: the ~4.5k
  MATH-test problems PRM800K trains on, plus MATH-500. It also skipped 14,992 quality-control / screening lines, 2,035
  lines with finish_reason other than solution / found_error, and 25 with labels diverged from the saved steps.
- It cached one sample per problem: 6,755 problems.
  - Against the official MATH files, 6,755 / 6,755 are in MATH train and 0 in MATH test.
  - On the final rows, 0 MATH-test matches and 0 non-train problems: `train/prov-gate.json`, `slice/prov-gate.json`.
- The jev1 `cmd_check` Index fingerprint dropped 447 judge_hard rows as MATH/GSM-style suite matches.

## Gates (all 0; GATES.json `all_pass` includes the shape gate)
**GATES.json `all_pass`: true** (sha256 `e449e11e…1974`).

Re-run on the FINAL train AND slice records. Every gate is 0 in both:

| gate | train | slice |
|---|---|---|
| split provenance (`jev3_prov_gate.py`) | 0 | 0 |
| strict full-row Index fp (`jev3_strict_fp.py`) | 0 | 0 |
| verbatim ≥ 6-word leaf vs JevBench + Index DEV / TEST / HELDOUT | 0 | 0 |
| `doc_overlap_guard` Index / JevBench | 0 / 0 | 0 / 0 |
| `attack_guard` | 0 | 0 |
| `jevbench_guard rescan` | 0 | 0 |
| `jev1_gates.py --generator data.jev3`: cohort v2 / v3, Index fp item tier, frozen held-out, D1-B, gold, shape | pass | pass |

The other checks also pass:
- gold-position χ² p ≥ 0.001: ambiguous 0.14, temporal_numeric 0.70, long_policy 0.41;
- token windows;
- slice / train disjointness.

**Shape gate:** passes for all 6 families. Draw vs public p90 shim tokens:

| family | draw p90 | public p90 |
|---|---|---|
| judge_hard | 896 | 380 |
| adequacy | 723 | 122 |
| policy | 301 | 137 |
| ambiguous | 846 | 624 |
| temporal_numeric | 1,767 | 1,009 |
| long_policy | 3,704 | 3,606 |

**ids-sha recipe:** sha256 of the sorted record ids, joined with "\n", with no trailing newline, UTF-8 (Python default str
sort on `id`). The variant WITH a trailing newline is recorded for loaders that use it:

| draw | without trailing newline | with trailing newline |
|---|---|---|
| train | `feb681f3…3120` | `f8dfe700a328349bcdf8b2fb18b8758183deefab7a17c34cd99f26b568b29a7f` |
| slice | `844233f2…3078` | `7d82e391382eaad04065650f7d2c91be43a83700d74ec27774701b1ac75e6cf3` |

**Pointer file sha256:**
- `bschema-draw-jev3.pointer.json`: `ce70500db28739ada3506a3db51888beca7ae3b1f58ee3ac999010dcbeaafdfd`
- `bschema-draw-jev3-slice.pointer.json`: `4d1cbba60302402dca33a7d92ea68496b3bb526b7be34e1ced4c913ad652d514`

**Records uncompressed sha256:**
- train: `368fd715d8c9290989da53927e2785c5802ee07e03490ca2bd8817ebb801e020`
- slice: `1f184bb925129e83d87dd295b2efead5f500640c6543f9924d1ec649a7c4ca69`

Manifest, content and lens shas are in the pointers.

## Coverage loss from the gates (rows dropped, by family)
- **jev1 `cmd_check`:**
  - Index fingerprint, item tier: judge_hard 447, adequacy 336, temporal_numeric 332, policy 21, ambiguous 6,
    long_policy ~5.
  - Checker index_item_text: judge_hard 27, adequacy 22.
  - Other: adequacy frozen_family 8 and dup_text 4; attack_template 1.
- **Strict full-row Index fp (pre-gate):**
  - Before A4: 748 rows. 438 were the long_policy score level "none of them", 117 the temporal_numeric criterion "that
    is the right answer" and 29 the judge_hard fragment "is it correct", all our own template strings.
  - After A4 rewording: 171 rows, all natural short phrases (TAT-QA table labels such as "december 31 2019", entity
    answers such as "united states"). By family: temporal_numeric 118, ambiguous 47, long_policy 6.
  - Those rows are dropped, never edited.
- **HelpSteer3 prompt dedupe vs HelpSteer2:** adequacy 20.
- **long_policy:** 1 row over 4,096 tokens. 44 rows of the last slice group (one policy document) beyond the 200 slice
  rows were dropped, never trained.

## Caveats
- **Label conventions are the datasets', not JevBench's.** The point is human or rubric judgements instead of rule
  engines. They may still differ from the board's own rubric.
- **Single-annotator labels in places.** PrivacyQA train relevance comes from one expert per (query, sentence). Aegis
  prompt labels are human. HelpSteer2 ratings come from one rater pool, and HelpSteer3 needs consensus.
- **"decline" is never gold in ambiguous.** AmbigNQ has only answer / clarify cases. The clarify gold is the human
  multipleQAs annotation.
  - The assistant notes are a neutral, seeded sample from a 44-sentence bank. One sentence states the clarify rule,
    and it is identical across labels.
- **judge_hard is false-leaning (44.7 % true).** Only 543 true rows survived the gates. PRM800K phase2 is active-learned
  toward convincing wrong solutions.
- **long_policy has 960 rows, not 1,000, from only 27 policies.** Shared policy text across queries is expected, but
  states stay unique. The slice is disjoint by POLICY DOCUMENT (4 documents). Type mix is choice 52 % / noul 36 % /
  score 12 %, short of the 60 % choice target because the choice pool is limited.
- **temporal_numeric distractors.** TAT-QA distractors are the same report's other human answers and table cells.
  TimeQA distractors are the same article's answers for other periods. Both are hard negatives, not labels.
- **Public items are eval-only.** Only the committed public-shape fingerprint (statistics) was used, as the shape gate's
  reference.
- **Raw downloads deleted after the build.** Each is re-fetchable by the revision / sha above with
  `scripts/jev3_fetch.py`. The two streamed caches (PRM800K, HelpSteer3) are rebuilt by their stream scripts.
