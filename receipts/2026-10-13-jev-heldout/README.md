# jev-heldout: a fresh, SEALED, JevBench-shaped secondary confirmation set (2,199 items)

Evidence lane (data worker), 2026-10-13. Branch `claude/ev-jev-heldout`, cut from `claude/ev-jev3-draw` @5c963b25. $0:
laptop CPU only, no cluster job of any kind.

- **Pre-registration:** state `PREREG.md` **EV-PREREG-JEV-HELDOUT**, state commit **d48e500c**, pushed before any heldout
  source was fetched.
- **Spec:** `FAMILY_SPEC_HELDOUT.md`, committed at 907bfa66 before any fetch or build. Amendments H1–H6 were written
  after pool-level COUNTS only. No item text, label or model output was read.
- **Use rule:** secondary confirmation only. Score it AT MOST ONCE per pre-registered cell, with
  `python -X utf8 -m jevkit score --items <ssdm-data/m2/build/jev-heldout/items.jsonl> --pred P --price ...`.
  - The cell's prereg must name jev-heldout before its first prediction exists.
  - Never use it for training, checkpoint selection, temperature or calibration fitting, or readout or prompt tuning.
  - It never substitutes for a JevBench claim (those stay on the 231 public items or the board).
  - Append every read to `ACCESS.log`.

## Seal / pointer
| what | value |
|---|---|
| pointer | `data/manifests/jev-heldout.pointer.json`, sha256 in the PROGRESS / ledger line |
| records (ssdm-train-v0) uncompressed sha256 | `09b5d22b8d116bbc3586d5bd0381ffdca53f3d57eb44636eb47bcc80bf445008` |
| items.jsonl (jevkit Item, split "heldout") sha256 | `17a40d43b13b0cfc66342fd54e6b5d20a6dbc797aa1942069b3d79046375dade` |
| ids sha256 (sorted, "\n"-joined, no trailing newline) | `c175d8a0a7e28a120af8ed3a00d6c5b99371f83b3fc5dd8c31864b17cd8cfd5d` |
| manifest / lens uncompressed sha256 | `fe278e4b…d76c` / `22d05503…f775` |
| location | `ssdm-data/m2/build/jev-heldout/` (records, items, manifest, lens, SEAL.json, ACCESS.log), outside git |
| generator commit | `125e17c9` (`data/jev_heldout.py`); gate / bookkeeping scripts as of this commit |

The candidate pool, which is a superset that contains the sealed items, is in `ssdm-data/m2/build/jev-heldout-work/`
under the same access rule. Two earlier finals were superseded before any scoring:
- `3a43fee9…`: HelpSteer2-only adequacy, 58.8 % true;
- `b1c9e350…`: the final-set jev-disjointness gate found 69 rows.

`ACCESS.log` records both.

## Composition (2,199; public 231 × 9.52)
| family | n | types | choice K | sources (split) | gold | string share (public) | noul true |
|---|---|---|---|---|---|---|---|
| adequacy | 114 | noul 114 | – | HelpSteer2 validation 52 + HelpSteer3 feedback validation 62 | human ratings / consensus verdicts | 1.00 (1.00) | 50.0 % |
| adversarial | 57 | choice 28 / noul 19 / score 10 | 4 | **rule engine** jev2b `_adv` (fresh indices) | engine | 1.00 (1.00) | 52.6 % |
| ambiguous | 67 | choice 67 | 3 (51), 4 (16) | AmbigNQ light **dev** | human single / multi annotation | 0.88 (0.86) | – |
| extraction | 229 | choice 229 | 4 (208), 5 (21) | MASSIVE en-US **test** (part 1) | human slot spans | 1.00 (1.00) | – |
| fact | 114 | noul 114 | – | BoolQ **validation** | human yes / no | 1.00 (1.00) | 50.0 % |
| intent | 229 | choice 229 | 5 | MASSIVE en-US **test** (part 0) | human intent | 1.00 (1.00) | – |
| judge_hard | 162 | noul 162 | – | PRM800K phase1/phase2 **test** (MATH-test problems) | human step ratings | 0.00 (0.00) | 50.0 % |
| long_policy | 181 | choice 114 / noul 48 / score 19 | 4 (73), 5 (31), 6 (10) | PrivacyQA **test** | annotator-majority relevance | 1.00 (1.00) | 50.0 % |
| multi_hop | 171 | choice 151 / noul 10 / score 10 | 4 (135), 6 (16) | MuSiQue-Ans **dev** | human-composed answer | 0.53 (0.56) | 50.0 % |
| ordinal | 114 | score 114 | – | SST-5 **test** | human 5-level sentiment | 1.00 (1.00) | – |
| policy | 114 | noul 114 | – | Aegis 2.0 **test** 69 + **validation** 45 | human labels only | 1.00 (1.00) | 50.0 % |
| probability | 95 | choice 67 / noul 28 | 3 | **rule engine** jev2b `_prob` | exact fractions | 1.00 (1.00) | 50.0 % |
| routing | 114 | choice 114 | 6 | MASSIVE en-US **dev** | human scenario | 1.00 (1.00) | – |
| routing_hard | 48 | choice 48 | 5 | **rule engine** jev1 `_model_route` (indices ≥ 10M) | engine | 0.00 (0.00) | – |
| temporal_numeric | 143 | choice 95 / noul 47 / score 1 | 3 (42), 4 (26), 5 (19), 6 (8) | TimeQA human-annotated **test** 100 + TAT-QA **dev** 43 | human answers | 0.67 (0.73) | 46.8 % |
| tool_selection | 114 | choice 114 | 5 | Glaive FC v2 **train rows no m2 build used** | **dataset call (synthetic, NOT human)** | 1.00 (1.00) | – |
| tradeoff | 57 | choice 38 / noul 19 | 4 (7), 5 (31) | **rule engine** jev2b `_tradeoff` | engine | 1.00 (1.00) | 52.6 % |
| trap | 76 | choice 28 / noul 38 / score 10 | 4 | **rule engine** jev2b `_trap` | engine | 1.00 (1.00) | 50.0 % |
| **total** | **2,199** | **choice 1,322 / noul 713 / score 164** | | | | | |

- **Type ratio.** The set is 1,322 / 713 / 164 = 60.1 / 32.4 / 7.5 %, against the public 139 / 74 / 18 =
  60.2 / 32.0 / 7.8 %.
  - temporal_numeric got 1 score row against a quota of 10, because TAT-QA dev has few count questions that pass the
    gates. The rest was filled with choice / noul and stays inside the ±10 pp type-share tolerance.
- **Human-labelled gold** covers 1,752 items (80 %) across 12 families.
- **Non-human gold** covers 447 items (20 %):
  - rule engines: 333 items in five families, the four jev2b engines plus routing_hard;
  - tool_selection: 114 items, gold = the dataset's own synthetic call.

  None of these families has a human source outside the Decision Index: BFCL / ToolRet / API-Bank / When2Call /
  BANKING77 / CLINC150 / GSM8K are Index suites.
- **Tiers** are assigned label-blind by hash in the public proportions (72 / 48 / 111): standard 631, easy 460,
  hard 1,108. They are a scoring weight only; heldout items have no real difficulty tier.
- **Shim tokens** (`jrt-h2o-shim:qwen3.5-4b`): every family's p90 is ≥ the public p90. Examples:
  - long_policy 3,812 vs 3,606;
  - multi_hop 2,878 vs 2,648;
  - probability 1,488 vs 1,239.

  long_policy lies in 2,000–4,096; all other families are ≤ 4,096. Full table in `final/shape-gate.json`.

## Gates (GATES.json `all_pass: true`; all re-run on the FINAL records, 0 required)
| gate | final |
|---|---|
| jev disjointness vs every `bschema-draw-jev*` record set (jev1, jev2a, jev2a-v2, jev2b, jev2c, jev2c-v2, jev2h*, jev2-mix*, jev3, jev3-slice, jev3-lmr, jev3-typed) + jev1 / jev3 candidate and checked pools: ids, source items, dataset row keys, content hashes, normalised states | 0 (key hits 0) |
| … plus exact ≥ 4-word lines and 13-word shingles (human-source rows) | 0 dropped-class hits; 92 rule-engine rows share only engine template lines (reported, H4) |
| split provenance (`jevho_prov_gate.py`: non-train file + row sha256; PRM800K MATH-test; AmbigNQ dev ∉ train; MASSIVE partition; Glaive unused; engine offset) | 0 fails |
| strict full-row Index fingerprint (`jev3_strict_fp.py`) | 0 |
| verbatim ≥ 6-word leaves vs JevBench + Index DEV / TEST / HELDOUT | 0 |
| exact ≥ 4-word lines vs JevBench + every Index row (`regate_line_exact.py`) | 0 |
| 13-gram overlap vs JevBench public / GSM8K test / Index DEV / TEST / HELDOUT / proxy600 (`regate_jev2a.py overlap`) | 0 / 0 / 0 / 0 / 0 / 0 rows |
| `train.doc_overlap_guard` Index / JevBench | 0 / 0 |
| `train.attack_guard` | 0 |
| `train.jevbench_guard rescan` | 0 |
| `jev1_gates.py --generator data.jev_heldout` (cohort v2 / v3, Index fp item tier, frozen check_files + check_generator, D1-B, gold-once, Item shape) | pass (0 fails, 0 refused, 0 D1-B hits) |
| build stage `jev1 cmd_check` (gold-once, Checker incl. SplitGuard / harness.heldout / Index item text, Index fp, JevBench fp + exact segment, attack_guard) | ran on the pool: 9,884 → 8,785 kept |
| shape gate (`jev2_shape_gate.py`) + two-sided tolerances (count ±15 %, type share ±10 pp, string share ±15 pp, criteria) | pass, all 18 families |
| gold-position χ² (presented order), p ≥ 0.001 | pass (min p 0.025, multi_hop) |

**Coverage loss.**
- Pre-gates dropped 2,547 of 9,884 candidate rows: multi_hop 749, tool_selection 406, temporal_numeric 344,
  judge_hard 274, adequacy 271 and fact 249. Most were Wikipedia passages or paragraphs shared with Index suites or
  with jev training data, plus Index-fingerprint matches.
- The build check dropped the Index item tier for judge_hard, multi_hop and temporal_numeric. It also dropped 166
  policy rows on the JevBench fingerprint.
- Details are in `pre/drop-ids-summary.json` and `check-report.json`.

## Power (from the existing paired bootstrap receipts, B = 1000, seed 16)
Half-widths are of the paired 95 % CI on the 231 public items, projected by sqrt(231 / n).

| quantity | at n = 231 | at n = 2,199 |
|---|---|---|
| **Capability** median, 27 arms of `claude/ev-scorecard` @e2fcd82b | ±8.30 | **±2.69** |
| Composite A, median of all arms | ±12.62 | ±4.09 |
| Composite A, 5 arms within 10 points of H2O | ±3.37 | ±1.09 |
| Composite A, cell-kit native_shimcal vs stock (`claude/ev-cell-kit` @ef25e482) | ±1.57 | ±0.51 |

The projection assumes the public items' per-item paired variance carries over. It is not a claim; the first scoring
reports the realised half-width.

## Sources (fetched to `ssdm-data/raw/jevho/`, deleted after the build; re-fetch with `fetch.sh`, shas in `fetch-log.jsonl`)
| source | file | licence | revision / sha256 |
|---|---|---|---|
| HelpSteer2 | validation.jsonl.gz | CC-BY-4.0 | hf 990b2711 / 610eeb52… |
| HelpSteer3 | feedback/validation.jsonl.gz | CC-BY-4.0 | hf f6d14577 / b86f7158… |
| Aegis 2.0 | validation.json, test.json | CC-BY-4.0 | hf d86bb8be / a97200e2…, b0a6d602… |
| PRM800K | phase1_test, phase2_test, math_splits/test | MIT | git 7ecc7947 / f4b3bc5b…, 6b172efa…, 35dc4108… |
| AmbigNQ light | ambignq_light.zip (dev_light.json) | CC-BY-SA-3.0 | 3f5dada6… |
| TAT-QA | tatqa_dataset_dev.json | CC-BY-4.0 | git 870accc4 / 8da095a8… |
| TimeQA | human_annotated_test.json (no human dev file exists: 404) | BSD-3-Clause | git 38b05989 / 70d61fca… |
| PrivacyQA | policy_test_data.csv | MIT | git 5178ff3e / 38224092… |
| BoolQ | validation parquet | CC-BY-SA-3.0 | hf 35b264d0 / 52355d11… |
| MuSiQue-Ans | musique_ans_v1.0_dev.jsonl | CC-BY-4.0 | hf bdsaglam/musique 22873a40 / 15fa6379… |
| SST-5 | test.jsonl (+ dev) | SST research release (mirror card unspecified) | hf SetFit/sst5 e51bdcd8 / 13842161… |
| MASSIVE 1.0 | en-US test / dev partitions (the jev1 on-disk tarball) | CC-BY-4.0 | dl-2026-10-10 copy |
| Glaive FC v2 | on-disk parquet; rows used by any of 75 m2 manifests / jev records (46,363 ids) excluded | Apache-2.0 | 7e7e32f0… |

## Caveats
- **Rule-engine and synthetic families.** probability, trap, tradeoff, adversarial and routing_hard come from rule
  engines; tool_selection uses Glaive's synthetic call as gold.
  - These reuse the jev2b / jev1 instruction banks, so cells trained on jev2b / jev1 have seen the templates, though
    never the items.
  - 92 such rows share an engine template line with a jev draw.
  - Their states are padded with neutral administrative notes to the public length range (H1).
- **Label conventions are the datasets', not JevBench's.** Several labels have a single annotator: SST-5,
  HelpSteer2 ratings, and Aegis.
- **Wikipedia overlap.** multi_hop and fact are Wikipedia-based. Every row sharing a 13-gram, a ≥ 4-word line or a
  fingerprint match with the Index or with jev training was dropped. Facts the models saw in pretraining remain
  possible.
- **Public share mismatches.**
  - temporal_numeric string share is 0.67 vs 0.73 public, and it has only 1 score row.
- **Heldout I is not board I.** jevkit counts split "heldout" as the sealed side, but it is not the board's sealed
  split.
- **Builder-side reads only.** The set was generated and gated by the builder; no model was scored on it.
