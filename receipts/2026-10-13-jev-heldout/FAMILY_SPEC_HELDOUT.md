# jev-heldout: family spec (written before any heldout source was fetched, generated or gated)

Evidence lane (data worker), 2026-10-13. Branch `claude/ev-jev-heldout`, cut from `claude/ev-jev3-draw` @5c963b25.
Pre-registration: state `PREREG.md` **EV-PREREG-JEV-HELDOUT** (state @d48e500c), pushed before this file.

## What this set is (and is not)
- A **fresh, sealed, JevBench-shaped evaluation set** of about 2,200 items (floor 2,000) for future pre-registered
  cells: a SECONDARY, higher-power confirmation set. At n = 231 the paired Capability CI half-width is ~±8.3; at
  n ≈ 2,000–2,200 the projection is ~±2.8 / ±2.7 (power section of the prereg).
- **Never** training data, never a checkpoint-selection slice, never a temperature / calibration fit set, never a
  readout / prompt tuning set. Scored at most once per pre-registered cell. Never a substitute for a JevBench claim.
- Items are written in jevkit `Item` form (`split: "heldout"`) so `jevkit score --items <sealed items.jsonl>` scores
  them unchanged, plus the `ssdm-train-v0` row form (`records.jsonl.gz`) so every existing leak gate runs unchanged.

## Inputs used to design each family
1. Family names and jevkit's docs (as in jev2 / jev3).
2. The committed public-shape fingerprint `receipts/2026-10-10-jev2-draw/public-shape-fingerprint.json` (statistics
   only: per family n, type counts, choice K, string-state share, shim-token quantiles).
3. The source datasets' own documentation and schemas.

No public item text, option or label is read; nothing is selected on public items.

## Composition targets (format level; scale 2,200 / 231 ≈ 9.52)
| family | target | public types (fingerprint) | heldout type quota | public K | public string share | public tok p50 / p90 |
|---|---|---|---|---|---|---|
| adequacy | 114 | noul 12 | noul 114 | – | 1.00 | 118 / 122 |
| adversarial | 57 | choice 3 / noul 2 / score 1 | choice 29 / noul 19 / score 9 | 4 | 1.00 | 267 / 321 |
| ambiguous | 67 | choice 7 | choice 67 | 3 (6), 4 (1) | 0.86 | 518 / 624 |
| extraction | 229 | choice 24 | choice 229 | 4 (21), 5 (3) | 1.00 | 113 / 123 |
| fact | 114 | noul 12 | noul 114 | – | 1.00 | 115 / 122 |
| intent | 229 | choice 24 | choice 229 | 5 | 1.00 | 134 / 148 |
| judge_hard | 162 | noul 17 | noul 162 | – | 0.00 | 313 / 380 |
| long_policy | 181 | choice 12 / noul 5 / score 2 | choice 114 / noul 48 / score 19 | 4, 5, 6 | 1.00 | 2,658 / 3,606 |
| multi_hop | 171 | choice 16 / score 1 / noul 1 | choice 152 / score 10 / noul 9 | 4 (14), 6 (2) | 0.56 | 2,201 / 2,648 |
| ordinal | 114 | score 12 | score 114 | – | 1.00 | 145 / 148 |
| policy | 114 | noul 12 | noul 114 | – | 1.00 | 129 / 137 |
| probability | 95 | choice 7 / noul 3 | choice 67 / noul 28 | 3 | 1.00 | 622 / 1,239 |
| routing | 114 | choice 12 | choice 114 | 6 | 1.00 | 157 / 161 |
| routing_hard | 48 | choice 5 | choice 48 | 5 | 0.00 | 305 / 307 |
| temporal_numeric | 143 | choice 10 / noul 4 / score 1 | choice 95 / noul 38 / score 10 | 3, 4, 5, 6 | 0.73 | 618 / 1,009 |
| tool_selection | 114 | choice 12 | choice 114 | 5 | 1.00 | 146 / 149 |
| tradeoff | 57 | choice 4 / noul 2 | choice 38 / noul 19 | 4 (1), 5 (3) | 1.00 | 741 / 816 |
| trap | 76 | noul 4 / score 1 / choice 3 | noul 38 / choice 29 / score 9 | 4 | 1.00 | 300 / 368 |
| **total** | **2,199** | choice 139 / noul 74 / score 18 | | | | |

Tolerances (gate): each family within ±15 % of its target; per-family type shares within ±10 pp of the public's; K
and types a subset of the public's per family; string-state share within ±15 pp; noul criteria on every noul item;
long_policy 2,000–4,096 shim tokens; every other family ≤ 4,096. A family below 70 % of its target after the gates is
reported SHORT, never padded from another source.

## Families, sources and gold
Human-labelled gold wherever a non-train split (or a provably unused train row) outside the Decision Index exists.

| family | source (split; licence) | gold | rendering (NEW instruction / criteria banks) |
|---|---|---|---|
| adequacy | HelpSteer2 **validation** (CC-BY-4.0); HelpSteer3 feedback **validation** only if HS2 validation cannot balance noul 45–55 % | human 0–4 attribute rating, bar ≥ 3 true / ≤ 1 false, 2 dropped (as jev3); HS3: consensus verdicts | plain-string "request / response" state; noul with explicit criteria |
| policy | Aegis 2.0 **validation + test** (CC-BY-4.0) | human prompt / response labels only (never llm_jury); REDACTED skipped | short prose policy + message (+ reply); noul message / reply / category views |
| judge_hard | PRM800K **phase1_test + phase2_test** (MIT): MATH-test problems, human step ratings | labeller's +1 / −1 step ratings and `finish_reason` (never the answer check) | dict `{request, response}` (public is dict); noul all / step / prefix views; ≤ 2 samples per problem, group = problem |
| ambiguous | AmbigNQ light **dev** (CC-BY-SA-3.0) | human singleAnswer vs multipleQAs | assistant notes (new bank) + question; choice K 3 (~85 %) / K 4; 86 % string |
| temporal_numeric | TAT-QA **dev** (CC-BY-4.0) ~65 % + TimeQA human-annotated **dev + test** (BSD-3) ~35 % | human answers / crowd spans | choice K 3–6 from the same document's other answers / cells; noul "is the answer X"; score = TAT-QA count; 73 % string |
| long_policy | PrivacyQA **test** (MIT) | expert relevance: a sentence is relevant when a majority of its annotators mark it | policy window 2,000–4,096 shim tokens + question; choice K 4–6 / noul / score as jev3, new banks |
| fact | BoolQ **validation** (CC-BY-SA-3.0) | human yes / no | passage + question as a plain string; noul with explicit criteria; passages short (public p90 122) |
| multi_hop | MuSiQue-Ans **dev**, answerable (CC-BY-4.0) | human-composed answer | paragraphs (supporting + distractors) + question; choice K 4 (~88 %) / 6, distractors = other paragraphs' entities of the same type / other dev answers; 56 % string; plus ~6 % score / noul views |
| intent | MASSIVE en-US **test** partition, hash part 0 (CC-BY-4.0) | human intent label | utterance + K 5 intent options (gold + 4 intents of other or same scenario) |
| extraction | MASSIVE en-US **test** partition, hash part 1 (CC-BY-4.0) | human slot annotation | utterance + "what is the <slot>?" K 4 (~88 %) / 5 over the utterance's spans |
| routing | MASSIVE en-US **dev** partition (CC-BY-4.0) | human scenario label | utterance + K 6 team options (scenario descriptions, new wording) |
| ordinal | SST-5 **test** (validation if short) (SST research release) | human 5-level sentiment | review text + score over 5 levels |
| tool_selection | Glaive function-calling v2 rows **never used by any jev pool** (Apache-2.0) — **non-human gold** (the dataset's own call; synthetic) | the call in the dataset | system tool list + user turn; choice K 5 (gold tool + 3 other listed / catalog tools + "no tool") |
| routing_hard | **rule engine** (jev1 `_model_route`, indices ≥ 10,000,000) | executable routing rule | dict state, choice K 5 |
| probability | **rule engine** (jev2b `_prob`, fresh indices) | exact fractions | jev2b render (banks reused; overlap reported) |
| trap | **rule engine** (jev2b `_trap`) | rule override / false premise engine | jev2b render |
| tradeoff | **rule engine** (jev2b `_tradeoff`) | constrained choice engine | jev2b render |
| adversarial | **rule engine** (jev2b `_adv`) | rules over embedded instructions | jev2b render |

Not used (in the Decision Index, or train-only, or licence): GSM8K test (Index suite 30), BANKING77 / CLINC150 test
(Index 4 / 5), BFCL / ToolRet / API-Bank / When2Call (Index 1 / 2 / 3 / 62), ContractNLI (Index 11), HoVer (Index 61;
HotpotQA-derived MuSiQue rows are gated by the strict full-row fingerprint), OPP-115 / PolicyQA (non-commercial).

## Common rules
- Row layout = jev3 layout (`ssdm-train-v0`, one question `q`, `aug.jev1.family8`, `aug.jev1.draw = "jev-heldout"`,
  `aug.jev1.group`, `aug.jev1.prov` = dataset / revision / licence / split / row key / row sha256). Ids end in
  `~view:jevho-<view>`, disjoint by construction from every jev1 / jev2 / jev3 id.
- `source_split` is the real split (`validation`, `dev`, `test`, `train-unused`, `synthetic`).
- Option keys keep a seeded per-row shuffle; gold appears once (normalised text); every noul row carries criteria.
- Label-blind sha256 rank (salt `jevho`) inside each source; then the noul balance (45–55 % true where both strata
  allow), the type quota and the K quota per family; one source item serves at most one row unless stated (judge_hard
  ≤ 2 samples per problem; long_policy / temporal_numeric documents may serve several questions — states stay unique).
- **State dedupe:** normalised state (whitespace collapsed, casefolded, dict states as canonical JSON) is unique over
  the whole set.

## Gates (0 hits on the FINAL records; GATES.json all_pass includes the shape gate)
1. vs jev1 / jev2a / jev2a-v2 / jev2b / jev2c / jev2c-v2 / jev3 / jev3-slice records (and pools on disk): shared ids,
   source items (parent ids + dataset row keys), content hashes, normalised states → 0 (`scripts/jevho_disjoint.py`).
2. vs the 231 JevBench public items: `train.jevbench_guard rescan` (13-gram) + `scripts/regate_exact_leaves.py`
   (verbatim ≥ 6-word leaves).
3. vs Decision Index DEV / TEST / HELDOUT / proxy600: `train.doc_overlap_guard scan`, `scripts/jev3_strict_fp.py`
   (strict full-row `data.dedup.Fingerprint`), `scripts/regate_line_exact.py`; the build-stage jev1 `cmd_check`
   (gold-once, Checker cohort v2 / v3 / SplitGuard / harness.heldout, Index fingerprint item tier, JevBench fingerprint
   + exact segment, attack_guard).
4. `train.attack_guard scan`.
5. Split provenance: every row's key resolves to its stated non-train file (`scripts/jevho_prov_gate.py`).
6. Gold-position χ² per family on the presented order, p ≥ 0.001.
7. Shape gate: `scripts/jev2_shape_gate.py check` + the two-sided tolerances above (`scripts/jevho_build.py final`).

## Sealing
`ssdm-data/m2/build/jev-heldout/`: `records.jsonl.gz`, `items.jsonl`, `manifest.jsonl.gz`, `lens.jsonl.gz`,
`SEAL.json`, `ACCESS.log`. The pointer `data/manifests/jev-heldout.pointer.json` publishes the shas and counts only.
The builder prints counts, never item text. If the build cannot finish on the laptop, a STAGE pointer
(`jev-heldout-stage.pointer.json`, role "stage, NOT sealed for use") records what exists.

## Amendments (written after the first candidate pool and its pre-gate COUNTS; no item text, label or model output read)
- **H1 (shape: engine-row length).** The reused jev2b engines write states far shorter than the public items of their
  families (first pool p90 vs public p90: probability 215 vs 1,239, trap 201 vs 368, tradeoff 410 vs 816, adversarial
  220 vs 321 shim tokens), which fails the shape gate's length rule. Their states are prefixed with neutral
  administrative "file notes" (a fixed 36-sentence bank: no digits, names, rules or facts; the header says they have no
  bearing on the question) up to a label-blind target drawn from U(public p50, public max x 1.05). Decision content
  and gold are unchanged.
- **H2 (long_policy negatives).** Gold relevance follows the annotator MAJORITY (as specified); the generator's extra
  "clean negative" filter (no sentence marked relevant by any annotator) left only 7 noul-false rows, so it is
  removed. Minority-marked sentences are still never used as negatives. All three view rounds are generated (plan 900).
- **H3 (template string vs the strict Index fingerprint).** The trap engine's score level "none of them" hits the
  strict full-row Index fingerprint (as in jev3 A4); it is reworded to "zero of them". Natural source text that hits a
  gate is dropped, never edited.
- **H4 (jev disjointness text rule).** Rule-engine rows hit ONLY by an exact >= 4-word line shared with a jev draw are
  the reused engines' own template sentences (329 rows), not a shared item; they are reported and kept. Every key hit
  (0 on the first pool) and every human-source text hit is dropped.
- **H5 (candidate plans).** The first pool's pre-gates dropped 1,472 / 7,204 rows (multi_hop 365 of 520, mainly
  Wikipedia paragraphs shared with Index suites / MuSiQue train; temporal_numeric 215; tool_selection 233 + 52 state
  duplicates; judge_hard 172). Plans are raised (multi_hop 1,100; judge_hard 800; fact 700; policy 700; tool_selection
  700; ambiguous 400; TAT-QA 600; TimeQA 450 with <= 5 questions per article) so the build-check stage still leaves
  each family at its target. Selection stays label-blind rank order; a larger plan only extends the same ranked list.
- **H6 (adequacy fallback, as pre-registered).** After the gates, HelpSteer2 validation left too few "false" rows for
  the 45-55 % noul balance (first final: 58.8 % true). The pre-registered fallback is used: HelpSteer3 feedback
  VALIDATION (nvidia/HelpSteer3 @f6d14577, CC-BY-4.0), human consensus verdicts exactly as jev3 A1 (>= 2 verdicts,
  strict majority on one side, none on the other; good = mostly / perfectly, bad = not / slightly), English general /
  stem, context + response <= 2,400 chars. Its rows join the adequacy candidate pool under the same label-blind rank;
  every gate is re-run on the whole pool. The first final (records sha 3a43fee9...) was superseded before any post gate
  or scoring read it.
