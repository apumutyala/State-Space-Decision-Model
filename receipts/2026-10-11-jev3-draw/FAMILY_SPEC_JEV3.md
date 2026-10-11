# bschema-draw-jev3: family spec (written before any build or gate ran)

Evidence lane (data worker), 2026-10-11. Branch `claude/ev-jev3-draw`, cut from `claude/ev-jev2-draw` @282a4867.

## Why this draw exists
Two H2O-host cells gained on our held-out jev slice but were flat or negative on the 231 public items: x7c42 (200 steps)
gained +4.4 pp and q668q (about one epoch) gained +9.0 pp. The x7c42 TRANSFER table
(`claude/ev-x7c42-rescore` @bf706854, `RESCORE_X7C42.md` section 7) gives three diagnoses:

- **(a) Wrong conventions.** Our rule-engine golds teach exact-rule conventions. JevBench's judgement families follow
  human/rubric conventions.
- **(b) Gains land in saturated families.** temporal_numeric, judge_hard and long_policy show no learnable headroom on
  the slice (+3.0 n.s., 0.0, +1.5 n.s.). That is where the public errors sit.
- **(c) Shape gaps.** Dict states vs prose, intent K, the adversarial type mix, and public states 2–6× longer.

jev3 therefore targets six families with **human-labelled TRAIN-split data**. The original human label is the gold; no
row is relabelled by a model or a rule engine. Shapes are matched at the **schema level only**.

Families:

- judge_hard
- adequacy
- policy
- ambiguous
- temporal_numeric
- long_policy

The target is about 1,000 TRAIN rows per family plus a 200-row held-out selection slice, so about 1,200 gated rows per
family. Each family generates about 1,700 candidates to absorb gate drops.

## Inputs used to design each family
1. The family name and jevkit's tier notes (as in jev2's spec).
2. The committed public-shape fingerprint `receipts/2026-10-10-jev2-draw/public-shape-fingerprint.json`. It holds
   statistics only: type counts, choice K, string-state share, noul-criteria presence and shim-token quantiles.
3. The source datasets' own documentation and schemas.

No public item text, option or label was read, and nothing is selected on public items. The fingerprint is used only as
the shape gate's reference.

## Families

| family | public shape (fingerprint) | jev3 source (TRAIN split, licence) | gold (human) | jev3 shape |
|---|---|---|---|---|
| judge_hard | noul 17/17; dict state; criteria filled; tok p90 380 | **PRM800K** phase2_train (openai/prm800k @7ecc7947, MIT), streamed. Only problems whose PRM800K `math_splits` unique_id starts with `train/` (MATH TRAIN). The 4.5k MATH-test problems PRM800K trains on are excluded, and so are the 500 MATH-500 problems. One sample per problem. Quality-control and screening samples are skipped. | Human step ratings (−1/0/+1) and `finish_reason`. The final-answer check against ground truth is never used as gold. | state `{request: problem, response: the saved pre-generated steps + "# Answer"}`; noul with explicit criteria. Views: "every step correct?" (gold = labeler reached `solution` with no −1 step), "is step k correct?" (k with a human rating of +1 or −1; 0-rated steps are never asked), "are steps 1..k all correct?". noul balanced 45–55 % by label-stratified selection. |
| adequacy | noul 12/12; string; criteria filled; tok p90 122 | **HelpSteer2** train (nvidia, CC-BY-4.0, already on disk `dl-2026-10-10/helpsteer2/train.jsonl.gz`). **Only prompts never used by jev1** (no jev1 pool row from helpsteer2 / hs2pref shares the normalised prompt). One row per prompt, with prompt+response ≤ 2,400 chars. Shortest-first is NOT used; selection is by sha256 rank within the char cap. | Human 0–4 ratings. bar = attribute ≥ 3 → true, ≤ 1 → false; the midpoint 2 is dropped (as jev1). | Plain-string state ("User request: …\n\nAI response: …"); noul with paraphrased bar criteria over helpfulness / correctness / instruction-following-style attributes (helpfulness, correctness, coherence). noul balanced 45–55 %. |
| policy | noul 12/12; string; criteria filled; tok p90 137 | **Aegis 2.0** train (nvidia, CC-BY-4.0, on disk). **Only ids never used by jev1.** Prompt labels are human (prompt_label_source human); response labels only where `response_label_source == "human"`; never llm_jury. REDACTED rows are skipped. Prompt ≤ 1,200 chars. | Human safe/unsafe label; human violated_categories for category views. | Plain-string state (a short content policy in prose, then the message or the message + reply). noul with explicit criteria. Views: message unsafe under policy, reply unsafe (human response label), and "violates category X". noul balanced 45–55 %. |
| ambiguous | choice 7/7 (K 3 ×6, K 4 ×1); string 86 %; tok p90 624 | **AmbigNQ light** train (`train_light.json`, nlp.cs.washington.edu/ambigqa ambignq_light.zip, CC-BY-SA-3.0). Questions whose annotations agree (all singleAnswer or all multipleQAs). | The human ambiguity annotation: multipleQAs → ask a clarifying question; singleAnswer → answer. | choice K 3 (85 %) / K 4 (15 %). Options: answer with X, ask a clarifying question, decline. K 4 adds a second answer (for an ambiguous question, another interpretation's human answer; otherwise another question's answer of the same coarse type). State = assistant operating notes (a seeded sample from a bank of ≥ 40 neutral guideline sentences that never mention the label) + the user's question. It is a plain string in 85 % of rows and a dict `{assistant_notes, conversation}` in 15 %. Notes length is varied so the p90 is ≥ the public p90. Gold balanced about 50/50 ask/answer. "decline" is never gold (stated caveat). |
| temporal_numeric | choice 10 (K 3,4,5,6) / noul 4 / score 1; string 73 %; tok p90 1,009 | **TAT-QA** train (NExTplusplus/TAT-QA @870accc4, data CC-BY-4.0), about 65 %. **TimeQA** human-annotated train (wenhuchen/Time-Sensitive-QA @38b05989 `human_annotated_train.json`, BSD-3-Clause), about 35 %. | TAT-QA: the human-written answer + scale. TimeQA: crowd-annotated answer spans for the time-scoped question. | TAT-QA: one question per document (hybrid table + text; arithmetic / span / count). The table is rendered as text in 75 % of rows; a dict `{table, paragraphs}` is used in 25 %. Choice distractors are the SAME document's other human answers (same report, same units), then cell values. noul "is the answer X?" (gold or a sibling answer). score = TAT-QA count questions on levels 0 / 1 / 2 / 3 / 4+. TimeQA: one question per row. The window holds the title paragraph plus the paragraphs carrying the gold and distractor answers, with neighbours, ≤ 7,000 chars. Distractors are the answers to the same document's other time-scoped questions, which differ only by period, so they are hard negatives. Types target choice ~65 % (K 3–6 all present) / noul ~27 % / score ~8 %. |
| long_policy | choice 12 (K 4,5,6) / noul 5 / score 2; string 19/19; tok p50 2,658, p90 3,606, max 3,953 | **PrivacyQA** train (`policy_train_data.csv`, AbhilashaRavichander/PrivacyQA_EMNLP @5178ff3e, MIT). Its 27 TRAIN policies are rebuilt from (DocID, SentID). **ContractNLI is excluded** (it overlaps the Index fingerprint). OPP-115 and PolicyQA are not used (non-commercial terms / partial context). | Expert sentence-relevance labels per (query, sentence). | Plain-string state: the policy text (a contiguous window of sentences in document order, 2,000–4,096 shim tokens; it holds every relevant sentence where they fit, else at least one), then the user's question. Views: noul "does the policy text answer the question?" (any relevant sentence in the window; false only for queries the experts found unanswerable); choice K 4–6 "which sentence answers it?" (one relevant sentence + hard-negative irrelevant sentences from the window by word overlap + a "none of these" option, which is gold for unanswerable queries); score "how many of these N sentences are relevant?" (levels none / one / two / three or more). At most 2 views per query. Types target choice ~60 % / noul ~28 % / score ~12 %. ≤ 4,096 tokens (RWKV7 recall note, coordinator 2026-10-11). |

**Common to all families.**

- Each row follows the jev1 layout: `ssdm-train-v0`, one question `q`, `aug.jev1.family8` = family, `aug.jev1.draw` =
  `jev3`. Option keys keep a seeded per-row shuffle. Every noul row carries explicit criteria.
- Instructions are drawn from ≥ 12 paraphrases per (family, view), and none of them gives away the rule order.
- **Provenance on every row** is recorded in `aug.jev1.prov`: dataset, pinned revision (commit or file sha256), licence,
  split `train`, and the source row key (the dataset's own id, or uid / line number) plus the sha256 of the canonical
  source record.
- **Token lengths are recorded per row.** The shim lens file holds them, and `aug.jev1` holds nothing that changes the
  content. Rows are capped at 4,096 shim tokens; long_policy is kept to the 2,000–4,096 window.

## Selection, balance and the held-out slice
- Within a source, candidates are ranked by sha256(`jev3|<source>|<key>`), which is label-blind. A noul family is then
  balanced to 45–55 % true by taking from the two label strata in rank order. That is selection on the TRAIN label, as
  recommended by JEV1_AUDIT fix 1, and it is reported.
- **Held-out selection slice:** 200 rows per family from the gated rows, chosen by sha256(`jev3-slice|<group>`) over
  source GROUPS. A group is one source item, or one document for TimeQA and PrivacyQA, where many questions share a
  document. The slice is disjoint from TRAIN by id, by source item / group, and by content hash (state + questions),
  and no slice row shares a state with a TRAIN row. Its ids sha256 (sorted, "\n"-joined) is reported.

## Gates (0 hits required; the shape gate is in all_pass)
- Build check stage = jev1 `cmd_check` unchanged:
  - gold-once;
  - Checker (cohort v2 / v3, SplitGuard, harness.heldout, Index DEV/TEST);
  - the production Index Fingerprint on item-tier texts;
  - the JevBench fingerprint + exact-segment on the full row;
  - attack_guard.
- Draw-level rescans on the final records:
  - `train.doc_overlap_guard scan` (Index + JevBench);
  - `train.attack_guard scan`;
  - `train.jevbench_guard rescan`;
  - `scripts/jev1_gates.py --generator data.jev3` (cohort, Index fingerprint item tier, frozen check_files /
    check_generator, D1-B held-out, gold, shape).
- **Strict full-row Index fingerprint:** `data.dedup.Fingerprint` on every string leaf of the full row (state +
  instructions + criteria), not only the item tier.
- **Verbatim ≥ 6-word-leaf rule:** `scripts/regate_exact_leaves.py`, from ev-jev2a-regate, against JevBench public items
  and Index DEV / TEST / HELDOUT.
- **Split provenance:** every row has `source_split == "train"` and a `prov` entry whose dataset / revision / licence
  matches the source table, and its key resolves to the dataset's TRAIN file.
  - For judge_hard, the problem's PRM800K math_splits unique_id starts with `train/`.
- Gold-position χ² on the presented order, per family, p ≥ 0.001.
- `scripts/jev2_shape_gate.py check` against the public-shape fingerprint, for each family:
  - types ⊆;
  - K ⊆;
  - noul criteria filled;
  - string share;
  - p90 length ≥ public.
- Slice disjointness: id / source item / group / content / state, all 0.
