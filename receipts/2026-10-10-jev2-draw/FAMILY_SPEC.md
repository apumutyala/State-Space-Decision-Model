# bschema-draw-jev2: family spec (written before any build or gate ran)

The 231 pinned JevBench public items are EVAL-ONLY. They span 18 families. bschema-draw-jev1 trains 8 of them
(routing, adequacy, policy, intent, ordinal/severity, extraction/enum, long_policy, state tracking). This draw adds
TRAIN-ONLY rows for the other 10 (111 public items, 48 %).

**How each family was understood.** Only three inputs were used:

1. jevkit's docs. `jevkit/README.md` and the upstream `datasets_manifest.json` tier notes, for example "explicit yes/no
   fact" and "one-obvious-tool selection" for the easy tier, and "judging whether a saved math answer is correct" for
   the judge tier.
2. The family name.
3. The public items' FORMAT, read by a structure-only script that printed no item text: question type counts, option
   count K, state type and top-level key names, state length, and instruction length.

No item text, option text or label was read, copied or paraphrased. The gates (JevBench fingerprint + exact-segment +
doc-overlap) are the backstop.

Every row has the jev1 schema: `ssdm-train-v0`, one question `q`, `state` + `questions{q: {type, instructions,
criteria}}` + `labels`, and `aug.jev1.family8` = family. Option keys stay in a seeded per-row shuffle, and the loader
must not re-sort them. Every row must be ≤ 2,048 tokens under `jrt-h2o-shim:qwen3.5-4b`.

## Format observed (structure only) → what we build

| family | public format (type mix; K; state) | train rows built from | gold |
|---|---|---|---|
| multi_hop | mostly choice (K 4–6), a few noul/score; long dict states (registers, archives, footnotes, amendments) or long strings | **jev2_ref_chase** generator: asset register + team table + site table + dated archive with ownership transfers, relocations, retirements→replacements, footnoted cancellations, contact amendment; questions chain 2–4 lookups (replacement → owner/site → team name/city/contact), plus yes/no and "how many effective transfers" score. **MuSiQue-Ans train** (CC-BY-4.0): supporting + 2 distractor paragraphs as `documents`; options = answer + the decomposition's intermediate answers (hard distractors) + same-type answers | rule engine replays the archive in date order (cancelled transfers skipped); MuSiQue gold answer |
| judge_hard | noul; state `{request, response}` | **GSM8K train** (MIT, hash partition 0): request = problem, response = the reference working (annotations stripped) left clean, with the last step's result corrupted, or with one intermediate result corrupted; asks "final answer correct?" or "every step correct?". **jev2_judge_check** generator: unit conversion, date arithmetic, weekday, sorting, constrained subject line, JSON extraction, discount+tax, invoice sums, letter counts; response correct or with one injected flaw | exact (computed truth vs response; constraint checks recomputed from the final text) |
| temporal_numeric | choice (K 3–6) / noul / score; string or `{rules, event}` | **jev2_temporal** generator: business-day SLA with cutoff and holidays, return windows, month-end warranty expiry, late-fee periods with cap, upgrade proration, fixed-offset time zones (invented office names), tenure eligibility, lateness buckets (score). Distractors = typical slips (calendar vs business days, off-by-one, ignored holiday or cap). **GSM8K train** (hash partition 1) as numeric choice | `datetime` / exact arithmetic |
| fact | noul; short string state | **jev2_fact_record** generator: one- or two-clause records (orders, flights, staff, invoices, rooms, products, loans, accounts). Question asks whether a slot has a value: the true value, a different value, or a slot the text never mentions (→ no). **BoolQ train** (CC-BY-SA-3.0): passage + yes/no question | slot value from the template; BoolQ label |
| tool_selection | choice K = 5; short request string; options = tool name → description | **Glaive FC v2 train** (Apache-2.0): first user turn whose first assistant turn calls a listed function; gold = that function; distractors = the row's other functions, then pool functions sharing no name token and < 2 description tokens with the gold. **jev2_tool_pick** generator: 33 tools in 11 domains with request templates; distractors = 1–3 same-domain siblings + other domains | the source's called function; the template's tool |
| probability (stage b) | choice K 3 / noul; long string | generator: urn/dice/card/queue/base-rate scenarios, Bayes updates, expected-value comparisons, with exact fractions (Fractions); options = correct + common fallacies (base-rate neglect, conjunction, gambler's) | exact rational arithmetic |
| trap (stage b) | noul / choice K 4 / score; string | generator: questions with a built-in false premise, a distractor number, or a "most obvious" answer that the stated rules make wrong. Gold follows the stated rules, or "the premise is false / cannot be determined" | rule engine |
| ambiguous (stage b) | choice K 3–4; string or dict (policy + conversation + candidates) | generator: an agent request with 0, 1 or ≥ 2 matching candidates (contacts, files, orders). Options: act on X / ask a clarifying question / decline. Gold = act when exactly one matches, else ask (or decline when the policy forbids) | match count + policy |
| tradeoff (stage b) | choice K 4–5 / noul; long string | generator: options with cost, latency, risk and quality attributes plus a stated priority rule (hard constraints, then lexicographic or weighted objective) | exact argmax under the stated rule |
| adversarial (stage b) | choice K 2–4 / noul / score; string | generator: a record carrying an injected instruction or misleading claim inside untrusted text (a ticket body, a review, a tool output). Gold follows the trusted task rules and ignores the injection. Uses our own training-only wording; the eval attack templates are excluded by `attack_guard` | rule engine |

## Gates (each row; 0 hits required, as in jev1)

- `scripts/jev2_build.py check`, which is jev1's `cmd_check` unchanged:
  - gold-once;
  - Checker (cohort v2/v3, SplitGuard, `harness.heldout`, Index DEV/TEST);
  - production Index Fingerprint;
  - JevBench fingerprint + exact-segment;
  - attack_guard.
- Draw-level rescans on the final records:
  - `train.doc_overlap_guard scan`;
  - `train.attack_guard scan`;
  - `train.jevbench_guard rescan`;
  - `scripts/jev1_gates.py --generator data.jev2` (cohort, fingerprint, frozen `check_files` + `check_generator`, D1-B held-out, gold, shape).
- Gold-position χ² on the presented order, per family (p ≥ 0.001).
