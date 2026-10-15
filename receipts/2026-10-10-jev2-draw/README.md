# bschema-draw-jev2: TRAIN-only rows for the JevBench families jev1 does not cover

Evidence lane, 2026-10-10. Branch `claude/ev-jev2-draw` (from `claude/b-jev1-draw` @60a6dd7).

- **Code:** `data/jev2.py` (stage a), `data/jev2b.py` (stage b) and `data/jev2c.py` (stage c), built by
  `scripts/jev2_build.py`. The check stage is jev1's `cmd_check`, unchanged.
- **Per-stage run:** `receipts/2026-10-10-jev2-draw/pipeline.sh`.
- **Family spec:** `FAMILY_SPEC.md`, written before any build. It was derived from jevkit docs, the family names and
  public-item FORMAT statistics only.
- **Row layout:** the same as jev1: `ssdm-train-v0`, one question `q`, `aug.jev1.family8`. Option keys keep the seeded
  per-row shuffle, and loaders must not re-sort them.

## Pointers (publish order)

| pointer | families | rows | tokens (shim, Qwen3.5-4B) | status |
|---|---|---|---|---|
| `bschema-draw-jev2a` (sha `53416094…`) | multi_hop, judge_hard, temporal_numeric, fact, tool_selection | 5 × 6,000 | ≤ 2,048 (max 1,789) | all leak gates 0; pre-audit format (dict states for multi_hop/temporal, short yes/no criteria) |
| `bschema-draw-jev2c` (sha `b9f82ab1…`) | long_policy | 1,500 | 2,001–3,347 | all leak gates 0; 5 % score rows; shape length_p90 fails (3,158 < 3,606) |
| `bschema-draw-jev2b` (sha `35620f5f…`) | probability, trap, ambiguous, tradeoff, adversarial | 5 × 1,500 | ≤ 2,048 (max 420) | all leak gates 0; score rows 523 / 740 / 478 / 543 / 872; explicit noul criteria; shape length_p90 short in all 5 |
| `bschema-draw-jev2c-v2` (sha `86960d90…`) | long_policy | 1,500 | 2,007–3,981 (p90 3,650) | all leak gates 0; score 294 (19.6 %); shape gate passes every check |
| `bschema-draw-jev2a-v2` (sha `eba70d2c…`) | multi_hop, judge_hard, temporal_numeric, fact, tool_selection | 5 × 1,500 | ≤ 4,096 (max 2,581) | all leak gates 0; prose states (multi_hop 55 %, temporal 71 %), explicit criteria; shape length_p90 short for judge_hard / multi_hop / temporal_numeric |

**Use these:**

- **`bschema-draw-jev2` (sha `bef9ec72…`):** the combined pointer, = jev2a-v2 + jev2b + jev2c-v2. That is 16,500 rows
  over 11 families. jev2a and jev2c are listed there as superseded, so do not mix them in.
- **`bschema-mix-jev1-jev2` (sha `71b503eb…`):** the mix manifest. It combines jev1 (with its long_policy replaced by
  jev2c-v2, per JEV1_AUDIT) and the combined jev2. That gives 18 families and 100,500 rows. The default option is
  `equal_family` (each family 1/18); `natural` is also recorded. max_len is 4,096. Cross-draw content overlap is 0.

Each stage's receipt directory holds:

- `GATES.json`: every gate plus `all_pass`, gold-position χ² p per family, score-row counts and the shape gate;
- `build-report.json`: per family sources, licences, subfamilies, types, label rates, token quantiles and the
  histogram;
- `target_provenance.json`;
- the raw gate receipts: `doc-overlap-guard.json`, `attack-guard.json`, `jevbench-guard-rescan.json`,
  `row-gates.json`, `shape-gate.json`;
- `sources.sha256`.

## Sources and licences

| source | split | licence | used in |
|---|---|---|---|
| MuSiQue-Ans v1.0 (HF mirror `bdsaglam/musique` @22873a40) | answerable train | CC-BY-4.0 | multi_hop |
| GSM8K (`openai/gsm8k` main/train, sources.lock) | train (two disjoint hash partitions) | MIT | judge_hard, temporal_numeric |
| BoolQ (`google/boolq` train, sources.lock) | train | CC-BY-SA-3.0 | fact |
| Glaive function-calling v2 (sources.lock) | train | Apache-2.0 | tool_selection |
| ContractNLI (HF mirror `kiddothe2b/contract-nli` @059927b9) | train | CC-BY-4.0 | long_policy candidates; **0 kept** (all hit the Index fingerprint) |
| own generators (`GEN_SPECS` in data/jev2*.py) | n/a | repo | every family |

## Caveats

- **The public items are eval-only.** Only their format statistics were read: types, K, state type, lengths, criteria
  presence. `public-shape-fingerprint.json` holds statistics only.
- **jev2a predates the JEV1 audit.** It has dict states for multi_hop and temporal_numeric, `yes`/`no` noul criteria
  on its synthetic yes/no rows, and multi_hop shorter than the public items. jev2a-v2 supersedes it for training
  (prose rendering, explicit criteria, longer multi_hop). Do not mix jev2a and jev2a-v2: they share source items.
- **jev2c (v1) is superseded by jev2c-v2.** v1 has too few score rows and a length p90 under the public p90.
- **Synthetic share is high.** Stages b and c are 100 % rule generators. Gold is exact, but the surface variety is
  bounded by templates; per-family duplicate drops are in each `check-report`.
- **GSM8K is an Index benchmark**; its train split is allowed. The Index guards dropped overlapping rows (counts in
  the build reports).
- **Choice options that no case can earn.** Some long_policy K = 5–6 options ("refer to the policy owner", "chapter
  does not apply", "postpone") are always wrong by construction; they exist only to vary K.
- **One label mapping is inherited.** The "ambiguous" family's gold is ask / act / decline by an explicit
  match-count + policy rule. It is the only family where an abstain-style answer is gold.
