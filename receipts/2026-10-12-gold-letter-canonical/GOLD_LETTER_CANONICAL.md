# Gold-letter prior under CANONICAL option order (JEV3-LM shim diagnostic)

Evidence lane, 2026-10-12, $0, CPU only. Raw numbers: `gold_letter_canonical.json` (same folder). Script:
`scripts/gold_letter_canonical.py` (streams the gz records; no GPU).

## Takeaway first
- **Canonical order creates a letter prior in TRAIN choice items. It does not create one on the public items.**
  - Train choice: gold = A on **35.0%** of items. "Always A" would score 28.0% by chance. χ² = 122 (df 14), p = 2.5e-19.
    Under the presented (shuffled) order, A is 28.7% (χ² = 9.8, p = 0.78).
  - Slice choice: A = **38.5%** vs 27.9% chance, p = 1.5e-4.
  - Public choice (139 items): A = **24.5%** vs 23.0% chance, p = 0.78. That is no prior.
- **One family drives it: `ambiguous`**, where gold A = 46.2% vs 32.2% chance (p = 1.4e-24).
  - Its train labels are only `answer` or `ask` (500 each). `decline` is never gold.
  - The decline descriptions ("Decline …", "Refuse …") sort after the "Answer …" / "Ask …" / "Check with …" ones, so the never-gold option usually lands at C or D.
  - `long_policy` and `temporal_numeric` choice items are uniform under canonical order (p = 0.47 and 0.65).
- **Consequence for the gate:** the slice has the same A bias as train. A letter-shortcut model would gain on the slice and **not** on public.
  - So a slice gain on choice items is not evidence against a letter shortcut.
  - Check choice gains on public, or on slice families other than `ambiguous`.
- **noul:** under canonical order the letter is mostly the label identity. Descriptions are templated, and the `true` option sits at B in 3035 of 3670 train items (83%); in `policy` it is at B in all of them.
  - So the letter prior equals the label prior: train A = 52.1% (p = 0.012), `judge_hard` A = 59.4% (p = 2.8e-9).
  - This is not a new shortcut, but the letter now carries the label prior.
  - Public noul descriptions are free text: A = 47.3%, p = 0.64.
- **score:** the letter is the level, so this is the gold-level prior. It does not depend on order.
  - Train is strongly non-uniform: χ² = 92 (df 7), p = 4e-17.
  - `temporal_numeric` score is **never** level 0 (0/85). `long_policy` score is level 0 on 49.5%.
  - Public score: p = 0.59 (n = 18).
- **Abstain-type options** (none / other / unknown / cannot / decline / refuse / escalate / neither):
  - Train choice: such an option sorts first on 219 of 2094 items (10.5%), and is first **and** gold on 36 (1.7%, all `long_policy` "none").
  - Slice: first on 53 of 416, first and gold on 10 (2.4%).
  - Public: first on 2 of 139, first and gold on 1.
  - The bigger prior is content, not position: `long_policy` choice gold is "none" on 224/503 = 44.5% of train items.
- **Share of choice items whose gold is the alphabetically-first option** (= canonical A under this casefolded key): train **35.0%**, slice **38.5%**, public **24.5%**.
- **Public: presented vs canonical.**

  | order | A | best constant letter | χ² p |
  |---|---|---|---|
  | presented order (criteria dict, which is key-sorted in the pinned files) | 21.6% | 24.5% | 0.88 |
  | item `labels` order (authored) | 18.7% | 27.3% | 0.96 |
  | canonical | 24.5% | 24.5% | 0.78 |

  Canonicalisation neither creates nor removes a public position prior.

## Canonical key (stated; the lead's implementation was not on `stack/qwen35/jev3-lm` at run time; the branch did not exist)
```
key(desc) = " ".join(unicodedata.normalize("NFC", str(desc)).split()).casefold()
order     = sorted by (key(desc), sha256(NFC(desc).utf8))   # content hash breaks key ties
tie_set   = options whose (key, hash) are equal (exact duplicates); gold in a tie_set counts separately
letters   = A, B, C, ... ; noul = the true/false criteria descriptions sorted the same way; score = level order
```
- No tie sets occurred in any set: 0 gold-in-tie and 0 items with any tie, across train, slice and public.
- So the results do not depend on how ties are broken. They could depend on the key if the lead's key differs, for example by not casefolding. Re-run the script with the lead's key once it is pushed.

## Headline table (canonical order)
| set | type | n | K mix | gold=A | chance A | best const. letter (acc) | chance acc | χ² vs uniform (df) | p |
|---|---|---|---|---|---|---|---|---|---|
| train | choice | 2094 | 3:1057 4:587 5:334 6:116 | **35.0%** | 28.0% | A (35.0%) | 28.0% | 122.0 (14) | 2.5e-19 |
| train | noul | 3670 | 2 | 52.1% | 50% | A (52.1%) | 50% | 6.3 (1) | 0.012 |
| train | score | 196 | 4:111 5:85 | 28.1% | 22.8% | B (31.6%) | 22.8% | 92.2 (7) | 4.4e-17 |
| slice | choice | 416 | 3:209 4:110 5:76 6:21 | **38.5%** | 27.9% | A (38.5%) | 27.9% | 41.5 (14) | 1.5e-4 |
| slice | noul | 741 | 2 | 50.9% | 50% | A (50.9%) | 50% | 0.2 (1) | 0.63 |
| slice | score | 43 | 4:28 5:15 | 27.9% | 23.3% | B (37.2%) | 23.3% | 24.4 (7) | 9.8e-4 |
| public | choice | 139 | 3:15 4:53 5:55 6:16 | 24.5% | 23.0% | A (24.5%) | 23.0% | 9.7 (14) | 0.78 |
| public | noul | 74 | 2 | 47.3% | 50% | B (52.7%) | 50% | 0.2 (1) | 0.64 |
| public | score | 18 | 4:17 5:1 | 27.8% | 24.7% | C (33.3%) | 24.7% | 5.6 (7) | 0.59 |

How to read the columns:
- χ² is stratified by K: one test per K against uniform over K letters, then the χ² values and the df are summed.
- For noul the uniform is over the 2 letters. For score it is over the levels.
- "Chance" is the mean of 1/K over the items.
- Small public strata have expected counts below 5, so their p values are rough.

## Per family (canonical; train / slice; chance in brackets)
| family | type | train gold=A | train χ² p | slice gold=A | slice χ² p | note |
|---|---|---|---|---|---|---|
| ambiguous | choice | **46.2%** (32.2%) | 1.4e-24 | **50.5%** (32.0%) | 1.0e-7 | gold only answer/ask; decline sorts late |
| long_policy | choice | 21.5% (22.0%) | 0.47 | 29.7% (22.3%) | 0.17 | "none" is gold on 44.5% (train); content, not position |
| temporal_numeric | choice | 27.6% (25.8%) | 0.65 | 25.2% (25.5%) | 0.93 | free-text values; clean |
| judge_hard | noul | 59.4% | 2.8e-9 | 56.0% | 0.09 | `true` at B in 88% of items, so A ≈ "false" (55.3% of gold) |
| adequacy | noul | 49.1% | 0.57 | 47.5% | 0.48 | |
| policy | noul | 50.5% | 0.75 | 47.5% | 0.48 | `true` always B |
| long_policy | noul | 47.7% | 0.39 | 53.5% | 0.55 | |
| temporal_numeric | noul | 48.1% | 0.51 | 52.9% | 0.63 | |
| long_policy | score | level 0 = 49.5% (25%) | 1.6e-8 | 42.9% | 8.4e-3 | |
| temporal_numeric | score | level 0 = **0%**, B 42.4% (20%) | 8.8e-11 | 0% | 0.013 | never level 0 |

Notes on the table:
- Presented-order χ² for train ambiguous is p = 0.14. So the ambiguous letter bias comes from canonicalisation.
- For comparison, the existing `decide.schema.canonical_options` order (choice sorted by option **id**) is worse on train: A = 38.5%, χ² = 423, p = 1.8e-81.
  - Under that order, the never-gold `decline` id always sorts last in `ambiguous`.
- Public per family (n ≤ 24 per cell, so read these as descriptive):
  - No family has a significant canonical letter prior except `long_policy` choice (D = 58%, p = 0.011, n = 12).
  - `long_policy` sits at A = 8%. A train-learned "A" shortcut would hurt there, not help.
  - The full table is in the JSON under `public.can.by_family`.

## Mapping in `ambiguous` (train, canonical)
| gold | A | B | C | D |
|---|---|---|---|---|
| answer (500) | 263 | 88 | 141 | 8 |
| ask (500) | 199 | 212 | 82 | 7 |

The letter is not a fixed function of the label: the description templates vary. "Never choose the decline-like letter" is a real shortcut (C/D gold on only 238 of 1000 items, against 356 expected uniform), and "prefer A" follows from it.

## What this means for JEV3-LM
1. **Risk is real but bounded.**
   - The only exploitable train letter prior is in `ambiguous` (1000 of 2094 choice items). "Always A" there gains +14 pp over chance.
   - On public choice, canonical A is 24.5% vs 23.0% chance. A learned A-bias transfers almost nothing and costs on `long_policy` (A = 8%).
   - The expected public harm is mild miscalibration toward A on choice. A wrong answer would be a large failure.
2. **Cheap mitigations, in order:**
   - (a) Train under canonical order but add `ambiguous` items where `decline` is gold. Without that, the label prior (decline never gold) lives in the letter under canonical order and in the content otherwise.
   - (b) Or train on random presented order and canonicalise only at inference. That gives zero letter prior in train (presented χ² p = 0.78), at the cost of train/serve order mismatch.
   - (c) Or, the minimal one: a per-letter calibration check on the slice restricted to non-`ambiguous` choice items.
3. **Gate hygiene:**
   - Report slice choice accuracy with and without `ambiguous`, because the slice shares the A bias.
   - Read the public-only choice letter-confusion before and after training. A shift of predicted-A share well above 24.5% on public is the shortcut signature.

## Provenance and inputs
- TRAIN: `data/manifests/bschema-draw-jev3.pointer.json` on `claude/ev-jev3-draw` @ 5c963b25.
  - Data: `ssdm-data/m2/build/bschema-draw-jev3/records.jsonl.gz`, uncompressed sha256 368fd715… (matches the pointer), 5960 rows.
- SLICE: `ssdm-data/m2/build/bschema-draw-jev3-slice/records.jsonl.gz`, 1200 rows (gz sha256 00fafdae…).
- PUBLIC: `jevkit.items.load_public()`, pin b6b8fff7, 231 items (sha256-verified on load).
  - Used for **format-level statistics only** (gold position).
  - Eval-only: never used for training, tuning or selection.
- Option order in train/slice records is the criteria dict order, which is the seeded per-row shuffle. Public "presented" order is the criteria dict order of the pinned files.
- Abstain regex: see `abstain_regex` in the JSON. It matches on the option id or on the normalised description.

Reproduce: `python -X utf8 scripts/gold_letter_canonical.py --repo . --out gold_letter_canonical.json` (about 10 s, CPU; `--data` defaults to `C:\Users\apumu\research\CHLabs\ssdm-data`).
