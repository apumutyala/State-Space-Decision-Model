# Exact 7,000-record C quota for clean B v2d

Delivered data: `C:/Users/apumu/research/CHLabs/ssdm-data/c-multirecord-7k-v2-allk-quota/train_set/`.
Pointer: `data/manifests/c-multirecord-7k-v2-allk-quota.pointer.json`.
Exporter commit: `a92d989` on `stack/rwkv7/c-multirecord-synth`.

The verified 13,575-record union starts with its 7,000 original base records,
then 6,575 compact high-K records. B's `append_extra --extra-limit 7000`
therefore loses every K73–77 record. B's final P3 selection re-sorts by seeded
ID hash, so merely reordering the full union does not guarantee coverage.
Source inspection: B `e58e3b957ca0b79e52b2386b0fd6517701388494`,
`data/b_schema.py:897–899` and `1224–1236`. No B record text was read.

This exact-size quota retains all 76 K values (2–77) and all 204 K/style
groups available in the admitted union, versus 71 K values and 189 groups
in its 7,000-record prefix. It contains 49,000 questions and 4,772 distinct
(K, style, canonical gold position) groups. K73–77 each receive 670 records.
The source does not contain all 228 possible K/style groups; no missing
group is invented. Each available group gets one record, then remaining
capacity is allocated proportionally by largest remainder. Seeded gold
position rotation maximizes coverage within each group's quota. This
preserves the source's substantial high-K weighting rather than imposing
uniform K weights. All selected record and sidecar bytes are copied exactly.

Whole-record bounds are inherited from the pinned source verification:
v2f <= 1,646 and v3w <= 2,048, trainer `9c3e435`, layout `g1v2`.
These are source upper bounds, not remeasured subset maxima. The original
13,575 pool and ongoing full100k compact export are unchanged.

## Integration contract

Only integrate into a fresh **clean B v2d** build. At delivery, no v2d
pointer or completion receipt was published. Older B v2b/v2c pools are
quarantined for NLI4CT contamination. The existing B candidates workdir
also pins old C v1 and must not be reused for this replacement.

Use this artifact as the sole C P3 pool with `--extra-quota 7000
--extra-limit 7000`, passing the committed pointer through `--extra-pointer`.
Keep B's exclusion, deduplication, tokenizer and admission checks. Require
all 7,000 `selected_ids` in `summary.json` exactly once after final selection;
fail the integration if any is removed. B may add its normal provenance
metadata, so compare source IDs and semantic content after that transform.
Do not append other overlapping original/compact C pools. A smaller final
quota does not inherit this coverage guarantee. No B files were changed.

## Evidence

- `summary.json`: pinned source verification, selected IDs, group allocation,
  baseline coverage, inherited bounds, and output hashes.
- `independent-verifier.py` / `independent-verification.json`: one output
  scan, exact original manifest/content sidecar comparison and record hash
  proof; no retokenization or repeated source semantic audit.
- `data/tests/test_c_multirecord_quota.py`: six independent CPU tests passed,
  including exact noncanonical byte preservation and fail-before-output cases.

CPU only, $0 GPU, no model load, B text, TEST, held-out Index or retention-v3
rows. These measurements demonstrate quota coverage and unchanged data
semantics; they do not demonstrate an Index gain or qualification.
