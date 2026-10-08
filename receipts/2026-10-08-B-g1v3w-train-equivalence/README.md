# jrt-rwkv-g1-v3w train == readout equivalence (CPU, real G1k tokenizer, no model)

**0 mismatches over 969 rows from 513 records** (510 real training records + 3 synthetic word-id records +
1 synthetic dict-instruction record). The check covers ids, option order, answer position, gold target token,
the word-id flag, the code-key label width and field coverage.

- Rows by k: 869 with k <= 16, 100 with k 17-255.
- Row kinds: 399 word-id rows (yes/no, true/false, call/skip, NLI labels, first/second, A-D letter ids,
  intent slugs, synthetic SUPPORTED/REFUTED/NOT_ENOUGH_INFO and positive/negative/neutral); 447 coded rows
  without a key; 123 code+key rows (label width 64: 106, 24: 1, none (over budget): 16); 369 rows with
  dict-valued instructions.
- Families: all 22 in sample-150000-seed20261006, first 12 records per stratum (family x type x k bucket x
  word-like ids x dict instructions x multi-question).

Command (m2-heads venv, `CUDA_VISIBLE_DEVICES=`):

    python -X utf8 scripts/check_g1v2_train_equivalence.py --template jrt-rwkv-g1-v3w --per-stratum 12 \
      --hist-vs jrt-rwkv-g1-v2f --records <ssdm-data>/m2/build/sample-150000-seed20261006/records.jsonl.gz \
      --tokenizer-dir <ssdm-data>/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba \
      --out equivalence.json

## Gold position, v2f vs v3w (`equivalence.json` gold_hist)

Both formats list the options in canonical order (no presentation permutation), so the gold's listing position
is the same in both (k=2: 466 at position 0, 129 at position 1).

The gold target token is where they differ. In v2f the gold is the code " A" in 546/969 rows (56 %). In v3w
that drops to 283 (29 %), because word-id rows are scored by their own first token instead: " no" 121,
" not" 84, " true" 34, " false" 21, " yes" 18, " call" 17. Distinct gold tokens: v2f 70, v3w 90.
