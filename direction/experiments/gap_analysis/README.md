# Why the one-pass gap to H2O-Lightning-4B is large (2026-10-15)

EVAL-ONLY diagnostic on the 231 pinned public JevBench items. Nothing was fitted or selected. All systems use the
same H2O shim prompt and shim post-processing (`softmax(lp/0.8)`, yes/no floor 0.801). Intelligence (I) is jevkit's
public-only estimate, not the board's I, which is 50% sealed items and applies a gap penalty.

| System (one pass unless noted) | I | Correct |
|---|---|---|
| stock Qwen3.5-4B | 54.3 | 170/231 |
| stock Qwen3.5-4B, re-scored through `keel.runtime` (adapter check) | 54.3 | 170/231 |
| stock Qwen3.5-9B | 61.2 | 178/231 |
| **distilled student (Qwen3.5-4B + LoRA, `distill-v0-20261015`)** | **65.4** | **189/231** |
| H2O-Lightning-4B v1.1 | 77.5 | 205/231 |
| stock Qwen3.5-4B, thinking (≤ 3,072 tokens) | 84.1 | 214/231 |

## Where the 23.2-point gap (H2O − stock 4B) sits

I is the mean of three chance-corrected type scores: choice, yes/no and score. Each is tier-weighted: easy 0.1,
standard 0.2, hard 0.4. "Share of I" is how much of I a cell carries.

| Cell | Items | Share of I | Stock CC | H2O CC | Gap in I points |
|---|---|---|---|---|---|
| yes/no, hard | 38 | 0.190 | 15.8 | 63.2 | +9.0 |
| choice, hard | 67 | 0.190 | 40.2 | 70.1 | +5.7 |
| yes/no, standard | 24 | 0.095 | 50.0 | 91.7 | +4.0 |
| score, hard | 6 | 0.222 | 47.6 | 62.9 | +3.4 |
| choice, standard | 36 | 0.095 | 89.5 | 100.0 | +1.0 |
| yes/no, easy | 12 | 0.048 | 83.3 | 100.0 | +0.8 |
| choice, easy | 36 | 0.048 | 100.0 | 100.0 | 0.0 |
| score, standard | 12 | 0.111 | 99.6 | 93.5 | −0.7 |

- **Hard items carry 78% of the gap** (18.1 of 23.2 points) and 60% of I, from 111 of 231 items.
- **Yes/no carries 60% of the gap** (13.8 points). Chance correction doubles yes/no errors: CC = (acc − 0.5)/0.5.
- **Six hard score items carry 22% of I.** One of them can move I by more than a point.
- **The stock model has a "yes" prior.** It says yes 59% of the time against a 47% gold rate. It is 80% accurate
  when the answer is yes but 59% when it is no. H2O: 97% / 82%. Thinking: 89% / 97%.
- **It is an elicitation gap, not a knowledge gap.** H2O gets 38 choice and yes/no items right that stock one-pass
  misses; the same 4B model thinking gets 29 of them. Thinking beats H2O overall (214 vs 205).

## Where the remaining 12.2-point gap (H2O − student) sits

| Cell | Student CC | H2O CC | 4B thinking CC | Gap in I points |
|---|---|---|---|---|
| yes/no, hard | 31.6 | 63.2 | 73.7 | +6.0 |
| score, hard | 51.5 | 62.9 | 67.9 | +2.5 |
| choice, hard | 60.1 | 70.1 | 80.1 | +1.9 |
| choice, standard | 89.5 | 100.0 | 100.0 | +1.0 |
| yes/no, standard | 83.3 | 91.7 | 100.0 | +0.8 |

The student closed 48% of the gap (+11.1 I). Its training targeted manipulation resistance on three template
generators, not capability, yet it recovered 22 of the 38 items H2O wins. Half of what remains is hard yes/no, and
it still says yes 59% of the time.

## Reproduce

```bash
source ~/ssdm/env.sh
python -X utf8 direction/experiments/gap_analysis/score_shim.py \
  --model qwen3.5-4b --tag stock4b --model /results/students/distill-v0-20261015/merged --tag student   # ~$0.30
python -X utf8 direction/experiments/gap_analysis/decompose.py   # writes gap.json
```

Per-item predictions stay in `$SSDM_PRIVATE/gap_analysis/` and `$SSDM_PRIVATE/experiments/reasoning_headroom/raw/`.
