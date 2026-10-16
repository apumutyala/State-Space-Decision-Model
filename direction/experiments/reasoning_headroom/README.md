# Reasoning-headroom probe (2026-10-14)

Question: how much JevBench Intelligence does a hybrid-SSM base gain when it may think before the same one-token
decision readout, and what does that cost on Speed and Cost?

EVAL-ONLY: the 231 pinned public JevBench items (fstandhartinger/jevbench @ b6b8fff7, MIT). Nothing was trained,
tuned or selected on them. Scores come from the repo's frozen `jevkit` CLI.

## Protocol
- Prompt: H2O-Lightning-4B v1.1's published shim (system text, `record:/question:/options:` layout, compact JSON
  state, labels A, B, ...). Content identical across arms; one-pass prompts are byte-identical to the shim's own.
- `nothink`: thinking off (`<think>\n\n</think>\n\nAnswer:` prefill), one forward pass, label-token log-probs.
- `think`: same turn, `<think>\n` open; vLLM sampling T 0.6 / top-p 0.95 / top-k 20 / seed 0, max 3,072 tokens,
  stop at `</think>`; truncated traces get Qwen's budget-forcing sentence; then `</think>\n\nAnswer:` and the same
  label read.
- Post-processing for both arms: the shim's `softmax(lp / 0.8)` and yes/no commit floor 0.801 (`shim`); `raw` =
  softmax(lp), no floor.
- Label coverage: every label token present in the top-256 log-probs for every item (0 fallbacks).
- Latency: single-stream wall time on the 72 standard items (jevkit's timed set), H100, in-process, prefix cache on
  and prompts already seen once: a local harness estimate, not card-parity timing.
- Price classes (board v1.6.1 frozen references, USD per M input / output tokens): Qwen3.5-2B 0.02 / 0.10,
  4B and H2O 0.03 / 0.15, 9B 0.10 / 0.15.

## Pins
| Model | Revision |
|---|---|
| Qwen/Qwen3.5-2B | main = 15852e8c16360a2fea060d615a32b45270f8a8fc (unchanged since 2026-03-02) |
| Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (the repo's pin) |
| Qwen/Qwen3.5-9B | main = c202236235762e1c871ad0ccb60c8ee5ba337b9a (unchanged since 2026-03-02) |
| h2oai/h2o-lightning-4b | 193ad740925b176a3b70a5a13a7cff2f2fadd01e (v1.1, the kit-parity pin) |

vLLM 0.30.0, Modal H100, one container per model, all four in parallel: 9.3 min wall. Logged GPU time 1,708 s
(≈ $1.9 at Modal's H100 list price) plus smoke runs and unlogged overhead; no invoice reconciled.

## Results (shim post-processing; local public-231 estimates, not board numbers)
| Model | Arm | I | C | S | K | Composite A | Correct /231 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | one-pass | 37.0 | 58.3 | 94.1 | 66.5 | 31.3 | 156 |
| Qwen3.5-2B | think | 68.8 | 72.5 | 59.2 | 30.8 | 19.5 | 190 |
| Qwen3.5-4B | one-pass | 54.3 | 66.0 | 93.1 | 61.2 | 66.0 | 170 |
| Qwen3.5-4B | think | 84.1 | 87.2 | 62.4 | 29.1 | 18.3 | 214 |
| Qwen3.5-9B | one-pass | 61.2 | 75.8 | 93.4 | 45.5 | 53.3 | 178 |
| Qwen3.5-9B | think | 83.3 | 88.4 | 61.1 | 27.4 | 15.8 | 218 |
| H2O-Lightning-4B v1.1 | one-pass | 77.5 | 82.8 | 93.4 | 61.2 | 76.9 | 205 |
| H2O-Lightning-4B v1.1 | think | 83.7 | 86.6 | 80.0 | 36.9 | 34.5 | 210 |

Paired bootstrap (B 1,000, seed 16, jevkit strata; point estimate [95% CI]): 4B think − 4B one-pass ΔI +29.75
[+21.05, +37.96]; 4B think − H2O one-pass ΔI +6.55 [−4.59, +15.65], ΔC +4.35 [−1.29, +13.35], ΔCapability +5.45
[−2.03, +14.17] (superiority over H2O unresolved). H2O one-pass matches its published public card on total (205/231),
tier counts (48/71/86) and the card's identity probes; probability-level parity is not established.

Caveats: one sampling seed; only the last 400 characters of each trace are kept; the confidence gate in
`gate_rh.py` uses floored confidences, so it never routes yes/no items (fit a real gate on unfloored, independent
dev data).

## Reproduce

Paid: `modal run` needs `--max-usd <cap>` within an owner-approved budget (`keel/runtime/budget.py`).
```bash
cd direction/experiments/reasoning_headroom
modal run rh_modal.py --models "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a,h2oai/h2o-lightning-4b@193ad740925b176a3b70a5a13a7cff2f2fadd01e" \
  --items-path public231.jsonl --out-dir full          # public231.jsonl = the pinned items + a _tier field
# from the repo root:
SSDM_DATA=~/ssdm/data python -X utf8 direction/experiments/reasoning_headroom/score_rh.py <out-dir>
SSDM_DATA=~/ssdm/data python -X utf8 direction/experiments/reasoning_headroom/gate_rh.py Qwen3.5-4B h2o-lightning-4b
```
`reports/` holds the jevkit reports (aggregates) committed here. Per-item outputs (`raw/`: log-probs, token counts,
trace tails) and `public231.jsonl` (the public item text) stay out of git, in
`$SSDM_PRIVATE/experiments/reasoning_headroom/` (default `~/ssdm/private`). `gate_rh.py` reads `raw/` from there;
both scripts import from the repo root.
