---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-classification
tags:
- decision-making
- calibration
- jevbench
---

# H2O-Lightning-4B

H2O-Lightning-4B is a 4B-parameter decision model from H2O.ai, built on
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It answers typed decision questions about a record (a
document, a ticket, a policy, a conversation) and returns a probability for every option:

- **choice**: picks one of a set of named options;
- **yes/no** (`noul`): gives the probability that a statement is true;
- **score**: gives an ordinal level on a stated scale, with its probability distribution.

It runs on **unmodified vLLM 0.30.0** with a small standard-library shim in front (`h2o_lightning_shim.py`). Each
decision is one forward pass and one output token, so the cost is input tokens only.

## Measured (JevBench's public set)

All figures come from **JevBench's own CLI**, with `--adapter typesafe`, run on the 231 public items
(`datasets/public/easy.jsonl`, `original.jsonl` and `hard.jsonl`; dataset hash `dc3995d8…`) with the commands below,
on one H100 SXM (80 GB) at JevBench commit `bb05a33`.

| | H100 SXM 80 GB |
|---|---|
| correct | 205 / 231 (88.7 %) |
| by tier: easy / standard / hard | 48/48 · 71/72 · 86/111 |
| answered and valid | 231 / 231 |
| mean input tokens per decision | 654 |
| latency p50 / p95, standard tier, serial | 0.032 s / 0.036 s |
| latency p50 / p95, all 231 items, serial | 0.033 s / 0.088 s |

Latency depends on the GPU.

**Identity check for an evaluator:** a correct setup reproduces 205/231 with the tier split above. Six hard items
sit on near-ties (top two within 0.03) that GPU numerics can flip. Four are wrong here:
`hard-opus-a-temporal_numeric-07`, `hard-opus-a-temporal_numeric-12`, `hard-opus-c-temporal_numeric-04` and
`hard-sol-b-temporal_numeric-03`. Two are right: `hard-opus-b-multi_hop-03` and `hard-sol-c-judge_hard-13`.

## Run it

Hardware measured: one H100 SXM (80 GB); the weights are 8.4 GB, and smaller cards have not been measured. Install
vLLM 0.30.0 in a fresh virtual environment (for example `python3 -m venv .venv && . .venv/bin/activate`):

```bash
pip install "vllm==0.30.0+cu129" "torchcodec==0.16.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
```

The shim needs Python 3.8 or later and nothing else.

```bash
hf download h2oai/h2o-lightning-4b --revision v1.1 --local-dir h2o-lightning-4b
```

**1. Serve the model.** Keep `config.json` as shipped: its `"head_dtype": "float32"` makes vLLM compute the logits in
fp32. Inputs over the context limit get HTTP 422.

```bash
vllm serve ./h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90
```

**2. Start the shim** on the same machine:

```bash
python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

`bash h2o-lightning-4b/serve.sh` runs steps 1 and 2 together. Before answering, the
shim checks that vLLM serves `h2oai/h2o-lightning-4b` and that every label is one token at the answer slot; if not, it
answers 503, so a misconfigured server stops a run instead of scoring it.

**3. Warm up** until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"true": "yes", "false": "no"}}}}'
```

**4. Run JevBench's harness**:

```bash
cd <JEVBENCH> && python3 -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8741 --key-env '' --model h2oai/h2o-lightning-4b \
  --cost-basis self_hosted_gpu --reserve-usd 0 --results <OUT>/results.jsonl --raw-dir <OUT>/raw \
  --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json --run-label h2o-lightning-4b --delay-s 0
```

**The API.** Send requests to `POST /v1/systemone`. One request can carry several questions about the same state:

```json
{"state": "Ticket 4471: since 09:10 the checkout page returns HTTP 500 for every customer; no orders have gone through in 40 minutes. Severity guide: P1 = a revenue path fully down; P2 = degraded but working; P3 = cosmetic.",
 "questions": {
   "priority":  {"type": "choice", "instructions": "Which priority does the severity guide assign?",
                 "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}},
   "all_users": {"type": "noul", "instructions": "The problem affects every customer.",
                 "criteria": {"true": "Affects everyone", "false": "Affects only some"}},
   "urgency":   {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Low", "Medium", "High"]}}}
```

The response (rounded here):

```json
{"model": "h2oai/h2o-lightning-4b",
 "answers": {"priority": {"type": "choice", "choice": "p1",
                          "probabilities": {"p1": 0.945, "p2": 0.039, "p3": 0.016}, "confidence": 0.917},
             "all_users": {"type": "noul", "noul": 0.927},
             "urgency": {"type": "score", "score": 1.791, "legend": {"0": "Low", "1": "Medium", "2": "High"},
                         "probabilities": {"0": 0.032, "1": 0.145, "2": 0.823}, "confidence": 0.735}},
 "probability_source": "native", "effort_used": {"tier": "fast", "passes_per_decision": 1},
 "usage": {"input_tokens": 231, "output_tokens": 0, "latency_ms": 124.8}}
```

`usage.input_tokens` counts the prompt head that the questions share once.

**Status codes.** An input the shim cannot take (over the context limit, more than 255 options, an unknown question
type, a malformed question) gets HTTP 422, which the runner scores as one wrong answer; an empty request gets 400.
vLLM's 401, 403 and 429 pass through; any other vLLM failure is a 502, which counts toward the runner's
three-consecutive-failures stop.

**Tests** (no GPU): `python3 h2o-lightning-4b/test_shim.py`.

## Limitations

- Evaluated mostly in English. Decides best with up to about 16 options; up to 255 are accepted.
- The probabilities are calibrated for decision questions of this kind, not for open-ended text.
- Not a chat model: it answers one decision per question and does not generate explanations.

## Disclosures

- No JevBench data, public or otherwise, was used for training.
- No rules keyed to any benchmark item's wording, id or answer; no network calls at serving time.
- Price basis for an evaluator: Qwen3.5-4B at bf16, one output token per decision, 654 mean input tokens on the public
  set.

## License

Apache-2.0. Built on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Alibaba Cloud),
Apache-2.0; served with unmodified [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
