---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- decision-making
- calibration
- jevbench
- multimodal
---

# H2O-Lightning-4B

H2O-Lightning-4B is a 4B-parameter decision model from H2O.ai, built on
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It answers typed decision questions about a record (a
document, a ticket, a policy, a conversation) and about images that come with the record (photos,
screenshots, scanned documents, charts). It returns a probability for every option:

- **choice**: picks one of a set of named options;
- **yes/no** (`noul`): gives the probability that a statement is true;
- **score**: gives an ordinal level on a stated scale, with its probability distribution.

It runs on **unmodified vLLM 0.30.0** with a small standard-library shim in front (`h2o_lightning_shim.py`). Each
decision is one forward pass and one output token, so the cost is input tokens only.

## #1 on JevBench (October 7, 2026)

On **JevBench v1.6.1**, H2O-Lightning-4B v1.1 is **#1 on the official Composite Score** of open-weight systems, and it
**outscores Jev itself: 72.5 to 71.5**. It is also ahead of every open 12B, 26B and 31B model on the board.
([leaderboard](https://benchmarkheaven.com/jev-models), read 2026-10-07)

![JevBench v1.6.1 Composite Score, open-weights board, 2026-10-07: H2O-Lightning-4B v1.1 at #1](assets/jevbench_composite_2026-10-07.png)

| | Composite | Intelligence | Calibration | Speed | Cost | $ per 1,000 decisions |
|---|---|---|---|---|---|---|
| **H2O-Lightning-4B v1.1** (4B) | **72.5** | 60.0 | 90.0 | 92.6 | 60.3 | $0.021 (est.) |
| Jev 1.13.0 (reference, hosted) | 71.5 | 63.6 | 90.6 | 91.5 | 54.7 | $0.032 |
| Quyet-1.0-Large (31B) | 71.4 | 73.4 | 90.0 | 86.9 | 50.5 | $0.045 (est.) |

The composite is the equal-weight harmonic mean of the four axes. The 31B models lead on raw intelligence, while this
4B matches their calibration at a fraction of the cost and latency.

| Four axes, against a 31B | Capability against cost |
|---|---|
| ![The four JevBench axes, H2O-Lightning-4B v1.1 (solid) against Quyet-1.0-Large 31B (dashed)](assets/jevbench_four_axes_2026-10-07.png) | ![Capability against cost per 1,000 decisions on the open-weights board, H2O-Lightning-4B highlighted](assets/jevbench_capability_vs_cost_2026-10-07.png) |

### Accuracy, latency and JevBench scores, side by side

Microsoft's [Microsoft-Decision-1 announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
compares accuracy, latency and calibration side by side. The chart below keeps Microsoft's accuracy numbers unchanged,
corrects the latency and puts it on one H100 basis, and shows both JevBench v1.6.1 scores. Each model has composite above capability; the animation sorts by composite, capability, latency and accuracy. Capability averages Intelligence and Calibration, excluding speed and cost.

![Accuracy (Microsoft's 36-benchmark comparison), median latency on an H100 basis and JevBench v1.6.1 composite and capability for H2O-Lightning-4B v1.1, Jev 1.13.0, Quyet-1.0-Large, deck-31B, Surogate Rune 26B-A4B, OpenAI Decisions, Microsoft-Decision-1 and Strands-Decider 2B, ordered by composite](assets/accuracy_latency_composite_2026-10-10.gif)

![Vertical comparison of JevBench composite and capability scores, with median decision latency](assets/jevbench_scores_latency_2026-10-10.png)

[Vector SVG](assets/jevbench_scores_latency_2026-10-10.svg) · [PDF](assets/jevbench_scores_latency_2026-10-10.pdf)

*Footnote: we corrected the latency from Microsoft's page. For self-hosted models it used the leaderboard's adjusted
figure (measured time x 2 + 0.15 s, which the board itself labels "assumption, not measured"), so this model appeared
at 210 ms instead of its measured 29 ms. The chart uses the board's measured median (`latency.p50_s_raw` in the
[JevBench v1.6.1 data](https://benchmarkheaven.com/api/jevbench/latest)) on one H100 basis: Quyet-1.0-Large (116 ms)
and deck-31B (127 ms) were timed on an H100 and are unchanged; Surogate Rune was timed on an RTX PRO 6000, which has
about half an H100's memory bandwidth and compute, so its 116 ms is halved to 58 ms (the most favourable conversion for
it); this model was timed by the board on an RTX 5090 at 29 ms, and our own serial H100 runs give 28-32 ms, so 29 ms is
kept. Microsoft-Decision-1 now uses JevBench’s measured API median **460 ms** (replacing Microsoft’s announcement’s 85 ms), with composite **69.1 / API #6**. OpenAI Decisions uses **298 ms** (about 0.30 s), with composite **62.5 / API #8**. Both and Jev include network time and are not rescaled. The accuracy numbers are
Microsoft's own, on benchmarks they chose, so they are likely to favour their model; the JevBench composite is scored on
held-out items that no submitter, us included, has seen. Composite and ranks: [open-weights](https://benchmarkheaven.com/jev-models)
and [API](https://benchmarkheaven.com/jev-models/api) boards, read 2026-10-10.*

**How we got there:**
- **A held-out lockbox:** independently written evaluation items that are never trained on, used to compare
  candidate models before a release.
- **Rules written down first:** ship / no-ship criteria are fixed before results are seen.
- **No benchmark test items in training:** all training data is deduplicated against every evaluation set, exact
  and near-duplicate.
- **Gap hunting:** comparing against other systems by topic and question type, running adversarial sets, and filling
  the gaps with new, verified data.
- **Data quality first:** real, licence-clean data, with labels from human annotation or agreement among several
  independent strong models.
- **Calibration as a first-class goal:** one forward pass, one output token, and probabilities that mean what they say.

## See it in action

Three live runs on one NVIDIA H100 (vLLM 0.30.0). The operations inbox and the web agent are shown in **real time**:
nothing is sped up or slowed down. DOOM, where the game waits for each answer, is shown on the model's **decision
clock**, which advances only by the model's own time per call. Companies, people and amounts are fictional.

<video src="https://huggingface.co/h2oai/h2o-lightning-4b/resolve/main/assets/demos/demos_reel.mp4" poster="https://huggingface.co/h2oai/h2o-lightning-4b/resolve/main/assets/demos/demos_reel_poster.jpg" controls autoplay muted loop playsinline width="100%"></video>

One panel, three demos in turn: **operations inbox** (1,000 claims against a written policy, 32 at a time on one GPU,
all decided in 63 s), **web agent** (8 real browsers, 192 vendors in 145 s, every decision correct against answers
written beforehand) and **DOOM** (a Freedoom level played to the exit, four decisions per call). Each demo is followed by what was checked afterwards. The clips are excerpts; the full runs follow.

**Full-length runs (uncut), on YouTube.** The same three demos, each run shown complete, with chapters. In DOOM, E1M1 is one
of the two maps the thin controller (aiming, routing) was tuned on; the model itself was never trained on Doom.

[![H2O-Lightning-4B: full-length demo runs on YouTube](https://img.youtube.com/vi/2Qp04Wu0A14/maxresdefault.jpg)](https://youtu.be/2Qp04Wu0A14)

## Measured

Measured on one **NVIDIA RTX PRO 4500 Blackwell (32 GB)** with `serve.sh` as shipped (vLLM 0.30.0; text temperature
0.8, image temperature 0.65, images up to 1,048,576 pixels).

| JevBench | public items | result |
|---|---|---|
| JevBench (text) | the 231 public items, JevBench's own CLI | **205 / 231 (88.7 %)**: easy 48/48 · standard 71/72 · hard 86/111 |
| ImageJevBench | the 8 published examples (the only public items released with images) | **8 / 8** |
| image validation sets | 1,877 questions over 16 public licence-clean sets (table below) | **87.4 %** |

Details follow.

### Text: JevBench's public set

From **JevBench's own CLI**, with `--adapter typesafe`, on the 231 public items (`datasets/public/easy.jsonl`,
`original.jsonl` and `hard.jsonl`; dataset hash `dc3995d8…`) at JevBench commit `bb05a33`, with the commands below.

The measured H100 was an **80 GB SXM5**, with a **700 W configured power limit** (board692-2G520-0200-000). These are not H100 PCIe measurements; the timing values below are unchanged.

| | RTX PRO 4500 Blackwell 32 GB | 1x H100 80 GB SXM5 |
|---|---|---|
| correct | 205 / 231 (88.7 %) | 205 / 231 (88.7 %) |
| by tier: easy / standard / hard | 48/48 · 71/72 · 86/111 | 48/48 · 71/72 · 86/111 |
| answered and valid | 231 / 231 | 231 / 231 |
| yes/no answers with P(yes) strictly between 0.2 and 0.8 | 0 of 74 | 0 of 74 |
| mean input tokens per decision | 654 | 654 |
| client latency p50 / p95, standard 72 items, serial | 29.6 ms / 29.8 ms | 28.8 ms / 29.3 ms |
| client latency p50 / p95, all 231 items, serial | 30.2 ms / 225.6 ms | 29.8 ms / 69.9 ms |
| server-reported latency p50 / p95, all 231 items, serial | 30.0 ms / 225.3 ms | 29.5 ms / 69.5 ms |
| raw LoRA: correct, separate run | 205 / 231 (88.7 %) | 204 / 231 (88.3 %) |
| raw LoRA: client latency p50 / p95, standard 72 items, serial | 57.3 ms / 58.0 ms | 69.2 ms / 72.1 ms |
| raw LoRA: client latency p50 / p95, all 231 items, serial | 58.0 ms / 272.6 ms | 70.7 ms / 148.1 ms |
| raw LoRA: server-reported latency p50 / p95, all 231 items, serial | 57.9 ms / 272.4 ms | 70.4 ms / 147.7 ms |

**Latency rechecked October 10, 2026, using the exact v1.2.3 release**
(`acaf0d4ea251e54de928c75ef4352670d33192d3`). Each GPU processes the same
**231 distinct public requests once**, one request at a time, over a persistent
localhost HTTP connection. **Server startup, compilation, and 10 separate warmup
requests are excluded from both client and server timing statistics.** No
cached repeat passes are included; the measured pass had **zero prefix-cache
hits** on both GPUs. Client latency covers the complete shim request and
response; server latency is the shim's reported processing time. Both use
BF16, 40,960-token context, default CUDA graphs and prefix caching, with matched
vLLM 0.30.0 / Torch 2.13.0 cu129, Transformers 5.19.0 and tokenizers 0.23.3.

**Separate raw LoRA measurement:** the additional rows use the released adapter from
that same pinned v1.2.3 revision on `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with its standard alpha/rank scaling
(128/64), deployment multiplier 1.0, BF16 and `max_num_seqs=64`.
The preceding rows measure the merged checkpoint with its shipped server defaults.
These are two deployment configurations and separate weight representations.
The LoRA run uses the same pinned runtime, calibration, 40,960-token context,
CUDA graphs, prefix caching, 231 distinct serial requests, and ten separate
warmups excluded from both timing statistics. Native loading and activation of
all 496 adapter tensors across 248 decoder projections were checked; separate
base→adapter→base controls verified an adapter effect and restored base routing.
The H100 LoRA run scored 204/231, compared with 205/231 on RTX, using the same pinned adapter and calibration.

For the merged-model rows, the wider p95 is across different prompt lengths, not parallel traffic: the
standard prompts contain 110–163 input tokens, while the hardest include up to
3,952. That longest request took 316.5 ms on RTX and 85.9 ms on H100. These are single-request timings, not throughput or concurrent-load
measurements; latency depends on the GPU and input length.
JevBench's own measurement of v1.1: median 29 ms (board data, latency.p50_s_raw).

**Identity check for the merged-model evaluator:** a correct setup reproduces 205/231 with the tier split above. Five hard items
sit on near-ties (top two within 0.03) that GPU numerics can flip. Four are wrong here:
`hard-opus-a-temporal_numeric-07`, `hard-opus-a-temporal_numeric-12`, `hard-opus-c-temporal_numeric-04` and
`hard-sol-b-temporal_numeric-03`. One is right: `hard-sol-c-judge_hard-13`.

### Images

**(a) The published ImageJevBench examples: 8 / 8.** ImageJevBench's own harness is not public, and only 8 of its items
are published with their images (on the benchmark's page: two everyday photos, four screenshots with five labelled
click markers, one geometry diagram, one financial table). We read all 8 through this API: 8 correct. These 8 items
are not a benchmark score.

**(b) Public validation sets with human or source ground truth.** Multiple-choice questions we built from each
dataset's own annotations (human labels, boxes, receipt fields, table cells, a chart's own data); none come from a
model. None of these images or questions was used for training. Read through this API, one image per request, as a
data URI in `state`.

| set (licence) | what is asked | n | correct | ECE |
|---|---|---:|---:|---:|
| Open Images V7 validation (CC BY 4.0 annotations) | is there a ‹class› in the photo (human-verified labels) | 150 | 88.0 % | 0.034 |
| Open Images V7 validation | how many ‹class› in the photo (human boxes, 2-8) | 150 | 63.3 % | 0.121 |
| VizWiz validation (CC BY 4.0) | can the question be answered from this photo | 120 | 84.2 % | 0.057 |
| VizWiz validation | the answer (6+ of 10 annotators) | 120 | 100.0 % | 0.015 |
| TextVQA validation (CC BY 4.0) | text in the photo (6+ of 10 annotators) | 120 | 99.2 % | 0.024 |
| CORD v2 test + validation (CC BY 4.0) | receipt total | 95 | 99.0 % | 0.014 |
| CORD v2 test + validation | one item's price | 29 | 96.5 % | 0.043 |
| CORD v2 test + validation | number of item lines | 113 | 85.0 % | 0.050 |
| TAT-QA dev (CC BY 4.0) | financial-report table questions (the table shown as an image) | 120 | 95.8 % | 0.037 |
| DocLayNet test (CDLA-Permissive-1.0) | which section heading is on the page | 120 | 100.0 % | 0.011 |
| DocLayNet test | how many tables are on the page (0-3) | 120 | 75.8 % | 0.057 |
| DocLayNet test | does the page contain a picture | 120 | 90.0 % | 0.039 |
| Our World in Data charts (CC BY 4.0) | which country is highest or lowest (the chart's own data) | 100 | 98.0 % | 0.058 |
| Our World in Data charts | did a value rise between two years | 100 | 95.0 % | 0.069 |
| Our World in Data charts | roughly what is a value | 100 | 98.0 % | 0.066 |
| Rico app screens (CC BY 4.0) | which of 5 marked elements reaches a goal (human widget captions) | 200 | 65.0 % | 0.094 |
| **all** | | **1,877** | **87.4 %** | **0.017** |

ECE: 10-bin expected calibration error of the top answer's probability.

## Routing thresholds

When a decision is uncertain you may want to hand it to a stronger model or a person. This table shows that trade-off:
at a confidence threshold, how many questions get **escalated** (top probability below the threshold) and how many
**slip through confidently wrong** (top probability at or above it, but wrong), as a share of all questions.

Measured on 3,474 held-out questions from two public third-party decision sets, with no overlap with our training
data; text temperatures as shipped (choice 0.75, yes/no 0.8, score 0.65). The thresholds were fixed before the
analysis was run.

| question type (n) | threshold | escalated | accuracy on the rest | confidently wrong |
|---|---|---|---|---|
| choice (1,854) | 0.8 | 34.1 % | 95.5 % | 3.0 % |
| choice | 0.9 | 47.5 % | 98.2 % | 1.0 % |
| yes/no (1,367) | 0.8 | 40.8 % | 96.0 % | 2.3 % |
| yes/no | 0.9 | 59.5 % | 99.1 % | 0.4 % |
| score (253) | 0.8 | 46.2 % | 96.3 % | 2.0 % |
| score | 0.9 | 59.7 % | 98.0 % | 0.8 % |
| all (3,474) | 0.8 | 37.7 % | 95.8 % | 2.6 % |
| all (3,474) | 0.9 | 53.1 % | 98.5 % | 0.7 % |

Without a threshold, accuracy is 82.7 % (choice), 84.7 % (yes/no) and 80.2 % (score).

These numbers use [routing mode](#optional-routing-mode) (the yes/no floor off), for deployments behind a fallback. As
shipped (the evaluated configuration), yes/no answers are reported at no less than 0.801, so none would be escalated below
0.8.

## Run it

Hardware measured: one RTX PRO 4500 Blackwell (32 GB); the weights are 9.1 GB. Install vLLM 0.30.0 in a fresh virtual environment
(for example `python3 -m venv .venv && . .venv/bin/activate`):

```bash
pip install "vllm==0.30.0+cu129" "torchcodec==0.16.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
```

The shim needs Python 3.8 or later and nothing else.

```bash
hf download h2oai/h2o-lightning-4b --revision v1.2.3 --local-dir h2o-lightning-4b
```

**1. Serve the model.** Keep `config.json` as shipped: its `"head_dtype": "float32"` makes vLLM compute the logits in
fp32. Inputs over the context limit get HTTP 422. The last two options set the image limits (up to 4 images
per request, each scaled to at most 1.6 megapixels inside vLLM).

```bash
vllm serve ./h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1605632}'
```

If vLLM stops at startup because FlashInfer cannot compile its sampling kernels (an older system CUDA toolkit), start it
with `VLLM_USE_FLASHINFER_SAMPLER=0` in the environment: the decisions do not use the sampler.

**2. Start the shim** on the same machine:

```bash
python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

Run the shim as shipped for evaluations and benchmarks (including JevBench): `serve_config.json` is the evaluated
configuration. For production routing, see [Optional: routing mode](#optional-routing-mode).

`bash h2o-lightning-4b/serve.sh` runs steps 1 and 2 together. Before answering, the
shim checks that vLLM serves `h2oai/h2o-lightning-4b` and that every label is one token at the answer slot; if not, it
answers 503, so a misconfigured server stops a run instead of scoring it. Before the first image request it also checks
that vLLM's chat template renders the same prompt as the text path.

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
                          "probabilities": {"p1": 0.956, "p2": 0.032, "p3": 0.013}, "confidence": 0.934},
             "all_users": {"type": "noul", "noul": 0.928},
             "urgency": {"type": "score", "score": 1.864, "legend": {"0": "Low", "1": "Medium", "2": "High"},
                         "probabilities": {"0": 0.016, "1": 0.104, "2": 0.880}, "confidence": 0.820}},
 "probability_source": "native", "effort_used": {"tier": "fast", "passes_per_decision": 1},
 "usage": {"input_tokens": 231, "output_tokens": 0, "latency_ms": 62.0}}
```

`usage.input_tokens` counts the prompt head that the questions share once.

**Images.** Put each image in the request as a `data:image/...;base64,...` URI. It can go anywhere inside `state` (as a
string, an object value or a list item), or in a top-level `images` list next to `state` and `questions`. The `images`
list also takes bare base64 strings and `http(s)` URLs (vLLM fetches a URL itself, so the server needs network access
for those; a URL it cannot fetch gets HTTP 422). Chat-style parts (`{"type": "image_url", "image_url": {"url": ...}}`) and multipart/form-data (the JSON in a
`request` field, each image as a file part) work too.
- Each image taken out of `state` leaves `[image N]` in its place, so the record can refer to it. Images in the
  top-level list are numbered first.
- Up to 4 images per request and 20 MB per image.
- Each image is scaled to at most 1,048,576 pixels, and image answers use their own temperature, 0.65 (`serve_config.json`,
  `"image"`); text answers use per-type temperatures (`serve_config.json`).
- A request without an image takes exactly the text path above.

This example uses `example_receipt.png` from this repository:

```python
import base64, json, urllib.request
img = base64.b64encode(open("h2o-lightning-4b/example_receipt.png", "rb").read()).decode()
req = {"state": {"receipt": "data:image/png;base64," + img,
                 "claim": {"employee": "J. Ortiz", "category": "office supplies", "amount": "444.26"}},
       "questions": {
         "matches":  {"type": "noul", "instructions": "The receipt total equals claim.amount.",
                      "criteria": {"true": "Same amount", "false": "Different amount"}},
         "items":    {"type": "choice", "instructions": "How many item lines are on the receipt?",
                      "criteria": {"three": "3", "four": "4", "five": "5", "six": "6"}},
         "legible":  {"type": "score", "instructions": "How legible is the receipt?",
                      "criteria": ["Unreadable", "Partly readable", "Fully readable"]}}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8741/v1/systemone", json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}))
print(json.dumps(json.load(r), indent=1))
```

The response (rounded here):

```json
{"model": "h2oai/h2o-lightning-4b",
 "answers": {"matches": {"type": "noul", "noul": 0.964},
             "items": {"type": "choice", "choice": "six",
                       "probabilities": {"three": 0.006, "four": 0.012, "five": 0.024, "six": 0.958}, "confidence": 0.943},
             "legible": {"type": "score", "score": 1.926,
                         "legend": {"0": "Unreadable", "1": "Partly readable", "2": "Fully readable"},
                         "probabilities": {"0": 0.011, "1": 0.052, "2": 0.937}, "confidence": 0.906}},
 "probability_source": "native", "effort_used": {"tier": "fast", "passes_per_decision": 1},
 "usage": {"input_tokens": 585, "output_tokens": 0, "latency_ms": 141.0, "images": 1}}
```

`usage.input_tokens` includes the image tokens; `usage.images` counts the images read.

**Status codes.** An input the shim cannot take gets HTTP 422, which the runner scores as one wrong answer. That covers:
- over the context limit;
- more than 255 options;
- an unknown question type or a malformed question;
- more than 4 images, or an image that is not valid base64 image data.

An empty request gets 400. vLLM's 401, 403 and 429 pass through. Any other vLLM failure is a 502, which counts toward the
runner's three-consecutive-failures stop.

**Tests** (no GPU): `python3 h2o-lightning-4b/test_shim.py`.

## Optional: routing mode

**Not for benchmark or leaderboard runs.** Those use the shipped `serve_config.json` unchanged.

As shipped, the shim reports each yes/no answer at no less than 0.801 on its chosen side (`"noul_floor": 0.801` in
`serve_config.json`). That is the evaluated configuration, and it is why no shipped yes/no answer falls between 0.2 and
0.8. If you put the model in front of a fallback, and escalate uncertain answers to a larger model or a person, you
usually want the raw yes/no probability instead, so uncertain yes/no answers can be flagged. In that deployment only,
start the shim with the floor off:

```bash
SHIM_NOUL_FLOOR=0 python3 h2o-lightning-4b/h2o_lightning_shim.py --config h2o-lightning-4b/serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

The floor never changes which answer is chosen, and choice and score answers are unaffected. See
[Routing thresholds](#routing-thresholds) for what each threshold escalates with the floor off.

## Hybrid mode: decisions and text from one server

`adapter/` holds H2O-Lightning-4B's decision adapter as a LoRA for the base model
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Serve the base model with the adapter on vLLM, and one
server answers both kinds of request:
- **decisions** with `model: "h2oai/h2o-lightning-4b"` (the adapter), through the same shim and readout as the
  merged model;
- **free text** with `model: "Qwen/Qwen3.5-4B"` (the base model, no adapter).

On JevBench's 231 public items, the adapter scores the same as the merged model (205/231) and gives the same top answer
on 230 of 231 items. Measured on one RTX A6000: 94 ms per decision (p50, versus 60 ms for the merged model);
about 1.3 s for a 90-token reply from the base model.

**0. Get the adapter** into the same folder as the download above (shim and `serve_config.json` included):

```bash
hf download h2oai/h2o-lightning-4b --include "adapter/*" --local-dir h2o-lightning-4b && cd h2o-lightning-4b
```

**1. Serve the base model with the adapter** (vLLM 0.30.0, from the folder that contains `adapter/`):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules h2oai/h2o-lightning-4b=./adapter \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1605632}'
```

- `--max-num-seqs`: Qwen3.5's linear-attention layers need one state block per running sequence, so keep this at or
  below what fits. vLLM stops at startup with a clear message if it's too high.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is needed only where FlashInfer can't compile its sampler (an older CUDA toolkit).
  The decisions don't use the sampler.

**2. Put the shim in front for decisions.** Use the same shim and config as the merged model. The LoRA's name is the
served model name the shim expects:

```bash
python3 h2o_lightning_shim.py --config serve_config.json --vllm http://127.0.0.1:8000 --port 8741
```

Decisions then go to `POST http://127.0.0.1:8741/v1/systemone` exactly as described above.

**3. Text from the base model on the same server:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "Qwen/Qwen3.5-4B",
  "messages": [{"role": "user", "content": "Write a two-sentence status update for customers about a checkout outage that is now fixed."}],
  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false}}'
```

**Check that the adapter is active.** The two lines must differ; if they are identical, vLLM is serving the base model
under the adapter's name:

```bash
for m in Qwen/Qwen3.5-4B h2oai/h2o-lightning-4b; do curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d "{\"model\": \"$m\", \"prompt\": \"Ticket: checkout is down for every customer. Priority (P1, P2 or P3):\", \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": 5}" | python3 -c "import json,sys; print(sys.argv[1], json.load(sys.stdin)['choices'][0]['logprobs']['top_logprobs'][0])" "$m"; done
```

> **Must-rename note for converted or re-exported adapters.** vLLM loads Qwen3.5-4B as
> `Qwen3_5ForConditionalGeneration`, so the adapter's tensor names must be
> `base_model.model.model.language_model.layers.N.…`. The file shipped here already uses those names. An adapter
> saved from a text-only load of the model (`base_model.model.model.layers.N.…`) is accepted by vLLM without any
> warning and changes nothing: the output equals the base model. Rename the keys (insert `language_model.` after
> `model.model.`) and run the check above.

The adapter's own scaling is the standard `lora_alpha / r`. Use it as is, with no extra scale.

## What it was trained on

H2O-Lightning-4B was fine-tuned from Qwen3.5-4B in three stages. Every training example is a record (a document,
ticket, policy, conversation, game state...) with one typed question about it and a graded answer. The first stage is
broad, the second concentrates on reading long documents and on numbers, dates and probabilities, and the last stage
mixes everything back in with more support, everyday-language, safety and coding cases.

![Share of training examples by use case in each of the three fine-tuning stages, and overall: documents, policies and rules 37%, numbers, dates and probability 22%, games and simulations 12%, everyday language and knowledge 10%, support and operations 10%, safety and guardrails 7%, coding and tool calls 2%, finance and commerce under 1%](assets/datamix_stages.png)

| Use case | Stage 1: broad | Stage 2: focused | Stage 3: final | All stages |
|---|---:|---:|---:|---:|
| Documents, policies & rules (contracts, policies, multi-step reading, judging answers) | 34% | 49% | 36% | **37%** |
| Numbers, dates & probability | 24% | 41% | 15% | **22%** |
| Games & simulations (state tracking, planning, control) | 23% | 2% | 6% | **12%** |
| Everyday language & knowledge | 8% | 1% | 15% | **10%** |
| Support & operations (tickets, routing, workflows) | 6% | 4% | 16% | **10%** |
| Safety & guardrails | 6% | 3% | 9% | **7%** |
| Coding & tool calls | - | - | 3% | **2%** |
| Finance & commerce | - | - | 1% | **0.5%** |

Use cases are assigned per data family, so the split is approximate.

**Question types.** About half of the examples are choice questions, with yes/no and score questions making up the
rest. Yes/no and score questions get a bigger share in the final stage.

![Share of training examples by question type in each stage: overall choice 54%, yes/no 29%, score 16%](assets/qtype_stages.png)

| Question type | Stage 1 | Stage 2 | Stage 3 | All stages |
|---|---:|---:|---:|---:|
| Choice | 62% | 62% | 46% | **54%** |
| Yes / no | 23% | 29% | 34% | **29%** |
| Score | 14% | 9% | 20% | **16%** |

**Text and images.** All fine-tuning examples were text, and almost all (over 99%) were in English. The model reads
images with Qwen3.5-4B's vision encoder, which fine-tuning did not change.

## Limitations

- Evaluated mostly in English. Decides best with up to about 16 options; up to 255 are accepted.
- The probabilities are calibrated for decision questions of this kind, not for open-ended text.
- Not a chat model: it answers one decision per question and does not generate explanations.
- Images: no video input; counting many small objects and reading values off line charts are the weakest image
  skills we measured.

## Disclosures

- No JevBench or ImageJevBench data, public or otherwise, was used for training.
- No rules keyed to any benchmark item's wording, id or answer; no network calls at serving time (except fetching an
  image URL a request supplies).
- Price basis for an evaluator: Qwen3.5-4B at bf16, one output token per decision; 654 mean input tokens on the
  public text set. Image tokens are counted in `usage.input_tokens`.

## License

Apache-2.0. Built on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Alibaba Cloud),
Apache-2.0; served with unmodified [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
