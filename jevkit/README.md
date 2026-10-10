# jevkit: a JevBench-compatible local evaluation kit

Scores any decision system the way the JevBench v1.6.1 board does: Intelligence, Calibration, Speed, Cost,
Capability, and Composite A/B/C with bootstrap intervals. It also runs our own decision backends over JevBench items
through one `/v1/systemone` path. Everything runs on CPU, and the tests use small fixtures with no network.

| module | what it does |
|---|---|
| `spec.py` | the v1.6.1 constants (tier/type weights, G_med, gates, options, caps, B / seed) |
| `items.py` | item record, JSONL I/O, the sha256-pinned public-item loader (`fetch`, `verify_public`, `load_public`), eval-only fingerprints and doc-overlap-guard rows |
| `answers.py` | answer validity (exact label set; renormalise within 2 %, else invalid) and answer-to-distribution conversion |
| `scoring.py` | the formulas, the per-item table, `aggregate`, the paired stratified bootstrap, and `score_system` |
| `board.py` | recomputes every published v1.6.1 row from its own published sub-fields and diffs the results |
| `systemone.py` | `SystemOneAdapter` over a `decide.Decider`, a Decision Index kit engine, or a request callable; Decision Index row ↔ item conversion; kit `results.jsonl` → predictions |
| `perturb.py` | 8 deterministic perturbation operators and the flip-rate / probability-shift battery |
| `guarantees.py` | (a) reorder/relabel invariance, (b) split conformal sets plus Learn-then-Test abstention on two DISJOINT calibration halves, (c) injection-robust sets from trusted-only logit intervals |
| `meter.py` | serial latency (p50/p95, raw and adjusted), tokens per decision, and $/1k under a declared `PriceSource` |
| `cli.py` | `python -m jevkit score / perturb / guarantees / board / items / fetch` |

## Reproduction of the published board

`python -m jevkit board --api api_v161.json` checks the published artifact (sha256 `5cd8c133…c83c1`, the bytes
the methodology page cites). The receipt is in `receipts/2026-10-09-jevkit-board/`. Every scored quantity
reproduces exactly at a tolerance of 1e-9 (max |diff| 1.4e-14):

| check | rows exact |
|---|---|
| Composite A / B / C from the four axes | 137/137 each (3 rows have a null axis, and both sides are null) |
| Capability = mean(I, C) | 140/140 |
| per split × type CC from the per-tier CCs (tier-weighted, clipped at 0) | 840/840 |
| I_open, I_sealed (types 1/3 each) | 140/140 |
| base, gap, excess, penalty, Intelligence | 140/140 |
| Calibration parts (choice / noul / score) and Calibration | 139 / 139 / 138 parts, 140 axes |
| Speed from adjusted p50/p95; adjusted latencies from raw (×2 + 0.15 s) | 140/140; 130/130 |
| Cost from USD per 1k | 137/137 |

**Not reproduced: the sort views.** These are not scored. With the same gates, the gated I/S/K views match on
135/137 rows. deck31b and deck-4b-v1-0 are off by up to 0.67. "Intelligence only" equals the published
Intelligence on only 23/140 rows, so the site appears to compute its views from an older Intelligence basis.
`board.py` still reports the views, but they do not affect the exit code.

### Rules the published numbers pin down (each one tested)

- The **overfit penalty is multiplicative**, as in METHOD-v1.5 §4: `I = clip(base · max(0, 1 − max(0, excess − 8)/100))`.
  The page wording "lose 1 point per excess point" would subtract instead. Four published rows have excess > 8
  (hopper, manchego-v21-v15, raw-qwen3-4b-instruct-2507, open-jev-zefan-2b). The multiplicative form reproduces all
  of them exactly. Subtraction misses by up to 1.84 I points.
- Each split × type CC is **clipped at 0**. Per-tier CCs are not clipped. Tier weights are renormalised over the
  tiers present.
- Each **calibration term is clipped at 0**, and that includes each half of the score part (raw-qwen3-0.6b 32.82,
  classone-gemma4-e2b 36.83). A supported type with no scoreable distribution contributes **0** (jade: all score
  answers are invalid, which gives 50.56).
- **Choice ECE is computed on hard-tier items only.** For every row, `n_ece_hard` equals the hard-tier choice count
  (320, or 307 after the 13 items excluded from C). When a local item set has no hard tier, jevkit falls back to all
  tiers and records that it did.
- An **invalid score answer gets nMAE = 1** (the maximum). jade's per-tier score CCs of about −160 to −355 are
  consistent with this.
- **Speed** is S = 80 − 10·log10(p50) − 10·log10(p95), with p in seconds (the adjusted values for self-hosted
  rows). That is the same as the mean of f(p) = 100 − 20·log10(p / 0.1) over p50 and p95. The self-hosted adjustment
  is p' = 2p + 0.15 s; API rows are unadjusted.
- **Cost is clipped to [0, 100].** verdict-small's raw value of 101.87 is published as 100. Speed is clipped the
  same way. No published row exercises the Speed clip.
- One row is **provider-confidence-only** (fastino): each part is 100(1 − 2·top-label ECE). `board.py` handles it.
  The item scorer expects distributions.

## What a local score approximates

1. **The sealed split.** Intelligence is 0.5·I_open + 0.5·I_sealed over 1,200 unpublished items, plus the gap
   penalty. A local run is open-only, so no penalty applies. The report says this in `intelligence.basis` and
   `approximations`. Our own held-out items with `split: private` act as the sealed side.
   **The pinned public files are not the board's open set.** They are the v1.2 public items (231). v1.6.1's open
   draw is P = 300 items (150 choice / 75 noul / 75 score), and it is not in the jevbench repo at the pinned commit.
   Local Intelligence on the 231 items therefore measures the same constructs on a different draw. It does not
   reproduce a published `open|*` cell.
2. **Latency.** The meter measures in-process wall time and has no evaluator network hop. The board's
   ×2 + 0.15 s self-hosted adjustment is its own assumption, and the board labels it as one. The board times about
   124 open standard + judge items. The public files have no judge tier, so a public-only run times the 72 standard
   items.
3. **Price for novel models.** `--price` is required and recorded: a list tariff in effect ≥ 30 days, a
   base-model reference (the board's floor), or a size-class estimate. For a model with no tariff, the number is
   only as good as the declared class.
4. **Bootstrap intervals** use the published B = 1000 and seed 16, but they cover the scored items only. The
   board's per-row intervals need sealed item-level data and cannot be derived.
5. **Perturbation operators.** MD1 disclosed its families but not its 8 operators, so ours are documented in
   `perturb.py`. state_paraphrase applies only where the rewrite table matches (104 of 231 public items).

## Usage

```bash
python -m jevkit fetch                                    # pinned public items -> ssdm-data/jevbench/<commit>/ (network)
python -m jevkit items --fingerprints out.json            # offline sha256 check + eval-only fingerprint list
python -m jevkit score --public --pred preds.jsonl --price "kind=list_tariff,in=0.05,out=10,source=<url>,as_of=2026-10-05,listed_since=2026-08-01" --out report.json
python -m jevkit perturb make --public --out pert.jsonl --seeds 0
python -m jevkit perturb score --public --perturbed pert.jsonl --pred base.jsonl --pred-perturbed pp.jsonl
python -m jevkit guarantees conformal --public --pred preds.jsonl --alpha 0.1 --epsilon 0.05 --delta 0.05
#   items -> test / calibration (--cal-frac); calibration -> FIT half (q_hat) / disjoint LTT half (--fit-frac); see below
python -m jevkit board --api api_v161.json
```

Predictions are JSONL, one line per item: `{"id", "answer": <SystemOne answer>, "status": "ok", "latency_s",
"usage": {"input_tokens", "output_tokens"}}`. `SystemOneAdapter.answer_item` writes exactly this format. Raw
`{answers: {item_id: answer}}` lines are also accepted.

## Conformal + LTT guarantee

`guarantees conformal` splits the items deterministically (sha256 of seed and id, never of an answer) into a test
set and a calibration set. It then splits the calibration decisions again, the same way, into two **disjoint**
halves:

- the **FIT half** fits the conformal threshold q_hat;
- the **LTT half** runs Learn-then-Test, choosing the act-or-abstain rule (act iff |C| = 1 and p_max ≥ λ).

The report records all three id sets by sha256, and `ltt_select` refuses LTT data that overlaps the fit data.

The guarantee has two parts:

1. **Coverage:** P(y ∈ C(x)) ≥ 1 − α, marginally, with q_hat taken from the FIT half.
2. **Acted-on error:** with probability ≥ 1 − δ over the LTT half, the error rate on acted decisions is ≤ ε.

The second part holds because q_hat and the act rule are fixed before the binomial test sees the LTT half. That half
is i.i.d. and independent of q_hat, so the acted decisions are i.i.d. given their count. Reusing the fit data would
void this bound. Both parts need exchangeability with deployment data; neither survives distribution shift.

## Public items are EVAL-ONLY (enforced)

The pinned items are github.com/fstandhartinger/jevbench @ `b6b8fff7e345b98c060ad26c13308860ddc67004` (MIT):
original 72 (standard tier), easy 48, and hard 111. Their sha256 values match the upstream `datasets/manifest.json`.
They live outside git under `ssdm-data/jevbench/<commit>/`. Never train, tune, calibrate or select on them.

The training-data gates enforce this through `train/jevbench_guard.py`, which keeps a **separate JevBench index,
fingerprint and pin**. The Index eval index, its pin, and every pinned `doc_clean` spec stay byte-identical.

- **What gets indexed:** every document unit of the 231 items (one per top-level state key, plus all instructions
  and all criteria / option descriptions), built with `doc_overlap_guard.build_index` under its own RULE.
- **Doc-overlap gate.** `train.doc_overlap_guard.scan` (used by `train_set.draw`, `run_cell`, `heldout_gate`,
  `clef_teacher.select` and the CLI) runs the JevBench index beside the Index eval index in the same streaming pass.
  - It applies the 13-gram R / E / U rule, with this draw's boilerplate and template exclusions.
  - It also fails a row whose normalised segment exactly equals a JevBench item segment of at least 6 words. This
    catches short items the 13-grams cannot reach. 3 of the 231 states are shorter; the builder fingerprint covers
    them.
  - **Templates are not hits.** An instruction or option segment shared by two or more items is a rubric frame or a
    tool-catalog description, so it is neither indexed for the exact gate nor fingerprinted. The first rescan showed
    why: 1,049 glaive_fc rows matched generic tool descriptions such as "send an email to a recipient". State text
    is never a template.
  - Every JevBench fail counts toward the scan's `n_fail`. A scan with no JevBench fail reports exactly what the
    Index-only scan did, plus a `jevbench` block.
  - `clean_spec` records the JevBench index / fingerprint sha, and `check_pinned` refuses a spec made against a
    different one.
- **Builder fingerprint.** `data.build.build_sample` runs the JevBench `data.dedup.Fingerprint` (FP v3: exact,
  ngram13, minhash, edit, contain) beside the Index fingerprint, as extra matchers named `jevbench_*`.
  - An item-tier hit drops the record.
  - Context-tier hits are logged.
- **Pin and shipping.** The pin is `data/manifests/doc_overlap/jevbench/pin.json`: doc index sha256 `0fe994c4…934b`,
  fingerprint arrays sha256 `211da3f1…c55c`, and exact-segment set sha256 `f9b94122…049d`.
  - The arrays are hashes only, about 3.3 MB, and are committed next to the pin. Every job box gets them with the
    code archive, with no context.data entry.
  - Load fails closed on any drift.
  - `python -X utf8 -m train.jevbench_guard verify` rebuilds from the pinned files and reproduces both shas.
- The id + sha256 list (no text) is `jevkit/data/jevbench_public_fingerprints.json`.

## GPU parity run (next; not run here)

Compare jevkit against the board on at least 3 open-weights systems that are on the board: H2O-Lightning-4B v1.1,
decider-12b v2, and stock gemma-4-12B (raw base).

**Needed first:** the v1.6.1 open set P300, as a file plus its sha256 (`v16.a2_supplement.set_manifest_sha256.P` =
`6b2321f8…16ce5ba`). Only that set gives cell-level parity against the published `open|choice/noul/score` tier CCs and
I_open. Without it, the 231 v1.2 public items can only test whether the three systems come out in the published order
(their published open/sealed values bracket the comparison).

For each system:

1. Pin the weights by HF revision and tokenizer sha.
2. Serve it through `SystemOneAdapter`: either `decide.Decider` over `decide.backends.hf_readout` with the system's
   own prompt template, or the system's own `/v1/systemone` server through the `request` kind.
3. Run the item set serially, one request at a time, on one H100 (the board measured on H100 / RTX 6000).
4. Score with `jevkit score --items P300.jsonl` and the row's published price basis (`cost.basis`).

Pass criteria per system:

- |I_open − published| within the bootstrap 95 % half-width.
- Every calibration part within its half-width.
- p50 within 20 % after the ×2 + 0.15 adjustment.

A miss splits into three causes: prompt template, readout, and decoding differences. Each one needs its own diff.
