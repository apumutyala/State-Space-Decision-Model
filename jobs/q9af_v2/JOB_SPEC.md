# q9af_v2: a fully attention-free Qwen3.5-9B (Q9-AF v2), all 8 attention layers -> GDN jointly, gated end to end

**Takeaway.** The full8 run (job-2xe5w) didn't fail because layers 3 and 27 can't be converted. It failed because of
how it converted them. The gate checked one cumulative quantity per layer against a fixed floor. The distillation
used 40M tokens of narrow wikitext, re-read about 4.7 times. The new layers started from a copy of a neighbouring GDN
instead of from the attention weights. v2 fixes all three in one job: it converts all 8 layers at once, initialises them
from the attention Q/K/V/gate/O, and distils longer on a broad chat + web + math mix that it never re-reads. It is
gated end to end, on held-out KL plus a stock Index proxy600 screen, never on a per-layer cosine. The result is
attention-free by construction. The open question is how much quality it keeps, and the screen measures exactly that.

Trigger: issue #140, option (b). #140 also set the re-open trigger: the RWKV7 gap to Clef-flash still above 15 pts
after the cell + debias work. That trigger has now fired: the champion is 38.67 (RWKV7 D1 + v3w, job-39gnn) and
Clef-flash is 55.78, a gap of 17.1. The spend is ≤ $40, one H100, one attempt.

## Diagnosis of full8 (job-2xe5w, receipts/2026-10-07-D-full8)

1. **The cosine gate was structurally unreachable for 8/8, and it is what rejected L3 and L27.**
   - `hidden_cos` is the *final-layer* cosine of the whole model against the teacher, so it is cumulative.
   - It fell with every accepted layer: 0.9945 → 0.9889 → 0.9797 → 0.9735 → 0.9663 → 0.9575, about 0.007 per layer.
     At that rate, 8 layers end near 0.94, below the 0.95 floor, whatever the order.
   - L3 was rejected on cosine alone (0.9487), even though its step dNLL was **negative** (−0.0051). The model got
     better at next-token prediction and the gate still refused it.
   - L27 (0.9259; dNLL +0.042, inside the 0.05 bound) is the hardest layer: it has the highest single-swap KL.
   - Both were tried last, on a model that already had 6 GDNs, with the fewest stage-2 tokens (18.7M and 20.5M,
     against 33.8M for the first layer; the cap was 60 min per layer, and tok/s falls as the student suffix grows).
2. **The −26 Index / −12.2 MMLU damage came from the distillation data and was invisible to the gate.**
   - Every token was wikitext-103: 40M tokens, re-read about 4.7×. Stage 1 used 13M; the 8 stage-2 runs used ~188M.
   - The gate and the ppl probe also used wikitext. So ppl rose only 2.8–4.0 % while full MMLU fell 12.2 pts and the
     Index fell 26.0. The model had overfitted the one distribution the gate could see.
   - The Index prompt is chatml with an empty think block plus option scoring. The run never distilled a single
     chat-formatted token.
   - The probe-subset MMLU curve collapsed at L7 (−3.5) and L19 (−7.0) while ppl moved about 1 %. A few attention
     layers carry the MCQ/retrieval function (as in Jet-Nemotron's PostNAS), and wikitext KD never exercised it.
     The KL-importance order, also measured on wikitext, could not see it either.
3. **Init.** full8 and probe9b used `clone_neighbor` (the previous GDN's weights). The attention projections were
   discarded. On 0.8B the attention-reuse init was "much better" than random (m1 pilot). `clone_neighbor` was never
   compared on 9B (`init_cmp.steps: 0`).
4. **Would task training (LoRA + head) recover the −26?** Largely, no. The measured training lifts are:
   - our cell recipe on RWKV7: **+7.6** over its best stock readout (31.07 → 38.67);
   - Clef-flash: **+17.5** over stock Q9 (38.27 → 55.78).

   Q9-H2 starts at about 12.3 (38.27 − 26.0), so a cell projects to roughly 20–30, below both the champion (38.67)
   and stock RWKV7's 24.85 once the RWKV7 readout gain is counted. The −26 *is* a pre-training artifact, but the
   artifact sits in the conversion, not in the missing task training. Stock RWKV7, a model that is attention-free from
   scratch, scores 24.85, double Q9-H2's 12.3. Attention-freedom itself does not cost 26 points; a poorly distilled
   conversion did. The fix is a better conversion, then the cell.

## Design (v2)

| Item | full8 | v2 | Why |
|---|---|---|---|
| Layers | 1 at a time, KL order, accept/rollback per layer | **all 8 at once**, no rollback | attention-free by construction; no order effects |
| Init | `clone_neighbor` | **`reuse_attn`** (q/k from per-head SVD of Wq/Wk, v/z/o from v_proj / q-gate / o_proj, decay/beta/norm from the preceding GDN, identity conv) + out-scale calibration | keeps the attention weights; 40-step `init_cmp` diagnostic, pre-registered switch only if `clone_neighbor` rel-MSE < 0.8× |
| Stage 1 | 400 steps (13M tok) | layer-wise MOHAWK alignment of all 8, **70 min** (~77M tok at 18.3k tok/s) | 3.4× cheaper per token than stage 2 |
| Stage 2 | 8 × 60 min, layers added one by one | **430 min** end-to-end KL with **all 8 GDNs training jointly from step 0** (~130–150M tok) | every layer gets the whole budget, not 18–34M |
| Data | wikitext-103 train, 40M, ~4.7 passes | **generic mix, 280M tokens, each window read at most once**: fineweb-edu 55 % / ultrachat (Qwen chat format, empty think block = the readout's) 30 % / finemath 15 % (`convert/mixdata.py`) | the teacher's real input distribution, incl. chat scaffolding |
| Decontamination | none | any doc sharing a token 13-gram with any of the 12,104 DEV rows is dropped (reported) | no Index row text can leak in |
| Gate | per-layer hidden_cos ≥ 0.95, dNLL | **end to end**: held-out generic KL (web/chat/math, 64 × 4096) + **stock Index proxy600** (644 rows, paired vs stock, same job, same readout path) | measures what we want; no structurally unreachable floor |
| Reported | — | KL curve every 60 min, wikitext gate numbers (comparable to 2xe5w), ppl/MMLU-285/MQAR/S-NIAH, full MMLU (14,042), throughput + call latency | diagnosis for v2.1 |

Teacher = the open-weight Qwen3.5-9B itself (same weights, attention mode). No TEST, held-out, cohort v3 or Index row is
ever trained on. The sources are generic public corpora, pinned to immutable revisions (`scripts/sf_fetch.py` stage
`q9af_v2`). The DEV rows only build the n-gram filter.

### After this job (not implemented here)
- **STRONG / ADVANCE:** `python -m jobs.q9af_cell.pin fill --export-json export.json --gdn gdn_converted.pt` in its own
  commit; the export format is full8's. Then the q9af cell runs our recipe: LoRA r64, then r256 (GDN in_proj_qkv / z /
  out_proj + MLP, all 32 layers); fork head (or the joint head); a v3w-equivalent Qwen readout (word-like option ids
  scored by the first token of " <id>", code key for k ≥ 9, on the chatml template); and the anti-collapse settings
  (cosine LR, dev cards + best-ckpt scoring, collapse guard; `claude/anti-collapse`). Screen on proxy600, then full
  subset, then full DEV.
- **v2.1 levers, if the screen is MARGINAL/FOLD:** read the held-out KL curve first. If it is still falling, buy more
  stage-2 tokens. Otherwise un-freeze the converted layers' MLP (LoRA, merged at export), or redistil L7 / L19 / L27
  with an MCQ/retrieval-heavy generic mix (still no Index rows).

## Decision rule (BINDING, pre-registered: `jobs/q9af_v2/rule.py`, rule `q9af-v2-rule-v1`, pinned by tests/test_q9af_v2.py)

Committed before any v2 run or data. First match wins:

| Verdict | When | Action |
|---|---|---|
| NO_MEASUREMENT | the export is not final + attention-free + exactly [3, 7, …, 31]; the compare is not on proxy600 (ids `7d42c398…`) or not B 2000 / seed 0; a binding number is missing | exit 9; nothing pinned |
| FOLD_KL | held-out mix KL > **0.25** nats/token | archive the export; v2.1 levers |
| STRONG | proxy600 paired diff (converted − stock) point ≥ **−4** and CI low ≥ **−8** | pin; full-subset stock replication + r64 cell |
| ADVANCE | point ≥ **−10** | pin; r64 cell + its proxy600 screen |
| MARGINAL | point ≥ **−17** | decision-needed: only a Clef-level lift (+17.5) beats the champion |
| FOLD | otherwise | archive; DEAD_ENDS with the numbers |

Where the thresholds come from (receipts that predate this rule, no v2 number):
- stock is 38.27 and the champion is 38.67, and the measured lifts are +7.6 (our cell recipe) and +17.5 (Clef-flash);
- −10 means a cell needs a +10.4 lift, between those two measured lifts;
- −17 means only a Clef-level lift is enough;
- Q9-H2's −26 folds under every threshold.

KL alone never advances anything: Q9-H2 had a wikitext KL of 0.094 and still lost 26 points.

## Pricing (measured basis: job-5cmat, job-2xe5w, job-4j3vu; `jobs/q9af_v2/preflight.py`)

| Phase | cap (min) | projected / expected | basis |
|---|---|---|---|
| setup / weights | 10 / 20 | 2.4 / 0.3 | 5cmat |
| D (download ~3.0 GB of pinned parquet, tokenise 280M, 13-gram filter) | 35 | ~15 (estimate) | no receipt |
| P (stock probes, init_cmp, stage 1 70 min, stage 2 430 min, final evals) | 550 | 500.1 capped + 19.7 uncapped (needs 525.7 with the 30 % margin on the uncapped part) | stage rates 5cmat / 2xe5w |
| X export | 15 | 3 | estimate |
| I_stock / I_conv (proxy600, fork path, 2.78M positions + 1,566 calls) | 20 / 20 | 8.4 each | readout_volume per_row |
| S | 10 | 1 | 5cmat |
| reserve | 15 | | |

`max_duration_minutes` 695 = caps + reserve. One attempt (`max_restarts` 0; a crash salvages the last stage-2
checkpoint as `gdn_salvage.pt`). **Worst case $37.37** = (60 build + 695) × $0.0495 (`costs.worst_case_usd`), against a
hard cap of $40. Expected ~$28 (571 min). At the platform's 2026-10-08 quote rate of $0.063/min the quoted max is $43.79.

Stop floors (exit 3/4): stage 1 9,476 tok/s, stage 2 2,600 tok/s (50 % of the 2xe5w layer-3 rate, 5,210), peak 76 GB
(measured 39.7). The token budget check: 280M ≥ 253M, which is the use predicted at 1.15× the measured rates.

## Runner, outputs

`jobs/q9af_v2/q9af_v2_job.sh` runs the phases setup → D → P → X → I_stock → I_conv → S → R. Each phase is exported
and verified (MANIFEST.jsonl). `/output/Q9AF2_DONE` is written only by finalize, with a valid decision. Outputs:
- `gdn_converted.pt` (+ `.sha256`) and `export.json`, consumed by jobs/q9af_cell unchanged;
- `decision.json`;
- `phases/*.tgz`, holding results.json (KL curve, init_cmp, stage receipts, probes, MMLU full, throughput), mix.json
  (per-source tokens, drops, file sha256s) and the Index arms with the compare.

The CPU dry run (`Q9AF2_DRYRUN=1`, tests/test_q9af_v2.py) runs the real runner. Only the fetch, GPU and kit steps are
stubbed.

## Clean rerun (mode `clean`: `job_params_clean.json`, `configs/q9af_v2_clean_h100.json`)

**Takeaway.** job-w3c2e's mix was filtered against DEV only (dev13-v1). The corpus check (job-vdk25) found eval
contact in it (TEST / held-out / DEV / frozen) and that the full filter would drop 1,066 of its 249,744 train docs. The
rerun is the same conversion on a mix filtered with `full-eval-v1` (dev13-v1 + the pinned DEV + TEST + held-out
eval-hash index, `index_sha256` 2c821e29…, shipped as hashes in the 12 `doc-overlap/218affe13e367d07/` files the doc
gate uses, + the eval attack templates). The 280M-token quota is refilled from later rows. **The rule is unchanged**
(`jobs/q9af_v2/rule.py`, `q9af-v2-rule-v1`: KL ≤ 0.25, then STRONG / ADVANCE / MARGINAL / FOLD at −4/−8, −10, −17).

What changes, and why the gate still holds:
- **Exports.** Phase D writes `mix.json` (filter, rule text, index + template shas, drops per reason) and
  `windows.json.gz` (`convert.mixdata.WindowManifest`: per doc its id, token offset and length; per train window
  w = tokens [w·4096, (w+1)·4096) its doc range and its rank in WindowStream's permutation). Both go to `/output/` and
  into every phase tgz, so `corpus_check compare` and window-level audits run offline.
- **Budget fit ($41).** The quote is `max_duration × $0.063`. With the lint's 60-min build allowance:
  (60 + 588) × 0.063 = **$40.82** worst case (`costs.worst_case_usd`, `batch_usd_per_min` set to the quote rate).
  The platform quote alone is $37.04. Trims, in order:
  1. Phase caps where w3c2e had slack (attempt 4 logs, 2026-10-09): setup + weights measured 3.3 min (caps 10 + 20 →
     8 + 10). D measured 4.9 min under dev13-v1; full-eval-v1 adds ~8.3 min of serial guard (0.45M words/s per core ×
     ~224M words), so D 35 → 30. X 15 → 8 (est. 3). I arms 20 → 15 each (projected 8.4). S 10 → 5 (measured 1).
  2. Stage-2 KD 430 → 385 min (−10 %, ~116–134M instead of ~130–150M stage-2 tokens). P 550 → 482 = 70 + 385 +
     1.3 × 19.2 uncapped + 1.9 slack (preflight need 480.0). Stage 1 (70 min), init, floors, peak stops, eval cadence
     and every held-out / proxy600 measurement are unchanged. So the KL and proxy600 numbers are measured exactly as in
     w3c2e. The trade-off: 10 % fewer KD tokens can only raise KL or lower the proxy diff. A pass is still a pass under
     the same thresholds. A FOLD_KL / MARGINAL near a threshold should read the KL curve (still falling = buy minutes).
- **Not changed:** the image (same Dockerfile), the decision rule, the proxy600 ids, B 2000 / seed 0, one attempt.

| Phase | cap (min) | basis |
|---|---|---|
| setup / weights | 8 / 10 | w3c2e attempt 4: 3.3 total |
| D (fetch + tokenise 280M + dev13 + full-eval-v1 guard + window manifest) | 30 | 4.9 measured + ~8.3 est. |
| P (stage 1 70, stage 2 385, uncapped ×1.3) | 482 | preflight need 480.0 |
| X / I_stock / I_conv / S | 8 / 15 / 15 / 5 | est. 3 / proj. 8.4 / 8.4 / meas. 1 |
| reserve | 15 | |

## Clef-flash diagnostic (mode `clefflash`: `job_params_clefflash.json`, `configs/q9af_v2_clefflash_h100.json`)

The clean recipe above, verbatim, on Clef-flash's weights; the I arms are replaced by four proxy600 arms (Clef's joint head and the LM-logit v3w readout, each stock vs converted). A diagnostic with no keep/discard rule: see `README_clefflash.md` (pre-registered metrics, recipe identity, budget). One attempt, 638 min: **worst case $43.97** = (60 + 638) x $0.063 (hard cap $44); platform quote $40.19.
