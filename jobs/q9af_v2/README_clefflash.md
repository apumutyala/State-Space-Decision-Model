# q9af_v2 mode `clefflash`: what does removing attention cost at Clef-flash's level? (DIAGNOSTIC)

**Takeaway.** This job applies job-7ji99's clean Q9-AF v2 conversion to Clef-flash instead of stock Qwen3.5-9B. All 8
attention layers become GDNs, distilled with Clef-flash itself as the teacher. Then it scores stock and converted
Clef-flash on proxy600, paired, in the same job, through two readouts:
- **A (primary):** Clef's official joint head.
- **B:** an LM-logit v3w readout.

Only B's converted arm is a fully attention-free system. This is a measurement, not a claim. There is **no keep /
discard rule**, and nothing is pinned or promoted from it.

## Pre-registered metrics (committed before any run; `jobs/q9af_v2/diag.py`, `q9af-v2-clefflash-diag-v1`)

Rows: proxy600 (644 DEV rows, ids sha256 `7d42c398…`, `jobs/common/dev_proxy600.json`). Paired bootstrap B 2000,
seed 0 (`jobs.common.dev_subset compare`). Edition **0.2.1** is primary. The **0.3** edition (`--omit-gsm8k03`, as
the Clef jobs ran it) is reported alongside.

| Arm | Backbone | Readout | Attention-free system? |
|---|---|---|---|
| `clef_stock` (C_stock) | Clef-flash as released | Clef's joint head: `baselines.clef_index_engine:ClefIndexEngine`, decide path, ssdm encoding, `gdn_path=reference`, transformers 5.10.2 (model card) | no |
| `clef_conv` (C_conv) | Clef-flash, 8 attention layers → converted GDN | the same engine and head (`jobs.q9af_v2.clef_af:ClefAFIndexEngine`) | no: the head cross-attends over every position (EvidenceRoutingLayer, TransformerDecoderLayer over `memory`) |
| `v3w_stock` (L_stock) | Clef-flash as released | LM-logit v3w: `make_v3w("qwen3.5-9b")` (PR #236) + `decide.backends.v3w_generic_backend`; next-token log-prob of each option's single token after `ANSWER:` | no |
| `v3w_conv` (L_conv) | converted Clef-flash | the same | **yes** |

Reported numbers (all of them are binding for completeness; none gates anything):
- **Δ_A** = clef_conv − clef_stock (point, 95 % CI): the primary number, in 0.2.1 and 0.3;
- **Δ_B** = v3w_conv − v3w_stock (point, CI), in 0.2.1 and 0.3;
- the four arm scores in both editions;
- the conversion's held-out mixture KL (KL(Clef-flash ‖ converted), nats/token);
- in results.json (as in every v2 run): the KL curve, the probes (ppl / MMLU-285 / MQAR / S-NIAH), full MMLU, and
  throughput.

Reading guide (context only, not a rule): stock Clef-flash scored 55.78 on the 2,008-row subset (job-sgihj). A
proxy600 Δ_A near 0 would mean that Clef-flash's quality survives the loss of attention in the backbone. Δ_B prices the
fully attention-free system under a generic LM readout, without Clef's trained head.

## The recipe is job-7ji99's (byte-identical apart from the base model)

- `configs/q9af_v2_clefflash_h100.json` equals `configs/q9af_v2_clean_h100.json` key for key, except for two things:
  - `model_dir` → `Cloudflare/clef-flash@17f0b0ad…`. The teacher is therefore Clef-flash in attention mode.
  - `tokenizer_dir` → the Qwen3.5-9B tokenizer at `c2022362…`, the one 7ji99 tokenized with.

  `tokenizer_dir` is a new key, and it is opt-in: `convert.model.tokenizer_dir` falls back to `model_dir`, so the
  clean / w3c2e configs behave exactly as before. Pinning it keeps the mixture (full-eval-v1 decontam, 280M tokens),
  the held-out windows and the probes token-identical to 7ji99's. Clef-flash ships a newer tokenizer.json: the same
  vocab, with a pre-tokenizer that keeps combining marks attached to letters. `scripts/check_clefflash_v3w.py` found
  that it tokenizes 1 of the 3,730 proxy600 readout questions differently.
- Same stage-1 / stage-2 wall clocks (70 / 385 min), LR, init (`reuse_attn` + init_cmp switch), layer set
  [3, 7, …, 31], floors, peak stops, eval cadence and seed. Same caps for weights / D / P / X. Same image layers.
  tests/test_q9af_clefflash.py pins all of this.
- Image: `jobs/q9af_v2/Dockerfile` unchanged, plus one RUN layer before the code COPY (a launcher rewrite):
  `/opt/clefenv`, a `--system-site-packages` venv with transformers 5.10.2 + torchvision 0.28.0. Only the C arms run
  in it. Every conversion phase, the L arms and the scoring run on the system python (transformers 5.17.0, asserted
  at build).
- Fetch stage `q9af_v2_clefflash`: the same mixture / probe sources at the same revisions, the whole Clef-flash release
  (its LFS sha256s are verified at setup, exit 8), and the 9B tokenizer files only.

## Pair B readout checks (`analysis/check_clefflash_v3w.json`, CPU, proxy600)

- 255 codes. Every scored token of all 3,730 questions (3,327 word-id, 403 code) is ONE token after the answer prefix,
  and the tokens are distinct within each question. No question needs the multi-token fallback.
- 5 rows exceed the 16,384-token limit and are refused as `exhausted` (scored 0) in both L arms. For the C arms, 4 rows
  exceed Clef's limit.
- Readout identity `2e4907ee…`. The job rebuilds it at setup from the shipped 9B tokenizer.json /
  tokenizer_config.json (sha256 pinned in `data_clefflash.sha256`) and stops (exit 10) on a mismatch, before any GPU
  phase.
- Fork volume: 1.92M positions and 1,401 calls (`readout_volume_v3w_proxy600.json`). The stock arm applies the KV
  fork budget.

## Budget (one attempt, 1×H100)

| Phase | cap (min) | basis |
|---|---|---|
| setup / weights | 10 / 10 | clean 8 / 10, + Clef snapshot sha256 verify, clefenv and readout identity checks |
| D / P / X | 30 / 482 / 8 | the clean caps, unchanged |
| C_stock / C_conv | 20 / 25 | job-sgihj row model on the 1,210,697 Clef-encoded proxy600 tokens = 1.15 min fast; × 4 reference slowdown (× 2 more for the converted arm, torch chunk kernel), + 3.5 min load / verify |
| L_stock / L_conv | 15 / 15 | the exact v3w fork volume at the measured prefill rate and call latency (~7 min) |
| S | 8 | 4 compares + 8 scores |
| reserve | 15 | |

`max_duration_minutes` is 638, under the 900-minute platform limit. The platform hold is 638 × $0.063 = **$40.19**.
Worst case with the 60-minute build allowance: (60 + 638) × $0.063 = **$43.97** (hard cap $44). Expected is about
$38 (~600 min, dominated by the 455 capped KD minutes).
