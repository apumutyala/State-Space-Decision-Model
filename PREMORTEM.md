# PREMORTEM: from here to a goal-satisfying submission (Track D, 2026-10-06)

**Premise.** Assume it is 2026-10-27, the $500 promo credits have expired, and we did not merge an attention-free model that beats Clef and Clef-flash on the full Decision Index 0.2.1 with a paired CI above zero. This document lists, stage by stage, the most likely reasons, with code evidence. It also gives a cheap probe for each failure that we can run now.

**Basis.** origin/agent-main @ 54ee89d, plus open branches `claude/q9-full8`, `claude/m2-first-cell` (PR #74) and the uncommitted `m2-data-150k-v2` worktree. Issues #3, #36, #68 and #87, and the `job-sgihj` platform record.

**Probes.** Every probe in §10 is a CPU test in `tests/test_premortem_probes.py` (P1-P7: 11 test cases, all pass, ~3 min on CPU, $0). Their receipts are in `receipts/2026-10-06-D-premortem-probes/`.

## 0. Where we are

| Item | Value |
|---|---|
| DEV subset (2,008 rows, sha 6e4cb8df…) | Clef-flash 55.8 [53.0, 58.5]; Qwen3.5-9B stock 38.3; Q9 with 1 of 8 attention layers converted 39.9; RWKV7-13.3B stock 24.9 |
| Full DEV (12,104 rows) | Clef-flash 56.8 |
| Clef (27B, 61.98 claimed) | not measured |
| Trained decision head | none yet. The first cell (RWKV7, PR #74) is blocked: train step OOMs from T≈1k (job-mden7). Profiling job-tvuat for PR #86 flags is running |
| Money | $525 envelope, $14.49 spent month to date, so ~$510 left. **$500 expires 2026-10-27T06:53Z: 20.3 days** |

**New primary-source fact (operator, Cloudflare blog).** Clef-flash is a FROZEN Qwen3.5-9B plus a jointly trained routing head plus rank-256 LoRA. It is trained with label-smoothed CE + Brier on synthetic data that permutes field orders, prompts and schemas, plus an RLCD secondary objective. Inference is prefill-only with parallel scoring of all schema choices. The head is a two-stage attention head: option-specific evidence routing, joint cross-field attention, and a lexical prior. Weights are Apache-2.0.

**Strategic consequence.**
- Clef-flash is our exact same-backbone comparator. Our backbone is that same Qwen3.5-9B with its 8 attention layers distilled to GDN, so it can only be equal or weaker.
- The whole 38 → 56 gap is closed by head + LoRA + data on a frozen backbone.
- To beat Clef-flash we must out-do Cloudflare's head, data and recipe, with an attention-free head and a weaker backbone. To beat Clef (27B) we must also make up a 27B-vs-9B backbone gap.
- The critical path therefore centres on **Q9-AF + trained head**. RWKV7 becomes the fallback.
- The cheapest decisive experiment is a head on cached, frozen Q9 states (§11).

## 1. Stage map and critical path

```
 S1 full-8 Q9-AF conversion ──┐
                              ├─> S2 Q9-AF head+LoRA cell ─> S2b attention-free routing head ─> S2c RLCD-like stage (optional)
 S3 data / S3b teacher KD ────┘          ^                                         |
 S8 proxy loop (cached frozen-Q9 features) ┘ (recipe chosen here, before S2 spend)  v
 S4a Clef + Clef-flash baselines (DEV, full Index) ──────────────> S4 eval: full DEV -> PREREG -> full Index (= TEST unseal)
 S5 safety gates (G12/G13 in-job; G6 GCG on the deployed model) ──┘          |
 S7 coordination (spend acks, #36 margins, Codex re-run) ────────────────────> independent reproduction -> Index PR (user approval)
```

**Critical path (longest chain to the deadline).**
S1 → S2 smoke → first Q9-AF cell → scale-up cell → full DEV paired vs Clef → PREREG → full Index runs (candidate, Clef, Clef-flash) → reproduction.
S8 (proxy) and S4a (baselines) run in parallel and feed S2 and S4. Both must finish before the cell they feed.

## 2. Stage S1: full-8 Q9-AF conversion (issue #87). Critical path

**Preconditions.**
- Qwen3.5-9B @ c2022362 (`heads/backbone.py:682-683`).
- Pins: fla 0.5.2, Triton ≥ 3.7.1 (`jobs/common/fla_guard.py:24-55`).
- A full-8 runner, config, price, binding rule, dry run and independent review (issue #87, gates 1-4). Only the rule exists so far (`origin/claude/q9-full8`: `jobs/full8/rule.json`, `scripts/decide_full8.py`, `tests/test_full8.py`).
- A Codex spend acknowledgement (issue #87 has 0 comments).

| # | Failure | Evidence | P | Detection cost if found late | Cheap probe |
|---|---|---|---|---|---|
| S1-F1 | No runner exists, and the current one is one-layer only | `jobs/probe9b/probe_job.sh:109` hardcodes `prog_1.pt`; `configs/probe9b_h100.json:64` has `max_layers: 1`; `preflight.py:51,60-61` projects one stage 2. No `gdn_partial.pt` export | ~1.0 unless fixed | A failed $28-74 job | $0: shell dry run with a fake 8-step `results.json` including one rollback |
| S1-F2 | Wall clock: about 17 + 8×62 ≈ 515 min of P phase. The deadline is checked only between layers (`convert/progressive.py:44`), so a late layer overruns by up to 60 min. Skipped probes give a missing final tag and a "no-measurement" verdict | job-5cmat P = 79 min for 1 layer (#68); stage-2 cap 60 min/layer | 0.3-0.5 | Full job cost and 1 day | $0: read `timing` + probe `seconds` from art-2bybh; size the deadline as ≥ 17 + 8×(60 + gate + probe) + margin |
| S1-F3 | A crash or timeout loses every trained layer. Checkpoints live in `$WORK/data/ckpt` and are never exported (`convert/pipeline.py:113`, `jobs/common/lib.sh:19`). `PHASE_HOOK` (`pipeline.py:67-72`) is unused, and `resume: none`, `max_restarts: 0` | job_params | 0.4 over a ~9.5 h job | Up to $74 and 1-2 days | $0: wire `PHASE_HOOK` to export `prog_<n>.pt` (2.16 GB each); CPU test that resume seeds from it |
| S1-F4 | Compounding error. The gate requires final hidden cos ≥ 0.95 (`convert/progressive.py:13-15`). Layer 23 cost 1−cos = 0.006; layer 3 alone measured cos 0.980 (job-4j3vu). The sum crosses 0.05 by layer 6-8, so late layers are rejected and the model stays a **hybrid**. `q9af_from_stock` then refuses it (`heads/backbone.py:710-713`), so the S2 path dies | measured per-layer numbers | 0.4-0.6 | The whole Q9-AF branch: about 1 week | $0: sum per-layer (1−cos, ΔNLL) from art-2bybh `importance.per_layer`. ≤ $3 GPU: stage 1, then `align.evaluate` with all 8 in student mode |
| S1-F5 | MMLU noise. The bound is 3.0 points on 285 items, but `decide_full8` estimates the SE at about 3.8 points. One layer already cost 1.75 | `jobs/full8/rule.json` | 0.3-0.45 | A spurious quality-fail after full spend | $0: paired per-item variance from job-5cmat MMLU outputs. Fix the rule before the run (larger item set or CI-based bound) |
| S1-F6 | Retrieval collapses once ALL attention is gone. At one layer the MQAR/S-NIAH probes still had 7 attention layers. The recall levers (JRT, a 20-30% retrieval mix) are not built: distillation data is wikitext only (`convert/data.py:21-25`) | PLAN fallback; literature in PLAN.md | 0.35-0.55 | Pre-registered fallback: report the cost-of-attention-freedom curve, ship RWKV7 or a hybrid (user decision) | ≤ $3: MQAR/S-NIAH with attention zeroed (`controls.zero`) and with all-8 stage-1-only GDNs |
| S1-F7 | Length mismatch. Distillation uses seq_len 4,096, but Index DEV prompts reach p90 6.2k, p99 11.6k and max 28.5k tokens (295 of 2,008 rows > 4k) | config line 6; `readout_volume_qwen3.5.json` | 0.2-0.35 | Index quality loss found only at S4 | $0: split job-5cmat's paired Index diff by prompt length ≤ / > 4,096 |
| S1-F8 | The gate/Index path (`convert.model.ConvertibleQwen` + `MixerSwap`, fp32 GDNs, `readout.py:92` `attention_free: False`) differs from the m2 path (`q9af_from_stock`, HF `Qwen3_5GatedDeltaNet` in bf16). Equivalence is tested only on a tiny model | `tests/test_m2_gates.py:299,468-487` | 0.15 | The parent number and trained-cell number are not comparable | $0 CPU: all-converted tiny model, compare logits and fork features through both paths |
| S1-F9 | Warm start is not implemented. DECISION_TREE "starting from gdn_converted.pt" has no `init_from`, so stage 1, the ranking and layer 23 are redone; nondeterminism can reorder the KL order | `pipeline.py:252-262` | 0.15 | An audit no-measurement | $0: CPU test with a pinned order and shuffled KLs through `decide_full8` |

**Cost.**
- Expected about $28 (≈ 560 min).
- Worst case about $74 at max_duration 735 (`jobs/common/costs.py:39-40`). `FULL8_ESTIMATE = 150` (`costs.py:28`) is a placeholder.
- Memory: about 43 GB of the 76 GB stop (memmodel), so OOM is unlikely.

**Owner:** D. **Critical path:** yes.

## 3. Stage S2: training the decision head on Q9-AF. Critical path

**Preconditions.**
- `q9af.converted_sha256` pinned. It is `null` in `jobs/m2_pilot/m2_pilot_cell_q9af.json`, and `run_cell` refuses until it is set (`train/run_cell.py:295-303`).
- The 150k set wired into `run_cell`. That is only on PR #74 (`train/train_set.py`); on agent-main a 150k file fails the PR #16 content-map check (`train/cohort.py:56-57`, `run_cell.py:117-131`).
- Approved margins (#36). A Codex spend ack (#3).

**What already works ($0 evidence, this PR).** `load_q9af_backbone` (`heads/backbone.py:763-782`) and `q9af_from_stock` (`:700-730`) plug into the real train step:
- P1: LoRA reaches the converted layer's GDN projections, `Trainer.train_step` runs with the cache fork and the Clef loss arm, and checkpoint/resume works.
- P2: cache-fork LoRA gradients equal recompute gradients, max relative error 9.4e-6 over 48 LoRA tensors. The in-place-cache patch `_assign_not_copy` (`heads/backbone.py:416-444`) is correct under autograd.

So interface compatibility is not the risk. Memory and speed are.

| # | Failure | Evidence | P | Late cost | Probe |
|---|---|---|---|---|---|
| S2-F1 | **None of PR #86's levers applies to Q9-AF.** `act_ckpt` raises: "RWKV-7 only" (`heads/act_ckpt.py:49-55`, `train/trainer.py:160-161`). `row_batch>1` silently falls back to one call per row: `can_fork_rows` needs `state=` (`heads/backbone.py:351-356`, `heads/features.py:116-121`). HF gradient checkpointing is refused under the cache fork (`heads/backbone.py:285-288`), and run_cell hard-sets `grad_checkpointing=False` (`train/run_cell.py:153`). Q9-AF therefore trains with full activations | **P3 (pass: blocker confirmed)** | 0.9 | First cell OOMs at T=2,048 (same as job-mden7 for RWKV7) | done, P3 |
| S2-F2 | Activation memory. At matched shapes the stand-in Q9-AF saves 0.97× RWKV7's bytes per token per layer per d. Anchored on job-mden7's ~1.0 MB/token/layer, that gives Q9-AF about 0.97 MB × 32 layers ≈ **31 GB per 1k tokens**, with 58.8 GB headroom: **max T ≈ 1.95k without checkpointing**. The same model gives RWKV7 ≈ 0.76k, which matches its observed "OOM from T≈1k". So the 2,048 ladder rung OOMs, and the 455/2,008 DEV rows longer than 2,048 are never trained on | **P4** (CPU torch GDN fallback; fla saves less; tiny head dims) | 0.7 | One failed cell (~$26-64) | done, P4. Next: port `act_ckpt` to the HF GDN layer (move `conv_states`/`recurrent_states` out of the checkpointed region, exactly as RWKV7 does). CPU equivalence test, $0 |
| S2-F3 | The fla #640 path is exercised. LoRA on GDN backpropagates through fla's gated `chunk_bwd_dqkwg` (the #640 Hopper miscompile). The cache-fork continuation uses the chunk kernel with `initial_state` (HF `modeling_qwen3_5.py:539-545`) | `jobs/common/fla_guard.py`; guard present in `m2_job.sh:80`, `first_cell_job.sh:108` | <0.05 with guard | — | Keep the guard; add a GPU grad-equivalence smoke (the cell's exit-21 smoke already checks feature diff and grad cosine) |
| S2-F4 | Training/eval length filter. `keep_item` drops rows with T > `eval_max_t` (2,048) or K > 16 (`train/run_cell.py:172-173`, `m2_gates.py:73,83`) on cohort roles. That is fine for dev, but the full Index must be scored without truncation or pruning (GOAL integrity). Long rows go untrained | PR #74 config | 0.5 | Length-dependent loss on the full Index | $0: per-length-bin paired diff (PR #74 already reports ≤2,048 vs >2,048) |
| S2-F5 | **Clef recipe deltas** (see table below). Each unexamined delta is a reason our head stays below Clef-flash on the same backbone | blog vs code | 0.6 | The whole m2 budget | Proxy loop S8 |
| S2-F6 | Throughput. Eager per-row calls (job-mden7: launch-bound, 1.4-1.8 M aten ops per 8-row step). Q9-AF has no batched-row path (S2-F1), so the RWKV7 projection of 13k rows per 190 min may not hold | `jobs/PERF_AUDIT.md:42` | 0.5 | The cell trains on <10% of 150k | ≤ $2: ladder + 10-step timing inside the S1 follow-up job |

**Clef-flash recipe vs our cell** (`jobs/m2_pilot/m2_pilot_cell_q9af.json`; PR #74 `m2_first_cell_rwkv7.json`):

| Item | Clef-flash (blog) | Ours | Delta / action |
|---|---|---|---|
| Backbone | frozen Qwen3.5-9B hybrid (8 attention layers) | frozen Q9-AF (0 attention) | Structural; capacity loss measured at S1 |
| LoRA | r = 256 | r = 64 (memory, PR #74 note) | A 4× capacity gap; test r ∈ {64, 128, 256} in the proxy and full-step ladder |
| Head | two-stage attention: option-specific evidence routing + joint cross-field attention + lexical prior | `fork` (final-state fork MLP, `heads/heads.py:145-166`); `integrator`, `snapshot`, `HeadB`; the Clef-style attention head exists only as a priced control (`ClefCrossAttentionControl`, `heads/heads.py:322`) | S2b |
| Loss | LS-CE + Brier | `clef` arm = LS-CE(0.1) + Brier + RPS on score questions (`train/losses.py:5-7, 69-78`) | RPS is extra (close to RLCD's "partial ordinal credit"); ablate in the proxy |
| RL | RLCD secondary objective | none | S2c |
| Data | synthetic, permuted field orders / prompts / schemas | `data/augment.py:1-12`: option-order permutation, id relabelling, paraphrase, key renaming/reordering, placement, reversed score scales; used in the build (`data/build.py:254,275`) | Matches in kind; the volume actually trained is about 9% of 150k (PR #74 JOB_SPEC) |
| Inference | prefill once + parallel scoring of all schema choices | cache fork: one prefix + K continuations per question (`heads/backbone.py:265-329`); `shared_prefix` for eval (`heads/prefix_features.py`) | Latency/cost: see S4 |
| Teacher | — | none (`data/TEACHERS.md:3` "plan only") | S3b |

### S2b: attention-free replacement for Clef's cross-field routing (new)

Clef's head attends over the backbone states (evidence routing) and across fields (joint scoring). Our candidates without softmax over positions:
- **HeadB**, a per-option selective recurrence (`heads/heads.py:240`).
- The **bounded-S snapshot fork** (`:169`), where option-conditioned routing equals a learned gate over S state snapshots.
- A **joint cross-field step** over the per-question fork features. This is a fixed-size set of Q×K vectors, not positions, so a DeepSets/linear mixer over fields is attention-free under GOAL's definition. It needs a disclosure line.

**Failure:** a softmax over *options* or *fields* could be judged "attention". Pre-register the definition (positions only) in SAFETY_SPEC / interface before training.

**Probe:** the proxy loop (S8) on frozen-Q9 cached features, comparing HeadB, snapshot, fork + field-mixer and `ClefCrossAttentionControl` on identical cached inputs.

**Owner:** D. **Critical:** yes, for beating Clef-flash.

### S2c: RLCD-like calibrated RL stage (new, optional)

- **Recipe:** after supervised training, a small policy-gradient or DPO-style stage. Reward: ordinal partial credit, whole-record precision, and a KL/reference penalty to the SFT head.
- **Failure:** calibration drifts (RL sharpens probabilities, and ECE is reported). Mitigate by refitting the temperature and re-checking debiased ECE (`harness/calibration.py:141`).
- **Probe:** in the proxy loop, the head is the only trainable part, so an RL step over cached features costs seconds.
- **Gate:** adopt only if the paired DEV gain is > 0 with ECE inside m_ece.
- **Owner:** D. **Critical:** no. Schedule after the first Q9-AF cell.

## 4. Stage S3: data, and S3b: teacher distillation

**Preconditions.**
- The PR #44 150k set: records outside git, sha 82fb395d, pointer only.
- v2 fixes the review findings F1-F4 but is uncommitted (`m2-data-150k-v2`: no README, recheck or pointer).
- A train-set path merged to agent-main (PR #74).

| # | Failure | Evidence | P | Late cost | Probe |
|---|---|---|---|---|---|
| S3-F1 | **Coverage, not volume.** Area shares match the Index weights (`receipts/2026-10-06-D-m2-data-150k/README.md:88-94`), but per benchmark: knowledge covers 2 of 10 benchmarks (GPQA, MuSR, SATA, CRUXEval, CLadder, HLE, BBH and ChessBench have none; `data/SOURCES.md:21-30`); arts covers 1 of 7 (Humicroedit only); tools uses proxies only (Glaive/ToolACE/When2Call, with no API-Bank, ToolRet or Home-appliances). Our gaps vs Clef-flash are exactly knowledge, tools and arts | `harness/INDEX_NOTES.md:32-36` | 0.7 | A gap that training cannot move | $0: per-benchmark decomposition of Q9 vs Clef-flash on the DEV subset (from art-8y6d8 and job-5cmat predictions), weighted by "has any training source" |
| S3-F2 | **Knowledge is a backbone limit.** A head and LoRA cannot add GPQA, HLE or BBH knowledge. But Clef-flash shares our backbone, so its knowledge score is an upper bound reachable from Q9 states. The question is head and data, not knowledge | blog fact | — | — | Proxy: frozen-Q9 cached head on knowledge rows only |
| S3-F3 | v1 cohort leak. The reviewer bounded it at 37 records, but v2's probe skips 1,105 (ANLI 683, BoolQ 156, …); the 30× gap is unexplained. If v1 leaks, G12 is inflated | v2 `build-summary.json:596-664` | 0.3 | A false G12 pass, found at review | $0: diff the v1 manifest against the v2 skip list; hand-check 50 hits |
| S3-F4 | Irreproducibility. v2 is uncommitted and edits `train/cohort.py`; the first cell trains on v1, which has known flaws | PR #74 JOB_SPEC:15 | 0.5 | A receipt that names a flawed set | $0: commit v2 + README + recheck + pointer; rebuild twice and compare sha |
| S3-F5 | Label noise / mislabels: Glaive `no_tool` deferral (4,908 of 7,858), the SST-5 reversed scale, coarse Humicroedit s1 grades | PR #44 review F3 | 0.4 | Tools abstention scores 0 | $0: hand-audit 200 records per source |
| S3-F6 | Licences: ANLI is CC-BY-NC; SST-5, NQ-pairs and Humicroedit have no stated licence | README:140-143 | 0.3 | Release blocked | $0: licence table before the Index PR |

### S3b: teacher distillation from Clef / Clef-flash per-option probabilities (new candidate stage)

Clef and Clef-flash are open-weight (Apache-2.0), so PLAN.md allows them as teachers.

**Pipeline status: none exists.**
- `data/TEACHERS.md:3` says "plan only, $0 spent". The target `p = α·onehot + (1−α)·softmax(z/T)` is at line 39.
- `train/losses.py:69-78` accepts only an integer `y`.

**Build ($0 CPU):**
1. A `soft_targets` field on `Item`.
2. A KD term in `decision_loss` (KL to the teacher distribution, with an α mix).
3. A teacher-logit cache job that reuses the Clef baseline image (`baselines/clef_backend.py`).

**Cost.** Clef-flash measured about 10.4 rows/s (job-sgihj), so 150k rows ≈ 4 h ≈ **$12**. Clef 27B is about 5.4× slower: about 20 h, $60, so use a subset.

**Contamination and licence checks:**
- Teacher outputs only on training records that already passed the content-index guard and the D1-B/cohort exclusions. Never Index DEV/TEST rows, not even for "calibration".
- Record the teacher revision sha in lineage.
- Apache-2.0 attribution in the model card.
- A teacher trained on unknown Cloudflare data that overlaps the Index is a contamination channel we cannot audit. Disclose it, and report the KD vs no-KD paired diff.

**Owner:** D. **Critical:** not for the first cell. It is the main lever for knowledge/tools/arts once the proxy shows the head ceiling.

## 5. Stage S4: evaluation. Critical path

| # | Failure | Evidence | P | Late cost | Probe |
|---|---|---|---|---|---|
| S4-F1 | **The full Index is not "DEV + sealed TEST".** A board submission scores every scoreable row (150,317; `harness/INDEX_NOTES.md:73,110`). DEV+TEST is about 24k of 120,226 counted rows. The CLI can score only dev/test (`harness/cli.py:163,169`) and renormalises missing benchmarks (`score.py:483-487`); the board does not. So the submission run IS the TEST unseal, and about 10% of it is DEV rows used for selection | code | 1.0 (structural) | No submittable number | $0: add a full-rows scoring path; selfcheck with perfect/uniform predictors over all rows; write the PREREG (model sha, rows, B, seed) before any full run |
| S4-F2 | **The TEST seal is procedural.** "Once" is not enforced: `harness/cli.py:52-56` only appends to the unseal log | code | 0.2 | An integrity claim we cannot make | $0: refuse when the unseal log is non-empty, plus a test |
| S4-F3 | **Clef (27B) is unmeasured, and the composite crash blocks it.** job-sgihj (Clef-flash A) died at stage `composite`, exit 12. Cause: `jobs/clef_baseline/summarize.py:24` uses `str.splitlines()`, which splits inside JSON strings on U+2028/U+0085/\x1c-\x1e, while the kit writes `ensure_ascii=False` and splits on `\n` only. **Reproduced on CPU (P7).** Consequences: S3 (score, ECE, compare), S4 (steer) and S5 (latency) never ran for Clef-flash, and job C (Clef) `depends_on` A `result.ok == true`, so it can never launch as specced | P7 | 1.0 until fixed | No Clef baseline = no goal | **Fixed probe exists.** Fix: `split("\n")`; then re-run A's S3-S5 and relax C's dependency |
| S4-F4 | Clef-flash per-row predictions exist only in artifact art-8y6d8 (sha eab45f29…, 91 MB). Not committed or pinned locally; paired CIs need them | — | 0.2 (artifact retention) | Re-buy the baseline | $0: download, check SHA256SUMS, commit the sha, recompute 56.8 |
| S4-F5 | Paired CI on identical rows: `harness/bootstrap.py:20-44,75-97` is a paired cluster bootstrap and raises on mismatched benchmark sets. It needs per-row predictions of both systems on the same rows, and we have none for Clef. MDE on DEV is 0.65-2.4 points, so on a single 12k slice the win must be more than 1-2.4 points | `INDEX_NOTES.md:87-92` | 0.5 | A "CI includes 0" verdict | $0: `harness mde` with the sgihj SE |
| S4-F6 | **Full-Index cost and time for our candidate.** The fork path is about 36 min per 2,008 rows: full DEV about 3.6 h (~$11), full Index about **45 h (~$134)** on one GPU, which collides with the deadline. Clef full Index is about 18-21 h (~$55-65); Clef-flash about 4 h (~$12) | readout volumes + job-tkfaj rates | 0.6 | Not finishable before 10-27 | $0: price with `shared_prefix=True` (`heads/prefix_features.py`: 55M → 5M tokens on RWKV7 DEV); plan 3-4 parallel nodes |
| S4-F7 | Latency/cost not reportable as required. No latency-v1 sampler (750 rows, seed 20260926), no RTX PRO 6000, no $/1k-decisions code (`jobs/clef_baseline/summarize.py:55-60` notes "NOT the board's latency-v1") | grep | 0.8 | A GOAL item missing at merge | $0: implement the sampler; $/1k = Σ `total_wall_ms` × $2.97/h |
| S4-F8 | Board ECE is not reproducible (unpublished sample; n = 72,594) | `INDEX_NOTES.md:52,81` | 0.5 | Calibration claim scoped down | Report debiased ECE / SmoothECE on identical rows for all three systems |
| S4-F9 | Independent reproduction: the m3 job is not specced (`HANDOFF.md:123`). Codex's CPU probes hit RAM floors (#34), and both agents act under one GitHub identity, so "another agent" is hard to audit | `COORDINATION.md:34,126` | 0.5 | The claim stays "one track only" | $0: make the m3 receipt CPU-recomputable from raw `results.jsonl` (`harness score/compare`); Codex re-scores |

**Owner:** D (Codex for reproduction). **Critical path:** yes (S4-F1, F3, F6).

## 6. Stage S5: safety gates (G12/G13, G4/G6)

| # | Failure | Evidence | P | Late cost | Probe |
|---|---|---|---|---|---|
| S5-F1 | **Margins are unapproved, and exit 0 masks it.** `MARGINS` is a constant marked "PENDING USER APPROVAL" (`train/promotion.py:40-42`). `promotable` is false while pending (`train/m2_gates.py:247-252`), but `exit_code` returns accept/0 regardless (`:256-263`). Nothing reads issue #36, and Codex has not replied | code + #36 | 0.6 | No promotable model at the end | $0: make pending a distinct non-zero exit; escalate #36 to the user as `decision-needed` with a deadline (COORDINATION allows that after two rounds) |
| S5-F2 | **GCG cannot attack the deployed model.** `gcg.Engine` drives `HFLetterDecider` (an `AutoModelForCausalLM` LM head; `safety_eval/hf_decider.py:62-69`). `run_smoke.py:98-99` refuses GCG with any backend except `letter`, so the Q9-AF + head model is unattackable. Adaptive attacks are a stub (`safety_eval/adaptive.py:62-67`). So the G6 headline (`safety_eval/PREREG.md:182-203`) is unmeasurable on our candidate | code | 0.9 | The safety headline is missing at merge | $0: a head-aware GCG objective (gradient through `DecisionModel.forward` log-probs instead of LM letters); CPU test on the tiny Q9-AF stand-in |
| S5-F3 | GCG cost at 9B: about 4 s/step, worst case 10.6 min per attack, about 8 h for 45 attacks; peak about 64 GiB; spec estimate $25-40 at m2 (`jobs/clef_baseline/JOB_SPEC.md:83,103`) | spec | 0.4 | Budget | Early-stop statistics from Clef GCG runs |
| S5-F4 | Power assumptions are unmeasured (steer 0.5, discordance 0.10, `train/cohort.py:65-77`). Measured Qwen3-0.6B D1 steer is 0.87, and trained-head discordance may be ≫ 0.10, so G12 power at 1,800 requests may be < 0.84 | code | 0.4 | An inconclusive gate | $0: `steer_power` sweep (discordance 0.1-0.3, p 0.5-0.9) |
| S5-F5 | Certify (G4) cost: 13-80 decide calls per request at k=1/2, with escalations at the 256-call cap. It runs only for G16 cells | `safety_eval/certify.py:40-43,159-160,324` | 0.3 | Report-only | `n_erasure_sets` over real segment counts |
| S5-F6 | SAFETY_SPEC is not merged (PR #18 open), and neither are PR #20 and #22 (PREREG v0.4) | gh | 0.5 | Gates cite unmerged rules | Merge or relax per COORDINATION:118-136 |

**Owner:** D (Codex approves the margins). **Critical:** S5-F1 and F2 are on the merge path, not the compute path.

## 7. Stage S6: budget and time (credits expire 2026-10-27T06:53Z)

**Remaining:** about $510.5, of which $500 expires.

**Spend ladder:** each rung needs the previous receipt plus a `spend` issue acknowledged by the other agent (`COORDINATION.md:93-100`). The workspace cap is shared and has no owner (DECISION_TREE "workspace cap when jobs overlap").

The schedule is in §12. The budget table is below, with expected and worst-case costs; worst cases use the `costs.py` convention.

| Stage | Expected $ | Worst $ | Starts | Depends on |
|---|---|---|---|---|
| Clef-flash A S3-S5 re-run (after the P7 fix) | 1.5 | 5 | 10-07 | reader fix |
| Clef (27B) DEV, job C | 5.4 | 11.9 | 10-07 | reader fix, relaxed dependency |
| S1 pre-probe (stage 1, all-8 student eval, zero-attention retrieval, memory) | 3 | 5 | 10-07 | runner skeleton |
| S8 feature cache on stock Q9 (+ 1-layer Q9): 20k train rows + 2,008 DEV rows | 4 | 8 | 10-07 | cache script ($0) |
| S1 full-8 conversion | 28 | 74 | 10-08 | pre-probe, review, ack |
| S3b Clef-flash teacher logits on 150k | 12 | 20 | 10-09 | KD loss ($0), guard |
| Q9-AF parent arm + S2 smoke/ladder | 6 | 15 | 10-10 | S1 pass, GDN act-ckpt port |
| First Q9-AF cell (PR #74 shape) | 26 | 64 | 10-11 | smoke, #36 or user ruling |
| Scale-up cell (best proxy recipe, KD, r256 if it fits) | 60 | 120 | 10-15 | first cell |
| G6 GCG on the candidate (head-aware) | 25 | 40 | 10-18 | S5-F2 code |
| Full DEV paired: candidate, Clef, Clef-flash | 18 | 30 | 10-19 | scale-up |
| Full Index: candidate (shared prefix, 3 nodes), Clef, Clef-flash | 110 | 210 | 10-22 | PREREG frozen |
| Reproduction re-run (Codex, reserve) | 20 | 40 | 10-24 | full Index |
| **Total** | **~319** | **~643** | | |

**Verdict.** The expected path fits the ~$510 left. The worst case does not, and full-Index runs dominate it. Two mitigations:
1. Price shared-prefix scoring before 10-15.
2. Pre-decide what to drop if worst cases materialise, in this order: Clef full Index (use DEV + TEST paired only), then the GCG scale, then the reproduction re-run (Codex recomputes from raw predictions).

**Time is the binding constraint.** The chain from S1 through the full Index needs about 17 working days with no slack for a second S1 attempt.

## 8. Stage S7: coordination with Track L

| # | Failure | Evidence | P | Late cost | Probe |
|---|---|---|---|---|---|
| S7-F1 | **Spend deadlock.** Codex explicitly does not acknowledge PR #74's $70 job (#3), has not acknowledged #87, and has not answered #36. Spend needs an "other agent ack" (`COORDINATION.md:100,128`), while `:132-136` lets a fresh subagent review stand in. The rules contradict each other | gh | 0.5 | Days of slip on the critical path | $0: a `decision-needed` issue asking the user to rule on :128 vs :132-136 for spend |
| S7-F2 | Review debt: 30 open Codex PRs need D review (`COORDINATION.md:33`); many are stale past the ">2 sessions" rule (:27) | gh | 0.6 | D time diverted | Bulk triage per :27 |
| S7-F3 | Cap race: overlapping D/L jobs under one workspace cap | DECISION_TREE | 0.2 | A preflight failure or overspend | Name one cap owner; check `list_jobs` / `get_usage` before each submit |
| S7-F4 | Interface drift. L binds parent identity (#33, #79, #85) to a D parent that changes from RWKV7 to Q9-AF | #34 | 0.3 | Track L re-work, not D critical | Announce the Q9-AF parent revision pin when S1 lands |
| S7-F5 | The independent re-run is unfunded or unspecced (see S4-F9) | PLAN:63,72 | 0.5 | One-track-only claim | Spec m3 now |

## 9. Top 10 blockers (probability × cost of late discovery)

Cost of late discovery is scored 1-5: 5 = the goal becomes unreachable before 10-27; 3 = more than $50 or more than 3 days; 1 = cosmetic.

| Rank | Blocker | P | Late cost | Score | Cheapest probe now | Owner | Critical |
|---|---|---|---|---|---|---|---|
| 1 | **Head/recipe ceiling vs the same-backbone Clef-flash.** Unknown whether ANY of our attention-free heads on frozen Q9 states reaches ~56, before we pay for S1 + S2 | 0.6 | 5 | 3.0 | S8 proxy on cached frozen-Q9 features (~$4 cache, then CPU) | D | yes |
| 2 | Full Index = TEST unseal; no full-rows scoring path; candidate full Index ≈ 45 h / $134 on the current path | 1.0 | 3 | 3.0 | Full-rows CLI + selfcheck; price `shared_prefix` scoring ($0) | D | yes |
| 3 | Clef unmeasured: composite crash (reproduced, P7) + the job-C dependency | 1.0 | 3 | 3.0 | One-line fix + test (done as a probe) | D | yes |
| 4 | Q9-AF train step has no cache-fork-compatible checkpointing: OOM above T≈1.95k (P3, P4) | 0.7 | 4 | 2.8 | Port `act_ckpt` to HF GDN; CPU equivalence test | D | yes |
| 5 | Full-8 hidden-cos compounding → hybrid → no Q9-AF | 0.5 | 5 | 2.5 | Sum per-layer (1−cos) from art-2bybh ($0); ≤ $3 all-student eval | D | yes |
| 6 | Full-8 runner missing / no checkpoint export / deadline sizing (S1-F1..F3) | 0.5 | 4 | 2.0 | Shell dry run + `PHASE_HOOK` export ($0) | D | yes |
| 7 | Retrieval collapse with all attention gone | 0.45 | 4 | 1.8 | ≤ $3 zero-attention MQAR/S-NIAH | D | yes |
| 8 | Data covers 2/10 knowledge, 1/7 arts benchmarks; no KD pipeline | 0.7 | 2.5 | 1.75 | Per-benchmark gap decomposition from existing predictions ($0) | D | partly |
| 9 | Spend/margin deadlock with Track L (#3, #36, #87) | 0.5 | 3 | 1.5 | `decision-needed` to the user now | D+L | yes |
| 10 | Safety headline unmeasurable on the deployed model (no head-aware GCG; adaptive stub); exit 0 masks unapproved margins | 0.9 | 1.5 | 1.35 | Head-aware GCG CPU test on tiny Q9-AF; non-zero pending exit | D | merge path |

Near misses:
- v1 cohort-leak ambiguity (1,105 vs 37).
- S1-F8 path divergence.
- MMLU-noise quality-fail (S1-F5).
- Independent-reproduction identity (S7).

## 10. CPU probes run now (`tests/test_premortem_probes.py`; all pass, ~3 min CPU)

| Probe | What it checks | Result | Receipt |
|---|---|---|---|
| P1 | Q9-AF built by `q9af_from_stock` (tiny Qwen3.5, 4 layers, layer 3 converted) through `Trainer.train_step`: cache-fork `fork` head, Clef loss arm, LoRA r4 on the q9af spec groups, checkpoint + resume | **PASS.** LoRA reaches `layers.3.linear_attn.*` (the converted layer) with non-zero gradients; losses finite; resume restores step 3 | — |
| P2 | Cache-fork vs recompute-fork LoRA gradients on Q9-AF | **PASS.** 48 LoRA tensors, max relative error 9.4e-6, cos > 0.999 | `p2_grad_equivalence.json` |
| P3 | PR #86 levers on Q9-AF | **Blocker confirmed.** `act_ckpt` raises; `row_batch=2` falls back to per-row (no error, no gain); HF gradient checkpointing + cache fork raises | `p3_levers.json` |
| P4 | Saved-for-backward bytes per token per layer per d (cache-fork train step, LoRA on), tiny Q9-AF vs tiny RWKV7, d=64, T ∈ {128, 512} | Q9-AF/RWKV7 = **0.97**. Anchored on job-mden7 (1.0 MB/token/layer, RWKV7): Q9-AF ≈ 31 GB per 1k tokens; **max T without checkpointing ≈ 1.95k** (RWKV7 ≈ 0.76k, matching its observed OOM from ~1k). Caveats: CPU torch GDN fallback (fla saves less); head dim 8 vs 128 | `p4_activation_bytes.json` |
| P5 | Proxy exactness: cached-feature head training (`train/proxy.py`) vs `Trainer` at LoRA rank 0 | **PASS.** Head parameters equal (atol 1e-5) after 8 steps with warmup, accumulation 4, Clef loss; integrator and fork heads, fp32 and bf16 autocast. Without the matching autocast the bf16 case diverged by 0.018-0.035 (PR #91 review) | — |
| P6 | Proxy transfer: 6 recipes (head × loss arm × lr) ranked by proxy (head-only, cached) vs head + LoRA r4 | Kendall τ = **1.0**, argmin agrees, proxy train loop **47-58× faster** (two runs). Caveat: on a tiny random backbone LoRA moves dev NLL by < 0.002, so τ = 1 is a mechanism check, not 9B evidence | `p6_proxy_transfer.json` |
| P7 | Clef composite crash | **Reproduced.** One ` ` inside a JSON string written with `ensure_ascii=False` makes `summarize.rows` raise "Unterminated string" (job-sgihj's exact error); a `\n` split reads it correctly | `p7_composite_crash.json` |

## 11. Stage S8: fast-iteration proxy loop (cached frozen-backbone features)

**Design.**

1. **Feature cache (once, ≈ $4).**
   - Run frozen stock Qwen3.5-9B under `no_grad` over N ≈ 20k training rows (from the 150k set, guard-checked) and the 2,008-row DEV subset.
   - Store each row's `HeadInput` (`train/proxy.py: cache_items`), with `shared_prefix` for multi-question records.
   - Optionally repeat with the 1-layer-converted Q9 (art-2bybh) and later Q9-AF to measure the conversion gap at the head level.
   - Volume: ~12M tokens at ≥ 3.7k tok/s ≈ 55 min ≈ $3.
   - Storage per row: fork/final-state heads need only `fork` [K, D] + `q_end` + `e` (≈ 30 KB bf16), so 20k rows ≈ 0.6 GB. Heads that read every position (integrator, a Clef-style routing control) need `h` [T, D] ≈ 5 MB per row, so cache them for ≤ 5k rows (≈ 25 GB, on a node, not the laptop).
2. **Iteration (seconds to minutes, CPU or ≤ $1 GPU).** `train_head_cached` trains any head/loss/lr/warmup/data-mix/KD-α recipe. For LoRA rank 0 it is **exactly** the m2 `Trainer` update (P5), in fp32 and in bf16: with `bf16` set the proxy caches the features and runs the head under the Trainer's bf16 autocast (run_cell, m2_gates and cell_step pass `bf16=cuda`), and it refuses cached features whose precision or fork mode differ from the config. `eval_cached` gives dev NLL/accuracy. Scoring the DEV subset needs the cached Index rows plus the kit scorer (`harness score`), so Index points come for free per iteration.
3. **Questions it answers first, in order.**
   - (a) Does an attention-free head on frozen Q9 states reach Clef-flash's ~56 on the DEV subset? If no head reaches ~50, no Q9-AF variant will. That is blocker #1, decided for about $4 before S1's $28-74.
   - (b) Rank heads: fork vs HeadB vs snapshot vs fork + field-mixer vs the `ClefCrossAttentionControl` upper reference.
   - (c) Loss: `ce` vs `clef` (± RPS) vs KD from Clef-flash soft labels (S3b).
   - (d) Data mix per area / benchmark coverage.
   - (e) An RLCD-like stage (S2c).
4. **Hold-out discipline.** Hill-climb on cohort dev plus a fixed half of the DEV subset. Report the other half untouched. Never touch TEST.

**Why proxy gains should transfer, and where they stop.**
- *Exact part:* for head-only recipes the proxy is the real training update (P5). Every head, loss, lr, data or KD decision is measured, not approximated, on the same frozen backbone.
- *Empirical part, the LoRA term:* Clef-flash's own recipe is frozen backbone + head + LoRA, and its head + LoRA moved Q9 from 38.3 to 55.8. The proxy measures the head share of that gain. Transfer to head + LoRA is a measurement: P6 runs it (Kendall τ between proxy and head + LoRA recipe rankings) and found τ = 1.0 on a tiny model.
  - At 9B, calibrate with **one** pair of real cells (the proxy's best and worst recipe, ≤ $10 each, short ladder) before trusting the ranking. Adopt the proxy ranking only if τ ≥ 0.6 over ≥ 4 recipes and argmin agrees.
- *Backbone transfer (frozen Q9 → Q9-AF):* the 1-layer-converted cache (art-2bybh) gives a second point. If a recipe's rank holds between stock and 1-layer features, it should hold on Q9-AF, whose states differ more. Re-check the top-2 recipes on Q9-AF features (one more ~$3 cache) before the scale-up cell.
- *Known blind spots:* recipes whose benefit comes from adapting the backbone (long-context recall, retrieval through LoRA) and from length > the cached T. These are measured only in S2.

## 12. Critical-path schedule (dates UTC; ★ = critical path)

| Dates | Work (parallel lanes) | $ expected |
|---|---|---|
| 10-07 | ★ $0: P7 reader fix + test; full-8 runner + `PHASE_HOOK` export + resume; GDN `act_ckpt` port (CPU equivalence); full-rows scoring path + unseal-once refusal; pending-margins exit; `decision-needed` on spend-ack rules and #36. Download art-8y6d8, pin sha. Commit v2 data. Lanes: Clef-flash S3-S5 re-run, **Clef DEV (job C)**, S1 pre-probe, S8 feature cache | ~14 |
| 10-07 → 10-08 | ★ S8 proxy questions (a)-(c) on CPU. **Go/no-go:** if no attention-free head on frozen Q9 reaches ~50, escalate to the user before S1 spend (report the measured head ceiling) | 0 |
| 10-08 → 10-09 | ★ S1 full-8 conversion (~10 h). In parallel: KD loss + teacher cache job; head-aware GCG code | 28 + 12 |
| 10-10 | ★ Q9-AF parent arm + S2 smoke/ladder (with GDN act-ckpt); proxy re-check on Q9-AF features | 9 |
| 10-11 → 10-13 | ★ First Q9-AF cell (best proxy recipe; G12 in-job) | 26 |
| 10-14 | Proxy-transfer calibration pair (if not covered by the first cell); S2c RL proxy | ≤ 10 |
| 10-15 → 10-18 | ★ Scale-up cell (KD, r128-256 if memory allows), G12/G13; G6 GCG on the candidate | 85 |
| 10-19 → 10-21 | ★ Full DEV paired: candidate vs Clef vs Clef-flash; ECE; latency-v1 sampler; $/1k decisions | 18 |
| 10-22 | ★ PREREG frozen (model sha, rows, seeds); the user approves the full-Index run = TEST unseal | 0 |
| 10-22 → 10-25 | ★ Full Index: candidate (shared prefix, multiple nodes), Clef, Clef-flash; paired CIs | 110 |
| 10-25 → 10-26 | Codex reproduction (re-score from raw; reserve re-run); Index PR drafted, **not submitted without the user's explicit approval** | 20 |
| 10-27 06:53Z | Promo credits expire | |

**Slack:** about 1 day. A second S1 attempt, or a failed first cell, consumes it. The fallback decision (RWKV7 or a hybrid; `decision-needed`) must be taken by **10-12** at the latest to keep the full-Index window.
