# Conclusions
1. **The project bought evidence and infrastructure, not a competitive model:** none of the capability targets was met; JEV3-LM-S is promising only as an invariant, statistically inconclusive near-stock model. [L][E]
2. **$785 was granted, not spent:** the final $11.67 balance implies **$773.33 consumed**; this audit attributes **$675.47**, leaving **$97.86 explicitly unallocated**. [L][B]
3. **The state ledger alone accounts for $593.38 across 98 charged jobs** after removing holds and duplicate costs; summing its numeric cost column gives the misleading $870.67. [L]
4. **The documented RWKV7 line consumed $492.90**, versus $68.48 for JevBench Qwen cells; the large attention-free stack was economically misaligned with the later Composite-first objective. [L][B][G]
5. **At least $92.84 across 30 jobs and two CPU nodes went to engineering failures or invalid measurements**, but valid scientific negatives cost much more: $330.41. [L][B]
6. **Integrity controls earned their place:** audits caught training/evaluation leakage, vacuous perturbations, tokenizer leakage and invalid independence assumptions; arithmetic rescores mostly agreed. [P][E]
7. **The main scarce resource was informative experimental cycles:** repeated runner failures, premature parallel expansion and a secondary language program competed with capability work. [P][PM]
8. **Keep jevkit, typed serving, provenance/leak gates, invariant readouts, saved predictions/checkpoints and scoped safety constructions; freeze Track L, Q9-AF and further RWKV13.3B spending.** [D][E]
9. **The strongest next capability experiment is a properly budgeted full-epoch JEV3-LM with shimcal declared primary**, not another readout selected after examining the same 231 public items. [P][E]
10. **Use the sealed 2,199-item jev-heldout set for one-shot secondary confirmation**, preserve its access restrictions, and reserve replication funding before starting another exploratory campaign. [H]

## 1. Scope, evidence and accounting

This is a read-only post-mortem of the initial October 6–13 campaign, researched October 14, 2026. It covers the requested state documents, repository lessons/premortem/candidates/rules, evidence report and implementation sections in `jevkit/`, `decide/`, `heads/`, `train/`, `controlsafe/`, `safety_eval/`, `data/` and `jobs/common/`. Other branches were inspected through Git, without checking them out. Only metadata—not sealed item contents—was read for jev-heldout.

**Evidence labels:** **MEAS** means a measurement; historical model/test measurements below are **source-reported**, not rerun here. **EXT** denotes an extrapolation. **SIM** denotes a historical simulation. **[INFERENCE]** denotes this report’s interpretation or proposal. **[UNVERIFIED]** marks missing evidence. No model execution, tests, build, paid API, GPU job, weight download, commit or push was run in this audit. The computations actually performed were lightweight ledger/history parsing and arithmetic.

### 1.1 Reconciliation method

The state ledger has **188 data rows**, not 188 independent experiments. I parsed it with Python’s CSV reader and decimal arithmetic. One malformed row, line 118 (`job-862xw`), contains 15 fields; its trailing status/receipt/note were realigned **in memory only**. Its approximately $0.35 cost remains outside exact totals because the cost field is empty and PROGRESS later says approximately $0.40. Missing prices are never interpreted as zero. [L][P]

Rules applied:

- Exclude every `submitted` reservation. **Ten numeric holds total $196.42**; additional holds appear only in notes and were never charges.
- Count each paid job once. D1’s $19.38 correction and D12f/D10w’s $26.20/$27.88 rescores repeat original costs. D1’s readout job `39gnn` has a provisional $8.02 and a later receipted $10.25: retain $10.25, not both.
- Treat `fgu2c`’s eight $1.54 allocations as **one $12.35 job**, using the note’s total; their rounded allocation sum is $12.32.
- Count separately named shards/CPU jobs, but do not split their aggregate charge into invented individual prices. Short forms such as `job-kxg97+2jdw9` identify two jobs.
- Replace `4g6m8`’s `~0.6` with its archived provider receipt, **$0.5796**. Keep other ledger costs at their recorded precision; this is not a cent-perfect invoice reconstruction. [L][R]

The arithmetic is:

`870.672 − 196.42 − 81.48 + 0.03 = 592.802`

`592.802 + 0.5796 = 593.3816`.

The supplemental sources add only jobs absent from that priced set:

| Reconciliation component | USD | Basis |
|---|---:|---|
| Deduplicated state-ledger charges | **593.3816** | 98 distinct charged jobs; includes resolved `4g6m8` |
| Nine missing individually receipted early jobs | 12.7356 | `4j3vu`, `5cmat`, `5mw9h`, `c4zkm`, `d2u54`, `jbw22`, `jxevr`, `mhc3h`, `tkfaj` [B] |
| Six further historical jobs | 5.4700 | `sgihj`, `4idg4`, `jucxx`, `jn47u`, `mden7`, `tvuat`; rounded actuals in old LEDGER [O] |
| D11, D7 and three D1 full-DEV attempts | 47.4432 | $20.8376 + $9.6012 + $17.0044 in provider batch usage [B][P] |
| Four early CPU nodes | 0.4530 | Metered usage, not immediate balance changes [B] |
| Captured build charges | 15.984614 | October 10 usage snapshot [B] |
| **Documented total, rounded** | **675.47** | 118 charged jobs, four nodes, plus captured builds |
| **Unallocated residual** | **97.86** | Missing/later charges, additional build costs and settlement/rounding differences; not assigned to a lane |
| **Implied campaign consumption** | **773.33** | **[INFERENCE]** $785 grants minus last reported $11.67 balance |

The archived October 10 usage snapshot is itself incomplete for attribution: **54 jobs/$97.7353** are grouped under an unlabeled remainder. Its total is $530.028114 at 04:13Z; billing reports $531.132647 at 04:33Z. These are different capture times, not evidence of a $1.10 accounting error. There is no final all-job settlement export in the inspected history. The $97.86 residual is **not** identified with that older $97.7353 remainder. [B]

### 1.2 Spend by research line and outcome

These are **documented charges**, not fabricated allocations of the residual. Line assignment and outcome classification are **[INFERENCE]** from recipes and terminal notes. A useful baseline or engineering measurement counts as a “usable result”; a usable result is not necessarily a benchmark win. “Scientific negative/non-positive” includes valid inconclusive results, not proof of zero effect. Infrastructure includes engineering-induced aborts, preemption/cancellation without the planned result and void measurement procedures.

| Line, including its evaluations | State-ledger USD | Documented USD incl. supplements | Charged jobs |
|---|---:|---:|---:|
| Clef-era RWKV7-13.3B cells, readouts and evaluations | 439.03 | **492.90** | **49** |
| Q9-AF conversions, qualification and recall probe | 62.25 | **70.55** | **7** |
| JevBench Qwen cells: size ladder, 4B custom cells and H2O-host training ablations | 68.48 | **68.48** | **19** |
| Clef/H2O baselines, host scans, comparator and parity evaluations | 22.3896 | **25.88** | **17** |
| CPU integrity/data/control work | 1.232 | **1.685** | **26**, plus four nodes |
| **Jobs/nodes subtotal** | **593.3816** | **659.4834** | **118**, plus four nodes |

Baseline rows include small QRWKV/Qwen scouts; mixed one-load jobs are allocated once, not fractionally guessed. H2O *training* is in the Qwen-cell row, not hidden among baseline evaluations.

| Outcome | State-ledger USD / jobs | All documented USD / jobs and nodes |
|---|---:|---:|
| Usable result/reference/engineering evidence | 222.412 / 50 | **236.2362 / 56 jobs + 2 nodes** |
| Scientific negative or non-positive result | 330.4096 / 32 | **330.4096 / 32 jobs** |
| Infrastructure failure/invalid measurement | 40.56 / 16 | **92.8376 / 30 jobs + 2 nodes** |
| Build charges without outcome attribution | — | **15.984614** |
| Residual without line/outcome attribution | — | **97.861986** |

Classification edge cases matter. JEV3-SYM’s $3.42 training job ended with a loader error, but its checkpoint was recovered and evaluated for $2.16: both are charged to its scientific negative, **not** to irrecoverable infrastructure waste. Conversely, the $2.66 first flip battery is classed as invalid measurement even though some non-order operations remained usable. D1’s early hijack runs retain narrow in-distribution diagnostic value; their retired steer claims do not become general safety evidence. [L][D]

### 1.3 Time accounting and activity

The provider snapshot already records **8,778.78 H100-minutes = 146.31 GPU-hours across 78 jobs**, plus **1,682 billed build minutes**, by October 10. Those are overlapping machine-hours, not elapsed project hours; final machine-hours are unknown. [B]

On result-entry dates, deduplicated state-ledger costs are October 7 **$30.03**, October 8 **$59.98**, October 9 **$282.54**, October 10 **$180.62**, October 11 **$26.7716**, October 12 **$13.44**. These are **not daily provider charges**: completion and rescore dates differ from runtime, and the ledger misses jobs. In particular, the context’s approximate daily spend series must not be presented as reconciled accounting.

Git history inspection found **2,072 distinct reachable commits**, **372 remote branches excluding the HEAD alias**: 153 `claude/`, 117 `codex/`, 98 `stack/`, four integration/other. There are 95 “Merge pull request” commits and 16 squash-style subjects; the highest matched PR number is 247. This does **not** establish 247 merged PRs. **320 commits touch language-growth/Codex-plan/interface paths**, including shared integration history. Namespace and path counts are not person-hours or productivity rankings.

PROGRESS reports roughly **20 hours** spent on validation/Stage 2 before the first resubmitted training result. Later entries repeatedly document low-memory admission failures, stale job-state coordination and source-audit work while Stage 1 remained unqualified. **[INFERENCE]** The activity-to-learning ratio, not GPU price alone, is the operating problem. Agent/API labor costs and exact human/agent hours are **unknown**. [P][PM]

## 2. What failed, and what the controls bought

### 2.1 Taxonomy

**Engineering/infrastructure:** shell and multiprocessing paths unexercised by unit tests; cache copies/leaks; model-load/JIT time counted inconsistently; missing stage caps; false lineage mismatches; platform artifact limits; missing early checkpoint export; fleet worker loss; and perturbation serialization that turned reorder tests into no-ops. These are not evidence against model hypotheses. [L][LS]

**Scientific negatives:** constant-LR continuation degraded models; plain Clef-flash KD did not improve D1; full attention removal destroyed recall despite acceptable KL; data improvements on the internal slice failed to transfer; and a symmetric fork preserved invariance while producing inadequate probabilities/rankings. These are real findings, but only for the tested recipes. [D][E]

**Measurement/interpretation:** single-seed uncertainty; only 231 public JevBench items; calibration and Composite discontinuities; inappropriate per-decision independence; changes in benchmark edition; latency variation; post-hoc family narratives; and conflating API parity, arithmetic parity and board-scale comparability. [E][J]

### 2.2 Ten largest money/time sinks

Rows overlap deliberately; this is a prioritization table, **not an additive budget**. A sink can be a valuable negative result whose repetition should stop.

| Sink | Quantified evidence and examples | Decision |
|---|---|---|
| **1. Scaling the wrong RWKV recipe before stabilizing the learning loop** | D10, D10L, D1-fla, D12f and KD cost **$140.43/5 jobs**. D10 fell to 3.61; D1-fla to 28.46 versus D1’s 35.12; KD 34.65 versus 35.12. [L][D] | Freeze these recipes; use matched steps, decay and slice-selected checkpoints before scaling. |
| **2. All-attention-removal before a credible task anchor** | Two full conversion attempts cost **$54.41**; clean v2 used 196M distillation tokens and moved proxy Index **35.82→14.05**. The $7.84 single-layer successor probe did not test compounded conversion. [L][D] | Freeze Q9-AF; do not mistake good KL or isolated layers for retained decision competence. |
| **3. Losing expensive r256 outputs** | D5’s artifact exceeded 5 GB; D11 was canceled for the same export risk: **$43.90 across two jobs**. [L][B][P] | Inference-only, size-bounded exports must happen when checkpoints are saved. |
| **4. Full-DEV infrastructure and premature expensive evaluation** | D1’s three attempts cost **$17.00**, with lineage, heartbeat and worker-disk problems. D13/D19 final evaluations cost **$54.25** and established large losses. [B][L] | Preserve those predictions; require a credible cheap screen before another full-DEV candidate. |
| **5. Bespoke fork/cache performance engineering** | Initial smoke OOM $3.69; size-cliff serialization/OOM/probe failures **$2.30**; earlier many-option cache copies produced asymmetric errors. [L][LS L-21, L-34] | Exercise the actual tiny-tokenizer/multiprocess/checkpoint path; stop estimating hybrid KV memory as constant RWKV state. |
| **6. Runner failures consuming the final capability experiment** | JEV3-LM lost **$4.35**, reaching 497/729 steps with no exported weights; the clock omitted 11.6 pretrain minutes and a head-inappropriate guard halved LR three times. [L] | Treat the full-epoch hypothesis as open; separate artifact survival from training/evaluation exit status. |
| **7. Increasing data volume without transfer** | Qwen-cell non-positive results account for **$57.70**. H2O-host scaling cost $12.05, raised slice accuracy +4.44/+8.97 pp, but public Composite moved −0.41/−0.97. [L][E] | Improve gold conventions and task distributions before buying more of the same data. |
| **8. Low-ceiling calibration/readout repairs** | RAGTruth diagnosis/rephrase jobs cost **$6.72 known**, plus the unpriced ~$0.35–0.40 lineage abort; ranking improved but final Index gains were 0 and +0.127 with CI touching zero. D19 temperature fit/evaluation lowered JevBench Composite. [L][D] | Keep generic readout discipline, retire benchmark-specific repairs with no claimable outcome. |
| **9. Gates whose operational design manufactured failures** | The production path itself violated the early kernel reference; compilation spikes triggered rate guards; cache-directory names triggered false lineage rejection; first flip files voided reorder operations. [LS L-06, L-26, L-28, L-29][L] | Fix the reference and execution semantics, not thresholds after results. |
| **10. Track L and coordination expansion before a qualified parent** | Historical transfer failures include 100/300, 0/100 and 29/360 comparisons; Stage 2 had no qualified real parent. Hundreds of path-touch commits and repeated admission/status exchanges produced no qualified language level. [D][P][G] | Archive the deliverables and freeze active language development; preserve reusable typed/provenance components. |

### 2.3 Controls with demonstrated value

**Leak audits materially changed the evidence.** The attack audit found all four D1-A steer templates verbatim in training—roughly 1,240 rows—retiring the headline safety claim. Subsequent heldout_v1 evidence supported D13 over Clef-flash: hijack **0.025 vs 0.115**, steer **0.003 vs 0.089**, with the four primary contrasts Holm-adjusted; superiority to Clef 27B was not established. Document-level checks caught NLI4CT shared clinical reports, copied MMLU/MMLU-Pro items in conversion data, ANLI heldout-pool copies and short items missed by long n-grams. The latest jev-heldout build rejected about **26%** of candidates. Keep provenance, source-item and state-level exclusion—not just exact row IDs. [L][D][H]

**Independent review prevented more serious errors than arithmetic rescores did.** It found G15’s real-BPE leakage, G16 duplicate/order dependence and `sort_keys` perturbation no-ops. Reanalyzing dependence changed D19’s apparent marginal actionability advantage into **zero cluster-valid certification**. The six-family “66–81% of the gap” narrative was withdrawn after split-half scrutiny. These are genuine integrity returns. [E §§5–7][P]

**Preregistration resisted convenient reinterpretation.** It kept the failed torchcast parity result on the record; rejected the lean size-cliff template after Intelligence fell 1.3 points against a 1-point threshold; retained the FLA inference rejection despite a nearly neutral Index point estimate; and did not promote secondary shimcal results to primary wins. It also exposed a weakness: the JEV3-LM noul-temperature fallback was specified only in code, not prose. Keep concise claims, actual decision rules, fallbacks and multiplicity explicit. [L][P][PR]

**What had low demonstrated yield:** later numerical rescores matched 67–235 fields essentially exactly, while large agent-written analyses and interpretations still needed correction. Incompatible test environments made one combined gate impossible; broad shell-text assertions went stale; excessive per-element numerical gates rejected production behavior; load-path identity blocked identical weights. These controls cost time without detecting the intended model defect. **[INFERENCE]** Automate routine replay/hash comparison once; spend independent attention on leakage, non-vacuity, assumptions and interpretation. Do not delete all review because arithmetic usually agreed. [LS L-24, L-28][E §7]

## 3. Established results versus overclaims

| Claim | Credible conclusion | What remains unsupported |
|---|---|---|
| Competitive capability | D19 full DEV **49.49**, versus Clef-flash **57.29** and Clef **61.89**; paired losses exclude zero. Best own stock 4B shim trails H2O locally by **9.4 Composite**. [L][E] | No board-scale win; no theorem that future SSDM architectures cannot win. |
| JEV3-LM-S matches stock | Primary **−0.78 [−16.36,+3.47]**; no gain claimed. [L] | “Equivalent” or “non-inferior” is too strong: the interval permits a large regression. |
| Exact invariance | Historical GPU evidence: **0/565** reorder/relabel flips for SYM and LM-S; LM-S also has a separate score-item gate. Constructive invariance is reusable. [L][E] | Not paraphrase invariance, answer correctness or universal floating-point batch invariance; option-description/tie assumptions apply. |
| Better general robustness | D13 beats Clef-flash on the specific heldout_v1 template battery. [L] | Universal prompt-injection safety or superiority over Clef 27B. LM-S’s **3.64% vs 7.55%** average flips is unpaired across jobs, not an MD1-like-for-like comparison. |
| Guaranteed abstention | Conformal sets have scoped exchangeability-based guarantees; empirical LM-S coverage was **0.916 at α=.10**. [L][E] | No LTT abstention certificate for own stacks in the cited tests. A larger set and fewer actions can improve coverage without making decisions more useful. |
| State-space safety | GDN’s local contraction result is supported under its assumptions; protected information flow has a real construction. [E][CS] | Global hybrid-model stability, semantic safety from contraction, or shipped bf16 RWKV7 certification. Dense Gramians are explicitly unscalable at production state size. |
| Clean data | Latest draws passed the implemented independent gates; receipts preserve exclusions/provenance. [E][H] | Absolute absence of contamination. Earlier draws were corrected, and 49 slice rows shared a training state in one H2O-host run. |
| Data is the only bottleneck | H2O-host slice gains did not transfer. [E] | On SYM, probability readout explains much of the loss, with real score/routing losses remaining. Architecture, readout and data effects were not fully separated. |

The public set has **139 choice, 74 noul and only 18 score items**, including six hard score items carrying approximately 22% of the Intelligence weighting. The bootstrap is therefore highly sensitive to a few outcomes. JEV1’s primary interval spans **−25.64 to +3.22**; 41.9% of its resamples cross the Intelligence floor, producing an asymmetric Composite tail. A single “±8” shorthand is not a reliable universal error bar. [E][H]

Paired item bootstraps condition on one trained checkpoint. They do not measure optimizer/seed variance. Recorded run-to-run nondeterminism and roughly 2.7 Speed-point variation for the stock RWKV measurement reinforce that limitation. Nor does repeated use of the same public set remain confirmatory simply because each new experiment receives a fresh preregistration. [P][E]

The external benchmark’s current README independently warns against comparing method versions or treating leaderboard performance as workload reliability. It identifies v1.6.1’s sealed-input and modeled-cost limitations. The project’s own public-231 results must remain separate from the board’s 300-open/1,200-sealed draw and its 72.5 target. [J]

## 4. Reusable assets and retirement decisions

### 4.1 Module inventory

“Maturity” below is a deployment judgment **[INFERENCE]**. Test status means existing files or historical receipts; **tests were not run here**. Important later functionality lives on experiment branches, not necessarily on checked-out `agent-main`.

| Asset | Maturity and evidence | Keep/use boundary |
|---|---|---|
| **`jevkit/`**: specification, scoring, meter, adapters, perturbations | Strong research infrastructure: historical board arithmetic max error ~1.4e−14; three public-card systems reproduced within declared tolerance. `jevkit/tests/` exists. Later perturbation/cluster fixes have separate branch provenance. [K][E] | Keep frozen arithmetic/pins, paired scoring and raw outputs. Consolidate reviewed fixes before a new campaign; never conflate board arithmetic with sealed-model parity. |
| **`decide/`**: schema, provenance, trust boundaries, typed outputs | Usable API layer with conformance/provenance/error-boundary tests. `schema.py` expressly distinguishes sorted-ID reorder invariance from stronger relabel invariance. [DS] | Keep typed probabilities, failure signaling, host provenance and policy boundaries. Canonical LM serving belongs to the newer LM lineage. |
| **`heads/`**: fork/typed/layout/activation checkpointing | Research-grade, with trained fork/joint descendants and equivalence tests. Current integrator/snapshot/HeadB implementations consume full feature tensors; bounded accumulators do **not** mean implemented streaming. Clef control is shape-matched, not a faithful recipe replica. [HD] | Keep stable feature/layout contracts and exact cache fixes; freeze unused head variants until an ablation needs one. |
| **`train/`**: LoRA, losses, calibration, lineage, exports, guards | Reused in actual training, but cross-mode runner contracts remained brittle through October 12. Later `a8bca917` LM package includes head-aware guard/best-export fixes. [L][TC] | Keep core training and guard logic. Reuse the corrected cell lineage, not an old integration-branch runner by assumption. |
| **Cached proxy heads** (`train/proxy_heads.py`) | Existing LM-logprob, lexical-prior, routing and field-mixing features; `no_lm` and other ablations already implemented. Frozen-Q9 campaign’s best SELECT was 39.24. [PH][D] | Keep as a cheap screen, not a frozen-feature ceiling on LoRA-adapted heads. Do not rebuild LM pooling as if absent. |
| **νPI/consistency branches** | Historical νPI job `urjpw`: reported 13/13 after float-operation-order fix; OFF-path and checkpoint round-trip evidence. Consistency suite reported 13/13, but νPI was never used in a paid training ablation. [L] | Keep CPU-tested mechanism; label control-of-learning efficacy open. |
| **`controlsafe/`**: G15/G16, snapshots, persistence | CPU/tiny-model structural machinery; later real-BPE G15 receipt 1000/1000; duplicate/batch effects required fixes. No trained G16 capability result. [CS][E] | Keep scoped protected-stream/veto/interval construction. Freeze dense Gramians and weak G14 detector as research, not product claims. |
| **`safety_eval/`**: attacks, paired metrics, certification | Useful heldout_v1 measurements and covered-insertion certificates. Exact erasure cost grows combinatorially; certified means unchanged under covered attacks, not correct. [SC][L] | Keep frozen threat models, controls and escalation semantics; do not pay for broad batteries before a candidate survives capability screening. |
| **`data/` plus training gates** | High-value builders, augmentation, fingerprints, provenance and exclusion manifests; repeatedly improved after real leaks. [DB][E] | Keep final gated draws and generators; quarantine superseded pools. Add state/source-group deduplication at build time. |
| **`jobs/common/`** | Useful SHA-pinned packaging, rates, size limits, strict result readers and receipts. Legacy submit/context maps, mode-specific caps and shell staging increased failure surface. [JC][LS] | Keep a thin provider adapter and immutable manifest; retire redundant mode wiring rather than another runner framework. |

### 4.2 Data, checkpoints and proofs

| Reusable object | Reference and current value | Qualification |
|---|---|---|
| **Index-directed data** | D19 v3b **253,046 rows**; D20 v4 **258,798**. [L] | Useful for reproducibility/Index research, not established JevBench-transfer data. Preserve corrected exclusions. |
| **jev1 / jev2** | jev1 **96,000** rows; jev2 combined **16,500**; mixed pointer `71b503eb` **100,500**, with balanced cell subsampling. [L][P] | Build assets, not a validated general capability curriculum. Old jev2a/c versions are superseded. |
| **jev3** | **5,960 train + 1,200 slice** before sampler/length exclusions; LM-S uses **5,836** training rows. [L][E] | Best current human-rubric foundation; keep provenance and exclusions rather than interpreting every source label as JevBench gold. |
| **jev-heldout** | **2,199** sealed items; items hash `17a40d43…`, IDs `c175d8a0…`; branch `claude/ev-jev-heldout@668b7472`. [H] | Secondary confirmation only; at most one scoring per preregistered cell, access logged. |
| **FS-CRR unseen pool sample** | **10,365 rows / 6,780 cases**, IDs `990b5911…`; `ev-guar-fscrr@4a5046c0`. [E] | No predictions; roughly $35 historical estimate requires comparator work currently prohibited unless explicitly authorized. |
| **D1 / D13 / D19 artifacts** | D1 `art-bm3e6`, step 1439; D13 `art-cr9a2`, best1600 `61c19d16…`; D19 `art-p3gpi`, best2800 `31062f48…`; full-DEV predictions `art-utxjp+art-6qppy`. [L] | Keep immutable research controls/predictions. D1 has contamination caveats; none is a target-winning release. |
| **JEV3-SYM / LM-S** | SYM `art-8xs32`, best200 `5cb7da42…`, eval `art-xkhwy`; LM-S `art-mddiz`, checkpoint150 `da7276b0…`, export `8e936008…`. [L] | Highest-value invariant trained artifacts; availability of remotely retained bytes was not checked. |
| **Other landed artifacts** | JEV1 `art-mv9cr`; H2O-host `art-cwk49`/`art-5si2g`; D20 link 1 recorded as retained, but no final screen. [L][B0] | Archive as controls/negative evidence. Do not resume D20 merely because its first link is sunk cost. |
| **Proofs and adversarial counterexamples** | `ev-guarantee-proofs`: reorder, conformal, injection-robust conformal and review; `ev-review-stability@e89b29cc`; typed score-op `ev-score-inv@c47c0a46`. [E] | Keep assumptions, counterexamples and tests together. These are conditional mathematical assets, not certified real-world alignment. |

**Freeze versus delete.** Freeze active Track L, Q9-AF v1/v2/v3 scaling, RWKV13.3B D20/D21 training, historical H2O/Clef engineering and benchmark-specific readout branches. Preserve receipts, weights and negative results. **[INFERENCE]** In a future authorized cleanup, delete obsolete runtime aliases, repeated submit/context tables, stale shell-order tests and superseded live candidate/status entries after callers migrate; do not destroy evidence or branches as part of this read-only assignment. [D][JC][G]

## 5. Best hypotheses still standing

Costs are **EXT planning ranges**, not quotes or spending authorization. Historical H100 rates were $0.063/min, with a Friday discount; build/restart reservations are not actual charges. Cost/hold must be repriced before submission. [O][LS L-49]

| Priority / hypothesis | Evidence for and against | Minimal decision and cost |
|---|---|---|
| **1. Full-epoch JEV3-LM; shimcal primary** | LM-S preserves exact invariance; secondary shimcal **+3.48 [−1.09,+12.56]** versus stock and better slice NLL motivate it. Against: primary remains inconclusive, score/routing weaknesses persist, and one epoch might overfit. Full-epoch run never completed. [L][P] | Freeze readout/checkpoint rule before fresh predictions; compare against matched custom/control paths, then sealed confirmation. **~$6–9 actual / $20–25 hold per seed**. At 5.07 s/step, 729 steps require **61.6 train minutes**, not the old 42-minute clock; include 11.6-minute pretrain overhead and screen. The ~$17 LM-400 estimate is not a measured full-epoch budget. |
| **2. Trained G16 with actual utility measurement** | Protected-stream intervals give a genuine construction, and tokenizer/duplicate bugs now have counterexamples. Against: no trained result; shrinking evidence influence can make safety trivial and utility poor. [CS][E] | Train custom 4B judge; measure clean capability, coverage, set size, certified useful actions and held-out attack utility against an ungated arm. Historical **~$6 expected / $21.67 hold** is a **single-cell analogy**; allow roughly twice that for a fresh matched pair, with overhead uncertain. |
| **3. νPI ON/OFF on a working base** | OFF-path, multiplier serialization and controller identities have historical CPU evidence. Against: no trained on/off result; SYM’s fixed λ is not evidence for νPI. [L][E] | Matched data, steps, seeds and losses except controller. Report constraint violation and capability, not multiplier plots alone. Historical **~$12 actual / $43.34 hold** for two short arms; full-epoch pair could cost more. |
| **4. Anchor-then-ablate a Clef-shaped head on stock Qwen3.5** | The October 6 premortem already identified same-backbone recipe/head ceiling as blocker #1. Cloudflare confirms frozen Qwen + routing + LoRA + LS-CE/Brier. Against: private training data/RLCD details remain unknown; SSDM’s current control is not an exact Clef head, and frozen features topped out at 39.24. [PM][CF][HD][D] | Build **our custom hybrid-stack anchor**, not new third-party benchmarking; then change one component. **~$10–30 actual / $30–80 reserve** for a small 4B pilot/pair is an EXT range anchored to $2.63–12.43 historical 4B cells, not a profiled promise. Profile before choosing r256. |
| **5. LM-logprob pooling inside the trained head** | Existing proxy mixer already implements a gated LM pool. Earlier fixed fusion was **+1.27 [−0.38,+2.83]**, in-sample; some stock capabilities complement the specialized head. Against: frozen-feature campaign was negative; gains are not additive, and extra passes may erase benefit. [PH][T][D] | First reuse eligible train-side cached features for a $0 local CPU ablation; never fit a gate on public evaluation outputs. Then an end-to-end `no_lm` pair on a successful anchor, **~$6–18 actual / $20–50 hold** for short 4B cells, conditional on shared features. |

These are hypotheses, not expected benchmark gains. The first is the most direct capability test; G16 and νPI are differentiator tests whose useful outcome can be a measured cost/benefit trade-off rather than leaderboard leadership. **[INFERENCE]** Do not launch all five concurrently. [E][PM]

## 6. Faster and cheaper without losing integrity

### 6.1 A cheap inner loop

**[INFERENCE; motivated by runner losses, transfer failures and late-stage budget starvation above]** Replace “one idea → full branch/runner → parallel jobs” with a small, fixed experiment contract:

1. **One immutable cell record:** data/source-group hashes, actual rendered-token limits, model/tokenizer/head/readout pins, selected checkpoint rule, seed, primary metric, fallbacks, runtime estimate and maximum financial exposure. Generate the live board from this record and terminal receipts. Keep claims separate from invoices. [L][JC]
2. **CPU admission on the real production path:** tiny weights plus the real tokenizer, longest rows, multiprocessing, checkpoint save/load, exported inference reload and a complete mocked transport-free job stage transition. These checks target actual bugs; a large count of unrelated unit tests is not admission evidence. [L rows 82–89, 175–183][LS]
3. **One rate/memory canary per execution shape**, then reuse its measured steady-state rates for unchanged shapes. Separate setup/JIT/tokenization, training and screen budgets. Save a bounded inference artifact independently of subsequent evaluation success. [LS L-26/L-29][L]
4. **Explore only on train-derived, source-group-disjoint development slices.** Before public evaluation, inspect overall loss, hard score/noul behavior, confidence-band rates, slice transfer proxies and exact structural checks. Frozen feature caches shortlist heads; end-to-end LoRA resolves feature adaptation. [P][PH]
5. **One short cell before replication/full epoch.** EXT: a **200-step development cell around $3–5** is grounded in LM-S’s $2.63 combined run, not guaranteed for G16 or long-context heads. Do not spend this cheaply and then declare success from a noisy public slice. [L]
6. **Reserve the confirmation and second seed before exploration.** Use the small cell to reject grossly bad recipes; only finalists receive a full epoch and sealed confirmation. Initial scheduling should be one experiment plus its matched control, not several unvalidated descendants of a changing runner. [P][E]

**Keep** leak audits, scoped threats, pinned scorers, independent review of novel invariance/certification logic, and a single automated rescore from saved raw predictions. **Simplify** repeated prose preregistrations, repeated ad hoc landing scripts, unconditional full-DEV scoring and duplicated package registries. **Delete from the critical path** Track L qualification work, old calibration repairs and negative-architecture replications without new mechanism-level evidence. These changes follow the observed error distribution rather than a generic “less process” slogan. [LS][D][E]

### 6.2 Evaluation power plan

Use **jev-heldout’s actual 2,199**, not “about 2,000,” but do not oversell it. It contains **1,752 human-labelled items**, **333 rule-engine items** and **114 synthetic Glaive calls**; tiers are assigned by label-blind hash, not real difficulty. Some generators/templates recur, and labels follow the source datasets rather than JevBench’s conventions. It is a useful internal confirmation set, **not** the official sealed board. [H]

| Quantity | Historical public-231 paired CI half-width | EXT at n=2,199 |
|---|---:|---:|
| Capability, median over 27 arms | ±8.30 | **±2.69** |
| Composite, median over all arms | ±12.62 | **±4.09** |
| Composite, five arms near H2O | ±3.37 | **±1.09** |
| One cell-kit shimcal comparison | ±1.57 | **±0.51** |

These are the heldout receipt’s `sqrt(231/2199)` projections, recomputed here, **not simulations or measured heldout precision**. Correlated source cases, changed difficulty and Composite thresholds can make them optimistic. [H]

Recommended protocol **[INFERENCE]**:

- Preserve the existing seal: preregister each cell before first predictions; score each at most once; log reads; no calibration, prompt/readout selection or checkpoint selection on heldout. Stop using a confirmation set for adaptive design after results have influenced the program. [H]
- Name one finalist/claim family and multiplicity rule. Report paired aggregate deltas, per-type descriptive outcomes, confidence-band rates and useful risk/coverage; bootstrap by source case where dependence exists. Do not replace the official primary with a favorable heldout secondary. [PR][E]
- Run **at least two training seeds** for a finalist, reporting both checkpoint-specific item CIs and between-seed variation. Two seeds are still weak variance estimation, but better than pretending item resampling supplies replication. [P]
- For intuition only, EXT normal-approximation paired-binary 80%-power MDE is `(1.96+.84)*sqrt(q/n)`: with discordance q=.10–.20, public-231 detects about **5.83–8.24 pp**, versus **1.89–2.67 pp** at 2,199. Composite requires its own nonlinear/source-cluster power analysis; these are not Composite MDEs.
- Keep guarantee calibration separate from the sealed confirmation set. Under the current ε=.05, δ=.10, 101-threshold Bonferroni binomial rule, a zero-error threshold needs **135 independent acted observations**; with one error it needs **181**. The historical ~75 acts/one error is underpowered even before asking about deployment shift. More total items will not help if the model rarely supplies correct confident actions. [L][E]

**Decision rule for the next tranche:** require a surviving full-epoch custom model with declared structural properties, a non-adaptive confirmation result and replicated direction of effect before expanding architecture or data scale. If that fails, retain the evaluator/safety components as reusable assets and close the performance campaign rather than financing another round of unseparated hypotheses. **[INFERENCE]** [D][E][H]

## 7. Computation evidence and source register

### Commands exercised in this audit

Accounting command:

```text
python /tmp/ssdm-scratch/W5Postmortem/accounting.py
rows 188
numeric_raw 870.672
numeric_submitted_holds 196.42
ledger_resolved 593.3816 jobs 98
documented_jobs_nodes 659.4834 118 4
documented_builds 15.984614
documented_including_builds 675.468014
inferred_consumed 773.33
unattributed_to_consumed 97.861986
```

The scratch script prints every included job, original ledger line, charge, classification and source. It imports only Python standard-library modules and reads repository data/Git objects; it executes no repository code. Outcome buckets are methodological judgments, not output from a model.

History commands were executed through Python `subprocess.check_output`:

```text
git for-each-ref --format=%(refname:short) refs/remotes/origin
# 373 refs, including the HEAD alias; 372 branches

git log --all --format=%H%x09%ai%x09%s
# 2,072 distinct reachable commits

git log --all --format=%H -- language_growth plans/codex interfaces
# 320 path-touch commits
```

Lightweight formula calculations through Python Eval returned full-epoch training time **61.6005 minutes**, the four projected heldout widths above, and LTT thresholds **135/181**. **Model tests/builds/linters: not run.** Historical “PASS/FAIL” labels quoted or summarized in this report belong to their cited receipts, not this session.

### Sources

Internal links point to the supplied read-only snapshot or private GitHub branches; branch/commit references are essential where `agent-main` is older. Dates below are source dates, not dates of a new model measurement.

- **[L]** [State experiment ledger](file:///tmp/ssdm-state/results/ledger.tsv), October 7–13, 2026; especially lines 43–75, 82–118, 123–189.
- **[B]** [Provider usage](file:///tmp/State-Space-Decision-Model/receipts/2026-10-10-D-sf-usage-billing/usage.json), [billing](file:///tmp/State-Space-Decision-Model/receipts/2026-10-10-D-sf-usage-billing/billing.json), [capture timestamps](file:///tmp/State-Space-Decision-Model/receipts/2026-10-10-D-sf-usage-billing/capture.json) and adjacent job receipts, October 10.
- **[R]** [Independent H2O job records](https://github.com/apumutyala/State-Space-Decision-Model/blob/claude/ev-q668q-rescore/receipts/2026-10-11-ev-q668q-rescore/job-records.json), October 11; `4g6m8` charge 579,600 microdollars.
- **[O]** [Historical spend ledger](file:///tmp/State-Space-Decision-Model/LEDGER.md), October 6–10; use actual-charge column, not planned costs.
- **[P]** [PROGRESS](file:///tmp/ssdm-state/PROGRESS.md), October 8–13; lines 3–75, 78–98, 469–480 and 494–518.
- **[D]** [DEAD_ENDS](file:///tmp/ssdm-state/DEAD_ENDS.md), through October 12; [CANDIDATES](file:///tmp/State-Space-Decision-Model/CANDIDATES.md), historical candidate/proxy definitions.
- **[B0]** [BOARD](file:///tmp/ssdm-state/BOARD.md) and [HANDOFF](file:///tmp/ssdm-state/HANDOFF.md), including October 11 custom-stack directive; older “in flight” entries are superseded by terminal ledger rows.
- **[PR]** [PREREG](file:///tmp/ssdm-state/PREREG.md), October 9–13, including JEV3 amendments and jev-heldout access rule.
- **[E]** [Evidence Report V2 and closing entries](file:///tmp/ssdm-docs/EVIDENCE_REPORT_V2.md), October 12–13; report’s early $19.10 balance superseded by closing $11.67.
- **[G]** [GOAL](file:///tmp/State-Space-Decision-Model/GOAL.md), [AGENTS](file:///tmp/State-Space-Decision-Model/AGENTS.md), October 9 onward; historical budget prose is not a live balance.
- **[LS]** [LESSONS](file:///tmp/State-Space-Decision-Model/LESSONS.md), October 6–9; cited lesson numbers identify individual cases.
- **[PM]** [PREMORTEM](file:///tmp/State-Space-Decision-Model/PREMORTEM.md), October 6, §§3, 7–10; planned prices are historical estimates.
- **[H]** [jev-heldout receipt](https://github.com/apumutyala/State-Space-Decision-Model/blob/668b7472/receipts/2026-10-13-jev-heldout/README.md), October 13; inspected through `git show`, not by opening sealed records.
- **[K]** [jevkit README](file:///tmp/State-Space-Decision-Model/jevkit/README.md), board parity, split and metric limitations; [specification](file:///tmp/State-Space-Decision-Model/jevkit/spec.py).
- **[DS]** [Typed schema implementation](file:///tmp/State-Space-Decision-Model/decide/schema.py), especially lines 1–31, 86 onward.
- **[HD]** [Head implementations](file:///tmp/State-Space-Decision-Model/heads/heads.py), lines 1–51 and 361 onward.
- **[TC]** [Training cell](file:///tmp/State-Space-Decision-Model/train/run_cell.py), stage/exit contracts and provenance; newer fixes referenced in ledger.
- **[PH]** [Cached proxy heads](file:///tmp/State-Space-Decision-Model/train/proxy_heads.py), lines 1–24 and 101 onward.
- **[CS]** [Control/safety scope](file:///tmp/State-Space-Decision-Model/controlsafe/CONTROLSAFE.md), protected streams, certificates, Gramians and withdrawn G14 claims.
- **[SC]** [Safety certificate implementation](file:///tmp/State-Space-Decision-Model/safety_eval/certify.py), lines 1–69.
- **[DB]** [Data builder](file:///tmp/State-Space-Decision-Model/data/build.py), source shares and held-out exclusion contract.
- **[JC]** [Job launcher](file:///tmp/State-Space-Decision-Model/jobs/common/launch.py), lines 1–24 and legacy registration tables.
- **[T]** [Clef recipe and gap analysis](file:///tmp/State-Space-Decision-Model/techniques/10-clef-recipe-and-gap-attribution.md), October 6 with October 9 rate note; §§5, 9–10.
- **[CF]** [Cloudflare’s Clef training description](https://blog.cloudflare.com/clef-decision-models/), published October 1, 2026; searched and fetched October 14.
- **[J]** [Official JevBench README](https://github.com/fstandhartinger/jevbench/blob/main/README.md), v1.6.1 published October 6 with addenda through October 8; searched and fetched October 14.

[L]: file:///tmp/ssdm-state/results/ledger.tsv
[B]: file:///tmp/State-Space-Decision-Model/receipts/2026-10-10-D-sf-usage-billing/usage.json
[R]: https://github.com/apumutyala/State-Space-Decision-Model/blob/claude/ev-q668q-rescore/receipts/2026-10-11-ev-q668q-rescore/job-records.json
[O]: file:///tmp/State-Space-Decision-Model/LEDGER.md
[P]: file:///tmp/ssdm-state/PROGRESS.md
[D]: file:///tmp/ssdm-state/DEAD_ENDS.md
[B0]: file:///tmp/ssdm-state/BOARD.md
[PR]: file:///tmp/ssdm-state/PREREG.md
[E]: file:///tmp/ssdm-docs/EVIDENCE_REPORT_V2.md
[G]: file:///tmp/State-Space-Decision-Model/GOAL.md
[LS]: file:///tmp/State-Space-Decision-Model/LESSONS.md
[PM]: file:///tmp/State-Space-Decision-Model/PREMORTEM.md
[H]: https://github.com/apumutyala/State-Space-Decision-Model/blob/668b7472/receipts/2026-10-13-jev-heldout/README.md
[K]: file:///tmp/State-Space-Decision-Model/jevkit/README.md
[DS]: file:///tmp/State-Space-Decision-Model/decide/schema.py
[HD]: file:///tmp/State-Space-Decision-Model/heads/heads.py
[TC]: file:///tmp/State-Space-Decision-Model/train/run_cell.py
[PH]: file:///tmp/State-Space-Decision-Model/train/proxy_heads.py
[CS]: file:///tmp/State-Space-Decision-Model/controlsafe/CONTROLSAFE.md
[SC]: file:///tmp/State-Space-Decision-Model/safety_eval/certify.py
[DB]: file:///tmp/State-Space-Decision-Model/data/build.py
[JC]: file:///tmp/State-Space-Decision-Model/jobs/common/launch.py
[T]: file:///tmp/State-Space-Decision-Model/techniques/10-clef-recipe-and-gap-attribution.md
[CF]: https://blog.cloudflare.com/clef-decision-models/
[J]: https://github.com/fstandhartinger/jevbench/blob/main/README.md
