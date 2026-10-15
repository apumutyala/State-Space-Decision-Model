# Independent review of SSDM direction memo v0

**Verdict: amend before presenting.** The deliberation result survives independent recomputation. The proposed distillation-and-security program does not follow from it. Approve a staged experiment, not the claimed mechanism or an unconditional two-week delivery promise.

**Labels:** MEAS = this review's CPU recomputation or an identified historical receipt; EXT = opened external publication/vendor statement, not independent reproduction; SIM = arithmetic conditional on stated inputs; INFERENCE = interpretation/proposal. UNVERIFIED means the necessary receipt was not available. “Fatal” below means fatal to the stated inference/claim, not to the project or the measured headroom.

**Citation shorthand:** M = [memo](local://ssdm-direction-memo-v0.md); W1–W5 = [landscape](local://ssdm-W1-landscape.md), [capability](local://ssdm-W2-capability.md), [alignment](local://ssdm-W3-alignment.md), [architecture](local://ssdm-W4-architecture.md), [postmortem](local://ssdm-W5-postmortem.md). RH = `/tmp/ssdm-scratch/cos/rh_modal.py`; SCORE = `score_rh.py` and GATE = `gate_rh.py` in that directory; RAW = its `full/` directory; KIT = `/tmp/State-Space-Decision-Model/jevkit`; LEDGER = `/tmp/ssdm-state/results/ledger.tsv`. Citations with `:N–M` name file lines.

## 1. What changes the decision

1. **Fatal inference: extra computation is not demonstrated recoverable single-pass capability.** Qwen4B's gain is real, but no student was trained, and its advantage over H2O is statistically unresolved. Recovering H2O's local I requires **78.0% of the measured thinking gain**, not the proposed +8-I pilot threshold. Relevant literature ranges from essentially no transfer to more than the teacher's gain; the closest answer-only mathematical experiment failed outright. [MEAS/SIM: §2 below; EXT: §4 below; M:9,54–56,92.]
2. **Fatal mechanism claim: uncertainty is not an injection detector.** JevOut explicitly produces confident wrong decisions. The current shim gives every yes/no answer confidence ≥0.801, so the advertised τ=0.6 gate cannot route any of the 74 yes/no items. Exact reorder invariance also says nothing about semantic steering. Delete “cannot be talked into a decision” and the claim that capability and security automatically share one mechanism. [EXT: [JevOut abstract](https://arxiv.org/abs/2609.30243); MEAS: GATE:24–29,38–70, recomputation below; W3:135–147,189–201; M:11,98.]
3. **Major missing control: the experiment is not the proposed invariant head.** P1 changes both training targets and readout. Historical canonical stock lost **22.29 Composite [−33.49,−3.27]** against the ordinary stock shim; LM-S's primary difference was **−0.78 [−16.36,+3.47]**, not established non-inferiority. The new result cannot be transferred wholesale into that head. First establish matched, zero-training readout parity. [MEAS: LEDGER:186,188; M:92,104.]
4. **Fatal to a confirmatory claim: the acceptance design is incomplete.** A ≥10-point ASR improvement is impossible against a comparator below 10%; A6 gives “≥” to both act-rate and confident-wrong rate; clean capability, multiplicity, shift and useful coverage are inadequately constrained. Scoring a lockbox at step 1, using the result to choose step 4, then calling step 5 confirmation on the same lockbox is adaptive reuse. [SIM/INFERENCE: M:78–84,104–108; W3:14,42,114–119,201–205.]

**Fixable, not fatal to headroom:** wrong paired point differences; a materially wrong 9B output tariff; overbroad H2O “exact reproduction”; unqualified cache-warm latency; the Quyet speed-bound error. Details follow.

## 2. Experiment audit and independent recomputation

### 2.1 Method and results

**MEAS.** From each of the four raw JSONs, independently normalized label logprobs, applied the stated temperature/floor, constructed answers with frozen `jevkit.answers`, then used `items.load_public`, `scoring.build_table` and `aggregate`. Public item hashes were validated by the frozen loader; raw state/question payloads matched the pinned items on **231/231**. All arms contain 231 valid responses. Saved report I/C agree with independent raw reconstruction to **≤1.5×10⁻¹⁴**. Correct counts use KIT's diagnostic definition: score items use argmax for this count, while I uses expected-level error. [KIT `items.py`:120–158, `answers.py`:94–131, `scoring.py`:182–251,265–324.]

| Model | Arm | I | C | Correct /231 | Missing label probabilities | Truncated traces |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | one-pass | 36.97089 | 58.29425 | 156 | 0 | — |
| Qwen3.5-2B | think | 68.80473 | 72.53419 | 190 | 0 | 103 |
| Qwen3.5-4B | one-pass | 54.30991 | 66.04628 | 170 | 0 | — |
| Qwen3.5-4B | think | 84.05794 | 87.18116 | 214 | 0 | 67 |
| Qwen3.5-9B | one-pass | 61.16780 | 75.76466 | 178 | 0 | — |
| Qwen3.5-9B | think | 83.26352 | 88.35663 | 218 | 0 | 54 |
| H2O-Lightning-4B | one-pass | 77.51033 | 82.83394 | 205 | 0 | — |
| H2O-Lightning-4B | think | 83.70006 | 86.62819 | 210 | 0 | 49 |

**MEAS. Paired bootstrap:** 1,000 common resamples, seed 16, preserving the frozen scorer's `(side,type,tier)` strata; percentile intervals. These are item-resampling intervals conditional on this one model/sampling run, not training-seed or dataset-shift uncertainty. [KIT `scoring.py`:328–385.]

| Paired comparison | ΔI | 95% paired CI | Fixed / broken items |
|---|---:|---:|---:|
| 2B think − one-pass | +31.83384 | [+19.22763,+43.43546] | 44 / 10 |
| 4B think − one-pass | **+29.74803** | **[+21.04553,+37.96068]** | 46 / 2 |
| 9B think − one-pass | +22.09572 | [+10.38774,+32.06730] | 43 / 3 |
| H2O think − one-pass | +6.18973 | [+0.81692,+13.10926] | 8 / 3 |
| 4B think − H2O one-pass | **+6.54761** | **[−4.58534,+15.65202]** | 20 / 11 |

For the final cross-model comparison, **ΔC = +4.34722 [−1.28671,+13.35360]** and **ΔCapability = +5.44742 [−2.02954,+14.17437]**. M:51's point values **6.3 / 6.2 / 6.3** are wrong; its cross-model CI endpoints are substantially right. M:9,51's **29.9 [21.1,38.0]** should be **29.75 [21.05,37.96]**. None of the three cross-model intervals establishes superiority over H2O.

**Exact replacement for M:51:** “On this public set and single generation seed, stock-4B think minus one-pass ΔI is +29.75, paired 95% CI [+21.05,+37.96]. Versus H2O one-pass, ΔI is +6.55 [−4.59,+15.65], ΔC +4.35 [−1.29,+13.35], and ΔCapability +5.45 [−2.03,+14.17]; superiority is unresolved.”

The hard-tier claim is correct: 4B **60→94/111**; standard **63→72/72**; easy **47→48/48**. All five named family transitions in M:51 reproduce. Mean/median thinking tokens also reproduce: 2B **1,893.97/2,459**, 4B **1,409.87/619**, 9B **1,310.56/609**, H2O **716.06/8**. [MEAS: RAW and frozen table reconstruction.]

### 2.2 Could the protocol manufacture the gap?

| Check | Finding and cheapest discriminating test |
|---|---|
| **Prompt parity** | **MEAS: passes the stated control.** Same system, state, question, order and labels; only assistant thinking prefix differs. Reconstructed one-pass prompts are byte-identical to the local published-shim `Contract` on all 231 items. “Reply with a single letter” is shared, not a one-pass-only handicap. This measures this particular shim/readout, not the best possible one-pass prompt. [RH:28–66; `h2o_lightning_shim.py`:73–145,206–252.] |
| **Readout after closed think block** | Both arms read option-token probabilities at `Answer:`; label tokens are checked to be single-token continuations. Normalized candidate likelihoods are valid for this scoring contract. They are conditional on a particular generated trace, **not** calibrated uncertainty over possible traces. Teacher logits after a confident rationale can transmit confidence without correctness. Cheapest test: train/dev comparison of single-trace logits, multi-trace votes and independently verified labels. [MEAS/INFERENCE: RH:80–99,113–128.] |
| **Budget forcing** | At 3,072 generated tokens, the harness appends “Considering the limited time…” and forcibly closes the block; this is a different intervention from ordinary completed reasoning. Nevertheless it uses no gold-dependent selection. Among 4B's 67 truncated items, correct counts rise **30→51**; among 164 naturally closed traces, **140→163**. Thus both groups contribute, and forced completion alone cannot explain all 44 net fixes. Do not remove truncations post hoc. Cheapest test: on train/dev, fixed-budget ablation of the forcing sentence and alternative stopping budgets. [MEAS: RH:31–32,113–128; RAW.] |
| **Shared T/floor** | The shared mapping does **not** create the thinking gain; it shrinks it. Without either adjustment, 4B I is **42.76634→84.09692**, a **41.33** gain, versus **29.75** with the shim. But T=0.8 is H2O's convention, not a proven optimal calibration for stock: stock one-pass C falls **72.06587→66.04628** with the combined shim. Report raw and benchmark modes separately; fit any product calibration only on independent train/dev data. [MEAS: SCORE:16–36,53; RAW.] |
| **H2O confidence boost** | Reproduced: raw I **53.82553**, T-only **59.50699**, floor-only **76.63255**, both **77.51033**. The claimed **+23.7** is correct. For stock4B, T-only I **46.70757**, floor-only **54.16985**, both **54.30991**. A floor commits uncertain answers; it does not demonstrate better probability calibration. [MEAS: raw-logprob recomputation; KIT `answers.py`:94–131.] |
| **Label coverage** | **0 missing options and 0 affected items in every arm.** The top-256 fallback therefore did not affect these results. Future runs should request exact label-token IDs and fail if absent, as the actual shim does; imputing `min(top)-1` is not exact readout. [MEAS: RAW `missing`; RH:89–99; shim:312–324.] |
| **Seed/reproducibility** | Engine and think sampler both use **seed 0**, T=0.6, top-p=0.95, top-k=20. No evidence of best-of-seed selection in this run. One stochastic trajectory per item does not establish a stable teacher. 4B/H2O revisions are pinned; 2B/9B use mutable `main`. Only the last 400 trace characters are retained, so full rationale auditing or exact reconstruction of the second prompt is unavailable. Cheapest next run: pin all revisions and retain full traces/token IDs; use fresh seeds on dev before confirmation. [MEAS: RH:75–76,113–128; RAW revision fields.] |
| **H2O card reproduction** | **Passes the published accuracy identity check:** 205/231, tiers **48/48,71/72,86/111**, mean input **653.597≈654**; all five named near-tie correctness probes match the card. But current card timing uses v1.2.3 `acaf0d…`, this experiment uses `193ad740…`; local I/C **77.510/82.834** are not identical to historical kit-parity **77.6/83.4**. Full probability and service parity are not established. Replace “card exactly” with “the card's total, tier counts and five identity probes.” [MEAS: RAW, LEDGER:78; EXT: [H2O card, public evaluation/identity sections](https://huggingface.co/h2oai/h2o-lightning-4b/raw/main/README.md).] |

### 2.3 Economics and gating need stronger qualifications

- **Wrong 9B tariff (material correction).** SCORE:16–17,62–64 uses output price = five times input for every base: 9B is charged **$0.50/M**, whereas its frozen reference is **$0.15/M**. Repricing the recorded tokens gives **$0.26221/1k**, **K=27.4404**, **Composite=15.8216**, not approximately **$0.72091, K=14.3, Composite=3.0**. I/C are unchanged; 9B still fails the cost cap. [MEAS/SIM: RAW 9B think usage, frozen `api_v161.json` ClassOne 9B `cost.basis`, W2:139–143.]
- **Timing is not card-parity timing.** RH enables prefix caching, processes every prompt in both batched arms, then times repeated standard prompts; think latency is generated again and its trace length is not saved. Actual cache-hit rates are UNVERIFIED. This is cache-eligible repeat timing, without the card's full HTTP envelope, versus its explicitly distinct-request, zero-cache-hit pass. Therefore S/Composite are local harness estimates, not deployment receipts; cold timings cannot be recovered on CPU. The unchanged result is that full 4B thinking is outside the cost cap even if latency improved. Cheapest test when authorized: distinct held-out requests once, warm engine but controlled prefix state, exact production head, per-request tokens and end-to-end timing. [MEAS: RH:75,105–144; EXT: H2O card “Latency rechecked October 10”; W4:144.]
- **The gate is not yet a deployed uncertainty policy.** CPU rerun reproduces H2O τ=0.6: **20/231** routed, Capability **80.2→83.5**, cost **$0.0455/1k**, Composite **76.9→73.3** under the script's accounting. **Zero of the 74 yes/no items and zero of the 72 timed standard items route.** Unchanged p50 is consequently not evidence of free reasoning on production's difficult requests. 4B τ=0.6 likewise routes 24 items, none timed and none yes/no. Fit and assess risk/utility routing on unfloored, independent data; include hard-case tail latency and deferral cost. [MEAS: GATE:24–70 and new selection-count computation.]
- **Token accounting is an assumption, not settled billing.** GATE:45 charges original input once despite the initial decision and selected think pass; RH:127 collapses generate-then-read into continuation-style usage. Declare a genuinely reused-continuation implementation or count every logical pass. Charging another full original input to H2O's selected τ=0.6 items adds **$0.003886/1k**; for stock4B it adds **$0.005017**, moving its roughly **$0.0633** estimate above the cap. These are conditional accounting sensitivities, not claims about an observed invoice. The script's separate “finished within B” approximation also omits the attempted thinking on failures, so it is not a costed implementable budget policy. [SIM/INFERENCE: GATE:38–58,71–74; RH:125–128; W4:144.]

## 3. Exhaustive claim audit: memo §§0, 2, 4 and 5

“Pass” means supported at the stated scope, not independently rerunning the historical experiment. Exact replacement text is quoted for amendments.

### §0

| Claim | Verdict, evidence and corrected wording |
|---|---|
| **Item 1: alternative/API, eight days, $773/$785, missed targets, −7.8, −9.4, trained below base, $11.67** | **Mostly pass; amend certainty.** The API/goal is documented in `GOAL.md`:3–19; W5:15 dates Oct 6–13. **$773.33 is inferred consumption**, not fully reconciled charges: $675.47 documented, $97.86 unallocated. Historical −7.8 and −9.4 gaps and $11.67 balance are supported. “Trained cells below base” hides an inconclusive primary and positive inconclusive secondary. **Replace:** “No capability target was met. Consumption inferred from grants and the final balance is $773.33 of $785; $675.47 is attributed. Best own stock shim trails H2O by 9.4 local Composite; LM-S's primary difference from stock is −0.78 [−16.36,+3.47], with no improvement or non-inferiority established.” [MEAS/INFERENCE: W5:3,47–51,129–130; LEDGER:186–188.] |
| **Item 2: architecture never a moat; leaders thin post-trains; 3:1 hybrid; 73% on 13.3B** | **Amend.** **492.90/675.47=72.97%** is correct; Qwen4B has **24 GDN/8 attention** layers. “Never” is untestable; not all leaders are thin LoRAs, and GDN is recurrent linear attention rather than classical Mamba. **Replace:** “Architecture alone has not produced a demonstrated competitive advantage. Several leaders adapt strong hybrid bases, while others use full fine-tuning, joint heads or frozen readouts. The documented RWKV13.3B line consumed 73% of attributed spend and was cost-disfavoured under the later objective.” [MEAS/SIM/EXT: W5:59–64; W4:2,79–93; W1:83–95; W2:58–62.] |
| **Item 3: 54.3→84.1/87.2, H2O77.5/82.8, +29.9/CI, Composite18, format-loss/distillation conclusion** | **Scores pass except paired difference; conclusion overclaimed.** Replace with §2.1's exact numbers, and: “This demonstrates deliberation headroom under the tested readout; whether it transfers to an invariant one-pass student is unknown.” Composite≈18 is a local timing/accounting estimate, not official performance. [MEAS/INFERENCE: §2.] |
| **Item 4: weak/silent incumbents; attack rates; floor boost; structural assets; superior discipline** | **Rates and scoped assets pass; universal framing fails.** JevOut **61.4–73.2%**, four systems/seven datasets, is conditional on initially correct decisions and **64 accepted target evaluations**. DecisionRobust's rounded rates match **Jev291/367=79.3%, Perplexity200/350=57.1%, H2O116/311=37.3%, OpenAI90/319=28.2%** on its curated text-injection set; denominators and selection status differ. MD1/Clef lack the requested adaptive steering numbers, not all safety disclosures. **Replace:** “Published attacks expose substantial steering vulnerability in tested systems; matched adaptive evidence for Clef/MD1 is missing. SSDM has 0/565 reorder/relabel receipts and scoped proof machinery, but the cluster fix is not on agent-main, G15 is CPU-tested rather than serving-GPU validated, and own stacks have not demonstrated useful cluster-certified abstention. Our measurement process is a strength, not established industry superiority.” Floor +23.7 passes §2.2. [EXT: [JevOut](https://arxiv.org/abs/2609.30243), [DecisionRobust CSV](https://yifan-lan.github.io/DecisionRobust/data/asr.csv); MEAS: W3:135–147; W5:131–134.] |
| **Item 5: north star** | **Reject as factual promise.** One-pass-by-default is a design choice; knowing when to think, semantic immunity, useful deferral and no capability/alignment trade-off are unmeasured. Use the replacement in §5, decision 1. [INFERENCE; evidence §§1,4.] |
| **Item 6: Quyet legality, Oct10, rank16, directive, price, +1–2/+5 I, flat leaders, quantization** | **Mixed; see all §4 rows below.** In particular the historical ungating date is UNVERIFIED, hypothetical FP8 gains are not measured, and “quantization … not the score” is false for S. Two H2O generator-data cells do not establish that continued training of leaders generally fails. |
| **Item 7: operating model; $1–5 eval/$3–8 train; four models/nine minutes/$3; seal/freeze** | **Operating proposal reasonable; prices are planning estimates.** `full.log`:14–18 and final timestamp span about nine minutes; raw metadata confirms four models/vLLM0.30.0. Summed recorded load, arm and timing spans total **1,708.08 GPU-seconds**, equivalent to **$1.87 GPU-only at $3.9492/h**, excluding unlogged overhead, CPU/RAM and smokes. **Actual ≈$3 invoice UNVERIFIED.** **Replace:** “Illustrative small-cell budgets are $1–5/evaluation and $3–8/short training cell, subject to profiling and all-in caps. Today's four-model run took roughly nine minutes; its final billed cost is not yet receipted. Keep independent confirmation and the proposed freezes.” The ranges cannot automatically cover 20k×4 labels, two training seeds or adaptive attacks. [MEAS/SIM: RAW timings, `full.log`; W4:174,190–200.] |

### §2

| Claim | Verdict, evidence and corrected wording |
|---|---|
| **Capability definition/caps/leaders; Composite leaders** | **Pass with board/date scope.** Capability=(I+C)/2, caps **$0.064594655/1k and adjusted p50 1.23295664s**; Quyet **81.69**, H2O Composite **72.52**, Sage **74.05**. Do not mix live eligibility headlines with uncapped/raw rankings. [MEAS: W1:2–3,44–78,156–158; local `api_v161.json`.] |
| **Index0.3.2,166,d3 66.26 first; changes only new entries** | **Amend.** Counts/score correct; d3 is ordinal first but tied with Linne Oak under the board's rule. Some score changes came from a coverage correction affecting 37 models. **Replace:** “166 entries; d3 66.26 is ordinal first and tied with Linne Oak. Jev/Perplexity's displacement reflects new rows; other ranks/scores also reflect methodology corrections.” [MEAS: W1:4–6,108–126.] |
| **Winning recipe common to H2O/JevK5/Kev/Perplexity/Nimble; frozen FP8 Cap77–80** | **Second clause passes; first is not a controlled or universal recipe.** Decisio **79.57**, deck **77.59**. Perplexity and Kev27 use full FT; Perplexity changes attention/readout and uses tasksource; Nimble's disclosed minimal pairs do not establish the same teacher-rationale process. **Replace:** “Strong bases plus suitable readout/calibration are common. Data/adaptation differ; teacher-authored verification is a promising, confounded hypothesis, not an isolated winning ingredient.” [EXT/MEAS: W1:63,69,203–205; W2:57–65.] |
| **Composite slopes, zero latency, half tokens, prices, 9B600-token bound** | **Pass as conditional arithmetic.** At H2O: derivatives **I .3648,C .1625,S .1533,K .3616**; zero raw latency **+.574**, half priced input **+2.956** holding quality fixed. Prices quoted are **base-specific input references**, not a universal size tariff; add **9B output $0.15/M**. The 600-input-token 9B ceiling is **65.28** at I100 holding H2O C/S fixed. [SIM/EXT: W4:99–144; W2:139–145.] |
| **H2O205/boardI60 vs JevK5≈203/38.5; only lockbox/board confirms** | **Numerical contrast passes; amend confirmation scope.** “Public231 does not reliably rank leaders; source-disjoint sealed evaluation is needed. Our 2,199-item lockbox confirms internal transfer, not official board performance.” Its label conventions/hash-assigned tiers differ from the board. [EXT/MEAS: W2:9,64; W5:170,210.] |
| **MD1 5,250/11 unnamed; Clef card no safety; typed steering** | **Pass, scoped to opened disclosures.** Do not infer absence of internal safety work or absence of all public safety results. MD1's post gives qualitative results; typed allowed-option steering remains possible. [EXT: [MD1 launch](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), [Clef card](https://huggingface.co/Cloudflare/clef/raw/main/README.md); W1:183–187,225.] |

### §4, including repeated claims in §0.6

| Question | Verdict and replacement |
|---|---|
| **Allowed: Apache, ungated since Oct10, NOTICE** | **License/current access pass; historical date UNVERIFIED.** Opened [card](https://huggingface.co/chinhnc/Quyet-1.0-Large/raw/main/README.md), [model API](https://huggingface.co/api/models/chinhnc/Quyet-1.0-Large), [commit history](https://huggingface.co/api/models/chinhnc/Quyet-1.0-Large/commits/main). Last-modified Oct10 does not prove when gating changed; W1:198 attributes the date to Main without an attached historical receipt. **Replace:** “Apache-2.0 with NOTICE retention; ungated when checked October 14. Historical ungating date not independently verified here.” [EXT.] |
| **Gemma31B + r16, letters/T; base does work** | **Recipe passes; causal allocation unknown.** **Replace:** “Gemma-4-31B-it with merged rank-16 LoRA and a native letter-readout/calibration contract; the released evidence does not isolate the adapter's gain.” [EXT: Quyet card; W1:202.] |
| **Dense/no SSM/directive conflict** | **Pass under current authority.** Dense attention is not the hybrid main line, and the Oct11 third-party restriction is broader than an architecture rule. A principal can revise either; neither proves technical inferiority. [`GOAL.md`:7–15; `/tmp/ssdm-state/BOARD.md`:32; W1:93.] |
| **K50.5, $.09 input, needed I78.5/86; best boardI73.4; FP8 S90 C92 gives72.2/+1–2** | **Arithmetic mostly passes; one factual error and an unmeasured scenario.** Recomputed required I **78.45655** to tie H2O and **86.14172** to tie Sage. At unchanged Quyet I and hypothetical S90/C92, Composite **72.21739**, requiring **+1.28042 I** to tie H2O. **Replace:** “Quyet's I73.40 is not the unrestricted board maximum—Perplexity has I75.72, though cost-ineligible. FP8 achieving S90 and refitting C92 is an assumed scenario, not a measured Quyet outcome.” The roughly +5-I current-speed gap is correct. [SIM/MEAS: `api_v161.json` Quyet/H2O/Sage/Perplexity rows; W1:51–52,71.] |
| **Capability81.7; derivative is Quyet's result, not ours** | **Score passes; ownership judgment is not evidence.** **Replace:** “Any derivative must credit Quyet and demonstrate its incremental gain against the same parent; derivative status alone does not determine customer value.” Apply the same standard to Qwen/H2O-derived SSDM. [MEAS/INFERENCE: W1:51,198; M:67.] |
| **Quantization: no K change; René46; decisio pricing; speed≤+.6; not score** | **First facts pass; bound and conclusion fail.** The +.574 ceiling is at fast H2O, not Quyet. At Quyet's recorded point, zero raw latency permits **+1.49030 Composite**; halving both raw latency quantiles gives **+.57458**. These are bounds/scenarios, not expected quantization gains. **Replace:** “Precision alone does not lower reference-priced K. Quantization may improve S and actual serving economics, and may alter I/C; measure paired retention and latency. No Quyet-specific speed gain has been demonstrated.” [SIM/EXT: recomputation from frozen JSON; W1:210; W4:99–119.] |
| **Expected Quyet+1I,80%[−3,+5],$40–100; H2O≈0,−.41/−.97** | **Forecast is subjective, not measured; H2O metric is mislabeled.** **Replace:** “W2's Quyet +1 I, subjective 80% range [−3,+5], and $40–100/cell are unvalidated planning estimates. Two H2O generator-data cells had ΔComposite −0.41 [−1.75,+1.39] and −0.97 [−3.38,+1.12], not those ΔI values; they do not estimate a new teacher-authored recipe's effect.” [MEAS/INFERENCE: W2:98,155–156; W5:110.] |
| **Control arm only** | **Reasonable conditional policy, not empirical dominance.** “One matched H2O-parent control only if explicitly exempted and it answers a distinct data-transfer question; no generic continued-training or quantization campaign.” Cheapest discriminator is a matched new-data pilot, not another same-generator cell. [INFERENCE: W2:126,156; governing directive.] |

### §5: targets are proposals, not achieved measurements

| Claim/target | Audit and exact amendment |
|---|---|
| **Trusted evidence, escalation, retention controls** | **Agree.** Define authority outside model-generated text and separate legitimate evidence changes from instructions. “Cannot execute silently” also requires the host to enforce abstention/action policy. [INFERENCE: W3:25–42,174–184.] |
| **A1≤5%,upper≤8%,≥10pp below each** | **Amend.** “On common paired clean-correct units and a preregistered adaptive attack budget, ASR≤5%, 95% upper bound≤8%, and improvement≥10pp when comparator ASR≥20%, otherwise≥50% relative; the adjusted paired contrast must exclude zero. Report clean and worst-case robust accuracy on the full common cohort.” This restores the relative alternative and protects against denominator selection. [INFERENCE/SIM: W3:14,42,114–119; low-baseline impossibility follows directly from nonnegative ASR.] |
| **A3≥90% comparator** | **Keep as relative target, add usefulness.** “Evidence-retention≥90% of comparator, clean accuracy within 3pp, and matched decision/act coverage; report uncertainty.” Otherwise a weak or mostly abstaining system can win ASR. [INFERENCE: W3:42,115–118.] |
| **A4 eight-op≤1.3%,reorder/relabel0** | **MD1's reported 1.3% passes, cross-battery equivalence does not.** “Target≤1.3% on our published eight-operator battery, with CI upper≤2%; call MD1 comparison provisional until measured on identical operators. Require bitwise reorder/relabel invariance under the stated deterministic-kernel/content/tie assumptions.” Paraphrase is not structurally covered. [EXT/MEAS: MD1 launch; W3:145,189–193.] |
| **A5 rebinding≤1%** | **Valid unachieved target.** “Definition-level rebinding≤1% on matched pairs, including yes/no polarity and score levels; do not infer it from handle invariance.” Typed polarity in the content remains a specific untested risk. [INFERENCE: W3:145,189–193.] |
| **A6 ε5%, act-rate/confident-wrong≥best** | **Fix direction and scope.** “At preregistered ε=5%, δ=0.10, certify cluster-weighted risk using disjoint fitting/calibration/evaluation and fixed weights; maximize certified act-rate, require a prespecified useful minimum, and minimize confident-wrong rate at matched coverage. Report failure to certify explicitly. No adversarial guarantee follows from ordinary exchangeability.” Own historical certified act-rate is zero, not an existing product. [MEAS/INFERENCE: LEDGER:187; W3:135–136,119,201–205.] |
| **A7 no harm/over-refusal regression** | **Reasonable, not operational as written.** “Predefine non-inferiority margins for both harmful-action recall and benign acceptance, matched thresholds/coverage, with paired intervals; report the curve.” [INFERENCE: W3:35–42,114.] |
| **Honesty: raw probabilities always** | **Agree with precision.** “Report unfloored distributions and their calibration separately from the benchmark commit transform; use a separately fitted, unfloored risk score for act/defer.” Raw T=1 is not automatically calibrated. [MEAS/INFERENCE: §2.2; H2O card routing mode; W3:201–205.] |
| **Stop general injection-conformal, host-mask alignment, MD1-like0 flips, static Clef superiority** | **Agree; numbers are scoped.** Toy prior-only set mean **3.1/4** is SIM, not deployment utility. **3.64% average/4.14% pooled** are distinct summaries, not a confidence interval; both exceed zero. **Replace:** “Report exact reorder/relabel results separately from empirical paraphrase/format flips; static-template evidence supports only that battery, not superiority over Clef27B.” [W3:135–146,151–163; MEAS LEDGER:186–188.] |

**Missing from the adopted card:** restore W3's **A8 shift/persistence**, clean-accuracy guard, multiplicity policy and explicit rule for claiming overall dominance. A collection of isolated threshold passes is not by itself “far more aligned.” Cheapest test of the proposed shared mechanism: on stored clean/attacked paired predictions, measure attack detection and robust selective accuracy at a fixed clean act-rate using unfloored confidence. If the required predictions do not exist, that measurement—not coupled νPI training—is the first authorized alignment cell. [INFERENCE: W3:38–42,114–119,201–205; M:78–84,98.]

## 4. Strongest case against the strategy

### 4.1 What actually transfers from System 2 to System 1?

**No defensible universal “typical fraction” exists.** These tasks, model scales, curricula and baselines differ. For illustration define gain recovery as `(student − no-CoT)/(CoT teacher − no-CoT)`, not `student/teacher`. The fractions below are SIM from the opened papers' reported accuracies, not SSDM forecasts.

| Opened source | Reported result | Gain recovery and limitation |
|---|---|---|
| [Yu et al., Distilling System 2 into System 1](https://arxiv.org/html/2407.06023v3), Table 4 | Llama2-70B GSM8K: no-CoT **7.58%**, CoT **52.77%**, answer-only distilled **7.13%** | **−1.0% of gain**—essentially none, despite majority-vote teacher labels. This is the most direct counterexample to “just distill final thinking answers.” |
| Same paper, Table 1 | Coin flip **56.1→77.2→75.69%**; last-letter **30.0→44.5→98.0%** | **92.8%** and **>100%** respectively; strong positive evidence, but task-specific data curation, 70B and multi-token answers, not a 4B one-label guarantee. |
| Same paper, Table 3 | Judge agreement OASST **32.0→49.1→58.4%**; MT-bench **28.1→64.5→72.4%** | **>100%** recovery; especially relevant positive evidence for judgment. Uses branch/solve/merge and order-consistency filtering; it does not establish arithmetic transfer. |
| [Deng et al., Stepwise Internalization](https://arxiv.org/html/2405.14838v1), Table 3 | GSM8K: Phi3-3.8B **28→74→31%**; Mistral7B **38→68→51%** | **6.5%** and **43.3%**. GPT2-small/medium recover **60.7%/66.7%**. Progressive rationale removal is a richer intervention than the memo's answer-label LoRA. |
| [Hsieh et al., Distilling step-by-step](https://ar5iv.labs.arxiv.org/html/2305.02301), §3.2 | Auxiliary rationale and label tasks enable small students to answer without rationale at inference | Positive no-CoT evidence, but **rationale supervision**, not merely relabeling. The paper's “80%” example refers to training-data fraction, **not** fraction of reasoning gain transferred. |
| [CODI](https://arxiv.org/html/2502.21074v3), §§3–4 | Reports **99%/90% of explicit-CoT accuracy** on its GPT2/Llama1B experiments | Absolute accuracy ratios, not gain recovery. Uses **six sequential latent thoughts plus two boundary tokens**; it is **not one-forward-pass inference**. Its OOD iCoT baseline can fall below no-CoT. |

**Relevant decision-model warning:** [reflex's opened distillation receipt](https://github.com/kshetrajna12/reflex/blob/main/docs/results/lora-distill-qwen3.5-4b.md) reports teacher-agreement **.655→.849** but public hard accuracy **.658→.613**, temporal **.40→.27**, multi-hop **.67→.56**. This was a System-1 teacher, so it does not falsify deliberation distillation; it does show why teacher fidelity and in-domain validation are insufficient. [EXT.]

**Implications (INFERENCE):** computation-depth loss, bad/correlated teacher labels, source-distribution mismatch, forgetting and confidence imitation can each defeat transfer. Preserve rationales for an auxiliary/curriculum arm; independently verify a train-side sample; compare teacher gains on the *training distribution*, not just public231. Four samples from the same teacher can share the same misconception. Averaging after the 0.801 floor would contaminate the soft target; distill unfloored distributions or verified votes.

**SIM:** matching H2O local I needs `(77.51033−54.30991)/(84.05794−54.30991)=77.99%` recovery. A +8-I result recovers **26.89%**, far below that promise. The +8 pilot can justify more research, not “H2O parity.” Failure of one 20k/r64 recipe to recover 25% does not prove “single-pass distillation fails at 4B” or that Composite is unreachable. Replace M:128 with: “This recipe fails its preregistered transfer gate; change the hypothesis only if another intervention has independent supporting evidence.”

### 4.2 The recommended north star is too broad and insufficiently falsifiable

- **It combines three unsolved deliverables:** competent invariant one-pass prediction, compute routing and semantic attack resistance. No measured result yet establishes their conjunction. P1's ordinary letter readout and P2's trust-controlled head are not interchangeable. A generic “cannot be talked into” claim also conflicts with retaining legitimate evidence in untrusted state. [MEAS/INFERENCE: LEDGER:186–188; W3:174–195; M:92–98.]
- **Security and difficulty need different evidence.** A clean-distribution confidence ranking may route hard arithmetic yet confidently accept fabricated authority. A trusted-only guarantee can exclude attacks only by excluding information; W3's own first falsifier is trusted-only accuracy below **85%** of full accuracy. Measure that information gap before investing in the certificate. [EXT/INFERENCE: JevOut; W3:174–184.]
- **Changing the goal after losses risks declaring success by moving the denominator.** W1:227 finds no established willingness to pay for certificates. A workload-scoped action/deferral utility objective could legitimately supersede Composite, but a slogan plus reported metrics does not choose a customer, authority boundary, minimum act-rate or cost of deferral. [INFERENCE: `GOAL.md`:3–19; W1:216–227; M:11,98.]
- **P1 is not W2's data experiment.** Relabeling existing generator rows leaves their state/rubric/source distribution unchanged. W2's main hypothesis requires **fresh teacher-authored scenarios**, independent verification and matched generator-vs-teacher controls. Its grade is C+, not proof of the winning recipe. [W2:10–11,98–126; M:56,104.]
- **Budget/time are not yet priced for the actual plan.** At the observed 4B batched rate, **20k×4 samples ≈4.80 H100-hours and 112.79M output tokens**, versus **≈1.20h** for one sample. The four-sample GPU-only extrapolation is **≈$19 at Modal's listed rate**, before overhead, verification, two-seed training or confirmation; throughput on actual training rows is UNVERIFIED. W3 budgets its alignment thesis **$60–120 and 3–4 agent-weeks**, not a demonstrated $40–80/two-week delivery. M:103–108 line items sum **$80–160**, while the title says $150–250; label reserves and missing work explicitly. [SIM: RAW 4B `think.wall_s=49.87976`,231 items; W4:174; W3:15; M:56,92–93,100–108.]

### 4.3 Better first purchase; no proven universally dominant replacement

**INFERENCE:** No opened evidence proves another complete model pathway dominates on capability, safety, cost and time. The memo should not invent one. A **measurement-first, provenance-controlled hybrid program** does dominate the proposed immediate 20k relabel-and-couple plan as an information purchase:

1. Recover existing data/receipts; establish untrained ordinary-shim versus exact-head behavior on the same train/dev cohort. Keep the ordinary readout as the capability control, not silently as the baseline for a different student head. [LEDGER:186–188.]
2. Use W2's approximately **2k-row matched provenance pilot**: current generator distribution versus newly authored, independently thinking-verified scenarios, same host/readout/steps and a base-retention anchor. Add a matched relabel-only contrast if affordable; otherwise test that hypothesis separately. This separates “better labels” from “better tasks.” Its $5–8 training-pair estimate is a planning range, not a new quote. [W2:125–127,219.]
3. Retain **full-epoch JEV3-LM with fixed shimcal** as an omitted, cheaper capability control: W5's highest-ranked remaining experiment was never completed, and its positive secondary is only a hypothesis. Do not claim the old lineage is exhausted because its truncated predecessor was inconclusive. [W5:109,184–185.]
4. In parallel only at the measurement/design level, choose one agent-control workload and compare a frozen-backbone + host policy + calibrated-deferral baseline with the custom trust mechanism at matched useful coverage. Buy custom safety training only if the baseline leaves an identifiable gap. A wrapper can be valuable but is not an SSM-exclusive moat. [W3:135–140,174–184,201–205.]

**Replacement for M:98:** “Our initial program is a falsifiable 4B hybrid transfer experiment plus a workload-scoped robustness evaluation. Deliberation-derived supervision, invariant readout and trust control are separate hypotheses with matched controls. We will not claim a shared uncertainty-based security mechanism. Scale only after independent capability transfer and useful, paired robustness gains are measured.”

**Confirmation amendment:** “Choose all training, calibration, routing and attack thresholds on source-group-disjoint development data. Reserve an untouched final split for the selected recipe; if early lockbox feedback informs design, that portion becomes development and cannot confirm the later model. A new tranche requires positive independent transfer with the prespecified uncertainty and retention guards.” This replaces the M:104 **AND** kill rule that allows a public-only gain to keep a non-transferring model alive. [INFERENCE: W4:232; W5:202,210,229.]

## 5. Ranked verdicts on the five §9 decisions

Rank is urgency of action, not the original numbering. All replacement language below is a proposal, not permission to spend or amend governing instructions.

### Rank 1 — Decision 5, data access: **AGREE WITH CHANGE**

**Strongest reason:** without the real draws and sealed manifests, the proposed transfer result cannot be reproduced or independently confirmed. The local `ssdm-data` directory contains public JevBench/board files, not the training draws or 2,199-item lockbox. [MEAS: directory read; LEDGER:189; W5:167–170.]

**Blind spot / cheapest test:** rebuilding public training data is not rebuilding the old sealed confirmation set. Check hashes, provenance and access logs without opening sealed labels to the development lane.

**Replacement:** “Obtain the existing training draws, checkpoint/receipt artifacts and sealed-set manifests through controlled access. Preserve the lockbox seal. If recovery is impossible, approve a newly sourced train/dev build and a separately constructed untouched confirmation set; do not substitute public training splits for the original lockbox.”

**Would flip:** existing assets fail provenance/disjointness checks or cannot be recovered; then rebuild rather than reuse, with a new seal.

### Rank 2 — Decision 4, freeze list: **AGREE WITH CHANGE**

**Strongest reason:** repeated unseparated recipes consumed resources without competitive evidence; freezing them protects the next experiment. [MEAS: W5:104–113.]

**Blind spot / cheapest test:** a categorical H2O freeze contradicts decision 2's control cell, and “all training failed” would erase the unfinished full-epoch hypothesis.

**Replacement:** “Freeze Track L, Q9-AF, RWKV13.3B product scaling and Decision-Index/Clef chasing. Freeze repetitions of the old H2O generator-data recipe. Preserve artifacts and permit only the explicitly approved, new-data matched H2O control. Do not classify the uncompleted full-epoch custom-head test as a completed negative.”

**Would flip:** a new mechanism has clean, retained capability and a cost-feasible pilot—not merely better KL or synthetic-slice accuracy. [W4:91,163; W5:184–191.]

### Rank 3 — Decision 2, comparator/control exemption: **AGREE WITH CHANGE**

**Strongest reason:** superiority cannot be established by comparing our adaptive results against someone else's static/private battery. [EXT/MEAS: W3:114–123,145–146.]

**Blind spot / cheapest test:** MD1 also has an explicit “never re-run” rule in `GOAL.md`:9, so the exemption must cover both governing restrictions. The $1–10 estimate is not a complete priced five-tier adaptive campaign.

**Replacement:** “Authorize a narrowly capped, matched alignment comparison on preregistered common units, with the attack budget and total target calls priced before launch; explicitly exempt MD1 from both relevant restrictions. Start with the minimum useful comparator set. Authorize at most one H2O-parent training control only for the new data hypothesis, with its own cap; no general fast-follower campaign.”

**Would flip:** existing receipts already match the exact threat model/units, eliminating a rerun, or the scoped adaptive test cannot fit the cap. The H2O training control is unnecessary if the pilot does not show useful transfer.

### Rank 4 — Decision 3, ≈$200/provider: **AGREE WITH CHANGE**

**Strongest reason:** $200 can buy decisive small comparisons, but the current plan does not justify releasing it against a promised integrated product. [SIM/INFERENCE: §4.2; W4:190–200; W5:229.]

**Blind spot / cheapest test:** SF is nearly depleted, **not exhausted**—last receipted balance is **$11.67**; grants and projected cheap cells are not funded capacity. Price the actual rendered-token pilot and full confirmation workload first. [MEAS: LEDGER:186; W4:11.]

**Replacement:** “Approve a $200 ceiling on an explicitly authorized provider: release $50 for data/readout parity, teacher-quality/provenance screening and scoped baseline evaluation; release up to $100 more only after the preregistered transfer/utility gate; reserve $50 for independent confirmation and overruns. Preserve the last reported $11.67 SF balance. New grants count only after award.”

**Would flip:** the first tranche cannot produce a source-disjoint evaluable result within its cap, or teacher labels fail basic utility/retention checks; stop model training and retain the audit assets.

### Rank 5 — Decision 1, adopt the proposed north star: **DISAGREE AS WORDED**

**Strongest reason:** “cannot be talked into a decision” promises semantic immunity without a threat model, while the proposed uncertainty mechanism demonstrably lacks that implication. [EXT: JevOut; MEAS: zero yes/no gate routing; W3:135–147,201.]

**Blind spot / cheapest test:** capability loss, low coverage or ignoring evidence can manufacture apparent robustness. Evaluate trusted-only information loss and paired robust accuracy at matched clean accuracy/act-rate before authorizing the objective change.

**Replacement:** “Keep the existing benchmark objectives until the principal explicitly selects a workload-level utility objective. Our proposed differentiator is a 4B hybrid with exact order/handle invariance and measured adaptive-steering resistance under an explicit trust boundary and attack budget, without sacrificing useful evidence retention or cost-feasible act-rate. Single-pass is the default; deliberation, deferral and distillation are separately tested options, not assumed security guarantees.”

**Would flip to agreement:** a source-disjoint, paired result demonstrates useful certified/robust actions at matched capability and evidence retention, within a measured serving budget, and the principal explicitly accepts that workload objective. An ASR-only win or a public231 thinking gain would not flip it.

## 6. Evidence boundary and reproducibility receipts

This review performed read-only CPU scoring/arithmetic, prompt/item comparisons and source inspection. It did **not** run models, GPUs, paid APIs, training, repository tests or modifications. No experimental result here establishes transfer to the official sealed board, cold deployment latency, training throughput on the proposed new corpus, a paid invoice, or an end-to-end safe student.

Raw/scorer SHA-256 receipts recorded during this review:

| File | SHA-256 |
|---|---|
| RAW `Qwen3.5-2B.json` | `8d4557712ae3df43d18a388248be274a1d0ecd26488b46c3037eef9f746de24d` |
| RAW `Qwen3.5-4B.json` | `1dd548efe4f6461cc3b939edbdc1f48e7471b638694a7fe2f11f28461f129f93` |
| RAW `Qwen3.5-9B.json` | `55b1e641ab777ef38d39df1e43165d0e53730469e65b9c5a5820c96be40c3dd2` |
| RAW `h2o-lightning-4b.json` | `626a846f59c67488a8cbef20f626e88578a5158d1c8cbbb8d47564a7bb66dbd5` |
| `cos/public231.jsonl` | `5dbc4e3392d90725227e105eb69471c3dfab80ba577d2384b63ac3e789700d71` |
| KIT `scoring.py` | `43c2c934e946b9fbcdf12b6ac0313fd6082e9c70b1684708f52203277fc0a9b3` |
| KIT `spec.py` | `71362bd9d68d651dcc7bea013f67dc9e7993459535b1260a1724ed4989d92171` |

Recomputation environment: repository root, `SSDM_DATA=/home/alp/ssdm-data`, `/home/alp/ssdm-venv/bin/python -B -X utf8`; reconstruct probabilities from raw logprobs rather than trusting saved predictions; align by pinned item ID; use the frozen table/aggregation and common stratified bootstrap draws. The tables in §2 are this review's numerical receipt. Future paid tests named above are proposed discriminators, not exercised verification.
