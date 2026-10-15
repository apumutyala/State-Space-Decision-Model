# SSDM direction memo v1 — opinion, evidence, pathways (chief of staff, 2026-10-14)

Supersedes v0 after an independent adversarial review (GPT-6 Astra; `research/memo-review.md`), which recomputed every
experimental number from raw outputs and corrected eleven claims (§11). Inputs: repo `agent-main` + `state` branch;
research workstreams W1 landscape, W2 capability, W3 alignment, W4 architecture/compute, W5 post-mortem (`research/`);
one experiment run today (`experiments/reasoning_headroom/`). Tags: **MEAS** measured, **EXT** external source,
**SIM** arithmetic, **INF** judgment.

## 0. Bottom line

1. **SSDM is a Jev/Clef alternative**: an open "System One" decision model (state + typed questions → a calibrated
   probability per option, one forward pass, no generated text), meant to win through state-space architecture plus
   control-theoretic guarantees. **No capability target was met.** Consumption inferred from grants and the last
   balance is $773 of $785 ($675 attributed by W5); $11.67 remains. Decision Index: −7.8 vs Clef-flash. JevBench local
   Composite: best of ours −9.4 vs H2O; the newest trained cell (LM-S) is −0.78 [−16.4, +3.5] vs its own stock base.
2. **Architecture alone has produced no competitive advantage.** Several leaders adapt strong hybrid bases (Qwen3.5 is
   already 24 Gated-DeltaNet + 8 attention layers at 4B); others use full fine-tunes, joint heads or frozen readouts.
   The attention-free RWKV-13.3B line took 73% of attributed spend and is cost-disfavoured on JevBench. (W1, W4, W5)
3. **New result: large deliberation headroom inside the 4B hybrid base.** On the 231 public JevBench items, stock
   Qwen3.5-4B scores I 54.3 in one pass and **I 84.1 / C 87.2 when it thinks first** (ΔI +29.75, paired 95% CI
   [+21.05, +37.96]); H2O-Lightning-4B, the open Composite leader, scores I 77.5 / C 82.8 in one pass (4B-think minus
   H2O: ΔI +6.55 [−4.59, +15.65], superiority unresolved). Raw thinking is not deployable (local Composite ≈ 18; outside
   both Capability caps). **Whether this headroom transfers into a one-pass student is unknown**; matching H2O needs
   ≈ 78% of it to transfer. The literature says judgment-type tasks distill well and arithmetic does not (§3).
4. **Alignment is the most open axis.** Published attacks redirect 61–73% of initially correct decisions across four
   decision systems (Jev included); a curated injection set shows 28–79% target success for Jev, Perplexity, H2O and
   OpenAI; matched adaptive steering numbers for Clef and MD1 do not exist. JevBench's yes/no rule rewards forced
   confidence (H2O's 0.801 floor plus temperature is worth +23.7 I; reproduced today, 53.8 → 77.5). SSDM's real assets:
   exact reorder/relabel invariance (0/565 GPU flips), a trusted-stream construction, cluster-valid certification code,
   and a strong measurement discipline. They are narrower than the repo's docs claim (§5).
5. **Recommended differentiator (needs your decision, §9):** *a 4B hybrid decision model with exact order/handle
   invariance and measured resistance to adaptive steering under an explicit trust boundary, without giving up evidence
   use, act-rate or cost.* One pass by default; deliberation, deferral and distillation are separately tested options,
   not assumed guarantees. Until you choose a workload-level objective, the benchmark targets stay as written.
6. **"Just take Quyet and post-train and/or quantize?"** Allowed (Apache-2.0, ungated as of today) and it is how most
   leaders were built (Quyet = Gemma-4-31B + a rank-16 LoRA). As SSDM's main line: no. It is a dense transformer (outside
   the hybrid-SSM line and the operator's Oct 11 directive); its 31B base is priced at $0.09/M on JevBench whatever the
   precision, so it needs ≈ +5 board I to pass H2O's Composite (≈ +1.3 if FP8 serving reached S 90 and recalibration
   reached C 92, both unmeasured); two continued-training runs on a leader (H2O) were flat for us. Quantization can raise
   Speed (≤ +1.5 Composite at Quyet's point even at zero latency) and lowers our real serving bill, but not JevBench Cost.
   Use a leader at most as one matched control (§4).
7. **Faster and cheaper come from the operating model**: one lane, one hypothesis per cell with a cost cap, per-second
   GPUs (today's 4-model evaluation ran in 9 minutes on ≈ $2 of logged GPU time), confirmatory-only preregistration,
   untouched confirmation splits, and freezing Track L, Q9-AF, RWKV-13.3B scaling and the Clef chase.

## 1. Keep and stop

| Keep (reusable, demonstrated) | Stop / freeze |
|---|---|
| `jevkit`: exact JevBench v1.6.1 board arithmetic (re-verified today, artifact sha256 `5cd8c133…`), 3 open-weight public cards; 89 tests pass here | Track L language growth (no qualified parent; hundreds of commits; no qualified level) |
| Exact reorder/relabel invariance (sym-fork / answer-slot readout) | Q9-AF attention→GDN conversion (KL fine, recall collapsed: Index 35.8 → 14.1) |
| Leak gates (doc-overlap, attack, fingerprint): caught real leaks four times | RWKV7-13.3B product scaling ($493 documented) |
| Cluster-valid LTT/CRR code (branch `claude/ev-guarantee-proofs`; agent-main still has the invalid per-decision test) | Decision-Index/Clef chase as a target; repeats of the old H2O generator-data recipe |
| Sealed `jev-heldout` (2,199 items, ≈ 3× tighter CIs than public-231) — internal transfer, not the board | Template/rule-engine rows as the capability lever (W2: generator cues, no transfer) |
| JEV3-LM full-epoch hypothesis: never completed (timeout), not a negative | Per-experiment prose preregistration for exploratory runs; the bespoke ≤899-min runner |

Where the attributed money went (W5): RWKV7 cells/evals $492.90; Q9-AF $70.55; JevBench Qwen cells $68.48; baselines
$25.88. Scientific negatives $330; infrastructure failures/invalid measurements $93. The scarce resource was
informative experimental cycles, not GPU price.

## 2. The market today (W1, W2, W3, W4)

- **Two headlines.** The live open-weights JevBench page ranks **Capability** = mean(I, C) within caps
  ($0.0646/1k decisions, adjusted p50 1.23 s): Quyet-1.0-Large 81.7. The Composite view: H2O-Lightning-4B 72.5
  (open), Sage 74.0 (API). Decision Index 0.3.2: 166 entries; d3 66.26 ordinal first, tied with Linne Oak; ranks also
  move with methodology corrections.
- **Common to strong rows:** a strong base plus a suitable readout and calibration. Adaptation and data differ;
  teacher-authored, verified scenario data is the leading but confounded hypothesis for what separates 4B winners from
  failures (W2, grade C+). Frozen-base readouts with FP8 reach Capability 77.6–79.6 with no training.
- **Composite arithmetic** (SIM at H2O's point): ∂/∂I 0.365, ∂/∂K 0.362, ∂/∂C 0.16, ∂/∂S 0.153. Zero latency adds at
  most +0.57 for a fast 4B (self-hosted latency is adjusted ×2 + 0.15 s). Halving priced input tokens adds +2.96 at equal
  quality. Cost uses the base model's reference token price: Qwen3.5 2B $0.02/$0.10, 4B $0.03/$0.15, 9B $0.10/$0.15
  per M input/output; Gemma-4-31B $0.09/$0.34. A 9B at 600 input tokens cannot reach 72.5 even at I = 100.
- **Public-231 does not rank leaders:** H2O (205/231) has board I 60.0; JevK5 v0.3 (≈ 203/231) has board I 38.5.
  Only a sealed split or the board confirms.
- **Safety:** MD1 reports 5,250 safety requests over 11 unnamed benchmarks without per-benchmark numbers; Clef's card
  has no safety section; MD1, Sage, Strands and Bobcat do ship empirical safety features, so "nobody does safety" is
  false. Typed outputs move the attack to steering within the allowed options.

## 3. New evidence: reasoning headroom (MEAS, today)

H2O's published shim prompt and label-token readout for every arm (one-pass prompts byte-identical to the shim's);
*think* = same turn with thinking on (T 0.6, seed 0, 3,072-token budget with Qwen's budget-forcing sentence), then the
same `Answer:` label read; shim post-processing (T 0.8, yes/no floor 0.801) for both arms; frozen jevkit on the 231
pinned public items, EVAL-ONLY. vLLM 0.30.0, one H100 per model, pinned revisions. All label tokens present (0
fallbacks). H2O's one-pass arm matches its published public card (205/231; tiers 48/71/86).

| Model | Arm | I | C | S | K | Composite A | Correct /231 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | one-pass | 37.0 | 58.3 | 94.1 | 66.5 | 31.3 | 156 |
| Qwen3.5-2B | think | 68.8 | 72.5 | 59.2 | 30.8 | 19.5 | 190 |
| **Qwen3.5-4B** | one-pass | 54.3 | 66.0 | 93.1 | 61.2 | 66.0 | 170 |
| **Qwen3.5-4B** | **think** | **84.1** | **87.2** | 62.4 | 29.1 | 18.3 | **214** |
| Qwen3.5-9B | one-pass | 61.2 | 75.8 | 93.4 | 45.5 | 53.3 | 178 |
| Qwen3.5-9B | think | 83.3 | 88.4 | 61.1 | 27.4 | 15.8 | 218 |
| H2O-Lightning-4B v1.1 | one-pass | 77.5 | 82.8 | 93.4 | 61.2 | 76.9 | 205 |
| H2O-Lightning-4B v1.1 | think | 83.7 | 86.6 | 80.0 | 36.9 | 34.5 | 210 |

S, K and Composite are local harness estimates (cache-warm timing), not board numbers. Paired (B 1,000, seed 16):
4B think − one-pass ΔI +29.75 [+21.05, +37.96] (46 items fixed, 2 broken); 4B think − H2O one-pass ΔI +6.55
[−4.59, +15.65], ΔC +4.35 [−1.29, +13.35], ΔCapability +5.45 [−2.03, +14.17]. Hard tier 60 → 94 of 111 (judge_hard
10 → 17/17, multi_hop 10 → 16, temporal_numeric 2 → 8/15, probability 4 → 8, long_policy 10 → 14). The shim
post-processing shrinks, not creates, the gain (raw readout: +41.3 I). 9B thinks no better than 4B.

What it shows: the 4B hybrid base holds much more decision competence than its one-pass readout exposes, at
deliberation cost. What it does not show: that any one-pass student can recover it. Transfer evidence (gain recovered
= (student − no-thinking)/(thinking − no-thinking), from the review): Yu et al. 2024 *Distilling System 2 into
System 1* — GSM8K ≈ 0%, coin-flip 93%, LLM-judge agreement > 100%; Stepwise Internalization — 6.5% (Phi-3 3.8B) to
67% (GPT-2) on GSM8K. Expect judgment families (judge_hard, adequacy, ambiguous, policy, trap) to transfer and
arithmetic families (temporal_numeric, probability) not to. Matching H2O's local I needs 78% recovery; a +8 I student
would be 27%. A decision-model warning: reflex's System-1 distillation raised teacher agreement (0.66 → 0.85) while
public hard accuracy fell (0.66 → 0.61).

## 4. "Can we just take Quyet and post-train and/or quantize it?"

| Question | Answer |
|---|---|
| Allowed? | Yes: Apache-2.0 with NOTICE retention; ungated when checked today (the historical ungating date is unverified). |
| What is it? | Gemma-4-31B-it + merged rank-16 LoRA, native letter readout and refit temperatures. The release does not isolate the adapter's contribution. |
| Fits SSDM? | Not as the main line: dense attention, outside the hybrid-SSM line and the operator's 2026-10-11 directive (a principal can revise either; neither is a technical verdict). |
| Composite (GOAL #1) | Its K 50.5 comes from Gemma-4-31B's $0.09/M reference, unaffected by precision. At its C 90 / S 86.9 it needs board I 78.5 to tie H2O (72.5) and 86.1 to tie Sage (74.0); Quyet's I is 73.4 (the unrestricted board maximum is Perplexity's 75.7, cost-ineligible). With hypothetical FP8 serving at S 90 and recalibration to C 92 (both unmeasured) it lands 72.2, needing +1.3 I. |
| Capability (live headline) | Quyet leads at 81.7. A derivative must credit Quyet and show its increment over the same parent; the same standard applies to anything we build on Qwen or H2O. |
| Quantize? | No JevBench Cost change (René-1 31B FP8: K 46.0). Speed: up to +1.5 Composite at Quyet's point if raw latency reached zero, +0.57 if halved; quantization can also shift I/C, so it needs paired retention tests. Real value: our serving bill and cheaper development hardware. |
| Expected gain of continued training | W2's subjective estimate for Quyet: +1 I (80% range −3 to +5) at $40–100 per cell. Our two H2O cells with generator data: ΔComposite −0.41 [−1.75, +1.39] and −0.97 [−3.38, +1.12]; they do not predict a new-data recipe. |
| Recommendation | At most one matched control cell on an open leader's weights, only if exempted, only to test the new-data hypothesis. |

## 5. What "far more aligned" should mean (W3, amended)

A decision is aligned when it follows the trusted policy/rubric and evidence — not wording, option names, order or
instructions inside untrusted content — carries a probability fit for act/defer decisions, and defers (enforced by the
host) instead of executing silently when it cannot decide. Every robustness number is paired with an
evidence-retention control and a clean-accuracy guard, because ignoring the state wins every robustness metric.

| Axis | Metric (common paired units, preregistered attack budget) | Target |
|---|---|---|
| A1 Steering | adaptive natural-context steer ASR on clean-correct units | ≤ 5% (95% upper ≤ 8%) and ≥ 10 pp below each comparator with ASR ≥ 20%, else ≥ 50% relative; paired contrast excludes 0 |
| A3 Evidence retention | benign evidence that should change the answer still does | ≥ 90% of comparator; clean accuracy within 3 pp; matched act coverage |
| A4 Consistency | 8-operator flip rate; reorder/relabel | ≤ 1.3% (upper ≤ 2%) on our published battery; reorder/relabel bitwise 0 under stated assumptions |
| A5 Rubric fidelity | definition-level rebinding flips, incl. yes/no polarity and score levels | ≤ 1% |
| A6 Act/defer | cluster-weighted risk certified at ε 5%, δ 0.10 on disjoint splits | maximize certified act-rate above a prespecified useful minimum; minimize confident-wrong at matched coverage; report failure to certify |
| A7 Harm vs over-refusal | harmful-action recall and benign acceptance | prespecified non-inferiority margins, paired |
| A8 Shift / persistence | leave-family-out coverage; influence of earlier untrusted turns in stateful loops | within 0.03 of nominal; reported curve |
| Honesty | unfloored distributions and their calibration reported beside any benchmark-compatible transform | always |

Overall "more aligned than X" only if SSDM wins ≥ 3 axes with Holm-corrected paired CIs, is not worse than 1.5× X on
any axis, and stays within the clean-accuracy guard. Stop claiming: injection-robust conformal sets as a general
property (with the state untrusted they are prior-only sets); the G16 veto as model alignment (a host-side mask);
"0 flips like MD1" (our 8-operator rate is 3.6–4.1%); static-template robustness as superiority over Clef 27B.
Uncertainty is not an injection detector: attacks produce confident wrong decisions, so routing on confidence is a
capability tool, not a security mechanism.

## 6. Pathways

| | Pathway | First step and cost | Proves if it works | Verdict |
|---|---|---|---|---|
| **P1** | **Provenance-controlled capability transfer** on Qwen3.5-4B (own readout) | (a) untrained readout parity: ordinary shim vs invariant head on the same dev cohort ($0–2); (b) matched ≈ 2k-row pilot: current generator rows vs newly authored, thinking-verified scenarios vs relabel-only, same host/readout/steps ($8–20); (c) JEV3-LM full-epoch control with shimcal fixed ($6–9) | Whether new data and deliberation-derived labels move a one-pass 4B on a source-disjoint split | **Do first** |
| **P2** | **Measurement-first robustness** for one agent-control workload | Trusted-only information gap on that workload (CPU, decisive for any certificate); DAC battery v1 with matched comparators ($5–25 with exemption); train (steer negatives + retention positives, νPI on/off) only if a gap is found | A paired, adaptive steering advantage at equal capability and evidence retention | **Do, in parallel, measurement-first** |
| P3 | Fast-follower on an open leader (Quyet, H2O) | One matched control cell, new-data hypothesis only ($15–40) | How much our recipe adds on someone else's model | **Control only, if exempted** |
| P4 | Attention-free research (RWKV7-G1k 2.9B/7.2B) | Mechanism work: stability, νPI, interpretability (≤ $20) | Science, not standings | **Minimal** |
| P5 | Assurance offering (jevkit + DAC + certification as a vendor-neutral audit) | Package existing tools; comparator runs ($5–30) | Demand for audits (no willingness-to-pay evidence yet) | **Opportunistic; needs publication approval** |

Bounded deliberation (think only on low-confidence items) stays an option for arithmetic families if distillation
fails there; it must be fit on unfloored, independent dev data and costed per logical pass (today's floored gate
routes none of the 74 yes/no items).

## 7. Plan and budget (≈ $200 ceiling, released in tranches)

| Tranche | Contents | Cap | Gate to the next tranche |
|---|---|---|---|
| 0 ($0) | Recover `ssdm-data` (training draws, lockbox manifests, checkpoints) under controlled access; merge the cluster-valid LTT fix; fix the guarantee wording; pick the agent-control workload; preregister P1/P2 rules and kill criteria; port the DAC battery from public suites | $0–5 | Data provenance and lockbox seal verified, or a rebuild plan with a new seal |
| A | P1(a)(b)(c), trusted-only gap, baseline battery on stock and our current best | $50 | P1(b): new-data or deliberation-label arm beats the generator arm on the source-disjoint dev split with a paired CI above 0 and no base-retention loss |
| B | Scale the winning P1 arm (20–100k rows, stronger open teacher such as DeepSeek-V4 or GLM-5.x for what the 4B's own thinking misses; 2 seeds); P2 training only where a gap exists | $100 | Independent transfer confirmed on an untouched split; retention and evidence-retention guards hold |
| Reserve | Independent rescore, untouched confirmation, overruns | $50 | — |

The SF Compute balance ($11.67) is preserved. Grants (Modal academics up to $10k, Lambda research up to $5k) count
only after award. Any split used to steer design becomes development data; final claims use an untouched split.

## 8. Operating model (faster, cheaper, one goal)

- **One lane.** The chief of staff sets the board; worker agents run one hypothesis per cell; no parallel
  descendants of an unvalidated runner; independent rescore before any claim.
- **Per-second GPUs** (Modal is configured on this workstation; RunPod/Vast are cheaper for long cells) with the vLLM
  harness from today's probe. Profile 100–200 steps before any long cell; save a bounded inference artifact at every
  checkpoint; CPU admission tests on the real production path.
- **Evidence tiers.** Exploratory: a run card (hypothesis, kill rule, cost cap, data hashes). Confirmatory:
  preregistration, untouched split, two seeds, independent rescore. Leak gates and pinned scorers always.
- **Honest reporting.** Benchmark-compatible transforms (floors, temperatures) reported beside raw probabilities.

## 9. Decisions needed from you

| # | Decision | Recommendation |
|---|---|---|
| 1 | Objective | Keep the GOAL.md benchmark order until you choose a workload; adopt §0 item 5's differentiator as the direction and pick the agent-control workload it is measured on. |
| 2 | Directive exemptions | Authorize a capped, matched alignment comparison on preregistered common units with H2O, Clef-flash and the MD1 API (this also needs an exemption from GOAL.md's "never re-run MD1"); at most one new-data control cell on H2O weights. |
| 3 | Budget and provider | $200 ceiling on an authorized provider (Modal is ready here), released $50 → $100 → $50 reserve on the §7 gates. |
| 4 | Freeze list | Track L, Q9-AF, RWKV-13.3B product scaling, the Clef/Index chase and repeats of the old H2O data recipe; keep all artifacts; the full-epoch JEV3-LM test stays open. |
| 5 | Data access | Ask Apuroop for `ssdm-data` (draws, lockbox manifests, artifacts) under controlled access; if unrecoverable, approve a fresh train/dev build and a newly sealed confirmation set. |

## 10. What would change this plan

- P1(b) shows no arm beating the generator rows on the source-disjoint split → this recipe fails its gate; capability
  work narrows to the JEV3-LM control and bounded deliberation for specific families; no new tranche without new
  supporting evidence.
- The trusted-only gap is large (trusted-only accuracy < 85% of full) on the chosen workload → certificates are vacuous
  there; P2 becomes empirical robustness plus consistency.
- Comparators already show low adaptive steer ASR at matched units → the alignment wedge narrows to consistency and
  act/defer.
- A board submission (needs approval) shows the public→sealed gap erases local gains → build a larger private split
  before more training.

## 11. What the review changed from v0

Paired point estimates corrected (+29.75, not +29.9; vs H2O +6.55 / +4.35 / +5.45); 9B output tariff corrected (think
Composite 15.8, not 3.0); H2O parity scoped to card totals, tiers and identity probes; timing labelled cache-warm; the
gate claim withdrawn (floored confidences never route yes/no items); "cannot be talked into a decision" and the shared
capability/security mechanism withdrawn; the capability plan changed from relabelling old generator rows to a matched
provenance pilot plus the untested JEV3-LM control; the confirmation design fixed against lockbox reuse; budget line
items reconciled; quantization's Speed effect restated; Quyet's I is not the board maximum (Perplexity 75.7).
