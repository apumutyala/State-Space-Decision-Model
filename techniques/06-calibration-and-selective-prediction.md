# 06 — Calibration and selective prediction at the decision level

Written 2026-10-06. Same legend as `01`. Already settled by the reviews and not re-argued: calibration is not an architectural property (`reviews/02` §3); the PID-on-ECE loop is dropped (§9: 15-bin ECE of a perfectly calibrated model is 0.057±0.015 at n=256); abstention scores as wrong on the Index (§8); conformal thresholds fit on clean data give 11.7–20.2% realised risk under attack at nominal 5% (`reviews/03` §1.2); P(error | singleton) ≠ α (2506.21802).

**Headline:** (1) Post-hoc temperature scaling with a *predicted* temperature (Thermometer / Adaptive Temperature Scaling) is the published form of the plan's "per-question temperature with a length feature". (2) Report debiased ECE (Kumar et al.) or SmoothECE, never binned ECE. (3) For selective deployment, the 2025–26 results say: monotone calibration cannot improve selective classification because it does not reorder (2510.20242); the optimal rejection rule under covariate shift is a likelihood-ratio test (2505.15008, ICLR 2026); evaluate with AUGRC not AURC-at-a-threshold (2407.01032); control selective risk with Selective CRC / LTT, with role- or family-stratified budgets, and expect coverage of ~40% under attack shift (lab). (4) Option-ID bias (PriDe) is zero by construction in a permutation-equivariant head — verify numerically, don't assume.

---

## Techniques

### 6.1 Temperature scaling and its multiclass extensions
- **Cite:** Guo et al. 2017 (arXiv 1706.04599, from memory **[ID unverified]**); Dirichlet calibration with ODIR regularisation, Kull et al., arXiv 1910.12656 (NeurIPS 2019); "The Well-Tempered Classifier: elementary properties of temperature scaling", arXiv 2602.14862 **[not fetched]**; classifier calibration survey, arXiv 2112.10327.
- **Evidence:** R.
- **Known:** one parameter, accuracy-preserving; Dirichlet/matrix scaling helps when classwise miscalibration differs; lab: ECE 4.61% → 1.80% on BANKING77 with TS.
- **Failure modes:** a single global T cannot fix *length-dependent* miscalibration (review §3: stateless increments are under-confident short, over-confident long); degrades under shift (6.4).
- **Cost:** $0.
- **Serves:** both.

### 6.2 Predicted / adaptive temperature (per task, per input, per token)
- **Cite:** Thermometer, Shen et al., arXiv 2403.08819 (ICML 2024): an auxiliary model maps LLM features → task temperature, trained across tasks, transfers to new tasks without labels; Adaptive Temperature Scaling, arXiv 2409.19817 (EMNLP 2024): per-token temperature from token-level features, fit on the SFT set; lowest ECE/Brier on all downstream benchmarks for Llama-2-7B-Chat.
- **Evidence:** S each; the idea is R (parametric/local TS has several antecedents).
- **Known:** accuracy-preserving (argmax unchanged); temperature as a function of features absorbs systematic effects (task, length, option count).
- **Failure modes:** the feature model can overfit the calibration split; needs held-out schemas to measure transfer.
- **Cost:** $0–$.
- **Serves:** decision model (Index metric) and LM (plan-decision confidence).
- **How we'd use it:** T(q) = MLP over [record length bin, option count, question type embedding, evidence-count proxy from the integrator] fitted by NLL on held-out synthetic schemas; audit split untouched; report ECE per length bin. This is the plan's lever, now with two citations.

### 6.3 Calibration-error estimation (what to report)
- **Cite:** Kumar, Liang, Ma, "Verified Uncertainty Calibration", arXiv 1909.10155 (NeurIPS 2019): debiased binned estimator; Roelofs, Cain, Shlens, Mozer, "Mitigating Bias in Calibration Error Estimation", arXiv 2012.08668 (AISTATS 2022): ECE_bin biased, worst for calibrated models; ECE_sweep / ECE_debias; Błasiok & Nakkiran, "Smooth ECE", arXiv 2309.12236: consistent, hyperparameter-free; Lp canonical calibration estimator, arXiv 2210.07810 **[not fetched]**.
- **Evidence:** R.
- **Cost:** $0.
- **How we'd use it:** report ECE_debias (Kumar) and SmoothECE with bootstrap CIs; never gate on ECE below the estimator's noise floor (review §9 table); reliability diagrams per length bin and per question type.

### 6.4 Calibration under shift
- **Cite:** Ovadia et al., arXiv 1906.02530 (NeurIPS 2019): post-hoc calibration fails under dataset shift; weighted conformal under covariate shift, Tibshirani, Foygel Barber, Candès, Ramdas, arXiv 1904.06019; robust validation under f-divergence shift, Cauchois, Gupta, Ali, Duchi, arXiv 2008.04267; Adaptive Conformal Inference (online), Gibbs & Candès, arXiv 2106.00170; "Phases of Uncertainty: confidence–calibration dynamics" (UncertaiNLP 2025, from search) **[not fetched]**.
- **Evidence:** R.
- **Known:** TS fit in-distribution becomes miscalibrated under shift; weighted CP needs a density ratio; robust CP needs a shift budget; ACI needs online feedback.
- **How we'd use it:** the Index is zero-shot across 38 benchmarks (review §8), so calibrate on synthetic schemas stratified by family and report *per-family* reliability; deployment uses ACI with feedback where labels arrive; attack-goal shift is treated as a stratum (lab: calibrate on attacked data, expect 38–46% coverage).

### 6.5 Learn-then-Test, Conformal Risk Control, Selective CRC, small-calibration variants
- **Cite:** LTT, Angelopoulos, Bates et al., arXiv 2110.01052; CRC, arXiv 2208.02814 (ICLR 2024; monotone losses; extensions to shift, quantile risk, adversarial); **Selective CRC, Xu, Guo, Wei, arXiv 2512.12844**: abstain on low confidence and control risk on the accepted subset; SCRC-T exact finite-sample (joint thresholds over calibration+test), SCRC-I PAC-style; Batched-Bootstrap CRC / Randomised Batched Weighted-Average CRC for small calibration sets, arXiv 2509.23007; role-stratified CRC, arXiv 2607.24343 (`05` 5.9); singleton error rate for binary CP, arXiv 2506.21802 (from `reviews/02`).
- **Evidence:** R (LTT/CRC); S (SCRC; BB-CRC; role-stratified).
- **Known:** CRC controls E[loss] of any monotone loss with split-conformal guarantees; SCRC is the formally correct way to say "act iff confident" with a bound on *selected* risk R = E[AE]/E[A] (partner plan's formula); stratification costs sample size (role prevalence p_r ⇒ effective budget α·p_r without stratification).
- **Failure modes:** exchangeability breaks under adaptive attack and zero-shot family shift (lab A3 result); zero-error certification needs ~2,995 accepted cases for a 0.1% one-sided 95% bound (partner plan); correlated cases must be clustered.
- **Cost:** $0.
- **How we'd use it:** deployment-only (off-Index): SCRC on selective risk with Mondrian strata = schema family × attack-goal; BB-CRC when strata are small; publish error–reject (AUGRC) curves; `chl-slt-experiments/chl/slm/envelope.py` already implements LTT + Clopper–Pearson.

### 6.6 Selective classification: scores, training, evaluation
- **Cite:** SelectiveNet, Geifman & El-Yaniv, arXiv 1901.09192; Deep Gamblers, arXiv 1907.00208; softmax-response is a strong baseline across 523 ImageNet classifiers, arXiv 2302.11874 **[not fetched]**; **"Know When to Abstain: Optimal Selective Classification with Likelihood Ratios", Heng & Soh, arXiv 2505.15008 (ICLR 2026)**: Neyman–Pearson view, LR test optimal, strongest under covariate shift; **Rabanser & Papernot, "What Does It Take to Build a Performant Selective Classifier?", arXiv 2510.20242**: five-term finite-sample gap decomposition (Bayes noise, approximation, ranking, statistical, shift/implementation slack); monotone calibration rarely helps because it does not reorder; feature-aware calibrators and DRO training do; **AUGRC, Traub et al., arXiv 2407.01032**: area under the generalised risk–coverage curve = average risk of undetected failures; changes metric rankings on 5/6 datasets; "Selective Classification Under Distribution Shifts" (TMLR 2024, generalised SC with label/covariate-shift error types) **[ID not found]**.
- **Evidence:** R (softmax response strength; AUGRC adoption growing); S (LR-optimal; gap decomposition).
- **Known:** the selection *score* must rank errors below correct answers; temperature cannot change that ranking; features that can (order-agreement, margin, evidence count, ensemble disagreement) are where the gain is.
- **How we'd use it:** selection score = learned reorderer over [max prob, margin, order-flip agreement (lab), evidence-count, 3-member LoRA ensemble disagreement if budgeted], fit on held-out schemas; LR-test variant needs a density-ratio estimate — try the SSM state's final-norm/entropy as a proxy and report whether it reorders; evaluate with AUGRC + risk–coverage at fixed coverages; report the five-term budget where estimable.

### 6.7 Multiple-choice selection bias and permutation debiasing
- **Cite:** PriDe, Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", arXiv 2309.03882 (ICLR 2024): token prior on option IDs; estimate prior by permuting on a few samples, divide it out; "Calibration of Selection Bias in MCQ" (ACL 2025, from search) **[not fetched]**; permutation-aware GRPO, arXiv 2603.21016 **[not fetched]**.
- **Evidence:** R (selection bias is replicated across 20+ LLMs).
- **Known:** letter-token priors dominate order effects; PriDe is label-free and transferable.
- **How we'd use it:** our heads score option *content*, not IDs, so the prior should be exactly uniform; the lab found order flips of 0–1.2% from bf16 noise — keep the order-agreement test and a PriDe-style permutation audit as a unit test of equivariance, not as a fix.

### 6.8 Decision calibration (utility-level)
- **Cite:** Zhao, Kim, Sahoo, Ma, Ermon, "Calibrating Predictions to Decisions", arXiv 2107.05719 (from memory **[ID unverified]**).
- **Evidence:** S.
- **Known:** calibrate so that a decision-maker's *expected utility under the model's probabilities* matches realised utility for all decision rules in a class — the right notion for act/abstain and for the partner plan's "probability estimation separate from action utility".
- **Serves:** both (deployment).

---

## How we'd use this domain

**Index (forced choice):** train LS-CE + Brier (Clef recipe); post-hoc predicted temperature (6.2) on held-out synthetic schemas; report ECE_debias/SmoothECE per length bin and per family (6.3); equivariance audit (6.7). No abstention.

**Deployment / LM plan decisions:** SCRC with family×goal strata and BB-CRC for small strata (6.5); learned reordering score, LR-test variant as an arm (6.6); AUGRC + risk–coverage reporting; ACI online where feedback exists (6.4); decision calibration for act/abstain utilities (6.8); sample-size statement for any "≤0.1% accepted error" claim.

## Could not verify this session
- IDs for Guo 2017, Zhao 2021, the TMLR generalised-SC paper, 2302.11874 details, 2602.14862, 2210.07810, ACL 2025 selection-bias paper, 2603.21016.
