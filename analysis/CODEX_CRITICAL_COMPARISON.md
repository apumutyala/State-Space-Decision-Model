# Critical comparison and language growth decision

> **Rate note (2026-10-09):** the dollar figures here use the 2026-10-06 SF Compute rates ($2.97/h = $0.0495/min batch, $3.96/h interactive) and are left as written. From 2026-10-08 the H100 list price is $5.04/h and batch jobs bill $3.78/h ($0.063/min; Fridays $0.05628/min), so batch figures scale by x 1.27. Source and history: `jobs/common/rates.py`.

6 October 2026. Authoring lane: Codex. Scope: the latest local Claude Code session, its completed reviews, the two saved plans, and targeted primary-source checks. This file is a review and proposal, not an agreement attributed to the concurrently running agent.

## Recommendation

Let Claude own the System 1 decision core and its capability experiments. Codex takes the language-growth path: semantic programs, checking, faithful realization, interpretation tests, and evaluation of what each language extension preserves. Join them through [the proposed interface](../interfaces/DECISION_LANGUAGE_V0.md). Preserve the long-term goal of capable, controlled, interpretable models; do not mistake a good inherited backbone or a checked output language for having already met that goal.

The strongest combined near-term bet is a capable recurrent parent plus a small, causally tested semantic-program interface. A from-initialization model remains a useful scientific comparator for internal design claims. It should not consume most of the $525 merely to rediscover that a small new model lacks frontier knowledge.

## What was actually read

The latest parent session is `C:/Users/apumu/.claude/projects/C--Users-apumu-research-CHLabs/167d8677-872b-45ed-b144-905f7c6dd3bc.jsonl`. Its final coordination message at 2026-10-06T07:24:19Z describes this repository and three new Fable reviewers. The first four reports in `reviews/01` through `04` were complete. Their detailed mathematical and provenance claims were inspected, not accepted solely from the parent summary. The three new reviewers were still researching at the initial snapshot; later arrivals are handled in the addendum below.

The parent model is recorded as `claude-opus-5-5`; the review logs record `claude-fable-5-1`. These identify the local work reviewed, not the exact external benchmark configurations for the user's Astra/Fable target.

## Where the plans are strongest

| Topic | Claude plan | Earlier Codex plan | Combined decision |
| --- | --- | --- | --- |
| Broad decision competence | Stronger survey and a concrete inherited-knowledge route | Small scratch pilot tests mechanisms but poorly predicts frontier competitiveness | Start capability triage on capable recurrent models; retain small models for controlled ablations |
| Attention removal | A concrete Qwen layer-conversion hypothesis | Too little attention to partial conversion | Test one replacement before extrapolating to all layers |
| Semantic language growth | Thin checker and copy-slot sketch | Explicit compositional programs and no free-text bypass | Codex owns this architecture and its proof obligations |
| Authority and provenance | Learned channel labels/gates plus output masks | Stronger dependency, training and cache arguments | Keep learned robustness separate from structural authority |
| Interpretation | Arithmetic traces and layer diagnostics | Causal intervention and semantic-sufficiency criteria | Measure both, but never label arithmetic attribution full interpretation |
| Budget | Practical capability experiments, but stale price assumptions | Verified SF rates and a staged cap, but too much scratch-training allocation | One shared ledger and a joint first $75 gate |

Both plans correctly emphasize wrong permitted actions, calibration under attack, held-out evaluation, and keeping expensive LLC/PID/JEPA machinery off the critical path. Their endpoint remains unproved. The review should improve the experiment, not choose a winner by rhetoric.

## Material corrections to Claude's plan

### The conversion route is plausible but not priced or validated

The Qwen3.5-9B config does contain 24 linear-attention and eight full-attention text layers. That is an independently verified fact. Removing those eight layers still changes precisely the mechanisms that may supply the missing recall/composition. Their fraction of layers does not bound their importance or the recovery cost. The Clef routing head also uses attention, and the model has a vision component; removing text attention alone does not make the full deployed multimodal model attention-free. [Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [Clef head code](https://huggingface.co/Cloudflare/clef-flash/raw/main/joint_schema_model.py).

MOHAWK provides a relevant precedent on Phi, with staged mixing/hidden-state/end-to-end alignment. Its ablations make final-output alignment consequential. It does not validate eight-layer Qwen-to-GDN conversion in 200M tokens or the quoted dollar figure. First measure one converted layer at representative lengths, teacher/student memory, throughput, decision quality and retrieval. Layerwise output matching alone is a diagnostic stage, not evidence of recovered end-to-end behavior. [MOHAWK](https://arxiv.org/html/2408.10189v2).

Clarify the teacher: converting stock Qwen is not the same experiment as preserving Clef's merged adaptation and custom head. State which teacher distributions and interfaces are being retained. A stock teacher result cannot be compared as though it retained the complete Clef system.

### Readable scores and learned gates are narrower than claimed

For z_k = sum_t delta_tk, the summands exactly reconstruct that particular logit. Each delta may depend on the whole preceding hidden history, so this is not the causal effect of token t, nor a guarantee of truthful evidence. Perturb the evidence, recompute the full recurrence, and compare predicted versus actual decision changes. Explain which source supplied a checked derivation separately from why the neural model selected it.

A learned channel embedding or gate is an empirical robustness mechanism. If untrusted text already affected shared state, a later trusted gate can still read contaminated content. Exact authority needs an explicit dependency restriction or external trusted state; label embeddings are not that restriction. Training-time shared embeddings, global clipping and normalization need the same audit if noninterference is claimed.

Snapshot forks may improve recall, but do not eliminate lossy compression. If snapshots are taken every fixed number of tokens, memory grows with sequence length. If only S snapshots are retained, memory is bounded in length but recall remains finite. Compare the cost and retrieval failures explicitly. An ordered schema prefix still defeats structural option permutation equivariance unless canonicalization or a symmetric construction makes the common prefix independent of permutation.

### Language guarantees must concern relations and whole meanings

The proposed P2 copied-slot guarantee proves a property of copied values. It does not establish the relation between those values. The review's extra requirement that a semantic mapping be injective does not repair this: a one-to-one mapping can assign every sentence a distinct unsupported proposition. What is needed is a sound rule mapping checked premises into the complete asserted meaning.

Free connective text is not semantically harmless. Words such as because, only, all, still, or no can introduce causal, exclusivity, temporal or negative claims. A learned verifier adds empirical detection; it cannot inherit the exact checked-language guarantee. The fraction of tokens in a constrained region is not a valid substitute for claim/answer-level soundness. Keep the initial output entirely within typed, meaning-preserving productions.

Using the same hidden state for a decision head and a generic LM head does not establish language grown through the decision contract. The language planner must causally use the typed interface. The renderer receives the checked program, not unrestricted state or raw evidence. A strong LM can propose programs or licensed paraphrases, but its proposals still pass the same contract.

### Benchmark and cost claims need tighter scope

A 5,000-row sample cannot reproduce published full-benchmark aggregate scores without the same cohort and per-item predictions. Distinguish adapter parity on shared fixtures from model-score reproduction. A test-suite sample used repeatedly for development is no longer an untouched final evaluation.

The official runner does not impose a blanket ban on legitimate training-split fine-tuning. The earlier Codex plan was too restrictive in presenting zero-shot status as the benchmark contract. Use the actual rules, disclose training overlap, and separately measure unfamiliar schemas/domains. [Decision Index runner](https://github.com/apolinario/decision-index).

The public SF rate card currently lists $2.97 per H100-hour for batch work and $3.96 interactively. Claude's 112-hour Stage 0 is about $332.64 at the batch rate before overhead; retaining its $245 reserve would total $577.64, above $525. The two agents must not each allocate a separate $525. [SF pricing](https://autoresearch.sfcompute.com/).

Do not infer prefill performance, adapter feasibility or optimizer memory from weights fitting in 80GB. A 13.3B BF16 weight tensor is approximately 26.6GB in decimal units, but teacher/student copies, activations, recurrent states and trainable optimizer state are additional. Qualify the exact inference/training path before adopting RWKV as the fallback.

## Corrections to the technical review itself

The reviews contain useful numerical counterexamples but occasionally promote numerical agreement to a theorem.

1. **Stability.** Spectral radius below one at every step is not a common contraction certificate for switching matrices. In contrast, an actual common induced-norm bound ||A_t|| <= rho < 1 gives ||A_T ... A_1|| <= rho^T. A nonnormal switching example does not disprove this theorem or fixed-basis diagonal contraction. Full-model semantic safety remains a separate issue.
2. **Bayes calibration.** Reusing marginal likelihood ratios for redundant observations is not using the correct conditional likelihood ratios. The simulation refutes the naive additive evidence model, not Bayes' rule. The correct conclusion is that a learned integrator is not automatically calibrated.
3. **Probability losses.** Label smoothing and reward bonuses can change the optimum away from true class probabilities. Matching Clef's recipe is an experimental arm, not a calibration guarantee. Compare unsmoothed proper losses and held-out calibration directly.
4. **Grammar complexity.** A DFA transition for one token can be constant time; computing all allowed subword tokens, avoiding dead ends and enforcing semantic state need not be. Bound grammar states, vocabulary work and program depth in the actual decoder.
5. **Circuit inference.** Linear time in a compiled circuit's size does not imply polynomial compilation, and exact marginals do not automatically supply every MAP query. Restrict the circuit/query class if used.

### A concrete issue with the claimed universal learning coefficient

The review states that coincident poles do not lower the proposed lambda = r + (n-r)/2. That assertion is not established by the reported Monte Carlo examples. There is a simple local warning case.

Take a two-mode real SISO response g_t = a1 lambda1^t + a2 lambda2^t representing a one-mode truth a* lambda*^t. Near a positive-weight coincident representation, write total amplitude a, weight w in (0,1), mean pole m, and pole difference d:

    a1 = a w;        a2 = a (1-w)
    lambda1 = m + (1-w)d; lambda2 = m - w d
    g0 = a
    g1 = a m
    g2 = a m^2 + a w(1-w)d^2

For squared error on these first three impulse coefficients, near the matching truth the loss is locally equivalent to u^2 + v^2 + d^4, with w a flat coordinate and a smooth positive prior. Its local integrability exponent is 1/2 + 1/2 + 1/4 = 5/4, rather than the proposed 3/2 for n=2,r=1. This is a restricted local derivation, not a general RLCT formula for trained SSMs; it is enough to require a careful singular-stratum analysis before declaring a universal equality. Retain the original hypothesis as unresolved, not a corrected theorem that simulation has proved.

## What I change in my own plan

- The $525 scratch-heavy campaign tested a real prerequisite but allocated too much money before a capable parent was selected. I now prioritize Claude's capability triage, a small conversion diagnostic if warranted, and a bounded language extension on the selected parent.
- A pretrained Mamba-3 SISO 1.5B checkpoint is now directly verified. My prior repository-only survey was incomplete. Its card specifies no attention layers, 100B pretraining tokens and a 2,048-token context; actual loading and tokenizer access still require qualification. [Model card](https://huggingface.co/state-spaces/mamba3-siso-1.5b).
- From-initialization control and inherited-backbone control must remain distinct claims. A successful language contract around a pretrained parent does not retroactively establish interpretable pretraining. The branch can validate modular language growth without pretending to finish the whole internal-interpretation objective.
- Small models remain useful for semantic sufficiency and causal experiments. Weak small-model leaderboard entries do not logically invalidate that research question, but they are not evidence that the small model will become the competitive product.
- My earlier language plan needed a concrete integration contract, task-relevance checks, unit/time rules and a realistic sequence of learnable operations. Those are now specified in the language plan and interface.

## Decision for this collaboration

Claude's parent selection/conversion and Codex's semantic-language work should meet at a backend-independent interface. The first language experiment starts with oracle and replay backends locally; the learned condition then calls the actual System 1 checkpoint. No new independent foundation model is trained merely to unblock language work. An opaque parent remains explicitly opaque until causal internal evidence improves that assessment.

The interface and spending proposal are offered for Claude's review in [the coordination note](../coordination/CODEX_LANGUAGE_HANDOFF.md). Its existing documents remain unchanged, and this review does not claim its pending reviewers have endorsed the proposal.

## Later review arrivals

At this file's initial creation, the three second-wave Fable reports were not yet saved. Their conclusions must be reconciled when they arrive; absence here is not disagreement or a claim that they completed.
