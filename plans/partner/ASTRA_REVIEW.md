# Astra review of the decision first SSM proposal

5 October 2026. This is the coordinator's edited synthesis of the requested Astra review, incorporating the user's final clarification. The reviewer read both pasted plans and relevant primary/local research. This is a conceptual and mathematical audit, not an independent reproduction of the cited experiments.

## The corrected direction

The intended lineage is coherent: a capable controlled System 1 decision core, followed by increasingly expressive semantic programs, followed by language that faithfully realizes those programs. The first target includes Clef, Clef-flash, Jev, Astra, and Fable on specified decision tasks. The second grows from that same model. A separately trained unrestricted LLM with a checker added afterward would not satisfy this request.

Keep the original proposal's typed probabilities, state-space architecture, semantic interfaces, evidence tracking, and verified-emission ambition. Replace its unsupported guarantees and slow architecture ladder. Start with a capable selective SSM and train the controls and semantic interfaces from initialization. Expand what the model can express and verify together.

## Errors that change the engineering plan

| Draft claim or design | Problem | Repair |
| --- | --- | --- |
| Schema validity means no hallucinations | A permitted answer may be false or misleading | Separate type validity, evidence entailment, real-world truth, and task correctness |
| Small supervised BANKING77/CLINC wins establish competitor superiority | Training regime and task coverage differ from zero-shot general evaluation | Keep full-suite and unseen-schema comparisons, with adapted tasks separately labelled |
| The CLINC gap identifies an OOS defect | Aggregate macro-F1 does not identify the failing classes or adapter | Inspect per-class errors before choosing the remedy |
| A 10M–300M ladder is the route to beating frontier decision capability | Those scales are pilots; the plan omits a validated source of broad competence | Develop language/knowledge representations, multi-task decisions, and measured scale-up |
| A pretrained opaque backbone with readable head meets the requirement | Interpretation covers the head, not the full model; inception claim is false | Use it as a competence reference and train the primary controlled architecture from initialization |
| All information crossing h_sem is meaningful to humans | A computational cut does not identify its semantics | Supervise and intervene on concepts; test completeness, leakage, causal mediation and bypasses |
| Per-mode stability gives semantic safety | Nonlinear mixing, gate changes, autoregressive feedback and immediate actions are outside that bound | Scope numerical bounds to the actual recurrence; enforce authority separately |
| A fork scorer is structurally permutation-equivariant | Its ordered schema prefix and option-specific biases can change every score | Symmetric set summary, shared scorer, no positional IDs; test mapped permutations |
| All joint constraints can be compiled into efficient circuits | Compilation size, treewidth, and some MAP queries can be intractable | Restrict the first constraint language and measure inference complexity |
| Inspectable evidence increments are calibrated Bayesian evidence | Learned increments are not known conditional likelihood ratios | Treat as a discriminative model; test probabilities with proper losses and independent calibration |
| Calibration can be regulated by an ECE PID and temperature | ECE is unsigned/nonmonotone; repeated calibration use causes adaptation | Begin with held-out NLL temperature fitting and a separate audit |
| Contraction supplies SNGP's bi-Lipschitz features | A constant map is contractive and destroys all distances | Do not inherit a lower distance bound; evaluate SNGP only if needed |
| Entropy is aleatoric uncertainty | Entropy does not by itself separate uncertainty sources | Report predictive entropy without that decomposition |
| Copying KB strings through a grammar proves grounding | Copied values can form a false relation or unsupported negation | Check relation-level proof objects and meaning-preserving composition |
| Safe-token fraction measures language guarantees | One unsafe token can reverse a complete proposition | Measure claim-, answer-, and transaction-level coverage |
| Low LLC or Hankel rank establishes interpretable alignment | They characterize different mathematical objects under different assumptions | Retain as optional narrow research; remove from the critical path |
| No accuracy cost can be asserted from the architecture | The semantic representation may discard information needed for the answer | Make capability preservation a measured acceptance gate; enrich the interface when needed |

## Mathematical corrections

### Semantic sufficiency is the central hypothesis

If the decision only observes C(X), preserving the unrestricted Bayes predictor requires, for the target distribution,

    P(Y | X) = P(Y | C(X)).

A small hand-authored concept list need not satisfy this. Naming or sparsifying coordinates does not make the equality true. An opaque residual channel may recover accuracy while defeating complete concept interpretation. The right experiment compares learned concepts, oracle concepts, and a richer representation on the same task, then expands the semantic interface when its expressivity is the problem.

### Accepted error has a denominator

For acceptance A and error E,

    E[A E] = P(A = 1) P(E = 1 | A = 1).

A model can act on 1% of cases and be wrong on every one: acted error is 1%, accepted error is 100%. The original selective-loss argument therefore does not prove its claim. Ordinary conformal coverage also does not automatically imply singleton-accepted accuracy. Select thresholds with a valid multiple-testing procedure or on independent data, then bound error on accepted cases with enough independent observations. [Learn then Test](https://arxiv.org/abs/2110.01052).

### The Bayes update is conditional on true likelihoods

The identity

    log p(k | x_1:t) = log p(k | x_1:t-1)
                        + log p(x_t | k, x_1:t-1) - log p(x_t | x_1:t-1)

does not make a learned recurrence a calibrated Bayes filter. Its increments would have to be the specified likelihoods. Proper scoring encourages accurate probabilities under population/model assumptions; it does not certify them for every input. Question-first scanning is a useful design, not a theorem requiring input-dependent gates.

### Control quantities need the right system

Closed-form LTI Gramians and balanced truncation bounds apply to the declared linear input-output system. Nonlinear stacks and input-dependent gates need a separate argument. Under a state change h' = T h, the Gramians transform as P' = T P T* and Q' = T^(-*) Q T^(-1); their minimum eigenvalues are not coordinate-free measures of semantic reachability.

Input-dependent B(x)x creates nonlinear additive input. A bilinear term sum_j x_j N_j h requires input-dependent A. Checking endpoints of a scheduling variable does not automatically certify nonlinear exponentials and all cross terms. A cross-Gramian block is not an induced private-to-public transfer norm. Infinite-horizon controllability is not bounded-actuator, finite-horizon reachability. A CBF on a learned latent score does not establish that its safe set excludes unwanted language.

Use contract enforcement for permissions and effects, and causal intervention tests for semantics. Preserve slow memory where useful; forgetting is neither necessary nor sufficient for resisting prompt injection.

### The LLC and Hankel bridge is unproved

Hankel rank characterizes minimal LTI realization. RLCT/LLC depends on likelihood, parameterization, prior, data distribution and singular stratum. Factorizing a Hankel matrix does not turn the SSM into unrestricted reduced-rank regression. A local inactive product bc can illustrate a singularity without establishing universal half-unit jumps for repeated poles, mode collisions or complex states.

Likewise, a transfer-function error bound from balanced truncation is not the equality loss = 4 sigma^2. The proposed sqrt(log n / (8n)) universal truncation threshold does not follow. These hypotheses belong in a small independent mathematical study if they prove useful; they do not gate the model build.

## How verified language can grow without a bypass

Expand decisions into a typed semantic program algebra: assertions, checked inference, comparisons, explanations, questions, quotations, hypotheses, and action plans. Give each constructor a defined meaning, evidence dependencies, permissions, and failure behavior. Composition preserves those obligations rather than only checking field types.

The realizer receives a checked program and licensed discourse choices. Start with templates, then compositional grammar, then learned choices within sound semantic rules. A free string slot or a fallible entailment classifier cannot inherit the hard guarantee. Richer meaning must enter through a new verified module or an explicitly unmet obligation.

Separate expression from discovery. An oracle program tests whether the language can express the correct answer. Oracle evidence tests whether the planner can derive it. Learned evidence tests perception. Own-history rollouts test composition under the errors the model will actually make. Improve the failing component while preserving the contracts.

Truth requires sound and current evidence. Alignment also requires a specification that captures intended behavior, including misleading omissions, framing and consequences. Those are not solved by formal typing. For the covered formal semantics, proof-carrying answers can rule out unsupported assertions. Broader open-world correctness and value alignment remain research targets.

## What the old experiments contribute

The [October 1 results](../../overnight/LATEST_PILOT_RESULTS_20261001.md) demonstrate scoped interfaces and a concrete capability deficit in one design. They do not disprove the new thesis. They warn against declaring masks or loss improvements sufficient: source separation lost 12.24 trajectory points against ordinary modularity, and prospective reweighting improved a loss without improving decisions. The recurrent extension did not establish an architecture advantage.

Use those implementations and failure examples as supporting evidence. The main research agenda remains the user's decision-to-language architecture, with full competitive capability and meaningful interpretation measured together.

## Recommended first decisive result

A useful System 1 SSM should solve unfamiliar typed decisions, preserve specified controls, and expose semantic edits whose effects it predicts correctly. It should then compose a modest semantic program and realize it faithfully without losing decision competence. Only then scale the same lineage toward the complete competitor target and broader language.

Neither categorical pessimism about the $5,000 budget nor a promise of victory is justified. The budget can buy discriminating experiments. Preserving intelligence and increasing control is the acceptance criterion; it is not a theorem the proposal already possesses.
