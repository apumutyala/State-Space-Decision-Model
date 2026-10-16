# Verifier adequacy before small-model RL

## Thesis and frozen protocol (2026-10-16)

Before paying for RLVR, audit the actual reward functions against an independent answer contract. This is a synthetic-input, real-library measurement, not a model experiment or a claim about natural error prevalence. No Keel or JevBench data are used.

**Targets, selected from public task descriptions before reading scorer implementations:** Reasoning Gym 0.1.25 `score_answer` for `basic_arithmetic`, `decimal_arithmetic`, `gcd`, `lcm`, `prime_factorization`, `number_sorting`, `polynomial_equations`, and `syllogism`; Hugging Face `math-verify==0.9.0` default `parse`/`verify`; OpenAI GSM8K's original answer extractor plus exact equality. Ten endpoints need not mean ten independent implementations. Package interfaces may share code; report that rather than inflate replication.

**Independent fixtures and labels:** 64 seeded base problems per endpoint, authored from task specifications, not selected by scorer success. Exact arithmetic uses `fractions.Fraction`; factorization uses trial division; ordered lists retain multiplicity; polynomial root sets use constructed factorizations and deduplicate; categorical syllogisms exhaust all 255 nonempty sets of three-predicate membership types (complete for these categorical formulas, including empty predicates). Candidate strings come from a restricted grammar with typed denotations and exact-equivalence labels, not from the tested verifier. A mutation that preserves value remains *valid*. Missing unique answers and extra false assertions are separately flagged; an empty root set is legitimately valid. `native`/`alternate`/`extra_content` are presentation categories, not claims that the upstream API promises every semantic format. Rejecting a valid but forbidden format is not automatically a bug. Decimal answers are compared after 12-significant-digit half-up rounding; roots after four-decimal rounding. Fixtures use exact decimal sums and integral roots, avoiding irrational root-label error and rounding-tie ambiguity.

**Family split fixed before implementation inspection:**

| Development (7) | Once-only held-out evaluation (7) |
|---|---|
| boundary arithmetic (+/−1) | duplicate members (valid for sets, invalid for lists/multisets) |
| sign/polarity flip | contradictory trailing assertion and neutral suffix |
| partial answer / omitted members | percentage/unit scaling (value-changing and equivalent controls) |
| multiple candidate answers | LaTeX fractions that change or preserve value |
| quantifier flip in categorical assertions | numeric-tolerance edges |
| whitespace (valid control) | alternative delimiters (valid control) |
| equivalent rational notation (valid control) | scientific notation (valid control) |

Only applicable endpoint × family cells are scored; absent denominators are null, not zero error. Baseline canonical controls accompany each split. No held-out scorer outputs are inspected before the final implementation/method freeze commit. Development may repair adapters and label/rendering defects, not select favorable endpoints. Held-out evaluation runs once; any failure or amendment is disclosed rather than silently rerun.

**Measurements:** primary false acceptance = fraction of independently incorrect answers with score exactly 1; false rejection = fraction of independently correct answers without score exactly 1. Secondary wrong-answer positive reward is *not* called full acceptance. Record exceptions separately and count them as nonacceptance. Report every endpoint × family with counts and 95% base-problem-cluster percentile bootstrap CIs (2,000 resamples; all variants of a problem remain together). Bootstrap intervals at zero/one can collapse; also report a Wilson interval on the base-problem indicator of any error to avoid implying certainty. These intervals condition on the hand-authored grammar, not on an unseen domain population. No pooled rate across heterogeneous verifiers is primary. Any paired contrast uses shared problem resamples. The development adequacy vs held-out adequacy association, if estimable over only ten dependent endpoints, is descriptive and does not establish prediction of learned reward hacking.

**Execution:** CPU only, `nice -n 10`, no-network bubblewrap, ≤60 minutes and ≤2 GB per job. Raw items, pinned source downloads and disclosure notes stay under `$SSDM_PRIVATE/research-verifier-adequacy/`. Commit only this README, experiment code, and compact aggregate JSON. No public issue filing.

## Prior work (opened 2026-10-16)

- Stojanovski et al. (2025), [Reasoning Gym](https://arxiv.org/abs/2505.24760): procedural generators and verifiers; [task gallery](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) specifies the tested answer interfaces.
- Rajan (2026), [Auditing Reward Hackability in Code RL Training Environments](https://arxiv.org/abs/2606.16062): incorrect code patches can pass training tests; not reproduced here.
- Gabor, Lynch and Rosenfeld (2025), [EvilGenie](https://arxiv.org/abs/2511.21654): coding reward hacking and detection, not this math/logic mutation design.
- Huang et al. (2025; [v3, 27 August 2026, full text opened](https://arxiv.org/html/2505.22203)), [From Accuracy to Robustness](https://arxiv.org/abs/2505.22203): rule-based mathematical verifier false negatives and learned-verifier reward hacking already directly cover the general thesis, including static probes and RL experiments.
- Helff et al. (2026), [LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149): extensional logical verification rewards shortcuts; isomorphic checks change the training incentive.
- [Math-Verify documentation](https://github.com/huggingface/Math-Verify) explicitly describes best-effort extraction, unit normalization, numerical tolerance and stricter RL normalization. Those documented behaviors are contract limitations, not automatically newly discovered bugs.
- Cobbe et al.'s [original GSM8K repository](https://github.com/openai/grade-school-math), archived, documents extraction after `####`; the [opened implementation](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py) is pinned to commit `3101c7d5072418e28b9008a6636bde82a006892c`, SHA256 `c22b81bccaaebfd1123be296e8b3c3dca3ddc704c46b70bf4a50603bb627e197`. The runner executes its original grading functions, selected by AST, not a reimplementation or its unrelated Torch training class.

## Implementation freeze and development run

The family/specification freeze is commit `c43eb0f3`; the executable freeze is the commit immediately preceding the held-out run, also recorded in the result JSON. Scorer source was read **after** the family freeze, for adapters and diagnosis. Thus this is an output-held-out family split, not a code-blind independent red team. No endpoint, family, threshold, or fixture was removed after development.

`dev01` ran once: **6,936 calls**, all 640 canonical controls accepted, **2.40 s / 146.36 MiB peak RSS**, CPU/no-network, $0. The run revealed sorting-value substitution, decimal-precision mismatch and empty-root parse acceptance; it is discovery evidence, not confirmation. No development implementation repair was needed. `basic_arithmetic`, `gcd`, `lcm` and `syllogism` share `ProceduralDataset.score_answer`, leaving **seven distinct implementations** across ten real task endpoints.

The final numeric grid is `10^-12, 10^-8, 10^-4, 10^-2, 1/2, 1, 2`; the same positive shift is applied to every element. Other held-out variants are: prepend a duplicate first member; append `; also 0 = 1` or `; exact.`; literal/equivalent percentages; boxed half-value/equivalent fractions; parentheses/unquoted-list/asterisk/semicolon/lowercase delimiters; and exponent-zero versus compensated exponent-one notation. For no-real-root fixtures (8/64), empty-string controls are valid. Family applicability is encoded in `reference.py`; the complete labelled records stay private.

The runner verifies exact library pins plus source hashes, logs every exception, enforces 2 GiB address space/3,500 CPU seconds, and creates an exclusive held-out-start marker before scoring. Bootstrap CIs concern full-credit errors. Fractional reward on a correct string is a false rejection at this operating point, **not necessarily zero reward**; positive reward on a wrong string is reported separately. Canonical controls are excluded from mutation rates. The base fixtures are shared across splits to isolate family shift; this is not held-out task generalization. Labels are exact within our authored grammar, but rendering/reference code was written by one AI researcher and has not had independent human review.


## Next step and cost

Conditional on useful, independently validated bugs and owner approval: a small custom policy RL run against original versus hardened reward at equal updates and matched seeds, capped at roughly **$3.95** (one H100-hour under the portfolio tariff). No paid run is authorized or performed here. A mutation-suite pass is not proof of soundness; training transfer, clean utility and held-out exploit discovery would be the outcomes, not reward alone.
