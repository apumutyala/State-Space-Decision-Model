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

## Held-out results (`held01`, executable freeze `6bfc494b`)

- **Run:** 9,392 scored rows over 10 endpoints, which share 7 distinct implementations.
- **Controls:** every canonical control was accepted (64/64 per endpoint).
- **Cost:** 2.1 s on CPU.
- **Determinism:** a second run into a scratch directory reproduced the aggregate exactly, timing aside.
- **Data:** aggregates are in [`results.json`](results.json); per-row records stay private.

**False acceptance** means an independently wrong answer received full credit. Nonzero cells only; every other
applicable endpoint × family cell was 0/n.

| Endpoint | Held-out family | Wrong answers with full credit | Minimal example | Cause (source read after the freeze) |
|---|---|---|---|---|
| Reasoning Gym `number_sorting` | numeric tolerance | 382/448. By shift: every level up to 1/2 64/64, 1 62/64, 2 0/64 | gold `["-92.49", "8.45", …]`; every element shifted by up to 1 still scores 1.0 | an absolute tolerance of 1 per element ("Increased tolerance to handle decimal differences") on values with two decimals |
| Reasoning Gym `decimal_arithmetic` | numeric tolerance | 129/385, shifts ≤ 10⁻⁴ only | `14.4331` for `14.433` | documented tolerance of one unit in the last decimal place |
| Reasoning Gym `polynomial_equations` | contradictory trailing text | 8/64, exactly the no-real-root fixtures | `; also 0 = 1` for the empty root set | text without numbers parses as "no roots" |
| `math-verify` 0.9.0 | units / percentages | 64/64 | `485%` for `485` | percentage normalisation (documented) |
| GSM8K original extractor | units / percentages | 64/64 | `#### 485%` for `#### 485` | the regex keeps digits only |
| GSM8K original extractor | contradictory trailing text | 64/64 | `#### 485; also 0 = 1` | extraction ignores everything after the number (documented) |

**Partial credit.** Reasoning Gym's default `ProceduralDataset.score_answer`, shared by `basic_arithmetic`, `gcd`,
`lcm` and `syllogism`, pays `len(gold)/len(answer)` to any answer that *contains* the gold string. Of the wrong
strings on those endpoints, 80% received positive reward, which in RL pays the policy for hedging with several
candidates.

**False rejection** means a valid variant was denied full credit. It is pervasive but mostly by contract:
- The shared Reasoning Gym scorer requires an exact string. It rejected 100% of valid alternative delimiters, LaTeX,
  scientific notation, neutral suffixes and units.
- `math-verify` rejected alternative delimiters (64/64) and half of the scientific-notation controls (64/128).
- The GSM8K extractor rejected LaTeX, units and delimiters.

These cases add reward noise for valid answers but cannot be exploited.

## Verdict

- **Useful as a free pre-RL gate.** It found, in seconds, one high-severity acceptance:
  - Reasoning Gym `number_sorting` gives full credit to sorted lists whose values are all wrong by up to 1.
  - Partial credit also rewards hedging.
  - Several documented normalisations accept value-changing answers.
  - Anyone training on these endpoints should harden or replace these scorers first.
- **Weak as standalone research.** The general thesis, that rule-based verifiers have false negatives and
  exploitable acceptances, is already published (Huang et al. 2025/2026; Helff et al. 2026). A publishable result
  would need the paid step: showing a small policy actually learns to exploit the tolerance or the partial credit,
  and that hardening removes the exploit at equal updates.
- **Disclosure.** The `number_sorting` tolerance appears deliberate (code comment), so it is a correctness issue,
  not a security vulnerability. A private note is in `$SSDM_PRIVATE/research-verifier-adequacy/`. No public issue
  was filed; whether to report upstream is the owner's decision.
- **Provenance.** The experiment, freeze and held-out run were done by the slice agent. The agent then stopped on a
  provider error, so this results section was written by the integrator from the frozen run's aggregate and
  records.

## Next step and cost

Conditional on useful, independently validated bugs and owner approval: a small custom policy RL run against original versus hardened reward at equal updates and matched seeds, capped at roughly **$3.95** (one H100-hour under the portfolio tariff). No paid run is authorized or performed here. A mutation-suite pass is not proof of soundness; training transfer, clean utility and held-out exploit discovery would be the outcomes, not reward alone.
