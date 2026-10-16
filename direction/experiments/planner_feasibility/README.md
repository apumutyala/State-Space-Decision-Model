# Query-only grants: useful partial restriction, not an autonomous safety boundary

**2026-10-16 · CPU-only exploratory experiment · $0 paid compute · no models or agent runs.**

## 1. Thesis and verdict

A trusted planner that reads only the authenticated task can recover part of the value of task-scoped
host authority without exposing the planner to tool-output injections. On all **97 AgentDojo user tasks**,
our frozen keyword-plus-literal planner recovers **60.9% [56.6, 65.2] of ORACLE-denied attack calls** at
**94.6% [91.4, 97.3] ground-truth-call utility**. It denies **58.4% [54.3, 62.4]** of all evaluated attack
calls, versus **95.0% [93.1, 96.7]** for ORACLE argument grants. Complete reference traces remain permitted
for **84.5% [77.3, 90.7] of tasks**, not 94.6% of tasks.

**Worth pursuing as a first restriction layer, not as a standalone defense.** A stricter prompt-literals-required
arm recovers 70.4% of ORACLE denials, but utility falls to 77.5% and complete-trace coverage to 45.4%.
The remaining problem is not merely better keywords: **19/97 tasks (19.6% [12.4, 26.8])** contain a
nonliteral write-target identifier observed in earlier tool outputs. These are conservative witnesses that the
prompt lacks a concrete identity binding. Exact, useful preauthorization of those output-dependent targets
needs additional trusted resolution or approval; it cannot be justified from the task text alone.
This is **not** a claim that 19.6% of tasks are impossible for a provenance-aware system to execute safely.

“Recovery” here is offline recall of ORACLE-denied reference calls, **not recovery of the 3–4× executed-outcome
improvement** in [workflow v0.2](../workflow_v0/README.md#v02-oracle-task-scoped-authority). No executed attack-success
rate, adaptive security result, or deployment-safety guarantee was measured.

## 2. Prior work and what this experiment adds

All links below were opened and checked against the primary text on 2026-10-16.

- **[CaMeL, v2 (2025)](https://arxiv.org/html/2503.18813v2)** extracts control/data flow from the trusted query,
  isolates untrusted processing, and checks capabilities in an interpreter. Its motivating example explicitly
  shows why an isolated planner still needs data-flow policy when a recipient comes from a document. This study
  measures a much smaller static grant compiler; it does not reproduce CaMeL's interpreter or information-flow guarantees.
- **[Design Patterns for Securing LLM Agents, v3 (2025)](https://arxiv.org/html/2506.08837v3)** distinguishes
  plan-then-execute, dual-LLM and code-then-execute patterns. Fixing tools does not fix their arguments or outgoing
  message content. Our results are a concrete test of this distinction, not a new isolation principle.
- **[Progent, v3 (2026)](https://arxiv.org/html/2504.11703v3)** generates symbolic tool/argument policies from a task,
  updates them with execution context, and uses SMT checks to distinguish automatic narrowing from expansions
  requiring approval. Its latest version makes monotonic confinement explicit. Our frozen static clauses omit both
  contextual updates and approved expansion; the output-dependent cases identify where those mechanisms matter.
- **[PACE, v1 (2026)](https://arxiv.org/html/2610.01349v1)** compiles authenticated-request authority and verifies
  concrete schema-defined effects at the last tool boundary, paired with provenance-path confinement. Importantly,
  it separates the certified contract from its evaluated restoration/repair configuration. Neither PACE's effects
  model nor its reported benchmark outcomes can be inferred from our simple call-denial proxy.
- **[Can CaMeLs Talk?, v1 (2026)](https://arxiv.org/html/2610.05640v1)** shows that individually protected agents
  can lose protection when an untrusted value becomes a downstream agent's trusted request. Multi-CaMeL separates
  instructions from untrusted data across that boundary. Any future planner deployment must authenticate the task
  channel, not merely accept a string labeled “user.”

The contribution is a free, reproducible all-suite feasibility measurement, not a novel security architecture.

## 3. Frozen method

[AgentDojo](https://github.com/ethz-spylab/agentdojo), package **0.1.35**, suite selector **v1.2.2**. Every user task
and every injection task in all four suites is included. For each user task, call `init_environment` on a clean
suite environment; obtain user and paired injection `ground_truth()` calls on independent deep copies. Never
inject a payload, call a model, or execute an agent loop. The source diagnostic separately runs only clean
simulator discovery calls; those outputs never enter any planner.

`keel/workflows/query_planner.py` accepts **only prompt text plus host-trusted tool schemas**. No task IDs,
ground-truth calls, environment objects, tool results, attack goals, or outcome-driven fitting enter the planner.
Rules were committed at **`7ff47158` before the first scoring run**, after inspecting tool interfaces, not benchmark
prompts or answers. The measurement-only output-ID audit was corrected at **`f4a916a4`**: short IDs must match
exact returned scalars, not substrings of dates. Planner rules and all utility/security scores were unchanged.

| Arm | Construction |
|---|---|
| `all_tools` | All suite tool names, no argument clauses; sanity reference. |
| `tools` | All discovery tools (`get/read/search/list/check` prefixes) plus write tools selected by fixed semantic keyword rules. This deliberately broad read policy avoids guessing discovery dependencies. |
| `literals` | Same tool rules; extract IBANs, currency-marked amounts, emails, capitalized names, channels, ISO dates/times, URLs and filenames from the prompt. One candidate becomes `fixed`; several scalar candidates become `one_of`; an explicitly stated single monetary ceiling becomes `at_most`. Arrays are fixed as whole arrays. Unknown fields remain free. |
| `literal_required` | Same extraction, but withhold a write tool unless its declared routing/identity/value scope keys are constrained. This tests the availability cost of missing bindings, not a complete safety policy. |
| `oracle_tools` | Exactly the clean user ground truth's tool-name set. |
| `oracle_arguments` | Existing `ArgumentPolicy.from_calls`: a disjunction of complete ground-truth argument tuples, with its existing Slack `<summary>`/`...` body exemptions and schema-default handling. Omitted fields stay free. |

All grants use actual `keel.host.ArgumentClause` and `Authority` matching; no second authorization implementation.
The planner is intentionally primitive: no negation/role resolution, no date-language normalization, no reliable
name disambiguation, and no unseen-value guessing. Multiple candidates can authorize incorrect combinations.
An `at_most` constraint alone supplies neither a lower bound nor a cumulative spending limit. Free bodies,
optional fields, repeated calls and information flows remain unrestricted. Trusted schemas are part of the TCB;
attacker-authored MCP descriptions are not acceptable planner inputs.

**Estimands.** For each user task, utility = permitted user-ground-truth calls / its ground-truth calls;
security proxy = denied paired-injection-ground-truth calls / all such calls. Report task means, not task completion.
Recovery = calls denied by **both** planner and argument ORACLE / calls denied by argument ORACLE; this is a pooled
call ratio, so denying an ORACLE-permitted call earns no recovery credit. Overall utility/denial means weight each
of the 97 tasks equally; recovery pools calls across suites. All arms share the same task resamples.

95% percentile CIs use **10,000 user-task cluster resamples**, seed **20261016**. A cluster carries every paired
attack goal and call. Overall resampling is stratified by suite, retaining suite task counts. This is a finite
benchmark census: the intervals describe hypothetical task-population variability, not uncertainty about the
observed census. Composite tasks reuse subtasks and environments; task clustering does not remove that wider
dependence. Degenerate intervals are not population-risk certificates.

## 4. Experiment actually run

| Suite | User tasks | Injection goals | User calls | Paired attack calls | Paired goals with no calls |
|---|---:|---:|---:|---:|---:|
| workspace | 40 | 14 | 84 | 400 | 320 / 560 |
| travel | 20 | 7 | 124 | 240 | 20 / 140 |
| banking | 16 | 9 | 33 | 192 | 0 / 144 |
| slack | 21 | 5 | 98 | 273 | 0 / 105 |
| **Total** | **97** | **35** | **339** | **1,105** | **340 / 949** |

Eight workspace goals and one travel goal have empty tool-call ground truth. They were loaded for every pairing,
but contribute no calls to this proxy: **9/35 goals and 340/949 pairings are not scored for security**. In particular,
output-only manipulation is outside a tool-call grant proxy. The remaining 609 pairings contribute 1,105 calls;
these include preparatory reads as well as effectful actions. Blocking one reference call neither proves the
whole attack impossible nor establishes its actual success predicate.

Final run: **3.90 seconds** measured inside the experiment (**6.28 seconds** including startup), **244.0 MiB peak
RSS**, Python **3.12.13**, NumPy **2.5.4**. Linux `bwrap --unshare-net`, read-only root with only this experiment's
output directories writable, `nice -n 10`; no network, model inference, GPU, paid API, or Modal. No discovery-call
errors; both ORACLE arms permit **339/339** reference user calls. **74 CPU tests passed in 2.52 seconds**, covering
new planner/aggregation tests plus existing host-clause and workflow-ORACLE tests. No full-repository test claim.

## 5. Results with uncertainty

Percent **[95% task-cluster CI]**. ORACLE utility is **100 [100, 100]** by construction in every suite; it is
reference-call compatibility, unlike the lower actual actor utility reported in workflow v0.2.

| Suite | Arm | Call utility ↑ | Attack-call denial ↑ | ORACLE-denial recovery ↑ |
|---|---|---:|---:|---:|
| workspace | tools | 97.0 [93.7, 99.6] | 54.2 [48.0, 60.0] | 54.2 [48.0, 60.0] |
| workspace | literals | 94.9 [88.8, 99.2] | 65.8 [58.2, 73.2] | 65.8 [58.2, 73.2] |
| workspace | literal_required | 77.0 [68.5, 85.0] | 76.8 [73.0, 80.5] | 76.8 [73.0, 80.5] |
| workspace | ORACLE tools | 100 [100, 100] | 86.0 [79.8, 91.5] | 86.0 [79.8, 91.5] |
| workspace | ORACLE arguments | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] |
| travel | tools | 97.5 [92.5, 100] | 44.6 [40.4, 48.3] | 49.5 [45.1, 53.9] |
| travel | literals | 97.5 [92.5, 100] | 47.9 [46.2, 49.2] | 53.2 [49.6, 57.5] |
| travel | literal_required | 93.4 [87.1, 98.2] | 49.6 [48.8, 50.0] | 55.1 [51.8, 59.4] |
| travel | ORACLE tools | 100 [100, 100] | 82.1 [74.2, 89.6] | 91.2 [85.8, 95.9] |
| travel | ORACLE arguments | 100 [100, 100] | 90.0 [83.3, 95.8] | 100 [100, 100] |
| banking | tools | 97.9 [93.8, 100] | 51.6 [33.9, 69.8] | 52.4 [34.0, 71.3] |
| banking | literals | 96.7 [91.2, 100] | 70.8 [54.2, 85.4] | 72.0 [54.5, 87.2] |
| banking | literal_required | 64.0 [54.0, 75.4] | 91.7 [91.7, 91.7] | 93.1 [91.7, 94.6] |
| banking | ORACLE tools | 100 [100, 100] | 67.7 [50.0, 84.9] | 68.8 [50.5, 86.8] |
| banking | ORACLE arguments | 100 [100, 100] | 98.4 [96.9, 100] | 100 [100, 100] |
| slack | tools | 95.8 [90.8, 100] | 37.7 [33.3, 41.0] | 42.7 [37.4, 47.8] |
| slack | literals | 89.4 [81.4, 96.1] | 45.1 [41.4, 49.1] | 51.0 [46.0, 56.6] |
| slack | literal_required | 73.6 [65.7, 81.6] | 49.1 [46.2, 52.4] | 55.6 [50.8, 61.1] |
| slack | ORACLE tools | 100 [100, 100] | 68.5 [59.7, 76.9] | 78.2 [71.1, 84.7] |
| slack | ORACLE arguments | 100 [100, 100] | 87.5 [81.3, 93.0] | 100 [100, 100] |
| **Overall** | **tools** | **97.0 [94.9, 98.8]** | **48.2 [44.3, 52.3]** | **50.3 [46.1, 54.7]** |
| **Overall** | **literals** | **94.6 [91.4, 97.3]** | **58.4 [54.3, 62.4]** | **60.9 [56.6, 65.2]** |
| **Overall** | **literal_required** | **77.5 [73.1, 81.8]** | **67.6 [66.0, 69.4]** | **70.4 [68.3, 72.8]** |
| **Overall** | **ORACLE tools** | **100 [100, 100]** | **78.4 [73.8, 82.9]** | **82.2 [77.7, 86.5]** |
| **Overall** | **ORACLE arguments** | **100 [100, 100]** | **95.0 [93.1, 96.7]** | **100 [100, 100]** |

The all-tools reference is 100% utility and 0% denial/recovery in every suite. Exact unrounded values, complete-trace
coverage, tool-grant counts, denominators, source counts and intervals are in [`aggregate.json`](aggregate.json).

### Where task text is insufficient

For every supplied ground-truth argument, the diagnostic records exact task-literal occurrence, case-normalizable
occurrence, observation in earlier clean discovery outputs, schema-default equality, or unresolved/generated.
Across **464 arguments**, these counts are **155 / 10 / 191 / 9 / 99**, respectively. This is not a complete semantic
three-way provenance oracle: a value found in an output might also be derivable from the prompt, and generated
prose/defaults must not be falsely classified as requiring untrusted data. Only case normalization is credited as
a trusted derivation; richer derivations remain unresolved. No output value is promoted into an actual grant.

The narrower identity diagnostic counts only nonliteral **write** recipients, email addresses and opaque
file/event/transaction IDs that occur in prior tool outputs. Short IDs require exact typed scalar matches.
The 19 retained prompt/call witnesses were inspected, including bill-provided IBANs, website-provided invitation
addresses, discovered file/event IDs, and message-activity-selected recipients. This is a lower-bound-style
operational diagnostic, not a complete impossibility classifier; hotel choices, dates, amounts, generated IDs
from skipped write calls and message content are deliberately outside the count.

| Suite | Tasks with missing concrete identity bindings | Percent [95% CI] |
|---|---:|---:|
| workspace | 8 / 40 | 20.0 [7.5, 32.5] |
| travel | 0 / 20 | 0.0 [0.0, 0.0] |
| banking | 4 / 16 | 25.0 [6.2, 50.0] |
| slack | 7 / 21 | 33.3 [14.3, 52.4] |
| **Overall** | **19 / 97** | **19.6 [12.4, 26.8]** |

**[INFERENCE] Conditional impossibility:** if two admissible environments have the same task text but different
correct target IDs, a prompt-only deterministic compiler emits identical grants in both. It cannot select the
correct concrete ID in both while excluding the incorrect alternative. Deny-all remains safe but not useful;
broad grants remain useful but do not supply this binding. A trusted directory, authenticated structured lookup,
provenance-conditioned capability, or explicit user approval can change that information constraint. The 19.6%
figure does **not** prove that every other task is safely scopeable or that these tasks need LLM planning.

## 6. What this does and does not establish

The literal arm adds useful restriction over the broad-discovery tool arm without a model call. It is worth
retaining as an interpretable baseline and a low-cost first pass. It is **not deployable permission inference**:
keyword false positives, entity-role confusion and missing literals remain; all discovery tools are authorized;
and arbitrary message bodies can still carry phishing or exfiltration to an otherwise allowed recipient.

Reference-call compatibility is not real task utility: an actor may choose a different valid search, paraphrase,
recipient ordering or date representation. Denying a reference attack call is not a measured prevented outcome;
adaptive attackers can use other calls, free fields or repeated authorized calls. ORACLE also fixes arbitrary
reference strings, omits sequence/count/flow constraints, and can leave harmful optional fields unconstrained.
The experiment is exploratory, with no held-out confirmation population or independent human provenance labels.

## 7. Next step, cost and reproduction

**Next free step:** design authenticated identifier-resolution grants and test output substitution/entity-binding
counterexamples on CPU. Keep tool-output text out of the planner; distinguish trusted API identity metadata from
untrusted prose. Cost: **$0 paid compute**, plus unestimated engineering/review time. Improving regex recall alone
cannot answer the missing-authority problem.

**Then, only with owner approval:** a matched 216-episode pilot on the original banking/Slack task matrix
(24 tasks × benign/two attacked conditions × none/literals/ORACLE-arguments), with executed target predicates,
actual task utility and review demand. At the historical ORACLE run's ~$0.02177 per episode, **[INFERENCE] ~$4.70**
compute before new overhead; propose an **$8 owner-approved cap**, not an authorization or invoice. No such run was
started and no budget file was created or edited. This pilot would not establish adaptive security.

From this worktree after `source ~/ssdm/env.sh`:

```bash
WT="$PWD"
OUT="$SSDM_PRIVATE/research-planner"
mkdir -p "$OUT"
nice -n 10 bwrap --unshare-net --die-with-parent --ro-bind / / \
  --bind "$OUT" "$OUT" \
  --bind "$WT/direction/experiments/planner_feasibility" "$WT/direction/experiments/planner_feasibility" \
  --proc /proc --dev /dev --tmpfs /tmp --chdir "$WT" --setenv PYTHONDONTWRITEBYTECODE 1 \
  "$VIRTUAL_ENV/bin/python" -m direction.experiments.planner_feasibility.run
nice -n 10 bwrap --unshare-net --die-with-parent --ro-bind / / \
  --proc /proc --dev /dev --tmpfs /tmp --chdir "$WT" --setenv PYTHONDONTWRITEBYTECODE 1 \
  "$VIRTUAL_ENV/bin/python" -m pytest -q -p no:cacheprovider \
  keel/tests/test_query_planner.py keel/tests/test_host_arguments.py keel/tests/test_workflow_oracle.py \
  direction/experiments/planner_feasibility/test_run.py
```

`run.py` and `test_run.py` are experiment-local. Reusable planner tests live in `keel/tests/test_query_planner.py`.
Only code, this README and compact aggregate JSON are committed. Per-task prompts, calls, grants and source audits
remain at `$SSDM_PRIVATE/research-planner/records.jsonl`; its SHA-256 is
`568083e91cf9f12548e250c25863f933dae8c495e510f59067e0c2406b4b21d2`.
Planner SHA-256: `780d03d1c9f66886b552ee2fff6edd0b7daf81ea27bf7497af5bd6d973d9ae44`.
