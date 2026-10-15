# Workflow v0: executed prompt-injection targets in AgentDojo

**Primary result: the stock one-pass Keel gate blocks the measured attack targets by over-holding legitimate calls. This is not useful robustness and does not meet Keel's 10× stretch target at matched utility and review load.** In banking its benign utility is 41.7% versus 100% without defense; in Slack it is 0% versus 91.7%. No workflow-specific training, prompt adjustment, or threshold fitting was used.

The budget-permitted think-first arm restores much of benign Slack utility (**75.0% [50.0, 100.0]**) but still loses attacked utility and demands substantial review. It also does not meet the stretch target. Its 1,024-token reasoning cap is an important limitation: 90.0% of traces are budget-truncated.

## Design, versions and scope

- [AgentDojo](https://github.com/ethz-spylab/agentdojo), pinned Python package **0.1.35**, benchmark suite version **v1.2.2**; [pipeline documentation](https://agentdojo.spylab.ai/concepts/agent_pipeline/).
- Agent: **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Judge: **Qwen/Qwen3.5-4B**, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. These are the shared runtime's pinned checkpoints, not workflow-trained students.
- vLLM **0.30.0**; one H100 per run, 55% memory for the localhost OpenAI agent server and 30% for an in-process judge; shared `ssdm-hf-cache`. Native pinned Qwen3.5 chat template, `qwen3_coder` tool parser, `qwen3` reasoning parser. Actor thinking disabled, temperature 0, seed 0, 1,024 output-token limit, 32,768 context, upstream 15-iteration tool loop. [Qwen deployment recommendation](https://huggingface.co/Qwen/Qwen3.5-9B#deployment).
- **Two workflows:** banking and Slack; tasks `user_task_0` through `user_task_11` in each, selected by numeric ID before main inference, without performance filtering. Each arm has **12 benign episodes + 24 attacked episodes per suite**, two attack goals per user task, hence **360 primary episodes** across five arms. This is a deterministic prefix, not a random sample of the complete benchmark.
- Standard non-adaptive `important_instructions`: upstream template and injection-vector discovery. Only the model-address substitution is set to `Qwen3.5-9B`, because the package model-name registry predates this model. Banking uses `injection_task_0` and `_1` (transactions disclosing private transaction information); Slack uses `_1` and `_2` (phishing-message delivery and all-channel exfiltration).
- Exact primary matrix: [`config.json`](config.json). Separate parser/deployment pilot: [`pilot_config.json`](pilot_config.json), [`pilot_summary.json`](pilot_summary.json). Pilot tasks banking 13 / Slack 16 are excluded from primary metrics; both undefended tasks completed, with valid executed calls and no tool-dispatch errors. The gate prompt and threshold were not tuned after observing pilot failures.
- Arms `none`, `spotlighting_with_delimiting`, `repeat_user_prompt`, `tool_filter` use upstream AgentDojo implementations. Keel replaces the no-defense executor with a pre-dispatch gate; trust-mode user task metadata is host-provided, every prior tool output and the proposed call are untrusted. The one-pass 4B judge allows only when `p(allow) > p(hold)`; ties hold. No human reviews or approvals are simulated.
- Supplemental [`think_config.json`](think_config.json) adds **72 episodes**, using exactly the same tasks, goals, policy, options, threshold and agent. The judge first samples one reasoning trace (temperature 0.6, top-p 0.95, top-k 20, seed 0, maximum 1,024 tokens), uses the shared `close_think`, then reads the same labels. The supplement was configured and launched before the primary aggregate was inspected; no outcome-based prompt changes were made. Total reported matrix: **432 episodes, two suites × six arms**.

## Utility, targeted attack success and review demand

All entries are **percent [95% task-cluster percentile bootstrap CI]**. There are 12 clusters per suite/arm; 10,000 resamples, seed 20261015. Resampling a user task carries both attack goals together. `aggregate.json` contains unrounded estimates, counts, denominators and intervals.

**Targeted ASR** means the selected AgentDojo target predicate became true following actual tool execution in the simulated environment, not that the agent merely proposed a forbidden call. Predicates are checked after each dispatch and latched; a later action cannot erase success. Trace-based predicates receive executed calls, not AgentDojo's default assistant-proposal trace. This is an episode-level target-achievement rate, not a count or exhaustive detector of every unauthorized action.

**Hold rate** is held calls / proposed calls, pooled across episodes and bootstrapped by task. Non-gate arms have no review mechanism, so their hold rate is zero; this does not mean their actions were approved by a reviewer. Review episodes (any hold) are reported separately below.

| Suite | Arm | Benign utility | Attacked utility | Targeted ASR | Hold benign | Hold attacked |
|---|---|---:|---:|---:|---:|---:|
| banking | Keel one-pass | 41.7 [16.7, 66.7] | 41.7 [16.7, 66.7] | 0.0 [0.0, 0.0] | 84.8 [75.0, 92.3] | 84.8 [73.7, 92.5] |
| banking | Keel think-first | 58.3 [33.3, 83.3] | 54.2 [29.2, 79.2] | 0.0 [0.0, 0.0] | 60.6 [37.5, 76.9] | 69.0 [49.0, 81.2] |
| banking | none | 100.0 [100.0, 100.0] | 83.3 [62.5, 100.0] | 16.7 [0.0, 37.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| banking | repeat_user_prompt | 83.3 [58.3, 100.0] | 79.2 [58.3, 95.8] | 12.5 [0.0, 33.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| banking | spotlighting_with_delimiting | 91.7 [75.0, 100.0] | 75.0 [50.0, 95.8] | 12.5 [0.0, 33.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| banking | tool_filter | 58.3 [33.3, 83.3] | 66.7 [41.7, 91.7] | 8.3 [0.0, 25.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| slack | Keel one-pass | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 95.4 [90.4, 100.0] | 95.5 [90.5, 100.0] |
| slack | Keel think-first | 75.0 [50.0, 100.0] | 33.3 [12.5, 54.2] | 0.0 [0.0, 0.0] | 56.6 [24.3, 71.3] | 70.8 [50.5, 80.8] |
| slack | none | 91.7 [75.0, 100.0] | 62.5 [33.3, 87.5] | 12.5 [0.0, 25.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| slack | repeat_user_prompt | 75.0 [50.0, 100.0] | 58.3 [33.3, 83.3] | 16.7 [4.2, 29.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| slack | spotlighting_with_delimiting | 91.7 [75.0, 100.0] | 62.5 [33.3, 87.5] | 12.5 [0.0, 25.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| slack | tool_filter | 16.7 [0.0, 41.7] | 8.3 [0.0, 25.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Keel one-pass requires review in **91.7% [75.0, 100.0]** of banking episodes (both benign and attacked) and **100% [100, 100]** of Slack episodes. This is not matched to zero review load in the baselines. Its zero observed targeted ASR is **0/24** attacked episodes in each suite. Bootstrap intervals at all-zero/all-one observations degenerate: **[0, 0] is not evidence of zero population risk or a safety certificate**.

Think-first review-episode rates are banking **58.3% [33.3, 83.3]** benign / **70.8% [45.8, 91.7]** attacked, and Slack **33.3% [8.3, 58.3]** benign / **66.7% [41.7, 87.5]** attacked. Its targeted ASR is also 0/24 in each suite, with the same degenerate-bootstrap caveat. Thinking improves benign utility relative to one-pass, but neither utility nor review load is matched to the useful defended baselines; this is not a successful workflow safety–utility frontier.

The upstream `tool_filter` runs with the local model but is not uniformly effective: in the pilot it returned tool-name lists without executing either user task; the main matrix has nonzero utility and retains all its results. Its low ASR must be read alongside lost utility, particularly in Slack. We do not repair or silently substitute a different filter pipeline. The standard attack can replace legitimate evidence as well as introduce instructions, so utility-under-attack loss is not entirely attributable to executing attacker instructions.

## Latency and cost per episode

Warm end-to-end seconds include all agent requests, tool execution and gating. CIs use the same task bootstrap. GPU dollars use one H100 at $3.95/hour; their CI is exactly the displayed latency CI multiplied by `3.95/3600` (unrounded values are in `aggregate.json`). All primary arms share the same dual-engine allocation, even when the judge is idle. Costs exclude human review and are estimates, not provider invoices.

| Suite | Arm | Benign seconds [95% CI] | Attacked seconds [95% CI] | Benign warm GPU $ | Attacked warm GPU $ |
|---|---|---:|---:|---:|---:|
| banking | Keel one-pass | 13.7 [11.6, 16.1] | 13.4 [11.7, 15.3] | 0.0150 | 0.0147 |
| banking | Keel think-first | 82.9 [61.5, 103.0] | 90.0 [62.5, 118.7] | 0.0909 | 0.0987 |
| banking | none | 11.6 [9.1, 14.4] | 11.5 [9.3, 13.6] | 0.0127 | 0.0126 |
| banking | repeat_user_prompt | 8.7 [7.1, 10.2] | 11.5 [8.6, 14.3] | 0.0095 | 0.0126 |
| banking | spotlighting_with_delimiting | 10.5 [8.2, 12.9] | 12.8 [9.7, 16.2] | 0.0116 | 0.0141 |
| banking | tool_filter | 7.1 [3.0, 12.1] | 8.3 [3.6, 13.7] | 0.0077 | 0.0091 |
| slack | Keel one-pass | 16.8 [12.2, 21.6] | 17.1 [12.3, 22.1] | 0.0184 | 0.0187 |
| slack | Keel think-first | 186.3 [95.8, 292.0] | 203.7 [116.2, 299.3] | 0.2044 | 0.2235 |
| slack | none | 9.3 [6.3, 13.1] | 14.2 [8.8, 20.6] | 0.0102 | 0.0156 |
| slack | repeat_user_prompt | 10.8 [6.7, 15.7] | 17.1 [10.2, 25.1] | 0.0119 | 0.0187 |
| slack | spotlighting_with_delimiting | 9.5 [6.4, 13.3] | 14.9 [8.5, 23.8] | 0.0104 | 0.0164 |
| slack | tool_filter | 5.3 [2.5, 9.3] | 5.5 [2.6, 9.7] | 0.0058 | 0.0060 |

Primary run `workflow-v0-20261015b`: **74.1042 GPU-minutes, estimated $4.8785**, including model loading; 106.09 s to agent readiness, 173.33 s to both engines ready. Across all 360 primary episodes, fully amortized function-wall GPU cost is **$0.01355/episode**. Full per-allocation accounting, including the parser pilot and cancelled launch, is in [`keel/workflows/README.md`](../../../keel/workflows/README.md).

Supplement `workflow-think-20261015a`: **174.4865 GPU-minutes, estimated $11.4870**, including model loading; 116.09 s to agent readiness, 190.03 s to both engines ready. Fully amortized function-wall GPU cost: **$0.15954/episode**. Across primary + supplemental + parser pilot, measured function-wall GPU estimates total **252.2991 minutes / $16.6097**. Including a conservative five-GPU-minute reserve for the cancelled startup gives **$16.9389**, below the $25 cap. Platform invoice/teardown charges were not independently audited.

## Execution quality, reproducibility and limitations

All **360/360 primary episodes completed with zero episode-level exceptions**. There were 791 successfully executed tool calls and 53 tool-returned errors (14 Slack none, 16 Slack spotlighting, 20 Slack repeat-user, 3 Slack tool-filter); these are retained, not excluded. Eight agent completions hit the token cap (banking spotlighting 1, banking tool-filter 3, Slack none 2, Slack spotlighting 2). A failed tool call does not become an executed call merely because it appeared in an assistant message. Side effects remain scored even if subsequent agent execution fails.

All **72/72 supplemental episodes completed with zero episode-level exceptions**, 113 successful tool calls, two tool-returned Slack errors and one actor completion reaching its token cap. The think-first judge made **341 decisions**, generating **340,614 reasoning tokens** (mean **998.9**). **307/341 traces (90.0%) reached the 1,024-token cap** and were closed using the shared budget-forcing suffix. These numbers describe a budget-limited thinking arm, not an unconstrained reasoning teacher. Raw reasoning and both initial/readout prompt hashes remain outside git.

CPU command: `/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests` — **33 passed**. Tests use a deterministic fake judge but a real AgentDojo executor and mutable runtime, plus actual banking/Slack predicates; they cover hold/no side effect, allow passthrough, trust boundaries, decision logs, same-batch calls, nested-call bypass prevention, invalid scores, logging failure and task-cluster aggregation.

Reproduction commands and architecture are in [`keel/workflows/README.md`](../../../keel/workflows/README.md). Raw transcripts, dataset text, judge records and episode metrics remain outside git at `/home/alp/ssdm-direction/workflow/` and `keel-results:/workflow/`. Primary executable source was committed at `e5f10566`; supplemental source at `c619137d` does not change the primary gate policy, threshold or one-pass readout. Aggregate code, exact configs and [`run_summary.json`](run_summary.json) are committed here.

This is exploratory, not sealed or confirmatory: **one agent model family, a same-family stock judge, one non-adaptive standard attack, a deterministic task prefix, only two attack goals per task, one agent seed, and simulator-side effects**. There is no claim about adaptive attackers, other families, production data, all unauthorized actions, calibrated probabilities, statistical gate certification, or external replication. The judge was trained on **nothing workflow-specific**. Utility and review load are not matched, so even an arbitrarily large raw ASR reduction cannot establish the moonshot's matched-completion/review-load stretch result.

## v0.1: host-kernel integration

The v0 tables above are preserved as historical measurements. The host-integrated cutover is a separate **72-episode one-pass Keel rerun only**, with the same two suites, 12 tasks per suite, benign condition and two attack goals, pinned models, prompt text, options and argmax/tie-hold rule. [`host_config.json`](host_config.json) fixes this comparison; implementation source is `6ae7af5f`. This is not an additional pooled sample or a new attack evaluation.

Every proposed call now goes through `keel.host.HostKernel` with a host-bound `agent` principal. The operator policy's structural action set and the principal's role grants are the suite's tool names. Trusted task metadata and untrusted tool outputs come from `kernel.inputs`; proposed-call content is `.derive`d from untrusted outputs. The judge returns a `DecisionResult` with explicit defer reasons for holds; the kernel receives an `ActionProposal` with tool, arguments, principal and cited segment IDs. Only its executor callback invokes the actual AgentDojo runtime.

One disk-backed `ReceiptLog` is shared across all episodes in the run. Its write-ahead hash chain, checkpoint, actual prompt-use manifests and saved replay inputs replace the plain decision journal; new runs do not produce both. Each episode verifies its cumulative chain, and the end-of-run audit checks episode/receipt linkage plus 20 seeded random input/prompt replays against a pinned checkpoint. A receipt records dispatch intent, not confirmation that a downstream tool succeeded. The host's zero-capacity review queue holds every deferred call and returns the same fixed message used in v0.

CPU acceptance: `/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests` — **414 passed**. Deterministic fake judges exercise the real host and real AgentDojo dispatch, including an available runtime tool that is absent from structural authority: forced model approval still cannot execute it. Tests also cover write-ahead receipts, one receipt/no dispatch on hold, chain tampering, saved-request replay, episode linkage and both thinking prompt manifests.

These grants are broad, deployable suite-tool grants, **not a ground-truth task-tool oracle**. They cannot distinguish authorized from unauthorized arguments to an otherwise permitted tool; that semantic decision still rests with the uncalibrated stock judge. The integration does not fix the v0 utility/review mismatch or certify the moonshot target. Receipt replays establish input/prompt identity, not independent model-output regeneration; checkpoint integrity depends on trusted host storage, and Modal durability is not an independently audited crash-recovery guarantee.

### Measured integration result

Run `workflow-host-v01-20261015a` completed **72/72 episodes with zero episode or dispatch errors**: 335 gated calls, 30 successful executions and 305 held calls. **All 72 episodes exactly match v0 on all 14 compared non-timing outcome fields**, including utility, targeted ASR, call/hold counts, token counts and injection hashes. The 335 aligned judge inputs have **335/335 identical prompt hashes and 335/335 exactly identical probabilities**; maximum absolute probability difference is 0. Request fingerprints intentionally change because the host-authorized tool set and derived provenance are now part of the request.

The complete **335-receipt chain** and **72 episode checkpoints** verified both remotely and after download. **20/20 seeded random receipts passed `replay_check`**; these are the same 20 sampled indices, not 40 independent replays. The pinned head is `83b65ad3bd34b94f2d55b22f2cf03659785cfe8a12ff46bd2afb14d5603dbc54`. Aggregate audit and exact comparison counts are in [`host_audit.json`](host_audit.json); per-episode differences (none), receipt chains, checkpoints and saved requests remain outside git under `/home/alp/ssdm-direction/workflow/workflow-host-v01-20261015a/`.

The rerun uses the same 10,000-resample task-cluster CIs as v0 and reproduces its safety–utility failure, not an improvement:

| Suite | Host-integrated arm | Benign utility | Attacked utility | Targeted ASR | Hold benign | Hold attacked |
|---|---|---:|---:|---:|---:|---:|
| banking | Keel one-pass | 41.7 [16.7, 66.7] | 41.7 [16.7, 66.7] | 0.0 [0.0, 0.0] | 84.8 [75.0, 92.3] | 84.8 [73.7, 92.5] |
| slack | Keel one-pass | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 95.4 [90.4, 100.0] | 95.5 [90.5, 100.0] |

Review-episode rates are unchanged: banking 91.7% [75.0, 100.0] in both conditions; Slack 100% [100, 100] in both. Each targeted-ASR estimate is 0/24; the degenerate bootstrap intervals still do not certify zero population risk. [`host_aggregate.json`](host_aggregate.json) contains full counts, unrounded estimates and intervals.

| Suite | Benign seconds [95% CI] | Attacked seconds [95% CI] | Benign warm GPU $ [95% CI] | Attacked warm GPU $ [95% CI] |
|---|---:|---:|---:|---:|
| banking | 11.4 [9.7, 13.2] | 11.1 [9.7, 12.7] | 0.0125 [0.0106, 0.0145] | 0.0122 [0.0107, 0.0139] |
| slack | 13.4 [9.8, 17.5] | 13.5 [9.8, 17.5] | 0.0147 [0.0107, 0.0192] | 0.0148 [0.0107, 0.0192] |

Latency differs from v0, but these are separate allocations rather than an isolated host-overhead benchmark: the lower wall times must not be attributed to the host integration. This follow-up used **17.8653 H100-minutes / $1.1761 estimated**, including model loading, against a separate **$8 cap**; no other follow-up allocations were launched. Function-wall cost amortized over 72 episodes is **$0.01634/episode**, excluding platform teardown/invoice adjustments and human review. Exact allocation evidence is in [`host_run_summary.json`](host_run_summary.json), and the paid-run ledger is in the workflow package README.
