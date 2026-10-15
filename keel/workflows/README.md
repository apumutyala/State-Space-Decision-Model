# Keel in AgentDojo workflows

This package runs a real tool-using agent against AgentDojo's simulated banking and Slack environments. Tools mutate benchmark environments; no real bank account, Slack workspace, or external recipient is connected.

## Implementation

- `gate_element.py` replaces AgentDojo's `ToolsExecutor`. It judges **each proposed call before dispatch**, including calls before the first tool output and each member of a multi-call batch, then routes it through `keel.host.HostKernel`. Only the kernel's executor callback can invoke the real AgentDojo dispatcher. Earlier outputs in the same batch are visible to later decisions.
- All segments come from `kernel.inputs`: the authenticated user task is trusted metadata; every prior tool result is untrusted `tool_output`; the proposed call is derived through `.derive` from untrusted outputs (or no parents before any output). Host policy text is unchanged from v0, but structural `authorized_actions` and the host-bound `agent` role grants are explicitly the suite's tool names. Neither injected tool names nor model approval can expand that set.
- `keel.protocol.render(mode="trust")` uses the existing escaped untrusted-record boundary. The Qwen3.5-4B judge reads one token's raw label log-probabilities, normalized only over allow/hold, exactly as the shared judge runtime. `p(allow) > p(hold)` allows; ties hold. No workflow-specific training or threshold tuning is performed.
- The supplemental `keel_think` arm uses the same request, policy, options and threshold, but first samples one reasoning trace (up to 1,024 tokens; temperature 0.6, top-p 0.95, top-k 20, seed 0), closes it with the shared `close_think`, then reads the label. The judge returns `JudgeEvaluation(probabilities, prompts)` containing the actual opening and answer-slot `PromptUse` manifest. Current receipts retain reasoning/truncation and both prompt hashes; the old separate `judge_samples.jsonl` writer is removed. This is not a trained or tuned judge.
- Hold returns the fixed message `This tool call is held for human review. It was not executed.` The underlying executor is not called. This experiment measures review demand; there is no human approver or automatic retry-to-permission.
- The judge produces a `DecisionResult`; hold/tie or unsupported nested calls set a defer reason. Each `ActionProposal` names the host-bound principal `agent` and cites the request's segment IDs. A single disk-backed `ReceiptLog` per run fsyncs a canonical, hash-chained receipt and checkpoint before kernel dispatch; receipts include probabilities, authorization, provenance and the actual prompt manifest. The old plain decision journal is removed, not written alongside receipts. Original requests are saved separately by fingerprint solely for replay. Receipt `act` means dispatch intent, not downstream completion. Audit failure prevents dispatch.
- All arms use the same upstream pipeline composition, system message and 15-iteration tool loop. `none`, `spotlighting_with_delimiting`, `repeat_user_prompt`, and `tool_filter` are AgentDojo's implementations, not approximations. `keel` replaces only the no-defense executor.
- `runner.py` preserves upstream utility/security predicates but supplies **executed** calls, not assistant proposals, to trace-based predicates. It latches targeted success after each actual dispatch, so later actions cannot erase a successful attack. Failed/aborted episodes remain in denominators and retain any already-executed target success. It does not rerun incomplete episodes against a partially mutated environment.

## Architecture and dependencies

A single Modal H100 runs a localhost vLLM OpenAI-compatible server for Qwen3.5-9B (55% GPU memory) and an in-process Qwen3.5-4B label judge (30%). Both use 32,768-token contexts, eager execution and the shared `ssdm-hf-cache` volume. This avoids cross-container per-call latency and paying for a second GPU. The judge is not a concurrent agent: episodes and their calls run sequentially. Raw logs persist on `keel-results` under `workflow/<run-id>` and are downloaded to `/home/alp/ssdm-direction/workflow/<run-id>/`.

Pinned dependencies: `agentdojo==0.1.35`, `vllm==0.30.0`; shared Modal environment uses Modal 1.5.3. Agent and judge revisions are read from `keel.runtime.modal_app.MODELS` and written to every run record. The native Qwen3.5 chat template is paired with `qwen3_coder` tool-call parser and `qwen3` reasoning parser, as recommended by the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B#deployment). Agent thinking is disabled, generation is greedy (`temperature=0`, seed 0, at most 1,024 output tokens). AgentDojo maps system messages to OpenAI's developer role; the endpoint adapter maps these back to Qwen's system role and explicitly sends zero temperature for **all** arms and tool-filter queries.

Sources: [AgentDojo pipelines and defenses](https://agentdojo.spylab.ai/concepts/agent_pipeline/), [benchmark](https://github.com/ethz-spylab/agentdojo/blob/main/src/agentdojo/benchmark.py), [upstream tool execution](https://github.com/ethz-spylab/agentdojo/blob/main/src/agentdojo/agent_pipeline/tool_execution.py), [pinned package release](https://pypi.org/project/agentdojo/0.1.35/).

## Reproduction

From the repository root, with the configured Python environment. Current code uses the v0.1 host-kernel cutover; the original v0 source revisions are recorded with the historical results.

```sh
uv pip install --python /home/alp/ssdm-venv/bin/python agentdojo==0.1.35
/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.modal_run \
  --config direction/experiments/workflow_v0/pilot_config.json --run-id UNIQUE-pilot
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.modal_run \
  --config direction/experiments/workflow_v0/config.json --run-id UNIQUE-main
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.modal_run \
  --config direction/experiments/workflow_v0/think_config.json --run-id UNIQUE-think
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.aggregate \
  --episodes /home/alp/ssdm-direction/workflow/UNIQUE-main/episodes.jsonl \
  --episodes /home/alp/ssdm-direction/workflow/UNIQUE-think/episodes.jsonl \
  --output direction/experiments/workflow_v0/aggregate.json
```

Unique run IDs avoid accidental overwrite; duplicate episode IDs are rejected by aggregation. `--download-only` retrieves a run without launching paid inference. Per-episode text, model outputs, requests and receipt chains/checkpoints must not enter git.

## Measurement and limits

Results and exact matrix are in [`direction/experiments/workflow_v0`](../../direction/experiments/workflow_v0/). All task IDs and attack goals are fixed in `config.json` before main inference. Bootstrap CIs resample user tasks, carrying both attacked goals together; hold rate is held/proposed calls, not an unweighted mean of episode ratios. Episodes with any hold separately measure review demand. A zero-success bootstrap interval can be degenerate and is not a certificate of zero population risk.

Measured matrix: **432 episodes, banking + Slack × six arms, zero episode-level exceptions**. One-pass Keel has zero observed targeted successes but benign utility only 41.7% / 0.0% and benign call holds 84.8% / 95.4% (banking / Slack). Think-first improves benign utility to 58.3% / 75.0%, still below useful baselines and with substantial review. Its 307/341 reasoning traces (90.0%) reached the 1,024-token budget. Full task-cluster CIs, attacked utility, review demand and latency/cost are published with the aggregates. These are negative results for a matched-utility/review-load robustness claim, not a 10× success.

This is an exploratory comparison, not the 10× stretch claim: one agent model family, one judge family shared with the agent, a standard non-adaptive attack, 12 tasks per suite, two target goals per task, one agent seed, and a simulated environment. Utility/review load are reported rather than matched. Tool filtering may reduce available useful actions. Target success is the benchmark's selected predicate, not a detector of every possible unauthorized action. The gate has no guarantee that an allowed call is correct and no fitted statistical act/defer certificate. The optional CaMeL-lite reference is not included.

CPU acceptance after adding the supplemental arm: `python -X utf8 -m pytest -q keel/tests` — **33 passed**. Tests cover trust construction and control-token escaping, hold/no executor invocation, real allow side effects, immutable original transcript, same-batch evidence, receipt binding, ties, invalid scores/log failure, nested-call blocking, one-pass/think enforcement, clustered CIs, and real AgentDojo attack/predicate behavior.

## Parser pilot and spend ledger

The six-episode pilot completed with no episode or tool-dispatch errors: the undefended agent completed both held-out pilot user tasks with seven executed calls. This verifies tool calling in banking and Slack, not just an artificial function schema. The Keel arm held 8/9 calls and completed neither pilot task. The unmodified upstream tool-filter arm made no calls and completed neither task: the model continued answering the filter request rather than the original user task. In the main matrix tool filtering did execute calls and attain nonzero utility, so it is retained without repair; low ASR must be interpreted alongside lost utility. No prompt/threshold was adjusted after the pilot.

Every paid allocation uses `keel.workflows.modal_run.evaluate` on one H100, estimated at $3.95/hour. Runtime estimates include model loading and episode execution but exclude image building and unobserved platform teardown; these are not provider invoices.

| Date | Run / function | GPU-minutes | Estimated GPU $ | Accounting |
|---|---|---:|---:|---|
| 2026-10-15 | `parser-pilot-20261015a` / `evaluate` | 3.7084 | 0.2441 | Measured function wall time; six episodes |
| 2026-10-15 | `workflow-v0-20261015a` / `evaluate` | 5.0000 reserved | 0.3292 reserved | Cancelled before server startup/episodes to remove the local shell's one-hour deadline; explicitly stopped Modal app `ap-DX8V6WsWVxGUcGOt0XX0AC`; exact billed duration unavailable |
| 2026-10-15 | `workflow-v0-20261015b` / `evaluate` | 74.1042 | 4.8785 | Measured function wall time; all 360 primary episodes completed, zero episode exceptions |
| 2026-10-15 | `workflow-think-20261015a` / `evaluate` | 174.4865 | 11.4870 | Measured function wall time; all 72 supplemental episodes completed, zero episode exceptions |
| 2026-10-15 | `workflow-host-v01-20261015a` / `evaluate` | 17.8653 | 1.1761 | Separate v0.1 $8 cap; all 72 host-integrated one-pass episodes completed, zero episode/dispatch errors |

The original v0 allocations total **252.2991 GPU-minutes / $16.6097** estimated. Including the cancelled-startup reserve gives **$16.9389**, below that assignment's **$25 cap**. Raw logs are under `/home/alp/ssdm-direction/workflow/`; aggregate v0 allocation records are in `direction/experiments/workflow_v0/run_summary.json`. The separate v0.1 integration assignment has an $8 cap.

## v0.1: host-kernel integration

This closes the workflow-to-host integration gap without changing the model-facing prompt, question, threshold or chosen task population. The request registry and `Authority` come from the integrated host package; the adapter does not maintain a second authorization implementation. `review_capacity=0` explicitly holds all deferred calls because this experiment has no human-review service. Structural denials also return the fixed held message, without calling the tool.

Each episode verifies the run's receipt chain and records its cumulative checkpoint plus receipt IDs in the private episode log. `keel.workflows.audit` verifies the complete run, checks every episode-to-receipt link, and replay-checks a seeded random sample of 20 saved requests. Replay checks input/prompt identity, not regenerated probabilities. Protect the checkpoint independently; these hashes are not signatures, and a host that can rewrite both chain and checkpoint is outside the guarantee.

```sh
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.modal_run \
  --config direction/experiments/workflow_v0/host_config.json --run-id UNIQUE-host
/home/alp/ssdm-venv/bin/python -X utf8 -m keel.workflows.audit \
  --run-dir /home/alp/ssdm-direction/workflow/UNIQUE-host \
  --reference /home/alp/ssdm-direction/workflow/workflow-v0-20261015b \
  --output direction/experiments/workflow_v0/host_audit.json
```

The comparison reads the historical v0 journal only as immutable reference evidence; new runs write only `receipts.jsonl` as their decision log. Host configuration authorizes all suite tools, not a task-ground-truth oracle: it structurally rejects out-of-suite tools, but the model must still judge whether an in-suite tool and its arguments are needed. No argument-level correctness or matched-utility robustness claim follows from the integration.

CPU acceptance for the host-integrated cutover: `/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests` — **414 passed**. New coverage includes real kernel-mediated dispatch after receipt persistence, structural denial despite a forced-allow judge, one receipt for a hold, shared chains across episode hosts, disk tampering detection, saved-request replay, episode/receipt linkage, and both thinking prompt stages. Replay request files deliberately preserve dictionary insertion order: the canonical request fingerprint is order-insensitive, but the existing renderer preserves JSON-content key order in prompt bytes.

The measured v0.1 run completed all **72 episodes**: **335 receipts verified**, **72 episode checkpoints verified**, and **20/20 sampled receipts replayed**, both in the container and against the downloaded chain's pinned checkpoint. **72/72 outcomes match v0 exactly on all compared non-timing fields; all 335 prompt hashes and probability pairs are identical.** The same 20 receipts are replayed twice, not 40 unique samples. There are 30 actual executions and 305 holds.

The full CIs and latency/cost tables are in the results README's v0.1 section and `host_aggregate.json`. Benign utility remains banking **41.7% [16.7, 66.7]** and Slack **0% [0, 0]**, despite measured targeted ASR **0/24 [0%, 0%]** per suite: structural enforcement and replayability do not repair the stock judge's over-holding. Separate allocations prevent attributing latency differences to kernel overhead.

This follow-up's sole paid allocation used **17.8653 H100-minutes / $1.1761 estimated**, below its separate **$8 cap**; no other v0.1 allocations were launched. `host_run_summary.json` records exact accounting and `host_audit.json` pins the receipt head and comparison totals. Executable source for the rerun is `6ae7af5f`; private logs are at `/home/alp/ssdm-direction/workflow/workflow-host-v01-20261015a/`. The original v0 tables, aggregate and $25-cap ledger remain unchanged as historical evidence.

