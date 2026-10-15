# Keel in AgentDojo workflows

This package runs a real tool-using agent against AgentDojo's simulated banking and Slack environments. Tools mutate benchmark environments; no real bank account, Slack workspace, or external recipient is connected.

## Implementation

- `gate_element.py` replaces AgentDojo's `ToolsExecutor`. It judges **each proposed call before dispatch**, including calls before the first tool output and each member of a multi-call batch. Earlier outputs in the same batch are visible to later decisions.
- The authenticated, host-provided user task is trusted metadata. Every previous tool result is an untrusted `tool_output` segment, sourced by tool name. Proposed tool names/arguments are also untrusted data, never interpolated into the system turn or question. The operator policy authorizes only actions needed for the user's task, including necessary information gathering.
- `keel.protocol.render(mode="trust")` uses the existing escaped untrusted-record boundary. The Qwen3.5-4B judge reads one token's raw label log-probabilities, normalized only over allow/hold, exactly as the shared judge runtime. `p(allow) > p(hold)` allows; ties hold. No workflow-specific training or threshold tuning is performed.
- The supplemental `keel_think` arm uses the same request, policy, options and threshold, but first samples one reasoning trace (up to 1,024 tokens; temperature 0.6, top-p 0.95, top-k 20, seed 0), closes it with the shared `close_think`, then reads the label. Supplemental `judge_samples.jsonl` records raw log-probabilities, reasoning, truncation and the final readout-prompt hash outside git. This is not a trained or tuned judge.
- Hold returns the fixed message `This tool call is held for human review. It was not executed.` The underlying executor is not called. This experiment measures review demand; there is no human approver or automatic retry-to-permission.
- Before dispatch, one JSONL record records request fingerprint, rendered prompt SHA-256, probabilities, outcome, call identity and judge latency. A score/logging exception aborts before the current call executes. Nested AgentDojo `FunctionCall` objects are held rather than allowing implicit unjudged subcalls; normal OpenAI JSON tool arguments are supported.
- All arms use the same upstream pipeline composition, system message and 15-iteration tool loop. `none`, `spotlighting_with_delimiting`, `repeat_user_prompt`, and `tool_filter` are AgentDojo's implementations, not approximations. `keel` replaces only the no-defense executor.
- `runner.py` preserves upstream utility/security predicates but supplies **executed** calls, not assistant proposals, to trace-based predicates. It latches targeted success after each actual dispatch, so later actions cannot erase a successful attack. Failed/aborted episodes remain in denominators and retain any already-executed target success. It does not rerun incomplete episodes against a partially mutated environment.

## Architecture and dependencies

A single Modal H100 runs a localhost vLLM OpenAI-compatible server for Qwen3.5-9B (55% GPU memory) and an in-process Qwen3.5-4B label judge (30%). Both use 32,768-token contexts, eager execution and the shared `ssdm-hf-cache` volume. This avoids cross-container per-call latency and paying for a second GPU. The judge is not a concurrent agent: episodes and their calls run sequentially. Raw logs persist on `keel-results` under `workflow/<run-id>` and are downloaded to `/home/alp/ssdm-direction/workflow/<run-id>/`.

Pinned dependencies: `agentdojo==0.1.35`, `vllm==0.30.0`; shared Modal environment uses Modal 1.5.3. Agent and judge revisions are read from `keel.runtime.modal_app.MODELS` and written to every run record. The native Qwen3.5 chat template is paired with `qwen3_coder` tool-call parser and `qwen3` reasoning parser, as recommended by the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B#deployment). Agent thinking is disabled, generation is greedy (`temperature=0`, seed 0, at most 1,024 output tokens). AgentDojo maps system messages to OpenAI's developer role; the endpoint adapter maps these back to Qwen's system role and explicitly sends zero temperature for **all** arms and tool-filter queries.

Sources: [AgentDojo pipelines and defenses](https://agentdojo.spylab.ai/concepts/agent_pipeline/), [benchmark](https://github.com/ethz-spylab/agentdojo/blob/main/src/agentdojo/benchmark.py), [upstream tool execution](https://github.com/ethz-spylab/agentdojo/blob/main/src/agentdojo/agent_pipeline/tool_execution.py), [pinned package release](https://pypi.org/project/agentdojo/0.1.35/).

## Reproduction

From the repository root, with the configured Python environment:

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

Unique run IDs avoid accidental overwrite; duplicate episode IDs are rejected by aggregation. `--download-only` retrieves a run without launching paid inference. Per-episode text, model outputs and decision journals must not enter git.

## Measurement and limits

Results and exact matrix are in [`direction/experiments/workflow_v0`](../../direction/experiments/workflow_v0/). All task IDs and attack goals are fixed in `config.json` before main inference. Bootstrap CIs resample user tasks, carrying both attacked goals together; hold rate is held/proposed calls, not an unweighted mean of episode ratios. Episodes with any hold separately measure review demand. A zero-success bootstrap interval can be degenerate and is not a certificate of zero population risk.

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

