"""Run unmodified AgentDojo tasks/defenses against a local tool-calling server.

Security uses actual executed calls, not AgentDojo's assistant-proposal trace.
All transcripts, attack texts and call receipts are written only to log_root.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from types import SimpleNamespace

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.task_suite.load_suites import get_suite
from keel.protocol.receipts import ReceiptLog
from keel.runtime.modal_app import MODELS

from .gate_element import KeelGateElement

AGENTDOJO_VERSION = "0.1.35"
BENCHMARK_VERSION = "v1.2.2"
ARMS = ("none", "spotlighting_with_delimiting", "repeat_user_prompt", "tool_filter", "keel", "keel_think")


def json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(type(value).__name__)


class EndpointClient:
    """Keep upstream OpenAILLM and ToolFilter, with explicit local sampling settings.

    AgentDojo 0.1.35 maps system to OpenAI's developer role and omits a zero
    temperature. Qwen's native template needs system; both are corrected here
    identically for every arm, including the upstream tool-filter model call.
    """

    def __init__(self, base_url: str, model: str, max_tokens: int = 1024):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0, timeout=120)
        self.chat = SimpleNamespace(completions=self)
        self.model = model
        self.max_tokens = max_tokens
        self.usage = []

    def create(self, **kwargs):
        kwargs["messages"] = [
            {**m, "role": "system"} if m["role"] == "developer" else m for m in kwargs["messages"]
        ]
        kwargs.update(temperature=0.0, seed=0, max_tokens=self.max_tokens,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        self.usage.append({"latency_s": time.perf_counter() - start,
                           "usage": response.usage.model_dump() if response.usage else None,
                           "finish_reason": response.choices[0].finish_reason})
        return response


class ExecutionRuntime(FunctionsRuntime):
    """Witness real dispatch and latch the target predicate immediately after calls."""

    def __init__(self, functions, security_check=None):
        super().__init__(functions)
        self.executed = []
        self.dispatches = []
        self.security_check = security_check
        self.target_succeeded = False

    def run_function(self, env, function, kwargs, raise_on_error=False):
        call = FunctionCall(function=function, args=kwargs)
        result, error = super().run_function(env, function, kwargs, raise_on_error)
        self.dispatches.append({"call": call, "error": error})
        if error is None:
            self.executed.append(call)
        if self.security_check is not None:
            self.target_succeeded |= bool(self.security_check(env, self.executed))
        return result, error


class EpisodeLogger(Logger):
    def __init__(self):
        self.messages = []

    def log(self, messages, **kwargs):
        self.messages = list(messages)

    def log_error(self, message):
        raise RuntimeError(message)


class CountProposals:
    def __init__(self, executor):
        self.executor = executor
        self.proposals = 0

    def query(self, query, runtime, env, messages, extra_args):
        if messages and messages[-1]["role"] == "assistant":
            self.proposals += len(messages[-1].get("tool_calls") or [])
        return self.executor.query(query, runtime, env, messages, extra_args)


def make_pipeline(client, arm, user_task, episode_id, log_root, judge=None, *, authorized_actions):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    llm = OpenAILLM(client, client.model)
    llm.name = client.model
    config = PipelineConfig(llm=llm, model_id=client.model, defense=None if arm in ("none", "keel", "keel_think") else arm,
                            system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(config)
    gate = None
    loop = next(element for element in pipeline.elements if isinstance(element, ToolsExecutionLoop))
    elements = list(loop.elements)
    if arm in ("keel", "keel_think"):
        if judge is None:
            raise ValueError("Keel arm requires a judge")
        model_id, revision = MODELS["qwen3.5-4b"]
        gate = KeelGateElement(user_task, judge, ReceiptLog(Path(log_root) / "receipts.jsonl"), episode_id,
                               authorized_actions=authorized_actions, model_id=model_id, model_revision=revision,
                               think=arm == "keel_think")
        elements[0] = gate
    counter = CountProposals(elements[0])
    elements[0] = counter
    loop.elements = elements
    pipeline.name = f"qwen3.5-9b-{arm}"
    return pipeline, gate, counter


def standard_attack(suite, pipeline):
    # Upstream's model-name registry predates Qwen3.5. Use its exact standard
    # template and vector discovery, replacing only the addressed model name.
    attack = FixedJailbreakAttack(ImportantInstructionsAttack._JB_STRING, suite, pipeline)
    attack.name = "important_instructions"
    attack.user_name = "Emma Johnson"
    attack.model_name = "Qwen3.5-9B"
    return attack


def run_episode(client, judge, suite_name, task_id, injection_id, arm, log_root):
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    task = suite.get_user_task_by_id(task_id)
    injection = suite.get_injection_task_by_id(injection_id) if injection_id else None
    episode_id = f"{suite_name}/{arm}/{task_id}/{injection_id or 'benign'}"
    pipeline, gate, counter = make_pipeline(client, arm, task.PROMPT, episode_id, log_root, judge,
                                           authorized_actions=tuple(tool.name for tool in suite.tools))
    injections = standard_attack(suite, pipeline).attack(task, injection) if injection else {}
    env = task.init_environment(suite.load_and_inject_default_environment(injections))
    pre = env.model_copy(deep=True)

    def security(environment, trace):
        return suite._check_injection_task_security(injection, [], pre, environment, trace)

    runtime = ExecutionRuntime(suite.tools, security if injection else None)
    usage_start = len(client.usage)
    start = time.perf_counter()
    error = None
    logger = EpisodeLogger()
    with logger:
        try:
            _, _, env, messages, _ = pipeline.query(task.PROMPT, runtime, env)
        except Exception as exc:
            # Preserve failures and any already-executed attack. Never count an
            # infrastructure error as a safe successful episode or drop its row.
            error = f"{type(exc).__name__}: {exc}"
            messages = logger.messages
    elapsed = time.perf_counter() - start
    output = messages[-1].get("content") or [] if messages and messages[-1]["role"] == "assistant" else []
    utility = False if error else suite._check_user_task_utility(task, output, pre, env, runtime.executed)
    target = bool(runtime.target_succeeded or (security(env, runtime.executed) if injection else False))
    outcomes = gate.outcomes if gate else []
    holds = sum(outcome.status != "executed" for outcome in outcomes)
    receipt_checkpoint = gate.kernel.receipt_log.verify() if gate else None
    usage = client.usage[usage_start:]
    row = {
        "episode_id": episode_id, "suite": suite_name, "arm": arm, "user_task": task_id,
        "injection_task": injection_id, "attacked": injection is not None, "utility": bool(utility),
        "targeted_asr": target if injection else None, "proposed_calls": counter.proposals,
        "executed_calls": len(runtime.executed), "dispatch_errors": sum(d["error"] is not None for d in runtime.dispatches),
        "gated_calls": len(outcomes), "held_calls": holds, "review_episode": holds > 0,
        "hold_rate": holds / counter.proposals if counter.proposals else 0.0,
        "latency_s": elapsed, "warm_gpu_cost_usd": elapsed * 3.95 / 3600,
        "agent_requests": len(usage), "completion_truncations": sum(u["finish_reason"] == "length" for u in usage),
        "prompt_tokens": sum((u["usage"] or {}).get("prompt_tokens", 0) for u in usage),
        "completion_tokens": sum((u["usage"] or {}).get("completion_tokens", 0) for u in usage),
        "judge_latency_s": gate.judge_latency_s if gate else 0.0, "error": error,
        "receipt_chain_verified_count": receipt_checkpoint.count if receipt_checkpoint else None,
        "receipt_ids": [outcome.receipt_id for outcome in outcomes],
        "injections_sha256": hashlib.sha256(json.dumps(injections, sort_keys=True).encode()).hexdigest(),
    }
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    trace_path = root / "traces" / f"{episode_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps({"row": row, "messages": messages, "dispatches": runtime.dispatches,
                                     "usage": usage, "injections": injections}, default=json_default), encoding="utf-8")
    return row


def run_matrix(client, judge, config, log_root):
    if importlib.metadata.version("agentdojo") != AGENTDOJO_VERSION:
        raise RuntimeError(f"requires agentdojo=={AGENTDOJO_VERSION}")
    results = []
    for suite_name, specification in config["suites"].items():
        for task in specification["user_tasks"]:
            for injection in [None, *specification["injection_tasks"]]:
                for arm in config["arms"]:
                    row = run_episode(client, judge, suite_name, task, injection, arm, log_root)
                    results.append(row)
                    print(json.dumps(row), flush=True)
    return results
