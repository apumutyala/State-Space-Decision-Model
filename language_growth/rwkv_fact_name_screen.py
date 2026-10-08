"""Prepared paired candidate-name CPU screen. Default never loads a model."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time

from language_growth.fact_binding_screen import generate_binding_screen
from language_growth.operation_cache import digest, initial_operation
from language_growth.rwkv_operation_cache import (
    Capture, _ProcessHandle, assert_environment, authorize_worker, budget_check,
    sha, state_digest, supervise, write,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "96f5bd0c7298690d90e36918d5ca73cb302d447a"
PLAN = "plans/codex/L0H0_PARENT_BINDING_SCREEN.md"
RUNNER_PLAN = "plans/codex/L0_RWKV_FACT_NAME_SCREEN.md"
PREPARED = ROOT / "receipts/2026-10-07-L-fact-name-screen-retry-prepared"
OPERATION_EQUALITY_EVIDENCE = (
    "all 32 members: identical initial-operation requests excluding only request_id; "
    "one successful finite-score evaluation plus deterministic exact-logit cache replay "
    "through the same adapter; weaker than two independent forwards, not temporal retention"
)
CONFIG = {"seed": 20261007, "worlds": 16, "members_per_world": 2,
          "styles": ["record_ref", "entity_name"], "threads": 2,
          "min_start_bytes": 9 * 1024**3, "min_running_bytes": 3 * 1024**3,
          "timeout_seconds": 1200, "max_batch_rows": 2, "max_context": 2048,
          "bootstrap_seed": 9203, "bootstrap_samples": 2000}


class InitialOperationScoreMemo:
    """Run-local, serial, bounded stock-score reuse; never caches served answers."""
    name = "l0h0_exact_initial_operation_score_memo"
    max_entries = 32

    def __init__(self, backend, check=lambda: None):
        self.backend, self.check = backend, check
        self.stock_revision = deepcopy(backend.revision())
        policy = {"route": "initial-operation-empty-prefix-only", "max_entries": 32,
                  "key": "full-canonical-request-minus-request_id-and-fixed-backend-revision",
                  "admission": "successful-adapter-response-and-finite-complete-stock-scores"}
        self._revision = {**deepcopy(self.stock_revision), "operation_score_memo": policy,
            "revision_id": self.stock_revision["revision_id"] + ";operation_memo=" + digest(policy)[:12]}
        self.entries = {}
        self.pending = None
        self.hits = self.live_initial_scores = 0

    def revision(self):
        return deepcopy(self._revision)

    def key(self, request):
        if self.backend.revision() != self.stock_revision:
            raise ValueError("fixed operation-cache backend revision drift")
        if not initial_operation(request):
            return None
        content = deepcopy(request)
        content["ssdm"].pop("request_id", None)
        return digest({"request": content, "backend_revision": self.stock_revision})

    def plan(self, request):
        self.check()
        key = self.key(request)
        # Always retain live tokenization/context validation, including on hits.
        plan = deepcopy(self.backend.plan(request))
        if key in self.entries:
            plan.inference_calls = plan.total_input_tokens = 0
        return plan

    def score(self, request, deadline):
        self.check()
        key = self.key(request)
        if key in self.entries:
            entry = self.entries[key]
            result = deepcopy(entry["result"])
            source = {"hit": True, "key": key, "source_request_id": entry["request_id"],
                      "source_inference_calls": result.inference_calls,
                      "source_total_input_tokens": result.total_input_tokens,
                      "source_backend": result.extras}
            result.inference_calls = result.total_input_tokens = 0
            result.extras = {"operation_score_memo": source, "call_latency_ms": []}
            self.hits += 1
            # deadline() counts a model forward; a lookup starts none. Decider
            # still checks its wall deadline after score(), and check() guards RAM/time.
            self.check()
            return result
        if key is not None:
            self.live_initial_scores += 1
        result = self.backend.score(request, deadline)
        self.check()
        if key is not None and len(self.entries) < self.max_entries:
            self.pending = (key, deepcopy(result))
        return result

    def finish(self, request, response):
        pending, self.pending = self.pending, None
        if pending is None or not isinstance(response, dict) or response.get("ssdm", {}).get("status") != "ok":
            return
        key, result = pending
        if key != self.key(request) or len(self.entries) >= self.max_entries:
            return
        questions, logits = request["questions"], result.logits
        if (not isinstance(logits, dict) or set(logits) != set(questions) or any(
                not isinstance(logits[qid], dict) or set(logits[qid]) != set(q["criteria"])
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in logits[qid].values()) for qid, q in questions.items())):
            return
        self.entries[key] = {"result": result, "request_id": request["ssdm"].get("request_id")}

    def accounting(self):
        return {"entries": len(self.entries), "hits": self.hits,
                "live_initial_score_attempts": self.live_initial_scores, "max_entries": self.max_entries}


class SuccessfulMemoProvider:
    """Admit scores only after the ordinary adapter successfully serves them."""
    def __init__(self, decider, memo):
        self.decider, self.memo = decider, memo

    def revision(self):
        return self.decider.revision()

    def decide(self, request):
        self.memo.pending = None
        response = None
        try:
            response = self.decider.decide(request)
            return response
        finally:
            self.memo.finish(request, response)


def screen_data():
    examples = generate_binding_screen(CONFIG["seed"], CONFIG["worlds"])
    if len(examples) != 32 or len({e["pair_id"] for e in examples}) != 16:
        raise ValueError("complete frozen 16 paired worlds required")
    return examples


def separated_data(examples):
    """Keep evaluator targets out of the saved visible inputs and raw calls."""
    inputs, targets = [], []
    for example in examples:
        tr = example["training_only"]
        identity = {k: example[k] for k in ("task_id", "pair_id", "pair_member")}
        inputs.append({**identity, "model_inputs": example["model_inputs"],
                       "host": tr["host"], "catalog": tr["catalog"]})
        targets.append({**identity, **{k: tr[k] for k in
            ("kind", "field", "target_record_ref", "program", "targets")}})
    return inputs, targets


def registration():
    # These constants do not inspect/open the snapshot or import a model runtime.
    from language_growth.rwkv7_cpu_parent import MODEL_ID, SNAPSHOT, PINNED_HASHES
    examples = screen_data()
    inputs, targets = separated_data(examples)
    sources = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
        for folder in ("language_growth", "decide", "third_party")
        for p in (ROOT / folder).rglob("*.py")) + [PLAN, RUNNER_PLAN]
    return {"schema": "ssdm.rwkv-fact-name-screen.v1", "status": "prepared_not_run",
        "base_commit": BASE_COMMIT, "config": CONFIG,
        "scope": "synthetic development read/explain candidate display screen; no L0 qualification",
        "counts": {"worlds": 16, "members": 32, "members_per_style": 32,
                   "read_members": 16, "explain_members": 16},
        "visible_inputs_sha256": digest(inputs), "evaluator_targets_sha256": digest(targets),
        "examples_sha256": digest(examples),
        "expected_parent": {"model_id": MODEL_ID, "snapshot": SNAPSHOT.name,
                            "file_sha256": PINNED_HASHES},
        "runtime": {"interpreter": str(Path(sys.executable).resolve()), "python": sys.version,
                    "device": "cpu", "cuda_visible_devices": "-1"},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "numpy", "psutil")},
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "order": "even world: record_ref then entity_name; odd world: reversed; both members per style",
        "direct_retention": OPERATION_EQUALITY_EVIDENCE,
        "measurement_amendment": "remove direct bookends; memoize successful finite initial-operation scores only; preserve logical calls and full-state hashes; equality is deterministic replay, not independent repeated inference",
        "planned_requests": 128, "planned_forwards": 192,
        "planned_model_scored_requests": 96, "planned_operation_cache_hits": 32,
        "request_count_basis": "nominal read/explain two-slot paths only; model-selected compare paths may exceed 1200 seconds",
        "start_guard_basis": "9 GiB operational reserve after resource abort; not a model-only RAM estimate",
        "state_retention": "separate pre/post hashes of all parent parameters and registered buffers, including LM head",
        "primary": "entity_name minus record_ref useful-answer accuracy, paired by world; all failures retained",
        "secondary": ["kernel acceptance", "read/explain useful answers", "both-member useful answers",
                      "operation and fact selections", "same-request deterministic operation replay equality"],
        "selection": "one fixed display comparison; no tuning, training, parser correction or retries",
        "decision": "complete 16 worlds; >=2 both-member world gains and zero world losses; exact fact gain; zero operation losses; useful CI lower>0; exact paired initial-operation equality and unchanged full state; screening only",
        "execution": "explicit --execute only, offline owned CPU child; 9 GiB start reserve / 3 GiB run / 2 threads / 1200s cap",
        "cost_usd": 0, "adoption": False}


def verify_registration(value):
    expected = json.loads((PREPARED / "registration.json").read_text(encoding="utf-8"))
    if value != expected:
        raise ValueError("frozen screen source/software/data registration drift")


def preparation(out):
    value = registration()
    if (PREPARED / "registration.json").exists():
        verify_registration(value)
    inputs, targets = separated_data(screen_data())
    out.mkdir(parents=True, exist_ok=False)
    write(out / "registration.json", value)
    write(out / "screen-inputs.json", inputs)
    write(out / "evaluator-targets.json", targets)
    return value


def decision_fields(response):
    """Exact served decisions; wall-clock latency/request correlation is volatile."""
    env = response.get("ssdm", {})
    return {"answers": response.get("answers"), **{k: env.get(k) for k in
        ("status", "error", "abstention", "raw_probabilities", "served_probabilities", "served_config")}}


def selection_metrics(example, calls):
    """Evaluator only, after the actual free-running call. No corrected answer."""
    tr = example["training_only"]
    desired_operation = "op001" if tr["kind"] == "read" else "op002"
    operation, selected_fact = None, None
    for call in calls:
        qid, q = next(iter(call["request"]["questions"].items()))
        choice = (call["response"] or {}).get("answers", {}).get(qid, {}).get("choice")
        if qid == "slot-001-operation":
            operation = choice
        if qid == "slot-002-fact" and choice in q["criteria"]:
            # IDs enumerate the visible permitted record/field set in order.
            from language_growth.planner import _as_options, _candidate_sets, _fact_description
            state = json.loads(call["request"]["state"])
            facts, _, _ = _candidate_sets(tr["host"], tr["catalog"], state["task_input"])
            _, bindings = _as_options(facts, _fact_description, "f")
            selected_fact = bindings.get(choice)
    return {"operation_correct": operation == desired_operation,
            "fact_slot_observed": selected_fact is not None,
            "record_correct": selected_fact is not None and selected_fact["record_ref"] == tr["target_record_ref"],
            "field_correct": selected_fact is not None and selected_fact["field"] == tr["field"],
            "fact_correct": selected_fact is not None and selected_fact["record_ref"] == tr["target_record_ref"]
                            and selected_fact["field"] == tr["field"]}


def evaluate_screen(examples, provider, *, model, revision, served_config, raw, check=lambda: None):
    from language_growth.free_running_dev import run_development_tasks
    groups = {}
    for example in examples:
        groups.setdefault(example["pair_id"], []).append(example)
    if len(groups) != 16 or any(sorted(e["pair_member"] for e in members) != [0, 1]
                                for members in groups.values()):
        raise ValueError("all 16 complete paired worlds required")
    rows = {style: [] for style in CONFIG["styles"]}
    order = []
    operation_calls = {}
    for world, (pair_id, members) in enumerate(sorted(groups.items())):
        styles = CONFIG["styles"] if world % 2 == 0 else list(reversed(CONFIG["styles"]))
        for style in styles:
            for example in sorted(members, key=lambda e: e["pair_member"]):
                check()
                phase = f"{pair_id}/{example['pair_member']}/{style}"
                capture = Capture(provider, raw, phase)
                observation = run_development_tasks([example], capture, model=model,
                    parent_revision=revision, expected_served_config=served_config,
                    candidate_style=style, allow_repair=False)["observations"][0]
                check()
                row = {**observation, "pair_id": pair_id, "pair_member": example["pair_member"],
                       **selection_metrics(example, capture.calls)}
                from language_growth.kernel import check_program
                checked = check_program(observation["program"], example["training_only"]["host"],
                    example["training_only"]["catalog"]) if observation["program"] is not None else None
                row["checked_outputs"] = [output.to_dict() for output in checked.outputs] if checked is not None else []
                rows[style].append(row)
                order.append({"task_id": example["task_id"], "style": style})
                if capture.calls:
                    first = capture.calls[0]
                    request = deepcopy(first["request"])
                    request["ssdm"].pop("request_id", None)
                    operation_calls[example["task_id"], style] = {
                        "request": request, "decision": decision_fields(first["response"] or {})}
                # Save each completed attempt so timeout/termination retains its outcomes.
                write(raw.parent / f"{style}.json", rows[style])
                write(raw.parent / "evaluation-order.json", order)
    operation_equal = all(operation_calls.get((e["task_id"], "record_ref")) is not None and
        operation_calls[e["task_id"], "record_ref"]["decision"]["status"] == "ok" and
        operation_calls.get((e["task_id"], "record_ref")) == operation_calls.get((e["task_id"], "entity_name"))
        for e in examples)
    return rows, order, operation_equal


def paired_summary(rows, examples):
    """Cluster both renamed/reordered members within each generated world."""
    by_style = {style: {r["task_id"]: r for r in values} for style, values in rows.items()}
    ids = {e["task_id"] for e in examples}
    if len(ids) != 32 or any(len(rows[style]) != 32 or set(by_style[style]) != ids for style in CONFIG["styles"]):
        raise ValueError("complete 32-member conditions required")
    metrics = ("task_correct", "kernel_accepted", "operation_correct", "record_correct", "field_correct", "fact_correct")
    summary = {}
    for kind in ("all", "read", "explain"):
        worlds = {}
        for e in examples:
            if kind == "all" or e["training_only"]["kind"] == kind:
                worlds.setdefault(e["pair_id"], []).append(e["task_id"])
        result = {}
        for metric in (*metrics, "both_members_task_correct"):
            before, after = [], []
            for pair_ids in worlds.values():
                for style, values in (("record_ref", before), ("entity_name", after)):
                    observations = [by_style[style][i] for i in pair_ids]
                    values.append(float(all(r["task_correct"] for r in observations)) if metric == "both_members_task_correct"
                                  else sum(r[metric] for r in observations) / len(observations))
            # Preserve fractional two-member means when resampling whole worlds.
            import random
            delta = [a - b for a, b in zip(after, before)]
            rng = random.Random(CONFIG["bootstrap_seed"])
            draws = sorted(sum(rng.choice(delta) for _ in delta) / len(delta)
                           for _ in range(CONFIG["bootstrap_samples"]))
            result[metric] = {"baseline": sum(before) / len(before), "entity_name": sum(after) / len(after),
                "estimate": sum(delta) / len(delta), "ci95": [draws[50], draws[1950]],
                "n_worlds": len(delta), "n_members": 2 * len(delta),
                "unit": "paired world, retaining both renamed/reordered members", "samples": 2000, "seed": 9203}
        summary[kind] = result
    return summary


def continuation_gate(rows, examples, summary, *, frozen_parent, operation_equal):
    """Apply the existing L0h0 plan; never promote a partial or regressing screen."""
    # paired_summary validates completeness/unique matching before this gate.
    paired_summary(rows, examples)
    by_style = {style: {r["task_id"]: r for r in values} for style, values in rows.items()}
    worlds = {}
    for e in examples:
        worlds.setdefault(e["pair_id"], []).append(e["task_id"])
    gained = lost = 0
    for ids in worlds.values():
        before = all(by_style["record_ref"][i]["task_correct"] for i in ids)
        after = all(by_style["entity_name"][i]["task_correct"] for i in ids)
        gained += after and not before
        lost += before and not after
    operation_losses = sum(by_style["record_ref"][i]["operation_correct"] and
                           not by_style["entity_name"][i]["operation_correct"] for i in by_style["record_ref"])
    gates = {"at_least_two_both_member_world_gains": gained >= 2,
        "no_previously_successful_world_lost": lost == 0,
        "exact_fact_binding_improved": summary["all"]["fact_correct"]["estimate"] > 0,
        "no_operation_regression": operation_losses == 0,
        "useful_answer_ci_lower_positive": summary["all"]["task_correct"]["ci95"][0] > 0,
        "read_explain_nonregression": all(summary[kind][metric]["estimate"] >= 0
            for kind in ("read", "explain") for metric in ("task_correct", "kernel_accepted")),
        "full_parent_state_equal": frozen_parent,
        "arm_operation_equal": operation_equal}
    return {"eligible_for_review": all(gates.values()), "gates": gates,
            "both_member_world_gains": gained, "both_member_world_losses": lost,
            "operation_member_losses": operation_losses}


def run(out):
    import psutil
    assert_environment()
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        raise MemoryError("9 GiB start reserve before Torch/checkpoint load")
    frozen = json.loads((out / "registration.json").read_text(encoding="utf-8"))
    verify_registration(frozen)
    if registration() != frozen:
        raise ValueError("runtime screen registration drift")
    examples = screen_data()
    inputs, targets = separated_data(examples)
    if (json.loads((out / "screen-inputs.json").read_text()) != inputs or
            json.loads((out / "evaluator-targets.json").read_text()) != targets):
        raise ValueError("saved screen inputs/targets drift")
    started = time.monotonic()
    check = lambda: budget_check(started, psutil.virtual_memory().available)
    import torch
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only guard")
    check()
    from language_growth.rwkv7_cpu_parent import load_cpu_parent
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.adapter import Decider
    plugin, parent, evidence = load_cpu_parent()
    check()
    parent.eval().requires_grad_(False)
    before_state = state_digest(parent)
    write(out / "state-before.json", {"full_state_sha256": before_state})
    backend = HFReadoutBackend(plugin, "jrt-rwkv-g1-v1", max_batch_rows=CONFIG["max_batch_rows"])

    class Bounded:
        name = backend.name
        def revision(self): return backend.revision()
        def plan(self, request):
            check()
            return backend.plan(request)
        def score(self, request, deadline):
            check()
            result = backend.score(request, lambda: (check(), deadline()))
            check()
            return result

    memo = InitialOperationScoreMemo(Bounded(), check=check)
    decider = Decider(memo, backend_max_input_tokens=CONFIG["max_context"])
    provider = SuccessfulMemoProvider(decider, memo)
    revision = provider.revision()["revision_id"]
    config = {"calibration": decider.calibration.identity(),
              "policy_masking": "none (served covers every supplied option)"}
    raw = out / "raw.jsonl"
    rows, order, operation_equal = evaluate_screen(examples, provider, model=plugin.model_id,
        revision=revision, served_config=config, raw=raw, check=check)
    check()
    after_state = state_digest(parent)
    write(out / "state-after.json", {"full_state_sha256": after_state})
    frozen_parent = before_state == after_state
    summary = paired_summary(rows, examples)
    gate = continuation_gate(rows, examples, summary,
                             frozen_parent=frozen_parent, operation_equal=operation_equal)
    if not frozen_parent or not operation_equal:
        raise ValueError("parent state or identical-input operation decision drift")
    write(out / "receipt.json", {"schema": "ssdm.rwkv-fact-name-screen-result.v1", "status": "complete",
        "registration_sha256": sha(out / "registration.json"), "parent_evidence": evidence,
        "revision": revision, "summary": summary, "order": order,
        "arm_operation_equal": operation_equal,
        "operation_equality_evidence": OPERATION_EQUALITY_EVIDENCE,
        "operation_score_memo": memo.accounting(),
        "full_state_sha256_before": before_state, "full_state_sha256_after": after_state,
        "frozen_parent": frozen_parent, **gate, "adopted": False,
        "cost_usd": 0, "elapsed_seconds": time.monotonic() - started,
        "artifact_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file() and
            p.name not in ("stdout.txt", "stderr.txt", "launcher.jsonl")}})


def record_incomplete(out, reason, exit_code, **details):
    """Terminal resource stops are results, never efficacy/success receipts."""
    import psutil
    write(out / "receipt.json", {"schema": "ssdm.rwkv-fact-name-screen-result.v1",
        "status": "incomplete", "stop_reason": reason, "exit_code": exit_code,
        "cost_usd": 0, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_commit": BASE_COMMIT, "command_argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "hardware": {"processor": platform.processor(), "platform": platform.platform(),
                     "total_physical_bytes": psutil.virtual_memory().total, "device": "cpu"},
        "artifact_sha256": {p.name: sha(p) for p in out.iterdir()
                            if p.is_file() and p.name != "receipt.json"}, **details})


def finalize_launch(out, exit_code):
    receipt = out / "receipt.json"
    if exit_code == 0 and receipt.exists() and json.loads(receipt.read_text())["status"] == "complete":
        return 0
    events = [json.loads(line) for line in (out / "launcher.jsonl").read_text().splitlines()]
    terminal = events[-1] if events else {}
    record_incomplete(out, terminal.get("reason", "worker_failed" if exit_code else "missing_complete_result"),
                      exit_code, launcher_terminal=terminal)
    return exit_code if exit_code else 1


def worker(out, pipe, launcher_pid, secret):
    authorize_worker(pipe, launcher_pid, secret)
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                run(out)
            except Exception as exc:
                write(out / "failure.json", {"status": "failed", "type": type(exc).__name__,
                    "message": str(exc), "cost_usd": 0})
                raise


def launch(out):
    import psutil
    from language_growth.l0_qwen_cpu_probe import _offline_env
    preparation(out)
    available_at_start = psutil.virtual_memory().available
    if available_at_start < CONFIG["min_start_bytes"]:
        record_incomplete(out, "start_memory_guard", 125,
                          available_bytes=available_at_start)
        return 125
    from language_growth.l0_qwen_cpu_probe import _write_event
    _write_event(out / "launcher.jsonl", "launcher_start", available_bytes=available_at_start,
                 required_start_bytes=CONFIG["min_start_bytes"])
    receive = send = process = None
    failure, cleanup_errors = None, []
    failure_code = 1
    phase = "spawn"
    try:
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        secret = secrets.token_bytes(32)
        process = context.Process(target=worker, args=(out, receive, os.getpid(), secret), daemon=True)
        old = dict(os.environ)
        try:
            os.environ.update(_offline_env())
            process.start()
        finally:
            os.environ.clear()
            os.environ.update(old)
        phase = "handshake"
        receive.close()
        send.send_bytes(secret)
        send.close()
        phase = "supervisor"
        handle = _ProcessHandle(process)
        code = supervise(handle, out / "launcher.jsonl", available=lambda: psutil.virtual_memory().available)
    except KeyboardInterrupt as exc:
        failure_code = 130
        failure = {"reason": "launcher_interrupted", "exception_type": type(exc).__name__,
                   "exception_message": str(exc)}
    except Exception as exc:
        failure = {"reason": f"{phase}_error", "exception_type": type(exc).__name__,
                   "exception_message": str(exc)}
    finally:
        for connection in (receive, send):
            if connection is not None:
                try:
                    connection.close()
                except Exception as exc:
                    cleanup_errors.append({"type": type(exc).__name__, "message": str(exc)})
        # A failed start may still have assigned a PID. Never join an unstarted
        # process, and never discover/kill processes outside this owned handle.
        if process is not None and process.pid is not None:
            try:
                if process.exitcode is None:
                    process.kill()
                process.join()
            except Exception as exc:
                cleanup_errors.append({"type": type(exc).__name__, "message": str(exc)})
    if failure is not None or cleanup_errors:
        failure = failure or {"reason": "cleanup_error"}
        reason = failure.pop("reason")
        record_incomplete(out, reason, failure_code, **failure, cleanup_errors=cleanup_errors)
        return failure_code
    try:
        return finalize_launch(out, code)
    except Exception as exc:
        record_incomplete(out, "finalization_error", 1,
                          exception_type=type(exc).__name__, exception_message=str(exc))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.execute:
        return launch(out)
    preparation(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
