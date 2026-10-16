"""CPU-only synthetic review service; not the kernel's nonexistent approval path."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
import math

import numpy as np

POLICIES = ("fifo", "fair", "cap", "reserved")
CONTROLS = ("infinite", "always_hold", "unsafe_fail_open")
SEED = 2026101603


@dataclass(frozen=True, slots=True)
class Job:
    arrival: float
    principal: int
    deferred: bool
    harmful: bool
    entitled: bool
    service: float
    deadline: float


class PendingQueue:
    """Bound waiting storage, independently of the number of active reviewers."""

    def __init__(self, policy: str, capacity: int):
        if policy not in POLICIES or type(capacity) is not int or capacity < 0:
            raise ValueError("unknown policy or invalid capacity")
        self.policy, self.capacity = policy, capacity
        self.items: dict[int, Job] = {}
        self.principals: OrderedDict[int, dict[int, Job]] = OrderedDict()
        self.unentitled = 0

    def admit(self, key: int, job: Job) -> str | None:
        if len(self.items) >= self.capacity:
            return "review_queue_full"
        if self.policy == "cap" and len(self.principals.get(job.principal, ())) >= max(1, self.capacity // 8):
            return "principal_cap"
        if self.policy == "reserved" and not job.entitled and self.unentitled >= self.capacity // 4:
            return "reserved_capacity"
        self.items[key] = job
        self.principals.setdefault(job.principal, {})[key] = job
        self.unentitled += not job.entitled
        return None

    def remove(self, key: int) -> Job:
        job = self.items.pop(key)
        own = self.principals[job.principal]
        del own[key]
        if not own:
            del self.principals[job.principal]
        self.unentitled -= not job.entitled
        return job

    def take(self, entitled_only: bool = False) -> tuple[int, Job] | None:
        if not self.items:
            return None
        if entitled_only:
            key = next((k for k, j in self.items.items() if j.entitled), None)
            if key is None:
                return None
        elif self.policy == "fair":
            principal = next(iter(self.principals))
            key = next(iter(self.principals[principal]))
            self.principals.move_to_end(principal)
        else:
            key = next(iter(self.items))
        return key, self.remove(key)


def simulate(jobs: list[Job], policy: str, capacity: int, reviewers: int = 4,
             trace: list | None = None) -> dict:
    """Replay frozen jobs; a perfect review rejects harm, never auto-releases."""
    if type(reviewers) is not int or reviewers < 0:
        raise ValueError("reviewers must be a nonnegative integer")
    if any(not all(math.isfinite(x) for x in (j.arrival, j.service, j.deadline))
           or j.service < 0 or j.deadline <= j.arrival for j in jobs):
        raise ValueError("jobs require finite times, nonnegative service, and positive lifetime")
    queue = PendingQueue(policy, capacity)
    # Completion (0) wins deadline ties (1); expiry wins arrival ties (2).
    events = [(j.arrival, 2, k, -1) for k, j in enumerate(jobs)]
    heapq.heapify(events)
    free = [True] * reviewers
    reserved_reviewers = reviewers * 3 // 4
    benign = sum(not j.harmful for j in jobs)
    need_review = sum(j.deferred and not j.harmful for j in jobs)
    completed = harmful = rejected_harmful = held_benign = started_benign = 0
    started = 0
    work = 0.0
    waits = []
    max_waiting = 0
    while events:
        now, kind, key, worker = heapq.heappop(events)
        job = jobs[key]
        if kind == 0:
            free[worker] = True
            if now <= job.deadline:
                rejected_harmful += job.harmful
                completed += not job.harmful
        elif kind == 1:
            if key in queue.items:
                queue.remove(key)
        else:
            if not job.deferred:
                completed += not job.harmful
                harmful += job.harmful
                status, reason = "executed", "authorized_dispatch"
            else:
                reason = queue.admit(key, job)
                if reason is not None:
                    held_benign += not job.harmful
                    status = "held"
                else:
                    status, reason = "deferred", "gate_deferred"
                    heapq.heappush(events, (job.deadline, 1, key, -1))
                    max_waiting = max(max_waiting, len(queue.items))
            if trace is not None:
                trace.append((key, status, reason))
        if not queue.items:
            continue
        for worker in range(reviewers):
            if not free[worker]:
                continue
            item = queue.take(policy == "reserved" and worker < reserved_reviewers)
            while item is not None and item[1].deadline <= now:
                item = queue.take(policy == "reserved" and worker < reserved_reviewers)
            if item is None:
                continue
            picked, active = item
            free[worker] = False
            started += 1
            work += active.service
            if not active.harmful:
                started_benign += 1
                waits.append(now - active.arrival)
            heapq.heappush(events, (now + active.service, 0, picked, worker))
    return {
        "completion": completed / benign if benign else None,
        "harmful_effects": harmful,
        "p95_wait": float(np.quantile(waits, .95)) if waits else None,
        "starvation": (need_review - started_benign) / need_review if need_review else 0.0,
        "held_benign": held_benign / benign if benign else None,
        "review_work": work,
        "reviews_started": started,
        "rejected_harmful": rejected_harmful,
        "max_waiting": max_waiting,
    }


def controls(jobs: list[Job], fifo: dict) -> dict[str, dict]:
    """Exact controls: no service, hold everything, and unsafe FIFO release.

    Unsafe release does not cancel in-service work. Waiting expiry/overflow still
    remove/omit the item exactly as in FIFO, so it shares FIFO's resource path.
    Its wait quantile refers to reviews that start, not release latency.
    """
    harmful = sum(j.harmful for j in jobs)
    zero = dict(completion=1.0, harmful_effects=0, p95_wait=0.0, starvation=0.0,
                held_benign=0.0, review_work=0.0, reviews_started=0,
                rejected_harmful=harmful, max_waiting=0)
    return {
        "infinite": {**zero, "reviews_started": sum(j.deferred for j in jobs)},
        "always_hold": {**zero, "completion": 0.0, "p95_wait": None,
                        "starvation": 1.0, "held_benign": 1.0, "rejected_harmful": 0},
        "unsafe_fail_open": {**fifo, "completion": 1.0,
                             "harmful_effects": harmful - fifo["rejected_harmful"],
                             "held_benign": 0.0},
    }


def schedule(rho: float, cv: int, replicate: int, rate: float, timing: str,
             identity: str, *, n: int = 1000, development: bool = False) -> list[Job]:
    """Independent schedules, common job-level draws across policies/rates/labels."""
    if rho <= 0 or cv not in (0, 1, 2) or rate < 0:
        raise ValueError("invalid utilization, service variability, or attack rate")
    if timing not in ("sustained", "burst") or identity not in ("authenticated", "sybil", "entitlement_breach"):
        raise ValueError("unknown attack timing or identity assumption")
    # Separate streams keep every benign request unchanged when attack count varies.
    seeds = np.random.SeedSequence([SEED, int(development), round(rho * 100), cv, replicate]).spawn(2)
    benign_rng, attack_rng = (np.random.default_rng(s) for s in seeds)
    phase = float(attack_rng.random())
    horizon = n * .8 / (4 * rho)
    jobs = []
    for is_attack, rng, count in ((False, benign_rng, n), (True, attack_rng, round(n * rate))):
        # Row-wise draws make attack rate curves nested prefixes of the same stream.
        draws = rng.random((count, 5))
        for u_time, u_id, u_gate, u_service, u_phase in draws:
            principal = int(u_id * (32 if identity != "authenticated" else 1)) + 8 if is_attack else int(u_id * 8)
            deferred = is_attack or u_gate < (.6 if principal < 4 else 1.0)
            t = u_time
            if is_attack and timing == "burst":
                t = ((math.floor(t * 10) + (t * 10 % 1) * .2) / 10 + phase) % 1.0
            arrival = t * horizon
            if cv == 0:
                service = 1.0
            elif cv == 1:
                service = -math.log1p(-u_service)
            else:
                # Box-Muller uses two frozen uniforms; mean-one lognormal CV=2.
                normal = math.sqrt(-2 * math.log(max(u_service, 1e-15))) * math.cos(2 * math.pi * u_phase)
                service = math.exp(math.sqrt(math.log(5)) * normal - math.log(5) / 2)
            jobs.append(Job(arrival, principal, bool(deferred), is_attack,
                            not is_attack or identity == "entitlement_breach", service, arrival + 8))
    return jobs


def validate_host() -> dict:
    """Concrete same-principal flooding, then a drain; not a review approval test."""
    from keel.host import ActionProposal, Authority, HostKernel
    from keel.protocol import Answer, DecisionResult, OperatorPolicy, Question
    from keel.protocol.receipts import ReceiptLog

    cases = []
    for capacity in (0, 1, 2, 3):
        calls = []
        host = HostKernel(Authority(OperatorPolicy("p", "Synthetic admission check", authorized_actions=("a",)),
                                  {"worker": ("a",)}, {"actor": ("worker",)}),
                          principal="actor", executor=calls.append, receipt_log=ReceiptLog(),
                          review_capacity=capacity, model_id="synthetic", model_revision="frozen",
                          gate_version="frozen", calibration_id="none")
        queue = PendingQueue("fifo", capacity)
        for k, role in enumerate(["attack"] * (capacity + 1) + ["benign_defer", "benign_act", "drain", "benign_defer"]):
            if role == "drain":
                real = host.review_queue.pop()
                sim = queue.take()
                assert (real is None) == (sim is None)
                continue
            deferred = role != "benign_act"
            job = Job(float(k), 0, deferred, role == "attack", True, 1, k + 8)
            reason = queue.admit(k, job) if deferred else None
            expected = ("held", reason) if reason else (("deferred", "gate_deferred") if deferred else ("executed", "authorized_dispatch"))
            segment = host.inputs.untrusted(f"s{k}", {"synthetic_role": role}, kind="tool_output")
            request = host.request(f"r{k}", (segment,), (Question("q", "choice", "Route?", {"a": "A", "b": "B"}),))
            result = DecisionResult(request.request_id, {"q": Answer("q", {"a": .8, "b": .2}, None if deferred else "a")},
                                    not deferred, ["synthetic_defer"] if deferred else [])
            outcome = host.process(request, ActionProposal("a", {}, "actor", (segment.segment_id,)), result)
            assert (outcome.status, outcome.reason) == expected
            assert len(queue.items) == len(host.review_queue)
            cases.append({"capacity": capacity, "role": role, "status": outcome.status, "reason": outcome.reason})
        assert len(calls) == 1
        assert host.receipt_log.verify().count == capacity + 4
    return {"matched_outcomes": len(cases), "capacities": [0, 1, 2, 3], "same_principal": True,
            "executor_calls_per_trace": 1, "cases": cases}
