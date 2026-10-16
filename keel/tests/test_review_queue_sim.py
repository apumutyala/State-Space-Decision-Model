"""Deterministic contracts for the reusable SIM engine, not experiment outcomes."""
from dataclasses import replace

import pytest

from direction.experiments.review_queue.sim import (
    Job, PendingQueue, POLICIES, controls, schedule, simulate, validate_host,
)


def job(t=0.0, principal=0, *, harmful=False, service=1.0, deadline=8.0, deferred=True):
    return Job(t, principal, deferred, harmful, not harmful, service, deadline)


def test_fifo_matches_real_host_same_principal_flood():
    check = validate_host()
    assert check["matched_outcomes"] == 22
    assert all(c["status"] == "held" for c in check["cases"]
               if c["role"] == "benign_defer" and c["capacity"] == 0)


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_storage_holds_even_with_idle_reviewers(policy):
    trace = []
    stats = simulate([job()], policy, 0, trace=trace)
    assert trace == [(0, "held", "review_queue_full")]
    assert stats["completion"] == 0 and stats["starvation"] == 1
    assert stats["reviews_started"] == 0


def test_completion_deadline_tie_counts_and_late_service_is_not_cancelled():
    tied = simulate([job(service=8)], "fifo", 1, reviewers=1)
    assert tied["completion"] == 1
    jobs = [job(harmful=True, service=10, deadline=2), job(t=3, deadline=5)]
    stats = simulate(jobs, "fifo", 1, reviewers=1)
    assert stats["completion"] == 0 and stats["starvation"] == 1
    assert stats["review_work"] == 10 and stats["harmful_effects"] == 0


def test_expiry_frees_storage_but_never_approves():
    trace = []
    jobs = [job(service=5), job(t=.1, deadline=1), job(t=1, deadline=8)]
    stats = simulate(jobs, "fifo", 1, reviewers=1, trace=trace)
    assert [status for _, status, _ in trace] == ["deferred"] * 3
    assert stats["completion"] == pytest.approx(2 / 3)
    assert stats["reviews_started"] == 2


def test_equal_deadline_waiters_do_not_start_when_expired():
    jobs = [job(service=2, deadline=4), job(t=.1, deadline=2), job(t=.2, deadline=2)]
    stats = simulate(jobs, "fifo", 2, reviewers=1)
    assert stats["reviews_started"] == 1


def test_round_robin_is_not_fifo_and_cap_counts_waiting_only():
    fair = PendingQueue("fair", 16)
    for k, principal in enumerate((0, 0, 1, 1)):
        assert fair.admit(k, job(principal=principal)) is None
    assert [fair.take()[0] for _ in range(4)] == [0, 2, 1, 3]
    cap = PendingQueue("cap", 16)
    assert cap.admit(0, job()) is None and cap.admit(1, job()) is None
    assert cap.admit(2, job()) == "principal_cap"
    cap.take()
    assert cap.admit(2, job()) is None
    assert cap.admit(3, job(principal=1)) is None


def test_reservation_depends_on_entitlement_not_harm_label():
    queue = PendingQueue("reserved", 4)
    assert queue.admit(0, job(principal=8, harmful=True)) is None
    assert queue.admit(1, job(principal=9, harmful=True)) == "reserved_capacity"
    assert queue.admit(1, replace(job(principal=9, harmful=True), entitled=True)) is None
    assert queue.take(entitled_only=True)[0] == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_frozen_schedules_conserve_storage_and_contain_deferred_harm(policy):
    jobs = schedule(.75, 2, 3, 4, "burst", "sybil", n=100, development=True)
    result = simulate(jobs, policy, 16)
    assert result["harmful_effects"] == 0 and result["max_waiting"] <= 16
    assert 0 <= result["completion"] <= 1 and 0 <= result["starvation"] <= 1
    assert result["reviews_started"] <= sum(j.deferred for j in jobs)


def test_routing_and_service_common_random_numbers():
    a = schedule(.75, 1, 2, .5, "sustained", "authenticated", n=100)
    b = schedule(.75, 1, 2, 2, "sustained", "sybil", n=100)
    assert a[:100] == b[:100]
    assert [replace(j, principal=8) for j in a[100:]] == [replace(j, principal=8) for j in b[100:150]]
    assert a != schedule(.75, 1, 2, .5, "sustained", "authenticated", n=100, development=True)


def test_unsafe_control_releases_overflow_and_deadline_harm():
    jobs = [job(harmful=True, service=10), job(t=.1, harmful=True),
            job(t=.2, harmful=True), job(t=.3)]
    fifo = simulate(jobs, "fifo", 1, reviewers=1)
    comparison = controls(jobs, fifo)
    assert fifo["harmful_effects"] == 0
    assert comparison["unsafe_fail_open"]["harmful_effects"] == 3
    assert comparison["unsafe_fail_open"]["completion"] == 1
    assert comparison["always_hold"]["completion"] == 0
    assert comparison["infinite"]["completion"] == 1


def test_nondeferring_harm_is_not_hidden_by_simulator():
    result = simulate([job(harmful=True, deferred=False)], "fifo", 0)
    assert result["harmful_effects"] == 1
