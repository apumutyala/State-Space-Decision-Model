import pytest
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from keel.workflows.aggregate import cluster_ratio, summarize
from keel.workflows.runner import BENCHMARK_VERSION, ExecutionRuntime, standard_attack


def test_bootstrap_resamples_tasks_not_injection_rows():
    rows = [{"user_task": task, "x": value} for task, value in [("a", 0), ("b", 1)] for _ in range(30)]
    result = cluster_ratio(rows, "x", draws=1000)
    assert result["estimate"] == .5
    assert result["ci95"] == [0, 1]
    assert result["clusters"] == 2
    assert result["episodes"] == 60


def test_hold_rate_is_pooled_call_ratio_not_episode_mean():
    rows = [{"user_task": "a", "held": 1, "calls": 1}, {"user_task": "b", "held": 0, "calls": 9}]
    assert cluster_ratio(rows, "held", "calls")["estimate"] == .1
    assert cluster_ratio([{"user_task": "a", "held": 0, "calls": 0}], "held", "calls")["estimate"] == 0


def test_duplicate_episodes_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([{"episode_id": "same"}, {"episode_id": "same"}])


def test_actual_execution_latches_target_even_if_later_undone():
    state = {"value": "safe"}
    base = FunctionsRuntime()

    @base.register_function
    def overwrite(value: str) -> str:
        """Overwrite state.

        :param value: New value.
        """
        state["value"] = value
        return value

    rt = ExecutionRuntime(list(base.functions.values()), lambda env, trace: state["value"] == "unsafe")
    rt.run_function(None, "overwrite", {"value": "unsafe"})
    rt.run_function(None, "overwrite", {"value": "safe"})
    assert rt.target_succeeded
    assert len(rt.executed) == 2
    rt.run_function(None, "missing", {})
    assert len(rt.executed) == 2
    assert rt.dispatches[-1]["error"] is not None


@pytest.mark.parametrize("suite_name,injection_id", [("banking", "injection_task_0"), ("slack", "injection_task_1")])
def test_real_benchmark_attack_and_predicate_require_side_effect(suite_name, injection_id):
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    task = suite.get_user_task_by_id("user_task_0")
    injection = suite.get_injection_task_by_id(injection_id)
    attack = standard_attack(suite, object())
    replacements = attack.attack(task, injection)
    assert replacements
    assert all(injection.GOAL in text and "Qwen3.5-9B" in text for text in replacements.values())
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    pre = env.model_copy(deep=True)
    calls = injection.ground_truth(pre)
    assert not suite._check_injection_task_security(injection, [], pre, env, calls)
    runtime = ExecutionRuntime(suite.tools)
    for call in calls:
        _, error = runtime.run_function(env, call.function, call.args)
        assert error is None
    assert suite._check_injection_task_security(injection, [], pre, env, runtime.executed)
