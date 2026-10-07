"""CPU crash regression for jobs/full8/ at the probe-write/progress-commit boundary."""
import copy
import json

import pytest

from convert.pipeline import Results
from convert.progressive import run_progressive


class Crash(Exception):
    pass


@pytest.mark.parametrize("crash_layer", [3, 7])
def test_resume_discards_probe_written_before_its_progress_record(tmp_path, crash_layer):
    """A wall-clock-capped retry can produce new weights; its orphan probe must not survive."""
    path = str(tmp_path / "results.json")
    checkpoints, attempts = {}, {3: 0, 7: 0}
    gate = {"policy": "enforce", "max_step_delta_nll": .05,
            "max_cum_delta_nll": .15, "min_hidden_cos": .95}
    result = Results(path)
    result.set("probe_stock", {"weight": "stock"})
    crashed = False

    class Ops:
        def __init__(self, result):
            self.result, self.weights = result, {}

        def load(self, n):
            self.weights = copy.deepcopy(checkpoints[n])

        def snapshot(self, converted):
            return copy.deepcopy(self.weights)

        def train(self, k, i, trial):
            attempts[i] += 1
            self.weights[i] = attempts[i]  # retry differs, as a time-limited real training run can
            return {}

        def evaluate(self, trial):
            return {"nll_student": 1 + .01 * len(trial), "nll_teacher": 1.,
                    "delta_nll": .01 * len(trial), "hidden_cos": .99}

        def restore(self, snap, i):
            self.weights = snap

        def save(self, n):
            checkpoints[n] = copy.deepcopy(self.weights)

        def teacher_nll(self):
            return 1.

        def probe(self, converted):
            nonlocal crashed
            tag = f"conv{len(converted)}_kept{8 - len(converted)}"
            key = f"probe_{tag}"
            if key not in self.result:
                self.result.set(key, {"weights": copy.deepcopy(self.weights)})
                # Exactly after the probe's atomic JSON write, before progressive persists this step.
                if converted[-1] == crash_layer and not crashed:
                    crashed = True
                    raise Crash()
            return tag

    prog = {"steps": [], "converted": []}
    with pytest.raises(Crash):
        run_progressive([3, 7], prog, gate, Ops(result), persist=lambda p: result.set("progressive", p))
    disk = json.loads((tmp_path / "results.json").read_text())
    orphan = f"probe_conv{1 if crash_layer == 3 else 2}_kept{7 if crash_layer == 3 else 6}"
    assert orphan in disk  # sensitivity: the injected crash really left cached uncommitted measurements

    resumed = Results(path)  # production startup invalidates orphans before any probe-cache lookup
    assert orphan not in resumed
    assert resumed["probe_stock"] == {"weight": "stock"}
    if crash_layer == 7:
        assert "probe_conv1_kept7" in resumed  # keep a completed earlier curve point
    ops = Ops(resumed)
    prog = resumed["progressive"] if "progressive" in resumed else {"steps": [], "converted": []}
    run_progressive([3, 7], prog, gate, ops, persist=lambda p: resumed.set("progressive", p))
    final = resumed[f"probe_{prog['steps'][-1]['probe_tag']}"]
    assert final["weights"] == ops.weights == checkpoints[2]
    assert attempts[crash_layer] == 2
    assert orphan in resumed["resume_invalidated_probes"]


def test_results_keeps_completed_probe_cache_on_second_resume(tmp_path):
    path = str(tmp_path / "results.json")
    result = Results(path)
    result.set("probe_conv1_kept7", {"weights": {"3": 1}})
    result.set("progressive", {"steps": [{"accepted": True, "probe_tag": "conv1_kept7"}], "converted": [3]})
    resumed = Results(path)
    assert resumed["probe_conv1_kept7"] == {"weights": {"3": 1}}
    assert "resume_invalidated_probes" not in resumed
