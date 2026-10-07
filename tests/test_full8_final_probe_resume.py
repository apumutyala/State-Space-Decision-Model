"""CPU resume regression for jobs/full8/ final-model measurements after a deadline stop."""
import contextlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from convert import pipeline


@pytest.mark.parametrize("reject_last", [False, True])
def test_resume_final_measurements_follow_the_last_accepted_model(tmp_path, monkeypatch, reject_last):
    """Run the production pipeline through 7 steps, resume the 8th, and resume the completed run again.

    Only model/data/measurement costs are scalar fakes. Progressive gating, rollback, checkpoint save/load,
    Results persistence and both final-measurement cache decisions execute unchanged.
    """
    layers = [3, 7, 11, 15, 19, 23, 27, 31]
    state = {"trained": [], "mmlu": [], "throughput": [], "stage1": 0}

    class Model:
        def __init__(self):
            self.lm = torch.nn.Linear(1, 1)
            self.cfg = None  # FLOP reporting is optional
            self.attn_layers = layers
            self.swaps = {i: SimpleNamespace(gdn=None) for i in layers}
            self.current_modes = {}

        @contextlib.contextmanager
        def modes(self, modes):
            old, self.current_modes = self.current_modes, modes
            try:
                yield
            finally:
                self.current_modes = old

    monkeypatch.delenv("PHASE_HOOK", raising=False)
    monkeypatch.delenv("DEADLINE_MINUTES", raising=False)
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: (Model(), None, "synthetic-scalar"))
    monkeypatch.setattr(pipeline, "env_info", lambda: {"synthetic": True})
    monkeypatch.setattr(pipeline, "init_gdn", lambda *a: torch.nn.Linear(1, 1))
    monkeypatch.setattr(pipeline, "attach_gdn", lambda m, i, g: setattr(m.swaps[i], "gdn", g))
    monkeypatch.setattr(pipeline.data, "build", lambda tok, s, **k: (s, {"split": s}))
    monkeypatch.setattr(pipeline.data, "load_tokens", lambda p: np.arange(100, dtype=np.uint32))
    monkeypatch.setattr(pipeline.data, "Batcher", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.align, "teacher_hidden", lambda *a: [])
    monkeypatch.setattr(pipeline.align, "Guard", lambda *a, **k: SimpleNamespace(rate=1., peak_gb=lambda: 0.))

    def stage1(*a, **k):
        state["stage1"] += 1
        return [{}, {}], 1, 1.

    def stage2(model, train_layers, modes, *a, **k):
        state["trained"].append(list(modes)[-1])
        with torch.no_grad():
            for i in train_layers:
                model.swaps[i].gdn.weight.add_(1)
        return [{}, {}], 1, 1.

    def evaluate(model, windows, modes, **k):
        d = .001 * len(modes)
        return dict(nll_teacher=2., nll_student=2. + d, delta_nll=d,
                    hidden_cos=.9 if reject_last and 31 in modes else .99,
                    kl=sum(layers.index(i) + 1 for i in modes))

    def mmlu(model, tok, **k):
        state["mmlu"].append(dict(model.current_modes))
        return dict(acc=1., n=14042, correct=[1] * 14042, items_sha256="scalar-fixture", seconds=0.)

    def throughput(model, modes, *a):
        state["throughput"].append(dict(modes))
        return {"4": {"tok_s": 1., "measured_modes": dict(modes)}}

    monkeypatch.setattr(pipeline.align, "stage1_align", stage1)
    monkeypatch.setattr(pipeline.align, "stage2_kl", stage2)
    monkeypatch.setattr(pipeline.align, "evaluate", evaluate)
    monkeypatch.setattr(pipeline.probes, "run_all", lambda *a: {
        "ppl": {"ctx1024": {"ppl": 1.}}, "niah": {"mean_acc": 1.},
        "mqar": {"mean_acc": 1.}, "mmlu": {"acc": 1.}})
    monkeypatch.setattr(pipeline.probes, "mmlu_full", mmlu)
    monkeypatch.setattr(pipeline, "measure_throughput", throughput)
    cfg = dict(seed=0, model_dir=None, ckpt_dir=str(tmp_path / "ck"), seq_len=4, batch=1,
               data={}, controls={"swa_windows": [], "zero": False}, init_cmp={"steps": 0},
               init="random", calibrate_out_scale=False, order="kl", stage1={"steps": 1, "lr": .1},
               stage2={"steps": 1, "lr": .1},
               progressive={"max_layers": 8, "probe_after_deadline": True},
               gate=dict(n_windows=2, max_step_delta_nll=.05, max_cum_delta_nll=.15,
                         min_hidden_cos=.95, policy="enforce"),
               mmlu_full={"batch_tokens": 16384}, throughput={"lengths": [4], "call_latency": False})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "ck").mkdir()
    out = tmp_path / "pipeline"
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config), "--out", str(out)])

    real_progressive = pipeline.run_progressive

    def stop_after_seven(*a, **k):
        args = list(a)
        args[5] = lambda: len(state["trained"]) >= 7  # production progressive deadline boundary
        return real_progressive(*args, **k)

    monkeypatch.setattr(pipeline, "run_progressive", stop_after_seven)
    pipeline._main()
    first = json.loads((out / "results.json").read_text())
    assert len(first["progressive"]["steps"]) == 7
    assert len(state["mmlu"]) == len(state["throughput"]) == 2
    assert sorted(map(int, first["mmlu_full"]["final"]["modes"])) == layers[:7]
    assert "converted_7" in first["throughput"]

    monkeypatch.setattr(pipeline, "run_progressive", real_progressive)
    pipeline._main()
    final = json.loads((out / "results.json").read_text())
    expected_layers = layers[:7] if reject_last else layers
    expected_calls = 2 if reject_last else 4
    assert len(final["progressive"]["steps"]) == 8
    assert final["progressive"]["converted"] == expected_layers
    assert sorted(map(int, final["mmlu_full"]["final"]["modes"])) == expected_layers
    assert set(final["throughput"]) == {"stock", f"converted_{len(expected_layers)}"}
    assert final["throughput"][f"converted_{len(expected_layers)}"]["4"]["measured_modes"] == {
        str(i): "student" for i in expected_layers}
    assert len(state["mmlu"]) == len(state["throughput"]) == expected_calls
    assert state["trained"] == layers and state["stage1"] == 1

    pipeline._main()  # completed checkpoint: retain matching measurements without repeating either arm
    assert len(state["mmlu"]) == len(state["throughput"]) == expected_calls
    assert state["trained"] == layers and state["stage1"] == 1
