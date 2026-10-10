"""jobs/m2_first_cell mode d19_jevbench (eval_only "jevbench"): params / runner consistency, the per-item jevkit loop
(serial, usage, resume) over a decide.Decider, and the paired report (CPU; oracle backends, never weights)."""
from __future__ import annotations

import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "jobs/m2_first_cell/job_params_d19_jevbench.json"
RUNNER = ROOT / "jobs/m2_first_cell/first_cell_job.sh"

from jobs.m2_first_cell import jev_eval as JE  # noqa: E402


def _items():
    from jevkit.items import default_data_dir, load_public

    if not default_data_dir().exists():
        pytest.skip("pinned public JevBench items missing (python -m jevkit fetch)")
    return load_public()


def test_params_runner_and_loader_agree():
    from jobs.common import lint
    from jobs.m2_first_cell.checkpoint_loader import MEMBER_RE, checkpoint_block

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    ck = checkpoint_block(p)
    assert MEMBER_RE.fullmatch(ck["member"]) and ck["member"].endswith("trainable-step0002800.safetensors")
    assert p["eval_only"] == "jevbench" and p["mode"] == "d19_jevbench" and "stage" not in p
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    d19 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_d19_v3w_v3b_joint.json").read_text(encoding="utf-8"))
    assert cfg == d19 and cfg["template"] == "jrt-rwkv-g1-v3w" and cfg["head"] == "joint_record"  # train == serve
    assert p["max_duration_minutes"] * 0.063 <= 15.0
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    src = RUNNER.read_text(encoding="utf-8")
    for ph in p["phase_caps_minutes"]:
        assert f"cap {ph}" in src or f"cap $PH" in src or ph in ("setup",)
    assert '"jevbench"' in src and "jobs.m2_first_cell.jev_eval score" in src
    bad, waived = lint.failures(str(PARAMS.relative_to(ROOT)).replace("\\", "/"),
                                lint.lint_params(str(PARAMS.relative_to(ROOT)).replace("\\", "/"), ROOT))
    assert not bad and not waived, bad
    dests = {d["dest"] for d in p["context"]["data"]}
    assert "checkpoint/cell.json" in dests and sum(d.startswith("jevbench/") for d in dests) == 3
    assert p["price"]["kind"] == "size_class_estimate" and p["price"]["usd_per_m_input"] == 0.05


def test_runner_shell_syntax():
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(RUNNER)], capture_output=True).returncode == 0


def _oracle_decider(items, margin):
    from decide import Decider
    from decide.backends.oracle import OracleBackend

    gold = {}

    class PerItem:
        """An oracle whose gold follows the current request (the item's expected label)."""
        name = "oracle-per-item"

        def __init__(self):
            self.b = None

        def revision(self):
            return {"revision_id": f"oracle-m{margin}", "backend": "oracle"}

        def plan(self, request):
            return self._b(request).plan(request)

        def score(self, request, check):
            return self._b(request).score(request, check)

        def _b(self, request):
            it = gold[json.dumps(request["state"], sort_keys=True, default=str) + json.dumps(request["questions"]["q"], sort_keys=True)]
            lab = {"q": ("true" if it.expected == "yes" else "false") if it.type == "noul" else str(it.expected)}
            return OracleBackend(labels=lab, margin=margin)

    for it in items:
        gold[json.dumps(it.state, sort_keys=True, default=str) + json.dumps(it.question, sort_keys=True)] = it
    return Decider(PerItem())


def test_run_items_report_and_paired_deltas(tmp_path, monkeypatch):
    items = _items()
    out = tmp_path / "jevbench"
    for arm, margin in (("child", 6.0), ("parent", 0.5)):
        res = JE.run_items(_oracle_decider(items, margin), items, out / arm, warmup=1)
        assert res["items"] == len(items)
        (out / arm / "identity.json").write_text(json.dumps({"arm": arm}), encoding="utf-8")
        recs = [json.loads(x) for x in (out / arm / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(recs) == len(items) and all(r["usage"]["output_tokens"] == 1 for r in recs)
    again = JE.run_items(_oracle_decider(items, 6.0), items, out / "child", warmup=0)  # resume: nothing re-scored
    assert again["items"] == 0 and again["resumed_from"] == len(items)
    a = Namespace(params=str(PARAMS), out_dir=str(out), elapsed_min=1)
    d = JE.report(a)
    assert JE.valid_result(d)
    pc = d["child_minus_parent"]
    assert pc["composite_A"]["ci95"] is not None and pc["intelligence"]["point"] > 0  # sharper oracle = better
    assert d["arms"]["child"]["cost"]["price"]["usd_per_m_input"] == 0.05
    assert not JE.valid_result({**d, "arms": {**d["arms"], "parent": {**d["arms"]["parent"], "complete": False}}})


def test_joint_head_v3w_serve_path_answers_every_public_item_type(tmp_path):
    """The child's serving path (DecisionModelBackend, joint_record head, jrt-rwkv-g1-v3w / g1v2: D19's native readout)
    answers JevBench requests of every type, ok, on a tiny CPU backbone."""
    torch = pytest.importorskip("torch")
    from decide import Decider
    from decide.prompt_g1v3 import G1V3W_ID
    from heads import ByteEncoder, DecisionModel
    from heads.backbone import tiny_backbone
    from heads.decide_backend import DecisionModelBackend

    items = _items()
    pick = [next(it for it in items if it.type == t) for t in ("choice", "noul", "score")] + items[-5:]
    m = DecisionModel(tiny_backbone(seed=0, hidden=16, fork_mode="cache"), "joint_record", seed=0).eval()
    with torch.no_grad():
        m.head.w.normal_()
        m.head.lex_scale.fill_(0.7)
    dec = Decider(DecisionModelBackend(m, ByteEncoder(), template=G1V3W_ID, layout="g1v2"))
    res = JE.run_items(dec, pick, tmp_path / "child", warmup=0)
    assert res["status"] == {"ok": len(pick)}, res
    recs = [json.loads(x) for x in (tmp_path / "child" / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["answer"]["type"] == it.type and r["usage"]["input_tokens"] > 0 for r, it in zip(recs, pick))
