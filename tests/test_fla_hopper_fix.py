"""CPU static tests for the fla #640 (Hopper + Triton < 3.7.1) image fix. No GPU, no Docker.

SF job-jbw22 (first H100 qualification) stopped in test_gdn's backward: fla 0.5.2 refuses the gated chunk_bwd_dqkwg
on Hopper with 3.4.0 <= Triton < 3.7.1. Fix: every pytorch/pytorch-based job image moves to torch 2.13.0 (pins
triton 3.7.1), pins fla's Triton backward (FLA_TILELANG=0), asserts the versions at build time, and every runner
calls jobs/common/fla_guard.py at setup (exit 5) before any GPU phase.

    python -m pytest -q tests/test_fla_hopper_fix.py
"""
from __future__ import annotations

import json
import re
from importlib.util import find_spec
from pathlib import Path

import pytest

from jobs.common import fla_guard as G

ROOT = Path(__file__).resolve().parents[1]
NEW_BASE = ("pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel"
            "@sha256:09c0260351b12b617408ff5711e08d88ccdbb9a21f7f6d7de67e39039b022ae1")
OLD_DIGEST = "a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"
# No exemptions. clef_baseline used to be exempt as "inference-only", which was false: its S1 --grad-probe and job B
# (GCG-lite) call backward() through fla's gated GDN kernels (PR #41 review). It is now on the same base, with
# transformers kept at the model card's 5.10.2 (declared deviation: torch 2.11 -> 2.13; see its Dockerfile/JOB_SPEC).
EXEMPT: set[str] = set()
# job -> runner; m2_pilot / m2_parent_arm build from the rwkv7 Dockerfile (their build_context.sh copies it).
RUNNERS = {"qual": "jobs/qual/qual_job.sh", "probe9b": "jobs/probe9b/probe_job.sh", "rwkv7": "jobs/rwkv7/rwkv7_job.sh",
           "m2_pilot": "jobs/m2_pilot/m2_job.sh", "m2_parent_arm": "jobs/m2_parent_arm/parent_arm_job.sh"}
GUARD_CALL = re.compile(r'^python -I /opt/ssdm/m1/jobs/common/fla_guard\.py --out "\$WORK/out/fla_guard\.json".*'
                        r'\|\| fail 5 ', re.M)
DOCKERFILES = sorted(p for p in (ROOT / "jobs").glob("*/Dockerfile"))


def _job_dockerfile(job: str) -> str:
    """The Dockerfile a job's image is built from (derived jobs copy another job's Dockerfile)."""
    own = ROOT / "jobs" / job / "Dockerfile"
    if own.exists():
        return own.read_text(encoding="utf-8")
    bc = (ROOT / "jobs" / job / "build_context.sh").read_text(encoding="utf-8")
    src = re.search(r'cp "\$OUTD/code/m1/(jobs/[a-z0-9_]+/Dockerfile)" "\$OUTD/Dockerfile"', bc)
    assert src, f"{job}: cannot find which Dockerfile build_context.sh uses"
    return (ROOT / src.group(1)).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- Dockerfiles carry the fix
def test_every_dockerfile_is_covered_or_explicitly_exempt():
    jobs = {p.parent.name for p in DOCKERFILES}
    assert EXEMPT <= jobs and {"qual", "probe9b", "rwkv7", "clef_baseline"} <= jobs
    assert jobs - EXEMPT == {"qual", "probe9b", "rwkv7", "clef_baseline"},         "new Dockerfile: add it to the fix (or justify an exemption)"


@pytest.mark.parametrize("job", sorted(RUNNERS) + ["clef_baseline"])
def test_job_image_carries_fix(job):
    d = _job_dockerfile(job)
    froms = re.findall(r"^FROM (\S+)", d, re.M)
    assert froms == [NEW_BASE], froms
    env = " ".join(re.findall(r"^ENV (.*)$", d, re.M))
    assert "FLA_TILELANG=0" in env and "PIP_BREAK_SYSTEM_PACKAGES=1" in env
    pip = re.search(r'^RUN pip install --no-cache-dir (.*?)(?<!\\)$', d, re.M | re.S).group(1)
    assert '"torch==2.13.0"' in pip and '"triton==3.7.1"' in pip
    assert re.search(r"^RUN python -c .*import torch, triton.*'2\.13\.0'.*'12\.6'.*'3\.7\.1'", d, re.M), \
        "build-time torch/cuda/triton assertion missing"
    # the build-time assertion must come after the pip line that could change torch/triton
    assert d.index("import torch, triton") > d.index('"triton==3.7.1"')
    assert "tilelang" not in pip, "the fix is the Triton path; installing tilelang would switch the backward kernel"
    if "flash-linear-attention" in d:
        assert '"flash-linear-attention==0.5.2"' in d and '"fla-core==0.5.2"' in d


def test_old_base_digest_is_gone_from_jobs():
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / "jobs").rglob("*")
            if p.is_file() and p.suffix in ("", ".md", ".sh", ".json", ".py") and OLD_DIGEST in p.read_text("utf-8", "ignore")]
    assert hits == []


def test_qual_spec_documents_new_image():
    spec = (ROOT / "jobs/qual/JOB_SPEC.md").read_text(encoding="utf-8")
    assert NEW_BASE in spec and "triton 3.7.1" in spec and "#640" in spec and "fla_guard.py" in spec


# ----------------------------------------------------------------------------- runners call the guard early
@pytest.mark.parametrize("job,runner", sorted(RUNNERS.items()))
def test_runner_calls_guard_before_gpu_work(job, runner):
    s = (ROOT / runner).read_text(encoding="utf-8")
    m = GUARD_CALL.search(s)
    assert m, f"{runner}: no fla_guard call failing with exit 5"
    setup = s.index('say "setup')
    assert setup < m.start()
    # every heavy step (weights fetch / snapshot download / tests / training / phases) comes after the guard
    heavy = [h.start() for h in re.finditer(r"sf_fetch\.py|snapshot_download|pytest|-m tests\.|-m train\.(?!cohort)|"
                                            r"^timeout |^export_phase (?!\(\))", s, re.M)]
    assert heavy and min(heavy) > m.start(), f"{runner}: GPU/heavy work before the guard"
    assert "\\n  ||" not in s, f"{runner}: literal \\n in a shell continuation"


CLEF_RUNNER = "jobs/clef_baseline/run_job.sh"
CLEF_GUARD = re.compile(r'^python -I /app/jobs/common/fla_guard\.py --out "\$OUT/env/fla_guard\.json".*\|\| fail 5 ', re.M)


def test_clef_runner_calls_guard_before_download_and_gpu_phases():
    """Clef's runner has its own layout (/app, $OUT); the guard must precede the weights download and every phase
    that touches the GPU (S1 probes incl. --grad-probe, the Index runs, safety_eval and GCG)."""
    s = (ROOT / CLEF_RUNNER).read_text(encoding="utf-8")
    m = CLEF_GUARD.search(s)
    assert m, f"{CLEF_RUNNER}: no fla_guard call failing with exit 5"
    heavy = [h.start() for h in re.finditer(r"^[^#\n]*(snapshot_download|baselines\.clef_probes|decision_index run|"
                                            r"baselines\.run_clef_safety)", s, re.M)]
    assert len(heavy) >= 5 and min(heavy) > m.start(), "Clef GPU/heavy work before the guard"
    assert "--grad-probe" in s  # the backward the guard exists for is still there (else re-derive the need)
    header = "".join(ln for ln in s.splitlines(keepends=True)[:12] if ln.startswith("#"))
    assert "5 setup: fla #640 guard" in header  # exit 5 documented in the header


def test_clef_image_keeps_card_transformers_and_no_tilelang():
    d = (ROOT / "jobs/clef_baseline/Dockerfile").read_text(encoding="utf-8")
    assert '"transformers==5.10.2"' in d  # the model card's version: Qwen3.5 model code + processor unchanged
    assert "tilelang" not in re.sub(r"^#.*$", "", d, flags=re.M)
    assert re.search(r'"causal-conv1d==\d+\.\d+\.\d+"', d), "causal-conv1d must be pinned"
    assert "torch==2.11" not in d and "whl/cu128" not in d
    # PR #43 review: ninja for the causal-conv1d source build; torchvision matched to cu126; the build-time assert also
    # pins transformers 5.10.2 + torchvision and imports AutoProcessor / Qwen3_5ForConditionalGeneration
    assert '"ninja==' in d and d.index('"ninja==') < d.index('"causal-conv1d==')
    assert re.search(r'"torchvision==0\.28\.0" --index-url https://download\.pytorch\.org/whl/cu126', d)
    assert re.search(r"^RUN python -c .*from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration.*'5\.10\.2'"
                     r".*'0\.28\.0'", d, re.M)
    assert d.index("from transformers import AutoProcessor") > d.index('"torchvision==0.28.0"')


def test_clef_spec_documents_fix():
    spec = (ROOT / "jobs/clef_baseline/JOB_SPEC.md").read_text(encoding="utf-8")
    assert NEW_BASE in spec and "#640" in spec and "fla_guard.py" in spec and "kernel_path" in spec
    assert "torch 2.11" in spec  # the deviation from the model card is declared, not hidden
    assert "H200" in spec and "H100" in spec and '"build_timeout_minutes": 90' in spec  # latency + build deviations
    assert '"result.s1_ok == true"' in spec and "result.parity_ok == true" not in spec  # job B gated on kernel_path


def test_no_literal_backslash_n_continuations_in_shell():
    bad = [str(p.relative_to(ROOT)) for p in (ROOT / "jobs").rglob("*.sh") if re.search(r"\\n\s+\|\|", p.read_text("utf-8"))]
    assert bad == []


# ----------------------------------------------------------------------------- the guard's rule (pure)
@pytest.mark.parametrize("v,t", [("3.7.1", (3, 7, 1)), ("3.7.1+git1a2b", (3, 7, 1)), ("3.8.0rc1", (3, 8, 0)),
                                 ("3.4", (3, 4, 0)), ("garbage", (0,)), ("", (0,))])
def test_vtuple(v, t):
    assert G.vtuple(v) == t


@pytest.mark.parametrize("hopper,triton,tl,buggy", [
    (True, "3.4.0", False, True), (True, "3.6.0", False, True), (True, "3.7.0", False, True),
    (True, "3.7.1", False, False), (True, "3.8.0", False, False), (True, "3.3.1", False, False),
    (True, "3.6.0", True, False), (False, "3.6.0", False, False)])
def test_buggy_rule_matches_fla_0_5_2(hopper, triton, tl, buggy):
    assert G.buggy_gated_bwd(hopper, triton, tl) is buggy


def test_policy():
    assert G.policy_problems(True, "3.7.1", False, False) == []
    assert G.policy_problems(False, "3.8.0", False, False) == []
    assert G.policy_problems(True, "3.4.0", False, True)          # job-jbw22's stack
    assert G.policy_problems(False, "3.4.0", False, False)        # image policy: Triton >= 3.7.1 everywhere
    assert G.policy_problems(True, "3.7.1", True, False)          # TileLang selected -> not the measured path
    assert any("disagrees" in p for p in G.policy_problems(True, "3.7.1", False, True))


def test_guard_without_fla_is_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "find_spec", lambda name: None)
    out = tmp_path / "g.json"
    assert G.main(["--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["fla_installed"] is False and rep["ok"] is True and rep["problems"] == []


def test_guard_refuses_with_exit_5(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "check", lambda smoke=True: ({"fla_installed": True}, ["triton 3.4.0 < 3.7.1"]))
    out = tmp_path / "g.json"
    assert G.main(["--out", str(out)]) == 5
    assert json.loads(out.read_text())["ok"] is False


def test_guard_crash_is_a_refusal(monkeypatch):
    def boom(smoke=True):
        raise ImportError("no fla.utils")
    monkeypatch.setattr(G, "check", boom)
    assert G.main([]) == 5


@pytest.mark.skipif(find_spec("fla") is None, reason="fla not installed here")
def test_rule_mirrors_installed_fla_source():
    """When fla is importable locally, its #640 condition must be the one the guard mirrors (read as text: no triton)."""
    base = Path(list(find_spec("fla").submodule_search_locations)[0])
    src = (base / "ops/common/chunk_o.py").read_text(encoding="utf-8")
    assert "if g is not None and IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:" in src
    compat = (base / "utils/_compat.py").read_text(encoding="utf-8")
    assert 'parse("3.7.1")' in compat and 'parse("3.4.0")' in compat
    tl = (base / "ops/common/backends/tilelang/__init__.py").read_text(encoding="utf-8")
    assert 'env_var = "FLA_TILELANG"' in tl and 'return val != "0"' in tl
