"""ENFORCING launch lint (LESSONS.md, jobs/LAUNCH_CHECKLIST.md): every jobs/<job>/job_params*.json passes every check in
jobs/common/lint.py or carries a dated waiver (legacy files only; stale waivers fail); repo-wide LF / CR /
backslash-n / control-character checks; CHAMPION.md + CANDIDATES.md structure. CPU, no network, a few seconds.

    python -X utf8 -m pytest -q tests/test_jobs_lint.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jobs.common import lint

ROOT = Path(__file__).resolve().parents[1]
PARAMS = lint.params_files(ROOT)
COSTS = lint.load_costs(ROOT)
TESTS = lint._test_sources(ROOT)
NL = chr(10)
GIT = shutil.which("git") is not None and (ROOT / ".git").exists()


def test_discovers_every_params_file():
    assert len(PARAMS) >= 10 and set(lint.LEGACY_PARAMS) <= set(PARAMS)
    for rel in PARAMS:
        assert rel.startswith("jobs/") and Path(rel).name.startswith("job_params")


@pytest.mark.parametrize("rel", PARAMS)
def test_params_file_passes_lint_or_is_waived(rel):
    bad, _ = lint.failures(rel, lint.lint_params(rel, ROOT, COSTS, TESTS))
    assert not bad, f"{rel} fails launch lint (fix it; only legacy files may be waived): {bad}"


def test_waivers_are_legacy_dated_and_not_stale():
    for rel, w in lint.WAIVERS.items():
        assert rel in lint.LEGACY_PARAMS, f"waiver for a non-legacy params file {rel}: new jobs must comply"
        res = lint.lint_params(rel, ROOT, COSTS, TESTS)
        for check, reason in w.items():
            assert check in lint.CHECKS
            assert re.match(r"20\d\d-\d\d-\d\d: \S", reason), f"undated waiver {rel}:{check}"
            assert not res[check][0], f"stale waiver {rel}:{check} now passes: delete it from lint.WAIVERS"


def test_new_job_cannot_be_waived():
    rel = "jobs/new_job/job_params.json"
    assert rel not in lint.LEGACY_PARAMS
    lint.WAIVERS[rel] = {"build_timeout": "2026-10-06: attempt"}
    try:
        bad, waived = lint.failures(rel, {"build_timeout": (False, "18")})
        assert bad == {"build_timeout": "18"} and not waived
    finally:
        del lint.WAIVERS[rel]


def test_build_timeout_and_schedule_rules():
    ok = {"build_timeout_minutes": 60, "max_duration_minutes": 10, "reserve_minutes": 1,
          "phase_caps_minutes": {"a": 4, "b": 5}}
    assert lint.chk_build_timeout(ok)[0] and lint.chk_schedule(ok)[0]
    assert not lint.chk_build_timeout({**ok, "build_timeout_minutes": 18})[0]  # job-4idg4 era publish timeouts
    assert not lint.chk_schedule({**ok, "max_duration_minutes": 11})[0]
    q = {"max_duration_minutes": 180, "reserve_minutes": 8,
         "phase_caps_minutes": {"setup": 15, "A_per_test": 3, "A_tests": 7, "B": 20, "C_weight_wait": 25, "C": 80}}
    assert lint.schedule_total(q) == 169


def test_backslash_n_regex_fires_on_known_bad_and_not_on_good():
    bs = chr(92)  # built from chr() so no heredoc / string-escape layer can collapse it (L-11)
    bad = f"python -m x {bs}n    --flag 1"
    good = f"python -m x {bs}\n    --flag 1"
    assert bad.count(bs) == 1 and "\n" not in bad
    assert lint.BACKSLASH_N.search(bad)
    assert not lint.BACKSLASH_N.search(good)
    assert lint.BACKSLASH_N.pattern == "[" + bs + bs + "]n[ \\t]+--"


def test_no_backslash_n_continuation_in_any_job_shell():
    assert lint.backslash_n_hits(ROOT) == []


def test_runner_execution_detector():
    runner = "jobs/x/x_job.sh"
    direct = 'import subprocess\nsubprocess.run(["bash", str(ROOT / "jobs/x/x_job.sh")])\n'
    split = ('import subprocess, shutil\nBASH = shutil.which("bash")\n'
             'subprocess.run([BASH, str(ROOT / "jobs" / "x" / "x_job.sh")])\n')
    via_name = 'import subprocess\np = ROOT / "jobs/x/x_job.sh"\nsubprocess.run(["bash", str(p)], check=True)\n'
    snippet = ('import subprocess\nlines = open("jobs/x/x_job.sh").read().splitlines()[:3]\n'
               'subprocess.run(["bash", "-c", "echo hi"])\n')
    assert lint.executes_runner(direct, runner) and lint.executes_runner(split, runner)
    assert lint.executes_runner(via_name, runner)
    assert not lint.executes_runner(snippet, runner)  # reading the runner's text is not executing it
    assert not lint.executes_runner(f"# {runner}\n", runner)


@pytest.mark.parametrize("command", [
    '["bash", "-n", runner]', '["bash", "-en", runner]', '["bash", "--noexec", runner]',
    '["bash", "-o", "noexec", runner]', '["bash", "--help", runner]',
    '["bash", "-c", "cat " + runner]', '["bash", "-c", "exec bash -n " + runner]',
    '["bash", "-c", "exec cat " + runner]', '["cat", runner]', '["head", runner]',
    '["grep", "flag", runner]', '["echo", runner]', '["bash", "other.sh", runner]',
])
def test_runner_execution_detector_rejects_read_only_commands(command):
    runner = "jobs/x/x_job.sh"
    src = NL.join(['import subprocess', f'runner = "{runner}"', f'cmd = {command}',
                   'subprocess.run(cmd, check=True)'])
    assert not lint.executes_runner(src, runner)


@pytest.mark.parametrize("command", [
    '["bash", "-eu", "--", runner]', '["bash", "-o", "pipefail", runner]', '[runner]',
    '["bash", "-c", "export X=1; exec bash jobs/x/x_job.sh"]',
])
def test_runner_execution_detector_accepts_script_arguments(command):
    runner = "jobs/x/x_job.sh"
    src = NL.join(['import subprocess as sp', f'runner = "{runner}"', f'cmd = {command}',
                   'sp.run(args=cmd, check=True)'])
    assert lint.executes_runner(src, runner)


def test_runner_execution_detector_rejects_non_shell_bindings():
    runner = "jobs/x/x_job.sh"
    src = NL.join(['import subprocess', 'BASH = "cat"',
                   f'subprocess.run([BASH, "{runner}"], check=True)'])
    assert not lint.executes_runner(src, runner)


@pytest.mark.parametrize("body", [
    'subprocess.run(["bash", runner], executable="cat")',
    'subprocess.run(["bash", runner], -1, "cat")',
    'subprocess.run(["bash", runner], **options)',
    'runner = "other.sh"\nsubprocess.run(["bash", runner])',
    'cmd = ["bash", runner]\ncmd[0] = "cat"\nsubprocess.run(cmd)',
    'cmd = ["bash", runner]\ncmd.insert(1, "-n")\nsubprocess.run(cmd)',
    'cmd = ["bash", runner]\ncmd += ["-n"]\nsubprocess.run(cmd)',
    'cmd = ["bash", runner]\nalias = cmd\nalias[0] = "cat"\nsubprocess.run(cmd)',
    'cmd = ["bash", runner]\nchange(cmd)\nsubprocess.run(cmd)',
    'cmd = ["bash", runner]\ndef change():\n    cmd[0] = "cat"\nchange()\nsubprocess.run(cmd)',
    'subprocess.run(cmd)\ncmd = ["bash", runner]',
    'def helper():\n    runner = "jobs/x/x_job.sh"\nrunner = "other.sh"\nsubprocess.run(["bash", runner])',
    'def test_shadow(runner):\n    subprocess.run(["bash", runner])',
    'def test_shadow():\n    runner = "other.sh"\n    subprocess.run(["bash", runner])',
    'def test_shadow():\n    subprocess.run(["bash", runner])\n    runner = "other.sh"',
    'def test_shadow(BASH):\n    subprocess.run([BASH, runner])',
    'def test_unreachable():\n    return\n    subprocess.run(["bash", runner])',
])
def test_runner_execution_detector_rejects_overrides_mutation_and_shadowing(body):
    src = NL.join(['import subprocess', 'runner = "jobs/x/x_job.sh"', body])
    assert not lint.executes_runner(src, "jobs/x/x_job.sh")


def test_runner_execution_detector_preserves_nested_literal_invocation():
    src = NL.join(['import subprocess', 'def test_outer():', '    def run_part():',
                   '        cmd = ["bash", "jobs/x/x_job.sh"]', '        subprocess.run(cmd)',
                   '    run_part()'])
    assert lint.executes_runner(src, "jobs/x/x_job.sh")


def test_rule_constants_are_literal_module_level():
    src = "A_MIN = 0.5\nBB = (1, 2)\nlower = 3\nC = f()\ndef g():\n    D = 1\n"
    assert lint.rule_constants(src) == ["A_MIN", "BB"]


@pytest.mark.parametrize("assertion", [
    "assert R.MIN_GAIN",  # bare truthiness does not freeze the margin
    "assert R.MIN_GAIN == 0.5",  # MAX_LOSS is unpinned
    "assert (R.MIN_GAIN, R.MAX_LOSS) == (0.6, 0.1)",  # wrong literal
    "assert R.MIN_GAIN == R.MIN_GAIN and R.MAX_LOSS == R.MAX_LOSS",
    "assert R.MIN_GAIN == 0.5 or R.MAX_LOSS == 0.1",  # either clause may pass alone
    "# assert R.MIN_GAIN == 0.5 and R.MAX_LOSS == 0.1",
    "assert R.MIN_GAIN == 0.5 and R.MAX_LOSS == 0.1",  # unrelated import
])
def test_decision_rule_rejects_unpinned_constants(tmp_path, assertion):
    (tmp_path / "rule.py").write_text("MIN_GAIN = 0.5" + NL + "MAX_LOSS = 0.1" + NL, encoding="utf-8")
    module = "other" if assertion == "assert R.MIN_GAIN == 0.5 and R.MAX_LOSS == 0.1" else "rule"
    src = NL.join([f"import {module} as R", "def test_pins():", f"    {assertion}", ""])
    ok, detail = lint.chk_decision_rule({"decision_rule": "rule.py"}, root=tmp_path, job="new",
                                        tests={"tests/test_rule.py": src})
    assert not ok, detail
    assert "missing" in detail


@pytest.mark.parametrize("source", [
    "import jobs.new.rule as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5 and 0.1 == R.MAX_LOSS\n",
    "from jobs.new import rule as R\ndef test_pins():\n    assert (R.MIN_GAIN, R.MAX_LOSS) == (0.5, 0.1)\n",
    "from jobs.new.rule import MIN_GAIN as G, MAX_LOSS\ndef test_pins():\n    assert G == 0.5\n    assert MAX_LOSS == 0.1\n",
    "import jobs.new.rule\ndef test_pins():\n    assert jobs.new.rule.MIN_GAIN == 0.5\n    assert jobs.new.rule.MAX_LOSS == 0.1\n",
])
def test_decision_rule_accepts_direct_and_tuple_equality_pins(tmp_path, source):
    (tmp_path / "jobs/new").mkdir(parents=True)
    (tmp_path / "jobs/new/rule.py").write_text("MIN_GAIN = 0.5" + NL + "MAX_LOSS = 0.1" + NL, encoding="utf-8")
    ok, detail = lint.chk_decision_rule({"decision_rule": "jobs/new/rule.py"}, root=tmp_path, job="new",
                                        tests={"tests/test_rule.py": source})
    assert ok, detail


def test_rule_pin_literals_preserve_container_types():
    consts = {"FLAGS": {"ok": True}, "LAYERS": (1, 2), "OPTIONAL": None}
    src = NL.join(["import rule as R", "def test_pins():", "    assert R.FLAGS == {'ok': 1}",
                   "    assert R.LAYERS == [1, 2]", "    assert R.OPTIONAL == None"])
    assert lint.pinned_rule_constants(src, "rule", consts) == {"OPTIONAL"}


@pytest.mark.parametrize("source", [
    "import rule as R\nimport other as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins():\n    import other as R\n    assert R.MIN_GAIN == 0.5\n",
    "from rule import MIN_GAIN\nfrom other import MIN_GAIN\ndef test_pins():\n    assert MIN_GAIN == 0.5\n",
    "import rule as R\nR = other\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins():\n    R = other\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins(R):\n    assert R.MIN_GAIN == 0.5\n",  # fixture shadows import
    "import other as R\ndef helper():\n    import rule as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n",
    "def helper():\n    import rule as R\n    assert R.MIN_GAIN == 0.5\ndef test_pins():\n    pass\n",
    "import rule as R\ndef test_pins():\n    def helper():\n        assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n    R = other\n",  # local throughout
    "import rule as R\ndef test_pins():\n    R.MIN_GAIN = 0.5\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins():\n    if condition:\n        import other as R\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_pins():\n    return\n    assert R.MIN_GAIN == 0.5\n",
])
def test_rule_pin_bindings_respect_test_scope_and_shadowing(source):
    assert lint.pinned_rule_constants(source, "rule", {"MIN_GAIN": 0.5}) == set()


@pytest.mark.parametrize("source", [
    "import other as R\ndef test_pins():\n    import rule as R\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef helper():\n    import other as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\ndef test_other():\n    import other as R\ndef test_pins():\n    assert R.MIN_GAIN == 0.5\n",
    "import rule as R\nclass TestRule:\n    def test_pins(self):\n        assert R.MIN_GAIN == 0.5\n",
])
def test_rule_pin_bindings_keep_module_and_test_local_imports_separate(source):
    assert lint.pinned_rule_constants(source, "rule", {"MIN_GAIN": 0.5}) == {"MIN_GAIN"}


def test_remaining_legacy_rule_constants_are_pinned():
    """L-07: close omissions exposed by requiring complete equality pins, without moving any rule."""
    from jobs.fla_rwkv7 import fla_equivalence as E
    from scripts import decide_9b as Q, decide_probe9b as P
    from train import promotion as D

    assert E.ARGS == ("r", "w_log", "k", "v", "kk", "a", "state")
    assert (E.CAPTURE_LAYERS, E.WARMUP_T, E.CHILD_EVAL_TOK_S) == ((0, 30, 60), (2048, 1000), 12000)
    assert E.DEVICE == "cuda" and E.MIX_PATH == None
    assert E.V2_IMPLS == (("eager_bf16", "eager"), ("chunked", "chunked"), ("fla", "fla"),
                          ("fla_safe_gate", "fla_safe_gate"))
    assert E.VARIANTS == ("fla", "fla_safe_gate")
    assert E.VERDICTS == ("ADOPT", "ADOPT_READOUT_ONLY", "ADOPT_TRAINING_ONLY", "REJECT_SPEED", "REJECT_NUMERICS",
                          "INCONCLUSIVE")
    assert Q.VERDICTS == ("open-$40-one-layer-probe-spend-issue", "re-price-lean-plan", "fold")
    assert P.REQUIRED == ("ppl_ratio_ctx1024", "ppl_ratio_ctx4096", "mmlu_drop_pts", "mqar_drop", "niah_mean_drop",
                          "index_diff_point", "index_diff_lo")
    assert P.BAND_OF == {"gate_delta_nll": "gate_delta_nll", "gate_kl": "gate_kl", "gate_hidden_cos": "gate_hidden_cos",
                         "ppl_ratio_ctx1024": "ppl_ratio", "ppl_ratio_ctx4096": "ppl_ratio",
                         "mmlu_drop_pts": "mmlu_drop_pts", "index_diff_point": "index_diff_point",
                         "index_diff_lo": "index_diff_lo"}
    assert D.CLEAN_ACC_DROP == 0.01
    # Approved #36 contract, including the frozen-set power requirements; never restore the superseded proposal.
    assert D.MARGINS == {"m_cap": 1.0, "m_steer": 0.02, "m_ece": 0.01, "m_ret_acc": 0.01,
                         "m_false_alarm": 0.01, "m_refusal": 0.01, "fa_abs_cap": 0.01,
                         "min_valid_draw_frac": 0.95,
                         "required_n": {"g13_retention_acc": 9300, "g13_retention_ece": 33300},
                         "alpha": 0.05, "area_family_alpha": 0.05, "template_family_alpha": 0.05,
                         "steer_bootstrap": "fixed_template", "null_pass_floor": 0.8, "approved": True,
                         "status": "approved: Codex cross-track decision, #36 comments 6028537932 + 6028610849 (2026-10-07); "
                                   "activated by the Track D coordinator under the user's delegation after independent re-reviews of "
                                   "#102 (FIX-THEN-MERGE findings resolved)"}
    assert D.E1_GROUPS == ("new_domain_proxy", "new_domain_frozen")
    assert D.G13_REPORT_EDITIONS == ("0.3",)


def test_current_candidate_rule_constants_are_pinned():
    """Keep current agent-main candidates covered when the stricter rule lint is integrated."""
    from scripts import decide_full8 as F
    from jobs.q9af_cell import rule as R

    assert F.REQUIRED == ("ppl_ratio_ctx1024", "ppl_ratio_ctx4096", "mmlu_drop_pts", "mqar_drop", "niah_mean_drop",
                          "index_diff_point", "index_diff_lo", "niah_length_drop")
    assert F.TOL == 1e-9
    assert F.VERDICTS == ("q9af-backbone", "retrieval-fail", "quality-fail", "partial-conversion", "no-conversion")
    assert R.ACTIONS == {
        "NO_MEASUREMENT": "fix the missing input / stop cause and re-run; no recipe conclusion",
        "THROUGHPUT": "re-price from the measured step time (run_cell exit 26); no recipe conclusion",
        "HYBRID_REPORT": "report as a hybrid (NOT attention-free); decision-needed issue to the user; never promoted",
        "FOLD": "stop this variant; log the negative result in CANDIDATES.md",
        "ESCALATE": "decision-needed issue to the user before any further spend, PREREG, full Index or TEST unseal",
        "PROMOTE": "CANDIDATES.md row; the variant is the scale-up base",
        "CONTINUE": "a scale-up cell may build on it; not a candidate",
        "INCONCLUSIVE": "expand the frozen retention evaluation and re-gate the same checkpoint; not a scale-up base",
    }


def test_no_control_characters_in_sources():
    """L-11: a heredoc turned '\\b' into a backspace (0x08) in a regex while this lint was written."""
    bad = []
    for sub, pat in (("jobs", "*.py"), ("jobs", "*.sh"), ("tests", "*.py"), ("scripts", "*.py")):
        for f in (ROOT / sub).rglob(pat):
            b = f.read_bytes()
            if re.search(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", b):
                bad.append(f.relative_to(ROOT).as_posix())
    assert bad == []


@pytest.mark.skipif(not GIT, reason="needs a git checkout")
def test_gitattributes_pin_lf():
    assert set(lint.eol_attrs(ROOT).values()) == {"lf"}, lint.eol_attrs(ROOT)


@pytest.mark.skipif(not GIT, reason="needs a git checkout")
def test_git_archive_head_has_no_cr_even_with_autocrlf():
    """L-14: what a Windows clone (core.autocrlf=true) would ship. -text paths and receipts/ are exempt."""
    assert lint.archive_cr_hits(ROOT, "HEAD", "true") == []


# ----------------------------------------------------------------------------- branch & champion policy
def _table_rows(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    head = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, (c.strip() for c in ln.strip("|").split("|")))) for ln in lines[2:]]


def test_champion_and_candidates_ledgers():
    champ = (ROOT / "CHAMPION.md").read_text(encoding="utf-8")
    for field in ("family", "branch", "merged sha", "checkpoint", "receipt", "promotion rule"):
        assert field in champ.lower(), f"CHAMPION.md lacks '{field}'"
    rows = _table_rows((ROOT / "CANDIDATES.md").read_text(encoding="utf-8"))
    assert rows, "CANDIDATES.md has no rows"
    for r in rows:
        assert re.search(r"(stack|claude|codex)/\S+", r.get("branch", "")), f"row without a branch: {r}"
        assert re.search(r"job-[a-z0-9]{5}|receipts/|art-[a-z0-9]{5}", r.get("receipt", "")), f"row without a receipt: {r}"
        assert r.get("verdict") and r.get("kept/folded"), f"row without verdict / kept-folded: {r}"


def test_docs_carry_the_branch_rule_and_the_incident_rule():
    for f in ("LESSONS.md", "jobs/LAUNCH_CHECKLIST.md"):
        assert "stack/<family>/<variant>" in (ROOT / f).read_text(encoding="utf-8"), f
    for f in ("COORDINATION.md", "AGENT_HANDOFF.md"):
        t = (ROOT / f).read_text(encoding="utf-8")
        for needle in ("LESSONS.md", "LAUNCH_CHECKLIST.md", "CHAMPION.md", "CANDIDATES.md"):
            assert needle in t, f"{f} does not link {needle}"


def test_lessons_entries_name_an_enforcing_check():
    text = (ROOT / "LESSONS.md").read_text(encoding="utf-8")
    entries = re.findall(r"^### (L-\d\d)", text, re.M)
    assert len(entries) >= 15 and len(entries) == len(set(entries))
    for block in re.split(r"^### ", text, flags=re.M)[1:]:
        assert "**Check:**" in block and "**Status:**" in block, block.splitlines()[0]


def test_coordination_has_no_superseded_spend_rule():
    """L-18: the later 'Review sufficiency' user decision governs spend; the old blocking sentence must not return."""
    t = (ROOT / "COORDINATION.md").read_text(encoding="utf-8")
    assert "Spend still needs the other agent's acknowledgement on a `spend` issue." not in t
    assert "never applies to `main` or to spend approval" not in t
    assert "requested but not blocking" in t
    assert "stop condition, acknowledged by the other agent." not in t  # spend ladder item 2 (PR #93 review)
    assert "reviewed per 'Review sufficiency' (the other agent's acknowledgement is requested but not blocking)" in t


def test_rate_guard_requires_an_executed_call():
    """PR #93 B2: a comment, an echo or a log path naming the guard is not a guard."""
    ok = ["python -m jobs.common.rate_guard preflight --tok-s 1",
          "  python -X utf8 -m jobs.probe9b.preflight --params p",
          "X=1 python3 -m jobs.common.rate_guard run --results r",
          "if ! python -m jobs.common.rate_guard preflight; then exit 15; fi",
          "bash jobs/x/preflight_check.sh"]
    bad = ["# TODO preflight", "# python -m jobs.common.rate_guard preflight",
           "echo 'python -m jobs.common.rate_guard'", "say phase preflight",
           "python -m jobs.x.score estimate > \"$WORK/out/preflight.json\"",
           "python -m jobs.common.rate_guard_v2 run"]
    for line in ok:
        assert lint.RATE_GUARD.search(f"#!/bin/bash{NL}{line}{NL}"), line
    for line in bad:
        assert not lint.RATE_GUARD.search(f"#!/bin/bash{NL}{line}{NL}"), line


def test_new_job_must_declare_decision_rule(tmp_path):
    """PR #93 B3: no `decision_rule` key -> fail, whatever the JOB_SPEC wording; null needs a reason."""
    (tmp_path / "jobs/new").mkdir(parents=True)
    (tmp_path / "jobs/new/JOB_SPEC.md").write_text("Decision: PASS if gain >= 0.5 else FAIL" + NL, encoding="utf-8")
    kw = dict(root=tmp_path, job="new", tests={}, rel="jobs/new/job_params.json")
    assert not lint.chk_decision_rule({}, **kw)[0]
    assert not lint.chk_decision_rule({"decision_rule": None}, **kw)[0]
    assert not lint.chk_decision_rule({"decision_rule": None, "no_verdict_reason": " "}, **kw)[0]
    assert lint.chk_decision_rule({"decision_rule": None, "no_verdict_reason": "throughput only"}, **kw)[0]
    # the legacy fallbacks never apply to a non-legacy params file
    assert not lint.chk_decision_rule({}, **{**kw, "job": "m2_parent_arm", "rel": "jobs/m2_parent_arm/job_params_x.json"})[0]
