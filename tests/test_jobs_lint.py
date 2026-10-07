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
    split = 'import subprocess\nsubprocess.run([BASH, str(ROOT / "jobs" / "x" / "x_job.sh")])\n'
    via_name = 'import subprocess\np = ROOT / "jobs/x/x_job.sh"\nsubprocess.run(["bash", str(p)], check=True)\n'
    snippet = ('import subprocess\nlines = open("jobs/x/x_job.sh").read().splitlines()[:3]\n'
               'subprocess.run(["bash", "-c", "echo hi"])\n')
    assert lint.executes_runner(direct, runner) and lint.executes_runner(split, runner)
    assert lint.executes_runner(via_name, runner)
    assert not lint.executes_runner(snippet, runner)  # reading the runner's text is not executing it
    assert not lint.executes_runner(f"# {runner}\n", runner)


def test_rule_constants_are_literal_module_level():
    src = "A_MIN = 0.5\nBB = (1, 2)\nlower = 3\nC = f()\ndef g():\n    D = 1\n"
    assert lint.rule_constants(src) == ["A_MIN", "BB"]


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


def test_coordination_candidate_tip_exception_replaces_blanket_paid_merge_rule():
    policy = " ".join((ROOT / "COORDINATION.md").read_text(encoding="utf-8").split())
    assert "Also merge before any paid run" not in policy
    assert "Before a paid champion/integration run, merge its exact code to `agent-main`." in policy
    assert ("A reviewed pre-promotion measurement may instead use the exact pushed "
            "`stack/<family>/<variant>` tip through `--candidate-branch`;") in policy
