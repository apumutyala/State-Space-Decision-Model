"""Harness v3: the SessionStart hook config, the compact session digest and the AGENTS.md map stay within budget.

CPU only. The digest tests run scripts/session_start.sh under Git Bash with autor faked or absent and `git fetch` made
to fail (GIT_ALLOW_PROTOCOL=none), so they need no network and never touch WSL. The INBOX/claims tests build a tiny
throwaway git repo with a local refs/remotes/origin/state (no exported GIT_DIR / GIT_WORK_TREE, LESSONS L-23 era
incident with core.worktree).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "session_start.sh"
MAX_BYTES = 8000          # the digest's own hard cap (scripts/session_start.sh)
HOOK_CHAR_CAP = 10_000    # Claude Code caps hook stdout added to context at 10,000 characters


def _git_bash() -> str | None:
    """Git Bash, never WSL's System32 bash.exe (that would run inside the Linux distro)."""
    git = shutil.which("git")
    if git:
        for cand in (Path(git).resolve().parents[1] / "bin" / "bash.exe", Path(git).resolve().parents[1] / "usr" / "bin" / "bash.exe"):
            if cand.exists():
                return str(cand)
    b = shutil.which("bash")
    if b and "system32" not in b.lower():
        return b
    return None


BASH = _git_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="Git Bash not available")


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "SSDM_AGENT")}
    env.update({"SSDM_WSL": "no-such-wsl-for-tests.exe", "GIT_ALLOW_PROTOCOL": "none", "GIT_TERMINAL_PROMPT": "0",
                "SSDM_FETCH_TIMEOUT": "5", "CUDA_VISIBLE_DEVICES": ""})
    env.update(extra)
    return env


def _run(script: Path, *args: str, cwd: Path | None = None, timeout: int = 150, **env: str):
    t0 = time.monotonic()
    r = subprocess.run([BASH, str(script), *args], cwd=str(cwd or ROOT), env=_env(**env), capture_output=True,
                       timeout=timeout)
    return r, time.monotonic() - t0


# ----------------------------------------------------------------------------- hook config
def test_settings_json_has_sessionstart_hook_with_the_four_matchers():
    cfg = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    groups = cfg["hooks"]["SessionStart"]
    covered: set[str] = set()
    cmds = []
    for g in groups:
        covered |= set(g["matcher"].split("|"))
        for h in g["hooks"]:
            assert h["type"] == "command"
            assert isinstance(h.get("timeout"), int) and 20 <= h["timeout"] <= 120, "seconds; the digest bounds itself"
            cmds.append(h["command"])
    assert {"startup", "resume", "clear", "compact"} <= covered, covered
    assert any("$CLAUDE_PROJECT_DIR/scripts/session_start.sh" in c and c.startswith("bash ") for c in cmds), cmds


def test_settings_json_is_committed_not_ignored():
    r = subprocess.run(["git", "check-ignore", "-q", ".claude/settings.json"], cwd=ROOT, env=_env())
    assert r.returncode == 1, ".claude/settings.json is gitignored; add !.claude/settings.json to .gitignore"


# ----------------------------------------------------------------------------- docs
def test_agents_md_is_a_short_map_and_claude_md_points_to_it():
    lines = (ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 100, f"AGENTS.md has {len(lines)} lines (max 100)"
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
    text = "\n".join(lines)
    for needle in ("jevkit", "receipts/2026-10-09-jevkit-board/", "autor billing", "SSDM_AGENT", "APPROVE"):
        assert needle in text, needle


def test_lessons_headings_are_one_line_titles_the_digest_can_index():
    heads = [ln for ln in (ROOT / "LESSONS.md").read_text(encoding="utf-8").splitlines() if ln.startswith("### ")]
    assert len(heads) >= 15
    for h in heads:
        assert re.fullmatch(r"### L-\d\d \S.{4,}", h), f"LESSONS heading must be '### L-NN <title>': {h!r}"
        assert len(h) <= 110, f"LESSONS title too long for the digest index ({len(h)} chars): {h!r}"


def test_targets_tsv_parses():
    rows = [ln.split("\t") for ln in (ROOT / "TARGETS.tsv").read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    assert rows[0][:5] == ["rank", "target", "leader", "score", "ours"]
    assert all(len(r) >= 5 for r in rows), rows
    assert rows[1][0] == "1" and "Composite" in rows[1][1]


# ----------------------------------------------------------------------------- digest on the real repo
@needs_bash
def test_digest_offline_exits_0_and_stays_under_budget_on_current_state():
    r, _ = _run(SCRIPT, SSDM_AGENT="agent-a")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8", "replace")
    assert len(r.stdout) <= MAX_BYTES, len(r.stdout)
    assert "## Objective" in out and "## Targets" in out
    assert "(unavailable)" in out  # balance without wsl/autor
    # Every section is capped by count and width, so the untruncated digest also fits the hook's context cap.
    r2, _ = _run(SCRIPT, SSDM_AGENT="agent-a", SSDM_DIGEST_MAX_BYTES="1000000")
    assert r2.returncode == 0
    assert len(r2.stdout.decode("utf-8", "replace")) < HOOK_CHAR_CAP, len(r2.stdout)


@needs_bash
def test_full_flag_still_gives_the_verbose_dump():
    r, _ = _run(SCRIPT, "--full")
    assert r.returncode == 0
    assert b"== BOARD (origin/state) ==" in r.stdout and b"== LEDGER (last 12 runs) ==" in r.stdout


# ----------------------------------------------------------------------------- hermetic repo: INBOX, claims, autor
STATE_FILES = {
    "BOARD.md": "# BOARD\n\n| # | Item | Lane | Owner | Status |\n|---|---|---|---|---|\n"
                "| 1 | Kit parity | evidence | E | open |\n| 2 | Host scan | lead | L | running |\n\nProposals: below\n- p1\n",
    "INBOX.md": "# INBOX\n\n- @lead (from evidence, rescore): LEAD-ONLY item\n"
                "- @agent-a / @agent-b (from C, data): AGENT-A-ITEM\n- @evidence (from lead, kit): EVIDENCE-ONLY item\n"
                "- @all (from C, notice): ALL-ITEM\n- @leadership (from x): NOT-FOR-LEAD\n",
    "DEAD_ENDS.md": "# DEAD_ENDS\n\n- **Bold dead idea (job-aaaaa):** long text that must not show.\n"
                    "- 2026-10-09 (A): constant LR past one epoch. More text that must not show.\n",
    "PROGRESS.md": "# PROGRESS\n" + "".join(f"- progress line {i}\n" for i in range(20)),
    "results/ledger.tsv": "utc\tagent\tjob\tsha\trecipe\tindex\tlo\thi\tcost_usd\tstatus\treceipt\tnote\n"
                          "2026-10-09T20:30Z\tclaude\tjob-zzzzz\tabc\tRECIPE-X\t43.56\t39.93\t48.84\t3.70\tkeep\t\tn\n",
    "claims/X1.lock": "owner: lead\njob: job-xxxxx\nstart_utc: 2026-10-09T00:00Z\ntime_limit: 2000-01-01T00:00Z\n",
    "claims/X2.lock": "owner: evidence\njob: job-yyyyy\nstart_utc: 2026-10-09T00:00Z\ntime_limit: 2999-01-01T00:00Z\n",
}

FAKE_AUTOR = """#!/usr/bin/env bash
case " $* " in
  *" job ls "*) printf 'ID         STATUS     CHIP EXIT LABEL  SUBMITTED\\n'
                printf 'job-live1  running    h100 -    LIVE-LABEL one  2026-10-09T22:06:07+00:00\\n'
                printf 'job-done1  succeeded  h100 -    DONE-LABEL  2026-10-09T21:58:01+00:00\\n' ;;
  *" billing "*) printf '{\\n  "balance": {\\n    "available_usd": 123.45,\\n    "burn_rate_usd_per_hour": 6.5,\\n    "runway_minutes": 999\\n  },\\n  "credit_expiries": [\\n    {\\n      "expires_at": "2026-10-27T06:53:54+00:00",\\n' ;;
esac
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
                   cwd=cwd, env=_env(), check=True, capture_output=True)


@pytest.fixture(scope="module")
def mini_repo(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("digest") / "agent-x"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "session_start.sh")
    for f in ("LESSONS.md", "TARGETS.tsv"):
        shutil.copy(ROOT / f, repo / f)
    _git(repo, "init", "-q", "-b", "work")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "work")
    _git(repo, "checkout", "-q", "--orphan", "statebr")
    _git(repo, "rm", "-rfq", ".")
    for name, text in STATE_FILES.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "state")
    _git(repo, "update-ref", "refs/remotes/origin/state", "HEAD")
    _git(repo, "checkout", "-q", "work")
    fake = repo.parent / "fake-wsl"
    fake.write_bytes(FAKE_AUTOR.encode("utf-8"))
    hang = repo.parent / "hang-wsl"
    hang.write_bytes(b"#!/usr/bin/env bash\nexec sleep 90\n")
    return repo


def _digest(repo: Path, **env: str) -> str:
    r, _ = _run(repo / "scripts" / "session_start.sh", cwd=repo, **env)
    assert r.returncode == 0, r.stderr
    return r.stdout.decode("utf-8", "replace")


@needs_bash
def test_inbox_is_filtered_to_this_agent(mini_repo):
    lead = _digest(mini_repo, SSDM_AGENT="lead")
    assert "LEAD-ONLY" in lead and "ALL-ITEM" in lead
    assert "EVIDENCE-ONLY" not in lead and "AGENT-A-ITEM" not in lead and "NOT-FOR-LEAD" not in lead
    ev = _digest(mini_repo, SSDM_AGENT="evidence")
    assert "EVIDENCE-ONLY" in ev and "LEAD-ONLY" not in ev
    a = _digest(mini_repo, SSDM_AGENT="agent-a")
    assert "AGENT-A-ITEM" in a and "LEAD-ONLY" not in a
    unknown = _digest(mini_repo)  # worktree "agent-x" and branch "work" name no lane
    assert "set SSDM_AGENT" in unknown and "LEAD-ONLY" not in unknown


@needs_bash
def test_digest_sections_from_state(mini_repo):
    out = _digest(mini_repo, SSDM_AGENT="lead")
    assert "1 | Kit parity | evidence | E | open" in out and "1 proposals" in out
    assert re.search(r"- X1 \| lead \| job-xxxxx \| until 2000-01-01T00:00Z EXPIRED", out)
    assert re.search(r"- X2 \| evidence \| job-yyyyy \| until 2999-01-01T00:00Z$", out, re.M)
    assert "job-zzzzz | 43.56 [39.93,48.84] | 3.70 | keep | RECIPE-X" in out
    assert "- Bold dead idea (job-aaaaa)" in out and "must not show" not in out
    assert "- 2026-10-09 (A): constant LR past one epoch" in out
    assert "progress line 19" in out and "progress line 10" not in out
    assert "L-48" in out and "L-01 " not in out  # LESSONS: titles only, latest 10
    assert "**Check:**" not in out


@needs_bash
def test_autor_sections_parse_fake_autor(mini_repo):
    out = _digest(mini_repo, SSDM_AGENT="lead", SSDM_WSL=str(mini_repo.parent / "fake-wsl"))
    assert "- job-live1 running 10-09T22:06 | LIVE-LABEL one" in out
    assert "recent ended: job-done1 succeeded" in out and "DONE-LABEL" not in out
    assert "available $123.45 | burn $6.5/h | runway 999 min | first credit expiry 2026-10-27" in out


@needs_bash
def test_hung_autor_is_bounded_and_the_digest_still_exits_0(mini_repo):
    r, dt = _run(mini_repo / "scripts" / "session_start.sh", cwd=mini_repo, SSDM_AGENT="lead",
                 SSDM_WSL=str(mini_repo.parent / "hang-wsl"), SSDM_AUTOR_TIMEOUT="3")
    assert r.returncode == 0
    assert b"## Balance" in r.stdout and b"(unavailable)" in r.stdout
    assert dt < 60, f"digest took {dt:.0f} s with a hung autor (fake sleeps 90 s; timeout 3 s)"
