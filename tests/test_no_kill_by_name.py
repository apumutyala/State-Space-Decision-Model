"""LESSONS L-23: never kill processes by image name or name pattern on the shared laptop (it kills other agents' and
Codex's processes); kill only PIDs you started yourself and recorded. Greps jobs/, scripts/, tests/ and the agent docs
(*.md at the repo root and under jobs/) for name-based kills. Allowed: PID-based use (``taskkill /PID <pid>`` with no
/IM or /FI filter, ``kill <pid>``, ``Stop-Process -Id <pid>``) and the two files that document the ban itself."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: name-based kills: taskkill with /IM or /FI (image name / window-title filter), pkill, killall, PowerShell
#: Stop-Process -Name or piped from Get-Process, wmic ... delete, kill $(pgrep ...) / kill `pgrep ...`
PATTERN = re.compile(
    r"(?i)\btaskkill\b[^\n]*(/IM|/FI|-IM|-FI)\b"
    r"|\bpkill\b|\bkillall\b"
    r"|\bStop-Process\b[^\n]*-Name\b"
    r"|\bGet-Process\b[^\n]*\|\s*Stop-Process\b"
    r"|\bwmic\b[^\n]*\bdelete\b"
    r"|\bkill\b[^\n]*\$\(\s*pgrep\b"
    r"|\bkill\b[^\n]*`\s*pgrep\b")
NAME_FLAG = re.compile(r"(?i)(/IM|/FI|-IM|-FI)\b")
DOC_ALLOW = {"LESSONS.md", "COORDINATION.md"}  # they state the rule (L-23) and quote the forbidden commands
SELF = Path(__file__).resolve()


def _files():
    for d in ("jobs", "scripts", "tests"):
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".sh", ".md", ".ps1", ".bat", ".cmd", ".json") and p != SELF:
                yield p
    for p in ROOT.glob("*.md"):
        yield p


def offences(text: str) -> list[str]:
    """Every line matching a name-based kill. ``/PID`` never exempts a line: the taskkill pattern only fires on an
    /IM or /FI filter, so a pure-PID taskkill never matches and a mixed one always does."""
    return [ln for ln in text.splitlines() if PATTERN.search(ln)]


def test_no_name_based_process_kills_in_jobs_scripts_tests_or_agent_docs():
    bad = []
    for p in _files():
        if p.name in DOC_ALLOW and p.parent == ROOT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        bad += [f"{p.relative_to(ROOT)}: {ln.strip()}" for ln in offences(text)]
    assert not bad, "name-based process kills (LESSONS L-23; kill only PIDs you started):\n" + "\n".join(bad)


def test_the_check_fires_on_known_bad_and_not_on_pid_use():
    sl, dl, pp, bt = chr(47), chr(36), chr(124), chr(96)  # built so this file never contains the samples literally
    for bad in (f"taskkill {sl}F {sl}IM python.exe",
                f"taskkill {sl}F {sl}FI \"WINDOWTITLE eq x\"",
                f"taskkill {sl}F {sl}PID 12 {sl}IM python.exe",  # a /PID mention does not exempt an /IM filter
                "pk" + "ill -f train",
                "kill" + "all python",
                "Stop" + "-Process -Name python -Force",
                "Get" + f"-Process python {pp} Stop" + "-Process",
                "wm" + "ic process where name='python.exe' delete",
                "kill " + f"{dl}(pg" + "rep -f train)",
                "kill -9 " + f"{bt}pg" + f"rep python{bt}"):
        assert offences(bad), bad
    for good in (f"taskkill {sl}F {sl}PID 1234", "kill 1234", "kill -TERM \"$WD_PID\"", "Stop" + "-Process -Id 1234"):
        assert not offences(good), good


def test_coordination_and_lessons_carry_the_rule():
    c = (ROOT / "COORDINATION.md").read_text(encoding="utf-8")
    assert "Never kill processes by image name" in c
    assert "### L-23" in (ROOT / "LESSONS.md").read_text(encoding="utf-8")
