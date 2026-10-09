"""Launch lint for every paid job: one check per failure class in LESSONS.md (CPU only, no network, < 1 s).

    python -m jobs.common.lint                  # every jobs/<job>/job_params*.json + the repo-wide checks
    python -m jobs.common.lint rwkv7            # one job

Enforced by tests/test_jobs_lint.py. Used by jobs/common/launch.py before any context is built. A failing check is either
fixed or listed in WAIVERS (dated, reasoned, legacy params files only); new jobs cannot be waived. Each check names the
LESSONS.md entry it prevents.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: L-09 (job-4idg4/jucxx-adjacent: image publish timeouts at build_timeout 18): every build gets >= 60 min.
MIN_BUILD_TIMEOUT = 60
#: L-10: a literal backslash-n followed by an indented flag in a shell runner (passed every Python-level test).
BACKSLASH_N = re.compile(r"[\\]n[ \t]+--")
#: L-14: file types whose shipped bytes must be LF (sha256sum -c and bash both break on CR).
LF_PATTERNS = ("*.sh", "*.py", "*.json", "*.sha256", "Dockerfile")
#: paths exempt from the archive CR scan: byte-exact by design (`-text` in .gitattributes) or immutable evidence.
CR_EXEMPT_PREFIXES = ("receipts/",)
#: L-08: a runner must EXECUTE a measured-rate preflight / live rate guard before its long phase: a non-comment line
#: that starts (after optional VAR=x assignments, `if`, `!`, `(`) with `python -m jobs.common.rate_guard` or
#: `python -m jobs.<job>.preflight`, or `bash <...>preflight<...>.sh`. A comment or an echo does not count (PR #93 B2).
RATE_GUARD = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:if\s+|!\s*|\(\s*)*"
    r"(?:python[0-9.]*\s+(?:-[A-Za-z]\s*\S+\s+)*-m\s+(?:jobs\.common\.rate_guard|jobs\.\w+\.preflight)\b"
    r"|bash\s+\S*preflight\S*\.sh\b)", re.M)

#: params files that existed when this lint landed (2026-10-06). Only these may carry waivers.
LEGACY_PARAMS = frozenset({
    "jobs/cell_step_profile/job_params.json", "jobs/fla_rwkv7/job_params.json",
    "jobs/fla_rwkv7/job_params_truth_v2.json", "jobs/m2_parent_arm/job_params.json", "jobs/m2_pilot/job_params.json",
    "jobs/probe9b/job_params.json", "jobs/qual/job_params.json", "jobs/qual/job_params_c_only.json",
    "jobs/rwkv7/job_params_lora.json", "jobs/rwkv7/job_params_readout.json",
})
#: legacy jobs with a verdict -> their pre-registered rule module (new jobs declare "decision_rule" in job_params).
LEGACY_RULES = {
    "cell_step_profile": "jobs/cell_step_profile/rule.py", "fla_rwkv7": "jobs/fla_rwkv7/fla_equivalence.py",
    "probe9b": "scripts/decide_probe9b.py", "qual": "scripts/decide_9b.py", "rwkv7": "scripts/decide_rwkv7.py",
    "m2_pilot": "train/promotion.py",
}
#: legacy jobs that declare no verdict in their JOB_SPEC.
LEGACY_NO_VERDICT = frozenset({"m2_parent_arm"})

_B = "2026-10-06: spec predates the >=60 min build rule (L-09); raise build_timeout_minutes before its next launch"
_T = "2026-10-06: no test executes the runner in a subprocess (L-10); add a dry-run shell test before its next launch"
_R = ("2026-10-06: measured-rate projection via `python -m jobs.m2_parent_arm.score_parent estimate` (exit 15), not "
      "jobs.common.rate_guard / a *.preflight module; route it through rate_guard before launch")
#: {params file: {check: dated reason}}. tests/test_jobs_lint.py fails if a waiver is stale (the check now passes),
#: so this list only shrinks. launch.py refuses waived failures unless --allow-waived, which is recorded in submit.json.
WAIVERS: dict[str, dict[str, str]] = {
    "jobs/cell_step_profile/job_params.json": {
        "rate_guard": "2026-10-06: per-step cap guard inside cell_step.py (no step starts that cannot finish before "
                      "its phase cap); no long unmeasured-rate phase"},
    "jobs/fla_rwkv7/job_params.json": {
        "build_timeout": _B, "shell_test": _T,
        "rate_guard": "2026-10-06: fixed-size kernel gate, every phase capped from job-jucxx/jn47u measurements"},
    "jobs/fla_rwkv7/job_params_truth_v2.json": {
        "build_timeout": _B, "shell_test": _T,
        "rate_guard": "2026-10-06: fixed-size kernel gate, every phase capped from job-jucxx/jn47u measurements"},
    "jobs/m2_parent_arm/job_params.json": {
        "build_timeout": _B, "shell_test": _T,
        "registered": "2026-10-06: DRAFT, blocked on PENDING margins; register in costs.PARAMS + DECISION_TREE first",
        "rate_guard": _R},
    "jobs/m2_pilot/job_params.json": {
        "build_timeout": _B, "shell_test": _T,
        "registered": "2026-10-06: DRAFT, blocked on PENDING margins; register in costs.PARAMS + DECISION_TREE first",
        "rate_guard": _R},
    "jobs/probe9b/job_params.json": {"build_timeout": _B, "shell_test": _T},
    "jobs/qual/job_params.json": {
        "build_timeout": _B, "shell_test": _T,
        "schedule": "2026-10-06: caps sum 169 + reserve 8 < 180 (slack, not overrun); historical (job-jbw22/c4zkm)",
        "rate_guard": "2026-10-06: MFU/tok-s floors in-step (exit 3), no jobs.common.rate_guard; historical"},
    "jobs/qual/job_params_c_only.json": {
        "build_timeout": _B, "shell_test": _T,
        "schedule": "2026-10-06: caps sum 130 + reserve 8 < 140 (slack, not overrun); historical (job-4j3vu)",
        "rate_guard": "2026-10-06: MFU/tok-s floors in-step (exit 3), no jobs.common.rate_guard; historical"},
    "jobs/rwkv7/job_params_lora.json": {"build_timeout": _B,
                                        "decision_rule": "2026-10-06: stage A is a measurement (no threshold); the stage B pass rule lives in "
                             "train.run_cell / jobs/rwkv7/pilot_cell.json, not pinned constants in scripts/decide_rwkv7.py"},
    "jobs/rwkv7/job_params_readout.json": {"build_timeout": _B,
                                           "decision_rule": "2026-10-06: stage A is a measurement (no threshold); the stage B pass rule lives in "
                             "train.run_cell / jobs/rwkv7/pilot_cell.json, not pinned constants in scripts/decide_rwkv7.py"},
}

CHECKS = ("build_timeout", "schedule", "rate_current", "worst_case_stated", "registered", "shell_test", "decision_rule",
          "rate_guard", "no_backslash_n", "submit_checks")

#: L-25: the platform's CheckSpec table (validate_job's own error table, probed 2026-10-07: "<type> takes only ...",
#: "needs ...", "has no meaning in a success check"). type -> (required keys, allowed keys, phases, value rules).
#: A combinator is {"all_of"|"any_of": [CheckSpec, ...]}. success takes only `check`; health takes check, interval_s,
#: grace_period_s, fail_threshold, on_fail (record|kill|reschedule; interval_s and on_fail required).
def _int(lo=None, hi=None):
    def ok(v):
        return isinstance(v, int) and not isinstance(v, bool) and (lo is None or v >= lo) and (hi is None or v <= hi)
    ok.desc = f"an integer{'' if lo is None else f' >= {lo}'}{'' if hi is None else f' <= {hi}'} (not a bool)"
    return ok


def _num_pos(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


_num_pos.desc = "a positive number (not a bool)"


def _abs_path(v):
    return isinstance(v, str) and v.startswith("/")


_abs_path.desc = "an absolute path string"


def _nonempty_str(v):
    return isinstance(v, str) and bool(v.strip())


_nonempty_str.desc = "a non-empty string"


def _any(v):
    return v is not None


_any.desc = "a value"

BOTH, HEALTH = ("success", "health"), ("health",)
CHECK_TYPES: dict[str, dict] = {
    "file_exists": {"req": {"path"}, "opt": set(), "phases": BOTH},
    "file_grows": {"req": {"path"}, "opt": set(), "phases": HEALTH},
    "file_size": {"req": {"path"}, "opt": {"min_bytes", "max_bytes"}, "phases": BOTH,
                  "one_of": {"min_bytes", "max_bytes"}},
    "jsonl_field": {"req": {"path", "field"}, "opt": {"require"}, "phases": BOTH},
    "jsonl_rate": {"req": {"path", "min_per_interval"}, "opt": set(), "phases": HEALTH},
    "heartbeat": {"req": {"path"}, "opt": set(), "phases": HEALTH},
    "gpu_util": {"req": {"min_pct"}, "opt": set(), "phases": HEALTH},
    "gemm_sane": {"req": set(), "opt": set(), "phases": HEALTH},
    "http": {"req": {"port"}, "opt": {"path"}, "phases": HEALTH},
    "tensor_changed": {"req": {"path"}, "opt": set(), "phases": HEALTH},
    "exec": {"req": {"command"}, "opt": set(), "phases": BOTH},
}
VALUE_RULES = {"path": _abs_path, "min_bytes": _int(0), "max_bytes": _int(0), "field": _nonempty_str,
               "require": _any, "min_per_interval": _num_pos, "min_pct": _int(1, 100), "port": _int(1, 65535),
               "command": _nonempty_str}
VALUE_RULES_HTTP = {"path": lambda v: isinstance(v, str) and v.startswith("/")}
PLATFORM_CHECK_TYPES = frozenset(CHECK_TYPES)
HEALTH_ON_FAIL = frozenset({"record", "kill", "reschedule"})
HEALTH_KEYS = frozenset({"check", "interval_s", "grace_period_s", "fail_threshold", "on_fail"})


def check_spec_problems(spec, where: str = "check", phase: str = "success") -> list[str]:
    """Why one CheckSpec is not valid for this phase ([] = valid). Never raises on malformed input."""
    if not isinstance(spec, dict):
        return [f"{where}: a CheckSpec must be an object, got {type(spec).__name__} {spec!r:.80}"]
    combos = [k for k in ("all_of", "any_of") if k in spec]
    if combos:
        if len(combos) > 1 or len(spec) != 1:
            return [f"{where}: a combinator object holds exactly one of all_of / any_of, got keys {sorted(map(str, spec))}"]
        items = spec[combos[0]]
        if not isinstance(items, list) or not items:
            return [f"{where}.{combos[0]}: must be a non-empty list"]
        return [x for i, c in enumerate(items) for x in check_spec_problems(c, f"{where}.{combos[0]}[{i}]", phase)]
    t = spec.get("type")
    if not isinstance(t, str) or t not in CHECK_TYPES:
        return [f"{where}.type: {t!r:.80} is not a platform check type {sorted(CHECK_TYPES)}"]
    rule = CHECK_TYPES[t]
    out = []
    if phase not in rule["phases"]:
        out.append(f"{where}: {t} has no meaning in a {phase} check (phases: {', '.join(rule['phases'])})")
    keys = {k for k in spec if k != "type"}
    for k in sorted(map(str, keys - rule["req"] - rule["opt"])):
        out.append(f"{where}.{k}: {t} takes only type, {', '.join(sorted(rule['req'] | rule['opt'])) or '(nothing else)'}")
    for k in sorted(rule["req"] - keys):
        out.append(f"{where}: {t} needs `{k}`")
    if rule.get("one_of") and not (rule["one_of"] & keys):
        out.append(f"{where}: {t} needs {' or '.join(sorted(rule['one_of']))} (or both)")
    for k in sorted(map(str, keys & (rule["req"] | rule["opt"]))):
        ok = (VALUE_RULES_HTTP if t == "http" and k in VALUE_RULES_HTTP else VALUE_RULES)[k]
        if not ok(spec[k]):
            out.append(f"{where}.{k}: must be {getattr(ok, 'desc', 'valid')}, got {spec[k]!r:.80}")
    if t == "file_size" and isinstance(spec.get("min_bytes"), int) and isinstance(spec.get("max_bytes"), int) \
            and spec["min_bytes"] > spec["max_bytes"]:
        out.append(f"{where}: file_size min_bytes > max_bytes")
    return out


def checks_problems(checks) -> list[str]:
    """Why a submit_job `checks` value is not the platform schema ([] = valid). A success check is required.
    Never raises: any input (lists, bools, nested junk) yields a list of problems."""
    if not isinstance(checks, dict):
        return [f"checks must be an object {{success, health}}, got {type(checks).__name__} {checks!r:.80}"]
    out = [f"checks: unknown key {k!r:.40}" for k in checks if k not in ("success", "health")]
    succ = checks.get("success")
    if not isinstance(succ, dict) or "check" not in succ:
        out.append("checks.success: required, as {\"check\": CheckSpec}")
    else:
        out += [f"checks.success.{k}: the success phase takes only `check`" for k in map(str, succ) if k != "check"]
        out += check_spec_problems(succ["check"], "checks.success.check", "success")
    if "health" in checks:
        h = checks["health"]
        if not isinstance(h, dict) or "check" not in h:
            out.append("checks.health: must be {\"check\": CheckSpec, \"interval_s\": N, \"on_fail\": ...}")
        else:
            out += [f"checks.health.{k}: unknown field (health takes {sorted(HEALTH_KEYS)})"
                    for k in map(str, h) if k not in HEALTH_KEYS]
            out += check_spec_problems(h["check"], "checks.health.check", "health")
            if not _int(1)(h.get("interval_s")):
                out.append("checks.health.interval_s: required positive integer (not a bool)")
            on_fail = h.get("on_fail")
            if not isinstance(on_fail, str) or on_fail not in HEALTH_ON_FAIL:
                out.append(f"checks.health.on_fail: one of {sorted(HEALTH_ON_FAIL)}, got {on_fail!r:.40}")
            for k in ("grace_period_s", "fail_threshold"):
                if k in h and not _int(0)(h[k]):
                    out.append(f"checks.health.{k}: non-negative integer (not a bool)")
    return out


# ----------------------------------------------------------------------------- discovery
def params_files(root: Path = ROOT) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in (root / "jobs").glob("*/job_params*.json"))


def job_of(params_rel: str) -> str:
    return params_rel.split("/")[1]


def mode_of(params_rel: str) -> str | None:
    stem = Path(params_rel).stem
    return stem[len("job_params_"):] if stem.startswith("job_params_") else None


def runner_of(root: Path, job: str, p: dict) -> str | None:
    if p.get("runner"):
        return p["runner"]
    cands = sorted((root / "jobs" / job).glob("*_job.sh"))
    return cands[0].relative_to(root).as_posix() if len(cands) == 1 else None


def load_costs(root: Path):
    spec = importlib.util.spec_from_file_location(f"_lint_costs_{abs(hash(str(root)))}", root / "jobs/common/costs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(root: Path, rel: str) -> str:
    f = root / rel
    return f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""


def _test_sources(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted((root / "tests").rglob("*.py"))}


# ----------------------------------------------------------------------------- per-job checks (True = pass)
def schedule_total(p: dict) -> int:
    c = dict(p["phase_caps_minutes"])
    if "A_per_test" in c:  # jobs/qual layout
        c["A"] = c.pop("A_per_test") * c.pop("A_tests")
    return sum(c.values()) + p["reserve_minutes"]


def chk_build_timeout(p: dict, **_) -> tuple[bool, str]:
    b = p.get("build_timeout_minutes", 0)
    return b >= MIN_BUILD_TIMEOUT, f"build_timeout_minutes {b} (need >= {MIN_BUILD_TIMEOUT})"


def chk_schedule(p: dict, **_) -> tuple[bool, str]:
    t = schedule_total(p)
    return t == p["max_duration_minutes"], f"phase caps + reserve = {t}, max_duration_minutes = {p['max_duration_minutes']}"


def chk_rate_current(p: dict, costs, **_) -> tuple[bool, str]:
    """L-49: a params file priced at a stale rate under-states its worst case (the 2026-10-08 list change moved the
    billed batch rate $0.0495 -> $0.063/min while every params file still said 0.0495)."""
    want = costs.rates.BATCH_USD_PER_MIN
    got = p.get("batch_usd_per_min")
    return got == want, f"batch_usd_per_min {got} (jobs/common/rates.py BATCH_USD_PER_MIN {want}, since {costs.rates.EFFECTIVE})"


def chk_worst_case_stated(p: dict, root: Path, job: str, costs, **_) -> tuple[bool, str]:
    w = costs.worst_case_usd(p)
    spec = _read(root, f"jobs/{job}/JOB_SPEC.md")
    ok = f"{w:.2f}" in spec and w <= p.get("hard_cap_usd", float("inf"))
    return ok, f"worst case ${w:.2f} (costs.worst_case_usd) stated in JOB_SPEC.md and <= hard_cap_usd {p.get('hard_cap_usd')}"


def chk_registered(p: dict, root: Path, job: str, rel: str, costs, **_) -> tuple[bool, str]:
    in_costs = rel in set(costs.PARAMS.values())
    tree = _read(root, "jobs/DECISION_TREE.md")
    m = mode_of(rel)
    in_tree = rel in tree or any(f"jobs/{job}/" in ln and (m is None or re.search(r"\b" + re.escape(m) + r"\b", ln))
                                 for ln in tree.splitlines())
    return in_costs and in_tree, f"in costs.PARAMS: {in_costs}; in DECISION_TREE.md: {in_tree}"


_SUBPROC = {"run", "call", "check_call", "check_output", "Popen"}


def _norm(s: str) -> str:
    return re.sub(r"""["']\s*/\s*["']""", "/", s)


def executes_runner(src: str, runner: str) -> bool:
    """True if some subprocess.<run|call|check_call|check_output|Popen>(...) in this test source executes `runner`:
    the path appears in the call itself or in an assignment / for-target of a name the call uses."""
    if Path(runner).name not in _norm(src):
        return False
    return any(runner in t for call in _subprocess_texts(src) for t in call)


@functools.lru_cache(maxsize=None)
def _subprocess_texts(src: str) -> tuple[tuple[str, ...], ...]:
    """Per subprocess call: its source plus the bound values of every name it uses (normalised paths)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ()
    binds: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        binds.setdefault(nm.id, []).append(ast.get_source_segment(src, n.value) or "")
        elif isinstance(n, (ast.For, ast.comprehension)):
            for nm in ast.walk(n.target):
                if isinstance(nm, ast.Name):
                    binds.setdefault(nm.id, []).append(ast.get_source_segment(src, n.iter) or "")
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _SUBPROC):
            continue
        texts = [ast.get_source_segment(src, n) or ""]
        for nm in ast.walk(n):
            if isinstance(nm, ast.Name):
                texts += binds.get(nm.id, [])
        out.append(tuple(_norm(t) for t in texts))
    return tuple(out)


def chk_shell_test(p: dict, root: Path, job: str, tests: dict[str, str], **_) -> tuple[bool, str]:
    runner = runner_of(root, job, p)
    if runner is None:
        return False, "no runner (exactly one jobs/<job>/*_job.sh, or 'runner' in job_params)"
    hits = [f for f, s in tests.items() if executes_runner(s, runner)]
    return bool(hits), f"tests executing {runner} in a subprocess: {hits or 'none'}"


def rule_constants(src: str) -> list[str]:
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]+", t.id) and n.value is not None:
                    try:
                        ast.literal_eval(n.value)
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        continue
                    out.append(t.id)
    return out


def chk_decision_rule(p: dict, root: Path, job: str, tests: dict[str, str], rel: str = "", **_) -> tuple[bool, str]:
    """New jobs MUST declare `decision_rule`: a rule module path, or null plus a non-empty `no_verdict_reason`. The
    JOB_SPEC wording is never trusted to mean "no verdict" (PR #93 B3). Legacy files use LEGACY_RULES/NO_VERDICT."""
    if "decision_rule" in p:
        rule = p["decision_rule"]
        if rule is None:
            why = p.get("no_verdict_reason")
            ok = isinstance(why, str) and bool(why.strip())
            return ok, f"decision_rule: null; no_verdict_reason: {why!r} (required, non-empty)"
    elif rel in LEGACY_PARAMS and job in LEGACY_NO_VERDICT:
        return True, "no verdict (legacy JOB_SPEC)"
    elif rel in LEGACY_PARAMS and job in LEGACY_RULES:
        rule = LEGACY_RULES[job]
    else:
        return False, "job_params has no 'decision_rule' (a rule module path, or null + 'no_verdict_reason')"
    src = _read(root, rule)
    if not src:
        return False, f"decision rule {rule} missing"
    consts = rule_constants(src)
    mod = rule[:-3].replace("/", ".")
    stem = Path(rule).stem
    pinned = []
    for f, s in tests.items():
        if mod not in s and not re.search(rf"\b{re.escape(stem)}\b", s):
            continue
        for line in s.splitlines():
            if ("==" in line or "assert" in line) and any(re.search(rf"\b{c}\b", line) for c in consts):
                pinned.append(f)
                break
    return bool(consts and pinned), f"{rule}: {len(consts)} literal constants; pinned by {pinned or 'no test'}"


def chk_rate_guard(p: dict, root: Path, job: str, **_) -> tuple[bool, str]:
    runner = runner_of(root, job, p)
    ok = bool(runner) and bool(RATE_GUARD.search(_read(root, runner)))
    return ok, f"{runner} calls a measured-rate preflight / jobs.common.rate_guard: {ok}"


def chk_no_backslash_n(p: dict, root: Path, job: str, **_) -> tuple[bool, str]:
    bad = backslash_n_hits(root, f"jobs/{job}")
    return not bad, f"literal backslash-n continuations: {bad or 'none'}"


def backslash_n_hits(root: Path, sub: str = "jobs") -> list[str]:
    hits = []
    for f in sorted((root / sub).rglob("*.sh")):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if BACKSLASH_N.search(line):
                hits.append(f"{f.relative_to(root).as_posix()}:{i}")
    return hits


def chk_submit_checks(p: dict, **_) -> tuple[bool, str]:
    """L-25: a submit block's checks must be the platform's CheckSpec schema (legacy jobs without a submit block get
    theirs from launch.py's LEGACY_SUBMIT, which launch.submit_args validates the same way)."""
    sub = p.get("submit")
    if not sub:
        return True, "no submit block (legacy table in launch.py; validated at submit_args)"
    bad = checks_problems(sub.get("checks"))
    return not bad, f"submit.checks: {bad or 'valid'}"


REQUIRES_TRANSFORMERS = re.compile(r"requires_transformers\(\s*[\"']([0-9][0-9.]*)[\"']\s*\)")


def collected_by(src: str, job: str, rel_test: str) -> bool:
    """launch.job_tests' collection rule, mirrored: a test file names jobs/<job>/ or jobs.<job>, or is the lint test."""
    return f"jobs/{job}/" in src or f"jobs.{job}" in src or rel_test.endswith("test_jobs_lint.py")


def pinned_tests_uncovered(root: Path = ROOT, tests: dict[str, str] | None = None) -> list[str]:
    """PR #103 review B2: every test file with requires_transformers(X) must be collected by at least one job whose
    test_env.transformers == X, or its pinned tests run in no launch at all (skipped everywhere, tested nowhere)."""
    tests = _test_sources(root) if tests is None else tests
    pins: dict[str, set[str]] = {}
    for rel in params_files(root):
        te = json.loads((root / rel).read_text(encoding="utf-8")).get("test_env") or {}
        if te.get("transformers"):
            pins.setdefault(te["transformers"], set()).add(job_of(rel))
    bad = []
    for f, src in tests.items():
        for want in sorted(set(REQUIRES_TRANSFORMERS.findall(src))):
            if not any(collected_by(src, j, f) for j in pins.get(want, ())):
                bad.append(f"{f}: requires_transformers({want}) but no job with test_env.transformers == {want} "
                           f"collects it (jobs with that pin: {sorted(pins.get(want, ()))})")
    return bad


def lint_params(rel: str, root: Path = ROOT, costs=None, tests=None) -> dict[str, tuple[bool, str]]:
    costs = costs or load_costs(root)
    tests = _test_sources(root) if tests is None else tests
    p = json.loads((root / rel).read_text(encoding="utf-8"))
    kw = dict(root=root, job=job_of(rel), rel=rel, costs=costs, tests=tests)
    return {c: globals()[f"chk_{c}"](p, **kw) for c in CHECKS}


def failures(rel: str, res: dict[str, tuple[bool, str]]) -> tuple[dict, dict]:
    """(unwaived failures, waived failures) for one params file."""
    w = WAIVERS.get(rel, {}) if rel in LEGACY_PARAMS else {}
    bad = {c: m for c, (ok, m) in res.items() if not ok}
    return {c: m for c, m in bad.items() if c not in w}, {c: f"{m} [WAIVED {w[c]}]" for c, m in bad.items() if c in w}


# ----------------------------------------------------------------------------- repo-wide checks (need git)
def eol_attrs(root: Path = ROOT) -> dict[str, str]:
    probes = ["x.sh", "x.py", "x.json", "x.sha256", "Dockerfile", "jobs/x/Dockerfile"]
    out = subprocess.run(["git", "-C", str(root), "check-attr", "eol", "--", *probes], capture_output=True, text=True,
                         check=True).stdout
    return {ln.split(":")[0].strip(): ln.rsplit(":", 1)[1].strip() for ln in out.splitlines()}


def archive_cr_hits(root: Path = ROOT, rev: str = "HEAD", autocrlf: str = "true") -> list[str]:
    """Files of the pinned types whose `git archive <rev>` bytes contain CR. autocrlf=true reproduces a Windows clone
    (the CRLF *.sha256 incident). Paths with `-text` (byte-exact vendored code/fixtures) and receipts/ are exempt."""
    import io
    import tarfile

    specs = [f":(glob)**/{p}" for p in LF_PATTERNS]
    raw = subprocess.run(["git", "-C", str(root), "-c", f"core.autocrlf={autocrlf}", "archive", rev, "--", *specs],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        cand = [m.name for m in t.getmembers() if m.isfile() and b"\r" in t.extractfile(m).read()]
    cand = [c for c in cand if not c.startswith(CR_EXEMPT_PREFIXES)]
    if not cand:
        return []
    attrs = subprocess.run(["git", "-C", str(root), "check-attr", "--source", rev, "text", "--", *cand],
                           capture_output=True, text=True, check=True).stdout
    unset = {ln.split(": text:")[0] for ln in attrs.splitlines() if ln.endswith("unset")}
    return [c for c in cand if c not in unset]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    costs, tests = load_costs(ROOT), _test_sources(ROOT)
    rc = 0
    for rel in params_files(ROOT):
        if argv and job_of(rel) not in argv:
            continue
        bad, waived = failures(rel, lint_params(rel, ROOT, costs, tests))
        print(f"{'FAIL' if bad else 'ok  '} {rel}")
        for c, m in {**bad, **waived}.items():
            print(f"     {c}: {m}")
        rc |= bool(bad)
    if not argv:
        attrs = eol_attrs(ROOT)
        cr = archive_cr_hits(ROOT)
        print(f"eol attrs {attrs}; archive CR hits {cr or 'none'}; backslash-n {backslash_n_hits(ROOT) or 'none'}")
        rc |= any(v != "lf" for v in attrs.values()) or bool(cr) or bool(backslash_n_hits(ROOT))
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
