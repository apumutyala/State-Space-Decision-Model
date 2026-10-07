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
import shlex
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
    "jobs/rwkv7/job_params_lora.json": {"build_timeout": _B, "shell_test": _T,
                                        "decision_rule": "2026-10-06: stage A is a measurement (no threshold); the stage B pass rule lives in "
                             "train.run_cell / jobs/rwkv7/pilot_cell.json, not pinned constants in scripts/decide_rwkv7.py"},
    "jobs/rwkv7/job_params_readout.json": {"build_timeout": _B, "shell_test": _T,
                                           "decision_rule": "2026-10-06: stage A is a measurement (no threshold); the stage B pass rule lives in "
                             "train.run_cell / jobs/rwkv7/pilot_cell.json, not pinned constants in scripts/decide_rwkv7.py"},
}

CHECKS = ("build_timeout", "schedule", "worst_case_stated", "registered", "shell_test", "decision_rule",
          "rate_guard", "no_backslash_n")


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
    the path is the executable or a shell's script argument, never a read-only command or syntax check."""
    if Path(runner).name not in _norm(src):
        return False
    return any(runner in t for call in _subprocess_texts(src) for t in call)


@functools.lru_cache(maxsize=None)
def _subprocess_texts(src: str) -> tuple[tuple[str, ...], ...]:
    """Executable script arguments and their bound paths for subprocess argv lists.

    A literal Bash -c command counts only when its last command explicitly execs the script.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ()
    binds: dict[str, list[ast.AST]] = {}
    subprocess_names = {"subprocess"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            subprocess_names.update(a.asname or a.name for a in n.names if a.name == "subprocess")
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        binds.setdefault(nm.id, []).append(n.value)
        elif isinstance(n, (ast.For, ast.comprehension)):
            for nm in ast.walk(n.target):
                if isinstance(nm, ast.Name):
                    binds.setdefault(nm.id, []).append(n.iter)

    def texts(expr):
        out = [ast.get_source_segment(src, expr) or ast.unparse(expr)]
        for nm in ast.walk(expr):
            if isinstance(nm, ast.Name):
                out += [ast.get_source_segment(src, value) or "" for value in binds.get(nm.id, [])]
        return tuple(_norm(t) for t in out)

    def argv_lists(expr):
        if isinstance(expr, (ast.List, ast.Tuple)):
            yield expr.elts
        elif isinstance(expr, ast.Name):
            for value in binds.get(expr.id, []):
                if isinstance(value, (ast.List, ast.Tuple)):
                    yield value.elts

    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _SUBPROC
                and isinstance(n.func.value, ast.Name) and n.func.value.id in subprocess_names):
            continue
        if any(k.arg == "shell" and not (isinstance(k.value, ast.Constant) and k.value.value is False)
               for k in n.keywords):
            continue
        args = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "args"), None)
        commands = list(argv_lists(args))
        for argv in commands:
            if not argv:
                continue
            first = texts(argv[0])
            # A direct executable path is also valid (its path is checked by executes_runner).
            executable = first[1:] if isinstance(argv[0], ast.Name) and binds.get(argv[0].id) else first
            shell = any(re.search(r"\bbash(?:\.exe)?\b|\bBASH\b", t) for t in executable)
            if not shell:
                out.append(first)
                continue
            i, skip = 1, False
            while i < len(argv):
                arg = argv[i]
                value = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
                if value == "--":
                    i += 1
                    break
                if value is None or not value.startswith(("-", "+")):
                    break
                if value == "-c" and i + 1 < len(argv) and isinstance(argv[i + 1], ast.Constant) \
                        and isinstance(argv[i + 1].value, str):
                    try:
                        terminal = shlex.split(argv[i + 1].value.split(";")[-1])
                    except ValueError:
                        terminal = []
                    if terminal[:1] == ["exec"]:
                        commands.append([ast.Constant(value=arg) for arg in terminal[1:]])
                if value in {"--noexec", "--help", "--version"} or (value.startswith("-")
                        and not value.startswith("--") and any(flag in value[1:] for flag in "ncs")):
                    skip = True
                    break
                if value in {"-o", "+o", "-O", "+O"}:
                    i += 1
                    if i >= len(argv) or not isinstance(argv[i], ast.Constant) or argv[i].value == "noexec":
                        skip = True
                        break
                i += 1
            if not skip and i < len(argv):
                out.append(texts(argv[i]))
    return tuple(out)


def chk_shell_test(p: dict, root: Path, job: str, tests: dict[str, str], **_) -> tuple[bool, str]:
    runner = runner_of(root, job, p)
    if runner is None:
        return False, "no runner (exactly one jobs/<job>/*_job.sh, or 'runner' in job_params)"
    hits = [f for f, s in tests.items() if executes_runner(s, runner)]
    return bool(hits), f"tests executing {runner} in a subprocess: {hits or 'none'}"


def _rule_literals(src: str) -> dict[str, object]:
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]*", t.id) and n.value is not None:
                    try:
                        value = ast.literal_eval(n.value)
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        continue
                    out[t.id] = value
    return out


def rule_constants(src: str) -> list[str]:
    return list(_rule_literals(src))


def _same_literal(a: object, b: object) -> bool:
    """Compare literal values without accepting bool/int or nested container type substitutions."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_literal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return len(a) == len(b) and all(any(_same_literal(k, k2) and _same_literal(v, v2)
                                           for k2, v2 in b.items()) for k, v in a.items())
    if isinstance(a, (set, frozenset)):
        return len(a) == len(b) and all(any(_same_literal(x, y) for y in b) for x in a)
    return a == b


@functools.lru_cache(maxsize=None)
def _rule_test_assertions(src: str) -> tuple[tuple[dict[str, str], ast.Compare], ...]:
    """Snapshot bindings at direct assertions in pytest test functions/methods.

    Helpers and nested functions cannot supply pins or imports. Reimports and writes invalidate old bindings;
    conditional writes invalidate conservatively, and assertions inside conditional/loop blocks do not count.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ()

    def writes(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {n.name}  # never descend into another scope
        if isinstance(n, ast.Lambda):
            return set()
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            return {n.id}
        if isinstance(n, ast.arg):
            return {n.arg}
        if isinstance(n, ast.Import):
            return {a.asname or a.name.split(".")[0] for a in n.names}
        if isinstance(n, ast.ImportFrom):
            return {a.asname or a.name for a in n.names}
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            # Changing an imported module's constant makes a subsequent equality vacuous.
            return {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
        return set().union(*(writes(child) for child in ast.iter_child_nodes(n)))

    def update(n, bindings):
        changed = writes(n)
        if "*" in changed:
            bindings.clear()
        for name in changed:
            bindings.pop(name, None)
        if isinstance(n, ast.Import):
            for alias in n.names:
                name = alias.asname or alias.name.split(".")[0]
                bindings[name] = alias.name if alias.asname else name
        elif isinstance(n, ast.ImportFrom) and not n.level:
            for alias in n.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{n.module}.{alias.name}"

    def required(expr):
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
            for part in expr.values:
                yield from required(part)
        elif isinstance(expr, ast.Compare):
            yield expr

    module_bindings = {}
    for n in tree.body:
        update(n, module_bindings)
    functions = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(n)
        elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
            functions.extend(m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assertions = []
    for fn in functions:
        if not fn.name.startswith("test_"):
            continue
        bindings = module_bindings.copy()
        # Python local names shadow module bindings throughout the function, including before their first write.
        locals_ = writes(fn.args) | set().union(*(writes(n) for n in fn.body))
        for name in locals_:
            bindings.pop(name, None)
        for n in fn.body:
            if isinstance(n, (ast.Return, ast.Raise)):
                break
            if isinstance(n, ast.Assert):
                assertions.extend((bindings.copy(), part) for part in required(n.test))
            update(n, bindings)
    return tuple(assertions)


def pinned_rule_constants(src: str, mod: str, consts: dict[str, object]) -> set[str]:
    def dotted(n):
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            base = dotted(n.value)
            return f"{base}.{n.attr}" if base else None
        return None

    def pins(ref, literal, bindings):
        name = dotted(ref)
        if name:
            base, _, attr = name.partition(".")
            resolved = bindings.get(base, "") + (f".{attr}" if attr else "")
            for constant, expected in consts.items():
                if resolved == f"{mod}.{constant}":
                    try:
                        actual = ast.literal_eval(literal)
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        return set()
                    return {constant} if _same_literal(actual, expected) else set()
        if isinstance(ref, (ast.Tuple, ast.List)) and type(ref) is type(literal) and len(ref.elts) == len(literal.elts):
            return set().union(*(pins(a, b, bindings) for a, b in zip(ref.elts, literal.elts)))
        return set()

    out = set()
    for bindings, expr in _rule_test_assertions(src):
        operands = [expr.left, *expr.comparators]
        for left, op, right in zip(operands, expr.ops, operands[1:]):
            if isinstance(op, ast.Eq):
                out |= pins(left, right, bindings) | pins(right, left, bindings)
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
    consts = _rule_literals(src)
    mod = rule[:-3].replace("/", ".")
    pinned = {f: hit for f, s in tests.items() if (hit := pinned_rule_constants(s, mod, consts))}
    covered = set().union(*pinned.values()) if pinned else set()
    missing = sorted(set(consts) - covered)
    return bool(consts and not missing), (f"{rule}: {len(covered)}/{len(consts)} literal constants equality-pinned; "
                                         f"missing {missing}; tests {list(pinned) or 'none'}")


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
