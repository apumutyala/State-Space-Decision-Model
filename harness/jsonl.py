"""Strict JSONL reading for results / side files: one record per ``\\n``-terminated line, nothing else.

Why this exists (job-sgihj, 2026-10-06). ``summarize.py composite`` read results with ``str.splitlines()``, which also
splits on U+2028/U+2029, U+0085, \\v, \\f and \\x1c-\\x1e. ``json.dumps(ensure_ascii=False)`` (the kit's writer) leaves
U+2028 raw inside strings, so one GPQA-Diamond row was cut in two and the composite crashed with "Unterminated
string". The file itself was complete (12,104 rows, final newline). The opposite failure is the kit's own
``load_results``: it stops silently at a partial last line, so a truncated or partial results file scored as if the
missing rows were unanswered. Both are fixed here:

* lines are split on ``\\n`` only (``\\r\\n`` tolerated), never on other Unicode line boundaries;
* a non-empty file whose last byte is not ``\\n`` is a truncated write -> ``ResultsFileError("truncated_tail")``;
* an unparseable line -> ``"malformed"`` (with path:line), a non-object line -> ``"malformed"``;
* duplicate ``run_id`` -> ``"duplicate"``; expected ids absent -> ``"incomplete"``.

``EXIT_RESULTS`` (3) is the exit code every CLI here uses for these errors, so a runner can map it to its own
"results file incomplete" code instead of a Python traceback.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

EXIT_RESULTS = 3


class ResultsFileError(ValueError):
    def __init__(self, kind: str, path: str | Path, detail: str):
        self.kind, self.path, self.detail = kind, str(path), detail
        super().__init__(f"{path}: {kind}: {detail}")


def _bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def iter_jsonl_strict(path: str | Path, *, allow_truncated_tail: bool = False):
    """Yield (line_number, record). See the module docstring for what raises."""
    p = Path(path)
    if not p.exists():
        raise ResultsFileError("missing", p, "file does not exist")
    try:
        text = _bytes(p).decode("utf-8")
    except (UnicodeDecodeError, OSError, EOFError) as e:
        raise ResultsFileError("decode", p, str(e)) from None
    lines = text.split("\n")
    tail = lines.pop()  # "" when the file ends with "\n"
    if tail.strip():
        if not allow_truncated_tail:
            raise ResultsFileError("truncated_tail", p, f"line {len(lines) + 1} has no terminating newline "
                                                         f"({len(tail)} chars): partial write")
    for i, line in enumerate(lines, 1):
        if line.endswith("\r"):
            line = line[:-1]
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ResultsFileError("malformed", p, f"line {i}: {e.msg} at column {e.colno}") from None
        if not isinstance(rec, dict):
            raise ResultsFileError("malformed", p, f"line {i}: not a JSON object")
        yield i, rec


def read_jsonl_strict(path: str | Path, *, allow_truncated_tail: bool = False) -> list[dict[str, Any]]:
    return [r for _, r in iter_jsonl_strict(path, allow_truncated_tail=allow_truncated_tail)]


def read_results_strict(path: str | Path, expected_ids: Iterable[str] | None = None, *,
                        allow_partial: bool = False, key: str = "run_id") -> dict[str, dict[str, Any]]:
    """{run_id: row}. Fails on a truncated tail, a malformed line, a duplicate id, and (unless ``allow_partial``) on
    any ``expected_ids`` with no row. ``allow_partial`` keeps the kit's semantics for a run in progress (a partial
    last line is dropped, missing rows score as unanswered) and must be asked for explicitly."""
    out: dict[str, dict[str, Any]] = {}
    for i, r in iter_jsonl_strict(path, allow_truncated_tail=allow_partial):
        rid = r.get(key)
        if rid is None:
            raise ResultsFileError("malformed", path, f"line {i}: no {key}")
        if rid in out:
            raise ResultsFileError("duplicate", path, f"line {i}: {key} {rid} already seen")
        out[rid] = r
    if expected_ids is not None and not allow_partial:
        missing = sorted(set(expected_ids) - set(out))
        if missing:
            raise ResultsFileError("incomplete", path, f"{len(missing)} expected {key}s have no row "
                                                       f"(e.g. {missing[:3]}); pass --allow-partial to score anyway")
    return out


def drop_truncated_tail(path: str | Path) -> str | None:
    """For append-only checkpoint files (resume after a kill): cut a partial last line off ``path`` so the next append
    starts on a fresh line. The dropped bytes are kept next to it as ``<name>.truncated-tail`` (evidence, never
    re-read). Returns the dropped text, or None when the file was clean. Plain (non-gz) files only."""
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    cut = raw.rfind(b"\n") + 1
    if cut == len(raw):
        return None
    tail = raw[cut:]
    with open(p.with_name(p.name + ".truncated-tail"), "ab") as fh:
        fh.write(tail + b"\n")
    with open(p, "r+b") as fh:
        fh.truncate(cut)
    return tail.decode("utf-8", errors="replace")


def load_results(path: str | Path, expected_ids: Iterable[str] | None = None, *,
                 allow_partial: bool = False) -> dict[str, dict[str, Any]]:
    """Strict replacement for ``decision_index.scoring.report.load_results`` (same ``compact`` rows)."""
    from decision_index.scoring.report import compact

    return {k: compact(v) for k, v in read_results_strict(path, expected_ids, allow_partial=allow_partial).items()}
