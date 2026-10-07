"""Regression tests for the job-sgihj composite crash (LESSONS L-15) and the strict results readers.

Root cause (reproduced on the job's own output, art-8y6d8): index/decide/results.jsonl was COMPLETE (12,104 rows, final
newline), but row 25:GPQA-Diamond:GPQA-Diamond:test:4 carries a raw U+2028 inside a JSON string (the kit writes
ensure_ascii=False), and summarize.py split the file with str.splitlines(), which treats U+2028 as a line break. The
record was cut in two -> JSONDecodeError "Unterminated string" -> composite exit 1 -> runner exit 12 before S3/S4.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUMMARIZE = ROOT / "jobs" / "clef_baseline" / "summarize.py"

from harness.jsonl import (EXIT_RESULTS, ResultsFileError, drop_truncated_tail, read_jsonl_strict,  # noqa: E402
                           read_results_strict)

LS = " "


def _row(rid, status="ok", text="plain", **kw):
    return {"run_id": rid, "catalog_id": 1, "status": status, "payload": {"state": text},
            "response": {"answers": {"q": {"choice": "a"}}}, "total_wall_ms": 10.0, **kw}


def _write(p: Path, rows, tail: str = ""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + tail, encoding="utf-8",
                 newline="\n")
    return p


def test_root_cause_splitlines_cuts_a_record_on_u2028(tmp_path):
    p = _write(tmp_path / "r.jsonl", [_row("a"), _row("b", text=f"x{LS}y"), _row("c")])
    text = p.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 4  # the old reader: one record became two lines
    with pytest.raises(json.JSONDecodeError):
        [json.loads(x) for x in text.splitlines() if x.strip()]
    rows = read_jsonl_strict(p)  # the fix: newline-only split
    assert [r["run_id"] for r in rows] == ["a", "b", "c"] and rows[1]["payload"]["state"] == f"x{LS}y"


@pytest.mark.parametrize("sep", [" ", "\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"])
def test_other_unicode_line_boundaries_do_not_split(tmp_path, sep):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"run_id": "a", "t": f"x{sep}y"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl_strict(p)[0]["t"] == f"x{sep}y"


def test_truncated_tail_fails_loudly(tmp_path):
    p = _write(tmp_path / "r.jsonl", [_row("a")], tail='{"run_id": "b", "sta')
    with pytest.raises(ResultsFileError) as e:
        read_jsonl_strict(p)
    assert e.value.kind == "truncated_tail"
    assert [r["run_id"] for r in read_jsonl_strict(p, allow_truncated_tail=True)] == ["a"]


def test_malformed_duplicate_incomplete(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"run_id": "a"}\n{"run_id": \n{"run_id": "c"}\n', encoding="utf-8")
    with pytest.raises(ResultsFileError) as e:
        read_jsonl_strict(p)
    assert e.value.kind == "malformed" and "line 2" in str(e.value)
    d = _write(tmp_path / "d.jsonl", [_row("a"), _row("a")])
    with pytest.raises(ResultsFileError) as e:
        read_results_strict(d)
    assert e.value.kind == "duplicate"
    q = _write(tmp_path / "q.jsonl", [_row("a")])
    with pytest.raises(ResultsFileError) as e:
        read_results_strict(q, expected_ids=["a", "b"])
    assert e.value.kind == "incomplete"
    assert set(read_results_strict(q, expected_ids=["a", "b"], allow_partial=True)) == {"a"}
    assert set(read_results_strict(q, expected_ids=["a"])) == {"a"}  # a superset file is fine


def test_drop_truncated_tail_keeps_evidence(tmp_path):
    p = _write(tmp_path / "gcg_attacks.jsonl", [{"x": 1}], tail='{"x": 2, "partial')
    assert drop_truncated_tail(p) == '{"x": 2, "partial'
    assert p.read_text(encoding="utf-8") == '{"x": 1}\n'
    assert (tmp_path / "gcg_attacks.jsonl.truncated-tail").read_text(encoding="utf-8").startswith('{"x": 2')
    assert drop_truncated_tail(p) is None


def _composite(tmp_path, dec, nat, probes):
    _write(tmp_path / "decide.jsonl", dec) if isinstance(dec, list) else None
    _write(tmp_path / "native.jsonl", nat) if isinstance(nat, list) else None
    (tmp_path / "probes.json").write_text(json.dumps(probes), encoding="utf-8")
    out = tmp_path / "comp" / "results.jsonl"
    r = subprocess.run([sys.executable, "-X", "utf8", str(SUMMARIZE), "composite", "--decide",
                        str(tmp_path / "decide.jsonl"), "--native", str(tmp_path / "native.jsonl"), "--probes",
                        str(tmp_path / "probes.json"), "--out", str(out)], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    return r, out


PROBES = {"encodings": {"rows": 3, "clef_would_truncate": [{"run_id": "c"}]}, "native_arm_run_ids": ["c"],
          "parity_sample_run_ids": ["a"]}


def test_composite_on_u2028_rows_succeeds_and_replaces_only_native_arm(tmp_path):
    dec = [_row("a"), _row("b", text=f"GPQA{LS}row"), _row("c", status="unsupported")]
    nat = [_row("a", raw_output={"native": True}), _row("c", raw_output={"native": True})]
    r, out = _composite(tmp_path, dec, nat, PROBES)
    assert r.returncode == 0, r.stderr
    got = read_jsonl_strict(out)
    assert [x["run_id"] for x in got] == ["a", "b", "c"]
    assert got[2]["status"] == "ok" and got[2]["raw_output"] == {"native": True}  # native arm row substituted
    assert "raw_output" not in got[0]  # parity-sample row keeps the decide output (the documented rule)
    prov = json.loads((out.parent / "composite.provenance.json").read_text(encoding="utf-8"))
    assert prov["replaced_run_ids"] == ["c"] and prov["native_rows_missing_from_native_run"] == []


def test_composite_truncated_native_exits_3_and_writes_nothing(tmp_path):
    _write(tmp_path / "native.jsonl", [_row("a")], tail='{"run_id": "c", "stat')
    r, out = _composite(tmp_path, [_row("a"), _row("b"), _row("c")], None, PROBES)
    assert r.returncode == 3 and "truncated_tail" in r.stderr and not out.exists()


def test_composite_partial_runs_exit_4(tmp_path):
    r, out = _composite(tmp_path, [_row("a"), _row("b"), _row("c")], [_row("a")], PROBES)  # native-arm row c missing
    assert r.returncode == 4 and "native-arm rows missing" in r.stderr and not out.exists()
    r, out = _composite(tmp_path, [_row("a"), _row("c")], [_row("c")], PROBES)  # decide stopped early: 2 of 3 rows
    assert r.returncode == 4 and "partial decide run" in r.stderr and not out.exists()


def test_composite_truncated_decide_exits_3(tmp_path):
    _write(tmp_path / "decide.jsonl", [_row("a"), _row("b")], tail='{"run_id": "c"')
    r, out = _composite(tmp_path, None, [_row("c")], PROBES)
    assert r.returncode == 3 and not out.exists()


def test_exit_code_constant():
    assert EXIT_RESULTS == 3


# ----------------------------------------------------------------------------- merged tree (PR #95 require_results)
def _dev_or_skip():
    pytest.importorskip("decision_index", reason="kit not installed")
    from harness import config

    if not (config.DATA_ROOT / "slices" / "dev-rows.jsonl.gz").exists():
        pytest.skip("private DEV rows not present")


def _rows_for(ids, tmp_path, name):
    rows = [{"run_id": i, "catalog_id": 0, "status": "ok", "response": {"answers": {}}} for i in ids]
    return _write(tmp_path / name, rows)


@pytest.mark.parametrize("edition", ["0.2.1", "0.3"])
def test_missing_row_exits_3_through_both_clis(tmp_path, edition):
    """On the merged tree the strict reader runs BEFORE #95's require_results (which would exit 1): a results file that
    misses one slice/subset row must exit 3, which run_job.sh maps to 17."""
    _dev_or_skip()
    import gzip

    from harness import cli
    from jobs.common import dev_subset

    man = [json.loads(x) for x in gzip.open(ROOT / "harness/manifests/dev-manifest.jsonl.gz", "rt", encoding="utf-8")]
    ids = [m["run_id"] for m in man]
    f = _rows_for(ids[1:], tmp_path, "dev.jsonl")  # every DEV row but one
    with pytest.raises(SystemExit) as e:
        cli.main(["score", "--results", str(f), "--slice", "dev", "--edition", edition, "--omit-gsm8k03", "--B", "10"])
    assert e.value.code == 3
    with pytest.raises(SystemExit) as e:
        cli.main(["compare", "--candidate", str(f), "--baseline", str(f), "--slice", "dev", "--edition", edition,
                  "--omit-gsm8k03", "--B", "10"])
    assert e.value.code == 3
    sub = dev_subset.load_subset()["run_ids"]
    g = _rows_for(sorted(sub)[1:], tmp_path, "sub.jsonl")
    assert dev_subset.main(["score", "--results", str(g), "--edition", edition, "--omit-gsm8k03", "--B", "10"]) == 3
    t = _write(tmp_path / "trunc.jsonl", [{"run_id": i, "status": "ok"} for i in sorted(sub)], tail='{"run_id": "x"')
    assert dev_subset.main(["score", "--results", str(t), "--edition", edition, "--B", "10"]) == 3
