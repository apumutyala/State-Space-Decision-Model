"""Stage ``screen`` (jobs/m2_first_cell/screen.py): reference pinning (shipped / downloaded artifact chain, fail
closed), the DROP / PROMOTE rule, the frozen-scorer report on real proxy rows, and the runner wiring."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from jobs.m2_first_cell import screen as SC

ROOT = Path(__file__).resolve().parents[1]
JOB = (ROOT / "jobs" / "m2_first_cell" / "first_cell_job.sh").read_text(encoding="utf-8")
D1_UNITS = Path(r"C:\Users\apumu\rescore_c7pq2\dl\x_final\out\units-child-index.json")
STOCK_V1 = Path(r"C:\Users\apumu\m2fc_uhxb5_out\x_cell\out\parent_arm\units-index.json")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tar_bytes(members: dict, gz=False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if gz else "w") as t:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _chain(tmp_path, units: bytes):
    inner = _tar_bytes({"out/units-child-index.json": units}, gz=True)
    outer = _tar_bytes({"./phases/final.tgz": inner})
    art = tmp_path / "art.tar"
    art.write_bytes(outer)
    ref = dict(SC.DEFAULT_REFERENCE, artifact="art-test", artifact_sha256=_sha(outer), member_sha256=_sha(inner),
               units_sha256=_sha(units), shipped="screen_reference/x/units-child-index.json")
    return ref, art


def test_default_reference_is_d1_and_validated():
    ref = SC.reference_block({})
    assert ref["artifact"] == "art-bm3e6" and ref["units_sha256"].startswith("545dbb94")
    with pytest.raises(SC.ScreenError):
        SC.reference_block({"screen_reference": dict(ref, units_sha256="nope")})
    with pytest.raises(SC.ScreenError):
        SC.reference_block({"screen_reference": dict(ref, shipped="../escape.json")})


def test_reference_shipped_then_downloaded_chain_and_fail_closed(tmp_path, monkeypatch):
    units = json.dumps({"index_results": {"a": {"status": "ok"}}}).encode()
    ref, art = _chain(tmp_path, units)
    data = tmp_path / "data"
    monkeypatch.delenv("REF_URL", raising=False)
    with pytest.raises(SC.ScreenError, match="REF_URL"):
        SC.locate_reference(ref, data, tmp_path / "w")  # nothing shipped, no URL: fail closed
    rec = SC.locate_reference(ref, data, tmp_path / "w", url=art.as_uri())  # artifact -> member -> inner, all pinned
    assert rec["how"] == "downloaded" and Path(rec["path"]).read_bytes() == units
    with pytest.raises(SC.ScreenError, match="artifact"):
        SC.locate_reference(dict(ref, artifact_sha256="0" * 64), data, tmp_path / "w2", url=art.as_uri())
    with pytest.raises(SC.ScreenError):
        SC.locate_reference(dict(ref, units_sha256="0" * 64), data, tmp_path / "w3", url=art.as_uri())
    shipped = data / ref["shipped"]
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(units)
    assert SC.locate_reference(ref, data, tmp_path / "w4")["how"] == "shipped"
    shipped.write_bytes(units + b" ")  # a shipped file with the wrong sha is never used
    with pytest.raises(SC.ScreenError):
        SC.locate_reference(ref, data, tmp_path / "w5")


def test_rule_and_restrict():
    assert SC.verdict(-2.51) == "DROP" and SC.verdict(-2.5) == "PROMOTE" and SC.verdict(3.0) == "PROMOTE"
    with pytest.raises(SC.ScreenError, match="miss 1"):
        SC.restrict({"a": 1}, {"a", "b"}, "reference")


def test_estimate_from_d1_rate():
    e = SC.estimate({"phase_caps_minutes": {"screen": 60}, "batch_usd_per_min": 0.0495})
    assert e["ok"] and e["minutes"] == pytest.approx(40.97, abs=0.1) and e["usd_batch"] == pytest.approx(2.03, abs=0.01)
    assert not SC.estimate({"phase_caps_minutes": {"screen": 40}, "batch_usd_per_min": 0.0495})["ok"]


@pytest.mark.skipif(not (D1_UNITS.is_file() and STOCK_V1.is_file()), reason="local D1 / stock v1 units not present")
def test_report_on_real_proxy_rows_pairs_with_the_frozen_scorer():
    from jobs.common import dev_proxy as P

    rows = [r for r in P.subset_rows() if r["_evaluation"]["run_id"] in P.proxy_ids()]
    prox = P.load_proxy()
    d1 = json.loads(D1_UNITS.read_text(encoding="utf-8"))["index_results"]
    v1 = json.loads(STOCK_V1.read_text(encoding="utf-8"))["index_results"]
    params = {"batch_usd_per_min": 0.0495}
    rep = SC.build_report(rows, prox, v1, v1, d1, {"artifact": "art-bm3e6"}, params, 1.0, B=200)
    assert rep["child_minus_parent"]["point"] == 0 and rep["child_minus_reference"]["point"] < -2.5
    assert rep["verdict"] == "DROP" and SC.valid_result(rep)
    rep = SC.build_report(rows, prox, d1, v1, d1, {"artifact": "art-bm3e6"}, params, 1.0, B=200)
    assert rep["verdict"] == "PROMOTE" and rep["child_minus_parent"]["point"] > 0 and SC.valid_result(rep)
    assert not SC.valid_result(dict(rep, verdict="DROP"))
    with pytest.raises(SC.ScreenError):  # a reference that does not cover every proxy row fails closed
        SC.build_report(rows, prox, d1, v1, {k: v for k, v in list(d1.items())[1:]}, {}, params, 1.0, B=50)


def test_runner_wiring_screen_stage_and_default_unchanged():
    assert 'st = p.get("stage", "gates")' in JOB and "read -r TRAIN_SUB TEACHER EVAL_ONLY STAGE" in JOB
    assert 'VALIDATOR=jobs.m2_first_cell.screen:valid_result DECISION=$WORK/out/screen.json' in JOB
    i = JOB.index
    assert i("screen reference --params") < i("python -m train.run_cell") < i("-m jobs.m2_first_cell.screen score")
    assert i("-m jobs.m2_first_cell.screen score") < i("-m jobs.m2_first_cell.screen report") < i("LATEST=")
    assert '--stage g12+index' in JOB  # the default (gates) path is still there, unchanged
