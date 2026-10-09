"""Stage ``screen`` (jobs/m2_first_cell/screen.py): the screen readout, reference / parent presets and their pinning
(shipped / downloaded artifact chain, fail closed), readout-mismatch refusal, the DROP / PROMOTE rule, the frozen-scorer
report on real proxy rows, and the runner wiring."""
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
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\m2")
V1, V3W = "jrt-rwkv-g1-v1", "jrt-rwkv-g1-v3w"
D1_CFG = {"template": V1, "layout": "ordered"}


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


def _chain(tmp_path, units: bytes, inner="out/units-child-index.json"):
    inner_tgz = _tar_bytes({inner: units}, gz=True)
    outer = _tar_bytes({"./phases/final.tgz": inner_tgz})
    art = tmp_path / "art.tar"
    art.write_bytes(outer)
    ref = dict(SC.REFERENCES["d1_v1"], artifact="art-test", artifact_sha256=_sha(outer), member_sha256=_sha(inner_tgz),
               inner_member=inner, units_sha256=_sha(units), shipped="screen_reference/x/" + Path(inner).name)
    return ref, art


def test_screen_readout_native_named_and_unknown():
    assert SC.screen_readout({}, D1_CFG) == {"name": "native", "template": V1, "layout": "ordered", "native": True,
                                             "reference_match": "exact", "family": V1}
    r = SC.screen_readout({"screen_readout": "v3w"}, D1_CFG)
    assert (r["template"], r["layout"], r["native"]) == (V3W, "g1v2", False)
    with pytest.raises(SC.ScreenError) as e:
        SC.screen_readout({"screen_readout": "v9"}, D1_CFG)
    assert e.value.code == 5


def test_presets_defaults_by_readout_and_fail_closed_on_readout_mismatch():
    assert SC.reference_block({}, V1)["artifact"] == "art-bm3e6"  # the D1 v1 preset stays the v1 default
    assert SC.reference_block({}, V3W)["artifact"] == "art-zskmx"  # v3w screens default to D1 + v3w
    assert SC.reference_block({"screen_reference": "d1_v1"}, V1)["preset"] == "d1_v1"
    with pytest.raises(SC.ScreenError, match="!= the screen readout") as e:
        SC.reference_block({"screen_reference": "d1_v1"}, V3W)  # a v1 reference on a v3w screen: refused
    assert e.value.code == 10
    with pytest.raises(SC.ScreenError, match="!= the screen readout"):
        SC.reference_block({"screen_reference": "d1_v3w"}, V1)
    with pytest.raises(SC.ScreenError):
        SC.reference_block({"screen_reference": dict(SC.REFERENCES["d1_v1"], units_sha256="nope")}, V1)
    # parent: in-job when its template is the screen's; v3w defaults to the pinned stock v3w family
    assert SC.parent_block({}, V1, V1) is None and SC.parent_block({}, V1, None) is None
    assert SC.parent_block({}, V3W, V1)["artifact"] == "art-ivu28"
    assert SC.parent_block({}, V3W, None)["artifact"] == "art-ivu28"
    with pytest.raises(SC.ScreenError, match="in-job parent arm readout"):
        SC.parent_block({"screen_parent": "in_job"}, V3W, V1)


def test_preset_pins_match_the_staged_copies():
    for blk in (SC.REFERENCES["d1_v1"], SC.REFERENCES["d1_v3w"], SC.PARENTS["stock_v3w"]):
        assert len(blk["units_sha256"]) == 64
    staged = {"d1_v3w": DATA / "screen_reference/art-zskmx/child.v3w.results.jsonl",
              "stock_v3w": DATA / "screen_parent/art-ivu28/v3w.results.jsonl"}
    for k, p in staged.items():
        if p.is_file():
            blk = SC.REFERENCES.get(k) or SC.PARENTS[k]
            assert _sha(p.read_bytes()) == blk["units_sha256"]
            assert blk["shipped"] == p.relative_to(DATA).as_posix()


def test_arm_template_from_revision():
    rev = ("rwkv7:RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854ce;w=0359db5847e4;tok=1c0bc8e5f355;prompt=jrt-rwkv-g1-v1."
           "e4125a57;cfg=fa8822f4;adapter=0.2.0")
    assert SC.arm_template({"revision": rev}) == V1 and SC.arm_template({}) is None


def test_reference_shipped_then_downloaded_chain_and_fail_closed(tmp_path, monkeypatch):
    units = json.dumps({"index_results": {"a": {"status": "ok"}}}).encode()
    ref, art = _chain(tmp_path, units)
    data = tmp_path / "data"
    monkeypatch.delenv("REF_URL", raising=False)
    with pytest.raises(SC.ScreenError, match="REF_URL"):
        SC.locate_reference(ref, data, tmp_path / "w")  # nothing shipped, no URL: fail closed
    rec = SC.locate_reference(ref, data, tmp_path / "w", url=art.as_uri())  # artifact -> member -> inner, all pinned
    assert rec["how"] == "downloaded" and Path(rec["path"]).read_bytes() == units
    assert SC.read_predictions(rec["path"]) == {"a": {"status": "ok"}}
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


def test_jsonl_predictions_chain(tmp_path):
    lines = b'{"run_id":"a","status":"ok"}\n{"run_id":"b","status":"ok"}\n'
    ref, art = _chain(tmp_path, lines, inner="out/readout_ab/child.v3w.results.jsonl")
    rec = SC.locate_reference(ref, tmp_path / "d", tmp_path / "w", url=art.as_uri())
    assert set(SC.read_predictions(rec["path"])) == {"a", "b"}
    dup = tmp_path / "dup.jsonl"
    dup.write_bytes(lines + b'{"run_id":"a","status":"ok"}\n')
    with pytest.raises(SC.ScreenError, match="duplicate"):
        SC.read_predictions(dup)


def test_locate_all_refuses_a_readout_this_code_cannot_render(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "template_available", lambda t: False)
    with pytest.raises(SC.ScreenError) as e:
        SC.locate_all({"screen_readout": "v3w"}, D1_CFG, tmp_path, tmp_path / "w")
    assert e.value.code == 5


@pytest.mark.skipif(not (DATA / "screen_reference/art-zskmx/child.v3w.results.jsonl").is_file(),
                    reason="staged screen references not present")
def test_locate_all_v3w_defaults_cover_the_proxy(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "template_available", lambda t: True)  # the v3w templates live on the readout branches
    rec = SC.locate_all({"screen_readout": "v3w"}, D1_CFG, DATA, tmp_path / "w")
    assert rec["readout"]["template"] == V3W and rec["reference"]["artifact"] == "art-zskmx"
    assert rec["parent"]["source"] == "pinned" and rec["parent"]["artifact"] == "art-ivu28"
    assert rec["proxy_rows_covered"] == 644 and rec["reference"]["how"] == rec["parent"]["how"] == "shipped"


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
    ro = {"name": "native", "template": V1, "layout": "ordered", "native": True}
    rid, pid = {"artifact": "art-bm3e6", "readout_template": V1}, {"source": "in_job", "readout_template": V1}
    rep = SC.build_report(rows, prox, v1, v1, d1, rid, params, 1.0, B=200, readout=ro, parent_ident=pid)
    assert rep["child_minus_parent"]["point"] == 0 and rep["child_minus_reference"]["point"] < -2.5
    assert rep["verdict"] == "DROP" and SC.valid_result(rep)
    rep = SC.build_report(rows, prox, d1, v1, d1, rid, params, 1.0, B=200, readout=ro, parent_ident=pid)
    assert rep["verdict"] == "PROMOTE" and rep["child_minus_parent"]["point"] > 0 and SC.valid_result(rep)
    assert not SC.valid_result(dict(rep, verdict="DROP"))
    assert not SC.valid_result(dict(rep, parent_identity={"source": "pinned", "readout_template": V3W}))
    with pytest.raises(SC.ScreenError):  # a reference that does not cover every proxy row fails closed
        SC.build_report(rows, prox, d1, v1, {k: v for k, v in list(d1.items())[1:]}, rid, params, 1.0, B=50)


def test_runner_wiring_screen_stage_and_default_unchanged():
    assert 'st = p.get("stage", "gates")' in JOB and "read -r TRAIN_SUB TEACHER EVAL_ONLY STAGE" in JOB
    assert 'VALIDATOR=jobs.m2_first_cell.screen:valid_result DECISION=$WORK/out/screen.json' in JOB
    assert 'screen reference --params "$PARAMS" --cell-config "$CELL"' in JOB
    i = JOB.index
    assert i("screen reference --params") < i("python -m train.run_cell") < i("-m jobs.m2_first_cell.screen score")
    assert i("-m jobs.m2_first_cell.screen score") < i("-m jobs.m2_first_cell.screen report") < i("LATEST=")
    block = JOB[i("-m jobs.m2_first_cell.screen score"):i("-m jobs.m2_first_cell.screen report")]
    assert '--reference "$WORK/out/screen-reference.json"' in block  # the child scores under the located readout
    assert '--stage g12+index' in JOB  # the default (gates) path is still there, unchanged
