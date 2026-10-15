"""jobs/clef_baseline JOB=H (JOB_SPEC 13): baselines.clef_hijack_matched on a mini cohort with the mocked Clef (CPU, no
weights). Plumbing, pins and fail-closed behaviour: exact unit ids / unit_sha, the art-4naug sha256 pin and layout, the
D1 reproduction check, and the pre-declared eligibility."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from baselines import clef_backend as CB  # noqa: E402
from baselines import clef_hijack_matched as H  # noqa: E402
from clef_mocks import FakeTok, MockProcessor, build_mock_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    """A mini cohort (steer roles only), as tests/test_m2_hijack_eval.py builds it."""
    from train import cohort as C

    d = tmp_path_factory.mktemp("hm")
    steer = C.steer_requests(6, 2)
    header = {"cohort": "test-mini-hijack", "sizes": {"dev": 0, "calibration": 0, "retention": 0, "steer_dev": 6,
                                                      "steer_gate": 2}, "sizing": None}
    lines = [json.dumps({"header": header}, sort_keys=True)]
    for role in ("steer_dev", "steer_gate"):
        for e in steer[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    p = d / "cohort.jsonl.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(text.encode())
    return str(p), hashlib.sha256(text.encode()).hexdigest()


def _mock(monkeypatch, seed):
    jsm = CB.load_jsm()
    model = build_mock_model(jsm, seed=seed)

    def fake_load(model_dir, release="clef-flash", *, encoding="ssdm", device="cpu", dtype="float32", **kw):
        be = CB.ClefBackend(model, FakeTok(), release=release, encoding=encoding, jsm=jsm, device="cpu",
                            dtype="float32", files_sha256={"note": "mock"})
        return be, model, MockProcessor(FakeTok())

    monkeypatch.setattr(CB, "load_clef", fake_load)
    monkeypatch.setenv("CLEF_DRY_RUN", "1")


@pytest.fixture(scope="module")
def scored(tmp_path_factory, cohort):
    """units-clef-flash.jsonl / units-clef.jsonl from two mock models, plus a D1 dir from a third and a fourth."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("scored")
    try:
        for rel, seed in (("clef-flash", 4), ("clef", 5)):
            _mock(mp, seed)
            ident = H.score("unused", rel, d, device="cpu", dtype="float32", verify=False, cohort_path=cohort[0],
                            cohort_sha256=cohort[1], log=lambda m: None)
            assert ident["gdn_path"] == "fast" and ident["variants"]["n_units"] > 0
        d1 = d / "d1"
        d1.mkdir()
        for k, seed in (("child", 6), ("parent", 7)):
            _mock(mp, seed)
            H.score("unused", "clef-flash", d / k, device="cpu", dtype="float32", verify=False, cohort_path=cohort[0],
                    cohort_sha256=cohort[1], log=lambda m: None)
            rows = H.read_units(d / k / "units-clef-flash.jsonl")
            (d1 / f"d1-{k}.json").write_text(json.dumps({"steer_rows": rows}), encoding="utf-8")
    finally:
        mp.undo()
    return d


def test_production_unit_set_is_job_j8wgy_shape_and_pinned():
    v = H.expected_variants()
    ident = H.check_production_variants(v)
    assert ident == {"n_units": 3600, "by_family": {"clean": 360, "hijack": 1800, "d1": 1440}, "n_clusters": 45,
                     "request_ids_sha256": H.VARIANTS_SHA256}
    from safety_eval import attacks

    assert {x["_meta"]["template"] for x in v if x["_meta"]["attack_family"] == "hijack"} == set(attacks.HIJACK_TEMPLATES)
    assert {x["_meta"]["template"] for x in v if x["_meta"]["attack_family"] == "d1"} == set(attacks.D1_TEMPLATES_A)


def test_compare_pairs_with_pre_declared_eligibility(scored, cohort):
    units = {r: scored / f"units-{r}.jsonl" for r in H.RELEASES}
    out = H.compare(scored / "d1", units, n_boot=50, cohort_path=cohort[0], cohort_sha256=cohort[1], declared=None)
    assert out["complete"] is True and out["eligibility"]["primary"] == "both"
    for rel in H.RELEASES:
        for f in H.FAMILIES:
            b = out["paired_vs_d1"][rel][f]
            assert set(b) >= {"both|two_way", "both|fixed_template", "reference_y|two_way", "reference_y|fixed_template"}
            assert b["both|two_way"]["eligibility"] == "both"
            assert b["both|two_way"]["n_eligible"] <= b["reference_y|two_way"]["n_eligible"]
            hl = out["headline"][rel][f]
            assert set(H.REPORTED) <= set(hl) and "clean_acc_model" in hl and "n_eligible_both" in hl
        own = out["per_model"][rel]
        assert {"clean", "hijack", "d1_A"} <= set(own) and own["hijack"]["complete"] is True
    assert set(out["per_model"]) == {"d1", "rwkv7_parent", *H.RELEASES}


def test_compare_fails_closed_on_unit_ids(scored, cohort, tmp_path):
    rows = (scored / "units-clef.jsonl").read_text(encoding="utf-8").splitlines()
    cut = tmp_path / "units-clef.jsonl"
    cut.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")  # one unit missing
    args = ["compare", "--d1-dir", str(scored / "d1"), "--units", str(scored / "units-clef-flash.jsonl"), str(cut),
            "--out", str(tmp_path / "s.json"), "--n-boot", "20", "--cohort", cohort[0], "--cohort-sha256", cohort[1],
            "--skip-declared-check"]
    assert H.main(args) == 21 and not (tmp_path / "s.json").exists()
    r0 = json.loads(rows[0])
    r0["unit_sha"] = "0" * 64  # same ids, different content
    cut.write_text("\n".join([json.dumps(r0)] + rows[1:]) + "\n", encoding="utf-8")
    assert H.main(args) == 21
    cut.write_text("\n".join(rows + rows[:1]) + "\n", encoding="utf-8")  # a duplicate unit
    assert H.main(args) == 21
    d1 = tmp_path / "d1"
    d1.mkdir()
    child = json.loads((scored / "d1" / "d1-child.json").read_text(encoding="utf-8"))["steer_rows"]
    (d1 / "d1-child.json").write_text(json.dumps({"steer_rows": child[:-1]}), encoding="utf-8")
    (d1 / "d1-parent.json").write_bytes((scored / "d1" / "d1-parent.json").read_bytes())
    cut.write_text("\n".join(rows) + "\n", encoding="utf-8")
    args[2] = str(d1)
    assert H.main(args) == 21  # D1 itself must cover the expected set


def test_d1_reproduction_check_refuses_other_numbers(scored, cohort):
    v = H.expected_variants(cohort[0], cohort[1])
    child, parent, _, _ = H.load_d1(scored / "d1")
    rep = H.d1_reproduction(child, parent, v, 20, declared=None)
    x, y = rep["rates_two_way"]["hijack"]["hijack_rate"]
    assert H.d1_reproduction(child, parent, v, 20, declared={"hijack": {"hijack_rate": (x, y)}})["complete"]
    with pytest.raises(H.UnitSetMismatch):
        H.d1_reproduction(child, parent, v, 20, declared={"hijack": {"hijack_rate": (x + 0.01, y)}})
    with pytest.raises(H.UnitSetMismatch):
        H.d1_reproduction(child, parent, v, 20, declared=None,
                          result={"hijack": {"hijack_rate": {"child": x + 0.5, "parent": y}}})
    assert H.D1_DECLARED["hijack"]["hijack_rate"] == (0.0, 0.349)
    assert H.D1_DECLARED["d1_A"]["attacked_acc"] == (0.992, 0.661)


def _artifact(tmp_path, scored, nested=True) -> Path:
    """An art-4naug-shaped artifact: a tar of /output whose phases/final.tgz holds out/ (lib.sh export_phase)."""
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as t:
        for k, member in (("child", H.D1_MEMBERS["child"]), ("parent", H.D1_MEMBERS["parent"])):
            b = (scored / "d1" / f"d1-{k}.json").read_bytes()
            ti = tarfile.TarInfo(member)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    art = tmp_path / "art.tar"
    with tarfile.open(art, "w") as t:
        for name, b in (("./phases/evals.tgz", b"not the newest"), ("./phases/final.tgz", inner.getvalue()),
                        ("./MANIFEST.jsonl", b"{}\n")):
            if name.endswith("evals.tgz"):
                e = io.BytesIO()
                with tarfile.open(fileobj=e, mode="w:gz"):
                    pass
                b = e.getvalue()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    return art


def test_fetch_d1_checks_the_sha256_pin_and_finds_the_phase_tarball(scored, tmp_path, monkeypatch):
    art = _artifact(tmp_path, scored)
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    with pytest.raises(H.ReferenceError_, match="sha256"):
        H.fetch_d1(tmp_path / "a", artifact=art)  # the real art-4naug pin
    monkeypatch.setitem(H.D1_ARTIFACT, "sha256", sha)
    ident = H.fetch_d1(tmp_path / "b", url=art.resolve().as_uri())
    assert ident["members_from"].endswith("phases/final.tgz") and ident["artifact_sha256_verified"] == sha
    assert (tmp_path / "b" / "d1-child.json").read_bytes() == (scored / "d1" / "d1-child.json").read_bytes()
    monkeypatch.delenv("D1_REF_URL", raising=False)
    with pytest.raises(H.ReferenceError_, match="D1_REF_URL"):
        H.fetch_d1(tmp_path / "c")
    monkeypatch.setenv("D1_REF_URL", "<get_artifact art-4naug url: fill at submit>")
    with pytest.raises(H.ReferenceError_):
        H.fetch_d1(tmp_path / "d")


def test_params_and_pins_agree():
    p = json.loads((ROOT / "jobs/clef_baseline/job_params_hijack_matched.json").read_text(encoding="utf-8"))
    assert p["d1_reference"]["sha256"] == H.D1_ARTIFACT["sha256"] and p["gdn_path"] == "fast"
    assert p["submit"]["env"] == {"RELEASE": "both", "JOB": "H", "D1_REF_URL": p["submit"]["env"]["D1_REF_URL"]}
    assert p["submit"]["env"]["D1_REF_URL"].startswith("<")  # filled from the submitting shell, never on disk
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    from jobs.common import costs

    assert costs.worst_case_usd(p) <= 12.0 and costs.worst_case_usd(p) <= p["hard_cap_usd"]
