"""train.m2_gates (G12/G13 + promotion), jobs/m2_parent_arm (cached parent units) and the Q9-AF loader, on tiny CPU
models. Numbers from tiny models mean nothing; these tests check plumbing, identities and fail-closed behaviour."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

TEMPLATE = "jrt-plain-v1"


def _kit_or_skip():
    from scripts.m2_e2e_dryrun import ensure_kit

    try:
        return ensure_kit()
    except ImportError:
        pytest.skip("Decision Index kit not importable")


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A mini cohort (toy decision records + real safety_eval dev requests), Index-shaped synthetic DEV rows."""
    _kit_or_skip()
    from scripts.m2_e2e_dryrun import synthetic_dev_rows
    from train import cohort as C
    from train.data import toy_records

    d = tmp_path_factory.mktemp("world")
    recs = {r["id"]: r for r in toy_records(18, seed=3)}
    for r in recs.values():  # toy states repeat; make content unique so roles are content-disjoint (as cohort v2 is)
        r["request"]["state"] += f"; ref={r['id']}"
    ids = sorted(recs)
    roles = {"dev": ids[:6], "calibration": ids[6:12], "retention": ids[12:18]}
    steer = C.steer_requests(3, 2)
    header = {"cohort": "test-mini", "sizes": {"dev": 6, "calibration": 6, "retention": 6, "steer_dev": 3,
                                               "steer_gate": 2}, "sizing": None}
    lines = [json.dumps({"header": header}, sort_keys=True)]
    for role, rids in roles.items():
        for i in rids:
            lines.append(json.dumps({"role": role, "id": i, "sha256": hashlib.sha256(i.encode()).hexdigest(),
                                     "source": "toy"}, sort_keys=True))
    for role in ("steer_dev", "steer_gate"):
        for e in steer[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    p = d / "cohort.jsonl.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(text.encode())
    cohort = C.Cohort(p, expected_sha256=hashlib.sha256(text.encode()).hexdigest())
    return {"dir": d, "cohort": cohort, "records": recs, "roles": roles, "index_rows": synthetic_dev_rows(3)}


def _decider(seed=0):
    from decide import Decider
    from heads import DecisionModel
    from heads.backbone import tiny_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder

    model = DecisionModel(tiny_backbone(seed=seed, hidden=16), "integrator", head_kwargs={"channels": 8}, seed=seed)
    return Decider(DecisionModelBackend(model, ByteEncoder(), template=TEMPLATE)), model


def _items(world, spec):
    from heads.layout import ByteEncoder
    from train import m2_gates as G
    from train.guard import SplitGuard

    g = SplitGuard.synthetic_only("toy records")
    return {r: G.decision_items([world["records"][i] for i in ids], ByteEncoder(), spec, g)
            for r, ids in world["roles"].items()}


SPEC = {"max_k": 16, "max_t": 100000, "template": TEMPLATE, "layout": "ordered"}


def _cache_parent(world, out, decide, steer, stages=("g12", "g13"), spec=SPEC, ident=None):
    from jobs.m2_parent_arm.score_parent import score_parent_arm

    return score_parent_arm(stages=stages, cohort=world["cohort"], spec=spec, decide=decide, steer=steer,
                            items_by_role=_items(world, spec), records_by_id=world["records"], out_dir=out,
                            parent_identity=ident or {"parent": {"parent": "tiny"}, "revision": "r0"},
                            index_rows=world["index_rows"])


def _gates(world, cache, decide, steer, stages=("g12", "g13"), out=None, expect_parent=None):
    from train import m2_gates as G

    return G.run_gates(stages=stages, cohort=world["cohort"], spec=SPEC, child_decide=decide, child_steer=steer,
                       child_identity={"revision_id": "child"}, items_by_role=_items(world, SPEC),
                       records_by_id=world["records"], parent_arm=cache, expect_parent=expect_parent,
                       index_rows=world["index_rows"], n_boot=200, out=out)


@pytest.fixture(scope="module")
def parent_cache(world):
    from safety_eval.standin import KeywordDecider

    dec, _ = _decider(0)
    cache = world["dir"] / "parent"
    _cache_parent(world, cache, dec.decide, KeywordDecider(gullibility=0.0))
    return cache, dec


# ----------------------------------------------------------------------------------------------- m2_gates
def test_identical_child_accepts_both_stages_but_is_not_promotable_with_pending_margins(world, parent_cache, tmp_path):
    from safety_eval.standin import KeywordDecider
    from train import m2_gates as G

    cache, dec = parent_cache
    code, rep = _gates(world, cache, dec.decide, KeywordDecider(gullibility=0.0), out=tmp_path / "report.json")
    assert code == 0, rep["promotion"]
    assert rep["promotion"]["verdict"] == "accept" and rep["promotion"]["margins_pending_user_approval"]
    assert rep["promotion"]["promotable"] is False
    crit = {c["criterion"] for s in rep["verdicts"].values() for c in s["criteria"]}
    assert {"g12_violation", "g12_steer_d1a", "g12_clean_acc", "g13_steer_d1b", "g13_capability", "g13_ece",
            "g13_retention_acc", "g13_retention_ece"} <= crit
    on_disk = json.loads((tmp_path / "report.json").read_text())
    assert G.valid_report(on_disk) and (tmp_path / "units-child-g13.json").exists()
    assert on_disk["stages"]["g13"]["units_identity"]["index_rows"]["n"] == len(world["index_rows"])


def test_more_gullible_child_is_rejected_at_g12(world, parent_cache):
    from safety_eval.standin import KeywordDecider

    cache, dec = parent_cache
    code, rep = _gates(world, cache, dec.decide, KeywordDecider(gullibility=6.0), stages=("g12",))
    assert code == 40 and "g12_steer_d1a" in rep["promotion"]["failed"]["g12"]


def test_degraded_decisions_reject_g12_accuracy_and_g13_retention(world, parent_cache, tmp_path):
    from safety_eval.standin import KeywordDecider

    _, dec = parent_cache
    gold = {json.dumps(r["request"]["state"]): r["labels"] for r in world["records"].values()}

    def informed(req):  # the parent puts most served mass on the gold option of every cohort question
        resp = dec.decide(req)
        labels = gold.get(json.dumps(req["state"]))
        for q, p in resp["ssdm"]["served_probabilities"].items():
            if labels and q in labels:
                g = str(labels[q]).lower() if str(labels[q]).lower() in p else str(labels[q])
                resp["ssdm"]["served_probabilities"][q] = {o: (0.9 if o == g else 0.1 / max(1, len(p) - 1)) for o in p}
        return resp

    cache = tmp_path / "informed"
    _cache_parent(world, cache, informed, KeywordDecider(gullibility=0.0))
    code, rep = _gates(world, cache, dec.decide, KeywordDecider(gullibility=0.0))
    f = rep["promotion"]["failed"]
    assert code == 42 and "g12_clean_acc" in f["g12"]
    assert {"g13_retention_acc", "g13_retention_ece"} & set(f["g13"])
    acc = rep["verdicts"]["g12"]["paired"]["dev"]
    assert acc["acc_parent"] > acc["acc_child"]


def test_a_failed_unit_fails_closed(world, parent_cache):
    from safety_eval.standin import KeywordDecider

    cache, dec = parent_cache
    bad = sorted(world["roles"]["dev"])[0]

    def flaky(req):
        resp = dec.decide(req)
        if req["state"] == world["records"][bad]["request"]["state"]:
            resp["ssdm"]["status"] = "error"
        return resp

    code, rep = _gates(world, cache, flaky, KeywordDecider(gullibility=0.0), stages=("g12",))
    assert code == 40 and "g12_clean_acc" in rep["promotion"]["failed"]["g12"]
    crit = next(c for c in rep["verdicts"]["g12"]["criteria"] if c["criterion"] == "g12_clean_acc")
    assert crit["evaluable"] is False
    assert rep["stages"]["g12"]["child_decision_errors"]["dev"]


def test_parent_cache_must_match_units_parent_and_hash(world, parent_cache, tmp_path):
    import shutil

    from safety_eval.standin import KeywordDecider
    from train import m2_gates as G

    cache, dec = parent_cache
    with pytest.raises(G.IdentityMismatch, match="parent arm identity"):
        _gates(world, cache, dec.decide, KeywordDecider(0.0), stages=("g12",), expect_parent={"parent": "rwkv7"})
    other = tmp_path / "other"
    _cache_parent(world, other, dec.decide, KeywordDecider(0.0), stages=("g12",), spec={**SPEC, "max_k": 3})
    with pytest.raises(G.IdentityMismatch, match="unit_spec"):
        _gates(world, other, dec.decide, KeywordDecider(0.0), stages=("g12",))
    with pytest.raises(G.IdentityMismatch, match="no g13 stage"):
        _gates(world, other, dec.decide, KeywordDecider(0.0), stages=("g13",))
    tampered = tmp_path / "tampered"
    shutil.copytree(cache, tampered)
    f = tampered / "units-g12.json"
    f.write_text(f.read_text().replace('"correct": 1.0', '"correct": 0.0', 1))
    with pytest.raises(G.IdentityMismatch, match="drifted"):
        _gates(world, tampered, dec.decide, KeywordDecider(0.0), stages=("g12",))
    with pytest.raises(G.IdentityMismatch, match="no parent arm cache"):
        _gates(world, tmp_path / "none", dec.decide, KeywordDecider(0.0), stages=("g12",))


def test_g13_variants_are_held_out_d1b_and_g12_never_sees_them(world):
    from safety_eval import attacks
    from train import m2_gates as G

    v12, v13 = G.stage_variants(world["cohort"], "g12"), G.stage_variants(world["cohort"], "g13")
    t12 = {v["_meta"]["template"] for v in v12 if v["_meta"]["attack_family"] == "d1"}
    t13 = {v["_meta"]["template"] for v in v13 if v["_meta"]["attack_family"] == "d1"}
    assert t12 == set(attacks.D1_TEMPLATES_A) and t13 == set(attacks.D1_TEMPLATES_B)
    assert not ({v["request_id"].split("::")[0] for v in v12} & {v["request_id"].split("::")[0] for v in v13})


def test_exit_codes_and_report_validator():
    from train import m2_gates as G

    ok, bad = {"accept": True, "failed": []}, {"accept": False, "failed": ["x"]}
    assert G.exit_code({"g12": ok, "g13": ok}) == 0
    assert G.exit_code({"g12": bad, "g13": ok}) == 40 and G.exit_code({"g12": ok, "g13": bad}) == 41
    assert G.exit_code({"g12": bad, "g13": bad}) == 42
    assert not G.valid_report({}) and not G.valid_report({"schema": G.REPORT_SCHEMA})
    rep = {"schema": G.REPORT_SCHEMA, "identities": {}, "verdicts": {"g12": ok},
           "promotion": G.promotion_verdict({"g12": ok})}
    assert G.valid_report(rep) and rep["promotion"]["promotable"] is False  # g13 not run + margins pending
    rep["promotion"]["verdict"] = "reject"
    assert not G.valid_report(rep)


def test_cli_usage_errors():
    from train import m2_gates as G

    assert G.main(["--cell", "x", "--ckpt-dir", "x", "--model-dir", "x", "--records", "x", "--guard-readiness", "x",
                   "--parent-arm", "x", "--out", "x"]) == G.EXIT["usage"]  # g13 without --index-rows


def test_served_child_uses_the_fitted_temperature(world):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from train import m2_gates as G

    _, model = _decider(1)
    items = _items(world, SPEC)
    ft = G.fit_child_temperature(model, items["calibration"], {"eval": items["dev"], "retention": items["retention"]})
    dec = Decider(DecisionModelBackend(model, ByteEncoder(), template=TEMPLATE), calibration=ft.to_serving())
    assert dec.revision()["revision_id"].endswith("+T:" + ft.identity()["id"])
    from train.calibration import item_content

    ft.to_serving().verify_disjoint("eval", item_content(items["dev"]))
    from train.calibration import CalibrationLeak

    with pytest.raises(CalibrationLeak):
        G.fit_child_temperature(model, items["calibration"], {"eval": items["calibration"][:1]})


# ----------------------------------------------------------------------------------------------- parent arm job
def test_parent_arm_estimate_and_cap():
    from jobs.m2_parent_arm import score_parent as S

    e = S.estimate(5000.0)
    assert e["tokens"]["g12"] == 3_487_676 + 5_378_738
    assert e["total_minutes"] == pytest.approx(sum(e["tokens"].values()) / 5000 / 60 + S.FIXED_MIN)
    assert S.main(["estimate", "--tok-s", "5000", "--cap-min", "180"]) == 0
    assert S.main(["estimate", "--tok-s", "1000", "--cap-min", "180"]) == 15
    with pytest.raises(ValueError):
        S.estimate(0)
    params = json.loads((Path(__file__).resolve().parents[1] / "jobs/m2_parent_arm/job_params.json").read_text())
    worst = (params["build_timeout_minutes"] + 2 * params["max_duration_minutes"]) * params["batch_usd_per_min"]
    assert worst <= params["hard_cap_usd"] <= 20.0


def test_parent_arm_job_scores_before_nothing_trains():
    s = (Path(__file__).resolve().parents[1] / "jobs/m2_parent_arm/parent_arm_job.sh").read_text(encoding="utf-8")
    assert s.index("jobs.m2_pilot.guard_ready") < s.index("jobs.m2_parent_arm.score_parent estimate") < \
        s.index("-m jobs.m2_parent_arm.score_parent --parent")
    assert "train.run_cell" not in s and "--slice test" not in s


# ----------------------------------------------------------------------------------------------- Q9-AF loader
def _stock_and_converted(seed_stock=0, seed_gdn=1):
    from heads.backbone import tiny_hf_backbone

    stock = tiny_hf_backbone("qwen3.5", seed=seed_stock)
    donor = tiny_hf_backbone("q9af", seed=seed_gdn)
    attn = [i for i, t in enumerate(stock.model.model.config.layer_types) if t == "full_attention"]
    sd = {i: {k: v.clone() for k, v in donor.model.model.layers[i].linear_attn.state_dict().items()} for i in attn}
    return stock, sd, attn


def test_q9af_from_stock_is_an_all_gdn_model_equal_to_the_reference():
    from heads.backbone import HiddenStateBackbone, q9af_from_stock, tiny_hf_backbone

    stock, sd, attn = _stock_and_converted()
    text = q9af_from_stock(stock.model.model, sd, {"kind": "q9af"})
    assert set(text.config.layer_types) == {"linear_attention"}
    assert all(not hasattr(text.layers[i], "self_attn") and text.layers[i].block_type == "linear_attention"
               for i in attn)
    ref = tiny_hf_backbone("q9af", seed=1).model.model
    ref.load_state_dict(text.state_dict(), strict=True)
    ids = torch.randint(5, 250, (1, 24))
    bb = HiddenStateBackbone(text, spec=stock.spec)
    assert torch.allclose(bb.features(ids), HiddenStateBackbone(ref).features(ids), atol=1e-6)
    prefix, conts = list(range(5, 29)), [[30, 31], [40], [50, 51, 52]]
    bb.fork_mode = "cache"
    _, fork = bb.fork_features(prefix, conts)
    bb.fork_mode = "recompute"
    rec = torch.stack([bb.features(torch.tensor([prefix + c]))[0, -1] for c in conts])
    assert torch.allclose(fork, rec, atol=1e-5)


def test_q9af_refuses_partial_conversion_and_records_provenance_in_lineage():
    from heads.backbone import HiddenStateBackbone, LineageError, q9af_from_stock
    from train.lineage import backbone_identity

    stock, sd, attn = _stock_and_converted()
    with pytest.raises(LineageError, match="attention layers"):
        q9af_from_stock(stock.model.model, {}, None)
    stock2, sd2, _ = _stock_and_converted()
    text = q9af_from_stock(stock2.model.model, sd2)
    bb = HiddenStateBackbone(text, spec=stock2.spec)
    plain = backbone_identity(bb)
    assert "provenance" not in plain
    bb.provenance = {"converted_sha256": "a" * 64}
    assert backbone_identity(bb)["provenance"] == {"converted_sha256": "a" * 64}


def test_q9af_pin_check(tmp_path):
    from heads.backbone import Q9AF_BASE_REPO, Q9AF_BASE_REVISION, LineageError, q9af_pin_check

    ck = tmp_path / "gdn.pt"
    torch.save({3: {"x": torch.zeros(1)}}, ck)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    pin = {"converted_sha256": sha, "base_repo": Q9AF_BASE_REPO, "base_revision": Q9AF_BASE_REVISION}
    (tmp_path / "config.json").write_text("{}")
    prov = q9af_pin_check(pin, ck, tmp_path)
    assert prov["converted_sha256"] == sha and len(prov["base_config_sha256"]) == 64
    for bad, msg in (({**pin, "converted_sha256": None}, "not pinned"), ({**pin, "converted_sha256": "0" * 64}, "!="),
                     ({**pin, "base_revision": "x"}, "base"), ({**pin, "base_config_sha256": "0" * 64}, "config")):
        with pytest.raises(LineageError, match=msg):
            q9af_pin_check(bad, ck, tmp_path)


def test_run_cell_refuses_unpinned_q9af_before_loading_anything(tmp_path):
    from train.run_cell import EXIT, main, parent_encoder

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "jobs/m2_pilot/m2_pilot_cell_q9af.json").read_text())
    assert cfg["parent"] == "q9af" and cfg["template"] == "jrt-chatml-v1"
    assert cfg["q9af"]["converted_sha256"] is None  # PENDING: the full-8 conversion has not run
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    args = ["--config", str(tmp_path / "c.json"), "--model-dir", str(tmp_path / "nomodel"), "--records", "x",
            "--out", str(tmp_path / "o"), "--ckpt-dir", str(tmp_path / "k")]
    assert main(args) == EXIT["lineage"]
    ck = tmp_path / "gdn.pt"
    torch.save({}, ck)
    assert main(args + ["--gdn-ckpt", str(ck)]) == EXIT["lineage"]
    with pytest.raises(ValueError):
        parent_encoder("mamba", "x")
