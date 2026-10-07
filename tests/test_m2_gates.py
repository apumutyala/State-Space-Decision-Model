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


def _gates(world, cache, decide, steer, stages=("g12", "g13"), out=None, expect_parent=None,
           parent_arm_sha256=None, **kw):
    from train import m2_gates as G

    return G.run_gates(stages=stages, cohort=world["cohort"], spec=SPEC, child_decide=decide, child_steer=steer,
                       child_identity={"revision_id": "child"}, items_by_role=_items(world, SPEC),
                       records_by_id=world["records"], parent_arm=cache, expect_parent=expect_parent,
                       index_rows=world["index_rows"], n_boot=200, out=out, parent_arm_sha256=parent_arm_sha256, **kw)


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
    from train.promotion import MARGINS

    pending = {**MARGINS, "approved": False, "status": "PENDING USER APPROVAL"}
    code, rep = _gates(world, cache, dec.decide, KeywordDecider(gullibility=0.0), out=tmp_path / "report.json",
                       margins=pending)
    assert code == G.EXIT["pending"] == 43, rep["promotion"]  # issue #36: never exit 0 while margins are pending
    assert rep["promotion"]["verdict"] == "accept" and rep["promotion"]["margins_pending_user_approval"]
    assert rep["promotion"]["promotable"] is False
    crit = {c["criterion"] for s in rep["verdicts"].values() for c in s["criteria"]}
    assert {"g12_violation", "g12_steer_d1a", "g12_false_alarm", "g12_clean_acc", "g13_steer_d1b", "g13_false_alarm",
            "g13_capability", "g13_ece", "g13_retention_acc", "g13_retention_ece"} <= crit
    assert rep["verdicts"]["g13"]["comparators"]["anchor"]["same_as"] == "last_accepted"
    approved = {**MARGINS, "approved": True, "status": "approved (test)"}
    code2, rep2 = _gates(world, cache, dec.decide, KeywordDecider(gullibility=0.0), margins=approved)
    assert code2 == 0 and rep2["promotion"]["promotable"] is True
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
    assert G.exit_code({"g12": ok, "g13": ok}, pending=False) == 0
    assert G.exit_code({"g12": ok, "g13": ok}, pending=True) == G.EXIT["pending"] == 43  # issue #36
    assert G.exit_code({"g12": bad, "g13": ok}, pending=True) == 40  # a rejection is reported as such
    assert G.exit_code({"g12": bad, "g13": ok}) == 40 and G.exit_code({"g12": ok, "g13": bad}) == 41
    assert G.exit_code({"g12": bad, "g13": bad}) == 42
    assert G.exit_code({}) == G.EXIT["usage"] != 0  # nothing evaluated is never an accept
    assert not G.valid_report({}) and not G.valid_report({"schema": G.REPORT_SCHEMA})
    rep = {"schema": G.REPORT_SCHEMA, "identities": {}, "verdicts": {"g12": ok},
           "promotion": G.promotion_verdict({"g12": ok})}
    assert G.valid_report(rep) and rep["promotion"]["promotable"] is False  # g13 not run
    rep["promotion"]["verdict"] = "reject"
    assert not G.valid_report(rep)


def test_cli_usage_errors():
    from train import m2_gates as G

    assert G.main(["--cell", "x", "--ckpt-dir", "x", "--model-dir", "x", "--records", "x", "--guard-readiness", "x",
                   "--parent-arm", "x", "--parent-arm-sha256", "0" * 64, "--out", "x"]) == G.EXIT["usage"]  # g13, no rows
    with pytest.raises(SystemExit):  # the arm.json pin is mandatory
        G.main(["--cell", "x", "--ckpt-dir", "x", "--model-dir", "x", "--records", "x", "--guard-readiness", "x",
                "--parent-arm", "x", "--out", "x", "--stage", "g12"])


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
    assert e["tokens"]["g12"] == sum(S.VOLUME_TOKENS["g12"].values())
    assert e["total_minutes"] == pytest.approx(sum(e["tokens"].values()) / 5000 / 60 + S.FIXED_MIN)
    assert S.main(["estimate", "--tok-s", "10000", "--cap-min", "180"]) == 0
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
    (tmp_path / "config.json").write_text("{}")
    cfg_sha = hashlib.sha256(b"{}").hexdigest()
    pin = {"converted_sha256": sha, "base_repo": Q9AF_BASE_REPO, "base_revision": Q9AF_BASE_REVISION,
           "base_config_sha256": cfg_sha}
    prov = q9af_pin_check(pin, ck, tmp_path)
    assert prov["converted_sha256"] == sha and prov["base_config_sha256"] == cfg_sha
    for bad, msg in (({**pin, "converted_sha256": None}, "not pinned"), ({**pin, "converted_sha256": "0" * 64}, "!="),
                     ({**pin, "base_revision": "x"}, "base"), ({**pin, "base_config_sha256": "0" * 64}, "config"),
                     ({**pin, "base_config_sha256": None}, "base_config_sha256 is not pinned")):
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


# ----------------------------------------------------------------------------------------------- round-2 review (B4 + nits)
def test_child_evals_preflight_counts_fit_and_card_not_only_the_arm():
    """B4: the m2_pilot evals phase scores the child arm AND forwards every calibration item (temperature fit) and every
    dev item (eval card). Issue #36: G13's Index term is the FULL DEV slice (projected 38.8 M fork positions, was the
    6.4 M subset), so at the 50-min evals cap the child phase needs >= ~23.4k tok/s (was ~12k); below that, G13 runs
    as a separate gate job (JOB_SPEC)."""
    from jobs.m2_parent_arm import score_parent as S
    from train.cohort import Cohort

    assert S.VOLUME_COHORT_SHA256 == Cohort().sha256  # volumes were measured on the committed cohort
    arm = S.estimate(1.0, fixed_min=0, mode="parent")["total_tokens"]
    child = S.estimate(1.0, fixed_min=0, mode="child")["total_tokens"]
    assert child == arm + sum(S.CHILD_EXTRA_TOKENS.values()) and 66e6 < child < 67e6
    need = S.min_tok_s(50)
    assert 23_000 < need < 24_000 and S.min_tok_s(50, mode="parent") < 21_000
    assert S.main(["estimate", "--tok-s", "10000", "--mode", "child", "--cap-min", "50", "--fixed-min", "0"]) == 15
    assert S.main(["estimate", "--tok-s", "21000", "--mode", "parent", "--cap-min", "50", "--fixed-min", "0"]) == 0
    assert S.main(["estimate", "--tok-s", str(round(need) + 1), "--mode", "child", "--cap-min", "50",
                   "--fixed-min", "0"]) == 0
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs/m2_pilot/m2_job.sh").read_text(encoding="utf-8")
    pre = job[job.index("score_parent estimate"):job.index("evals-preflight.json")]
    assert "--mode child" in pre and '--cap-min "$CAP_E"' in pre
    assert job.index("evals-preflight.json") < job.index("python -m train.run_cell")  # before the paid cell phase
    spec = (root / "jobs/m2_pilot/JOB_SPEC.md").read_text(encoding="utf-8")
    assert "~9,000 tok/s" not in spec and "23,400" in spec  # full Index DEV (issue #36), projected fork positions


@pytest.mark.parametrize("runner", ["jobs/m2_pilot/m2_job.sh", "jobs/m2_parent_arm/parent_arm_job.sh"])
def test_runner_exit_codes_tok_s_and_q9af_snapshot(runner):
    s = (Path(__file__).resolve().parents[1] / runner).read_text(encoding="utf-8")
    trap = s.index("trap finish EXIT")
    assert 'fail 4 "TOK_S' in s[trap:] and "${TOK_S:?" not in s  # a clear exit 4 after the trap, never rc 127/1
    assert 'fail 2 "cell' not in s  # exit 2 is only "bad PARENT"
    assert 'sha256sum -c "$CODE/jobs/m2_pilot/q9af_snapshot.sha256"' in s


def test_tok_s_unset_gives_a_clear_exit(tmp_path):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    root = Path(__file__).resolve().parents[1]
    text = (root / "jobs/m2_pilot/m2_job.sh").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if 'fail 4 "TOK_S' in ln)
    script = "set -Euo pipefail\nfail() { echo \"FAIL($1): $2\"; exit \"$1\"; }\nunset TOK_S || true\n" + line + "\necho PAST"
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 4 and "TOK_S" in r.stdout and "PAST" not in r.stdout


def test_m2_pilot_pins_parent_arm_sha_and_checks_it():
    root = Path(__file__).resolve().parents[1]
    params = json.loads((root / "jobs/m2_pilot/job_params.json").read_text())
    assert set(params["parent_arm_sha256"]) >= {"rwkv7", "q9af"}
    job = (root / "jobs/m2_pilot/m2_job.sh").read_text(encoding="utf-8")
    assert "parent_arm_sha256" in job and '--parent-arm-sha256 "$ARM_SHA"' in job
    assert job.index("parent arm.json sha256 != pinned") < job.index("python -m train.run_cell")


def test_arm_json_pin_and_parent_repo_revision_are_enforced(world, parent_cache):
    from safety_eval.standin import KeywordDecider
    from train import m2_gates as G

    cache, dec = parent_cache
    sha = hashlib.sha256((cache / "arm.json").read_bytes()).hexdigest()
    code, _ = _gates(world, cache, dec.decide, KeywordDecider(0.0), stages=("g12",), parent_arm_sha256=sha)
    assert code in (0, G.EXIT["pending"])  # accepted (43 only while the margins are pending)
    with pytest.raises(G.IdentityMismatch, match="pinned"):
        _gates(world, cache, dec.decide, KeywordDecider(0.0), stages=("g12",), parent_arm_sha256="0" * 64)
    exp = G.expected_parent_identity("rwkv7")
    assert exp["repo"] == "RWKV/RWKV7-G1k-13.3B-20260930" and len(exp["revision"]) == 40
    q = G.expected_parent_identity("q9af", {"converted_sha256": "a" * 64})
    assert q["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a" and q["converted_sha256"] == "a" * 64
    with pytest.raises(G.IdentityMismatch):
        G.expected_parent_identity("q9af", {})
    with pytest.raises(G.IdentityMismatch, match="revision"):  # same family, other snapshot -> refused
        _gates(world, cache, dec.decide, KeywordDecider(0.0), stages=("g12",),
               expect_parent={"parent": "tiny", "revision": "other"})


def test_q9af_cell_pins_base_config_and_snapshot_file():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "jobs/m2_pilot/m2_pilot_cell_q9af.json").read_text())["q9af"]
    lines = dict(reversed(ln.split("  ", 1)) for ln in
                 (root / "jobs/m2_pilot/q9af_snapshot.sha256").read_text().splitlines() if ln.strip())
    assert cfg["base_config_sha256"] == lines["config.json"]
    assert sum(f.endswith(".safetensors") for f in lines) == 4 and "tokenizer.json" in lines
    from train.tokenizer_check import PINS

    for f, h in PINS["qwen3.5-9b"]["files"].items():  # the same snapshot the tokenizer check pinned
        assert lines[f] == h


def test_convert_gdn_matches_the_hf_gdn_layer():
    """Parity: convert.gdn.GDN (what convert/ trains and saves) == HF Qwen3_5GatedDeltaNet (what Q9-AF serves) on the
    same weights, so a converted state dict means the same function after q9af_from_stock."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from convert.gdn import GDN

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention"])
    hf = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0.0, 0.2)
    for kernel in ("recurrent", "chunk"):
        ours = GDN.from_config(cfg, kernel=kernel).float().eval()
        ours.load_state_dict(hf.state_dict(), strict=True)  # parameter-compatible, no renames
        back = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()
        back.load_state_dict(ours.state_dict(), strict=True)  # and back: what q9af_from_stock does
        x = torch.randn(2, 37, 32)
        with torch.no_grad():
            a, b, c = ours(x), hf(x), back(x)
        a = a[0] if isinstance(a, tuple) else a
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (kernel, (a - b).abs().max())
        assert torch.equal(b, c)


def test_temperature_status_and_from_dict_tamper():
    import dataclasses

    from decide.calibration import (CalibrationNotServable, FeatureTemperature, content_set_sha256,
                                    disjointness_proof_sha256)

    manual = FeatureTemperature(w=(0.2, 0.1, 0.0))
    assert manual.identity()["status"] == "unfitted_manual"
    with pytest.raises(CalibrationNotServable):
        manual.assert_servable()
    assert FeatureTemperature().identity()["status"] == "uncalibrated_identity"
    calib = ("c" * 64,)
    fo = content_set_sha256(calib)
    dj = {"eval": content_set_sha256({"e" * 64})}
    ok = FeatureTemperature(w=(0.3, 0.2, -0.4), fitted_on=fo, disjointness=tuple(dj.items()), calib_content=calib,
                            proof_sha256=disjointness_proof_sha256(fo, dj))
    assert FeatureTemperature.from_dict(ok.to_dict()) == ok
    d = ok.to_dict()
    del d["id"]
    with pytest.raises(ValueError, match="no id"):
        FeatureTemperature.from_dict(d)
    stale = ok.to_dict()  # id still matches (disjointness is outside the id hash), but the proof no longer does
    stale["disjointness"] = {"eval": "0" * 64}
    with pytest.raises(ValueError, match="own proof"):
        FeatureTemperature.from_dict(stale)
    swapped = ok.to_dict()
    swapped["calib_content"] = ["d" * 64]
    with pytest.raises(ValueError, match="own proof"):
        FeatureTemperature.from_dict(swapped)
    assert dataclasses.replace(ok).identity()["id"] == ok.identity()["id"]


# ----------------------------------------------------------------------------------------------- issue #36 comparators
def test_g13_needs_both_comparators_and_the_child_arm_is_reusable(world, parent_cache, tmp_path):
    """Anti-creep: the last accepted checkpoint (here identical to the child) accepts, but the lineage anchor (an
    'informed' parent the child is worse than) rejects, so G13 rejects. G12 uses the last accepted only."""
    from safety_eval.standin import KeywordDecider
    from train import m2_gates as G

    cache, dec = parent_cache
    gold = {json.dumps(r["request"]["state"]): r["labels"] for r in world["records"].values()}

    def informed(req):
        resp = dec.decide(req)
        labels = gold.get(json.dumps(req["state"]))
        for q, p in resp["ssdm"]["served_probabilities"].items():
            if labels and q in labels:
                g = str(labels[q]).lower() if str(labels[q]).lower() in p else str(labels[q])
                resp["ssdm"]["served_probabilities"][q] = {o: (0.9 if o == g else 0.1 / max(1, len(p) - 1)) for o in p}
        return resp

    anchor = tmp_path / "anchor"
    _cache_parent(world, anchor, informed, KeywordDecider(gullibility=0.0))
    code, rep = _gates(world, cache, dec.decide, KeywordDecider(gullibility=0.0), anchor_arm=anchor,
                       out=tmp_path / "run" / "report.json")
    comps = rep["verdicts"]["g13"]["comparators"]
    assert comps["last_accepted"]["accept"] and not comps["anchor"]["accept"]
    assert code == 41 and "g13_retention_acc" in rep["promotion"]["failed"]["g13"]
    assert set(rep["stages"]["g13"]["comparator_arms"]) == {"last_accepted", "anchor"}
    assert "comparators" in rep["verdicts"]["g12"] and set(rep["stages"]["g12"]["comparator_arms"]) == {"last_accepted"}
    # the child's own units were cached as an arm: a later run can use it as its last accepted checkpoint
    child_arm = tmp_path / "run" / "child-arm"
    code2, rep2 = _gates(world, child_arm, dec.decide, KeywordDecider(gullibility=0.0), anchor_arm=cache)
    assert rep2["verdicts"]["g13"]["accept"] and code2 in (0, G.EXIT["pending"])


def test_index_arm_from_the_served_decider_is_bitwise_the_untempered_arm(world):
    _kit_or_skip()
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from train import m2_gates as G

    _, model = _decider(2)
    items = _items(world, SPEC)
    ft = G.fit_child_temperature(model, items["calibration"], {"eval": items["dev"], "retention": items["retention"]})
    be = DecisionModelBackend(model, ByteEncoder(), template=TEMPLATE)
    served, plain = Decider(be, calibration=ft.to_serving()), Decider(be)
    a = G.index_predict(served.decide, world["index_rows"], untempered_from_raw=True)
    b = G.index_predict(plain.decide, world["index_rows"])
    assert all(a[k]["status"] == "ok" for k in a)
    assert {k: v["response"] for k, v in a.items()} == {k: v["response"] for k, v in b.items()}
    assert all("served_response" in v for v in a.values())
