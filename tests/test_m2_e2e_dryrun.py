"""m2 end-to-end dry run (CPU, tiny random models) plus regression tests for the integration bugs it found.

The end-to-end test runs ``scripts/m2_e2e_dryrun.py`` with the ``ci`` profile (target < 3 min on a laptop
CPU). It uses the PR #16 record file when present (outside git) and synthetic records of the same shape
otherwise; it needs the official Decision Index kit (``decision_index``, pinned clone) for harness scoring.
"""

import gzip
import hashlib
import json
import math
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _kit_or_skip():
    from scripts.m2_e2e_dryrun import ensure_kit

    try:
        return ensure_kit()
    except ImportError:
        pytest.skip("Decision Index kit (decision_index) not importable and no pinned local clone")


# ----------------------------------------------------------------------------------------- end to end
def test_m2_dry_run_end_to_end(tmp_path):
    _kit_or_skip()
    from scripts.m2_e2e_dryrun import ARMS, BACKBONES, HEADS, run

    t = time.perf_counter()
    rec = run(tmp_path, profile="ci")
    took = time.perf_counter() - t
    s = rec["summary"]
    assert s["n_runs"] == len(BACKBONES) * len(HEADS) * len(ARMS) == 16
    assert s["all_losses_finite"] and s["all_resumes_ok"]
    eq = s["resume_equals_uninterrupted"]
    assert len(eq) == 16 and all(v == 0.0 for v in eq.values()), eq  # every run, not a sample
    assert all(n == {"different_data": "refused", "different_config": "refused"} for n in s["lineage_negative_checks"])
    assert s["guard_refuses_all_probes"]
    probes = rec["stages"]["data"]["guard_probes"]
    assert probes["laundered_exact_by_kit_payload_hash"].startswith("refused")
    assert probes["laundered_reordered_by_content_hash"].startswith("refused")
    assert s["steer_complete"] and s["index_scored"]
    assert s["n_full_cards"] == 4 and len(s["promotion_verdicts"]) == 2
    for r in rec["stages"]["runs"]:
        assert all(math.isfinite(x) for x in r["temperature"]["w"]), r["head"]
        assert math.isfinite(r["cards"]["child"]["tempered"]["nll"])
        assert (tmp_path / f"{r['backbone']}-{r['head']}-{r['arm']}" / "ckpt" / "latest.json").exists()
        if r["full_card"]:
            v12, v13 = r["verdict_g12"], r["verdict_g13"]
            assert {c["criterion"] for c in v12["criteria"]} == {"g12_violation", "g12_steer_d1a", "g12_clean_acc"}
            assert {c["criterion"] for c in v13["criteria"]} == {"g13_steer_d1b", "g13_capability", "g13_ece"}
            assert v12["accept"] == (not v12["failed"]) and v13["accept"] == (not v13["failed"])
            assert "PENDING USER APPROVAL" in v12["margins"]["status"]
            assert r["verdict_g12_self"]["failed_evaluable"] == []  # self-comparison never fails on a CI
            assert r["cards"]["child"]["g4_exact_k1"]["n_variants"] > 0
            assert "lam_hat" in r["cards"]["child"]["g5_ltt"]
    assert any(r["temperature"].get("refuses_eval_fit") for r in rec["stages"]["runs"])
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["summary"] == json.loads(
        json.dumps(s, default=float))
    assert took < 180, f"dry run took {took:.0f}s (budget 180s)"


# ----------------------------------------------------------------------------------------- PR #16 records
def _pr16_record(rid="m2:test:1", state="", instr="Judge it.\nEvidence: the sky is green."):
    return {"id": rid, "schema": "ssdm-train-v0", "source": "anli", "family": "nli", "source_split": "train",
            "state": state, "untrusted_fields": [], "labels": {"answer": "neutral"},
            "questions": {"answer": {"type": "choice", "instructions": instr,
                                     "criteria": {"entailment": "e", "neutral": "n", "contradiction": "c"}}}}


def test_pr16_top_level_records_become_rows_and_are_hash_guarded():
    from decide.schema import canonical_json, sha256_text
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE, items_from_records
    from train.guard import ForbiddenRow, SplitGuard, record_request

    rec = _pr16_record(state={"premise": "p"})
    req = record_request(rec)
    assert req["state"] == {"premise": "p"} and req["questions"] is rec["questions"]
    items = items_from_records([rec], ByteEncoder(), template=TOY_TEMPLATE, guard=SplitGuard.synthetic_only("t"))
    assert len(items) == 1 and items[0].row.option_ids[items[0].label] == "neutral"
    # the request-hash check used to be skipped for this shape (no "request" key)
    g = SplitGuard(hashes={sha256_text(canonical_json(req))})
    with pytest.raises(ForbiddenRow, match="payload hash"):
        g.check(rec)


def test_empty_state_rows_use_reread_instruction_as_evidence():
    from heads.layout import ByteEncoder, build_row
    from train.data import TOY_TEMPLATE, items_from_records
    from train.guard import SplitGuard, record_request

    rec = _pr16_record()  # PR #16 "placement: instructions": state "" (~10% of the m2 sample)
    enc = ByteEncoder()
    row = build_row(record_request(rec), "answer", enc, template=TOY_TEMPLATE)
    a, b = row.state_span
    text = bytes(i - 4 for i in row.ids[a:b]).decode()
    assert row.meta["evidence"] == "reread_instructions" and "sky is green" in text and b < len(row.ids)
    with pytest.raises(ValueError, match="empty"):
        build_row(record_request(rec), "answer", enc, template=TOY_TEMPLATE, empty_state="error")
    assert items_from_records([rec], enc, template=TOY_TEMPLATE, guard=SplitGuard.synthetic_only("t"))
    full = build_row(record_request(_pr16_record(state={"premise": "p"})), "answer", enc, template=TOY_TEMPLATE)
    assert full.meta["evidence"] == "state"


def test_load_m2_records_verifies_the_committed_manifest(tmp_path):
    from decide.schema import canonical_json
    from train.data import ManifestMismatch, load_m2_records

    recs = [_pr16_record(f"m2:test:{i}", state={"p": str(i)}) for i in range(3)]
    lines = [canonical_json(r) for r in recs]
    rp, mp = tmp_path / "records.jsonl.gz", tmp_path / "m.manifest.jsonl.gz"
    with gzip.open(rp, "wt", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    man = [{"id": r["id"], "sha256": hashlib.sha256(x.encode()).hexdigest()} for r, x in zip(recs, lines)]
    with gzip.open(mp, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(m) for m in man) + "\n")
    got, ident = load_m2_records(rp, mp, keep=lambda r: r["id"] != "m2:test:1")
    assert [r["id"] for r in got] == ["m2:test:0", "m2:test:2"] and ident["manifest_verified"]
    man[2]["sha256"] = "0" * 64
    with gzip.open(mp, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(m) for m in man) + "\n")
    with pytest.raises(ManifestMismatch):
        load_m2_records(rp, mp)


# ----------------------------------------------------------------------------------------- calibration
def test_temperature_fit_with_mixed_option_counts_is_finite():
    from train.calibration import FeatureTemperature, features

    g = torch.Generator().manual_seed(0)
    n = 64
    ks = [2 + (i % 3) for i in range(n)]
    lp = torch.full((n, 4), float("-inf"), dtype=torch.float64)
    for i, k in enumerate(ks):
        lp[i, :k] = torch.log_softmax(torch.randn(k, generator=g, dtype=torch.float64) * 4, -1)
    y = torch.tensor([i % k for i, k in enumerate(ks)])
    phi = features([50 + i for i in range(n)], ks)
    ft = FeatureTemperature.fit(lp, y, phi, [str(i) for i in range(n)])
    assert all(math.isfinite(w) for w in ft.w)
    assert ft.fit_report["nll_after"] < ft.fit_report["nll_before"]
    out = ft.apply(lp, phi)
    assert torch.isfinite(out[torch.isfinite(lp)]).all() and torch.isinf(out[~torch.isfinite(lp)]).all()
    assert torch.equal(out.argmax(-1), lp.argmax(-1))


# ----------------------------------------------------------------------------------------- decide backend
def test_decision_model_backend_through_decide_and_safety_bridge():
    from decide import Decider
    from heads import DecisionModel
    from heads.backbone import tiny_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from safety_eval import attacks, data, metrics
    from safety_eval.decide_bridge import DecideBridge
    from train.data import TOY_TEMPLATE

    model = DecisionModel(tiny_backbone(seed=0, hidden=16), "integrator", head_kwargs={"channels": 8}, seed=0)
    dec = Decider(DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE))
    req = {"model": "t", "state": {"note": "colour red"},
           "questions": {"c": {"type": "choice", "criteria": {"red": "r", "blue": "b"}},
                         "n": {"type": "noul", "instructions": "is it red?"},
                         "s": {"type": "score", "criteria": ["low", "mid", "high"]}}}
    resp = dec.decide(req)
    assert resp["ssdm"]["status"] == "ok", resp["ssdm"]
    assert abs(sum(resp["answers"]["c"]["probabilities"].values()) - 1) < 1e-3
    assert dec.backend.revision()["revision_id"].startswith("decision_head:")
    v = attacks.build_variants(data.build(1, 0, split="dev"), hijack=[], d1=None)
    rows = metrics.run(DecideBridge(dec), v)
    assert len(rows) == len(v) and all(r["status"] == "ok" for r in rows)


# ----------------------------------------------------------------------------------------- B1: laundering
def _kit_style_row(rid="32:SYNTH:0:0", gid="SYNTH-32-0"):
    """A row in the kit's frozen format, hashed exactly as decision_index/suite/build/freeze.py does."""
    from train.guard import KIT_REFERENCE_MODEL

    state = {"note": "the parcel is red", "sender": "ops"}
    questions = {"answer": {"type": "choice", "instructions": "Which colour?",
                            "criteria": {"red": "the colour red", "blue": "the colour blue", "amber": "amber"}}}
    payload = json.dumps({"model": KIT_REFERENCE_MODEL, "state": state, "questions": questions}, ensure_ascii=False,
                         separators=(",", ":"))
    return {"_evaluation": {"run_id": rid, "catalog_id": 32, "group_id": gid,
                            "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()},
            "id": "orig", "state": state, "questions": questions, "expected": {"answer": "red"}, "metadata": {}}


def _launder(row, reorder=False, model=None):
    def rev(d):
        return {k: (rev(v) if isinstance(v, dict) else v) for k, v in reversed(list(d.items()))}

    st, qs = row["state"], row["questions"]
    if reorder:
        st, qs = (rev(st) if isinstance(st, dict) else st), rev(qs)
    rec = {"id": "m2:fresh:1", "schema": "ssdm-train-v0", "source": "banking77", "state": st, "questions": qs,
           "labels": {}}
    if model:
        rec["model"] = model
    return rec


def test_kit_payload_hash_is_the_kits_own():
    _kit_or_skip()
    from decision_index.constants import REFERENCE_MODEL
    from decision_index.suite.io import dumps

    from train.guard import KIT_REFERENCE_MODEL, kit_payload_hash

    assert KIT_REFERENCE_MODEL == REFERENCE_MODEL
    row = _kit_style_row()
    want = hashlib.sha256(dumps({"model": REFERENCE_MODEL, "state": row["state"],
                                 "questions": row["questions"]}).encode()).hexdigest()
    assert kit_payload_hash(row["state"], row["questions"]) == want == row["_evaluation"]["payload_sha256"]


@pytest.mark.parametrize("model", [None, "ssdm-train", "jev-1.13.0", "anything"])
def test_laundered_row_refused_by_manifest_payload_hash(model):
    from train.guard import ForbiddenRow, SplitGuard

    row = _kit_style_row()
    g = SplitGuard(hashes={row["_evaluation"]["payload_sha256"]})  # what a dev/test manifest line carries
    with pytest.raises(ForbiddenRow, match="payload hash"):
        g.check(_launder(row, model=model))


def test_reordered_laundered_row_needs_and_hits_the_content_index(tmp_path):
    from train.guard import ForbiddenRow, SplitGuard, content_hash

    row = _kit_style_row()
    rec = _launder(row, reorder=True)
    SplitGuard(hashes={row["_evaluation"]["payload_sha256"]}).check(rec)  # key order defeats the exact kit hash
    p = tmp_path / "content.txt.gz"
    with gzip.open(p, "wt", encoding="ascii") as f:
        f.write(content_hash(row["state"], row["questions"]) + "\n")
    g = SplitGuard(hashes={row["_evaluation"]["payload_sha256"]}).load_content_index(p)
    assert g.identity()["content_index"]["n"] == 1
    with pytest.raises(ForbiddenRow, match="content hash"):
        g.check(rec)
    assert "none loaded" in SplitGuard().identity()["content_index"]


def _local(path):
    from harness import config as hc

    p = hc.DATA_ROOT / path
    if not p.exists():
        pytest.skip(f"{p} not present (outside git)")
    return p


def test_laundered_real_dev_rows_are_refused_by_the_production_guard():
    """Real-format rows from the local DEV slice (guard test only; nothing is scored), laundered three ways."""
    from train.guard import ForbiddenRow, SplitGuard

    dev = _local("slices/dev-rows.jsonl.gz")
    ci = _local("m2/index-content-hashes-v1.txt.gz")
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "harness" / "manifests"
    g = SplitGuard.from_manifests([root / "dev-manifest.jsonl.gz", root / "test-manifest.jsonl.gz"], content_index=ci)
    rows = []
    with gzip.open(dev, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 400 == 0:
                rows.append(json.loads(line))
    assert len(rows) >= 20
    for r in rows:
        for rec in (_launder(r), _launder(r, model="ssdm-train"), _launder(r, reorder=True)):
            with pytest.raises(ForbiddenRow):
                g.check(rec)


def test_pr16_sample_passes_the_strengthened_guard():
    from train.data import load_m2_records
    from train.guard import SplitGuard

    recs_p = _local("m2/build/sample-20000-seed20261006/records.jsonl.gz")
    ci = _local("m2/index-content-hashes-v1.txt.gz")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    g = SplitGuard.from_manifests([root / "harness/manifests/dev-manifest.jsonl.gz",
                                   root / "harness/manifests/test-manifest.jsonl.gz"], content_index=ci)
    recs, _ = load_m2_records(recs_p, root / "data/manifests/sample-20000-seed20261006.manifest.jsonl.gz")
    for r in recs:
        g.check(r)  # the PR #16 dedup is stricter than exact-content matching: nothing may be refused


# ----------------------------------------------------------------------------------------- B2: empty states
@pytest.mark.parametrize("state", ["", "   \n", {}, [], None, {"a": ""}, {"a": {}, "b": [" "]}])
def test_every_empty_state_uses_the_reread_instruction(state):
    from decide.prompt import build_prompt
    from heads.layout import ByteEncoder, build_row, is_empty_state
    from train.data import TOY_TEMPLATE
    from train.guard import record_request

    req = record_request(_pr16_record(state=state))
    assert is_empty_state(state)
    row = build_row(req, "answer", ByteEncoder(), template=TOY_TEMPLATE)
    a, b = row.state_span
    assert row.meta["evidence"] == "reread_instructions"
    assert "sky is green" in bytes(i - 4 for i in row.ids[a:b]).decode()
    # the prompt text itself is unchanged: byte-identical to decide's prompt (so "{}" is still rendered)
    assert bytes(i - 4 for i in row.ids).decode() == build_prompt(TOY_TEMPLATE, req, "answer")
    with pytest.raises(ValueError, match="empty"):
        build_row(req, "answer", ByteEncoder(), template=TOY_TEMPLATE, empty_state="error")


@pytest.mark.parametrize("state", ["x", {"a": "x"}, {"n": 0}, [False], {"a": {"b": "y"}}])
def test_non_empty_states_keep_the_state_span(state):
    from heads.layout import ByteEncoder, build_row, is_empty_state
    from train.data import TOY_TEMPLATE
    from train.guard import record_request

    assert not is_empty_state(state)
    row = build_row(record_request(_pr16_record(state=state)), "answer", ByteEncoder(), template=TOY_TEMPLATE)
    assert row.meta["evidence"] == "state"


def test_census_all_20k_pr16_records_span_correctly():
    """Every record of the PR #16 sample: rows build, and the evidence span is the state iff the state has
    content (2,060 "" + 2,226 {} records take the re-read instruction)."""
    import collections

    from heads.layout import is_empty_state
    from train.data import items_from_records, load_m2_records

    recs_p = _local("m2/build/sample-20000-seed20261006/records.jsonl.gz")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    recs, _ = load_m2_records(recs_p, root / "data/manifests/sample-20000-seed20261006.manifest.jsonl.gz")
    assert len(recs) == 20000

    class LenEnc:  # length-faithful stand-in tokenizer (1 id per byte); fast on long rows
        def identity(self):
            return {"kind": "len"}

        def __call__(self, text, trusted=True):
            return [5] * len(text.encode())

    empty = collections.Counter()
    seen = collections.Counter()
    for r in recs:
        e = is_empty_state(r["state"])
        empty[repr(r["state"]) if e else "content"] += 1
        for it in items_from_records([r], LenEnc(), template="jrt-plain-v1"):
            seen[(it.row.meta["evidence"], e)] += 1
            assert it.row.state_span[1] > it.row.state_span[0]
    assert empty["''"] == 2060 and empty["{}"] == 2226
    assert set(seen) == {("state", False), ("reread_instructions", True)}, seen


# ----------------------------------------------------------------------------------------- non-blocking fixes
def test_temperature_raises_on_nan_logits():
    from train.calibration import FeatureTemperature, features

    lp = torch.log_softmax(torch.randn(4, 3, dtype=torch.float64), -1)
    lp[1, 2] = float("nan")
    with pytest.raises(FloatingPointError):
        FeatureTemperature().apply(lp, features([5] * 4, [3] * 4))
    with pytest.raises(FloatingPointError):
        FeatureTemperature.fit(lp, torch.zeros(4, dtype=torch.long), features([5] * 4, [3] * 4), list("abcd"))


def test_backend_keeps_control_token_code_raises_on_nan_and_caches_revision(monkeypatch):
    from decide import Decider
    from decide.untrusted import ControlTokenError
    from heads import DecisionModel
    from heads.backbone import tiny_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE

    model = DecisionModel(tiny_backbone(seed=0, hidden=16), "integrator", head_kwargs={"channels": 8}, seed=0)
    req = {"model": "t", "state": {"note": "red"}, "questions": {"c": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}}

    def refusing(text, trusted=True):
        if not trusted:
            raise ControlTokenError("control ids")
        return ByteEncoder()(text)

    resp = Decider(DecisionModelBackend(model, refusing, template=TOY_TEMPLATE)).decide(req)
    assert resp["ssdm"]["status"] == "unsupported"
    assert resp["ssdm"]["error"]["code"] == ControlTokenError.code
    be = DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE)
    assert be.revision() is be.revision()
    with torch.no_grad():
        for p in model.head.parameters():
            p.fill_(float("nan"))
    resp = Decider(be).decide(req)
    assert resp["ssdm"]["status"] != "ok"


# ----------------------------------------------------------------------------------------- B3 + G12/G13 rules
@pytest.fixture(scope="module")
def standin():
    from safety_eval import attacks, data, metrics
    from safety_eval.standin import KeywordDecider

    reqs = data.build(6, 4, split="dev")
    v = attacks.build_variants(reqs, hijack=[], d1=None)
    gate = attacks.build_variants(reqs, hijack=[], d1=sorted(attacks.D1_TEMPLATES_B), final=True)
    return {"v": v, "gate": gate,
            "parent": metrics.run(KeywordDecider(gullibility=0.0), v),
            "same": metrics.run(KeywordDecider(gullibility=0.0), v),
            "gullible": metrics.run(KeywordDecider(gullibility=3.0), v),
            "gate_parent": metrics.run(KeywordDecider(gullibility=0.0), gate),
            "gate_same": metrics.run(KeywordDecider(gullibility=0.0), gate),
            "gate_gullible": metrics.run(KeywordDecider(gullibility=3.0), gate)}


DEC_OK = {"identical_items": True, "acc_diff": {"point": 0.0}, "acc_child": 0.5, "acc_parent": 0.5}


def _g12(s, child="same", dec=DEC_OK, **kw):
    from train.promotion import verdict_g12

    return verdict_g12(decision=dec, steer_child_rows=s[child], steer_parent_rows=s["parent"],
                       expected_variants=s["v"], n_boot=200, **kw)


def test_g12_accepts_a_non_regressing_checkpoint(standin):
    v = _g12(standin)
    assert v["accept"], v["criteria"]
    steer = next(c for c in v["criteria"] if c["criterion"] == "g12_steer_d1a")
    assert steer["n_eligible"] > 0 and steer["complete"]


def test_g12_rejects_more_gullible_child(standin):
    v = _g12(standin, child="gullible")
    assert v["failed"] == ["g12_steer_d1a"]


def test_g12_clean_accuracy_drop_rejects(standin):
    assert _g12(standin, dec={**DEC_OK, "acc_diff": {"point": -0.02}})["failed"] == ["g12_clean_acc"]
    assert _g12(standin, dec={"identical_items": False})["failed"] == ["g12_clean_acc"]


@pytest.mark.parametrize("drop", ["attacked", "clean", "both_arms_attacked"])
def test_dropped_units_make_the_comparison_incomplete_and_reject(standin, drop):
    """B3 regression: compare_arms without expected_ids inner-joins units present in either arm, so a unit
    dropped from BOTH arms vanished silently. With the expected unit set it is an explicit error."""
    from train.promotion import verdict_g12

    att = next(r["request_id"] for r in standin["same"] if r["attack_family"] == "d1")
    cln = next(r["request_id"] for r in standin["same"] if r["attack_family"] == "clean")
    gone = cln if drop == "clean" else att
    child = [r for r in standin["same"] if r["request_id"] != gone]
    parent = [r for r in standin["parent"] if r["request_id"] != gone] if drop == "both_arms_attacked" else standin["parent"]
    v = verdict_g12(decision=DEC_OK, steer_child_rows=child, steer_parent_rows=parent,
                    expected_variants=standin["v"], n_boot=200)
    assert not v["accept"] and "g12_steer_d1a" in v["failed"] and "g12_violation" in v["failed"]
    steer = next(c for c in v["criteria"] if c["criterion"] == "g12_steer_d1a")
    assert not steer["evaluable"] and not steer["complete"]


def test_g12_rejects_served_violations(standin):
    bad = [dict(r) for r in standin["same"]]
    next(r for r in bad if r["status"] == "ok")["violation"] = True
    from train.promotion import verdict_g12

    v = verdict_g12(decision=DEC_OK, steer_child_rows=bad, steer_parent_rows=standin["parent"],
                    expected_variants=standin["v"], n_boot=200)
    assert v["failed"] == ["g12_violation"]


@pytest.mark.parametrize("lo,hi,pt,ok,why", [
    (0.01, 0.015, 0.012, False, "regression demonstrated"),  # lower CI > 0 rejects even inside the margin
    (-0.01, 0.03, 0.01, False, "upper CI"),                  # upper CI beyond +m_steer
    (-0.01, 0.015, 0.005, True, None),                       # positive point estimate: reported, not rejected
    (-0.02, 0.0, -0.01, True, None),
])
def test_steer_rule_decision_table(monkeypatch, standin, lo, hi, pt, ok, why):
    from safety_eval import metrics as M
    from train import promotion

    real = M.compare_arms

    def fake(*a, **k):
        out = real(*a, **k)
        out["steer_rate"]["diff"] = {"point": pt, "lo": lo, "hi": hi}
        return out

    monkeypatch.setattr(M, "compare_arms", fake)
    c = promotion.steer_criterion("g12_steer_d1a", standin["same"], standin["parent"], standin["v"], "d1_A",
                                  promotion.MARGINS["m_steer"], n_boot=50)
    assert c["pass"] is ok and c["evaluable"]
    assert c["regression_point_estimate"] is (pt > 0)
    if why:
        assert why in c["reason"]


def _results(rows, pick):
    """Kit results for harness rows: ``pick(row, key, q) -> (choice, p_choice)``."""
    res = {}
    for r in rows:
        ans = {}
        for key, q in r["questions"].items():
            ch, p = pick(r, key, q)
            others = [o for o in q["criteria"] if o != ch]
            probs = {o: (1 - p) / len(others) for o in others}
            probs[ch] = p
            ans[key] = {"type": "choice", "choice": ch, "confidence": p, "probabilities": probs}
        res[r["_evaluation"]["run_id"]] = {**r["_evaluation"], "status": "ok", "response": {"answers": ans}}
    return res


def test_g13_retention_and_gate_steer(standin):
    _kit_or_skip()
    from scripts.m2_e2e_dryrun import synthetic_dev_rows
    from train.promotion import retention_comparison, verdict_g13

    rows = synthetic_dev_rows(6)
    gold = _results(rows, lambda r, k, q: (r["expected"][k], 0.8))
    worse = _results(rows, lambda r, k, q: (next(o for o in q["criteria"] if o != r["expected"][k]), 0.9))
    same = retention_comparison(rows, gold, gold, B=200, seed=0)
    assert same["identical_items"] and same["capability"]["lo"] == same["capability"]["hi"] == 0
    assert same["ece_diff"]["point"] == same["ece_diff"]["lo"] == same["ece_diff"]["hi"] == 0
    kw = dict(steer_parent_rows=standin["gate_parent"], expected_variants=standin["gate"], n_boot=200)
    assert verdict_g13(retention=same, steer_child_rows=standin["gate_same"], **kw)["accept"]
    assert verdict_g13(retention=same, steer_child_rows=standin["gate_gullible"], **kw)["failed"] == ["g13_steer_d1b"]
    bad = retention_comparison(rows, worse, gold, B=200, seed=0)
    assert set(verdict_g13(retention=bad, steer_child_rows=standin["gate_same"], **kw)["failed"]) == {
        "g13_capability", "g13_ece"}
    dropped = dict(gold)
    dropped.pop(rows[0]["_evaluation"]["run_id"])
    gone = retention_comparison(rows, dropped, gold, B=200, seed=0)
    assert not gone["identical_items"]
    assert set(verdict_g13(retention=gone, steer_child_rows=standin["gate_same"], **kw)["failed"]) == {
        "g13_capability", "g13_ece"}
    assert "g13_capability" in verdict_g13(retention=None, steer_child_rows=standin["gate_same"], **kw)["failed"]
    # G13 steer uses the held-out D1-B units: the development D1-A set cannot stand in for them
    v = verdict_g13(retention=same, steer_child_rows=standin["same"], steer_parent_rows=standin["parent"],
                    expected_variants=standin["v"], n_boot=200)
    assert "g13_steer_d1b" in v["failed"]


def test_paired_decision_identical_checkpoints_have_zero_difference():
    from train.promotion import paired_decision

    f = {f"r{i}/q": {"conf": 0.3 + 0.05 * (i % 10), "correct": float(i % 3 == 0), "cluster": f"r{i // 2}"}
         for i in range(40)}
    d = paired_decision(f, f, n_boot=100)
    assert d["identical_items"] and d["acc_diff"]["point"] == 0 and d["ece_diff"]["hi"] == 0 == d["ece_diff"]["lo"]
    assert not paired_decision(f, {k: v for k, v in list(f.items())[1:]})["identical_items"]
