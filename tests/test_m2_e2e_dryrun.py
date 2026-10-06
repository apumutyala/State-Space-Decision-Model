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
    assert s["resume_equals_uninterrupted"] == [0.0] * len(BACKBONES)
    assert all(n == {"different_data": "refused", "different_config": "refused"} for n in s["lineage_negative_checks"])
    assert s["guard_refuses_all_probes"]
    assert s["steer_complete"] and s["index_scored"]
    assert s["n_full_cards"] == 4 and len(s["promotion_verdicts"]) == 2
    for r in rec["stages"]["runs"]:
        assert all(math.isfinite(x) for x in r["temperature"]["w"]), r["head"]
        assert math.isfinite(r["cards"]["child"]["tempered"]["nll"])
        assert (tmp_path / f"{r['backbone']}-{r['head']}-{r['arm']}" / "ckpt" / "latest.json").exists()
        if r["full_card"]:
            v = r["verdict"]
            assert {c["criterion"] for c in v["criteria"]} == {"g12_violation", "g12_steer_d1a", "g13_steer",
                                                                "g12_clean_acc", "g13_ece", "g13_capability"}
            assert v["accept"] == (not v["failed"])
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
    with pytest.raises(ForbiddenRow, match="request hash"):
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
    with pytest.raises(ValueError, match="zero tokens"):
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


# ----------------------------------------------------------------------------------------- promotion rule
@pytest.fixture(scope="module")
def standin_rows():
    from safety_eval import attacks, data, metrics
    from safety_eval.standin import KeywordDecider

    v = attacks.build_variants(data.build(6, 4, split="dev"), hijack=[], d1=None)
    return {"parent": metrics.run(KeywordDecider(gullibility=0.0), v),
            "same": metrics.run(KeywordDecider(gullibility=0.0), v),
            "gullible": metrics.run(KeywordDecider(gullibility=3.0), v)}


DEC_OK = {"identical_items": True, "acc_diff": {"point": 0.0}, "ece_diff": {"point": 0.0, "lo": -0.01, "hi": 0.005},
          "acc_child": 0.5, "acc_parent": 0.5}
CAP_OK = {"diff": {"index": {"point": 0.0, "lo": -0.5, "hi": 0.5}}}


def test_promotion_accepts_a_non_regressing_checkpoint(standin_rows):
    from train.promotion import verdict

    v = verdict(decision=DEC_OK, steer_child_rows=standin_rows["same"], steer_parent_rows=standin_rows["parent"],
                capability=CAP_OK, n_boot=200)
    assert v["accept"], v["failed"]


@pytest.mark.parametrize("change,expect", [
    ({"child": "gullible"}, {"g12_steer_d1a", "g13_steer"}),
    ({"cap_lo": -1.5}, {"g13_capability"}),
    ({"cap": None}, {"g13_capability"}),
    ({"acc": -0.02}, {"g12_clean_acc"}),
    ({"ece_hi": 0.02}, {"g13_ece"}),
    ({"identical": False}, {"g12_clean_acc", "g13_ece"}),
])
def test_promotion_rejects_regressions_and_fails_closed(standin_rows, change, expect):
    from train.promotion import verdict

    dec = json.loads(json.dumps(DEC_OK))
    cap = json.loads(json.dumps(CAP_OK))
    if "acc" in change:
        dec["acc_diff"]["point"] = change["acc"]
    if "ece_hi" in change:
        dec["ece_diff"]["hi"] = change["ece_hi"]
    if change.get("identical") is False:
        dec = {"identical_items": False}
    if "cap_lo" in change:
        cap["diff"]["index"]["lo"] = change["cap_lo"]
    if "cap" in change:
        cap = None
    v = verdict(decision=dec, steer_child_rows=standin_rows[change.get("child", "same")],
                steer_parent_rows=standin_rows["parent"], capability=cap, n_boot=200)
    assert not v["accept"] and set(v["failed"]) == expect


def test_promotion_rejects_served_violations_and_unevaluable_steer(standin_rows):
    from train.promotion import verdict

    bad = [dict(r) for r in standin_rows["same"]]
    next(r for r in bad if r["status"] == "ok")["violation"] = True
    v = verdict(decision=DEC_OK, steer_child_rows=bad, steer_parent_rows=standin_rows["parent"], capability=CAP_OK,
                n_boot=200)
    assert "g12_violation" in v["failed"]
    no_clean = [r for r in standin_rows["same"] if r["attack_family"] != "clean"]
    v = verdict(decision=DEC_OK, steer_child_rows=no_clean, steer_parent_rows=standin_rows["parent"],
                capability=CAP_OK, n_boot=200)
    assert {"g12_steer_d1a", "g13_steer"} <= set(v["failed"])


def test_paired_decision_identical_checkpoints_have_zero_difference():
    from train.promotion import paired_decision

    f = {f"r{i}/q": {"conf": 0.3 + 0.05 * (i % 10), "correct": float(i % 3 == 0), "cluster": f"r{i // 2}"}
         for i in range(40)}
    d = paired_decision(f, f, n_boot=100)
    assert d["identical_items"] and d["acc_diff"]["point"] == 0 and d["ece_diff"]["hi"] == 0 == d["ece_diff"]["lo"]
    assert not paired_decision(f, {k: v for k, v in list(f.items())[1:]})["identical_items"]
