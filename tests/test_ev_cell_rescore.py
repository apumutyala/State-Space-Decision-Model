"""scripts/ev_cell_rescore.py: G1 on synthetic invariant / non-invariant arms (from files, fail closed), the file-order
check (a sort_keys dump of perturbed.jsonl is vacuous), the declared non-bitwise tolerance, Holm, and one end-to-end
CLI run on a synthetic artifact. CPU only, the pinned public items."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from jevkit import perturb as P
from jevkit.answers import answer_from_probs
from jevkit.items import load_public
from scripts import ev_cell_rescore as K

ITEMS = load_public()
SUB = ITEMS[:60]  # enough choice + noul items for every G1 operator, fast


def _h(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def invariant_model(item, noise: float = 0.0):
    """Reads option CONTENT only, in a canonical (content-sorted) order: bit-identical under reorder and relabel."""
    if item.type == "noul":
        return {"type": "noul", "noul": 0.1 + 0.8 * _h(json.dumps(item.state, sort_keys=True)) + noise}
    crit = item.question.get("criteria")
    keys = list(crit) if isinstance(crit, dict) else list(item.labels)
    descs = {k: json.dumps(crit[k], sort_keys=True) if isinstance(crit, dict) else str(i) for i, k in enumerate(keys)}
    ex = {k: math.exp(3 * _h(descs[k])) for k in keys}
    z = sum(ex[k] for k in sorted(keys, key=lambda k: descs[k]))
    probs = {k: ex[k] / z for k in keys}
    if noise:
        k0 = keys[0]
        probs = {k: (v + noise if k == k0 else v - noise / (len(keys) - 1)) for k, v in probs.items()}
    return answer_from_probs(item, probs)


def first_listed_model(item):
    """Position-biased: 0.7 on whichever option is listed first (the non-invariant arm)."""
    if item.type != "choice":
        return invariant_model(item)
    keys = list(item.question["criteria"])
    rest = 0.3 / (len(keys) - 1)
    return answer_from_probs(item, {k: (0.7 if n == 0 else rest) for n, k in enumerate(keys)})


def _rec(item, ans):
    return {"id": item.id, "answer": ans, "status": "ok", "usage": {"input_tokens": 100}, "latency_s": 0.1}


def make_g1_dir(d, model, items=SUB, sort_keys=False, drop_ops=(), noise_pert=0.0):
    d.mkdir(parents=True, exist_ok=True)
    ps = [p for p in P.make_perturbations(items, K.G1_OPS, (0,)) if p.op not in drop_ops]
    (d / "base.preds.jsonl").write_text("".join(json.dumps(_rec(it, model(it))) + "\n" for it in items), encoding="utf-8")
    pm = (lambda it: invariant_model(it, noise_pert)) if noise_pert else model
    (d / "perturbed.preds.jsonl").write_text("".join(json.dumps(_rec(p.item, pm(p.item))) + "\n" for p in ps),
                                             encoding="utf-8")
    (d / "perturbed.jsonl").write_text("".join(json.dumps(p.to_dict(), sort_keys=sort_keys) + "\n" for p in ps),
                                       encoding="utf-8")
    return d


def run_g1(d, max_dp=0.0):
    return K.g1_check(ITEMS, K.rj(d / "perturbed.jsonl"), K.rj(d / "base.preds.jsonl"),
                      K.rj(d / "perturbed.preds.jsonl"), max_dp, K.G1_OPS, B=50, seed=16)


def test_g1_invariant_arm_passes_bitwise(tmp_path):
    g = run_g1(make_g1_dir(tmp_path / "inv", invariant_model))
    assert g["g1_pass"] and g["bitwise"] and not g["non_bitwise_within_tolerance"], g["g1_failures"]
    for op in K.G1_OPS:
        v = g["per_operator"][op]
        assert v["n"] > 0 and v["flips"] == 0 and v["invalid"] == 0 and v["max_abs_dp"] == 0.0
        fo = v["file_order"]
        assert fo["matches_regenerated"] == fo["n"] == fo["changed_vs_base"]


def test_g1_non_invariant_arm_fails(tmp_path):
    g = run_g1(make_g1_dir(tmp_path / "pos", first_listed_model))
    assert not g["g1_pass"]
    assert g["per_operator"]["option_reverse"]["flips"] > 0 and g["per_operator"]["option_shuffle"]["flips"] > 0
    assert any("flips" in f for f in g["g1_failures"])
    lo, hi = g["per_operator"]["option_reverse"]["flip_rate_ci95"]
    assert 0 < lo <= g["per_operator"]["option_reverse"]["flip_rate"] <= hi


def test_g1_sort_keys_file_is_vacuous_even_for_an_invariant_arm(tmp_path):
    g = run_g1(make_g1_dir(tmp_path / "sk", invariant_model, sort_keys=True))
    assert not g["g1_pass"]
    for op in ("option_reverse", "option_shuffle"):
        fo = g["per_operator"][op]["file_order"]
        assert fo["changed_vs_base"] == 0 and fo["matches_regenerated"] < fo["n"]
    assert any("NOT in the in-memory perturbation order" in f for f in g["g1_failures"])
    assert any("reorder check vacuous" in f for f in g["g1_failures"])


def test_g1_tolerance_is_flagged_non_bitwise(tmp_path):
    d = make_g1_dir(tmp_path / "tol", invariant_model, noise_pert=1e-6)
    strict = run_g1(d, 0.0)
    assert not strict["g1_pass"] and any("max|dp|" in f for f in strict["g1_failures"])
    loose = run_g1(d, 1e-3)
    assert loose["g1_pass"] and not loose["bitwise"] and loose["non_bitwise_within_tolerance"]


def test_g1_missing_operator_fails_closed(tmp_path):
    g = run_g1(make_g1_dir(tmp_path / "miss", invariant_model, drop_ops=("key_relabel",)))
    assert not g["g1_pass"] and any(f.startswith("key_relabel: no valid") for f in g["g1_failures"])


def test_g1_in_job_report_diff(tmp_path):
    d = make_g1_dir(tmp_path / "rep", first_listed_model)
    g = run_g1(d)
    fake = {"g1_pass": True, "per_operator": {op: dict(g["per_operator"][op]) for op in K.G1_OPS}}
    fake["per_operator"]["option_reverse"] = dict(fake["per_operator"]["option_reverse"], flips=0)
    g2 = K.g1_check(ITEMS, K.rj(d / "perturbed.jsonl"), K.rj(d / "base.preds.jsonl"), K.rj(d / "perturbed.preds.jsonl"),
                    0.0, K.G1_OPS, B=20, seed=16, in_job=fake)
    assert g2["agrees_with_in_job_counts"] is False


def test_holm_step_down():
    h = K.holm({"a": 0.01, "b": 0.04, "c": 0.2})
    assert h["a"]["level"] == pytest.approx(0.05 / 3) and h["a"]["pass_p"]
    assert h["b"]["level"] == pytest.approx(0.025) and not h["b"]["pass_p"]
    assert h["c"]["tested"] is False and not h["c"]["pass_p"]
    h2 = K.holm({"x7c42": 0.565, "q668q": 0.81})
    assert h2["x7c42"]["level"] == pytest.approx(0.025) and not h2["x7c42"]["pass_p"] and not h2["q668q"]["tested"]


def test_mcnemar_exact():
    assert K.mcnemar(0, 0) == 1.0
    assert K.mcnemar(244, 84) < 1e-15
    assert K.mcnemar(2, 3) == 1.0


PRICE = ("kind=size_class_estimate,in=0.03,out=0,source=declared reference price for Qwen3.5-4B (test),"
         "as_of=2026-10-09,model=Qwen/Qwen3.5-4B")


def test_cli_end_to_end_synthetic_artifact(tmp_path):
    art = tmp_path / "art"
    jd = art / "out" / "screen" / "jevkit"
    jd.mkdir(parents=True)
    for arm, model in (("stock_shim", first_listed_model), ("native_shimcal", invariant_model)):
        (jd / f"{arm}.preds.jsonl").write_text("".join(json.dumps(_rec(it, model(it))) + "\n" for it in ITEMS),
                                               encoding="utf-8")
    (art / "out" / "screen.json").write_text(json.dumps({"jevkit": {}}), encoding="utf-8")
    make_g1_dir(jd / "g1", invariant_model)
    pos = make_g1_dir(tmp_path / "g1pos", first_listed_model)
    out = tmp_path / "out"
    rc = K.main(["--art", str(art), "--out", str(out), "--price", PRICE, "--B", "40",
                 "--g1", f"sym={jd / 'g1'}", "--g1", f"posbias={pos}", "--holm-ext", "other=0.9"])
    r = json.loads((out / "cell_rescore.json").read_text(encoding="utf-8"))
    assert rc == 3 and r["verdict"] == "FLAGGED"  # no cell.json / slice rows / screen arms: flagged, never silent
    assert any("slice_floor_rows.jsonl missing" in f for f in r["flags"])
    assert r["6_g1"]["sym"]["g1_pass"] and r["6_g1"]["sym"]["bitwise"]
    assert not r["6_g1"]["posbias"]["g1_pass"]
    assert r["6_g1"]["sym"]["base_preds_equal_arm"]["native_shimcal"] == len(SUB)
    p = r["1_primary"]
    assert p["ci95"][0] <= p["point"] <= p["ci95"][1] and 0.0 <= p["p_one_sided_boot"] <= 1.0
    assert set(p["holm"]["family"]) == {"native_shimcal", "other"}
    assert r["4_integrity"]["per_arm"]["native_shimcal"]["n_order_not_criteria_order"] == 0
    md = (out / "CELL_RESCORE.md").read_text(encoding="utf-8")
    for sec in ("(1) PRIMARY", "(2) SECONDARY", "(3) TARGET", "(4) INTEGRITY", "(5) SELECTION", "(6) G1", "(7) FLIP",
                "(8) TRANSFER"):
        assert sec in md
