"""A-PREREG-JEV3-LM-R (DRAFT): the two seed modes jev3_lm_r_s0 / jev3_lm_r_s1 = the JEV3-LM-S recipe on jev3 + the
JEV3-TYPED coverage rows (K subsampled to the public K) at the full 1-epoch budget, PRIMARY native_shimcal, no per-type
T_noul rule, both KEEP gates, A-share flag, canon_stock, slice_first; budgets from cell_budget.py; the draw's K rule.
CPU only, no torch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / "jobs/m2_first_cell"


def _j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("seed", [0, 1])
def test_jev3_lm_r_modes(seed):
    from jobs.common import costs
    from jobs.m2_first_cell import screen
    from jobs.m2_first_cell.cell_budget import check

    mode = f"jev3_lm_r_s{seed}"
    p, c = _j(D / f"job_params_{mode}.json"), _j(D / f"m2_first_cell_rwkv7_{mode}.json")
    s, ps = _j(D / "m2_first_cell_rwkv7_jev3_lm_s.json"), _j(D / "job_params_jev3_lm_s.json")
    assert p["mode"] == mode and p["cell_config"] == f"jobs/m2_first_cell/m2_first_cell_rwkv7_{mode}.json"
    assert c["seed"] == seed and c["variant"] == mode
    # recipe unchanged from JEV3-LM-S except the data, the step budget and the clock
    for k in ("head", "layout", "train_layout", "template", "lora_rank", "lr_head", "lr_lora", "warmup_steps",
              "label_smoothing", "label_smoothing_noul", "dev_card_records", "assert_no_truncation", "dev_card_every",
              "pass_rule", "lr_schedule", "lr_min_frac", "collapse_guard", "best_ckpt", "qwen35", "cohort_sha256"):
        assert c[k] == s[k], k
    assert c["train_set"]["exclude_files"] == s["train_set"]["exclude_files"]
    assert c["train_set"]["pointer"] == "data/manifests/bschema-draw-jev3-lmr.pointer.json"
    assert c["n_train_records"] == c["min_train_items"] == 7636 and c["data_pins"]["status"] == "final"
    assert c["max_steps"] == c["lr_horizon_steps"] == 7636 // 8 == 954 and "lr_horizon_frac" not in c
    # budget: cell cap covers the slow-end measurement for the whole epoch; schedule; worst case stated
    b = check(p, c)
    assert b["pass"] and b["expected_steps"] == 954
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 153
    w = costs.worst_case_usd(p)
    assert w == 23.06 and w <= p["hard_cap_usd"]
    assert f"{w:.2f}" in (D / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"job_params_{mode}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert costs.expected_usd(p) == 6.36
    e = screen.estimate(p)
    assert e["ok"] and e["proxy"] is False and p["screen_proxy"] is False and p["cell_budget_check"] is True
    # readout + gates
    jk, js = p["jevkit"], ps["jevkit"]
    assert jk["paired_primary"] == "native_shimcal" and jk["shim_calibration"] and jk["stock_shim"]
    assert "per_type" not in jk["tfit"] and jk["tfit"]["stock_band"] is True  # the banded T_noul rule is gone
    for k in ("g1_keep_gate", "g1_max_dp", "score_inv", "letter_prior", "canon_stock", "slice_first", "slice_floor",
              "flip_ops_all", "perturb_ops", "target_ref", "price", "by_family"):
        assert jk[k] == js[k], k
    assert jk["score_inv"]["keep_gate"] and jk["letter_prior"]["flag_pp"] == 5 and jk["g1_max_dp"] == 0.0
    assert jk["slice_state_disjoint"]["train_records"] == f"{p['train_dir']}/records.jsonl.gz" == "m2-jev3-lmr/records.jsonl.gz"
    assert p["jev_heldout"]["prereg"] == "EV-PREREG-JEV-HELDOUT" and p["jev_heldout"]["consumed_by_runner"] is False
    # the context ships the lmr draw under its pins, and the cell / pointer / pins files
    dests = {x["dest"]: x for x in p["context"]["data"]}
    for f in ("records.jsonl.gz", "bschema-draw-jev3-lmr.manifest.jsonl.gz", "bschema-draw-jev3-lmr.content.jsonl.gz"):
        assert dests[f"m2-jev3-lmr/{f}"]["pin"] == "jobs/m2_first_cell/data_jev3_lmr.sha256"
    assert not any(k.startswith("m2-jev3/") for k in dests)
    pins = (D / "data_jev3_lmr.sha256").read_text(encoding="utf-8").split()
    assert sum(1 for x in pins if x.startswith("m2-jev3")) == 4
    for f in (p["cell_config"], "jobs/m2_first_cell/data_jev3_lmr.sha256", "data/manifests/bschema-draw-jev3-lmr.pointer.json"):
        assert f in p["context"]["manifest_files"]


def test_lmr_pointer_matches_the_cell_pins():
    import hashlib

    ptr_path = ROOT / "data/manifests/bschema-draw-jev3-lmr.pointer.json"
    ptr = _j(ptr_path)
    typed = _j(ROOT / "data/manifests/bschema-draw-jev3-typed.pointer.json")
    assert ptr["ids_sha256"] == typed["ids_sha256"] and ptr["n_records"] == typed["n_records"] == 7760
    assert ptr["public_k"] == {"intent": 5, "routing": 6} and ptr["n_changed"] == 814
    for seed in (0, 1):
        c = _j(D / f"m2_first_cell_rwkv7_jev3_lm_r_s{seed}.json")
        assert c["data_pins"]["pinned"]["train_pointer_sha256"] == hashlib.sha256(ptr_path.read_bytes()).hexdigest()
        assert c["data_pins"]["expected"]["train_ids_sha256"] == ptr["ids_sha256"]


def test_k_subsample_keeps_gold_and_order():
    sys.path.insert(0, str(ROOT / "scripts"))
    import jev3_lm_r_draw as R

    crit = {f"o{i}": f"option {i}" for i in range(12)}
    r = {"id": "m2:x:1~view:jev1-route", "questions": {"q": {"type": "choice", "criteria": crit}},
         "labels": {"q": "o9"}, "aug": {"jev1": {"family8": "routing"}}}
    out = R.subsample(r, 6)
    keys = list(out["questions"]["q"]["criteria"])
    assert len(keys) == 6 and "o9" in keys and keys == [k for k in crit if k in keys]  # original relative order
    assert R.subsample(r, 6) == out  # deterministic
    assert R.subsample({**r, "questions": {"q": {"type": "choice", "criteria": dict(list(crit.items())[:5])}},
                        "labels": {"q": "o1"}}, 6) is None  # K <= public K: unchanged
    assert R.subsample({**r, "questions": {"q": {"type": "noul"}}}, 6) is None
    assert r["questions"]["q"]["criteria"] == crit  # the input row is not mutated
