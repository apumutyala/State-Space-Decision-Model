"""JEV3-TYPED (A-PREREG-JEV3-TYPED): the typed sym layout (score levels as content, the canonical scale in the prefix),
KEEP gate 2 (the evidence lane's score-item invariance op, vendored from claude/ev-score-inv @c47c0a46), the warm start,
and the job params / cell config / data pins. CPU only."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from heads.layout import ByteEncoder, build_row  # noqa: E402
from heads.symmetric import (SYM_LAYOUT, SYM_TYPED_LAYOUT, level_content, parse_level,  # noqa: E402
                             typed_score_axis)
from train.data import TOY_TEMPLATE  # noqa: E402

ENC = ByteEncoder()
EV_SHA = "c47c0a463936d8aff586ee69728291521bc31336"
PINS = {"scripts/ev_score_perturb.py": "803eeb9019f45f8a077f289ab6eab4f930510cf3ee7e5f847bdd855cbcf3e1da",
        "scripts/ev_score_flip.py": "b59f21655522187b3f8453e8092f41e3943c24d6c9fe6dc83beb16fb5961b6b2",
        "tests/test_ev_score_inv.py": "7f87bf1e7fef644bfebf8c71b7a3a2af94ab73e7fee33b77b6cd56b9a98d558b"}
RUB = ["no impact", "minor impact, workaround exists", "major impact on a core function", "total outage, data loss",
       "catastrophic"]


def _text(seq) -> str:
    return bytes(x - 4 for x in seq).decode("utf-8")


def _score_req(k: int, state: str = "incident 7: users affected 3") -> dict:
    return {"state": state, "questions": {"q": {"type": "score", "instructions": "Rate the severity.",
                                               "criteria": RUB[:k]}}}


def _typed_req(k: int, display: list[tuple[str, str]], state: str = "incident 7: users affected 3") -> dict:
    cont = [level_content(v, k, RUB[v]) for v in range(k)]
    return {"state": state, "questions": {"q": {"type": "score", "instructions": "Rate the severity.",
                                               "scale": cont, "options": dict(display)}}}


def _tiny(seed: int = 0):
    from heads.backbone import tiny_hf_backbone
    from heads.model import DecisionModel

    torch.manual_seed(seed)
    return DecisionModel(tiny_hf_backbone("qwen3.5", seed=seed, fork_mode="cache"), "fork", seed=seed).float().eval()


def _probs(model, req) -> dict[str, float]:
    row = build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
    with torch.no_grad():
        lp = model([row]).log_probs[0, : row.n_options].double().exp()
    return {o: float(p) for o, p in zip(row.option_ids, lp)}


# ------------------------------------------------------------------------------------------------ vendored op
def test_vendored_evidence_scripts_are_byte_identical_to_the_pin():
    for f, sha in PINS.items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == sha, f"{f} differs from {EV_SHA}"
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_jev3_typed.json").read_text(encoding="utf-8"))
    si = p["jevkit"]["score_inv"]
    assert si["scripts_sha256"] == {k: v for k, v in PINS.items() if k.startswith("scripts/")}
    from jobs.m2_first_cell.screen import score_inv_scripts_check

    assert score_inv_scripts_check(si["scripts_sha256"]) == []
    assert score_inv_scripts_check({**si["scripts_sha256"], "scripts/ev_score_flip.py": "0" * 64})


def test_rendering_matches_the_evidence_op_byte_for_byte():
    """Training rows, the native public readout and the gate-2 requests must agree: R_score == render_level."""
    from scripts import ev_score_perturb as SP

    assert SP.RENDERING.startswith("ev-score-level-as-content/1")
    for k in (2, 4, 5):
        for v in range(k):
            for rub in (RUB[v], {"a": 1, "é": [1, 2]}, "  spaced  "):
                assert level_content(v, k, rub) == SP.render_level(v, k, rub)
    from jevkit.items import load_public

    sc = [it for it in load_public() if it.type == "score"]
    assert len(sc) == 18
    for it in sc:
        req = {"state": it.state, "questions": {"q": dict(it.question)}}
        row = build_row(req, "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
        assert [_text(s) for s in row.option_seqs] == SP._contents(it)
        assert row.option_ids == [str(v) for v in range(it.k)]
        base = SP.base_request(it)  # the gate-2 base request renders to the SAME row as the wire item
        rb = build_row({"state": base["state"], "questions": {"q": {
            "type": "score", "instructions": base["question"]["instructions"], "scale": base["question"]["scale"],
            "options": base["question"]["options"]}}}, "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
        assert rb.ids == row.ids and rb.continuations == row.continuations and rb.option_ids == row.option_ids


# ------------------------------------------------------------------------------------------------ typed rows
def test_score_display_permutations_and_relabels_are_bitwise_invariant_rows():
    """Every display order (K <= 4: all K!), fresh handles and permuted handles: the prefix and every continuation are
    bitwise identical; only option_ids move with the handles; the level order of the axis is the level VALUE order."""
    for k in (2, 3, 4):
        base = build_row(_score_req(k), "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
        cont = [level_content(v, k, RUB[v]) for v in range(k)]
        prefix = _text(base.ids)
        assert all(c in prefix for c in cont) and prefix.index(cont[0]) < prefix.index(cont[-1])  # scale in the prefix
        assert base.meta["score_levels"] == list(range(k)) and base.meta["typed_score"] is True
        handle_sets = [[str(v) for v in range(k)], [f"s_{v:06x}" for v in (7, 3, 11, 5)[:k]],
                       [str(v) for v in ([1, 0] + list(range(2, k)))]]
        for hs in handle_sets:
            for order in itertools.permutations(range(k)):
                disp = [(hs[v], cont[v]) for v in order]
                r = build_row(_typed_req(k, disp), "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
                assert r.ids == base.ids and r.continuations == base.continuations
                assert r.option_seqs == base.option_seqs and r.option_ids == hs
                for h in hs:  # no handle string reaches the model unless it is also content
                    if not h.isdigit():
                        assert h not in _text(r.ids + sum(r.continuations, []))


def test_score_level_value_fails_closed():
    k = 3
    cont = [level_content(v, k, RUB[v]) for v in range(k)]
    bad = [("0", cont[0]), ("1", "minor impact"), ("2", cont[2])]  # no level value in content
    with pytest.raises(ValueError, match="no level value"):
        build_row(_typed_req(k, bad), "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
    with pytest.raises(ValueError, match="not exactly"):
        typed_score_axis([("a", level_content(0, 3, "x")), ("b", level_content(0, 3, "y")),
                          ("c", level_content(2, 3, "z"))])
    with pytest.raises(ValueError, match="not exactly"):  # K in content disagrees with the option count
        typed_score_axis([("a", level_content(0, 4, "x")), ("b", level_content(1, 4, "y"))])
    with pytest.raises(ValueError, match="scale"):
        typed_score_axis([(str(v), cont[v]) for v in range(k)], scale=list(reversed(cont)))
    with pytest.raises(ValueError, match="duplicate"):
        typed_score_axis([("a", cont[0]), ("a", cont[1])])
    assert parse_level("level 12 of 13: x") == (12, 13)
    with pytest.raises(ValueError):
        parse_level("level 01 of 3: x")  # one canonical spelling of each value


def test_choice_and_noul_rows_are_the_sym_fork_rows():
    req = {"state": "s", "questions": {"c": {"type": "choice", "instructions": "pick",
                                             "criteria": {"b": "beta", "a": "alpha"}},
                                       "n": {"type": "noul", "instructions": "is it?"}}}
    for qid in ("c", "n"):
        a = build_row(req, qid, ENC, template=TOY_TEMPLATE, layout=SYM_LAYOUT)
        b = build_row(req, qid, ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
        assert (a.ids, a.continuations, a.option_ids, a.option_seqs) == (b.ids, b.continuations, b.option_ids,
                                                                        b.option_seqs)
    with pytest.raises(ValueError):
        from heads.symmetric import build_row_sym

        build_row_sym(req, "c", ENC, TOY_TEMPLATE, SYM_TYPED_LAYOUT, score_options=[("a", "level 0 of 1: x")])


def test_tiny_model_distribution_is_bitwise_invariant():
    model = _tiny()
    k = 4
    cont = [level_content(v, k, RUB[v]) for v in range(k)]
    p0 = _probs(model, _score_req(k))
    rng = random.Random(0)
    for _ in range(6):
        order = list(range(k))
        rng.shuffle(order)
        hs = [f"s_{rng.getrandbits(24):06x}" for _ in range(k)]
        p1 = _probs(model, _typed_req(k, [(hs[v], cont[v]) for v in order]))
        assert all(p1[hs[v]] == p0[str(v)] for v in range(k))  # bitwise after mapping back


# ------------------------------------------------------------------------------------------------ gate 2 end to end
def _gate2(tmp_path, head):
    from tests import test_ev_score_inv as T
    from scripts import ev_score_flip as SF
    from scripts import ev_score_perturb as SP

    items = T._items()
    base, pert = SP.make(items, SP.OPS, (0, 1, 2))
    d = tmp_path / "req"
    d.mkdir(parents=True)
    SP.write_rows(base, d / "base.jsonl")
    SP.write_rows(pert, d / "perturbed.jsonl")
    for name, rows in (("base", SF.rj(d / "base.jsonl")), ("perturbed", SF.rj(d / "perturbed.jsonl"))):
        (d / f"{name}.preds.jsonl").write_text("".join(json.dumps({"id": r["id"], "answer": head(r), "status": "ok"})
                                                       + "\n" for r in rows), encoding="utf-8")
    res = SF.run_dir(items, d, max_dp=0.0)
    return res, {r["perturbation"]["op"] for r in pert}


def test_ev_score_flip_passes_the_typed_head_and_fails_a_handle_reading_head(tmp_path):
    from jobs.m2_first_cell.screen import _score_inv_request
    from tests import test_ev_score_inv as T

    model = _tiny(1)

    def typed_head(row):
        row2 = build_row(_score_inv_request(row), "q", ENC, template=TOY_TEMPLATE, layout=SYM_TYPED_LAYOUT)
        with torch.no_grad():
            lp = model([row2]).log_probs[0, : row2.n_options].float()
        return {"type": "score", "probabilities": {o: math.exp(float(lp[i])) for i, o in enumerate(row2.option_ids)}}

    res, ops = _gate2(tmp_path / "typed", typed_head)
    assert ops == {"score_reverse", "score_shuffle", "score_relabel", "score_permute"}
    assert res["score_inv_pass"] is True and res["bitwise"] is True, res["failures"]
    assert all(v["n"] > 0 and v["flips"] == 0 and v["max_abs_dp"] == 0.0 for v in res["per_operator"].values())
    bad, _ = _gate2(tmp_path / "handle", T.handle_head)
    assert bad["score_inv_pass"] is False
    assert any(bad["per_operator"][op]["flips"] > 0 or bad["per_operator"][op]["max_abs_dp"] > 0
               for op in ("score_relabel", "score_permute"))


def test_screen_score_inv_report_fails_closed(tmp_path):
    from jobs.m2_first_cell.screen import score_inv_report

    r = score_inv_report(tmp_path / "none")
    assert r["native"]["score_inv_pass"] is False and "MISSING" in r["native"]["status"]


# ------------------------------------------------------------------------------------------------ warm start
def test_warm_start_copies_weights_checks_sha_and_lineage(tmp_path):
    from heads.backbone import tiny_backbone
    from heads.model import DecisionModel
    from train.ckpt_export import export_inference
    from train.data import items_from_records
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    recs = []
    for i in range(8):
        recs.append({"id": f"r{i}", "request": {"model": "toy", "state": f"case {i}",
                                                "questions": {"q": {"type": "score", "instructions": "rate",
                                                                    "criteria": RUB[:3]}}}, "labels": {"q": str(i % 3)}})
    guard = SplitGuard.synthetic_only("unit tests: warm start")

    def mk(layout, seed=0):
        items = items_from_records(recs, ENC, template=TOY_TEMPLATE, layout=layout)
        m = DecisionModel(tiny_backbone(seed=0, hidden=16), "fork", seed=seed)  # same frozen backbone, new head init
        torch.manual_seed(seed)
        cfg = TrainConfig(head="fork", layout=layout, lora_rank=4, batch_size=2, grad_accum=1, max_steps=2,
                          loss=LossConfig(arm="clef"), ckpt_dir=str(tmp_path / layout), seed=seed)
        return Trainer(m, cfg, items, guard=guard)

    src = mk(SYM_LAYOUT)
    src.train()
    pt = src.save()
    out = tmp_path / "export"
    export_inference(pt, out)
    exp = sorted(out.glob("trainable-step*.safetensors"))[-1]
    sha = hashlib.sha256(exp.read_bytes()).hexdigest()
    dst = mk(SYM_TYPED_LAYOUT, seed=5)  # different layout + init: weights come from the export
    with pytest.raises(ValueError, match="sha256"):
        dst.warm_start(exp, "0" * 64)
    w = dst.warm_start(exp, sha)
    assert w["sha256"] == sha and w["source_step"] == 2 and "layout" in w["lineage_differs_expected"]
    sp = {n: p for n, p in src.model.named_parameters() if p.requires_grad}
    dp = dict(dst.model.named_parameters())
    for n, p in sp.items():
        assert torch.equal(dp[n].detach().to(torch.bfloat16), p.detach().to(torch.bfloat16))
    assert dst.step == 0 and not dst.history
    hist = dst.train()
    assert len(hist) == 2 and all(math.isfinite(h["loss"]) for h in hist)
    with pytest.raises(ValueError, match="fresh run"):
        dst.warm_start(exp, sha)


# ------------------------------------------------------------------------------------------------ params / cell
def _params():
    return json.loads((ROOT / "jobs/m2_first_cell/job_params_jev3_typed.json").read_text(encoding="utf-8"))


def _cell():
    return json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_typed.json").read_text(encoding="utf-8"))


def test_params_budget_schedule_and_registration():
    from jobs.common import costs
    from jobs.m2_first_cell import screen

    p = _params()
    assert p["mode"] == "jev3_typed" and p["cell_config"] == "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_typed.json"
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    w = costs.worst_case_usd(p)
    assert w <= 18.0 and p["hard_cap_usd"] == 18.0 and f"{w:.2f}" in (ROOT / "jobs/m2_first_cell/JOB_SPEC.md").read_text(
        encoding="utf-8")
    assert "job_params_jev3_typed.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert screen.estimate(p)["ok"]  # the setup preflight (job-svvvn exit 15 at cap 32)
    jk = p["jevkit"]
    assert jk["paired_primary"] == "native_shimcal" and jk["shim_calibration"] and jk["stock_shim"]
    assert jk["g1_keep_gate"] and jk["g1_max_dp"] == 0.0 and jk["perturb_ops"] == ["option_reverse", "option_shuffle",
                                                                                    "key_relabel"]
    assert jk["score_inv"]["keep_gate"] and jk["score_inv"]["max_dp"] == 0 and jk["slice_first"] and jk["flip_ops_all"]
    assert jk["slice_state_disjoint"]["train_records"] == "m2-jev3-typed/records.jsonl.gz"
    blk = screen.jevkit_block(p, _cell()["template"])
    assert blk["score_inv"]["keep_gate"] and blk["paired_primary"] == "native_shimcal"


def test_warm_start_block_and_job_wiring():
    from jobs.m2_first_cell.checkpoint_loader import checkpoint_block

    p = _params()
    ws = p["warm_start"]
    checkpoint_block({"checkpoint": {k: v for k, v in ws.items() if k != "cell_config"}})
    assert ws["member"] == "./ckpt/trainable-step0000258.safetensors"
    assert ws["ckpt_sha256"] == "b0cc27089620823d809844b7edc6f31340f55c88f2f0b0a8fcb8b290efd5e5a2"
    assert ws["cell_config"] == "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_sym_s0.json"
    assert "warmstart" in p["phase_caps_minutes"] and p["submit"]["env"]["CKPT_URL"].startswith("<")
    sh = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert '"${WARM_ARGS[@]}"' in sh and "--block warm_start" in sh


def test_cell_config_is_the_seed0_stack_with_the_typed_layout():
    c, c0 = _cell(), json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_jev3_sym_s0.json").read_text(
        encoding="utf-8"))
    assert c["layout"] == SYM_TYPED_LAYOUT and c0["layout"] == SYM_LAYOUT
    for k in ("head", "template", "loss_arm", "lora_rank", "lr_head", "lr_lora", "grad_accum", "consistency",
              "label_smoothing", "label_smoothing_noul", "dev_card_records", "dev_card_rows", "lr_schedule",
              "lr_horizon_frac", "collapse_guard", "best_ckpt", "t_ladder", "qwen35"):
        assert c[k] == c0[k], k
    assert c["train_set"]["pointer"] == "data/manifests/bschema-draw-jev3-typed.pointer.json"
    ptr = json.loads((ROOT / c["train_set"]["pointer"]).read_text(encoding="utf-8"))
    assert c["n_train_records"] == ptr["n_records"] - 136 and c["max_steps"] == c["n_train_records"] // 8
    assert {"routing", "intent", "severity"} <= set(ptr["families"])
    pins = (ROOT / "jobs/m2_first_cell/data_jev3_typed.sha256").read_text(encoding="utf-8")
    assert "m2-jev3-typed/records.jsonl.gz" in pins and "m2-jev3-calib/records.jsonl.gz" in pins


# ------------------------------------------------------------------------------------------------ fit (real tokenizer)
def test_typed_rows_fit_the_cell_caps_with_the_real_tokenizer():
    """assert_no_truncation (run_cell exit 25) pre-mortem: the typed rendering lengthens SCORE rows (the scale in the
    prefix, 'level v of K: ' in each continuation). Every question of the typed draw (train caps) and of the jev3 slice
    (eval caps) is rendered with the real Qwen3.5-4B tokenizer under sym-fork and sym-fork-typed: typed must keep every
    row sym-fork keeps. Runs where the job ships the data + tokenizer (jobs/jev3_typed_cpu); skipped elsewhere."""
    import gzip
    import os

    data = Path(os.environ.get("SSDM_DATA", "/nonexistent"))
    recs = {"train": data / "m2/build/bschema-draw-jev3-typed/records.jsonl.gz",
            "slice": data / "m2/build/bschema-draw-jev3-slice/records.jsonl.gz"}
    tokdir = os.environ.get("SSDM_BPE_TOKENIZER")
    if not (all(p.is_file() for p in recs.values()) and tokdir and Path(tokdir).is_dir()):
        pytest.skip("typed draw / slice records or SSDM_BPE_TOKENIZER not present (jobs/jev3_typed_cpu ships them)")
    from decide.shim_format import bind_template
    from heads.layout import resolve_template
    from train.guard import record_request
    from train.run_cell import keep_item

    cell = _cell()
    bind_template(cell["template"], tokdir)
    tpl = resolve_template(cell["template"])
    caps = {"train": (int(cell["train_max_k"]), int(cell["t_ladder"][-1])), "slice": (int(cell["max_k"]),
                                                                                        int(cell["eval_max_t"]))}
    rep = {}
    for part, p in recs.items():
        mk, mt = caps[part]
        n = lost = kept_both = 0
        grow, worst, lost_ids = [], 0, []
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                req = record_request(rec)
                for qid in rec["labels"]:
                    a = build_row(req, qid, None, template=tpl, layout=SYM_LAYOUT)
                    b = build_row(req, qid, None, template=tpl, layout=SYM_TYPED_LAYOUT)
                    n += 1
                    ka, kb = keep_item(a, mk, mt), keep_item(b, mk, mt)
                    lost += int(ka and not kb)
                    if ka and not kb:
                        lost_ids.append(rec["id"])
                    kept_both += int(ka and kb)
                    if b.qtype == "score":
                        grow.append(len(b.ids) - len(a.ids))
                        worst = max(worst, len(b.ids))
                    else:
                        assert (a.ids, a.continuations) == (b.ids, b.continuations)
        rep[part] = {"n": n, "lost_by_typed": lost, "lost_ids": sorted(lost_ids), "kept_both": kept_both,
                     "score_rows": len(grow),
                     "prefix_growth_max": max(grow) if grow else 0, "longest_score_prefix": worst, "cap_t": mt}
    print("TYPED_FIT " + json.dumps(rep))
    # train rows the typed rendering pushes over the cap must be in the cell's sha-pinned exclude_files (removed from
    # the pool before the draw, so assert_no_truncation never sees them); the slice must lose nothing
    excl = set()
    for spec in cell["train_set"].get("exclude_files") or []:
        excl |= {x.strip() for x in (ROOT / spec["file"]).read_text(encoding="utf-8").splitlines() if x.strip()}
    assert rep["slice"]["lost_by_typed"] == 0, rep
    assert set(rep["train"]["lost_ids"]) <= excl, rep
