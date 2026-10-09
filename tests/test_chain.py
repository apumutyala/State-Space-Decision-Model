"""Chained cells (train.chain): one training run carried over several SF jobs through their /output artifacts.

Tiny CPU models only. The chain path is the real one: link 1 trains and saves, ``train.chain.export`` writes
/output/chain, the test tars /output as the platform does, ``train.chain.install`` fetches it (file:// URL, sha256
pins) into a FRESH checkpoint dir, a FRESH Trainer resumes and finishes. It must equal one uninterrupted run bit for
bit (losses, weights, AdamW state, schedule, dev cards, best, collapse-guard window), and refuse every mismatch."""

import hashlib
import io
import json
import random
import math
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from train import chain as C  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")
ROOT = Path(__file__).resolve().parents[1]
JOB = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
#: the cell config every link shares (any dict: its canonical JSON is the chain id; run_cell compares it exactly)
CELL = {"seed": 0, "head": "snapshot", "lora_rank": 4, "max_steps": 20, "max_minutes": 60, "lr_schedule": "cosine"}
TOTAL, BOUNDARY, HORIZON = 20, 9, 20


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)


@pytest.fixture(scope="module")
def dev():
    return items_from_records(toy_records(12, seed=1), ENC, template=TOY_TEMPLATE)


def fresh(seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), "snapshot")


def cfg(d, **kw):
    c = dict(head="snapshot", lora_rank=4, batch_size=8, max_steps=TOTAL, lr_head=1e-2, lr_lora=1e-3, warmup_steps=2,
             lr_schedule="cosine", lr_min_frac=0.1, dev_card_every=4, dev_card_rows=6, best_ckpt=True,
             collapse_guard=True, collapse_window=3, collapse_loss_frac=0.95, collapse_acc_frac=0.0,
             ckpt_dir=str(d), ckpt_every=5)
    c.update(kw)
    return TrainConfig(**c)


def train_to(tr, step, collapse_at=None):
    """run_cell's loop shape at the Trainer level: the horizon is fixed once after a few steps (here 3)."""
    while tr.step < step:
        tr.train(1)
        if tr.step == 3:
            tr.set_horizon(HORIZON)
        if collapse_at is not None and tr.step == collapse_at:
            tr.collapse("forced")


def link_end_cell(tr, link=1, of=2, cell_cfg=CELL):
    """What run_cell writes to cell.json when an intermediate link ends (train.run_cell link_end branch)."""
    return {"config": cell_cfg, "ladder": {"T_max": 512}, "train_clock": {"step": tr.step, "used_s": 12.5},
            "train": {"stopped": "link_end", "steps": tr.step, "link": link, "of": of},
            "chain": {"link": link, "of": of, "chain_id": C.chain_id(cell_cfg)}, "train_cards": []}


def export_link(tr, tmp, budget=C.BUDGET, link=1, of=2, cell_cfg=CELL):
    """Link ``link`` ends: final save, cell.json, export to <tmp>/output/chain, the platform's tar of /output."""
    tr.save()
    cell = tmp / f"cell-link{link}.json"
    cell.write_text(json.dumps(link_end_cell(tr, link, of, cell_cfg)), encoding="utf-8")
    out = tmp / f"output{link}"
    rec = C.export(ck_dir=tr.cfg.ckpt_dir, cell_path=cell, cfg=cell_cfg, ch={"link": link, "of": of},
                   out_dir=out / "chain", budget=budget)
    art = tmp / f"artifact{link}.tar"
    with tarfile.open(art, "w") as t:
        for f in sorted((out / "chain").iterdir()):
            t.add(f, arcname=f"./chain/{f.name}")
    return rec, art


def block(rec, art, link=2, of=2):
    return {"link": link, "of": of, "from": {"job": "job-test", "artifact": "art-test",
                                            "artifact_sha256": hashlib.sha256(art.read_bytes()).hexdigest(),
                                            "manifest_sha256": rec["manifest_sha256"], "url_env": "CHAIN_URL"}}


def same_state(a, b):
    sa, sb = a.trainable_state(), b.trainable_state()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    oa, ob = a.opt.state_dict(), b.opt.state_dict()
    assert oa["param_groups"] == ob["param_groups"]
    for i in oa["state"]:
        for k, v in oa["state"][i].items():
            assert torch.equal(v, ob["state"][i][k]), (i, k)
    assert a.sched.state_dict() == b.sched.state_dict()
    assert (a.step, a.epoch, a.cursor) == (b.step, b.epoch, b.cursor)
    assert a.anti_collapse_state() == b.anti_collapse_state()


def run_chain(toy, dev, tmp, budget=C.BUDGET, collapse_at=None):
    l1 = Trainer(fresh(), cfg(tmp / "ck1"), toy, eval_items=dev, guard=G)
    train_to(l1, BOUNDARY, collapse_at if collapse_at and collapse_at <= BOUNDARY else None)
    rec, art = export_link(l1, tmp, budget)
    ck2 = tmp / "ck2"  # a fresh job: an empty checkpoint slot
    got = C.install(block(rec, art), cfg=CELL, ck_dir=ck2, work=tmp / "work2", cell_out=tmp / "out2" / "cell.json",
                    url=art.as_uri())
    l2 = Trainer(fresh(), cfg(ck2), toy, eval_items=dev, guard=G)
    assert l2.resume() and l2.loaded_format == "full" and l2.step == BOUNDARY
    l2.window_at_resume = list(l2._guard_window)  # the collapse guard's loss window continues across the link
    assert l2.window_at_resume == l1._guard_window and len(l1._guard_window) == 3
    train_to(l2, TOTAL, collapse_at)
    return rec, got, l2


# ------------------------------------------------------------------ the chain IS the uninterrupted run
def test_two_link_chain_equals_one_uninterrupted_run(toy, dev, tmp_path):
    ref = Trainer(fresh(), cfg(tmp_path / "ref"), toy, eval_items=dev, guard=G)
    train_to(ref, TOTAL)
    rng_ref = (torch.get_rng_state().clone(), random.getstate())
    rec, got, l2 = run_chain(toy, dev, tmp_path)
    assert torch.equal(torch.get_rng_state(), rng_ref[0]) and random.getstate() == rng_ref[1]  # RNG streams carried
    assert C.valid_result(rec) and rec["step"] == BOUNDARY and got["from_link"] == 1 and not got["reused"]
    # the boundary falls between dev cards (every 4): the best is an older step, carried FULL under the budget
    assert rec["best"] and rec["best"]["state"] == "full" and rec["best"]["step"] < BOUNDARY
    assert [x["loss"] for x in l2.history] == [x["loss"] for x in ref.history]
    same_state(ref, l2)
    assert l2.horizon == ref.horizon == HORIZON and l2.dev_cards == ref.dev_cards and l2.best == ref.best
    assert l2._guard_window == ref._guard_window and len(ref._guard_window) == 3


def test_chain_with_a_slim_best_is_still_exact_without_a_collapse(toy, dev, tmp_path):
    probe = Trainer(fresh(), cfg(tmp_path / "probe"), toy, eval_items=dev, guard=G)
    train_to(probe, BOUNDARY)
    probe.save()
    rec0 = json.loads((tmp_path / "probe" / "latest.json").read_text())
    full = (tmp_path / "probe" / rec0["path"]).stat().st_size
    budget = full + int(0.6 * full)  # latest + a full best do not fit, latest + a trainable-only best do
    ref = Trainer(fresh(), cfg(tmp_path / "ref"), toy, eval_items=dev, guard=G)
    train_to(ref, TOTAL)
    rec, _, l2 = run_chain(toy, dev, tmp_path, budget=budget)
    assert rec["best"]["state"] == "trainable" and rec["export_bytes"] <= budget
    best = torch.load(Path(l2.cfg.ckpt_dir) / rec["best"]["file"], map_location="cpu", weights_only=False)
    assert best["optimizer"] is None and best["chain_state"] == "trainable"
    assert [x["loss"] for x in l2.history] == [x["loss"] for x in ref.history]
    same_state(ref, l2)
    # scoring the carried best loads the SAME fp32 weights as the uninterrupted run's best checkpoint
    s_chain = Trainer(fresh(), cfg(tmp_path / "s2"), toy, guard=G)
    s_chain.load(Path(l2.cfg.ckpt_dir) / rec["best"]["file"])
    assert s_chain.loaded_format == "trainable"
    full_best = torch.load(tmp_path / "ck1" / rec["best"]["file"], map_location="cpu", weights_only=False)
    assert all(torch.equal(full_best["trainable"][k], v) for k, v in s_chain.trainable_state().items())
    with pytest.raises(ValueError, match="not a resume point"):  # a trainable-only file is never a resume point
        (Path(l2.cfg.ckpt_dir) / "latest.json").write_text(json.dumps({"path": rec["best"]["file"], "step": 8,
                                                                        "config_hash": l2.cfg.hash()}))
        Trainer(fresh(), cfg(l2.cfg.ckpt_dir), toy, guard=G).resume()


def test_collapse_after_the_boundary_rolls_back_exactly_with_a_full_best(toy, dev, tmp_path):
    ref = Trainer(fresh(), cfg(tmp_path / "ref"), toy, eval_items=dev, guard=G)
    train_to(ref, TOTAL, collapse_at=11)
    rec, _, l2 = run_chain(toy, dev, tmp_path, collapse_at=11)
    forced = [e for e in ref.collapse_events if e["trigger"] == "forced"]
    assert rec["best"]["state"] == "full" and forced[0]["action"] == "reload_best"
    assert forced[0]["best_step"] < BOUNDARY  # the rollback target came from link 1
    assert [x["loss"] for x in l2.history] == [x["loss"] for x in ref.history]
    same_state(ref, l2)
    assert all("optimizer" not in e for e in l2.collapse_events)


def test_collapse_to_a_slim_best_keeps_the_moments_and_says_so(toy, dev, tmp_path):
    probe = Trainer(fresh(), cfg(tmp_path / "probe"), toy, eval_items=dev, guard=G)
    train_to(probe, BOUNDARY)
    probe.save()
    full = (tmp_path / "probe" / json.loads((tmp_path / "probe" / "latest.json").read_text())["path"]).stat().st_size
    rec, _, l2 = run_chain(toy, dev, tmp_path, budget=full + int(0.6 * full), collapse_at=11)
    ev = [e for e in l2.collapse_events if e["trigger"] == "forced"][0]
    assert ev["best_step"] < BOUNDARY and rec["best"]["state"] == "trainable" and ev["action"] == "reload_best"
    assert ev["optimizer"].startswith("kept") and l2.lr_scale == 0.5
    best = torch.load(Path(l2.cfg.ckpt_dir) / rec["best"]["file"], map_location="cpu", weights_only=False)
    assert l2.step >= TOTAL and best["step"] == ev["best_step"]


def test_schedule_is_continuous_across_links(toy, dev, tmp_path):
    ref = Trainer(fresh(), cfg(tmp_path / "ref"), toy, eval_items=dev, guard=G)
    lrs = []
    while ref.step < TOTAL:
        ref.train(1)
        if ref.step == 3:
            ref.set_horizon(HORIZON)
        lrs.append([g["lr"] for g in ref.opt.param_groups])
    l1 = Trainer(fresh(), cfg(tmp_path / "ck1"), toy, eval_items=dev, guard=G)
    train_to(l1, BOUNDARY)
    rec, art = export_link(l1, tmp_path)
    C.install(block(rec, art), cfg=CELL, ck_dir=tmp_path / "ck2", work=tmp_path / "w", url=art.as_uri())
    l2 = Trainer(fresh(), cfg(tmp_path / "ck2"), toy, eval_items=dev, guard=G)
    l2.resume()
    assert l2.horizon_fixed and l2.horizon == HORIZON and l2.sched.last_epoch == BOUNDARY
    l2.set_horizon(5)  # link 2's own clock projection never moves the horizon fixed in link 1
    assert l2.horizon == HORIZON
    got = []
    while l2.step < TOTAL:
        l2.train(1)
        got.append([g["lr"] for g in l2.opt.param_groups])
    assert got == lrs[BOUNDARY:]
    assert lrs[-1][0] == pytest.approx(1e-2 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (TOTAL - 2) / (HORIZON - 2)))))


# ------------------------------------------------------------------ refusals
@pytest.fixture()
def link1(toy, dev, tmp_path):
    l1 = Trainer(fresh(), cfg(tmp_path / "ck1"), toy, eval_items=dev, guard=G)
    train_to(l1, BOUNDARY)
    return export_link(l1, tmp_path)


def _refuses(rec, art, tmp, *, msg, ch=None, cell_cfg=CELL):
    with pytest.raises(C.ChainError, match=msg):
        C.install(ch or block(rec, art), cfg=cell_cfg, ck_dir=tmp / "ckx", work=tmp / "wx", url=art.as_uri())
    assert not (tmp / "ckx" / "latest.json").exists()  # nothing half-installed


def test_install_refuses_another_cell_config(link1, tmp_path):
    rec, art = link1
    _refuses(rec, art, tmp_path, msg="chain_id", cell_cfg={**CELL, "lr_schedule": "constant"})


def test_install_refuses_the_wrong_link_or_chain_length(link1, tmp_path):
    rec, art = link1
    _refuses(rec, art, tmp_path, msg="link", ch=block(rec, art, link=3, of=3))
    _refuses(rec, art, tmp_path, msg="'of'", ch=block(rec, art, link=2, of=3))


def test_install_refuses_unpinned_bytes(link1, tmp_path):
    rec, art = link1
    b = block(rec, art)
    _refuses(rec, art, tmp_path, msg="artifact art-test sha256", ch={**b, "from": {**b["from"], "artifact_sha256": "0" * 64}})
    _refuses(rec, art, tmp_path, msg="CHAIN.json sha256", ch={**b, "from": {**b["from"], "manifest_sha256": "0" * 64}})
    # a tampered checkpoint inside a re-pinned artifact: CHAIN.json (pinned) still names the original sha256
    bad = tmp_path / "tampered.tar"
    with tarfile.open(art) as src, tarfile.open(bad, "w") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name.endswith(f"/{rec['best']['file']}"):
                data = data[:-1] + bytes([data[-1] ^ 1])
            m.size = len(data)
            dst.addfile(m, io.BytesIO(data))
    tb = block(rec, bad)
    with pytest.raises(C.ChainError, match=re.escape(rec["best"]["file"])):
        C.install(tb, cfg=CELL, ck_dir=tmp_path / "cky", work=tmp_path / "wy", url=bad.as_uri())
    assert not (tmp_path / "cky" / "latest.json").exists()


def test_install_reuses_its_own_restart_and_refuses_a_foreign_checkpoint(link1, toy, dev, tmp_path):
    rec, art = link1
    ck = tmp_path / "ck2"
    C.install(block(rec, art), cfg=CELL, ck_dir=ck, work=tmp_path / "w", url=art.as_uri())
    l2 = Trainer(fresh(), cfg(ck), toy, eval_items=dev, guard=G)
    l2.resume()
    l2.train(3)
    l2.save()  # link 2 checkpointed past the installed step, then the attempt was severed
    again = C.install(block(rec, art), cfg=CELL, ck_dir=ck, work=tmp_path / "w", url="file:///nonexistent")
    assert again["reused"] and json.loads((ck / "latest.json").read_text())["step"] == BOUNDARY + 3
    other = tmp_path / "foreign"
    other.mkdir()
    (other / "latest.json").write_text("{}")
    with pytest.raises(C.ChainError, match="not installed from this chain link"):
        C.install(block(rec, art), cfg=CELL, ck_dir=other, work=tmp_path / "w2", url=art.as_uri())


def test_resume_refuses_lineage_and_config_mismatch(link1, toy, dev, tmp_path):
    rec, art = link1
    ck = tmp_path / "ck2"
    C.install(block(rec, art), cfg=CELL, ck_dir=ck, work=tmp_path / "w", url=art.as_uri())
    with pytest.raises(ValueError, match="lineage"):  # other training items (lineage hashes them)
        Trainer(fresh(), cfg(ck), toy[:-1], eval_items=dev, guard=G).resume()
    with pytest.raises(ValueError, match="config"):  # a trajectory-changing key (in the TrainConfig hash)
        Trainer(fresh(), cfg(ck, lr_head=2e-2), toy, eval_items=dev, guard=G).resume()
    # run_cell's chain gate (exit 29): link 2 that did not resume, or a cell.json of another chain / link
    cell = json.loads((ck / "cell.json").read_text())
    ch = {"link": 2, "of": 2}
    assert C.resume_check(ch, cell, CELL, resumed=True) is None
    assert "found no checkpoint" in C.resume_check(ch, cell, CELL, resumed=False)
    assert "chain" in C.resume_check(ch, cell, {**CELL, "seed": 1}, resumed=True)
    assert "resumes only link" in C.resume_check({"link": 4, "of": 4}, {**cell, "chain": {**cell["chain"], "of": 4}},
                                                 CELL, resumed=True)
    assert C.resume_check({"link": 1, "of": 2}, {}, CELL, resumed=False) is None


def test_export_refuses_what_is_not_a_finished_link(toy, dev, tmp_path):
    l1 = Trainer(fresh(), cfg(tmp_path / "ck1"), toy, eval_items=dev, guard=G)
    train_to(l1, 4)
    l1.save()
    cell = link_end_cell(l1)
    for bad, msg in (({**cell, "config": {**CELL, "seed": 9}}, "config"),
                     ({**cell, "train_clock": {"step": 3}}, "train_clock step"),
                     ({**cell, "train": {"steps": 4}}, "did not end its link"),
                     ({**cell, "chain": {"link": 2}}, "chain link")):
        p = tmp_path / "cell.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(C.ChainError, match=msg):
            C.export(ck_dir=tmp_path / "ck1", cell_path=p, cfg=CELL, ch={"link": 1, "of": 2}, out_dir=tmp_path / "o")
    p.write_text(json.dumps(cell))
    with pytest.raises(C.ChainError, match="budget"):
        C.export(ck_dir=tmp_path / "ck1", cell_path=p, cfg=CELL, ch={"link": 1, "of": 2}, out_dir=tmp_path / "o",
                 budget=1000)


# ------------------------------------------------------------------ params, sizes, schedule of a link job
def test_chain_block_validation():
    ok = {"cell_config": "jobs/m2_first_cell/x.json", "chain": {"link": 1, "of": 3}}
    assert C.chain_block(ok)["of"] == 3 and C.chain_block({}) is None
    pin = {"artifact": "art-x", "artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64, "url_env": "CHAIN_URL"}
    assert C.intermediate(C.chain_block({**ok, "chain": {"link": 2, "of": 3, "from": pin}}))
    assert not C.intermediate(C.chain_block({**ok, "chain": {"link": 3, "of": 3, "from": pin}}))
    for bad in ({"link": 0, "of": 2}, {"link": 3, "of": 2}, {"link": 1, "of": 1}, {"link": 1, "of": 2, "from": pin},
                {"link": 2, "of": 2}, {"link": 2, "of": 2, "from": {**pin, "manifest_sha256": "xyz"}},
                {"link": 2, "of": 2, "from": {**pin, "url_env": "lower"}}):
        with pytest.raises(C.ChainError):
            C.chain_block({**ok, "chain": bad})
    with pytest.raises(C.ChainError, match="cell_config"):
        C.chain_block({"chain": {"link": 1, "of": 2}})
    with pytest.raises(C.ChainError, match="eval_only"):
        C.chain_block({**ok, "eval_only": "fulldev"})


def test_size_check_r64_fits_the_artifact_and_r256_is_refused():
    """The 13.3B RWKV7 spec's LoRA shapes: r64 = 287.8M LoRA params (+ the head) -> latest full ~3.5 GB + a
    trainable-only best ~1.2 GB < the 4.9 GB budget; r256 = 1.151B -> 13.8 GB, over the 5 GiB artifact cap."""
    cell = {"parent": "rwkv7", "head": "fork", "lora_rank": 64, "best_ckpt": True}
    p = {"cell_config": "x.json", "chain": {"link": 1, "of": 2}}
    r64 = C.check(p, cell)
    assert r64["trainable"]["lora"] == 287_834_112 and r64["fits"]
    assert 3.4e9 < r64["bytes"]["latest_full"] < 3.6e9 and r64["export_bytes_worst"] < C.BUDGET < C.ARTIFACT_CAP
    with pytest.raises(C.ChainError, match="cannot chain"):
        C.check(p, {**cell, "lora_rank": 256})
    final = C.check({**p, "chain": {"link": 2, "of": 2, "from": {"artifact": "a", "artifact_sha256": "a" * 64,
                                                                  "manifest_sha256": "b" * 64, "url_env": "U"}}},
                    {**cell, "lora_rank": 256})
    assert not final["fits"] and final["trainable"]["lora"] == 1_151_336_448  # the final link exports nothing


def test_link_stop_reason_and_cell_minutes():
    ch = {"link": 1, "of": 3}
    assert C.stop_reason(100, 200, 100, 5, 10, None, ch) == "link_end"
    assert C.stop_reason(200, 200, 100, 5, 10, None, ch) == "complete"  # the global clock ran out first
    assert C.stop_reason(100, 200, 100, 10, 10, None, ch) == "complete"  # max_steps
    assert C.stop_reason(100, 200, 100, 5, 10, "collapse_max_events", ch) == "complete"
    assert C.stop_reason(100, 200, 100, 5, 10, None, {"link": 3, "of": 3}) == "complete"  # the final link ends the run
    assert C.stop_reason(100, 200, None, 5, 10, None, ch) == "complete"
    p = {"max_duration_minutes": 899, "reserve_minutes": 7,
         "phase_caps_minutes": {"setup": 25, "cell": 830, "chain_export": 10}}
    assert C.cell_minutes(p, 45 * 60) == 899 - 45 - 10 - 7
    assert C.cell_minutes(p, 45 * 60 + 1) == 899 - 46 - 10 - 7  # rounds the elapsed time up


#: D18 (stack/rwkv7/d18-v2e-joint job_params_d18_v3w_v2e_joint.json @1a75caf) caps / expected minutes / clock
D18 = {"phase_caps_minutes": {"setup": 25, "guard": 5, "heldout": 15, "parent_g12": 85, "parent_index": 65, "cell": 531,
                              "screen": 157, "report": 5},
       "expected_minutes": {"build": 15, "setup": 15, "guard": 3, "heldout": 8, "parent_g12": 65, "parent_index": 48,
                            "cell": 521, "screen": 149, "report": 4},
       "reserve_minutes": 7, "max_duration_minutes": 895, "stage": "screen"}


def test_link_caps_keep_the_launch_schedule_rule_and_plan_the_d18_chain():
    pl = C.plan(D18, {"max_minutes": 380}, 4, pre_train_min=30)
    for x in pl["links"]:
        caps = x["phase_caps_minutes"]
        assert sum(caps.values()) + D18["reserve_minutes"] == x["max_duration_minutes"] == 899  # lint chk_schedule
        mid = x["link"] < 4
        assert ("chain_export" in caps) == mid and ("screen" in caps) != mid and ("parent_g12" in caps) != mid
        assert ("chain" in caps) == (x["link"] >= 2)
        assert {"setup", "guard", "heldout", "cell"} <= set(caps)  # the guards run in every link
    l1, l2, l4 = pl["links"][0], pl["links"][1], pl["links"][3]
    assert l2["phase_caps_minutes"]["cell"] == 899 - (25 + 5 + 15 + 15 + 10) - 7 == 822
    assert l2["train_min_floor"] == 822 - 151 - 8 == 663 and l1["train_min_floor"] == 678
    assert l2["train_min_expected"] >= 750 and l4["train_min_floor"] > 300  # vs D18's 380 per job
    assert pl["max_minutes_guaranteed"] == sum(x["train_min_floor"] for x in pl["links"])


def test_valid_result():
    good = {"schema": C.SCHEMA, "verdict": "link_done", "manifest_sha256": "a" * 64, "link": 1, "of": 2, "step": 9,
            "export_bytes": 4_700_000_000}
    assert C.valid_result(good)
    for k, v in (("verdict", "measured"), ("link", 2), ("step", 0), ("export_bytes", C.ARTIFACT_CAP + 1),
                 ("manifest_sha256", "x")):
        assert not C.valid_result({**good, k: v})
    assert not C.valid_result({})


# ------------------------------------------------------------------ the runner
def test_runner_runs_the_guards_every_link_and_evals_only_in_the_final_one():
    i = JOB.index
    skip = "no parent arm (its evals run in link"
    assert i("phase guard") < i("phase heldout") < i("phase chain (install") < i(skip)
    assert i(skip) < i("ARM=$CK/parent_arm") < i("fi  # INTERMEDIATE") < i('say "phase cell"')
    assert i('say "phase cell"') < i("phase chain_export") < i("phase screen")
    # intermediate links skip the parent arm (else-branch) and exit after the export, before any eval phase
    block = JOB[i(skip):i("fi  # INTERMEDIATE")]
    assert "phase parent_" in block and "re-price" in block
    exp = JOB[i("phase chain_export"):i("phase screen")]
    assert "RC=0\n  exit 0" in exp and "size_guard.py" in exp and "--cap-bytes \"$SIZE_CAP\"" in exp
    assert "--train-deadline" in JOB and "train.chain cell-minutes" in JOB and "train.chain check" in JOB
    assert 'VALIDATOR=train.chain:valid_result' in JOB


def test_runner_parses():
    import shutil

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    r = subprocess.run([bash, "-n", str(ROOT / "jobs/m2_first_cell/first_cell_job.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _fake_code(tmp, params: dict, mode: str):
    """A code root with only what the runner touches before the TOK_S check (and its finalize): the real runner,
    lib.sh, watchdog.sh, finalize.py, size_guard.py and train.chain."""
    import shutil

    code = tmp / "code"
    for rel in ("jobs/m2_first_cell/first_cell_job.sh", "jobs/common/lib.sh", "jobs/clef_baseline/watchdog.sh",
                "jobs/common/finalize.py", "jobs/common/size_guard.py", "train/__init__.py", "train/chain.py",
                "jobs/__init__.py", "jobs/m2_first_cell/__init__.py", "jobs/m2_first_cell/summarize.py"):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, code / rel)
    (code / "jobs/m2_first_cell/shared_cell.json").write_text(json.dumps(CELL), encoding="utf-8")
    (code / f"jobs/m2_first_cell/job_params_{mode}.json").write_text(json.dumps(params), encoding="utf-8")
    return code


@pytest.mark.shell_runner
@pytest.mark.parametrize("link,of,rc,says", [
    (2, 3, 4, "chain link 2 of 3 (train.chain; intermediate=1"),  # resolved, then the TOK_S check (exit 4)
    (3, 3, 4, "chain link 3 of 3 (train.chain; intermediate=0"),
    (0, 3, 5, "chain link/of must be ints"),  # a malformed block stops in mode resolution (exit 5)
])
def test_runner_resolves_a_chain_link_mode(tmp_path, link, of, rc, says):
    import os
    import shutil
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_m2_first_cell_shell import _bash, _posix

    bash = _bash()
    if bash is None:
        pytest.skip("no POSIX bash with coreutils")
    pin = {"artifact": "art-x", "artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64, "url_env": "CHAIN_URL"}
    params = {"mode": "chaint_l", "cell_config": "jobs/m2_first_cell/shared_cell.json",
              "chain": {"link": link, "of": of, **({"from": pin} if link >= 2 else {})},
              "phase_caps_minutes": {"setup": 25}, "reserve_minutes": 7, "max_duration_minutes": 899}
    code = _fake_code(tmp_path, params, "chaint_l")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("M2FC_", "GMN_", "TOK_S", "CALL_S"))}
    env.update(M2FC_CODE=_posix(code), M2FC_WORK=_posix(tmp_path / "work"), M2FC_DATA=_posix(tmp_path / "data"),
               GMN_OUTPUT_DIR=_posix(tmp_path / "out"), GMN_CHECKPOINT_DIR=_posix(tmp_path / "ck"),
               GMN_METADATA_URL="http://127.0.0.1:9", HF_HUB_CACHE=_posix(tmp_path / "hf"), PYTHONUTF8="1",
               PATH=os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")]))
    if shutil.which("python", path=env["PATH"]) is None:
        pytest.skip("no python on PATH for the runner")
    # no pipes: the runner's watchdog leaves a ``sleep`` behind for up to a minute, which would hold a captured pipe
    r = subprocess.run([bash, _posix(code / "jobs/m2_first_cell/first_cell_job.sh"), "chaint_l"], cwd=code, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    log = (tmp_path / "work/logs/run.log").read_text(encoding="utf-8", errors="replace")
    assert r.returncode == rc, log[-2000:]
    assert says in log, log[-2000:]
    if rc == 4:
        assert "cell jobs/m2_first_cell/shared_cell.json" in log  # the SHARED cell config, not ..._<mode>.json
        res = json.loads((tmp_path / "out" / "result.json").read_text())
        assert res["exit_code"] == 4 and not (tmp_path / "out" / "M2_FIRST_CELL_DONE").exists()
