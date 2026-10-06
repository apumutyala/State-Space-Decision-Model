"""jobs/m2_first_cell: config, cost and schedule arithmetic, train-set loader guards, the held-out gate, the weighted
live rate guard, the measurement-only Index stage, the readout diagnostics, the runner's static guarantees, and the
CPU end-to-end dry run on a tiny RWKV7 (skipped when the private local inputs are absent)."""
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CELL = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json").read_text(encoding="utf-8"))
PARAMS = json.loads((ROOT / "jobs/m2_first_cell/job_params.json").read_text(encoding="utf-8"))
JOB = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
TKFAJ = {"TOK_S": 3757.0, "CALL_S": 0.117}  # job-tkfaj probe.json


# ----------------------------------------------------------------------------- config / cost / schedule
def test_cell_config_is_the_specified_cell():
    assert CELL["parent"] == "rwkv7" and CELL["head"] == "fork" and CELL["loss_arm"] == "clef"
    assert CELL["lora_rank"] == 64 and CELL["max_k"] == 16 and CELL["eval_max_t"] == max(CELL["t_ladder"]) == 2048
    assert CELL["require_guard_readiness"] and CELL["cohort_sha256"] == PARAMS["cohort_sha256"]
    assert CELL["train_set"]["pointer"] == PARAMS["train_set_pointer"]
    assert (ROOT / CELL["train_set"]["pointer"]).exists()
    assert CELL["rwkv7_wkv"] == "chunked"  # the measured path; "fla" only after the gate's ADOPT + GPU_VERIFIED
    assert f"{PARAMS['phase_caps_minutes']['cell']}-min cell phase" in CELL["max_minutes_note"]
    from heads.rwkv7_wkv import GPU_VERIFIED

    assert GPU_VERIFIED is False
    assert CELL["min_steps"] * CELL["grad_accum"] * CELL["batch_size"] >= 4800
    assert CELL["n_train_records"] * 0.9 >= CELL["max_steps"] * CELL["grad_accum"] * 0.9  # < 1 epoch at max_steps


def test_cost_worst_case_and_expected():
    from jobs.common.costs import cap_ok, expected_usd, schedule_ok, worst_case_usd

    assert worst_case_usd(PARAMS) == round((20 + 2 * 637) * 0.0495, 2) == 64.05
    assert PARAMS["phase_caps_minutes"]["parent_index"] == 65
    assert cap_ok(PARAMS) and PARAMS["hard_cap_usd"] == 70.0 and schedule_ok(PARAMS)
    assert CELL["max_minutes"] == 190  # pre-registered
    assert expected_usd(PARAMS) == round(sum(PARAMS["expected_minutes"].values()) * 0.0495, 2)
    assert 20 <= expected_usd(PARAMS) <= 30
    assert sum(PARAMS["phase_caps_minutes"].values()) + PARAMS["reserve_minutes"] == PARAMS["max_duration_minutes"]
    # training wall clock fits its phase with the non-training work (load, smoke, ladder, two cards, checkpoints)
    assert CELL["max_minutes"] + 40 <= PARAMS["phase_caps_minutes"]["cell"]


@pytest.mark.parametrize("stages,mode,cap", [("g12", "parent", "parent_g12"), ("index", "parent", "parent_index"),
                                             ("g12,index", "child", "evals")])
def test_eval_preflights_pass_at_the_measured_rates_and_stop_when_slower(stages, mode, cap):
    pytest.importorskip("numpy")
    from jobs.m2_parent_arm.score_parent import estimate

    fixed = PARAMS["evals_fixed_minutes"][mode]
    cap_min = PARAMS["phase_caps_minutes"][cap]
    est = estimate(TKFAJ["TOK_S"], stages=stages.split(","), fixed_min=fixed, mode=mode, call_s=TKFAJ["CALL_S"])
    assert est["total_minutes"] <= 0.95 * cap_min, est
    ok = estimate(TKFAJ["TOK_S"] / 1.25, stages=stages.split(","), fixed_min=fixed, mode=mode,
                  call_s=TKFAJ["CALL_S"] * 1.25)
    assert ok["total_minutes"] <= 0.95 * cap_min  # a box 25 % slower than job-tkfaj completes the phase
    k = 1.35  # a box 35 % slower than job-tkfaj's (rates and per-call latency) is over every cap: exit 15
    slow = estimate(TKFAJ["TOK_S"] / k, stages=stages.split(","), fixed_min=fixed, mode=mode, call_s=TKFAJ["CALL_S"] * k)
    assert slow["total_minutes"] > 0.95 * cap_min
    # the expected minutes in job_params are the projection (+ load), not a guess
    exp = PARAMS["expected_minutes"][cap]
    assert abs(exp - est["total_minutes"]) <= 6


def test_estimate_legacy_projection_unchanged_without_call_s():
    from jobs.m2_parent_arm.score_parent import CHILD_EXTRA_TOKENS, VOLUME_TOKENS, estimate

    e = estimate(10_000)
    assert e["total_tokens"] == sum(VOLUME_TOKENS["g12"].values()) + sum(VOLUME_TOKENS["g13"].values())
    assert e["call_s"] == 0.0 and math.isclose(e["total_minutes"], e["total_tokens"] / 10_000 / 60 + 35)
    c = estimate(10_000, mode="child")
    assert c["total_tokens"] - e["total_tokens"] == sum(CHILD_EXTRA_TOKENS.values())


def test_lora_rank_memory_estimate_leaves_headroom():
    pytest.importorskip("torch")
    from heads.backbone import BACKBONES
    from train.estimate import train_point

    pt = train_point(BACKBONES["rwkv7-13.3b"], "fork", 2048, CELL["max_k"], rank=CELL["lora_rank"], fork_mode="cache",
                     grad_ckpt=False)
    big = train_point(BACKBONES["rwkv7-13.3b"], "fork", 2048, CELL["max_k"], rank=256, fork_mode="cache",
                      grad_ckpt=False)
    assert pt["mem_gb"]["total"] <= 64 < CELL["pass_rule"]["max_peak_gb"] < big["mem_gb"]["total"] + 6


def test_min_steps_feeds_the_measured_rate_stops():
    from train.run_cell import steps_needed, too_slow

    assert steps_needed(CELL["pass_rule"], CELL["min_steps"]) == 600
    assert steps_needed({"window": 50}) == 100  # unchanged for cells without min_steps
    assert too_slow(20.0, CELL["max_minutes"], 600)["too_slow"]  # 20 s/step -> 540 steps < 600: exit 26
    assert not too_slow(8.0, CELL["max_minutes"], 600)["too_slow"]


def test_train_config_hash_unchanged_for_cells_without_warmup():
    pytest.importorskip("torch")
    from train.run_cell import train_config

    pilot = json.loads((ROOT / "jobs/m2_pilot/m2_pilot_cell.json").read_text(encoding="utf-8"))
    assert train_config(pilot, 10).warmup_steps == 0
    assert train_config(CELL, 10).warmup_steps == 50 and train_config(CELL, 10).loss.arm == "clef"


# ----------------------------------------------------------------------------- train set loader
def _write_gz(p: Path, lines: list[str]) -> str:
    text = "".join(x + "\n" for x in lines)
    with gzip.open(p, "wt", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _rec(i, extra=""):
    return {"id": f"m2:t:{i}", "schema": "ssdm-train-v0", "source": "anli", "family": "nli", "source_split": "train",
            "state": {"premise": f"synthetic premise {i} {extra}"}, "untrusted_fields": [], "labels": {"a": "x"},
            "questions": {"a": {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "x", "y": "y"}}}}


def _train_set(tmp_path, recs, n_pointer=None, content_override=None):
    from decide.schema import canonical_json
    from train.cohort import record_content_hash

    d = tmp_path / "set"
    d.mkdir()
    lines = [canonical_json(r) for r in recs]
    rs = _write_gz(d / "records.jsonl.gz", lines)
    man = [json.dumps({"id": r["id"], "sha256": hashlib.sha256(x.encode()).hexdigest()}) for r, x in zip(recs, lines)]
    ms = _write_gz(d / "m.manifest.jsonl.gz", man)
    cm = [json.dumps({"content_sha256": (content_override or {}).get(r["id"], record_content_hash(r)), "id": r["id"]},
                     sort_keys=True) for r in recs]
    cs = _write_gz(d / "m.content.jsonl.gz", cm)
    ptr = tmp_path / "p.pointer.json"
    ptr.write_text(json.dumps({"build": "test", "records_uncompressed_sha256": rs, "manifest": "m.manifest.jsonl.gz",
                               "manifest_uncompressed_sha256": ms, "content_map": "m.content.jsonl.gz",
                               "content_map_uncompressed_sha256": cs, "n_records": n_pointer or len(recs)}))
    cfg = {"train_set": {"pointer": str(ptr)}, "n_train_records": 4, "seed": 0, "cohort_sha256": CELL["cohort_sha256"]}
    return d, cfg


def test_train_set_draw_verifies_and_excludes(tmp_path):
    from train.train_set import draw

    recs = [_rec(i) for i in range(10)]
    d, cfg = _train_set(tmp_path, recs)
    got, tr, ident = draw(cfg, d)
    assert len(tr) == 4 and set(got) == set(tr) and ident["lines_manifest_verified"] == 10
    assert got[tr[0]]["request"]["questions"] and ident["held_out_in_set"] == {"by_id": 0, "by_content": 0}
    assert draw(cfg, d)[1] == tr  # deterministic
    cfg2 = {**cfg, "train_set": {**cfg["train_set"], "heldout_excluded": {"ids": tr[:1]}}}
    assert tr[0] not in draw(cfg2, d)[1]


def test_train_set_refuses_tampering_and_cohort_records(tmp_path):
    from train.cohort import Cohort
    from train.train_set import TrainSetMismatch, draw

    recs = [_rec(i) for i in range(6)]
    (tmp_path / "a").mkdir()
    d, cfg = _train_set(tmp_path / "a", recs)
    ptr = json.loads(Path(cfg["train_set"]["pointer"]).read_text())
    Path(cfg["train_set"]["pointer"]).write_text(json.dumps({**ptr, "records_uncompressed_sha256": "0" * 64}))
    with pytest.raises(TrainSetMismatch, match="records uncompressed sha256"):
        draw(cfg, d)
    held = Cohort().ids("dev")[0]  # a cohort id inside the set: the reviewed build never has one
    (tmp_path / "b").mkdir()
    d, cfg = _train_set(tmp_path / "b", recs + [{**_rec(99), "id": held}])
    with pytest.raises(TrainSetMismatch, match="cohort-held-out"):
        draw(cfg, d)
    (tmp_path / "c").mkdir()
    d, cfg = _train_set(tmp_path / "c", recs, content_override={r["id"]: "f" * 64 for r in recs})
    with pytest.raises(TrainSetMismatch, match="differ in content"):
        draw({**cfg, "n_train_records": 6}, d)


def test_heldout_gate_passes_only_on_the_pinned_hits(tmp_path):
    from train.train_set import heldout_gate

    recs = [_rec(i) for i in range(4)] + [_rec(9, "he went to the teacher rather than relying on the others")]
    d, cfg = _train_set(tmp_path, recs)
    from safety_eval.attacks import D1_TEMPLATES_B, template_set_sha256

    sha = template_set_sha256(D1_TEMPLATES_B)
    assert heldout_gate(cfg, d) == 3  # an unpinned hit
    pinned = {**cfg, "train_set": {**cfg["train_set"], "heldout_excluded": {"ids": ["m2:t:9"], "templates_sha256": sha}}}
    assert heldout_gate(pinned, d, tmp_path / "h.json") == 0
    assert json.loads((tmp_path / "h.json").read_text())["pass"]
    wrong = {**cfg, "train_set": {**cfg["train_set"], "heldout_excluded": {"ids": ["m2:t:9", "m2:t:1"],
                                                                         "templates_sha256": sha}}}
    assert heldout_gate(wrong, d) == 3  # a pinned id that no longer hits: the file or the scanner changed


def test_cell_pins_exactly_the_observed_heldout_hits():
    assert CELL["train_set"]["heldout_excluded"]["ids"] == ["m2:mmlu_aux:21118", "m2:mmlu_aux:39640",
                                                            "m2:mmlu_aux:39643"]
    from safety_eval.attacks import D1_TEMPLATES_B, template_set_sha256

    assert CELL["train_set"]["heldout_excluded"]["templates_sha256"] == template_set_sha256(D1_TEMPLATES_B)


def test_parallel_cell_items_equal_serial():
    pytest.importorskip("torch")
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE, toy_records
    from train.guard import SplitGuard
    from train.run_cell import cell_items

    recs = {r["id"]: r for r in toy_records(24, seed=3)}
    cfg = {"layout": "ordered", "template": TOY_TEMPLATE, "max_k": 16}
    g = SplitGuard.synthetic_only("t")
    a = cell_items(recs, sorted(recs), cfg, ByteEncoder(), g, 2048, workers=1)
    b = cell_items(recs, sorted(recs), cfg, ByteEncoder(), g, 2048, workers=2)
    assert [(x.id, x.row.ids, x.label) for x in a] == [(x.id, x.row.ids, x.label) for x in b] and len(a) == 48


# ----------------------------------------------------------------------------- weighted live rate guard
def _child(tmp_path, per_unit_s, n=40):
    prog = tmp_path / "p.jsonl"
    code = (f"import json,time,pathlib\np=pathlib.Path({str(prog)!r})\ntime.sleep(0.5)\n"
            f"pathlib.Path(str(p)+'.total.json').write_text(json.dumps({{'total_weight': {n * 10}, 'n_units': {n}}}))\n"
            f"for i in range({n}):\n    time.sleep({per_unit_s})\n"
            f"    open(p,'a').write(json.dumps({{'w': 10}})+'\\n')\n")
    return prog, [sys.executable, "-c", code]


def test_run_weighted_stops_a_slow_phase_and_passes_a_fast_one(tmp_path):
    from jobs.common.rate_guard import EXIT_TOO_SLOW, project_weighted, run_weighted

    assert project_weighted(60, 120, 25, 100) == (60 + 60 * 4) / 60
    prog, cmd = _child(tmp_path, 0.3)  # 40 x 0.3 s = 12 s of scoring against a 0.1-min (6 s) cap
    rep = tmp_path / "r.json"
    assert run_weighted(cmd, prog, cap_min=0.1, check_after_min=0.03, poll_s=0.2, report=rep, min_units=3) == EXIT_TOO_SLOW
    assert "> cap" in json.loads(rep.read_text())["stopped"]
    (tmp_path / "f").mkdir()
    prog, cmd = _child(tmp_path / "f", 0.01)
    assert run_weighted(cmd, prog, cap_min=1.0, check_after_min=0.005, poll_s=0.1, min_units=3) == 0


# ----------------------------------------------------------------------------- measurement stage + diagnostics
def test_index_stage_never_enters_promotion_or_exit_code():
    from train.m2_gates import EXIT, exit_code, promotion_verdict, valid_report

    v = {"g12": {"accept": True, "failed": []}, "index": {"accept": False, "failed": ["g13_capability"]}}
    pv = promotion_verdict(v)
    assert pv["verdict"] == "accept" and pv["stages"] == ["g12"] and exit_code(v) == 0
    assert exit_code({"index": v["index"]}) == EXIT["usage"]
    rep = {"schema": "ssdm.m2.gates.v1", "promotion": pv, "verdicts": v, "identities": {}}
    assert valid_report(rep)


def test_index_measurement_keeps_rows_both_arms_refuse_for_the_same_reason(monkeypatch):
    """Dry-run finding: 8 of the 2,008 subset rows exceed the 16,384-token input cap in BOTH arms (job-tkfaj too). The
    strict paired criteria require every row ok, so before this fix the result had no scores (exit 9 after ~9 h)."""
    from scripts.m2_e2e_dryrun import ensure_kit

    try:
        ensure_kit()
    except ImportError:
        pytest.skip("Decision Index kit not importable")
    from train import m2_gates as G
    from train import promotion as P

    seen = {}

    def fake_ret(rows, c, p, B, seed):
        seen["rows"] = [r["_evaluation"]["run_id"] for r in rows]
        return {"identical_items": False, "reason": "stub"}

    monkeypatch.setattr(P, "retention_comparison", fake_ret)
    rows = [{"_evaluation": {"run_id": f"r{i}"}} for i in range(4)]
    err = json.dumps({"code": "input_tokens_exceeded"})
    ok = {"status": "ok"}
    child = {"r0": ok, "r1": {"status": "exhausted", "error": err}, "r2": ok, "r3": {"status": "error", "error": "x"}}
    parent = {"r0": ok, "r1": {"status": "exhausted", "error": err}, "r2": ok, "r3": ok}
    import harness.bootstrap as HB

    monkeypatch.setattr(HB, "compare", lambda *a, **k: {"candidate": 1, "baseline": 2, "diff": 3})
    import harness.score as S

    monkeypatch.setattr(S, "decompose", lambda rows, res: None)
    import harness.calibration as HC

    monkeypatch.setattr(HC, "fields", lambda rows, res: None)
    monkeypatch.setattr(HC, "report", lambda f: {})
    out = G.index_measurement(rows, child, parent, P.MARGINS, n_boot=10)
    assert seen["rows"] == ["r0", "r2", "r3"]  # r1 refused by both for the same reason; r3 only by the child: kept
    assert out["rows_not_ok_in_both_arms"]["n"] == 1 and out["scores"]["child"] == 1


def _row(rid, gold, opts=("a", "b", "c")):
    return {"_evaluation": {"run_id": rid, "catalog_id": 1, "group_id": rid},
            "state": "s", "questions": {"q": {"type": "choice", "criteria": {o: o.upper() for o in opts}}},
            "expected": {"q": gold}}


def test_relabel_reverses_the_presented_order_and_maps_back():
    from decide.schema import canonical_options
    from jobs.m2_first_cell.readout_diag import relabel

    row = _row("r1", "alpha", ("delta", "alpha", "charlie"))
    f, bf = relabel(row, "forward")
    r, br = relabel(row, "reversed")
    pres_f = [bf["q"][o] for o, _ in canonical_options(f["questions"]["q"])]
    pres_r = [br["q"][o] for o, _ in canonical_options(r["questions"]["q"])]
    assert pres_f == ["alpha", "charlie", "delta"] and pres_r == pres_f[::-1]
    assert [d for _, d in canonical_options(r["questions"]["q"])] == ["DELTA", "CHARLIE", "ALPHA"]


def test_order_diagnostic_detects_a_first_option_reader():
    from jobs.m2_first_cell.readout_diag import order_diagnostic, paired_order, position_bias

    rows = [_row(f"r{i}", "abc"[i % 3]) for i in range(30)]

    def first_reader(req):  # always picks the first PRESENTED option
        from decide.schema import canonical_options

        q = req["questions"]["q"]
        c = canonical_options(q)[0][0]
        return {"answers": {"q": {"choice": c}}, "ssdm": {"status": "ok"}}

    def oracle(req):
        q = req["questions"]["q"]
        gold = next(r for r in rows if r["state"] == req["state"])["expected"]["q"]
        c = next(k for k in q["criteria"] if k.endswith("-" + gold) or k == gold)
        return {"answers": {"q": {"choice": c}}, "ssdm": {"status": "ok"}}

    a = order_diagnostic(first_reader, rows)
    s = a["summary"]
    assert s["agreement"] == 0.0 and s["p_chose_presented_first"] == {"forward": 1.0, "reversed": 1.0}
    rows[:] = [{**r, "state": r["_evaluation"]["run_id"]} for r in rows]  # the oracle finds its row by state
    b = order_diagnostic(oracle, rows)
    assert b["summary"]["acc_forward"] == b["summary"]["acc_reversed"] == 1.0 and b["summary"]["agreement"] == 1.0
    p = paired_order(b, order_diagnostic(first_reader, rows), n_boot=100)
    assert p["identical_questions"] and p["agreement"]["diff"]["point"] == 1.0
    res = {r["_evaluation"]["run_id"]: {"status": "ok", "response": {"answers": {"q": {"choice": "a"}}}} for r in rows}
    pb = position_bias(rows, res)
    assert pb["p_chosen_first"] == 1.0 and math.isclose(pb["p_gold_first"], 1 / 3) and pb["n_questions"] == 30


# ----------------------------------------------------------------------------- runner static guarantees
def test_runner_orders_guards_before_training_and_prices_every_eval_first():
    i = JOB.index
    train = i("python -m train.run_cell")
    for s in ("score_parent estimate", "jobs.m2_pilot.guard_ready", "train.train_set heldout", "train.train_set verify",
              "sha256sum -c \"$CODE/jobs/rwkv7/snapshot.sha256\"", "fla_guard.py"):
        assert i(s) < train, s
    # review fix 1: the parent arm (live rate guard) and the measured re-price run BEFORE any training spend
    assert i("jobs.m2_parent_arm.score_parent --parent") < i("-m jobs.m2_first_cell.reprice") < train < i("-m train.m2_gates")
    assert "export_phase heldout" in JOB  # review fix 3
    assert '--report "$ARM/rate-parent-$STAGE.json"' in JOB  # rate reports survive a restart (reprice reads them)
    assert JOB.count("rate_guard run-weighted") == 2  # parent stages (loop) + child evals
    assert "jobs/common/finalize.py" in JOB and not re.search(r"touch [^\n]*DONE", JOB)
    assert not re.search(r"timeout \d+m", JOB)  # every cap comes from job_params
    assert "--final" not in JOB and "SSDM_UNSEAL_TEST" not in JOB and "--slice test" not in JOB
    assert "case $r in 0|40)" in JOB  # a G12 reject is a measured outcome here
    assert "--stage g12+index" in JOB and "VALIDATOR=jobs.m2_first_cell.summarize:valid_result" in JOB
    for ph in PARAMS["phase_caps_minutes"]:
        assert f"cap {ph}" in JOB or f"cap \"$3\"" in JOB or f"cap parent_$STAGE" in JOB, ph


@pytest.mark.parametrize("var", ["TOK_S", "CALL_S"])
def test_rates_unset_give_exit_4(var):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    lines = [ln for ln in JOB.splitlines() if var in ln and "fail 4" in ln]
    script = ("set -Euo pipefail\nfail() { echo \"FAIL($1): $2\"; exit \"$1\"; }\n"
              "num() { case \"$1\" in *[!0-9.]*|\"\") return 1 ;; esac; }\nTOK_S=1; CALL_S=1\n"
              f"unset {var}\n" + "\n".join(lines) + "\necho PAST")
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 4 and "PAST" not in r.stdout
    r = subprocess.run([bash, "-c", script.replace(f"unset {var}", f"{var}=abc")], capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 4


def test_summarize_validator_refuses_incomplete_results():
    from jobs.m2_first_cell.summarize import valid_result

    assert not valid_result({}) and not valid_result({"schema": "ssdm.m2.first_cell.v1", "verdict": "measured"})


# ----------------------------------------------------------------------------- CPU end-to-end dry run (tiny RWKV7)
def test_first_cell_dry_run_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from scripts.m2_first_cell_dryrun import missing_inputs

    if missing_inputs():
        pytest.skip(f"local private inputs absent: {missing_inputs()}")
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/m2_first_cell_dryrun.py"), "--out",
                        str(tmp_path / "dry"), "--role-n", "12", "--index-n", "24"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=1500)
    rec = json.loads((tmp_path / "dry" / "receipt.json").read_text(encoding="utf-8"))
    assert r.returncode == 0 and rec["all_ok"] and rec["result_valid"], r.stdout[-3000:]
    res = rec["result"]
    assert res["training"]["steps"] == 6 and res["training"]["data"]["train_set"]["n_pool"] == 149_997
    assert res["index_dev_subset"]["rows"] >= 24 and res["steer_hijack_d1a"]["complete"]
    assert set(res["index_dev_subset"]["child_minus_parent"]["areas"]) == set(res["index_dev_subset"]["child"]["areas"])
    assert res["readout_diagnostics"]["order"]["paired"]["identical_questions"]
    assert "PENDING CODEX APPROVAL (#36)" in res["promotion"]["margins_status"]
    assert [p["phase"] for p in rec["phases"]].index("reprice") < [p["phase"] for p in rec["phases"]].index("cell")
    assert res["index_by_prompt_length"]["split_t"] == 2048


# ----------------------------------------------------------------------------- review fixes (FIX-THEN-LAUNCH)
def test_training_budget_is_cumulative_over_restarts():
    from train.run_cell import train_seconds_used

    assert train_seconds_used({"train_clock": {"step": 400, "used_s": 3000.0}}, resumed=True) == 3000.0
    assert train_seconds_used({"train_clock": {"step": 400, "used_s": 3000.0}}, resumed=False) == 0.0
    assert train_seconds_used({}, resumed=True) == 0.0
    src = (ROOT / "train/run_cell.py").read_text(encoding="utf-8")
    assert 't_end = t_start + 60 * cfg["max_minutes"] - used_before' in src
    assert src.count('cell["train_clock"] = {"step": tr.step') == 2  # every checkpoint and the final save


def _rate_report(d, st, scoring_min, exit_=0):
    (d / f"rate-parent-{st}.json").write_text(json.dumps({"exit": exit_, "scoring_started_min": 2.0,
                                                           "elapsed_min": 2.0 + scoring_min}))


@pytest.mark.parametrize("slow,code", [(1.0, 0), (1.25, 0), (1.3, 15)])
def test_reprice_stops_a_slow_box_before_training(tmp_path, slow, code):
    from jobs.m2_first_cell.reprice import main
    from jobs.m2_parent_arm.score_parent import estimate

    for st in ("g12", "index"):
        p = estimate(TKFAJ["TOK_S"], stages=[st], fixed_min=0, mode="parent", call_s=TKFAJ["CALL_S"])["total_minutes"]
        _rate_report(tmp_path, st, slow * p)
    args = ["--arm-dir", str(tmp_path), "--params", str(ROOT / "jobs/m2_first_cell/job_params.json"),
            "--tok-s", str(TKFAJ["TOK_S"]), "--call-s", str(TKFAJ["CALL_S"]), "--out", str(tmp_path / "r.json")]
    assert main(args) == code
    assert math.isclose(json.loads((tmp_path / "r.json").read_text())["factor"], slow)


def test_reprice_without_reports_keeps_the_setup_preflight(tmp_path):
    from jobs.m2_first_cell.reprice import main

    assert main(["--arm-dir", str(tmp_path), "--params", str(ROOT / "jobs/m2_first_cell/job_params.json"),
                 "--tok-s", "3757", "--call-s", "0.117"]) == 0


def test_index_lengths_file_matches_the_subset_and_cell():
    from jobs.common.dev_subset import load_subset
    from jobs.m2_first_cell import index_lengths as IL

    d = IL.load()
    sub = load_subset()
    assert d["subset_ids_sha256"] == sub["ids_sha256"] and sorted(d["lengths"]) == sorted(sub["run_ids"])
    assert d["split_t"] == max(CELL["t_ladder"]) == 2048 and d["n_over_split"] == 455 and d["n_unmeasurable"] == 0


def test_group_bins_never_cut_a_group():
    from jobs.m2_first_cell.index_lengths import group_bins

    rows = [{"_evaluation": {"run_id": f"r{i}", "catalog_id": 1, "group_id": f"g{i // 2}"}} for i in range(6)]
    lengths = {"r0": 100, "r1": 3000, "r2": 100, "r3": 200, "r4": None, "r5": 50}
    b = group_bins(rows, lengths)
    ids = {k: [r["_evaluation"]["run_id"] for r in v] for k, v in b.items()}
    assert ids == {"<=2048": ["r2", "r3"], ">2048": ["r0", "r1"], "unmeasurable": ["r4", "r5"]}


def test_no_literal_backslash_n_continuations_in_job_scripts():
    """A literal backslash-n where a line continuation was meant passes `bash -n` but hands the command a stray 'n'
    argument (first_cell_job.sh reprice: argparse rc 2 -> exit 5 right after the paid parent arm). Catch it in every
    runner."""
    import re
    bad = re.compile(r"[\\]n[ \t]+--")  # a backslash, the letter n, blanks, then an option
    assert bad.search('--call-s "$CALL_S" ' + chr(92) + "n  --out") and not bad.search("--call-s x " + chr(92))
    hits = [(str(p.relative_to(ROOT)), i + 1) for p in sorted((ROOT / "jobs").rglob("*.sh"))
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()) if bad.search(line)]
    assert not hits, hits
