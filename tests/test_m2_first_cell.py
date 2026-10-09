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
    # job-tvuat ADOPT (rule v1-A2): ckpt_rows_lean; "fla" only after the fla gate's ADOPT + GPU_VERIFIED
    assert {k: CELL[k] for k in ("act_ckpt", "row_batch", "row_tokens", "rwkv7_wkv", "t_ladder")} == {
        "act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean", "t_ladder": [512, 1024, 2048]}
    assert CELL["row_batch"] == CELL["batch_size"] * CELL["grad_accum"]  # 8 rows per optimiser step (grouped by row_tokens per call)
    assert f"{PARAMS['phase_caps_minutes']['cell']}-min cell phase" in CELL["max_minutes_note"]
    from heads.rwkv7_wkv import GPU_VERIFIED

    assert GPU_VERIFIED is False
    assert CELL["min_steps"] * CELL["grad_accum"] * CELL["batch_size"] >= 4800
    assert CELL["n_train_records"] * 0.9 >= CELL["max_steps"] * CELL["grad_accum"] * 0.9  # < 1 epoch at max_steps


def test_cost_worst_case_and_expected():
    from jobs.common.costs import cap_ok, expected_usd, schedule_ok, worst_case_usd

    assert worst_case_usd(PARAMS) == round((60 + 2 * 637) * 0.0495, 2) == 66.03
    assert PARAMS["build_timeout_minutes"] >= 60  # launch lint L-09
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
    held = Cohort().ids("dev")[0]  # a cohort id inside the set
    (tmp_path / "b").mkdir()
    d, cfg = _train_set(tmp_path / "b", recs + [{**_rec(99), "id": held}])
    # since #105, cohort v3 (which holds every v2 role byte for byte) is removed from the pool BEFORE the draw, by id,
    # content and source item, so the held-out record can never be drawn even when the whole pool is drawn
    got, tr, ident = draw({**cfg, "n_train_records": len(recs)}, d)
    assert held not in tr and ident["cohort_v3_excluded"]["n"] >= 1 and ident["cohort_v3_excluded"]["drawn_overlap"] == 0
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
    # #102: a run that accepts but did not run both G12 and G13 is partial (44), never 0; index never counts
    assert pv["verdict"] == "accept" and pv["stages"] == ["g12"] and exit_code(v) == EXIT["partial"] == 44
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
    assert "case $r in 0|40|43|44|45)" in JOB  # PR #102: 40 / 43 / 44 / 45 are measured, not promotable
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
@pytest.mark.xfail(strict=True, reason="attack guard (train.attack_guard): the first-cell 150k config trains on the 8,049 D1-A eval-template rows and pins no train_set.attack_clean spec, so its heldout phase refuses (exit 3 -> job 33). Pin data/manifests/attack_clean/sample-150000-seed20261006.cell-m2_first_cell_rwkv7.json to re-enable.")
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
    ts = res["training"]["data"]["train_set"]
    # 150,000 - 3 pinned D1-B exclusions - cohort v3 (#105, removed before the draw: 38,202 records)
    assert res["training"]["steps"] == 6 and ts["n_pool"] == 149_997 - ts["cohort_v3_excluded"]["n"] == 111_795
    assert ts["cohort_v3_excluded"]["drawn_overlap"] == 0
    assert res["index_dev_subset"]["rows"] >= 24 and res["steer_hijack_d1a"]["complete"]
    assert set(res["index_dev_subset"]["child_minus_parent"]["areas"]) == set(res["index_dev_subset"]["child"]["areas"])
    assert res["readout_diagnostics"]["order"]["paired"]["identical_questions"]
    assert "#36" in res["promotion"]["margins_status"] and res["promotion"]["promotable"] is False
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


# ----------------------------------------------------------------------------- job-tvuat ADOPT wiring (PR #86 hook)
def test_execution_flags_reach_the_trainer_and_both_arms_read_the_wkv():
    pytest.importorskip("torch")
    from train.run_cell import train_config
    from train.trainer import with_execution

    t = with_execution(train_config(CELL, 10), CELL)
    assert (t.act_ckpt, t.row_batch, t.row_tokens) == (True, 8, 8192)
    assert t.hash() == train_config(CELL, 10).hash()  # execution flags are not in the checkpoint identity
    for f in ("train/run_cell.py", "train/m2_gates.py"):
        assert "with_execution(train_config(cfg," in (ROOT / f).read_text(encoding="utf-8"), f
    assert 'wkv=cfg.get("rwkv7_wkv")' in (ROOT / "train/run_cell.py").read_text(encoding="utf-8")
    assert 'wkv=cfg.get("rwkv7_wkv")' in (ROOT / "jobs/m2_parent_arm/score_parent.py").read_text(encoding="utf-8")
    from heads.rwkv7_wkv import LEAN

    assert CELL["rwkv7_wkv"] in LEAN  # needs no GPU gate (CPU equivalence, tests/test_cell_perf.py)


def test_training_reprice_from_the_measured_step():
    step = PARAMS["measured_rates_reference"]["train_step_s"]
    assert step == 8.25
    steps = int(60 * CELL["max_minutes"] / step)
    assert steps == 1381 or steps == 1382  # job-tvuat projection: 1,382 steps / 11,056 rows in 190 min
    assert CELL["max_steps"] > steps and CELL["n_train_records"] * 0.9 > steps * CELL["row_batch"]  # < 1 epoch
    assert 60 * CELL["max_minutes"] / CELL["min_steps"] >= 1.25 * step  # exit-26 floor tolerates a >= 25 % slower box


def test_decision_rule_constants_pinned():
    """job_params decision_rule = train/promotion.py: the issue #36 margins (approved in PR #102) and the G12
    clean-accuracy drop."""
    from train import promotion

    assert PARAMS["decision_rule"] == "train/promotion.py"
    assert {k: promotion.MARGINS[k] for k in ("m_cap", "m_steer", "m_ece")} == {"m_cap": 1.0, "m_steer": 0.02,
                                                                                 "m_ece": 0.01}
    assert promotion.CLEAN_ACC_DROP == 0.01


def test_registered_for_the_launch_lint():
    from jobs.common import costs

    assert costs.PARAMS["m2_first_cell"] == "jobs/m2_first_cell/job_params.json"
    tree = (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert "`jobs/m2_first_cell/` | $66.03 | $70 |" in tree
    assert f"{costs.worst_case_usd(PARAMS):.2f}" in (ROOT / "jobs/m2_first_cell/JOB_SPEC.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- m2_gates exit 43 (Codex #36, PR #102)
@pytest.mark.parametrize("code,measured", [(0, True), (40, True), (43, True), (44, True), (45, True), (41, False),
                                           (10, False), (2, False)])
def test_evals_exit_43_is_a_measurement_not_a_crash(code, measured):
    """Until PR #102 lands the #36 margins, m2_gates exits 43 (promotion PENDING). The runner must treat it like 0 / 40:
    no fail(), so the report phase writes the result and exports the checkpoint after the evals export."""
    bash = shutil.which("bash")
    if not bash or re.search(r"system32|windowsapps", bash, re.I):
        pytest.skip("no POSIX bash")
    lines = JOB.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("case $r in 0|40|43|44|45)"))
    block = "\n".join(lines[i:i + 2])
    script = ("set -Euo pipefail\nfail() { echo \"FAIL($1): $2\"; exit \"$1\"; }\n"
              f"r={code}\n{block}\necho MEASURED")
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)
    assert ("MEASURED" in r.stdout and r.returncode == 0) is measured, r.stdout
    assert JOB.index("gate \"evals\" $r; export_phase evals") < JOB.index("case $r in 0|40|43|44|45)")  # evals exported first
    assert JOB.index("case $r in 0|40|43|44|45)") < JOB.index("train.ckpt_export inference")  # then the checkpoint export


def test_pending_partial_underpowered_are_valid_measured_results():
    """PR #102 exit codes: 43 pending / 44 partial / 45 underpowered = trained + measured, not promotable, with the
    reason; only 0 is promotable (and this cell, G12 + index without G13, can never get 0)."""
    from jobs.m2_first_cell.summarize import MEASURED_EVAL_EXITS, promotion_block

    assert set(MEASURED_EVAL_EXITS) == {0, 40, 43, 44, 45}
    for code in (40, 43, 44, 45):
        pv = promotion_block({"exit_code": code, "promotion": {"verdict": "accept", "promotable": False}})
        assert pv["promotable"] is False and pv["evals_exit_code"] == code
        assert pv["not_promotable_reason"] == MEASURED_EVAL_EXITS[code]
    assert "underpowered" in MEASURED_EVAL_EXITS[45] and "partial" in MEASURED_EVAL_EXITS[44]
    ok = promotion_block({"exit_code": 0, "promotion": {"verdict": "accept", "promotable": True}})
    assert ok["promotable"] is True and ok["not_promotable_reason"] is None
    src = (ROOT / "jobs/m2_first_cell/summarize.py").read_text(encoding="utf-8")
    assert 'in MEASURED_EVAL_EXITS' in src and '"verdict": "measured"' in src


def test_launch_blocks_and_data_pins():
    p = PARAMS
    assert p["submit"]["command"] == "bash /opt/ssdm/m1/jobs/m2_first_cell/first_cell_job.sh"
    assert p["submit"]["env"] == {"TOK_S": "3757", "CALL_S": "0.117"}
    assert p["context"]["cmd_rewrite"] == ["jobs/rwkv7/rwkv7_job.sh", "jobs/m2_first_cell/first_cell_job.sh"]
    dests = {d["dest"] for d in p["context"]["data"]}
    assert dests == {"m2/records.jsonl.gz", "slices/dev-rows.jsonl.gz", "m2-150k/records.jsonl.gz",
                     "m2-150k/sample-150000-seed20261006.manifest.jsonl.gz",
                     "m2-150k/sample-150000-seed20261006.content.jsonl.gz", "suite-0.2/selected-rows.jsonl.gz",
                     "suite-0.2/added-rows.jsonl.gz", "suite-0.2/excluded-questions.json",
                     "parent_arm_seed/arm.json", "parent_arm_seed/units-g12.json", "parent_arm_seed/units-index.json"}
    from jobs.common.launch import data_pin, pins

    files = {"jobs/m2_first_cell/job_params.json": ((ROOT / "jobs/m2_first_cell/job_params.json").read_bytes(), 0)}
    for d in p["context"]["data"]:  # every shipped file has a pin: a line under its context path, or a job_params key
        if "pin_json" in d:  # the L-34 parent arm seed: pinned by job_params parent_arm_seed
            assert d["pin_json"][0] == "jobs/m2_first_cell/job_params.json" and len(data_pin(files, d)) == 64, d
            continue
        assert d["dest"] in pins((ROOT / d["pin"]).read_text(encoding="utf-8")), d
    # the runner reads exactly these context paths (M2FC_DATA defaults to /data)
    for rel in ("$DATA/m2-150k", "$DATA/m2/records.jsonl.gz", "$DATA/suite-0.2", "$DATA/slices/dev-rows.jsonl.gz",
                "$DATA/parent_arm_seed"):
        assert rel in JOB, rel


def test_parent_arm_identity_carries_the_wkv_and_refuses_another(tmp_path):
    """PR #74 review N1: a parent arm scored with "chunked" (e.g. an old cache) must not pass for a chunked_lean child."""
    from train import m2_gates as G

    exp = G.expected_parent_identity("rwkv7", wkv=CELL["rwkv7_wkv"])
    assert exp["wkv"] == "chunked_lean" and G.expected_parent_identity("rwkv7")["wkv"] == "chunked"
    ident = {"stage": "index", "id": "x"}
    for name, wkv in (("lean", "chunked_lean"), ("old", "chunked")):
        G.write_arm_cache(tmp_path / name, "index", {"stage": "index"}, ident,
                          {"parent": {"parent": "rwkv7", "repo": exp["repo"], "revision": exp["revision"], "wkv": wkv}})
    G.read_arm_cache(tmp_path / "lean", "index", ident, exp)
    with pytest.raises(G.IdentityMismatch, match="wkv"):
        G.read_arm_cache(tmp_path / "old", "index", ident, exp)
    src = (ROOT / "jobs/m2_parent_arm/score_parent.py").read_text(encoding="utf-8")
    assert '"wkv": str(plug.kernel_stack.get("wkv_implementation"))' in src
    assert 'expected_parent_identity(parent, prov, cfg.get("rwkv7_wkv"))' in (ROOT / "train/m2_gates.py").read_text(
        encoding="utf-8")


def test_tvuat_receipt_backs_the_execution_config_and_the_repricing():
    """PR #74 review N2: the job-tvuat decision the re-pricing rests on is committed, and the cell uses exactly it."""
    rd = ROOT / "receipts/2026-10-07-D-cell-step-profile-tvuat"
    d = json.loads((rd / "decision.json").read_text(encoding="utf-8"))
    rec = json.loads((rd / "receipt.json").read_text(encoding="utf-8"))
    assert d["verdict"] == rec["verdict"] == "ADOPT" and d["chosen"] == "ckpt_rows_lean" and rec["job"] == "job-tvuat"
    assert {k: CELL[k] for k in d["cell_config"]} == d["cell_config"]
    assert round(d["projection"]["median_step_s"], 2) == PARAMS["measured_rates_reference"]["train_step_s"]
    assert d["projection"]["steps_190min"] == 1382 and d["variants"]["ckpt_rows_lean"]["oom_steps"] == []
    for f, h in rec["files_sha256"].items():
        assert hashlib.sha256((rd / f).read_bytes()).hexdigest() == h, f
    assert "job-tvuat" in (ROOT / "LEDGER.md").read_text(encoding="utf-8")
    assert "receipts/2026-10-07-D-cell-step-profile-tvuat" in (ROOT / "jobs/m2_first_cell/JOB_SPEC.md").read_text(
        encoding="utf-8")


def test_tvuat_receipt_bytes_survive_an_autocrlf_checkout(tmp_path):
    """PR #74 final review N1: receipts are byte-exact (``receipts/** -text``), so a core.autocrlf=true checkout or
    archive keeps the tvuat MANIFEST.jsonl (and every receipt file) equal to its recorded sha256."""
    import io
    import tarfile

    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    rd = "receipts/2026-10-07-D-cell-step-profile-tvuat"
    attr = subprocess.run([git, "-C", str(ROOT), "check-attr", "text", "--", f"{rd}/MANIFEST.jsonl"],
                          capture_output=True, text=True, check=True).stdout
    assert attr.strip().endswith("text: unset"), attr
    raw = subprocess.run([git, "-C", str(ROOT), "-c", "core.autocrlf=true", "archive", "HEAD", "--", rd],
                         capture_output=True, check=True).stdout
    rec = json.loads((ROOT / rd / "receipt.json").read_text(encoding="utf-8"))
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        got = {m.name.rsplit("/", 1)[1]: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    assert b"\r" not in got["MANIFEST.jsonl"]
    for f, h in rec["files_sha256"].items():
        assert hashlib.sha256(got[f]).hexdigest() == h, f


# ----------------------------------------------------------------------------- job-uhxb5: smoke OOM (L-34)
def _smoke_bytes(act_ckpt_flag: bool, monkeypatch):
    """Run train.run_cell.smoke_stage on a tiny RWKV7 (chunked_lean, like the cell) and return (smoke record,
    per-mode saved-for-backward bytes of build_head_input, per-mode act_ckpt state seen inside the call, backbone)."""
    import torch

    from heads import act_ckpt, features
    from heads.backbone import tiny_hf_backbone
    from heads.rwkv7_wkv import select_wkv
    from jobs.fla_rwkv7 import train_mix as TM
    from train import run_cell
    from train.data import Item

    def shape(T, conts):  # the tests/test_cell_perf.py row layout
        a = T - 2 * len(conts) - 2
        return {"T": T, "conts": conts, "qtype": "choice", "state_span": [4, a], "field_span": [1, 3],
                "option_spans": [[a + 2 * j, a + 2 * j + 1] for j in range(len(conts))]}

    shapes = [shape(150, [3, 2, 4]), shape(120, [2, 5])]
    rows = TM.rows_from_shapes(shapes, vocab=260, seed=0)
    items = [Item(x.id, x.row, 0) for x in rows]
    bb = tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache")
    select_wkv(bb.model, CELL["rwkv7_wkv"])
    cfg = {**CELL, "act_ckpt": act_ckpt_flag}
    real = features.build_head_input
    seen, saved = {}, {}

    def spy(backbone, rws, **kw):
        tot = [0]

        def pack(t):
            tot[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = real(backbone, rws, **kw)
        seen[backbone.fork_mode] = act_ckpt.is_enabled(backbone.model)
        saved[backbone.fork_mode] = tot[0]
        return out

    with monkeypatch.context() as mp:  # undone on return: a second call must not wrap this spy
        mp.setattr(features, "build_head_input", spy)
        rec = run_cell.smoke_stage(bb, items, cfg, cuda=False)
    return rec, saved, seen, bb


def test_smoke_runs_under_the_cells_activation_checkpointing(monkeypatch):
    """job-uhxb5 (L-34): the cell trains with per-block activation checkpointing (act_ckpt, job-tvuat ADOPT), but the
    pre-training smoke ran its cache-vs-recompute fork comparison WITHOUT it (the flag was applied only inside the
    Trainer). The recompute fork batches every (row, option) continuation through the 13.3B backbone with full
    activations kept: CUDA OOM at 77 GiB after 3 minutes, 71 minutes into the paid job. The smoke must run under the
    cell's execution flags (both modes), keep far fewer activations, still pass, and leave the backbone as found."""
    pytest.importorskip("torch")
    from heads import act_ckpt
    from heads.rwkv7_lean import NAME
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    try:
        rec_on, saved_on, seen_on, bb_on = _smoke_bytes(True, monkeypatch)
        rec_off, saved_off, seen_off, _ = _smoke_bytes(False, monkeypatch)
    finally:
        M.RWKV7_WKV_FUNCTIONS.pop(NAME, None)
    assert CELL["act_ckpt"] is True
    assert seen_on == {"cache": True, "recompute": True}, seen_on
    assert seen_off == {"cache": False, "recompute": False}, seen_off
    for mode in ("cache", "recompute"):  # only block inputs + states are kept for backward
        assert saved_on[mode] * 2 < saved_off[mode], (mode, saved_on, saved_off)
    assert rec_on["pass"] and rec_off["pass"], (rec_on, rec_off)
    assert rec_on["act_ckpt"] is True and rec_off["act_ckpt"] is False
    assert not act_ckpt.is_enabled(bb_on.model)  # the Trainer applies its own flags afterwards
    assert bb_on.fork_mode == "cache" and not any("lora_" in n for n, _ in bb_on.model.named_parameters())

# ----------------------------------------------------------------------------- parent arm seed for a resubmit (L-34)
def _fake_arm(d, cell, wkv=None):
    """A parent arm dir shaped like train.m2_gates.write_arm_cache output, for this cell."""
    from train.m2_gates import ARM_SCHEMA, expected_parent_identity, unit_spec

    d.mkdir(parents=True, exist_ok=True)
    man = {"schema": ARM_SCHEMA, "stages": {}}
    for st in ("g12", "index"):
        text = json.dumps({"stage": st, "decision": {}}, sort_keys=True)
        (d / f"units-{st}.json").write_text(text, encoding="utf-8", newline="\n")
        man["stages"][st] = {"file": f"units-{st}.json", "sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "arm": {"parent": expected_parent_identity("rwkv7", wkv=wkv or cell.get("rwkv7_wkv"))},
                             "units_identity": {"stage": st, "cohort_sha256": cell["cohort_sha256"],
                                                "unit_spec": unit_spec(cell)}}
    (d / "arm.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    (d / "rate-parent-g12.json").write_text("{}", encoding="utf-8")
    return hashlib.sha256((d / "arm.json").read_bytes()).hexdigest()


def test_arm_seed_installs_only_a_pinned_matching_arm(tmp_path):
    """job-uhxb5: a resubmit starts with an empty checkpoint slot; a pinned export of the parent arm is installed into
    it (arm.json + units, NOT the earlier box's rate reports) and the runner's sha-verified cache then skips scoring."""
    pytest.importorskip("torch")
    from jobs.m2_first_cell.arm_seed import SeedError, check, seed

    src = tmp_path / "seed"
    pin = _fake_arm(src, CELL)
    dst = tmp_path / "ck" / "parent_arm"
    rec = seed(src, dst, pin, CELL)
    assert rec["seeded"] and sorted(p.name for p in dst.iterdir()) == ["arm.json", "seed-provenance.json",
                                                                   "units-g12.json", "units-index.json"]
    assert hashlib.sha256((dst / "arm.json").read_bytes()).hexdigest() == pin
    assert seed(src, dst, pin, CELL)["seeded"] is False  # a restart's own cache is left as is
    assert seed(tmp_path / "absent", tmp_path / "ck2", pin, CELL)["seeded"] is False  # nothing shipped
    assert not (tmp_path / "ck2" / "arm.json").exists()
    for bad_pin in (None, "0" * 64):  # a shipped seed without (or against) the pin is refused
        with pytest.raises(SeedError):
            seed(src, tmp_path / "ck3", bad_pin, CELL)
    assert not (tmp_path / "ck3" / "arm.json").exists()
    (src / "units-index.json").write_text("{}", encoding="utf-8")  # drifted units file
    with pytest.raises(SeedError, match="units-index.json sha256"):
        check(src, pin, CELL)
    other = tmp_path / "other"  # an arm scored with another WKV implementation
    with pytest.raises(SeedError, match="parent identity"):
        check(other, _fake_arm(other, CELL, wkv="chunked"), CELL)
    spec = tmp_path / "spec"
    with pytest.raises(SeedError, match="unit spec"):
        check(spec, _fake_arm(spec, {**CELL, "eval_max_t": 4096}), CELL)


def test_arm_seed_pin_is_the_uhxb5_export():
    """The pinned seed is job-uhxb5's exported parent arm (art-sq6mp, out/parent_arm/arm.json), whose two stages were
    scored at agent-main 9c76c93 under this exact cell config."""
    pin = PARAMS["parent_arm_seed"]
    assert pin["arm_sha256"] == "f8a114f3050a82cafd9e45364aa7bee2ae5ef98b2d0516281b610c3f33482145"
    assert pin["units_sha256"] == {"g12": "c16c24db65124b78af782305a55d2092b8b4163459a9501714b314e4897ef395",
                                   "index": "f450d000912f5e479eee9943a57ebdb145660396d1fda73fd21c3d7140e57001"}
    assert "--src \"$DATA/parent_arm_seed\"" in JOB and "jobs.m2_first_cell.arm_seed seed" in JOB
    assert JOB.index("arm_seed seed") < JOB.index("for STAGE in g12 index")


def test_arm_seed_is_atomic_a_failed_check_never_becomes_the_cache(tmp_path, monkeypatch):
    """PR #147 review B: the seed is copied to a private temp dir, the COPY is checked, and only then renamed into
    place, so a copy that fails its check can never be reused as $CK/parent_arm by a restart."""
    pytest.importorskip("torch")
    from jobs.m2_first_cell import arm_seed as AS

    src = tmp_path / "seed"
    pin = _fake_arm(src, CELL)
    dst = tmp_path / "ck" / "parent_arm"
    real = AS.check

    def fail_on_copy(d, *a, **k):
        if Path(d) != src:
            raise AS.SeedError("copy corrupted in transit")
        return real(d, *a, **k)

    monkeypatch.setattr(AS, "check", fail_on_copy)
    with pytest.raises(AS.SeedError, match="in transit"):
        AS.seed(src, dst, pin, CELL)
    assert not (dst / "arm.json").exists() and not list((tmp_path / "ck").glob(".*seed-tmp"))
    monkeypatch.setattr(AS, "check", real)
    dst.mkdir(parents=True)  # the runner's empty dir is fine ...
    assert AS.seed(src, dst, pin, CELL)["seeded"] and (dst / "arm.json").exists()
    part = tmp_path / "ck2" / "parent_arm"  # ... a partial cache without arm.json is refused, never overwritten
    part.mkdir(parents=True)
    (part / "units-g12.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AS.SeedError, match="partial cache"):
        AS.seed(src, part, pin, CELL)


def test_seed_probe_comparison_rule():
    from jobs.m2_parent_arm.score_parent import PROBE_MAX_ABS_DCONF, compare_probe

    cached = {f"r{i}/q": {"conf": 0.6, "correct": 1.0, "cluster": f"r{i}"} for i in range(20)}
    same = {k: dict(v, conf=v["conf"] + 0.004) for k, v in cached.items()}  # cross-box bf16 noise
    assert compare_probe(cached, same)["ok"]
    drift = dict(same, **{"r0/q": dict(cached["r0/q"], conf=0.6 + 2 * PROBE_MAX_ABS_DCONF)})
    assert not compare_probe(cached, drift)["ok"]
    flips = {k: dict(v, correct=0.0 if i < 3 else 1.0) for i, (k, v) in enumerate(same.items())}  # 3/20 > 10 %
    assert not compare_probe(cached, flips)["ok"]
    assert not compare_probe(cached, {k: v for k, v in same.items() if k != "r5/q"})["ok"]  # an item not scored
    assert not compare_probe({}, {})["ok"]


def test_seed_restart_after_install_before_result_keeps_atomic_provenance(tmp_path, monkeypatch):
    from jobs.m2_first_cell import arm_seed as AS

    src, dst = tmp_path / "seed", tmp_path / "ck" / "parent_arm"
    pin = _fake_arm(src, CELL)
    rename = Path.rename

    def interrupt_after_install(path, target):
        result = rename(path, target)
        if Path(target) == dst:
            raise KeyboardInterrupt("process died after install, before the result/old shell marker")
        return result

    monkeypatch.setattr(Path, "rename", interrupt_after_install)
    with pytest.raises(KeyboardInterrupt):
        AS.seed(src, dst, pin, CELL)
    assert (dst / AS.PROVENANCE).exists() and not (dst.parent / "arm-seed.json").exists()
    monkeypatch.setattr(Path, "rename", rename)
    # The installed provenance remains sufficient even when the original shipped source is unavailable.
    resumed = AS.seed(tmp_path / "absent", dst, pin, CELL)
    assert not resumed["seeded"] and resumed["verify_seed"] and resumed["arm_sha256"] == pin
    with pytest.raises(AS.SeedError, match="provenance pin"):
        AS.seed(tmp_path / "absent", dst, "0" * 64, CELL)


def test_seed_restart_recognizes_legacy_install_without_any_marker(tmp_path):
    from jobs.m2_first_cell import arm_seed as AS

    src, dst = tmp_path / "seed", tmp_path / "ck" / "parent_arm"
    pin = _fake_arm(src, CELL)
    shutil.copytree(src, dst)
    assert not (dst / AS.PROVENANCE).exists()
    assert AS.seed(src, dst, pin, CELL)["verify_seed"]
    assert AS.seed(tmp_path / "absent", dst, pin, CELL)["verify_seed"]  # recognized by the pinned manifest
    (dst / "units-index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AS.SeedError, match="units-index.json sha256"):
        AS.seed(src, dst, pin, CELL)


def test_shipped_seed_does_not_reclassify_an_ordinary_restart_cache(tmp_path):
    from jobs.m2_first_cell import arm_seed as AS

    src, dst = tmp_path / "seed", tmp_path / "ck" / "parent_arm"
    pin = _fake_arm(src, CELL)
    shutil.copytree(src, dst)
    manifest = json.loads((dst / "arm.json").read_text(encoding="utf-8"))
    manifest["stages"]["g12"]["arm"]["revision"] = "scored by this job"
    (dst / "arm.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    before = (dst / "arm.json").read_bytes()
    assert hashlib.sha256(before).hexdigest() != pin
    resumed = AS.seed(src, dst, pin, CELL)
    assert not resumed["seeded"] and not resumed["verify_seed"]
    assert (dst / "arm.json").read_bytes() == before


def test_cache_verifier_refuses_full_item_guard_and_revision_mismatches(tmp_path):
    from types import SimpleNamespace

    from jobs.m2_parent_arm.score_parent import verify_cache
    from train import m2_gates as G

    cohort = SimpleNamespace(sha256=CELL["cohort_sha256"], steer_requests=lambda role: [])
    item = SimpleNamespace(id="record/question")
    items = {"dev": [item]}
    spec, guard = G.unit_spec(CELL), {"manifests": {"path": "hash"}, "n_ids": 1}
    parent = G.expected_parent_identity("rwkv7", wkv=CELL["rwkv7_wkv"])
    for st in ("g12", "index"):
        identity = G.units_identity(st, cohort, spec, {r: items[r] for r in G.STAGE_ROLES[st]}, [], [])
        G.write_arm_cache(tmp_path, st, {}, identity, {"parent": parent, "revision": "pinned", "guard": guard})
    pin = hashlib.sha256((tmp_path / "arm.json").read_bytes()).hexdigest()
    args = dict(arm_dir=tmp_path, arm_sha256=pin, cfg=CELL, parent="rwkv7", cohort=cohort, spec=spec,
                stages=["g12", "index"], items_by_role=items, records_by_id={}, index_rows=[], guard_identity=guard)
    assert verify_cache(**args)[0] == 0
    rc, record = verify_cache(**{**args, "items_by_role": {"dev": [SimpleNamespace(id="other/question")]}})
    assert rc == 10 and "units differ" in record["error"]
    rc, record = verify_cache(**{**args, "guard_identity": {"manifests": {"path": "other"}, "n_ids": 1}})
    assert rc == 10 and "another guard" in record["error"]
    rc, record = verify_cache(**args, probe_n=1,
                              decider=lambda: (None, {"parent": parent, "revision": "changed"}))
    assert rc == 10 and "parent decider" in record["error"]


def test_reprice_uses_the_seed_probe_only_without_rate_reports(tmp_path):
    from jobs.m2_first_cell.reprice import main

    probe = tmp_path / "seed-verify.json"
    probe.write_text(json.dumps({"ok": True, "probe": {"factor_estimate": 0.7, "records": 24}}), encoding="utf-8")
    params = str(ROOT / "jobs/m2_first_cell/job_params.json")
    out = tmp_path / "r.json"
    assert main(["--arm-dir", str(tmp_path), "--params", params, "--tok-s", "3757", "--call-s", "0.117",
                 "--out", str(out), "--probe", str(probe)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["factor"] == 0.7 and rec["source"].startswith("seed probe") and "child_projection" in rec
    probe.write_text(json.dumps({"ok": True, "probe": {"factor_estimate": 9.0}}), encoding="utf-8")  # a slow box
    assert main(["--arm-dir", str(tmp_path), "--params", params, "--tok-s", "3757", "--call-s", "0.117",
                 "--probe", str(probe)]) == 15
    assert main(["--arm-dir", str(tmp_path), "--params", params, "--tok-s", "3757", "--call-s", "0.117"]) == 0


def test_runner_verifies_a_seed_before_training_and_sets_the_tvuat_allocator():
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in JOB
    assert JOB.index("export PYTORCH_CUDA_ALLOC_CONF") < JOB.index("python -m train.run_cell")
    v = JOB.index("score_parent --verify-cache")
    assert JOB.index("arm_seed seed") < v < JOB.index("-m jobs.m2_first_cell.reprice") < JOB.index('say "phase cell"')
    assert '--probe-n "$PROBE_N"' in JOB and '"${PROBE[@]}"' in JOB and PARAMS["parent_arm_seed"]["probe_records"] >= 8
    assert 'cp "$ARM"/*.json "$WORK/out/parent_arm/"' in JOB[v:JOB.index('say "phase cell"')]  # comparator units kept


def test_train_max_k_lifts_training_cap_only():
    """D8: ``train_max_k`` widens the TRAINING K filter (and the m2_gates lineage rebuild, which calls cell_items with
    the default cap); ``max_k`` stays the eval / unit-spec cap."""
    pytest.importorskip("torch")
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE, toy_records
    from train.guard import SplitGuard
    from train.m2_gates import unit_spec
    from train.run_cell import cell_items, train_max_k

    recs = {r["id"]: r for r in toy_records(24, seed=3)}
    g = SplitGuard.synthetic_only("t")
    base = {"layout": "ordered", "template": TOY_TEMPLATE, "max_k": 1, "t_ladder": [2048]}
    lifted = {**base, "train_max_k": 255}
    assert train_max_k(base) == 1 and train_max_k(lifted) == 255
    assert cell_items(recs, sorted(recs), base, ByteEncoder(), g, 2048, workers=1) == []
    tr = cell_items(recs, sorted(recs), lifted, ByteEncoder(), g, 2048, workers=1)
    assert len(tr) == 48
    assert cell_items(recs, sorted(recs), lifted, ByteEncoder(), g, 2048, workers=1, max_k=lifted["max_k"]) == []
    assert unit_spec(lifted)["max_k"] == 1
    with pytest.raises(ValueError, match="train_max_k"):
        train_max_k({**base, "max_k": 16, "train_max_k": 8})


def test_d10_is_d3_v2f_fast_on_the_b1_100k_draw():
    """d10_v2f_b100k = d3_v2f_fast with only the training set (B1 100k-v2 draw, empty heldout exclusions) and the D8
    training-K lift changed; _t380 adds only the 380-min clock and max_steps 4,500 (operational). Eval (max_k 16,
    eval_max_t 2,048), T ladder (<= 2,048), flags and parent-arm handling are d3_v2f_fast's. The readout format
    (template / layout / r1_format_commit) is the cell JSON's own swap point: a v3w D10 changes only those keys, in
    both cell JSONs together (data, flags, caps and job params unchanged)."""
    d = ROOT / "jobs/m2_first_cell"
    load = lambda n: json.loads((d / n).read_text(encoding="utf-8"))
    d3 = load("m2_first_cell_rwkv7_d3_v2f_fast.json")
    notes = lambda c: {k for k in c if k.endswith("_note") or k.startswith("_")}
    changed = {"n_train_records": 100000, "train_max_k": 255, "row_options": 160}
    FORMAT = {"template", "layout", "r1_format_commit"}
    fmt = {m: {k: load(f"m2_first_cell_rwkv7_{m}.json").get(k) for k in FORMAT}
           for m in ("d10_v2f_b100k", "d10_v2f_b100k_t380")}
    assert fmt["d10_v2f_b100k"] == fmt["d10_v2f_b100k_t380"] and fmt["d10_v2f_b100k"]["layout"] == "g1v2"
    for mode, extra in (("d10_v2f_b100k", {}), ("d10_v2f_b100k_t380", {"max_minutes": 380, "max_steps": 4500})):
        c = load(f"m2_first_cell_rwkv7_{mode}.json")
        want = {**changed, **extra, "variant": mode}
        assert {k: c[k] for k in want} == want
        assert c["train_set"]["pointer"] == "data/manifests/b1-draw100k-v2.pointer.json"
        assert c["train_set"]["heldout_excluded"]["ids"] == []
        skip = set(want) | {"train_set"} | FORMAT | notes(c) | notes(d3)
        assert {k: v for k, v in c.items() if k not in skip} == {k: v for k, v in d3.items() if k not in skip}
        assert c["max_k"] == 16 and c["eval_max_t"] == 2048 and max(c["t_ladder"]) == 2048
        p = load(f"job_params_{mode}.json")
        assert p["mode"] == mode and p["train_dir"] == "m2-b100k"
        assert p["cell_config"] == f"jobs/m2_first_cell/m2_first_cell_rwkv7_{mode}.json"
        assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}")
        assert p["parent_arm_seed"]["arm_sha256"] is None
        assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
        worst = (p["build_timeout_minutes"] + 2 * p["max_duration_minutes"]) * p["batch_usd_per_min"]
        assert worst <= p["hard_cap_usd"]
        pins = (d / "data_d10_b100k.sha256").read_text(encoding="utf-8")
        for e in p["context"]["data"]:
            if e["dest"].startswith("m2-b100k/"):
                assert e["pin"] == "jobs/m2_first_cell/data_d10_b100k.sha256" and e["dest"] in pins


def test_train_wkv_split_keeps_the_d1_parent_and_unit_identity(tmp_path):
    """job-b4fy8: a cell with rwkv7_wkv "chunked_lean" (scoring) + train_rwkv7_wkv "fla" (training steps only) has the
    SAME parent / unit identity and TrainConfig hash as D1 base, so the pinned D1 parent arm seed is accepted; only the
    training steps switch the backbone to fla (and back), under the fla strict refusal."""
    pytest.importorskip("torch")
    from heads.backbone import tiny_hf_backbone
    from jobs.m2_first_cell.arm_seed import check
    from train import run_cell as R
    from train.m2_gates import expected_parent_identity, unit_spec
    from train.trainer import with_execution

    assert CELL["rwkv7_wkv"] == "chunked_lean" and "train_rwkv7_wkv" not in CELL
    split = {**CELL, "train_rwkv7_wkv": "fla"}
    assert expected_parent_identity("rwkv7", wkv=split["rwkv7_wkv"]) == expected_parent_identity("rwkv7", wkv=CELL["rwkv7_wkv"])
    assert unit_spec(split) == unit_spec(CELL)
    h = lambda c: with_execution(R.train_config(c, 10), c).hash()  # noqa: E731
    assert h(split) == h(CELL)
    src = tmp_path / "seed"
    pin = _fake_arm(src, CELL)  # the D1 base arm (scored chunked_lean)
    check(src, pin, split)  # accepted for the split cell (job-b4fy8 refused the all-fla cell here)
    with pytest.raises(Exception, match="parent identity"):
        check(src, pin, {**CELL, "rwkv7_wkv": "fla"})  # the PR #225 behaviour for rwkv7_wkv "fla" is unchanged
    assert R.train_wkv(split) == "fla" and R.train_wkv(CELL) == "chunked_lean"
    assert R.fla_requested(split) and not R.fla_requested(CELL)
    assert R.fla_record(split)["requested"] == "fla" and R.fla_record(split)["scoring_wkv"] == "chunked_lean"
    bb = tiny_hf_backbone("rwkv7", fork_mode="cache")
    from heads.rwkv7_wkv import select_wkv

    select_wkv(bb.model, split["rwkv7_wkv"])  # load_parent_backbone: the SCORING wkv
    with R.training_wkv(bb, split):
        assert select_wkv(bb.model, None) == "fla"
    assert select_wkv(bb.model, None) == "chunked_lean"
    with R.training_wkv(bb, CELL):  # default (no train_rwkv7_wkv): a no-op
        assert select_wkv(bb.model, None) == "chunked_lean"
    sh = JOB
    assert 'c.get("train_rwkv7_wkv")' in sh  # fla is installed / verified if EITHER key is fla
