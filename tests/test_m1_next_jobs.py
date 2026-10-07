"""CPU tests for the m1 next-rung jobs (jobs/probe9b, jobs/rwkv7, jobs/DECISION_TREE.md): config validation, cost
arithmetic, decision rules and the qualification -> next-job mapping. No GPU, no weights.

    python -m pytest -q tests/test_m1_next_jobs.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from jobs.common import costs, dev_subset
from jobs.common import finalize as fin
from jobs.common.launch import LEGACY_SUBMIT, workspace_cap_increment
from jobs.common.readout_engine import map_response
from scripts import decide_9b, decide_probe9b, decide_rwkv7

ROOT = Path(__file__).resolve().parents[1]
FULL = json.loads((ROOT / "configs/probe9b_h100.json").read_text())
LEAN = json.loads((ROOT / "configs/probe9b_h100_lean.json").read_text())
PP = costs.load("probe9b")
RA, RB = costs.load("rwkv7_readout"), costs.load("rwkv7_lora")
CELL = json.loads((ROOT / "jobs/rwkv7/pilot_cell.json").read_text())


# ----------------------------------------------------------------------------- cost arithmetic and schedules
def test_worst_case_arithmetic_counts_build_and_a_retry():
    assert costs.attempts(PP) == 2 and costs.attempts(RB) == 2
    assert costs.worst_case_usd(PP) == round((30 + 2 * 270) * 0.0495, 2) == 28.21
    assert costs.worst_case_usd(RA) == round((30 + 2 * 150) * 0.0495, 2) == 16.34
    assert costs.worst_case_usd(RB) == round((30 + 2 * 180) * 0.0495, 2) == 19.3
    assert costs.worst_case_usd(costs.load("qual")) == 18.81  # unchanged qual job


@pytest.mark.parametrize("name", list(costs.PARAMS))
def test_every_job_fits_its_hard_cap_and_schedule(name):
    p = costs.load(name)
    assert costs.cap_ok(p) and costs.schedule_ok(p)
    if p.get("expected_minutes"):
        assert costs.expected_usd(p) < costs.worst_case_usd(p)


def test_probe_cap_is_the_planned_40_and_rate_matches_plan():
    assert PP["hard_cap_usd"] == 40.0  # PLAN.md: "~$40 one-layer probe"
    assert all(costs.load(n)["batch_usd_per_min"] == round(2.97 / 60, 4) for n in costs.PARAMS)


def test_branch_ledger_fits_the_envelope_and_lines():
    led = costs.branch_ledger()
    assert led["spent"] == 8.43 and led["envelope"] == 525.0
    assert led["joint_gate_fits"]
    for b in led["branches"].values():
        assert b["fits_envelope"] and b["system1_left_for_m2"] > 0


def test_pipeline_fits_its_phase_cap():
    s1, s2 = FULL["stage1"]["guard"]["max_minutes"], FULL["stage2"]["guard"]["max_minutes"]
    # stage caps + probes/importance/throughput overhead (~40 min measured-free estimate) <= deadline < P cap
    assert s1 + s2 + 40 <= PP["pipeline_deadline_minutes"] < PP["phase_caps_minutes"]["P"]
    assert LEAN["stage1"]["guard"]["max_minutes"] + LEAN["stage2"]["guard"]["max_minutes"] <= s1 + s2


def test_reprice_reports_tokens_and_stop_rule():
    q = {"stage2_steady_tok_s": 1500.0, "stage1_steady_tok_s": 5000.0, "stage2_mfu": 0.17, "peak_gb": 55.0}
    r = costs.reprice_probe(q, FULL, LEAN, PP)
    assert r["lean"]["stage2_tokens"] == 1500.0 * 60 * 45 and r["full"]["stage2_tokens"] == 1500.0 * 60 * 60
    assert r["full"]["passes_stop_rule"] is False  # 1500 < 2730: the full config would stop with exit 3
    assert r["lean"]["passes_stop_rule"] is True  # 1500 >= 1365 and 5000 >= 4738
    m = PP["measured"]  # the job-4j3vu receipt passes the full config's floors with a 50% margin
    r = costs.reprice_probe({"stage2_steady_tok_s": m["stage2_steady_tok_s"], "stage1_steady_tok_s":
                             m["stage1_steady_tok_s"]}, FULL, LEAN, PP)
    assert r["full"]["passes_stop_rule"] is True


# ----------------------------------------------------------------------------- probe config validation
def test_probe_config_is_one_layer_enforced_kl_order():
    for c in (FULL, LEAN):
        assert c["progressive"]["max_layers"] == 1 and c["gate"]["policy"] == "enforce" and c["order"] == "kl"
        assert c["init_cmp"]["steps"] == 0 and c["controls"]["swa_windows"] == [] and c["controls"]["zero"] is False
        for st in ("stage1", "stage2"):
            g = c[st]["guard"]
            assert g["max_peak_gb"] == 76 and g["max_minutes"] > 0 and g["min_tok_s"] > 0
        assert c["probes"]["mmlu_k"] > 0 and c["probes"]["mqar_pairs"]
        assert c["model_dir"].startswith("${SSDM_DATA_ROOT}/hf/models--Qwen--Qwen3.5-9B")
    assert min(FULL["probes"]["niah_lengths"]) == 1024 and max(FULL["probes"]["niah_lengths"]) == 16384
    assert FULL["stage1"]["guard"]["min_tok_s"] == 9476 and FULL["stage2"]["guard"]["min_tok_s"] == 2730  # 50% of job-4j3vu


def test_lean_changes_only_cost_knobs_never_the_rule():
    allowed = {"_comment", "batch", "stage1", "stage2", "probes", "throughput"}
    assert {k for k in FULL if FULL[k] != LEAN[k]} <= allowed
    assert LEAN["success_rule"] == FULL["success_rule"] and LEAN["gate"] == FULL["gate"]
    assert LEAN["stage2"]["guard"]["min_tok_s"] * 2 == FULL["stage2"]["guard"]["min_tok_s"]


# ----------------------------------------------------------------------------- probe decision rule
def _results(gate_pass=True, conv=None, stock=None, stopped=None):
    stock = stock or {"ppl": {"ctx1024": {"ppl": 10.0}, "ctx4096": {"ppl": 8.0}}, "mmlu": {"acc": 0.70},
                      "mqar": {"mean_acc": 0.90}, "niah": {"L1024": {"acc": 1.0}, "L16384": {"acc": 0.9}, "mean_acc": 0.95}}
    conv = conv or json.loads(json.dumps(stock))
    step = {"layer": 3, "gate_pass": gate_pass, "accepted": gate_pass, "step_delta_nll": 0.01 if gate_pass else 0.09,
            "gate_eval": {"delta_nll": 0.01, "hidden_cos": 0.98, "kl": 0.02}}
    imp = {"order": [3, 7, 11, 15, 19, 23, 27, 31], "per_layer": {"3": {"kl": 0.03}}}
    if gate_pass:
        step["probe_tag"] = "conv1_kept7"
    R = {"progressive": {"steps": [step], "converted": [3] if gate_pass else []}, "probe_stock": stock,
         "importance": imp}
    if gate_pass:
        R["probe_conv1_kept7"] = conv
    if stopped:
        R["stopped"] = stopped
    return R


CMP_OK = {"diff": {"index": {"point": -0.4, "lo": -1.6, "hi": 0.8}}}


def test_probe_proceeds_when_every_criterion_passes():
    d = decide_probe9b.decide(_results(), CMP_OK, FULL)
    assert d["verdict"] == "proceed-all-8" and decide_probe9b.valid_decision(d)


def test_probe_retrieval_fail_is_separate_from_quality():
    R = _results()
    R["probe_conv1_kept7"]["niah"] = {"L1024": {"acc": 1.0}, "L16384": {"acc": 0.5}, "mean_acc": 0.75}
    d = decide_probe9b.decide(R, CMP_OK, FULL)
    assert d["verdict"] == "retrieval-fail" and decide_probe9b.valid_decision(d)


@pytest.mark.parametrize("mutate", ["ppl", "mmlu", "index_point", "index_lo"])
def test_probe_quality_fail(mutate):
    R, cmp = _results(), json.loads(json.dumps(CMP_OK))
    if mutate == "ppl":
        R["probe_conv1_kept7"]["ppl"]["ctx4096"]["ppl"] = 8.0 * 1.03
    elif mutate == "mmlu":
        R["probe_conv1_kept7"]["mmlu"]["acc"] = 0.66
    elif mutate == "index_point":
        cmp["diff"]["index"]["point"] = -1.2
    else:
        cmp["diff"]["index"]["lo"] = -2.6
    d = decide_probe9b.decide(R, cmp, FULL)
    assert d["verdict"] == "quality-fail" and decide_probe9b.valid_decision(d)


def test_rejected_gate_is_a_valid_quality_fail_without_index():
    d = decide_probe9b.decide(_results(gate_pass=False), None, FULL)
    assert d["verdict"] == "quality-fail" and decide_probe9b.valid_decision(d)


def test_missing_measurements_never_pass():
    d = decide_probe9b.decide(_results(), None, FULL)  # Index arms missing
    assert d["verdict"].startswith("no-measurement") and not decide_probe9b.valid_decision(d)
    d = decide_probe9b.decide(_results(stopped={"exit_code": 3, "reason": "slow"}), CMP_OK, FULL)
    assert not decide_probe9b.valid_decision(d)
    assert not decide_probe9b.valid_decision({"verdict": "proceed-all-8", "gate": {"pass": True}, "numbers": {}})


def test_finalize_writes_marker_only_on_valid_decision(tmp_path):
    def run(rc, ef, dec):
        p = tmp_path / "decision.json"
        if dec is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(json.dumps(dec))
        (tmp_path / "PROBE_DONE").write_text("stale")
        code = fin.main([str(tmp_path / "r.json"), str(rc), "{}", str(ef), "1", str(p), str(tmp_path), "PROBE_DONE",
                         "scripts.decide_probe9b:valid_decision"])
        return code, (tmp_path / "PROBE_DONE").exists()

    good = decide_probe9b.decide(_results(), CMP_OK, FULL)
    assert run(0, 0, good) == (0, True)
    assert run(0, 1, good) == (6, False)
    assert run(0, 0, None) == (9, False)
    assert run(0, 0, {"verdict": "proceed-all-8"}) == (9, False)
    assert all(run(rc, 0, good) == (rc, False) for rc in (2, 3, 4, 5, 7, 8, 10, 12, 13))


# ----------------------------------------------------------------------------- the decision tree mapping
def test_every_qual_verdict_maps_to_next_jobs():
    assert set(costs.QUAL_TO_NEXT) == set(decide_9b.VERDICTS)
    assert costs.next_jobs_after_qual("open-$40-one-layer-probe-spend-issue") == ["probe9b:full", "rwkv7_readout"]
    assert costs.next_jobs_after_qual("re-price-lean-plan")[0] == "probe9b:lean"
    assert costs.next_jobs_after_qual("fold") == ["rwkv7_readout", "rwkv7_lora"]
    assert costs.next_jobs_after_qual(None, 3) == costs.next_jobs_after_qual(None, 4) == ["rwkv7_readout", "rwkv7_lora"]
    for rc in (2, 5, 6, 7, 8, 9, 1):
        assert costs.next_jobs_after_qual("fold", rc)[0].startswith("diagnose")
    assert costs.next_jobs_after_qual("no-measurement (stage 2 did not run): ...")[0].startswith("diagnose")


def test_every_probe_verdict_maps_to_next_jobs():
    assert set(costs.PROBE_TO_NEXT) == set(decide_probe9b.VERDICTS)
    assert costs.next_jobs_after_probe("proceed-all-8")[0].startswith("full8")
    assert "rwkv7_lora" in costs.next_jobs_after_probe("retrieval-fail")
    assert costs.next_jobs_after_probe("quality-fail") == ["rwkv7_lora"]
    assert costs.next_jobs_after_probe(None, 4) == ["rwkv7_lora"]
    assert costs.next_jobs_after_probe("proceed-all-8", 12)[0].startswith("diagnose")


def test_decision_tree_doc_quotes_the_computed_costs():
    doc = (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    for name in costs.PARAMS:
        assert f"{costs.worst_case_usd(costs.load(name)):.2f}" in doc, name


# ----------------------------------------------------------------------------- DEV subset
def test_dev_subset_is_reproducible_dev_only_and_covers_every_benchmark():
    sub = dev_subset.load_subset()
    entries = dev_subset.read_manifest()
    assert dev_subset.build_subset(entries, sub["n_target"], sub["min_per_bench"], sub["seed"]) == sub["run_ids"]
    dev_ids = {e["run_id"] for e in entries}
    assert set(sub["run_ids"]) <= dev_ids
    with gzip.open(ROOT / "harness/manifests/test-manifest.jsonl.gz", "rt", encoding="utf-8") as f:
        test_ids = {json.loads(x)["run_id"] for x in f if x.strip()}
    assert not set(sub["run_ids"]) & test_ids
    assert sub["summary"]["benchmarks"] == len({e["catalog_id"] for e in entries})
    groups = {}
    for e in entries:
        groups.setdefault((e["catalog_id"], e["group_id"]), set()).add(e["run_id"])
    s = set(sub["run_ids"])
    assert all(g <= s or not (g & s) for g in groups.values())  # whole groups only
    assert 1900 <= len(s) <= 2200


def test_largest_remainder_sums():
    assert sum(dev_subset.largest_remainder(17, {"a": 3, "b": 5, "c": 1}).values()) == 17


# ----------------------------------------------------------------------------- readout engine mapping
def test_engine_maps_exhausted_to_unsupported_and_errors_raise():
    class U(Exception):
        pass

    ok = {"model": "m", "answers": {"q": {}}, "usage": {"input_tokens": 9},
          "ssdm": {"status": "ok", "raw_probabilities": {}, "diagnostics": {"latency_ms": 3.0}}}
    resp, raw = map_response(ok, U)
    assert resp["answers"] == {"q": {}} and raw["sequence_tokens"] == 9
    with pytest.raises(U):
        map_response({**ok, "ssdm": {"status": "exhausted", "error": {"code": "c", "message": "m"}}}, U)
    with pytest.raises(RuntimeError):
        map_response({**ok, "ssdm": {"status": "error", "error": {"code": "c", "message": "m"}}}, U)


def test_converted_layer_spec_parsing():
    from convert.readout import parse_layers

    assert parse_layers("none") == parse_layers("") == parse_layers(None) == []
    assert parse_layers("31,3") == [3, 31] and parse_layers([7]) == [7]


# ----------------------------------------------------------------------------- RWKV7 identities and pilot cell
def test_vendored_rwkv7_code_matches_pins_and_snapshot_list():
    import third_party.rwkv7_hf as V

    assert all(V.vendored_ok().values())
    pins = dict(reversed(line.split("  ")) for line in (ROOT / "jobs/rwkv7/snapshot.sha256").read_text().splitlines())
    assert sum(k.endswith(".safetensors") for k in pins) == 28
    assert all(pins[f] == h for f, h in V.CODE_SHA256.items()) and pins["config.json"] == V.CONFIG_SHA256
    for p in (RA, RB):
        assert p["model"] == f"{V.REPO}@{V.REVISION}"
    assert '"transformers==5.17.0"' in (ROOT / "jobs/rwkv7/Dockerfile").read_text()


def test_verify_snapshot_code_refuses_a_changed_file(tmp_path):
    import third_party.rwkv7_hf as V

    for f in V.CODE_SHA256:
        shutil.copy(V.HERE / f, tmp_path / f)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError):
        V.verify_snapshot_code(tmp_path)


def test_rwkv7_lora_names_match_the_real_bundled_modules():
    torch = pytest.importorskip("torch")
    from heads.backbone import BACKBONES
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7Model
    from train.lora import apply_lora, remove_lora

    spec = BACKBONES["rwkv7-13.3b"]
    m = Rwkv7Model(Rwkv7Config(vocab_size=64, hidden_size=32, num_heads=4, head_dim=8, num_hidden_layers=3,
                               intermediate_size=64, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                               v_low_rank_dim=8))
    found = apply_lora(m, [(g, rx, c * 3 // spec.n_layers) for g, rx, c in spec.lora_groups], rank=2)
    assert len(found["att"]) == 12 and len(found["ffn"]) == 6
    assert remove_lora(m) == 18 and not any("lora" in n for n, _ in m.named_parameters())
    assert isinstance(torch.nn.Linear(1, 1), torch.nn.Module)


def test_pilot_cell_config_and_memory_estimate():
    torch = pytest.importorskip("torch")  # noqa: F841  (train.estimate imports heads -> torch)
    from heads.backbone import BACKBONES
    from train.estimate import train_point

    assert CELL["head"] == "fork" and CELL["loss_arm"] == "ce" and CELL["max_k"] <= 16
    assert CELL["pass_rule"]["max_peak_gb"] <= 76 and 0 < CELL["pass_rule"]["loss_drop_frac_min"] < 1
    assert CELL["max_minutes"] + 5 <= RB["phase_caps_minutes"]["train"]
    assert CELL["smoke"]["max_rel_feature_diff"] <= 0.05 and CELL["smoke"]["min_grad_cos"] >= 0.99
    pt = train_point(BACKBONES["rwkv7-13.3b"], "fork", max(CELL["t_ladder"]), CELL["max_k"], rank=CELL["lora_rank"],
                     fork_mode="cache", grad_ckpt=False)
    assert pt["mem_gb"]["total"] <= 64  # estimate leaves >= 12 GB under the 76 GB stop at the largest ladder rung


def test_run_cell_data_helpers_and_pass_rule():
    from train import run_cell as rc

    r = {"id": "m2:x:1", "state": "s", "questions": {"q": {"type": "noul"}}, "labels": {"q": "true"}, "source": "x"}
    assert rc.to_train_record(r) == {"id": "m2:x:1", "request": {"model": "m2", "state": "s",
                                                                  "questions": {"q": {"type": "noul"}}},
                                     "labels": {"q": "true"}}
    ids = [f"r{i}" for i in range(100)]
    tr, ev = rc.split_ids(ids, 60, 20, 0)
    assert len(tr) == 60 and len(ev) == 20 and not set(tr) & set(ev) and rc.split_ids(ids, 60, 20, 0) == (tr, ev)
    rule = CELL["pass_rule"]
    hist = [{"loss": 2.0}] * 50 + [{"loss": 1.5}] * 50
    cards = [{"raw": {"nll": 2.0}}, {"raw": {"nll": 1.6}}]
    assert rc.pass_rule(hist, cards, 60.0, rule)["pass"]
    assert not rc.pass_rule(hist, cards, 77.0, rule)["pass"]
    assert not rc.pass_rule([{"loss": 2.0}] * 100, cards, 60.0, rule)["pass"]
    assert not rc.pass_rule(hist[:-1] + [{"loss": float("nan")}], cards, 60.0, rule)["pass"]


def test_rwkv7_decisions():
    assert decide_rwkv7.decide_lora({"smoke": {"pass": True}, "train": {"steps": 10, "verdict": {
        "pass": False, "loss_first": 2.0, "peak_gb": 50.0}}})["verdict"] == "pilot-fail"
    d = decide_rwkv7.decide_lora({"smoke": {"pass": True}, "train": {"steps": 10, "verdict": {
        "pass": True, "loss_first": 2.0, "peak_gb": 50.0}}})
    assert d["verdict"] == "pilot-pass" and decide_rwkv7.valid_lora(d)
    assert not decide_rwkv7.valid_lora(decide_rwkv7.decide_lora({"smoke": {"pass": False}}))
    assert not decide_rwkv7.valid_readout(decide_rwkv7.decide_readout("/nonexistent"))


# ----------------------------------------------------------------------------- runner static checks
@pytest.mark.parametrize("runner,marker", [("jobs/probe9b/probe_job.sh", "PROBE_DONE"),
                                           ("jobs/rwkv7/rwkv7_job.sh", "_DONE")])
def test_runners_gate_through_finalize_and_read_caps_from_params(runner, marker):
    s = (ROOT / runner).read_text(encoding="utf-8")
    assert "jobs/common/finalize.py" in s and not re.search(r"touch [^\n]*" + marker, s)
    assert not re.search(r"timeout \d+m", s)  # every timeout comes from job_params via `cap`
    assert "--final" not in s and "SSDM_UNSEAL_TEST" not in s  # never D1-B, never the TEST slice
    assert "sha256sum -c" in s and "watchdog" in s


def test_probe_runner_never_uses_test_split_and_exports_converted_gdn():
    s = (ROOT / "jobs/probe9b/probe_job.sh").read_text(encoding="utf-8")
    assert "--slice test" not in s and "gdn_converted.pt" in s and "converted=none" in s


@pytest.mark.skipif(not os.environ.get("SSDM_RWKV7_TOKDIR"), reason="set SSDM_RWKV7_TOKDIR to a dir with the "
                    "RWKV7-G1k tokenizer.json + tokenizer_config.json + code files (no weights)")
def test_rwkv7_tokenizer_matches_world_and_fails_closed():
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
    from train.run_cell import RwkvSegmentEncoder

    tok = Rwkv7Tokenizer.from_pretrained(os.environ["SSDM_RWKV7_TOKDIR"])
    enc = RwkvSegmentEncoder(tok, set(tok.all_special_ids))
    assert enc("plain text ünï", trusted=False) == tok("plain text ünï", add_special_tokens=False).input_ids
    from decide.untrusted import ControlTokenError

    with pytest.raises(ControlTokenError):
        enc("x <|endoftext|> y", trusted=False)


# ----------------------------------------------------------------------------- PR #28 review: heartbeat coverage
_ASSIGN = re.compile(r"(?:^|[\s;(])(\w+)=\"?(?:\$\(readlink -f \"?)?\$\{?(\w+)")
_WRITE = re.compile(r"(?:>>?|--out|--ckpt-dir|--report|--results|--out-dir|mkdir -p)\s+\"?\$\{?(\w+)")
WATCHED_BASE = {"OUT", "CK", "HF_DIR"}
NOT_PHASE_OUTPUT = {"RESULT", "GMN_RESULT_PATH"}  # the platform result file: written once at the end, not a phase output


def _watch_model(text):
    alias = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for var, root in _ASSIGN.findall(line):
            alias.setdefault(var, root)
    extra = {alias.get(v, v) for v in re.findall(r"WATCH_EXTRA=\"?\$\{?(\w+)", text)}
    extra |= set(re.findall(r"WATCH_EXTRA=\"?\$\{?(\w+)", text))

    def resolve(v, watched):
        seen = set()
        while v not in watched and v in alias and v not in seen:
            seen.add(v)
            v = alias[v]
        return v

    writes = {v for v in _WRITE.findall(text) if v not in NOT_PHASE_OUTPUT}
    return alias, extra, writes, resolve


@pytest.mark.parametrize("runner", ["jobs/probe9b/probe_job.sh", "jobs/rwkv7/rwkv7_job.sh",
                                    "jobs/clef_baseline/run_job.sh"])
def test_every_phase_writes_under_a_watched_dir(runner):
    text = (ROOT / runner).read_text(encoding="utf-8")
    assert "watchdog.sh" in text
    alias, extra, writes, resolve = _watch_model(text)
    watched = WATCHED_BASE | extra
    bad = sorted(v for v in writes if resolve(v, watched) not in watched)
    assert writes and not bad, f"{runner}: writes under unwatched dirs {bad}"


def test_watch_test_is_not_vacuous_for_rwkv7():
    text = (ROOT / "jobs/rwkv7/rwkv7_job.sh").read_text(encoding="utf-8")
    alias, extra, writes, resolve = _watch_model(text.replace("WATCH_EXTRA=$WORK", "true"))
    assert extra == set()
    assert "WORK" in {resolve(v, WATCHED_BASE) for v in writes}  # the original bug would be caught


def test_qual_heartbeat_is_unconditional_so_it_cannot_false_kill():
    s = (ROOT / "jobs/qual/qual_job.sh").read_text(encoding="utf-8")
    assert re.search(r'while true; do date -u \+%FT%TZ > "\$OUT/heartbeat"; sleep 60; done', s)
    assert "watchdog.sh" not in s


WATCH_SCRIPT = r"""
set -u
OUT="$1/out"; CK="$1/ck"; HF_DIR="$1/hflink"; W="$1/work"; mkdir -p "$OUT" "$CK" "$W" "$1/hfreal"
ln -sfn "$1/hfreal" "$HF_DIR" 2>/dev/null || mkdir -p "$HF_DIR"
MAIN_PID=$$; HB_EVERY_S=1; STALL_S=3; WATCH_EXTRA=${3:-}
. "$2"
start_watchdog
sleep 4
rm -f "$OUT/heartbeat"
for i in 1 2 3 4; do echo "$i" >> "$W/train_log.jsonl"; sleep 1; done
if [ -f "$OUT/heartbeat" ]; then echo touched; else echo withheld; fi
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_watchdog_counts_progress_under_watch_extra(tmp_path):
    import subprocess

    p = tmp_path / "t.sh"
    p.write_text(WATCH_SCRIPT, encoding="utf-8", newline="\n")
    wd = (ROOT / "jobs/clef_baseline/watchdog.sh").as_posix()

    def run(extra):
        r = subprocess.run(["bash", str(p), tmp_path.as_posix(), wd, extra], capture_output=True, text=True,
                           timeout=60)
        return r.stdout.split()[-1]

    assert run((tmp_path / "work").as_posix()) == "touched"
    assert run("") == "withheld"  # without WATCH_EXTRA the same progress is invisible (the PR #28 finding)


# ----------------------------------------------------------------------------- measured stop conditions
def test_rate_guard_preflight_and_projection():
    from jobs.common import rate_guard as rg

    assert rg.preflight(1000.0, 600_000, 40)["ok"] is True  # 600k x 2 / 1000 / 60 = 20 min <= 38
    assert rg.preflight(300.0, 600_000, 40)["ok"] is False  # 66.7 min
    assert rg.project(100, 600.0, 2008) == pytest.approx(600 / 100 * 2008 / 60)
    assert rg.subset_proxy_tokens(str(dev_subset.SUBSET)) > 0


def _child(code_py):
    import sys as _s

    return [_s.executable, "-c", code_py]


def test_rate_guard_run_stops_slow_passes_fast_and_propagates(tmp_path):
    from jobs.common import rate_guard as rg

    res = tmp_path / "results.jsonl"
    head = "import time\nf = open(" + repr(str(res)) + ", 'a')\n"
    fast = rg.run(_child(head + "for i in range(30):\n    f.write('x' + chr(10)); f.flush()\n"), res, 30,
                  cap_min=1.0, check_after_min=0.0, poll_s=0.05)
    assert fast == 0
    res.unlink()
    slow = rg.run(_child(head + "for i in range(1000):\n    f.write('x' + chr(10)); f.flush(); time.sleep(0.05)\n"),
                  res, 1000, cap_min=0.2, check_after_min=0.02, poll_s=0.1, report=tmp_path / "rep.json")
    assert slow == rg.EXIT_TOO_SLOW
    assert "measured" in json.loads((tmp_path / "rep.json").read_text())["stopped"]
    assert rg.run(_child("import sys; sys.exit(3)"), tmp_path / "none.jsonl", 10, cap_min=1.0, poll_s=0.05) == 3
    assert rg.run(_child("import time; time.sleep(30)"), tmp_path / "none.jsonl", 10, cap_min=0.02,
                  check_after_min=0.0, poll_s=0.05) == 124


def test_training_rate_stop_condition():
    from train import run_cell as rc

    need = rc.steps_needed(CELL["pass_rule"])
    assert need == 100
    assert rc.too_slow(60.0, CELL["max_minutes"], need)["too_slow"] is False  # 110 steps affordable
    assert rc.too_slow(70.0, CELL["max_minutes"], need)["too_slow"] is True  # 94 < 100
    assert rc.EXIT["too_slow"] == 26
    s = (ROOT / "jobs/rwkv7/rwkv7_job.sh").read_text(encoding="utf-8")
    assert "21|22|23|24|25|26)" in s and "15) fail 15" in s


@pytest.mark.parametrize("runner", ["jobs/probe9b/probe_job.sh", "jobs/rwkv7/rwkv7_job.sh"])
def test_readout_phases_use_measured_rate_guards(runner):
    s = (ROOT / runner).read_text(encoding="utf-8")
    assert "rate_guard preflight --tok-s" in s and "rate_guard run --results" in s
    assert "timeout ${CAP_I" not in s  # the guard owns the cap for the Index arms


# ----------------------------------------------------------------------------- manifest hashes, chained caps
def test_dev_subset_records_which_manifest_bytes_are_hashed():
    import hashlib

    sub = dev_subset.load_subset()
    gz = (ROOT / "harness/manifests/dev-manifest.jsonl.gz").read_bytes()
    assert sub["dev_manifest_sha256_gz"] == hashlib.sha256(gz).hexdigest()
    assert sub["dev_manifest_sha256_uncompressed"] == hashlib.sha256(gzip.decompress(gz)).hexdigest()
    slices = json.loads((ROOT / "harness/manifests/slices.json").read_text())
    assert sub["dev_manifest_sha256_uncompressed"] == slices["dev"]["manifest_sha256"]
    assert "dev_manifest_sha256" not in sub  # no ambiguous key


def test_chained_and_concurrent_workspace_caps():
    assert costs.concurrent_cap_usd(["rwkv7_readout", "rwkv7_lora"]) == 36
    assert costs.concurrent_cap_usd(["probe9b", "rwkv7_readout"]) == 45
    assert costs.concurrent_cap_usd(["rwkv7_lora"]) == 20
    lora = json.loads((ROOT / "jobs/rwkv7/job_params_lora.json").read_text(encoding="utf-8"))
    assert lora["workspace_cap_jobs"] == ["rwkv7_readout", "rwkv7_lora"]
    increment, jobs = workspace_cap_increment(costs, lora, LEGACY_SUBMIT[("rwkv7", "lora")],
        costs.PARAMS["rwkv7_lora"], costs.worst_case_usd(lora))
    assert (increment, jobs) == (36, ["rwkv7_readout", "rwkv7_lora"])
    spec = (ROOT / "jobs/rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    assert "MTD + 36" in spec and "depends_on" in spec
    assert "MTD + 45" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("spec", ["jobs/probe9b/JOB_SPEC.md", "jobs/rwkv7/JOB_SPEC.md"])
def test_health_fields_documented(spec):
    s = (ROOT / spec).read_text(encoding="utf-8")
    for k in ("grace_period_s: 1800", "fail_threshold: 2", "interval_s: 600", "STALL_S", "WATCH_EXTRA"):
        assert k in s, k
