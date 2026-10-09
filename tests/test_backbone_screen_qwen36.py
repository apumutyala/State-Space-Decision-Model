"""jobs/backbone_screen MODE=qwen36_a3b (stock Qwen3.6-35B-A3B, m0 readout WITHOUT the prefix fork, primary stratified
half of proxy600 first): a tiny random Qwen3_5MoeForCausalLM (transformers' own qwen3_5_moe code: GDN + gated attention
+ MoE, torch GDN path on CPU) with the REAL Qwen3.6 vocabulary size and tokenizer.

job-gen87 failed its canary (fork vs recompute 0.26 nat in bf16 on the real model). In fp32 the HF cache fork equals
full recompute here (test_m0_fork_equals_recompute, tol 1e-3; measured 4e-6), so the fork has no cache-copy bug; the
scout scores without it. Checks: the no-fork canary (batched == batch-1), the engine builds a no-fork backend, the
half subset (identity, stratification, order), the per-row cost weights, the rate guard's cost-weighted projection,
the parts check, the summarizer's primary mode, params / pins / hold, and the runner."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/backbone_screen/screen_job.sh"
PARAMS = ROOT / "jobs/backbone_screen/job_params_qwen36_a3b.json"
PIN = ROOT / "jobs/backbone_screen/snapshot_qwen36_a3b.sha256"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS  # noqa: E402

TOK = DEFAULT_TOKENIZERS / BACKBONES["qwen3.6-35b-a3b"].tokenizer_dir
needs_tok = pytest.mark.skipif(not (TOK / "tokenizer.json").is_file(), reason=f"no local tokenizer at {TOK}")
needs_moe = pytest.mark.skipif(not hasattr(transformers, "Qwen3_5MoeForCausalLM"), reason="transformers without qwen3_5_moe")
V3W_SHA = "56612ff01973d13b862fb9a2b90580d7ff425fadcd9eed18f3490381c96e11cf"


def params():
    return json.loads(PARAMS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(vocab_size=248320, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                               linear_key_head_dim=8, linear_value_head_dim=8, num_experts=8, num_experts_per_tok=2,
                               moe_intermediate_size=16, shared_expert_intermediate_size=16,
                               layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                               max_position_embeddings=32768)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def plugin(model, budget=131072):
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.state_fork import HFCacheStateOps
    from decide.v3w_generic import HFEncoder

    enc = HFEncoder(TOK)
    return ModelPlugin("qwen3.6-moe", "tiny", None, lambda text, trusted=True: enc(text, trusted),
                       lambda ids, n: model(input_ids=ids, logits_to_keep=n, use_cache=False).logits,
                       special_ids=enc.control_ids, device="cpu", dtype="float32", max_context=262144,
                       state_ops=HFCacheStateOps(model), fork_state_token_budget=budget)


@pytest.fixture(scope="module")
def v3w():
    from decide.v3w_generic import make_v3w

    return make_v3w("qwen3.6-35b-a3b")


@needs_tok
def test_v3w_identity(v3w):
    assert v3w.identity()["sha256"] == V3W_SHA
    assert v3w.fmt.no_think == "enable_thinking=False" and v3w.fmt.tail.endswith("<think>\n\n</think>\n\nANSWER:")
    assert len(v3w.codes) == 255


@needs_tok
def test_m0_template_is_the_models_own_no_think_turn():
    """jrt-chatml-v1 (the m0 readout) == Qwen3.6's own chat template with enable_thinking=False, around the body."""
    from decide.prompt import TEMPLATES
    from decide.v3w_generic import format_from_chat_template

    fmt = format_from_chat_template("qwen3.6-moe", TOK)
    t = TEMPLATES["jrt-chatml-v1"]
    assert fmt.head == t.head and fmt.tail == t.tail


@needs_tok
@needs_moe
def test_m0_fork_equals_recompute(tiny):
    """fp32: the HF cache fork (KV + GDN conv/recurrent states, MoE) equals full recompute -> no cache-copy bug."""
    from jobs.common.readout_engine import QWEN36_FORK_MAX_ROWS, QWEN36_M0, TEMPLATES, fork_selfcheck

    rep = fork_selfcheck(plugin(tiny), TEMPLATES[QWEN36_M0], QWEN36_FORK_MAX_ROWS, tol=1e-3)
    assert rep["ok"], {k: rep[k] for k in ("max_abs_delta", "same_argmax")}


@needs_tok
@needs_moe
@pytest.mark.parametrize("budget", [131072, 64])
def test_v3w_fork_equals_recompute_under_state_budget(tiny, v3w, budget):
    """The v3w backend packs at most state_budget // prefix rows per call; scores do not depend on the packing."""
    from decide.backends.v3w_generic_backend import V3WGenericBackend
    from decide.prompt import CANARY_REQUEST
    from jobs.common.readout_engine import v3w_selfcheck

    b = V3WGenericBackend(plugin(tiny, budget), v3w)
    assert b.state_budget == budget
    L, packs = b._calls(b._questions(CANARY_REQUEST))
    assert all(len(p) * L <= budget or len(p) == 1 for p in packs)
    rep = v3w_selfcheck(b, tol=1e-3)
    assert rep["ok"], {k: rep[k] for k in ("max_abs_delta", "same_argmax")}


def test_state_budget_split_is_pure():
    from decide.backends.v3w_generic_backend import V3WGenericBackend

    b = V3WGenericBackend.__new__(V3WGenericBackend)
    b.pack_rows, b.pack_tokens, b.state_budget = 16, 4096, 100
    qs = [{"qid": f"q{i}", "prompt": list(range(40)) + [100 + i, 7], "coded": True} for i in range(9)]
    L, packs = b._calls(qs)
    assert L == 40 and [len(p) for p in packs] == [2, 2, 2, 2, 1]
    b.state_budget = None
    assert [len(p) for p in b._calls(qs)[1]] == [9]


def test_engine_m0_plugin_cache(monkeypatch):
    from decide.backends import hf_readout as H
    from jobs.common import readout_engine as E

    calls = []
    monkeypatch.setattr(H, "load_qwen35", lambda d, **k: calls.append((d, k)) or object())
    monkeypatch.setattr(E, "_HF_PLUGINS", {})
    a = E.build_plugin(E.QWEN36_M0, {"model_dir": "/snap"})
    b = E.build_plugin(E.QWEN36_M0, {"model_dir": "/snap"})
    assert a is b and len(calls) == 1 and calls[0][1]["family"] == "qwen3.6-moe"
    assert E.TEMPLATES[E.QWEN36_M0] == "jrt-chatml-v1" == E.TEMPLATES["qwen3.5"]
    assert E.QWEN36_MAX_BATCH_ROWS == 4


@needs_tok
@needs_moe
def test_engine_m0_scores_without_fork_and_canary_passes(tiny, tmp_path, monkeypatch):
    pytest.importorskip("decision_index")  # the engine subclasses the kit's Engine
    from jobs.common import readout_engine as E

    monkeypatch.setattr(E, "build_plugin", lambda fam, opts: plugin(tiny))
    eng = E.ReadoutIndexEngine(family=E.QWEN36_M0, model_dir="/snap", max_batch_rows="4",
                               selfcheck_out=str(tmp_path / "sc.json"))
    assert eng.backend.fork is False
    rep = json.loads((tmp_path / "sc.json").read_text(encoding="utf-8"))
    assert rep["ok"] and rep["path"] == "no-fork" and rep["max_abs_delta"] < 1e-3


@needs_tok
@needs_moe
def test_nofork_equals_fork_in_fp32(tiny):
    """The fork's scores are right in fp32 (job-gen87's 0.26 nat is bf16 path-dependence, not a cache bug)."""
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.prompt import CANARY_REQUEST

    pl = plugin(tiny)
    nf = HFReadoutBackend(pl, "jrt-chatml-v1", max_batch_rows=4, fork=False).score(CANARY_REQUEST, lambda: None).logits
    fk = HFReadoutBackend(pl, "jrt-chatml-v1", max_batch_rows=4, fork=True, fork_max_rows=32).score(
        CANARY_REQUEST, lambda: None).logits
    assert max(abs(nf[q][o] - fk[q][o]) for q in nf for o in nf[q]) < 1e-3


def _manifest_proxy():
    from jobs.common import dev_subset as D

    return D.read_manifest(), D.load_subset("jobs/common/dev_proxy600.json")


def test_half_subset_identity_and_stratification():
    from collections import Counter

    from jobs.backbone_screen import half_subset as HS
    from jobs.common import dev_subset as D

    entries, proxy = _manifest_proxy()
    half = json.loads((ROOT / params()["screen"]["primary_subset"]).read_text(encoding="utf-8"))
    assert half["run_ids"] == HS.choose(entries, proxy["run_ids"])  # reproducible from the seed
    assert half["ids_sha256"] == D.ids_sha256(half["run_ids"]) and len(half["run_ids"]) == 341
    assert set(half["run_ids"]) < set(proxy["run_ids"])
    assert half["dev_manifest_sha256_uncompressed"] == proxy["dev_manifest_sha256_uncompressed"]
    cat = {e["run_id"]: e["catalog_id"] for e in entries}
    grp = {e["run_id"]: (e["catalog_id"], str(e["group_id"])) for e in entries}
    full, kept = Counter(cat[i] for i in proxy["run_ids"]), Counter(cat[i] for i in half["run_ids"])
    assert set(full) == set(kept) and min(kept.values()) >= 3
    kept_groups = {grp[i] for i in half["run_ids"]}
    assert not any(grp[i] in kept_groups for i in set(proxy["run_ids"]) - set(half["run_ids"]))  # groups whole


def test_order_parts_and_weights(tmp_path):
    import gzip

    from jobs.backbone_screen import half_subset as HS

    ids = [f"1:x:{i}" for i in range(10)]
    first = ids[3:7]
    rows = tmp_path / "rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as fh:
        for i in ids:
            fh.write(json.dumps({"_evaluation": {"run_id": i}, "state": "s"}) + "\n")
    (tmp_path / "first.json").write_text(json.dumps({"run_ids": first}), encoding="utf-8")
    cost = {"rows": {i: {"weight": n + 1} for n, i in enumerate(ids)}}
    (tmp_path / "cost.json").write_text(json.dumps(cost), encoding="utf-8")
    rep = HS.write_order(str(rows), str(tmp_path / "first.json"), str(tmp_path / "o.jsonl.gz"),
                         str(tmp_path / "cost.json"), str(tmp_path / "w.json"))
    with gzip.open(tmp_path / "o.jsonl.gz", "rt", encoding="utf-8") as fh:
        got = [json.loads(ln)["_evaluation"]["run_id"] for ln in fh]
    assert sorted(got[:4]) == sorted(first) and sorted(got) == sorted(ids) and rep["first"] == 4
    assert json.loads((tmp_path / "w.json").read_text()) == {i: float(cost["rows"][i]["weight"]) for i in first}
    # parts: complete lines only, duplicates dropped, a cut last line ignored
    res = tmp_path / "results.jsonl"
    body = "".join(json.dumps({"run_id": i, "status": "ok"}) + "\n" for i in got[:5])
    res.write_text(body + json.dumps({"run_id": got[0], "status": "dup"}) + "\n" + '{"run_id": "1:x:9", "st',
                   encoding="utf-8")
    (tmp_path / "full.json").write_text(json.dumps({"run_ids": ids}), encoding="utf-8")
    p = HS.parts(str(res), str(tmp_path / "scored.jsonl"), str(tmp_path / "first.json"), str(tmp_path / "full.json"))
    assert p["primary_done"] and not p["full_done"] and p["rows_written"] == 5
    assert (tmp_path / "scored.jsonl").read_text(encoding="utf-8") == body


def test_cost_file_covers_proxy600_and_matches_volume():
    s = params()["screen"]
    cost = json.loads((ROOT / s["cost_weights"]).read_text(encoding="utf-8"))
    _, proxy = _manifest_proxy()
    rows = cost["rows"]
    assert set(rows) == set(proxy["run_ids"]) and cost["max_batch_rows"] == 4 and cost["input_cap"] == 16384
    assert sum(r["over_cap"] for r in rows.values()) == 5
    assert sum(r["positions"] for r in rows.values() if not r["over_cap"]) == 84_239_337
    half = json.loads((ROOT / s["primary_subset"]).read_text(encoding="utf-8"))["run_ids"]
    assert sum(rows[i]["positions"] for i in half if not rows[i]["over_cap"]) == 40_029_828


def test_rate_guard_weighted_projection(tmp_path):
    from jobs.common import rate_guard as R

    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"run_id": "a", "x": 1}) + "\n" + json.dumps({"run_id": "b"}) + "\n" + '{"run_id": "c',
                   encoding="utf-8")
    done = R._done_ids(res)
    assert done == {"a", "b"}
    w = {"a": 1.0, "b": 1.0, "c": 98.0}
    assert R.weighted_rows(done, w, 3) == pytest.approx(0.06)
    # 2 of 3 rows done but 2% of the cost: the projection is 50x the elapsed time, not 1.5x
    assert R.project_weighted_rows(R.weighted_rows(done, w, 3), 600, 3) == pytest.approx(500.0)
    assert R.project_weighted_rows(R.weighted_rows(done, w, 3), 660, 3, startup_s=60) == pytest.approx(501.0)
    assert R.project(2, 600, 3) == pytest.approx(15.0)  # the unweighted guard is unchanged


def test_rate_guard_weighted_run_stops_on_cost(tmp_path):
    """25 cheap rows written fast, 1 row holding 99.9% of the cost not done: a row-count guard would pass, the
    cost-weighted one stops (15)."""
    import sys

    from jobs.common import rate_guard as R

    res = tmp_path / "r.jsonl"
    ids = [f"r{i}" for i in range(25)]
    child = ("import json,sys,time\nf=open(sys.argv[1],'w')\n"
             "[f.write(json.dumps({'run_id':i})+'\\n') for i in sys.argv[2:]]\nf.flush()\ntime.sleep(30)\n")
    w = {i: 1.0 for i in ids} | {"big": 25000.0}
    rc = R.run([sys.executable, "-c", child, str(res), *ids], res, 26, cap_min=1.0, check_after_min=0.02, poll_s=0.5,
               report=tmp_path / "rep.json", weights=w)
    assert rc == R.EXIT_TOO_SLOW
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["weighted"]["rows"] == 26 and rep["checks"][-1]["rows_equiv"] < 0.1


def test_summarize_primary_mode(tmp_path):
    from jobs.backbone_screen.summarize import build, valid_screen

    p = params()
    half = json.loads((ROOT / p["screen"]["primary_subset"]).read_text(encoding="utf-8"))["run_ids"]
    ix = tmp_path / "index"
    (ix / "stock").mkdir(parents=True)
    idx = {"point": 40.0, "lo": 35.0, "hi": 45.0, "se": 2.8}
    (ix / "score.json").write_text(json.dumps({"rows": 341, "subset_ids_sha256": "h", "slice_index": {"index": idx}}))
    (ix / "compare_stock_v3w.json").write_text(json.dumps({"diff": {"index": {"point": 12.0}}, "mde": {}}))
    (ix / "ece.json").write_text(json.dumps({"ece": 0.1}))
    (ix / "selfcheck.json").write_text(json.dumps({"ok": True, "max_abs_delta": 0.01, "same_argmax": True}))
    (ix / "parts.json").write_text(json.dumps({"primary_done": True, "full_done": False}))
    (ix / "stock" / "results.scored.jsonl").write_text(
        "".join(json.dumps({"run_id": i, "status": "ok"}) + "\n" for i in half)
        + json.dumps({"run_id": "zz", "status": "x"}) + "\n")
    d = build(p, tmp_path)
    assert d["rows"] == d["expected_rows"] == 341 and d["row_status"] == {"ok": 341} and "full" not in d
    assert valid_screen(d) and "arm_b" not in d


def test_check_arm_a_without_target():
    from jobs.backbone_screen.summarize import check_arm_a

    sc = {"slice_index": {"index": {"point": 40.0}}, "rows": 644}
    rep = check_arm_a({"screen": {}}, sc)
    assert rep["ok"] and rep["expected"] is None


def test_params_pins_and_hold():
    from jobs.common import costs

    p = params()
    s = p["screen"]
    assert p["chip"] == "h100" and p["chip_count"] == 1 and p["mode"] == "qwen36_a3b"
    assert "arm_b" not in s and not s.get("one_load_arms") and s["family"] == "qwen3.6-moe-m0"
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 95
    assert costs.quoted_max_cost_usd(p) <= 6.0 and costs.worst_case_usd(p) <= p["hard_cap_usd"]
    pin = dict(reversed(ln.split("  ", 1)) for ln in PIN.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert len(pin) == 34 and sum(k.startswith("model-") for k in pin) == 26
    import fnmatch

    assert all(any(fnmatch.fnmatch(f, pat) for pat in s["allow_patterns"]) for f in pin)
    for f in ("jobs/backbone_screen/snapshot_qwen36_a3b.sha256", s["primary_subset"], s["cost_weights"]):
        assert f in p["context"]["manifest_files"]


def test_runner_accepts_mode_and_primary():
    text = RUNNER.read_text(encoding="utf-8")
    assert "qwen36_a3b)" in text and "half_subset order" in text and "half_subset parts" in text
    assert '--weights "$WORK/weights.json"' in text and 'if [ -n "$PKG" ]' in text
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(RUNNER)], capture_output=True).returncode == 0
