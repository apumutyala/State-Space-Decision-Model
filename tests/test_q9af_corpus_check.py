"""jobs/q9af_corpus_check (rebuild job-w3c2e's mix, scan every window vs DEV/TEST/held-out, frozen, attack): CPU only.

Synthetic sources with planted eval material and a synthetic eval index (tests/q9af_corpus_check_stub.py); the
shell-level dry run executes jobs/q9af_corpus_check/q9af_corpus_check_job.sh end to end.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pyarrow")

from jobs.q9af_corpus_check import corpus_check as C  # noqa: E402
from tests import q9af_corpus_check_stub as S  # noqa: E402

PARAMS = json.loads((ROOT / "jobs/q9af_corpus_check/job_params.json").read_text(encoding="utf-8"))
RUNNER = (ROOT / "jobs/q9af_corpus_check/q9af_corpus_check_job.sh").read_text(encoding="utf-8")
CFG = json.loads((ROOT / "configs/q9af_v2_h100.json").read_text(encoding="utf-8"))
#: every family name in the pinned eval index (index-pin 2c821e29..., meta rows), 2026-10-09
INDEX_FAMILIES = [
    "ACOS-category-sentiment", "ANLI", "API-Bank-tool-selection", "ARC-Challenge", "ARC-Easy", "Amazon-ESCI",
    "BANKING77", "BBH fixed-option tasks", "BFCL-tool-selection", "BPoMP-original-limerick", "BRIGHT-retrieval",
    "CFColor-preference", "CLINC150+OOS", "CLadder", "CRUXEval-output-choice", "ChessBench-legal-move", "ContractNLI",
    "FinEntity-entity-given", "ForecastBench-binary", "GPQA-Diamond", "GSM8K-10", "GSM8K-4", "HLE-text-MC",
    "Habermas-consensus", "HellaSwag", "HoVer claim verification", "Home-Appliance", "Humicroedit", "MMLU", "MMLU-Pro",
    "MuSR", "NLI4CT-2024", "New Yorker caption matching", "POP909-chord-pitch-class", "PhishNChips phishing decisions",
    "RAGTruth response-level hallucination", "RouterBench-0shot", "RouterBench-5shot", "SATA-Bench",
    "SGD-service-given-intent", "SimpleBench", "ToolRet-retrieval", "VAST", "When2Call MCQ", "WinoGrande",
    "iSarcasmEval-A-Ar", "iSarcasmEval-A-En", "iSarcasmEval-B-En", "iSarcasmEval-C-Ar", "iSarcasmEval-C-En"]


def _run(tmp_path, workers=0, expect=None):
    root = S.prepare(tmp_path / "root")
    roots = S.write_sources(root / "src")
    code, rep = C.run(S.tiny_config(), str(root / "data/slices/dev-rows.jsonl.gz"),
                      str(root / "data/doc-overlap" / S.KEY16), str(root / "mix"), str(root / "out"), workers, expect,
                      None, (root / "index.sha").read_text(), tok=C.FakeTok(), roots=roots)
    return code, rep, root


# ----------------------------------------------------------------------------- frozen catalogs, windows, seeds
def test_frozen_family_map_names_exactly_the_nine_frozen_catalogs():
    got = {f: C.frozen_of(f) for f in INDEX_FAMILIES if C.frozen_of(f)}
    assert got == {"BBH fixed-option tasks": (58,), "BPoMP-original-limerick": (20,), "CLadder": (44,),
                   "ChessBench-legal-move": (31,), "Home-Appliance": (9,), "MuSR": (32,),
                   "New Yorker caption matching": (64,), "POP909-chord-pitch-class": (22,), "SATA-Bench": (33,)}
    assert sorted({n for v in got.values() for n in v}) == [9, 20, 22, 31, 32, 33, 44, 58, 64]


def test_window_ranks_invert_the_training_permutation():
    from convert.mixdata import WindowStream

    toks = np.zeros(64 * 50 + 1, dtype=np.uint32)
    ws = WindowStream(toks, 64, 5, seed=CFG["seed"] + C.JOINT_SEED_OFFSET, device="cpu")
    order = ws.order.copy()
    rank = C.window_ranks(ws.n, CFG["seed"] + C.JOINT_SEED_OFFSET)
    assert (rank[order] == np.arange(ws.n)).all()
    assert 'WindowStream(train, L, B, seed=cfg["seed"] + 7' in (ROOT / "convert/joint.py").read_text(encoding="utf-8")
    assert C.JOINT_SEED_OFFSET == 7


# ----------------------------------------------------------------------------- the census on a synthetic mix
def test_census_classes_windows_and_tiling(tmp_path):
    code, rep, root = _run(tmp_path)
    assert code == 0 and rep["verdict"] == "CONTAMINATED" and C.valid_report(rep) is False  # synthetic index: unpinned
    t = rep["streams"]["train"]
    assert rep["tiling"]["ok"] and t["tokens"] == rep["mix"]["train_tokens"]
    pf = t["policy_fail"]
    assert pf["TEST"]["docs"] == 1 and pf["TEST"]["examples"][0]["id"] == "train/web/f0/rg0/r1"
    assert pf["TEST"]["examples"][0]["eval_run_id"] == "T:1"
    assert pf["HELDOUT"]["docs"] == 1 and pf["FROZEN"]["docs"] == 1 and pf["DEV"]["docs"] == 0
    assert pf["FROZEN"]["examples"][0]["frozen"] == [58] and pf["FROZEN"]["per_source"] == {"math": pf["FROZEN"]["per_source"]["math"]}
    assert t["contact"]["DEV"]["docs"] == 1 and t["contact"]["DEV"]["examples"][0]["id"] == "train/web/f0/rg0/r2"
    assert t["attack"]["docs"] == 1 and t["attack"]["per_source"].keys() == {"chat"}
    assert t["full_eval_v1_would_drop"]["docs"] == 4
    assert rep["verdict_counts"] == {"TEST": 1, "HELDOUT": 1, "FROZEN": 1, "ATTACK": 1}
    assert rep["dropped_seen_by_hook"] == {"train/web/dev_ngram": 1}  # dev13-v1 dropped the DEV-equal doc
    assert rep["mix"]["sources"]["web"]["docs_dropped_dev_ngram"] == 1
    # windows: the TEST doc's span maps to its windows, each with a permutation rank
    fl = [json.loads(x) for x in gzip.open(root / "out/flagged.jsonl.gz", "rt", encoding="utf-8")]
    d = next(x for x in fl if x["id"] == "train/web/f0/rg0/r1")
    L = S.tiny_config()["seq_len"]
    assert d["windows"] == list(range(d["token_offset"] // L, (d["token_offset"] + d["n_tokens"] - 1) // L + 1))
    rank = C.window_ranks(rep["windows"]["n_windows"], C.JOINT_SEED_OFFSET)
    assert d["window_ranks"] == [int(rank[w]) for w in d["windows"]]
    assert pf["TEST"]["windows"] == len(d["windows"])
    res = C.resolve(str(root / "out/flagged.jsonl.gz"), rep["windows"]["n_windows"])  # all windows trained
    assert res["policy:TEST"]["docs_in_trained_windows"] == 1 and res["attack"]["docs"] == 1
    assert C.resolve(str(root / "out/flagged.jsonl.gz"), 0)["policy:TEST"]["trained_windows"] == 0


def test_no_eval_text_in_any_output(tmp_path):
    _, _, root = _run(tmp_path)
    blob = "".join(p.read_bytes().decode("utf-8", "replace") if p.suffix != ".gz" else
                   gzip.decompress(p.read_bytes()).decode() for p in (root / "out").iterdir())
    for text in (S.TEST_DOC, S.FROZEN_DOC, S.DEV_DOC):
        grams = text.split()
        assert not any(" ".join(grams[i:i + 4]) in blob for i in range(len(grams) - 3))


def test_parallel_workers_give_the_same_census(tmp_path):
    _, a, _ = _run(tmp_path / "a", workers=0)
    _, b, _ = _run(tmp_path / "b", workers=2)
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("secs", "workers", "eval_index")},
                                 sort_keys=True, default=list)
    assert strip(a) == strip(b)


def test_reproduction_gate(tmp_path):
    _, rep, root = _run(tmp_path / "a")
    mix = json.loads((root / "out/mix.json").read_text())
    assert C.compare_mix(mix, mix)["match"]
    bad = json.loads(json.dumps(mix))
    bad["train_sha256"] = "0" * 64
    bad["sources"]["chat"]["docs_kept"] += 1
    res = C.compare_mix(mix, bad)
    assert not res["match"] and {d["key"] for d in res["diffs"]} == {"train_sha256", "sources.chat.docs_kept"}
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"status": "pinned", "mix": bad}))
    code, rep2, _ = _run(tmp_path / "b", expect=str(pin))
    assert code == C.EXIT_NOT_REPRODUCED and rep2["verdict"].startswith("NOT_REPRODUCED/")
    pin.write_text(json.dumps({"status": "pinned", "mix": mix}))
    code, rep3, _ = _run(tmp_path / "c", expect=str(pin))
    assert code == 0 and rep3["reproduction"]["match"] is True
    pend = json.loads((ROOT / "jobs/q9af_corpus_check/w3c2e_mix_pin.json").read_text())
    assert C.read_expect(str(ROOT / "jobs/q9af_corpus_check/w3c2e_mix_pin.json")) is None and pend["status"] == "pending"


def test_refuses_a_non_w3c2e_filter_and_a_stale_mix(tmp_path):
    root = S.prepare(tmp_path / "root")
    cfg = S.tiny_config()
    cfg["mix"]["decontam"] = "full-eval-v1"
    with pytest.raises(C.CheckError):
        C.run(cfg, "x", "x", str(root / "mix"), str(root / "out"), 0, None, tok=C.FakeTok(), roots={})
    (root / "mix").mkdir()
    (root / "mix/mix.json").write_text("{}")
    with pytest.raises(C.CheckError) as e:
        C.run(S.tiny_config(), "x", "x", str(root / "mix"), str(root / "out"), 0, None, tok=C.FakeTok(), roots={})
    assert e.value.code == C.EXIT_TILE


def test_index_must_match_the_committed_pin(tmp_path):
    root = S.prepare(tmp_path / "root")
    with pytest.raises(C.CheckError) as e:  # a synthetic index is not the pinned 2c821e29...
        C.load_index(str(root / "data/doc-overlap" / S.KEY16))
    assert e.value.code == C.EXIT_INDEX
    pin = json.loads((ROOT / "data/manifests/doc_overlap/index-pin.json").read_text())
    assert pin["index_sha256"].startswith("2c821e29") and pin["key"].startswith(S.KEY16)


# ----------------------------------------------------------------------------- params, pins, preflight, registration
def test_params_preflight_and_registration():
    from jobs.common import costs
    from jobs.q9af_corpus_check import preflight

    res = preflight.check(PARAMS, CFG)
    assert res["ok"], json.dumps(res["checks"], indent=1)
    assert PARAMS["chip"] == "cpu-8" and PARAMS["max_restarts"] == 0 and PARAMS["decision_rule"] is None
    assert costs.worst_case_usd(PARAMS) == 1.44
    assert "1.44" in (ROOT / "jobs/q9af_corpus_check/JOB_SPEC.md").read_text(encoding="utf-8")
    assert costs.PARAMS["q9af_corpus_check"] == "jobs/q9af_corpus_check/job_params.json"
    assert "jobs/q9af_corpus_check/" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert CFG["mix"]["decontam"] == C.W3C2E["decontam"] == "dev13-v1"


def test_data_pins_ship_the_dev_rows_and_every_index_array():
    from train import doc_overlap_guard as G

    pins = {ln.split()[1].removeprefix("/data/"): ln.split()[0]
            for ln in (ROOT / "jobs/q9af_corpus_check/data.sha256").read_text().splitlines() if ln.strip()}
    want = {"slices/dev-rows.jsonl.gz"} | {f"doc-overlap/{S.KEY16}/{k}.npy" for k in G.EvalIndex.ARRAYS} | {
        f"doc-overlap/{S.KEY16}/meta.json"}
    assert set(pins) == want == {d["dest"] for d in PARAMS["context"]["data"]}
    assert pins["slices/dev-rows.jsonl.gz"] == (ROOT / "jobs/clef_baseline/data.sha256").read_text().split()[0]
    assert all(d["pin"] == "jobs/q9af_corpus_check/data.sha256" for d in PARAMS["context"]["data"])


def test_fetch_stage_takes_the_tokenizer_not_the_weights():
    src = (ROOT / "scripts/sf_fetch.py").read_text(encoding="utf-8")
    block = src.split('"q9af_corpus_check": [', 1)[1].split("],\n}", 1)[0]
    assert "safetensors" not in block and 'Qwen/Qwen3.5-9B", "model", ["*.json", "*.txt", "*.jinja"]' in block
    v2 = src.split('"q9af_v2": [', 1)[1].split("    ],", 1)[0]
    for rev in re.findall(r'"([0-9a-f]{40})"', v2):  # the same pinned revisions as job-w3c2e's fetch
        if rev not in ("b08601e04326c79dfdd32d625aee71d232d685c3", "c30699e8356da336a370243923dbaf21066bb9fe"):
            assert rev in block


def test_runner_dry_seam_is_refused_in_the_image():
    assert 'if [ -d /opt/ssdm/m1 ] || [ -z "${Q9CC_DRYRUN_ROOT:-}" ]; then echo "Q9CC_DRYRUN refused here"; exit 5; fi' \
        in RUNNER
    assert not PARAMS["submit"]["env"]
    src = (ROOT / "jobs/q9af_corpus_check/corpus_check.py").read_text(encoding="utf-8")
    assert 'in_image = Path("/opt/ssdm/m1").is_dir()' in src and "if (a.fake_tokenizer or a.index_sha) and in_image:" in src
    assert 'return bool(os.environ.get("Q9CC_DRYRUN")) and not Path("/opt/ssdm/m1").is_dir()' in src


# ----------------------------------------------------------------------------- the SHELL-LEVEL dry run
BASH = shutil.which("bash")


def _to_posix(p):
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if re.match(r"^[A-Za-z]:/", s) else s


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_shell_dry_run(tmp_path):
    root = S.prepare(tmp_path / "root")
    shim = root / "shim"
    shim.mkdir()
    (shim / "curl").write_text('#!/usr/bin/env bash\nexit 0\n', newline="\n")
    (shim / "curl").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GMN_", "Q9CC_"))}
    path = [_to_posix(shim), _to_posix(os.path.dirname(sys.executable))]
    env.update({"Q9CC_DRYRUN": "1", "Q9CC_DRYRUN_ROOT": _to_posix(root), "CUDA_VISIBLE_DEVICES": "",
                "GMN_METADATA_URL": "http://metadata.invalid", "Q9CC_PATH_PREFIX": ":".join(path), "Q9CC_WORKERS": "2"})
    if os.name == "nt":
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p.endswith("site-packages")]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [BASH, "-c", 'export PATH="$Q9CC_PATH_PREFIX:$PATH"; exec bash jobs/q9af_corpus_check/q9af_corpus_check_job.sh']
    with open(root / "console.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
    log = (root / "console.log").read_text(encoding="utf-8", errors="replace")
    out = root / "output"
    assert rc == 0, log[-4000:]
    res = json.loads((out / "result.json").read_text())
    assert res["ok"] and res["verdict"] == "CONTAMINATED" and (out / "Q9CC_DONE").exists()
    rep = json.loads((out / "report.json").read_text())
    assert rep["verdict_counts"] == {"TEST": 1, "HELDOUT": 1, "FROZEN": 1, "ATTACK": 1} and rep["workers"] == 2
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert {m["phase"] for m in man} == {"setup", "fetch", "C", "final"} and all(m["ok"] for m in man)
