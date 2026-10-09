"""Mode readout_ab (jobs/m2_first_cell/readout_ab.py): a saved layout-g1v2 checkpoint scored under several readouts in
one process. CPU checks with a tiny model: the eval readouts' rows are the stock R3 prompts, every option is forked
with the token the readout SHOWS for it (presented code under v3p / v2fp, the id's own first token under v3w) and maps
back to its option id, the trained-head backend scores them, the driver writes one results file per readout, pricing,
validator, params and pin filling."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from decide.prompt_g1v3 import (G1V3P_ID, G1V3W_ID, load_v3_codes, presentation_order, question_codes,
                                render_g1v3)
from heads.layout import G1V2FP_ID, ByteEncoder, _g1v2_code_table, build_row
from jobs.m2_first_cell import readout_ab as RA

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "jobs/m2_first_cell/job_params_readout_ab_d3.json"
D1DRY = ROOT / "jobs/m2_first_cell/job_params_readout_ab_d1dry.json"
D1 = ROOT / "jobs/m2_first_cell/job_params_readout_ab_d1.json"
V3 = [e["code"] for e in load_v3_codes()[0]]
CAT = {f"opt{i:02d}": f"choice number {i}, a long catalog description that runs well past the label cut" for i in range(12)}  # k = 12 >= 9: v3 code key
REQ = {"model": "ssdm-m2", "state": {"note": "The customer lost the card yesterday."},
       "questions": {"intent": {"type": "choice", "instructions": "Pick the intent.",
                                "criteria": {"lost_card": "lost or stolen card", "balance": "check balance",
                                             "zz_other": "something else"}},
                     "which": {"type": "choice", "instructions": "Which catalog entry?", "criteria": CAT},
                     "ok": {"type": "choice", "instructions": "Is it urgent?",
                            "criteria": {"yes": None, "no": None, "maybe": None}}}}


class WordEncoder(ByteEncoder):
    """ByteEncoder + three whole-word tokens (ids 1-3, inside the tiny model's vocab): " yes", " no", " maybe"."""
    WORDS = {" maybe": 3, " yes": 1, " no": 2}

    def identity(self) -> dict:
        return {"kind": "byte+4+words"}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        out, i = [], 0
        while i < len(text):
            w = next((w for w in self.WORDS if text.startswith(w, i)), None)
            if w:
                out.append(self.WORDS[w])
                i += len(w)
            else:
                out += [b + 4 for b in text[i].encode("utf-8")]
                i += 1
        return out

    def token_bytes(self, i: int) -> bytes:
        inv = {v: k for k, v in self.WORDS.items()}
        return inv[i].encode() if i in inv else bytes([i - 4])


def _shown(text: str, qid: str) -> dict[str, str]:
    """option id -> the code listed next to it in the RE-READ block of a rendered prompt."""
    block = text[text.index("QUESTION (re-read"):]
    block = block[:block.index("Reply with")]
    return {m.group(2): m.group(1) for m in re.finditer(r"\n([A-Z]{1,2})\. ([^:\n]+)", block)}


@pytest.mark.parametrize("tid,table,cat,perm", [(G1V3P_ID, V3, 64, True), (G1V2FP_ID, None, None, True)])
@pytest.mark.parametrize("qid", ["intent", "which"])
def test_permuted_readout_rows_are_the_stock_prompt_and_fork_the_shown_code(tid, table, cat, perm, qid):
    enc = ByteEncoder()
    table = table or _g1v2_code_table()
    row = build_row(REQ, qid, enc, template=tid, layout="g1v2")
    text = render_g1v3(REQ, qid, table, cat, perm, True)
    assert row.ids == [0] + enc(text)  # byte-equal to the stock R3 render (incl. the k >= 9 code key for v3p)
    assert ("CODE KEY" in text) == (tid == G1V3P_ID and qid == "which")
    shown = _shown(text, qid)
    q = REQ["questions"][qid]
    codes = question_codes(q, presentation_order(q, qid), table)
    assert row.option_ids == sorted(q["criteria"])  # logits stay keyed by canonical option id
    for k, oid in enumerate(row.option_ids):
        assert row.continuations[k] == enc(" " + shown[oid]) == enc(" " + codes[k])  # the code PRESENTED for oid
    assert row.meta["codes"] == [shown[o] for o in row.option_ids]
    assert presentation_order(q, qid) != list(range(len(q["criteria"])))  # the permutation is not the identity here
    dec = lambda a, b: bytes(i - 4 for i in row.ids[a:b]).decode()  # noqa: E731
    for k, oid in enumerate(row.option_ids):  # option spans follow the canonical option, not the listing slot
        assert dec(*row.option_spans[k]).startswith(oid)


def test_v3w_forks_each_word_id_with_its_own_first_token_and_letters_elsewhere():
    enc = WordEncoder()
    row = build_row(REQ, "ok", enc, template=G1V3W_ID, layout="g1v2")
    assert row.meta["word_ids"] and row.option_ids == ["maybe", "no", "yes"]
    assert row.continuations == [[3], [2], [1]]  # " maybe", " no", " yes": the id's first answer token
    text = render_g1v3(REQ, "ok", V3, 64, False, True, frozenset({"ok"}))
    assert row.ids == [0] + enc(text) and "\n- yes" in text and "option id." in text
    lc = build_row(REQ, "intent", enc, template=G1V3W_ID, layout="g1v2")  # opaque-free but first tokens are bytes " "
    assert not lc.meta["word_ids"] and lc.continuations == [enc(" " + c) for c in V3[:3]]


def test_label_ladder_steps_down_like_the_stock_backend(monkeypatch):
    import heads.layout as L

    enc = ByteEncoder()
    n64 = len(enc(render_g1v3(REQ, "which", V3, 64, True, True))) + 1
    n24 = len(enc(render_g1v3(REQ, "which", V3, 24, True, True))) + 1
    assert n24 < n64
    monkeypatch.setattr(L, "CATALOG_BUDGET", n64 - 1)
    row = build_row(REQ, "which", enc, template=G1V3P_ID, layout="g1v2")
    assert row.meta["catalog_chars"] == 24 and row.ids == [0] + enc(render_g1v3(REQ, "which", V3, 24, True, True))
    monkeypatch.setattr(L, "CATALOG_BUDGET", 10)
    row = build_row(REQ, "which", enc, template=G1V3P_ID, layout="g1v2")
    assert row.meta["catalog_chars"] is None and "CODE KEY" not in bytes(i - 4 for i in row.ids[1:]).decode()


@pytest.mark.parametrize("tid", ["jrt-rwkv-g1-v2f", G1V3P_ID, G1V3W_ID, G1V2FP_ID])
def test_trained_head_backend_scores_every_readout_keyed_by_option_id(tid):
    torch = pytest.importorskip("torch")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    enc = WordEncoder()
    be = DecisionModelBackend(model, enc, template=tid, layout="g1v2", checkpoint={"sha256": "x"})
    res = be.score(REQ, lambda: None)
    for qid, q in REQ["questions"].items():
        row = build_row(REQ, qid, enc, template=tid, layout="g1v2")
        lp = model([row]).log_probs[0].detach().float()
        assert set(res.logits[qid]) == set(q["criteria"])
        for k, oid in enumerate(row.option_ids):  # the backend's logit of oid IS the head on oid's own fork
            assert res.logits[qid][oid] == pytest.approx(float(lp[k]), abs=1e-4)
    out = Decider(be).decide(REQ)
    assert out["ssdm"]["status"] == "ok" and set(out["answers"]) == set(REQ["questions"])
    assert be.revision()["prompt_template"]["id"] == tid


def test_check_readouts_needs_the_training_template_first():
    d3 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d3.json").read_text(encoding="utf-8"))
    assert RA.check_readouts(d3, ["v2f", "v3w"])["matched_control"]
    with pytest.raises(ValueError, match="training template"):
        RA.check_readouts(d3, ["v3w", "v2f"])
    d1 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        RA.check_readouts(d1, ["v2f", "v3w"])
    assert not RA.check_readouts(d1, ["v2f", "v3w"], allow_unmatched_control=True)["matched_control"]
    assert RA.check_readouts(d1, ["v1", "v3w"])["matched_control"]  # D1's own v1 / ordered = its control
    with pytest.raises(ValueError):
        RA.check_readouts(d3, ["v1", "v3w"])
    with pytest.raises(ValueError):
        RA.parse_readouts("v2f")
    with pytest.raises(ValueError):
        RA.parse_readouts("v2f,v9")


@pytest.mark.parametrize("f,ro", [(D3, ["v2f", "v3w"]), (D1, ["v1", "v3w"])])
def test_estimate_prices_two_readouts_under_the_target_and_three_still_fit(f, ro):
    p = json.loads(f.read_text(encoding="utf-8"))
    est = RA.estimate(p)
    refit = sum(est["refit_minutes_per_readout"].values()) * p["batch_usd_per_min"]  # D1: v3w's own temperature
    assert est["ok"] and est["readouts"] == ro and est["usd_batch_expected"] <= 8.0 + refit
    assert est["minutes"]["readout_ab"] == pytest.approx(p["expected_minutes"]["readout_ab"], abs=0.5)
    three = copy.deepcopy(p)
    three["readouts"] = ro + ["v3p"]
    assert RA.estimate(three)["ok"]
    slow = copy.deepcopy(three)
    slow["measured_rates_reference"]["child_rows_per_min"] /= 1.2
    assert not RA.estimate(slow)["ok"]


def test_params_are_consistent_with_the_runner_and_the_shared_loader():
    from jobs.common.costs import schedule_minutes, worst_case_usd
    from jobs.m2_first_cell.checkpoint_loader import checkpoint_block

    for f, cell in ((D3, "m2_first_cell_rwkv7_d3_v2f_gold.json"), (D1, "m2_first_cell_rwkv7.json"),
                    (D1DRY, "m2_first_cell_rwkv7.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        assert p["eval_only"] == "readout_ab" and schedule_minutes(p) == p["max_duration_minutes"]
        assert worst_case_usd(p) <= p["hard_cap_usd"] and p["readouts"][1] == "v3w"
        assert p.get("allow_unmatched_control", False) == (f == D1DRY)
        assert "<" in p["submit"]["env"]["CKPT_URL"]  # never a signed URL in git
        cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
        assert cfg == json.loads((ROOT / "jobs/m2_first_cell" / cell).read_text(encoding="utf-8"))
    hij = json.loads((ROOT / "jobs/m2_first_cell/job_params_hijack_eval.json").read_text(encoding="utf-8"))
    ck1 = checkpoint_block(json.loads(D1.read_text(encoding="utf-8")))
    assert ck1["artifact"] == "art-bm3e6" and all(ck1[k] == hij["checkpoint"][k] for k in ck1 if not k.startswith("_"))
    with pytest.raises(Exception):  # D3 pins are placeholders until the artifact lands
        checkpoint_block(json.loads(D3.read_text(encoding="utf-8")))
    sh = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert "jobs.m2_first_cell.readout_ab:valid_result" in sh and "readout_ab estimate" in sh
    assert subprocess.run(["bash", "-n", str(ROOT / "jobs/m2_first_cell/first_cell_job.sh")]).returncode == 0


def _result(**over):
    ci = {"index": {"point": 30.0, "lo": 28.0, "hi": 32.0}}
    d = {"schema": RA.SCHEMA, "verdict": "measured", "rows": RA.N_SUBSET, "readouts": ["v2f", "v3w"],
         "arms": {"child": {r: {"status": {"ok": RA.N_SUBSET}, "index": ci} for r in ("v2f", "v3w")}},
         "paired": {"child:v3w-v2f": {}}, "checkpoint": {"ckpt_sha256": "a" * 64, "loaded": {"sha256": "a" * 64}}}
    d.update(over)
    return d


def test_valid_result_requires_every_readout_the_pair_and_the_pinned_checkpoint():
    assert RA.valid_result(_result())
    assert not RA.valid_result(_result(paired={}))
    assert not RA.valid_result(_result(checkpoint={"ckpt_sha256": "a" * 64, "loaded": {"sha256": "b" * 64}}))
    short = _result()
    short["arms"]["child"]["v3w"]["status"] = {"ok": RA.N_SUBSET - 1}
    assert not RA.valid_result(short)


def _subset_rows(n=3):
    return [{"_evaluation": {"run_id": f"SYNTH:{i}", "group_id": f"g{i}", "catalog_id": 32},
             "state": {"note": f"row {i}: the card was lost"}, "questions": copy.deepcopy(REQ["questions"])}
            for i in range(n)]


def test_score_writes_one_results_file_per_readout_from_one_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("decision_index")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_first_cell import fulldev_eval as F

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    loads = []

    def fake_load_child(a):
        loads.append(1)
        be = DecisionModelBackend(model, WordEncoder(), template="jrt-rwkv-g1-v2f", layout="g1v2",
                                  checkpoint={"path": "ckpt-step0000001.pt", "sha256": "a" * 64})
        return Decider(be), True, {"arm": "child", "checkpoint": be.checkpoint}

    monkeypatch.setattr(F, "load_child", fake_load_child)
    monkeypatch.setattr(RA, "N_SUBSET", 3)
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": json.loads(
        (ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d3.json").read_text(encoding="utf-8"))}))
    rows = tmp_path / "rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as f:
        for r in _subset_rows():
            f.write(json.dumps(r) + "\n")
    out, ck = tmp_path / "out", tmp_path / "ck"
    argv = ["score", "--arm", "child", "--readouts", "v2f,v3w,v3p", "--cell", str(cell), "--ckpt-dir", str(ck),
            "--model-dir", "x", "--records", "x", "--train-dir", "x", "--guard-readiness", "x", "--rows", str(rows),
            "--resume-dir", str(ck), "--out-dir", str(out), "--progress", str(tmp_path / "p.jsonl"), "--device", "cpu"]
    assert RA.main(argv) == 0 and loads == [1]  # ONE model load for every readout
    for r in ("v2f", "v3w", "v3p"):
        res = [json.loads(x) for x in (out / f"child.{r}.results.jsonl").read_text().splitlines()]
        assert [x["run_id"] for x in res] == ["SYNTH:0", "SYNTH:1", "SYNTH:2"] and all(x["status"] == "ok" for x in res)
        assert set(res[0]["response"]["answers"]) == set(REQ["questions"])
        assert (out / f"child.{r}.results_served.jsonl").is_file() and (out / f"plan-child-{r}.json").is_file()
    ident = json.loads((out / "identity-child.json").read_text())
    assert ident["readout_control"]["matched_control"]
    assert {r: v["template"] for r, v in ident["readouts"].items()} == {r: RA.READOUTS[r] for r in ("v2f", "v3w", "v3p")}
    assert len([x for x in (tmp_path / "p.jsonl").read_text().splitlines() if x]) == 9  # 3 rows x 3 readouts
    assert RA.main(argv) == 0  # a restart resumes: nothing re-scored
    assert len([x for x in (tmp_path / "p.jsonl").read_text().splitlines() if x]) == 0


def _tar(tmp_path, cell_bytes: bytes) -> Path:
    t = tmp_path / "art" / "artifact.tar"
    t.parent.mkdir()
    pt, latest = b"fake checkpoint bytes", json.dumps({"path": "ckpt-step0000042.pt"}).encode()
    with tarfile.open(t, "w") as tf:
        for name, b in (("./ckpt/latest.json", latest), ("./ckpt/ckpt-step0000042.pt", pt), ("./cell.json", cell_bytes)):
            info = tarfile.TarInfo(name)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    return t


def test_fill_ckpt_pins_fills_a_block_the_loader_accepts(tmp_path, monkeypatch):
    import importlib.util

    from jobs.m2_first_cell.checkpoint_loader import install

    spec = importlib.util.spec_from_file_location("fill_ckpt_pins", ROOT / "scripts/fill_ckpt_pins.py")
    FP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FP)
    monkeypatch.setenv("SSDM_DATA", str(tmp_path / "data"))
    cfg = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d3.json").read_text(encoding="utf-8"))
    cell_bytes = json.dumps({"config": cfg, "ladder": {"T_max": 2048}}).encode()
    tar = _tar(tmp_path, cell_bytes)
    params = tmp_path / "params.json"
    params.write_text(D3.read_text(encoding="utf-8"), encoding="utf-8")
    assert FP.main([str(tar.parent), "--params", str(params), "--artifact", "art-zz9yy", "--job", "job-93fpy",
                    "--write"]) == 0
    p = json.loads(params.read_text(encoding="utf-8"))
    ck = p["checkpoint"]
    assert ck["artifact"] == "art-zz9yy" and ck["member"] == "./ckpt/ckpt-step0000042.pt"
    assert ck["artifact_sha256"] == hashlib.sha256(tar.read_bytes()).hexdigest()
    assert ck["cell_json_sha256"] == hashlib.sha256(cell_bytes).hexdigest()
    assert p["submit"]["env"]["CKPT_URL"] == "<get_artifact art-zz9yy url: fill at submit>"
    shipped = tmp_path / "data/m2/checkpoints/job-93fpy/cell.json"
    assert shipped.read_bytes() == cell_bytes
    rec = install(ck, cell_src=shipped, cell_config=ROOT / p["cell_config"], ck_dir=tmp_path / "ck",
                  work=tmp_path / "w", url=tar.as_uri())
    assert rec["sha256"] == ck["ckpt_sha256"] and rec["file"] == "ckpt-step0000042.pt"
    bad = json.dumps({"config": {**cfg, "seed": 1}, "ladder": {"T_max": 2048}}).encode()
    tar2 = _tar(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, bad)
    with pytest.raises(SystemExit, match="config"):
        FP.main([str(tar2), "--params", str(params), "--artifact", "art-zz9yy"])


def test_throughput_reports_latency_and_tokens_per_second():
    res = [{"run_id": f"r{i}", "status": "ok", "total_wall_ms": 1000.0} for i in range(30)]
    sp = RA.throughput(res, {f"r{i}": 500 for i in range(30)})
    assert sp["rows"] == 20 and sp["tokens"] == 10_000 and sp["tokens_per_s"] == 500.0 and sp["rows_per_min"] == 60.0
    assert sp["latency"]["median_ms"] == 1000.0 and sp["latency"]["n"] == 20


def test_d1_control_is_the_ordered_v1_backend_and_v3w_rides_the_same_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("decision_index")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_first_cell import fulldev_eval as F

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    base = Decider(DecisionModelBackend(model, WordEncoder(), template="jrt-rwkv-g1-v1", layout="ordered",
                                        checkpoint={"path": "ckpt-step0001439.pt", "sha256": "a" * 64}))
    monkeypatch.setattr(F, "load_child", lambda a: (base, True, {"arm": "child", "checkpoint": base.backend.checkpoint}))
    monkeypatch.setattr(RA, "N_SUBSET", 3)
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": json.loads(
        (ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d1.json").read_text(encoding="utf-8"))}))
    rows = tmp_path / "rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as f:
        for r in _subset_rows():
            f.write(json.dumps(r) + "\n")
    out, ck = tmp_path / "out", tmp_path / "ck"
    argv = ["score", "--arm", "child", "--readouts", "v1,v3w", "--cell", str(cell), "--ckpt-dir", str(ck),
            "--model-dir", "x", "--records", "x", "--train-dir", "x", "--guard-readiness", "x", "--rows", str(rows),
            "--resume-dir", str(ck), "--out-dir", str(out), "--device", "cpu"]
    assert RA.main(argv) == 0
    ident = json.loads((out / "identity-child.json").read_text())
    assert ident["readout_control"]["matched_control"] and ident["readouts"]["v1"]["layout"] == "ordered"
    assert ident["readouts"]["v3w"]["layout"] == "g1v2"
    for r in ("v1", "v3w"):
        plan = json.loads((out / f"plan-child-{r}.json").read_text())
        assert all(isinstance(t, int) and t > 0 for t in plan["tokens"].values())
    t1 = json.loads((out / "plan-child-v1.json").read_text())["tokens"]
    tw = json.loads((out / "plan-child-v3w.json").read_text())["tokens"]
    assert t1 != tw  # two different prompts, not the control twice
