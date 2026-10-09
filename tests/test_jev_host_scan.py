"""jobs/jev_host_scan: candidate pins, vendored readouts, the no-fork v3w path, and the scan end to end on tiny
random-weight models of each loader family (CPU; tokenizers from SSDM_DATA/tokenizers, never weights)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/jev_host_scan/scan_job.sh"
PARAMS = ROOT / "jobs/jev_host_scan/job_params.json"
TOKROOT = Path(os.environ.get("SSDM_TOKENIZERS", Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
                              / "tokenizers"))
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))

from jobs.jev_host_scan import host_scan as HS  # noqa: E402


def _need_tok(*kinds):
    from jobs.jev_host_scan.tiny import TOK

    miss = [k for k in kinds if not (TOKROOT / TOK[k] / "tokenizer.json").exists()]
    if miss:
        pytest.skip(f"private tokenizers missing under {TOKROOT}: {miss}")


def _need_items():
    from jevkit.items import default_data_dir

    if not default_data_dir().exists():
        pytest.skip("pinned public JevBench items missing (python -m jevkit fetch)")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ static checks
def test_candidates_pins_prices_and_schedule():
    from decide.v3w_generic import BACKBONES
    from jevkit.meter import parse_price

    doc = HS.load_candidates()
    cands = doc["candidates"]
    ids = [c["id"] for c in cands]
    assert len(ids) == len(set(ids))
    assert ids[0] == "h2o-lightning-4b" and ids[-1] == "rwkv7-g1k-13.3b"  # anchor first, old reference last
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    need = 0
    for c in cands:
        lines = [ln.split(None, 1) for ln in (ROOT / c["snapshot_pin"]).read_text(encoding="utf-8").splitlines()]
        names = {n.strip() for _, n in lines}
        assert all(len(h) == 64 for h, _ in lines)
        assert "config.json" in names and any(n.endswith(".safetensors") for n in names), c["id"]
        assert c["revision"] and len(c["revision"]) == 40
        for arm in c["arms"]:
            assert arm["readout"] in ("h2o_shim", "m0", "v3w", "d1")
            if arm["readout"] == "v3w":
                assert c["backbone_id"] in BACKBONES
        tmp = ROOT / ".pytest_price.json"
        try:
            tmp.write_text(json.dumps(HS.price_doc(c)), encoding="utf-8")
            assert parse_price(str(tmp)).usd_per_m_input > 0
        finally:
            tmp.unlink(missing_ok=True)
        need += c["cap_min"] + c["download_cap_min"] + p["score_cap_minutes"] * len(c["arms"])
    assert need <= p["phase_caps_minutes"]["scan"], (need, p["phase_caps_minutes"])
    assert p["max_duration_minutes"] * 0.063 <= 40.0  # the platform hold stays <= $40


def test_params_lint_clean():
    from jobs.common import lint

    rel = "jobs/jev_host_scan/job_params.json"
    bad, waived = lint.failures(rel, lint.lint_params(rel, ROOT))
    assert not bad and not waived, bad


def test_vendored_code_pins():
    import third_party.d1_3b as D
    import third_party.h2o_lightning_shim as H
    import third_party.rwkv7_hf as V

    assert all(D.vendored_ok().values()) and all(H.vendored_ok().values()) and all(V.vendored_ok().values())
    assert V.CONFIG_SHA256_BY_REPO[V.REPO] == V.CONFIG_SHA256


def test_runner_shell_syntax():
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(RUNNER)], capture_output=True).returncode == 0
    src = RUNNER.read_text(encoding="utf-8")
    assert "python -m jobs.common.rate_guard run" in src and "\r" not in src


# ------------------------------------------------------------------------------------------------ the H2O shim readout
class _FakePlugin:
    """encode = the H2O tokenizer (vLLM /tokenize, add_special_tokens false); forward = deterministic logits."""

    def __init__(self, tok_dir: Path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        self.device = "cpu"
        self.notes = {"fp32_head": True}
        self.V = len(self.tok)

    def encode(self, text, trusted=True):
        return self.tok(text, add_special_tokens=False).input_ids

    def logits(self, ids):
        import torch

        g = torch.Generator().manual_seed(int(hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:12], 16))
        return torch.randn(self.V, generator=g) * 3

    def forward(self, ids, n_last):
        return self.logits(ids[0].tolist())[None, None, :]


class _FakeVLLM:
    def __init__(self, plug, model):
        self.p, self.model, self.base = plug, model, "fake"

    def call(self, method, path, body=None):
        return 200, {"data": [{"id": self.model, "max_model_len": 40960}]}

    def tokenize(self, text):
        return self.p.encode(text)

    def label_logprobs(self, prompt, ids):
        import torch

        lp = torch.log_softmax(self.p.logits(self.p.encode(prompt)).float(), dim=-1)
        vals = lp.index_select(0, torch.tensor(ids)).tolist()
        return dict(zip(ids, vals)), {"prompt_tokens": len(self.p.encode(prompt))}


def test_shim_engine_equals_the_vendored_shim_on_every_public_item():
    """ShimEngine (one transformers forward) answers exactly what the vendored shim's own decide() answers over a
    vLLM returning the same log-probabilities: same prompt, label ids, temperature, floor and answer format."""
    _need_tok("h2o")
    _need_items()
    from jevkit.items import load_public
    from jevkit.systemone import item_request

    from jobs.jev_host_scan.tiny import TOK

    import third_party.h2o_lightning_shim as H

    plug = _FakePlugin(TOKROOT / TOK["h2o"])
    eng = HS.ShimEngine(plug)
    mod, contract = H.load_contract()
    shim = mod.Shim(contract, _FakeVLLM(plug, contract.model))
    for it in load_public():
        req = item_request(it)
        a, b = eng(req)["answers"], shim.decide(req)["answers"]
        assert a == b, it.id
    assert eng.c.temperature == 0.8 and eng.c.noul_floor == 0.801 and len(eng.label_ids) == len(eng.c.labels)


def test_shim_env_override_is_ignored(monkeypatch):
    import third_party.h2o_lightning_shim as H

    monkeypatch.setenv("SHIM_TEMPERATURE", "2.0")
    monkeypatch.setenv("SHIM_NOUL_FLOOR", "0")
    _, c = H.load_contract()
    assert c.temperature == 0.8 and c.noul_floor == 0.801


# ------------------------------------------------------------------------------------------------ tiny end to end
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from jobs.jev_host_scan.tiny import build

    root = tmp_path_factory.mktemp("tiny")
    cache = {}

    def get(kind):
        _need_tok(kind)
        if kind not in cache:
            cache[kind] = build(kind, root / kind, TOKROOT)
        return cache[kind]

    return get


def _run(c, model_dir, out, items, monkeypatch=None):
    kind, handle = HS.load_model(c, Path(model_dir), "cpu")
    res = {}
    for arm in c["arms"]:
        ad, ident = HS.build_arm(c, arm, kind, handle, Path(model_dir), TOKROOT)
        res[arm["arm"]] = (HS.run_arm(ad, items, Path(out) / arm["arm"], warmup=1), ident)
    return res


def _recs(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.mark.parametrize("cid", ["h2o-lightning-4b", "qwen3.5-4b"])
def test_tiny_qwen35_arms(tiny, tmp_path, cid):
    _need_items()
    from jevkit.items import load_public

    items = load_public()[:6] + load_public()[-3:]
    res = _run(HS.candidate(cid), tiny("h2o"), tmp_path, items)
    for arm, (st, ident) in res.items():
        recs = _recs(tmp_path / arm / "preds.jsonl")
        assert st["status"]["ok"] == len(items) == len(recs), (arm, st)
        assert all(r["usage"]["output_tokens"] == 1 and r["usage"]["input_tokens"] > 0 for r in recs)
        assert all(isinstance(r["answer"], dict) for r in recs)
    if cid == "h2o-lightning-4b":
        assert res["h2o_shim"][1]["fp32_head"] is True  # tiny config carries head_dtype float32, as H2O v1.1


def test_tiny_rwkv7_g1i_codeless(tiny, tmp_path, monkeypatch):
    _need_items()
    import third_party.rwkv7_hf as V
    from jevkit.items import load_public

    d = tiny("rwkv7_g1i")
    c = HS.candidate("rwkv7-g1i-2.9b")
    monkeypatch.setitem(V.CODELESS_CONFIG_SHA256, c["repo"], _sha(d / "config.json"))
    with pytest.raises(RuntimeError):  # a code-less pin never accepts a snapshot that ships its own code
        (d / "modeling_rwkv7.py").write_text("x")
        V.verify_snapshot_code(d, c["repo"])
    (d / "modeling_rwkv7.py").unlink()
    items = load_public()[:5]
    res = _run(c, d, tmp_path, items)
    assert res["v3w"][0]["status"]["ok"] == 5
    assert res["v3w"][1]["prompt_template"]["id"] == "jrt-rwkv-g1-v3w"


def test_v3w_nofork_equals_g1v3_words_fork_fp32(tiny, monkeypatch):
    """The no-fork v3w readout reads the SAME prompt ids and tokens as the production G1V3Backend(mode='words'):
    in fp32 the two scoring paths agree to rounding."""
    _need_items()
    import third_party.rwkv7_hf as V
    from decide.backends.hf_readout import load_rwkv7_hf
    from decide.backends.rwkv_g1_v3 import G1V3Backend
    from decide.backends.v3w_nofork import V3WNoForkBackend
    from decide.v3w_generic import make_v3w
    from jevkit.items import load_public
    from jevkit.systemone import item_request

    d = tiny("rwkv7")
    monkeypatch.setitem(V.CONFIG_SHA256_BY_REPO, V.REPO, _sha(d / "config.json"))
    plug = load_rwkv7_hf(d, device="cpu", dtype="float32")
    a = V3WNoForkBackend(plug, make_v3w("rwkv7-g1k-13.3b", tokenizers_root=TOKROOT))
    b = G1V3Backend(plug, mode="words")
    for it in load_public()[::40]:
        req = item_request(it)
        la, lb = a.score(req, lambda: None).logits["q"], b.score(req, lambda: None).logits["q"]
        assert la.keys() == lb.keys()
        assert max(abs(la[k] - lb[k]) for k in la) < 2e-3, it.id
        assert a.plan(req).max_call_input_tokens == len(make_v3w("rwkv7-g1k-13.3b", tokenizers_root=TOKROOT)
                                                        .questions(req)[0]["prompt"])


def test_tiny_d1_system_one(tiny, tmp_path, monkeypatch):
    _need_items()
    import third_party.d1_3b as D
    from jevkit.items import load_public

    d = tiny("d1")
    monkeypatch.setattr(D, "CONFIG_SHA256", _sha(d / "config.json"))
    items = load_public()[:4] + load_public()[-2:]
    res = _run(HS.candidate("d1-3b"), d, tmp_path, items)
    recs = _recs(tmp_path / "d1" / "preds.jsonl")
    assert res["d1"][0]["status"]["ok"] == 6 and all(r["answer"]["type"] == it.type for r, it in zip(recs, items))


@pytest.mark.parametrize("kind,cid", [("nemotron_h", "nemotron-3-nano-4b"), ("lfm2", "lfm2.5-2.6b")])
def test_tiny_stock_hf_v3w(tiny, tmp_path, kind, cid):
    _need_items()
    from jevkit.items import load_public

    res = _run(HS.candidate(cid), tiny(kind), tmp_path, load_public()[:4])
    assert res["v3w"][0]["status"]["ok"] == 4


def test_strict_load_refuses_missing_weights_but_not_unused_ones(tiny, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    from decide.backends.hf_readout import load_hf_causal

    src = tiny("lfm2")
    d = tmp_path / "extra"
    shutil.copytree(src, d)
    sd = load_file(d / "model.safetensors")
    sd["model.vision_tower.unused.weight"] = torch.zeros(2)  # a multimodal checkpoint's tower: unexpected, allowed
    save_file(sd, d / "model.safetensors", metadata={"format": "pt"})
    assert load_hf_causal(d, family="x", device="cpu").notes["unexpected_keys"]["n"] == 1
    d = tmp_path / "broken"
    shutil.copytree(src, d)
    sd = load_file(d / "model.safetensors")
    sd.pop(sorted(k for k in sd if "layers.0" in k)[0])
    save_file(sd, d / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="weights do not match"):
        load_hf_causal(d, family="x", device="cpu")


def test_score_summarize_and_validator(tiny, tmp_path):
    """The shell's scoring + summarize steps on a tiny run: jevkit report per arm, decision.json, the validator, the
    H2O identity block."""
    _need_items()
    from jevkit.items import load_public

    root = tmp_path / "out"
    items = load_public()
    c = HS.candidate("h2o-lightning-4b")
    d = root / "cand" / c["id"]
    _run(c, tiny("h2o"), d, items)
    HS.main(["price", "--cand", c["id"], "--out", str(d / "price.json")])
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "jevkit", "score", "--public", "--pred",
                        str(d / "h2o_shim" / "preds.jsonl"), "--price", str(d / "price.json"), "--out",
                        str(d / "h2o_shim" / "report.json")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    (d / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    dec = HS.summarize(root)
    assert set(dec["status"]) == {x["id"] for x in HS.load_candidates()["candidates"]}
    row = [x for x in dec["rows"] if x["candidate"] == c["id"]][0]
    assert row["complete"] and row["composite"]["A"] is not None and row["cost"]["usd_per_1000"] > 0
    assert row["speed"]["n"] == 72 and "h2o_identity_check" in row
    assert HS.valid_scan(dec)
    assert not HS.valid_scan({**dec, "rows": []})
