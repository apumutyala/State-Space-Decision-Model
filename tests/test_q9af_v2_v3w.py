"""Mode q9af_v2_v3w (Q9-AF v2 first cell) on CPU: the generic v3w spec pin and registry, the V3WGenericBackend
readout (shared-prefix fork == a full per-question forward), train rows == readout ids through heads.layout, the screen's
cross-backbone readout family, the mode's configs, the launch pin filler and the runner's q9af branch.
Real-tokenizer tests skip when the Qwen3.5-9B tokenizer files are not on this machine. The full 1,919-row
B1 100k-v2 check is scripts/check_q9af_v3w_equivalence.py (receipts/2026-10-09-q9af-v2-v3w/)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from decide.v3w_generic import (BACKBONES, DEFAULT_TOKENIZERS, PINNED_SPEC_SHA256, TEMPLATE_BACKBONE, bind_template,
                                is_generic_template, template_id)

ROOT = Path(__file__).resolve().parents[1]
TPL = "jrt-v3w-generic:qwen3.5"
QDIR = DEFAULT_TOKENIZERS / BACKBONES["qwen3.5-9b"].tokenizer_dir
needs_qwen = pytest.mark.skipif(not (QDIR / "tokenizer.json").is_file(), reason="Qwen3.5-9B tokenizer files")
CELL = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json").read_text(encoding="utf-8"))
PARAMS = json.loads((ROOT / "jobs/m2_first_cell/job_params_q9af_v2_v3w.json").read_text(encoding="utf-8"))
D16 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_d16_v3w_b100k_safe.json").read_text(encoding="utf-8"))
JOB = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")

REQ = {"state": {"passage": "The museum opened in 1902 and moved to the harbour in 1964.",
                 "notes": {"curator": "M. Ito", "rooms": 12}},
       "questions": {"year": {"type": "choice", "instructions": "When did the museum move?",
                              "criteria": {"A": "1902", "B": "1964", "C": "1971"}},
                     "mood": {"type": "choice", "instructions": {"task": "pick the tone", "candidate": "x"},
                              "criteria": {"neutral": "Neutral", "positive": "Positive", "negative": "Negative"}},
                     "cat": {"type": "choice", "instructions": "Which category?",
                             "criteria": {f"c{i:02d}": f"category number {i}" for i in range(40)}}}}


def test_registry_and_pin():
    assert template_id("qwen3.5-9b") == TPL and TEMPLATE_BACKBONE[TPL] == "qwen3.5-9b"
    assert is_generic_template(TPL) and not is_generic_template("jrt-rwkv-g1-v3w")
    assert len(PINNED_SPEC_SHA256[TPL]) == 64


@needs_qwen
def test_bind_checks_the_pinned_identity(tmp_path):
    spec = bind_template(TPL, QDIR)
    assert spec.identity()["sha256"] == PINNED_SPEC_SHA256[TPL] and len(spec.table) == 255
    # another copy of the SAME files (any dir name, e.g. an HF snapshot) gives the same identity
    d = tmp_path / "snapshot"
    d.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json"):
        (d / f).write_bytes((QDIR / f).read_bytes())
    assert bind_template(TPL, d).identity()["sha256"] == PINNED_SPEC_SHA256[TPL]
    # a changed chat template fails closed
    cfg = json.loads((QDIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg["chat_template"] = cfg["chat_template"].replace("<|im_start|>assistant", "<|im_start|>assistant ", 1)
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        bind_template(TPL, d)
    bind_template(TPL, QDIR)  # restore the process-wide binding for later tests


class _CausalOps:
    """A deterministic causal 'model' over the real vocabulary: logits(t) depend only on ids[:t+1]. ``extend`` returns
    a lazy [B, W, V] whose rows are computed only where the backend reads them (248k-vocab logits per position would
    not fit the shared laptop)."""

    def __init__(self, V):
        import torch

        self.V, self.torch = V, torch
        self.w = torch.randn(64, generator=torch.Generator().manual_seed(0))

    def last_logits(self, seq):
        t, acc = self.torch, 0.0
        for x in seq:
            acc = 0.9 * acc + float(self.w[x % 64]) + 0.001 * (x % 997)
        idx = t.arange(self.V, dtype=t.float64)
        return t.sin(idx * (0.001 + abs(acc) % 0.01) + acc).float()

    def prefill(self, ids):
        return None, ids[0].tolist()

    def extend(self, state, ids, n_last):
        ops = self

        class Lazy:
            device = "cpu"

            def __getitem__(self, key):
                rows, cols = key
                return ops.torch.stack([ops.last_logits(state + ids[r, : c + 1].tolist())
                                        for r, c in zip(rows.tolist(), cols.tolist())])

        return Lazy(), None

    def select(self, state, i):
        return state


@needs_qwen
def test_backend_fork_equals_full_forward_and_train_rows():
    import math
    from types import SimpleNamespace

    import torch

    from decide.backends.v3w_generic import V3WGenericBackend
    from heads.layout import build_row, resolve_template

    spec = bind_template(TPL, QDIR)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QDIR))
    V = len(tok)
    ops = _CausalOps(V)
    plug = SimpleNamespace(family="qwen3.5", model_id="stub", snapshot=None,
                           encode=lambda s, trusted=True: tok(s, add_special_tokens=False).input_ids, forward=None,
                           special_ids=frozenset(), files={}, max_context=262144, dtype="float32", device="cpu",
                           kernel_stack={}, state_ops=ops, fork_state_token_budget=None, notes={},
                           untrusted_encoder_id="stub")
    be = V3WGenericBackend(plug, spec)
    assert ";prompt=jrt-v3w-generic:qwen3.5." in be.revision()["revision_id"]
    res = be.score(REQ, lambda: None)
    qs = {q["qid"]: q for q in spec.questions(REQ)}
    assert qs["mood"]["words"] and all(q["coded"] for q in qs.values())
    assert any(q["codes"] for q in qs.values())  # at least one code-key question besides the word ids
    for qid, q in qs.items():
        full = torch.log_softmax(ops.last_logits(q["prompt"]).double(), -1)
        want = {o: float(full[t]) for o, t in zip(q["options"], q["tokens"])}
        assert all(math.isclose(res.logits[qid][o], want[o], abs_tol=1e-4) for o in want), qid
        row = build_row(REQ, qid, None, template=resolve_template(TPL), layout="g1v2")
        assert row.ids == q["prompt"] and row.continuations == [[t] for t in q["tokens"]]
        assert row.option_ids == q["options"] and row.answer_pos == len(q["prompt"]) - 1


@needs_qwen
def test_decide_backend_identity_is_the_spec():
    from heads.decide_backend import _template_identity
    from heads.layout import resolve_template

    spec = bind_template(TPL, QDIR)
    assert _template_identity(resolve_template(TPL)) == spec.identity()


def test_screen_family_and_regex():
    from jobs.m2_first_cell import screen as S

    assert S.readout_family(TPL) == S.readout_family("jrt-rwkv-g1-v3w") == "v3w"
    assert S.same_readout("jrt-rwkv-g1-v3w", TPL, "family") and not S.same_readout("jrt-rwkv-g1-v3w", TPL, "exact")
    assert not S.same_readout("jrt-rwkv-g1-v1", TPL, "family")
    rev = "qwen3.5:x+conv3-7@local;w=abc;tok=def;prompt=jrt-v3w-generic:qwen3.5.2e4907ee;cfg=1234abcd;adapter=0.2.0"
    assert S.arm_template({"revision": rev}) == TPL
    assert S.arm_template({"revision": "rwkv7:m@x;w=a;tok=b;prompt=jrt-rwkv-g1-v3w.0a1b2c3d;cfg=1;adapter=0"}) == "jrt-rwkv-g1-v3w"
    ro = S.screen_readout(PARAMS, CELL)
    assert ro["template"] == TPL and ro["layout"] == "g1v2" and ro["native"] and ro["reference_match"] == "family"
    blk = S.reference_block(PARAMS, TPL)
    assert blk["artifact"] == "art-zskmx"
    with pytest.raises(S.ScreenError):
        S.reference_block({**PARAMS, "screen_reference_match": "exact"}, TPL)
    assert S.parent_block(PARAMS, TPL, TPL) is None  # in-job parent under the same generic template
    with pytest.raises(S.ScreenError):
        S.parent_block(PARAMS, TPL, "jrt-rwkv-g1-v3w")  # the parent is exact, never family
    assert S.template_available(TPL)


def test_mode_configs_are_d16_on_the_q9af_parent():
    from jobs.common import costs

    assert CELL["parent"] == "q9af" and CELL["template"] == TPL and CELL["layout"] == "g1v2"
    assert CELL["row_batch"] == 1 and CELL["row_tokens"] == 0 and CELL["compile_blocks"] is False
    assert "rwkv7_wkv" not in CELL and CELL["act_ckpt"] is True
    for k in ("train_max_k", "row_options", "lr_schedule", "lr_min_frac", "dev_card_every",
              "dev_card_rows", "best_ckpt", "collapse_guard", "lr_head", "lr_lora", "lora_rank", "max_minutes",
              "max_steps", "min_steps", "t_ladder", "eval_max_t", "max_k", "cohort_sha256", "head",
              "loss_arm", "row_plan", "cont_plan", "seed", "batch_size", "grad_accum"):
        assert CELL[k] == D16[k], k
    assert CELL["train_set"]["attack_clean"]["n_excluded"] == 5201 and CELL["train_max_k"] == 255
    assert CELL["train_set"]["doc_clean"]["n_excluded"] == 1801 and CELL["n_train_records"] == 93013
    q = CELL["q9af"]
    assert q["base_repo"] == "Qwen/Qwen3.5-9B" and q["hybrid"] is False
    assert q["base_config_sha256"] == "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
    assert PARAMS["mode"] == "q9af_v2_v3w" and PARAMS["cell_config"].endswith("m2_first_cell_rwkv7_q9af_v2_v3w.json")
    assert PARAMS["stage"] == "screen" and PARAMS["screen_readout"] == "v3w" and PARAMS["screen_parent"] == "in_job"
    assert PARAMS["screen_reference"] == "d1_v3w" and PARAMS["screen_reference_match"] == "family"
    assert PARAMS["context"]["dockerfile"] == "jobs/probe9b/Dockerfile"
    assert sum(PARAMS["phase_caps_minutes"].values()) + PARAMS["reserve_minutes"] == PARAMS["max_duration_minutes"]
    assert costs.worst_case_usd(PARAMS) <= PARAMS["hard_cap_usd"]
    dests = {d["dest"]: d for d in PARAMS["context"]["data"]}
    assert dests["gdn/gdn_converted.pt"]["pin"] == dests["gdn/export.json"]["pin"] == "jobs/m2_first_cell/gdn_q9af_v2.sha256"
    snap = (ROOT / "jobs/m2_first_cell/q9af_snapshot_v3w.sha256").read_text(encoding="utf-8")
    assert "chat_template.jinja" in snap and "tokenizer.json" in snap and "config.json" in snap


def test_runner_q9af_branch():
    assert 'case "$PARENT" in rwkv7) ;; q9af) GDN_ARGS=(--gdn-ckpt "$DATA/gdn/gdn_converted.pt")' in JOB
    assert 'jobs/m2_first_cell/q9af_snapshot_v3w.sha256' in JOB and "jobs.q9af_cell.pin check" in JOB
    assert "bind_template(c[\"template\"], sys.argv[2])" in JOB
    for s in ('score_parent --parent "$PARENT" "${GDN_ARGS[@]}"', 'train.run_cell --config "$CELL" --model-dir "$SNAP" "${GDN_ARGS[@]}"',
              '--model-dir "$SNAP" "${GDN_ARGS[@]}" --records "$RECORDS" --train-dir "$TRAIN_DIR"'):
        assert s in JOB, s
    assert JOB.index("q9af_snapshot_v3w.sha256") < JOB.index("phase cell")


def _fake_export(tmp_path, layers, verdict="ADVANCE", attention_free=True):
    import torch

    from jobs.q9af_cell.hooks import sha256

    pt = tmp_path / "gdn_converted.pt"
    torch.save({i: {"w": torch.ones(2)} for i in layers}, pt)
    ex = tmp_path / "export.json"
    ex.write_text(json.dumps({"ok": True, "mode": "final", "attention_free": attention_free, "converted": layers,
                              "file": "gdn_converted.pt", "sha256": sha256(pt),
                              "base_config_sha256": CELL["q9af"]["base_config_sha256"],
                              "attention_layers_kept": []}), encoding="utf-8")
    from jobs.q9af_v2 import rule

    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps({"rule_version": rule.RULE_VERSION, "verdict": verdict, "action": "x", "problems": [],
                               "numbers": {"proxy_diff_point": -6.0, "proxy_diff_lo": -9.0, "heldout_kl": 0.2},
                               "report": {"binding": ["proxy_diff_point", "proxy_diff_lo"]}}), encoding="utf-8")
    return pt, ex, dec


def test_fill_conversion_pins(tmp_path):
    from jobs.q9af_v2.rule import ATTENTION_LAYERS
    from scripts import fill_conversion_pins as F

    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps(CELL), encoding="utf-8")
    pin = tmp_path / "gdn.sha256"
    pt, ex, dec = _fake_export(tmp_path, ATTENTION_LAYERS)
    args = ["--export-json", str(ex), "--gdn", str(pt), "--decision", str(dec), "--cell", str(cell), "--data-pin", str(pin)]
    assert F.main(args) == 0
    c = json.loads(cell.read_text(encoding="utf-8"))
    assert len(c["q9af"]["converted_sha256"]) == 64 and "ADVANCE" in c["q9af"]["converted_from"]
    assert pin.read_text(encoding="utf-8").splitlines()[0].endswith("  gdn/gdn_converted.pt")
    _, _, dec2 = _fake_export(tmp_path, ATTENTION_LAYERS, verdict="MARGINAL")
    assert F.main(args[:4] + ["--decision", str(dec2)] + args[6:]) == 10
    pt3, ex3, dec3 = _fake_export(tmp_path, ATTENTION_LAYERS[:-1], attention_free=False)
    assert F.main(["--export-json", str(ex3), "--gdn", str(pt3), "--decision", str(dec3), "--cell", str(cell),
                   "--data-pin", str(pin), "--dry"]) == 10
