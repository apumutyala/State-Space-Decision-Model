"""A-PREREG-JEV1-4B (mode jev1_q35_4b) on CPU, light (the laptop has ~1-2 GB free): the shim rows keep the REQUEST's
option order (the loader never re-sorts option keys), the mode's configs (the jev1 pointer, the fixed 1,121-step
budget, the jevkit stock + shim arm), the stock-shim engine (LoRA off, the tied LM head's label rows == the full
log-softmax readout through the vendored shim) and the paired-vs-stock report. Real-tokenizer tests skip without the
Qwen3.5-4B tokenizer files. The draw itself is verified in-job (heldout phase: attack / doc-overlap / JevBench guards);
train == serve on real draw rows: scripts/check_jev1_train_serve.py (receipts/2026-10-10-jev1-4b/)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
CELL = json.loads((M / "m2_first_cell_rwkv7_jev1_q35_4b.json").read_text(encoding="utf-8"))
PARAMS = json.loads((M / "job_params_jev1_q35_4b.json").read_text(encoding="utf-8"))
R4 = json.loads((M / "m2_first_cell_rwkv7_sizecliff_q35_4b.json").read_text(encoding="utf-8"))
TPL = "jrt-h2o-shim:qwen3.5-4b"


def _tokdir():
    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS

    return DEFAULT_TOKENIZERS / BACKBONES["qwen3.5-4b"].tokenizer_dir


needs_tok = pytest.mark.skipif(not (_tokdir() / "tokenizer.json").is_file(), reason="Qwen3.5-4B tokenizer files")

#: criteria in a NON-sorted order (as the jev1 banking77 / clinc150 rows are)
REQ = {"state": {"customer": "my card payment was declined twice", "region": "EU"},
       "questions": {"q": {"type": "choice", "instructions": "Which intent?",
                           "criteria": {"wrong_exchange_rate": "rate looks wrong", "activate_my_card": "activate",
                                        "card_declined": "a card payment was declined", "beneficiary": "add payee"}}}}


@needs_tok
def test_shim_rows_keep_the_request_option_order_never_resorted():
    from decide.shim_format import bind_template
    from heads.layout import build_row, resolve_template
    from train.run_cell import to_train_record

    spec = bind_template(TPL, _tokdir())
    rec = to_train_record({"id": "m2:t:1", "state": REQ["state"], "questions": REQ["questions"], "labels": {"q": "card_declined"}})
    q = rec["request"]["questions"]["q"]
    given = list(REQ["questions"]["q"]["criteria"])
    assert list(q["criteria"]) == given  # the loader keeps the builder's key order (json + dict insertion order)
    row = build_row(rec["request"], "q", None, template=resolve_template(TPL), layout="g1v2")
    text = spec.tok.decode(row.ids)
    pos = [text.index(f"\n{lab}) {name}: ") for lab, name in zip("ABCD", given)]
    assert pos == sorted(pos)  # A) wrong_exchange_rate, B) activate_my_card, C) card_declined, D) beneficiary
    assert row.option_ids == sorted(given)  # the option AXIS is canonical; each option keeps ITS label below
    for o, cont in zip(row.option_ids, row.continuations):
        assert cont == [spec.label_ids["ABCD"[given.index(o)]]]
    # a re-sorted request would be a DIFFERENT prompt (so a silent re-sort could not pass train == serve)
    resorted = {**rec["request"], "questions": {"q": {**q, "criteria": dict(sorted(q["criteria"].items()))}}}
    assert build_row(resorted, "q", None, template=resolve_template(TPL), layout="g1v2").ids != row.ids


def test_mode_is_the_r4_recipe_on_the_jev1_draw_with_a_fixed_step_budget():
    from jobs.common import costs

    same = ("template", "layout", "head", "loss_arm", "lora_rank", "lr_head", "lr_lora", "warmup_steps", "batch_size",
            "grad_accum", "lr_schedule", "lr_min_frac", "dev_card_every", "dev_card_rows", "best_ckpt",
            "collapse_guard", "act_ckpt", "row_batch", "t_ladder", "train_max_k", "max_k", "cont_chunk",
            "cont_chunk_candidates", "cont_chunk_peak_frac", "parent", "qwen35", "seed", "cohort_sha256")
    for k in same:
        assert CELL[k] == R4[k], k
    assert (CELL["max_steps"], CELL["min_steps"], CELL["lr_horizon_steps"]) == (1121, 1121, 753)
    assert CELL["max_minutes"] == 270 and PARAMS["max_duration_minutes"] <= 899  # r2: 1,121 x 10.92 s x 1.3
    assert CELL["train_set"]["pointer"] == "data/manifests/bschema-draw-jev1.pointer.json"
    assert CELL["n_train_records"] == 96000 and "attack_clean" not in CELL["train_set"]
    ptr = json.loads((ROOT / CELL["train_set"]["pointer"]).read_text(encoding="utf-8"))
    assert ptr["records_uncompressed_sha256"].startswith("45735109") and ptr["n_records"] == 96000
    assert PARAMS["mode"] == "jev1_q35_4b" and PARAMS["train_dir"] == "m2-jev1"
    assert PARAMS["jevkit"]["stock_shim"] is True and PARAMS["jevkit"]["paired_primary"] == "native"
    assert sum(PARAMS["phase_caps_minutes"].values()) + PARAMS["reserve_minutes"] == PARAMS["max_duration_minutes"]
    assert PARAMS["max_duration_minutes"] * 0.063 <= 25.5  # r2 hold $25.01 (270-min clock) and costs.worst_case_usd(PARAMS) <= PARAMS["hard_cap_usd"]
    dests = {d["dest"] for d in PARAMS["context"]["data"]}
    assert {"m2-jev1/records.jsonl.gz", "m2-jev1/bschema-draw-jev1.manifest.jsonl.gz",
            "m2-jev1/bschema-draw-jev1.content.jsonl.gz"} <= dests and not any(d.startswith("m2-b100k/") for d in dests)
    pin = (M / "data_jev1.sha256").read_text(encoding="utf-8")
    assert "m2-jev1/records.jsonl.gz" in pin and "\r" not in pin


@needs_tok
def test_stock_shim_engine_equals_the_full_lm_head_shim_readout():
    """LoRA off (scaling 0 restored after), the label rows of the tied head == the shim on full log-softmax logits."""
    import types

    torch = pytest.importorskip("torch")
    import third_party.h2o_lightning_shim as H
    from decide.shim_format import bind_template
    from jobs.m2_first_cell.screen import StockShimEngine
    from train.lora import LoRALinear

    spec = bind_template(TPL, _tokdir())
    V, D = len(spec.tok), 8

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.emb = torch.nn.Embedding(V, D)
            with torch.no_grad():
                self.emb.weight.copy_(torch.randn(V, D, generator=g))
            self.proj = LoRALinear(torch.nn.Linear(D, D), rank=2, generator=g)
            with torch.no_grad():
                self.proj.lora_B.normal_()

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids, use_cache=False):
            h = self.emb(input_ids).cumsum(1) / (1 + torch.arange(input_ids.shape[1])[None, :, None])
            return types.SimpleNamespace(last_hidden_state=torch.tanh(self.proj(h)))

    text = Text()
    backend = types.SimpleNamespace(model=types.SimpleNamespace(backbone=types.SimpleNamespace(model=text)),
                                    template=types.SimpleNamespace(id=TPL))
    snap = _tokdir()
    eng = StockShimEngine(backend, str(snap))
    mod, c = H.load_contract()
    with eng.lora_off() as n:
        assert n == 1 and text.proj.scaling == 0.0
        got = eng(REQ)["answers"]["q"]
    assert text.proj.scaling == 1.0  # restored
    t, names, descs, labels = c.question_prompt(mod.render_state(REQ["state"], c.state_compact), REQ["questions"]["q"])
    ids = torch.tensor([spec.ids(t)])
    text.proj.scaling = 0.0
    with torch.no_grad():
        h = text(ids).last_hidden_state[0, -1]
        lp = torch.log_softmax(text.emb.weight @ h, -1)
    text.proj.scaling = 1.0
    want = mod.format_answer("choice", names, descs, [float(lp[spec.label_ids[x]]) for x in labels], c.temperature,
                             c.noul_floor)
    assert got["choice"] == want["choice"]
    assert all(abs(got["probabilities"][k] - want["probabilities"][k]) < 1e-5 for k in want["probabilities"])


def test_paired_vs_stock_report_shape_and_success_rule():
    pytest.importorskip("numpy")
    from jevkit import scoring as JS
    from jevkit.answers import answer_from_probs
    from jevkit.items import load_public
    from jobs.m2_first_cell.screen import paired_vs

    try:
        items = load_public()
    except Exception:  # noqa: BLE001 - pinned public items not on this machine
        pytest.skip("JevBench public items")
    good = {it.id: answer_from_probs(it, {l: (0.9 if str(l) == str(it.expected) else 0.1 / (len(it.labels) - 1))
                                          for l in it.labels}) for it in items}
    flat = {it.id: answer_from_probs(it, {l: 1.0 / len(it.labels) for l in it.labels}) for it in items}
    tables = {"native": JS.build_table(items, good), "stock_shim": JS.build_table(items, flat)}
    axes = {"native": (90.0, 60.0), "stock_shim": (90.0, 60.0)}
    d = paired_vs(tables, axes, "native", "stock_shim", B=200)
    assert d["B"] == 200 and d["seed"] == 16
    for k in ("I", "C", "capability", "composite_A"):
        assert d[k]["point"] is not None and d[k]["ci95"] is not None
    assert d["I"]["point"] > 0 and d["I"]["ci95"][0] > 0  # an oracle-ish arm beats the uniform one on every draw
