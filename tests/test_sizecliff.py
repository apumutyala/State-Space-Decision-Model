"""A-PREREG-SIZECLIFF modes sizecliff_q35_{0p8b,2b,4b} on CPU: the stock Qwen3.5 parent (pins, loader, LoRA groups,
the padded single-token cache fork), the per-size generic v3w templates and the 4B lean template (train rows == readout
ids, spans), the screen without a parent arm (+ cross-readout comparators, the jevkit block), the mode configs (one
recipe, one draw, one rank, one clock, budget) and the runner's qwen35 / parent_arm none branches. Real-tokenizer and
real-data tests skip when the files are not on this machine. The full receipt is scripts/check_sizecliff_cpu.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS, PINNED_SPEC_SHA256, TEMPLATE_BACKBONE, bind_template

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs" / "m2_first_cell"
ARMS = ("0p8b", "2b", "4b")
TPL = {a: f"jrt-h2o-shim:qwen3.5-{a.replace('p', '.')}" for a in ARMS}  # PRIMARY: the trainable H2O shim format
V3W = {a: f"jrt-v3w-generic:qwen3.5-{a.replace('p', '.')}" for a in ARMS}  # score-only jevkit arm
LEAN = "jrt-h2o-shim:qwen3.5-4b-lean"
LEAN_V3W = "jrt-v3w-generic:qwen3.5-4b-lean"
JOB = (M / "first_cell_job.sh").read_text(encoding="utf-8")
CELLS = {a: json.loads((M / f"m2_first_cell_rwkv7_sizecliff_q35_{a}.json").read_text(encoding="utf-8")) for a in ARMS}
PARAMS = {a: json.loads((M / f"job_params_sizecliff_q35_{a}.json").read_text(encoding="utf-8")) for a in ARMS}
Q9 = json.loads((M / "m2_first_cell_rwkv7_q9af_v2_v3w.json").read_text(encoding="utf-8"))


def _tokdir(tpl: str) -> Path:
    from decide.shim_format import FAMILIES, is_shim_template

    bb = FAMILIES[tpl.split(":", 1)[1]][0] if is_shim_template(tpl) else TEMPLATE_BACKBONE[tpl]
    return DEFAULT_TOKENIZERS / BACKBONES[bb].tokenizer_dir


def _bind(tpl: str):
    from decide import shim_format as SF

    return (SF.bind_template if SF.is_shim_template(tpl) else bind_template)(tpl, _tokdir(tpl))


ALL_TPL = [*TPL.values(), LEAN, *V3W.values(), LEAN_V3W]
needs_tok = pytest.mark.skipif(not all((_tokdir(t) / "tokenizer.json").is_file() for t in ALL_TPL),
                               reason="Qwen3.5 small tokenizer files")

REQ = {"state": {"passage": "The museum opened in 1902 and moved to the harbour in 1964.", "rooms": 12},
       "questions": {"year": {"type": "choice", "instructions": "When did the museum move?",
                              "criteria": {"A": "1902", "B": "1964", "C": "1971"}},
                     "mood": {"type": "choice", "instructions": {"task": "pick the tone"},
                              "criteria": {"neutral": "Neutral", "positive": "Positive", "negative": "Negative"}},
                     "ok": {"type": "noul", "instructions": "Did it move?",
                            "criteria": {"true": "It moved.", "false": "It did not."}},
                     "cat": {"type": "choice", "instructions": "Which category?",
                             "criteria": {f"c{i:02d}": f"category number {i}" for i in range(40)}}}}


def test_templates_registered_and_pinned():
    from decide import shim_format as SF

    for t in [*V3W.values(), LEAN_V3W]:
        assert t in TEMPLATE_BACKBONE and len(PINNED_SPEC_SHA256[t]) == 64
    for t in [*TPL.values(), LEAN]:
        assert SF.is_shim_template(t) and len(SF.PINNED_SPEC_SHA256[t]) == 64
    assert not SF.is_shim_template("jrt-h2o-shim:qwen3.5-9b") and not SF.is_shim_template(V3W["4b"])
    assert BACKBONES["qwen3.5-4b-lean"].lean and not any(BACKBONES[f"qwen3.5-{s}"].lean for s in ("0.8b", "2b", "4b"))
    assert PINNED_SPEC_SHA256["jrt-v3w-generic:qwen3.5"] == "2e4907ee4302d9d196052fc04c5618e88fc8058446644715345eaad93a3a2543"


@needs_tok
def test_bind_every_size_and_lean_train_rows_equal_readout():
    from heads.layout import build_row, resolve_template

    from decide import shim_format as SF

    for t in ALL_TPL:
        spec = _bind(t)
        pins = SF.PINNED_SPEC_SHA256 if SF.is_shim_template(t) else PINNED_SPEC_SHA256
        assert spec.identity()["sha256"] == pins[t]
        for q in spec.questions(REQ):
            row = build_row(REQ, q["qid"], None, template=resolve_template(t), layout="g1v2")
            assert row.ids == q["prompt"] and row.option_ids == q["options"], (t, q["qid"])
            assert row.continuations == ([[x] for x in q["tokens"]] if q["coded"] else q["conts"])
            assert row.field_span[1] > row.field_span[0] and all(b > a for a, b in row.option_spans)
            assert row.state_span[1] > row.state_span[0]
    lean, full = _bind(LEAN), _bind(TPL["4b"])
    for ql, qf in zip(lean.questions(REQ), full.questions(REQ)):
        assert "<|im_start|>system" not in ql["text"] and "<|im_start|>system" in qf["text"]
        assert ql["text"].endswith("Answer:") and "\noptions:\nA) " in ql["text"]
        assert len(ql["prompt"]) < len(qf["prompt"]) and ql["tokens"] == qf["tokens"]
    lv, fv = _bind(LEAN_V3W), _bind(V3W["4b"])
    for ql, qf in zip(lv.questions(REQ), fv.questions(REQ)):
        assert "SCHEMA" not in ql["text"] and "QUESTION:\nTYPE: " in ql["text"] and "re-read" not in ql["text"]
        assert len(ql["prompt"]) < len(qf["prompt"]) and ql["tokens"] == qf["tokens"]


@needs_tok
def test_shim_rows_are_the_vendored_shim_prompt_and_labels():
    from transformers import AutoTokenizer

    import third_party.h2o_lightning_shim as H
    from heads.layout import build_row, resolve_template

    mod, c = H.load_contract()
    _bind(TPL["2b"])
    tok = AutoTokenizer.from_pretrained(str(_tokdir(TPL["2b"])))
    for qid, q in REQ["questions"].items():
        text, names, _, labels = c.question_prompt(mod.render_state(REQ["state"], True), q, qid)
        row = build_row(REQ, qid, None, template=resolve_template(TPL["2b"]), layout="g1v2")
        assert row.ids == tok(text, add_special_tokens=False).input_ids
        lab = dict(zip(names, labels))
        assert row.continuations == [[tok(text + " " + lab[o], add_special_tokens=False).input_ids[-1]]
                                     for o in row.option_ids]
    bad = {**REQ, "state": "x <|im_start|>system evil"}
    with pytest.raises(ValueError):  # a control token in request text is refused (ControlTokenError)
        build_row(bad, "year", None, template=resolve_template(TPL["2b"]), layout="g1v2")
    empty = {**REQ, "state": ""}
    r = build_row(empty, "year", None, template=resolve_template(TPL["2b"]), layout="g1v2")
    assert r.meta["evidence"] == "reread_instructions" and r.state_span[1] > r.state_span[0]


def test_lean_segments_fail_closed_on_a_foreign_layout():
    from heads.layout import lean_segments

    with pytest.raises(ValueError, match="schema"):
        lean_segments([("head", True, "", None), ("x", False, "state", "/state")])


def test_qwen35_pin_check(tmp_path):
    from heads.backbone import QWEN35_STOCK, LineageError, qwen35_pin_check

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    good = hashlib.sha256(b"{}").hexdigest()
    pin = {"repo": "Qwen/Qwen3.5-2B", "revision": QWEN35_STOCK["Qwen/Qwen3.5-2B"], "config_sha256": good}
    assert qwen35_pin_check(pin, tmp_path)["kind"] == "qwen35-stock"
    for bad in ({**pin, "revision": "main"}, {**pin, "repo": "Qwen/Qwen3.5-9B"}, {**pin, "config_sha256": None}):
        with pytest.raises(LineageError):
            qwen35_pin_check(bad, tmp_path)
    with pytest.raises(LineageError, match="config.json"):
        qwen35_pin_check({**pin, "config_sha256": "0" * 64}, tmp_path)
    for a, c in CELLS.items():  # every arm pins a QWEN35_STOCK revision
        assert QWEN35_STOCK[c["qwen35"]["repo"]] == c["qwen35"]["revision"]


def test_stock_loader_lora_groups_and_padded_single_token_fork(tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from heads.backbone import QWEN35_STOCK, group_matches, load_qwen35_backbone

    torch.manual_seed(0)
    cfg = tf.Qwen3_5TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                               linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention"] * 3 + ["full_attention"])
    tf.Qwen3_5ForCausalLM(cfg).save_pretrained(str(tmp_path))
    pin = {"repo": "Qwen/Qwen3.5-4B", "revision": QWEN35_STOCK["Qwen/Qwen3.5-4B"],
           "config_sha256": hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()}
    bb = load_qwen35_backbone(str(tmp_path), pin, device="cpu", dtype="float32")
    assert bb.spec.attention_free is False and bb.provenance["repo"] == "Qwen/Qwen3.5-4B"
    found = group_matches(bb.model, bb.spec.lora_groups)
    assert {g: len(found[g]) for g, _, _ in bb.spec.lora_groups} == {"gdn": 9, "attn": 4, "mlp": 12}
    assert all(not p.requires_grad for p in bb.model.parameters())
    prefix = list(range(5, 60))
    conts = [[7], [8], [9]]
    bb.fork_mode = "cache"
    _, fc = bb.fork_features(prefix, conts)
    bb.fork_mode = "recompute"
    full = torch.stack([bb.features(torch.tensor([prefix + c]))[0, -1] for c in conts])
    assert torch.allclose(fc.float(), full.float(), atol=1e-5)


def test_mode_configs_one_recipe_one_draw_one_rank_one_clock():
    from jobs.common import costs

    allowed = {"_comment", "parent", "template", "max_minutes", "max_steps", "max_minutes_note", "lora_rank_note",
               "execution_note", "qwen35", "v3w_note", "shim_note", "variant", "q9af", "census_note",
               "cont_chunk", "cont_chunk_candidates", "cont_chunk_peak_frac", "cont_chunk_note",
               "min_steps", "lr_horizon_steps", "step_parity_note"}
    for a, c in CELLS.items():
        assert c["parent"] == "qwen35" and c["template"] == TPL[a] and "q9af" not in c
        for k in set(c) | set(Q9):
            if k not in allowed:
                assert c.get(k) == Q9.get(k), (a, k)  # = q9af_v2_v3w's D16 keys (data, attack/doc clean, LR, ...)
        assert c["n_train_records"] == 94786 and c["lora_rank"] == 64
        assert c["train_set"]["doc_clean"]["n_excluded"] == 14 and c["train_set"]["attack_clean"]["n_excluded"] == 5201
    # the 4B's execution-only option chunking (job-w2t73); the 0.8B / 2B arms ran without it (same math)
    assert CELLS["4b"]["cont_chunk"] == "auto" and "cont_chunk" not in CELLS["0p8b"] and "cont_chunk" not in CELLS["2b"]
    for k in ("train_set", "n_train_records", "lora_rank", "seed", "lr_lora", "head",
              "lr_head", "batch_size", "grad_accum", "loss_arm", "lr_schedule", "warmup_steps", "t_ladder", "train_max_k"):
        assert len({json.dumps(c[k], sort_keys=True) for c in CELLS.values()}) == 1, k
    # PREREG Amendment 2: the 0.8B / 2B ran the 100-min clock; the 4B (r4) runs the 2B's step count with its LR horizon
    assert CELLS["0p8b"]["max_minutes"] == CELLS["2b"]["max_minutes"] == 100
    c4 = CELLS["4b"]
    assert (c4["max_steps"], c4["min_steps"], c4["lr_horizon_steps"], c4["max_minutes"]) == (1121, 1121, 753, 200)
    assert max(c4["t_ladder"]) == 2048 and c4["cont_chunk_candidates"] == [8, 16, 32, 48, 64]
    hold = 0.0
    for a, p in PARAMS.items():
        assert p["mode"] == f"sizecliff_q35_{a}" and p["parent_arm"] == "none" and p["screen_parent"] == "none"
        assert p["stage"] == "screen" and p["screen_reference"] == "d1_v3w" and p["screen_reference_match"] == "cross"
        assert p["screen_readout"] == "native" and p["jevkit"]["extra_templates"] == {"v3w": V3W[a]}
        assert p["jevkit"]["shim_calibration"] is True
        assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
        assert costs.worst_case_usd(p) <= p["hard_cap_usd"]
        assert f"{costs.worst_case_usd(p):.2f}" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")
        assert not any(d["dest"].startswith("gdn/") for d in p["context"]["data"])
        assert {c["name"] for c in p["screen_comparators"]} == {"clef_flash", "clef_27b"}
        hold += p["max_duration_minutes"] * 0.063
    assert hold <= 50.0  # r4 4B hold $20.60 (budget constraint lifted by the coordinator)
    assert PARAMS["4b"]["jevkit"]["lean_template"] == LEAN and "lean_template" not in PARAMS["2b"]["jevkit"]
    for a, usd in (("0p8b", 0.01), ("2b", 0.02), ("4b", 0.03)):
        from jevkit.meter import parse_price

        assert parse_price(PARAMS[a]["jevkit"]["price"]).usd_per_m_input == usd


def test_screen_blocks_parent_none_comparators_and_jevkit(tmp_path):
    from jobs.m2_first_cell import screen as S

    p = PARAMS["4b"]
    with pytest.raises(S.ScreenError):
        S.jevkit_block({"jevkit": {"price": "kind=nope,in=1,out=0,source=x,as_of=2026-10-09"}}, TPL["4b"])
    with pytest.raises(S.ScreenError):
        S.jevkit_block({"jevkit": {**p["jevkit"], "lean_template": "jrt-rwkv-g1-v3w"}}, TPL["4b"])
    with pytest.raises(S.ScreenError):  # the lean template must be in the screen readout's family (shim)
        S.jevkit_block({"jevkit": {**p["jevkit"], "lean_template": LEAN_V3W}}, TPL["4b"])
    jk = S.jevkit_block(p, TPL["4b"])
    assert jk["lean_template"] == LEAN and jk["price_parsed"]["usd_per_m_input"] == 0.03
    assert S.readout_family(TPL["4b"]) == "h2o_shim" and S.template_available(TPL["4b"])
    assert S.same_readout("jrt-rwkv-g1-v3w", TPL["4b"], "cross")
    assert not S.same_readout("jrt-rwkv-g1-v3w", TPL["4b"], "family")
    ro = S.screen_readout(p, CELLS["4b"])
    assert ro["template"] == TPL["4b"] and ro["native"] and ro["reference_match"] == "cross"
    assert S.reference_block(p, TPL["4b"])["artifact"] == "art-zskmx"
    with pytest.raises(S.ScreenError):  # comparator file missing / wrong sha -> refused
        S.locate_comparators(p, tmp_path, {"x"})
    rec = {"readout": {"template": TPL["4b"]}, "parent": {"source": "none"}}
    assert S.parent_predictions(rec, "unused", "none") == (None, {"source": "none", "readout_template": TPL["4b"]})
    d = {"schema": S.SCHEMA, "measured": True, "verdict": "PROMOTE", "readout": {"template": TPL["4b"],
                                                                              "reference_match": "cross"},
         "child": {"point": 40.0, "lo": 38.0, "hi": 42.0}, "reference": {"point": 38.0, "lo": 36.0, "hi": 40.0},
         "child_minus_reference": {"point": 2.0, "lo": 0.0, "hi": 4.0}, "parent_stock_readout": None,
         "child_minus_parent": None, "status": {"child": {"ok": 644}}, "proxy": {"rows": 644},
         "reference_identity": {"readout_template": "jrt-rwkv-g1-v3w"},
         "parent_identity": {"source": "none", "readout_template": TPL["4b"]}}
    assert S.valid_result(d)
    assert not S.valid_result({**d, "parent_identity": {"source": "in_job", "readout_template": TPL["4b"]}})


DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")


@pytest.mark.skipif(not (DATA / "m2/screen_comparators").is_dir() or not (DATA / "slices/dev-rows.jsonl.gz").is_file(),
                    reason="comparator files / DEV rows")
def test_comparators_cover_the_proxy_and_pair_with_the_frozen_scorer():
    import sys

    from harness import config as HC

    if (HC.KIT_CLONE / "decision_index").is_dir() and str(HC.KIT_CLONE) not in sys.path:
        sys.path.insert(1, str(HC.KIT_CLONE))
    pytest.importorskip("decision_index")
    from jobs.common.dev_proxy import proxy_ids
    from jobs.m2_first_cell import screen as S

    ids = proxy_ids("proxy600")
    comps = S.locate_comparators(PARAMS["2b"], DATA / "m2", ids)
    rows, _ = S.proxy_rows(DATA / "slices/dev-rows.jsonl.gz")
    flash = S.read_predictions(comps[0]["path"])
    out = S.comparator_report(rows, flash, comps, 200)
    assert round(out["clef_flash"]["comparator"]["point"], 2) == 59.03
    assert round(out["clef_27b"]["comparator"]["point"], 2) == 64.56
    assert out["clef_flash"]["child_minus_comparator"]["point"] == 0.0


def test_runner_qwen35_and_parent_arm_none():
    assert "qwen35) ;;" in JOB and 'elif [ "$PARENT" = qwen35 ]; then' in JOB
    assert 'sha256sum -c "$CODE/$Q35_SHA"' in JOB and "qwen35_pin_check" in JOB and "readout-spec.json" in JOB
    assert "S.bind_template if S.is_shim_template(t) else V.bind_template" in JOB
    assert JOB.index('elif [ "$PARENT" = qwen35 ]') < JOB.index("phase cell")
    assert 'PARENT_ARM=$(python -I -c' in JOB and '[ "$PARENT_ARM" = none ] && EVAL_PRICES=()' in JOB
    i, j = JOB.index('if [ "${PARENT_ARM:-in_job}" = none ]; then'), JOB.index("fi  # parent_arm in_job")
    assert i < JOB.index("arm_seed seed") < JOB.index("phase parent_$PSTAGE") < j < JOB.index("phase cell")
    for a in ARMS:
        assert PARAMS[a]["submit"]["command"].endswith(f"first_cell_job.sh sizecliff_q35_{a}")
        assert (ROOT / CELLS[a]["qwen35"]["snapshot_sha256_file"]).is_file()
        man = (ROOT / CELLS[a]["qwen35"]["snapshot_sha256_file"]).read_text(encoding="utf-8")
        assert f"{CELLS[a]['qwen35']['config_sha256']}  config.json" in man and "chat_template.jinja" in man
        assert ".safetensors" in man and "\r" not in man


@pytest.mark.skipif(not (DATA / "m2/screen_reference/art-zskmx").is_dir() or not (DATA / "slices/dev-rows.jsonl.gz").is_file(),
                    reason="reference file / DEV rows")
def test_report_without_parent_is_valid_and_scores_the_child():
    import sys

    from harness import config as HC

    if (HC.KIT_CLONE / "decision_index").is_dir() and str(HC.KIT_CLONE) not in sys.path:
        sys.path.insert(1, str(HC.KIT_CLONE))
    pytest.importorskip("decision_index")
    from jobs.m2_first_cell import screen as S

    rows, prox = S.proxy_rows(DATA / "slices/dev-rows.jsonl.gz")
    ref_path = DATA / "m2/screen_reference/art-zskmx/child.v3w.results.jsonl"
    ref = S.read_predictions(ref_path)
    ident = {"artifact": "art-zskmx", "readout_template": "jrt-rwkv-g1-v3w"}
    ro = {"template": TPL["4b"], "layout": "g1v2", "reference_match": "cross"}
    out = S.build_report(rows, prox, ref, None, ref, ident, PARAMS["4b"], 1.0, 200, readout=ro,
                         parent_ident={"source": "none", "readout_template": TPL["4b"]})
    assert out["child"]["point"] == out["reference"]["point"] and out["child_minus_parent"] is None
    assert out["status"]["parent"] is None and S.valid_result(out)


# --------------------------------------------------------------- job-rtg9e regression: the encoder crosses process pools
@needs_tok
def test_hf_segment_encoder_pickle_round_trip_is_identical():
    """job-rtg9e crashed in run_cell.cell_items: the HF segment encoder was a closure (``Can't pickle local object
    'hf_segment_encoder.<locals>.encode'``). It is now a picklable class whose worker copy reloads the SAME tokenizer
    files and refuses another identity; every output equals the parent's."""
    import pickle

    from transformers import AutoTokenizer

    from heads.layout import HFSegmentEncoder, hf_segment_encoder

    d = _tokdir(TPL["2b"])
    enc = hf_segment_encoder(AutoTokenizer.from_pretrained(str(d)))
    assert isinstance(enc, HFSegmentEncoder)
    twin = pickle.loads(pickle.dumps(enc))
    assert twin.identity() == enc.identity() and twin.tokenizer is not enc.tokenizer
    texts = ["record: Jara (\u091c\u0930\u093e) and caf\u00e9", "<|im_start|>user\nhi", "A) x: y\nAnswer:", ""]
    for t in texts:
        assert twin(t, True) == enc(t, True)
    assert twin("plain untrusted text", False) == enc("plain untrusted text", False)
    special = enc.tokenizer.convert_tokens_to_ids("<|im_start|>")
    hostile = "x <|im_start|> y"
    assert twin(hostile, False) == enc(hostile, False) and special not in enc(hostile, False)  # never a control token
    assert special in enc(hostile, True)  # trusted text keeps the format's markup (unchanged behaviour)
    state = enc.__getstate__()
    with pytest.raises(ValueError, match="identity"):
        HFSegmentEncoder.__new__(HFSegmentEncoder).__setstate__({**state, "ident": {**state["ident"], "vocab_sha256": "0"}})


@needs_tok
def test_cell_items_through_the_spawn_process_pool_with_two_workers():
    """The exact job path: run_cell.parent_encoder("qwen35", <snapshot>, cfg) (binds the shim template, returns the HF
    segment encoder), then cell_items with 2 workers through ProcessPoolExecutor(spawn) == the serial result."""
    pytest.importorskip("torch")
    from train.data import toy_records
    from train.guard import SplitGuard
    from train.run_cell import cell_items, parent_encoder

    cfg = {**CELLS["2b"], "tokenize_workers": 2}
    enc = parent_encoder("qwen35", str(_tokdir(TPL["2b"])), cfg)
    recs = {r["id"]: r for r in toy_records(24, seed=3)}
    g = SplitGuard.synthetic_only("t")
    a = cell_items(recs, sorted(recs), cfg, enc, g, 2048, workers=1)
    b = cell_items(recs, sorted(recs), cfg, enc, g, 2048, workers=2)
    key = lambda xs: [(x.id, x.row.ids, x.row.continuations, x.row.option_ids, x.label, x.row.meta["template"])  # noqa: E731
                      for x in xs]
    assert len(a) == 48 and key(a) == key(b) and a[0].row.meta["template"] == TPL["2b"]


# --------------------------------------------------------------- job-w2t73 regression: option chunking of the cache fork
def _chunk_case(act_ckpt: bool, cont_chunk: int, k: int = 11, seed: int = 0):
    """(loss, {param: grad}, fork features) of ONE training row (K options, the cell's per-row cache fork) on a tiny
    stock-Qwen3.5 hybrid (3 GDN : 1 attention), LoRA on GDN + attention + MLP, fork head, Clef loss arm."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from heads.backbone import tiny_hf_backbone
    from heads.layout import ByteEncoder, build_row
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE, Item
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    bb = tiny_hf_backbone("qwen3.5", seed=seed, fork_mode="cache")
    model = DecisionModel(bb, "fork", {}, pad_id=0, seed=seed)
    req = {"state": "the museum moved to the harbour in 1964; rooms 12; curator Ito",
           "questions": {"q": {"type": "choice", "instructions": "which option?",
                               "criteria": {f"o{i:02d}": f"option number {i}" for i in range(k)}}}}
    row = build_row(req, "q", ByteEncoder(), template=TOY_TEMPLATE, layout="ordered")
    cfg = TrainConfig(head="fork", layout="ordered", fork_mode="cache", lora_rank=4, batch_size=1, grad_accum=1,
                      loss=LossConfig(arm="clef"), act_ckpt=act_ckpt, cont_chunk=cont_chunk, seed=seed)
    it = Item("toy/q", row, 3)
    tr = Trainer(model, cfg, [it], [], guard=SplitGuard.synthetic_only("t"))
    torch.manual_seed(1)
    with torch.no_grad():  # non-zero LoRA B so the LoRA gradients are not trivially equal
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0.0, 0.05)
    model.train()
    model.zero_grad(set_to_none=True)
    fw = model([row])
    loss = tr._supervised_loss(fw, [it], 1.0)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), grads, bb.cont_chunk


@pytest.mark.parametrize("act_ckpt", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 5])
def test_cont_chunk_gradients_equal_the_unchunked_step(act_ckpt, chunk):
    """cont_chunk (the 4B arm's execution knob) changes only how many forked cache copies are alive at once: the loss
    and EVERY LoRA / head gradient equal the un-chunked step (same row, same init), with and without act_ckpt."""
    torch = pytest.importorskip("torch")
    l0, g0, c0 = _chunk_case(act_ckpt, 0)
    l1, g1, c1 = _chunk_case(act_ckpt, chunk)
    assert c0 == 0 and c1 == chunk
    assert abs(l0 - l1) < 1e-6
    assert set(g0) == set(g1) and any("lora_A" in n for n in g0) and any(n.startswith("head") for n in g0)
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-4, atol=1e-6), n


def test_cont_chunk_is_execution_only_and_refuses_unsupported_paths():
    from train.trainer import TrainConfig, with_execution

    base = TrainConfig(head="fork", lora_rank=4)
    assert TrainConfig(head="fork", lora_rank=4, cont_chunk=16).hash() == base.hash()  # same recipe identity
    assert with_execution(base, {"cont_chunk": 16}).cont_chunk == 16
    assert with_execution(base, {"cont_chunk": "auto"}).cont_chunk == 0  # resolved by run_cell's probe


def test_cont_chunk_keeps_only_one_chunk_of_forked_copies_saved():
    """The point of the knob: with chunking, the forward saves (for backward) far fewer bytes, because each chunk's K
    cache copies are recomputed in backward instead of being held from the forward (reentrant checkpoint)."""
    torch = pytest.importorskip("torch")
    from heads.backbone import tiny_hf_backbone
    from train.lora import apply_lora

    bb = tiny_hf_backbone("qwen3.5", seed=0, fork_mode="cache")
    apply_lora(bb.model, bb.spec.lora_groups, 4, generator=torch.Generator().manual_seed(0))
    bb.train()
    prefix, conts = list(range(5, 200)), [[30 + i, 40] for i in range(24)]

    def saved(chunk):
        bb.cont_chunk = chunk
        total = [0]

        def pack(t):
            total[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _, f = bb.fork_features(prefix, conts)
        f.pow(2).sum().backward()
        return total[0]

    full, chunked = saved(0), saved(4)
    assert chunked < 0.2 * full, (full, chunked)


# --------------------------------------------------------------- job-anx5y regressions: the probe and the cache leak
class _FakeAllocator:
    """A fake device: each candidate's step needs fixed + per_option x chunk GB; OOM over the capacity. ``leak`` GB per
    finished step stays allocated unless the caller frees (the job-anx5y failure: freed=False)."""

    def __init__(self, fixed=10.0, per_option=0.9, capacity=79.0, leak=0.0):
        self.fixed, self.per, self.cap, self.leak, self.held, self.calls = fixed, per_option, capacity, leak, 0.0, []

    def step(self, chunk):
        self.calls.append(chunk)
        need = self.held + self.fixed + self.per * chunk
        if need > self.cap:
            return {"ok": False, "oom": "fake OOM", "peak_gb": self.cap}
        self.held += self.leak
        return {"ok": True, "peak_gb": need, "step_s": 1.0}


def test_cont_chunk_probe_chooses_the_largest_passing_candidate():
    from train.run_cell import choose_cont_chunk

    fa = _FakeAllocator()  # 8: 17.2, 16: 24.4, 32: 38.8, 48: 53.2, 64: 67.6 GB; limit 64.6
    out = choose_cont_chunk([64, 32, 16, 8, 48], fa.step, 64.6)
    assert out["chosen"] == 48 and fa.calls == [8, 16, 32, 48, 64]  # smallest first, stops at the first failure
    assert out["points"]["64"]["peak_gb"] > 64.6
    oom = _FakeAllocator(fixed=80.0)  # even the smallest OOMs -> nothing chosen (the job refuses: never a failed chunk)
    assert choose_cont_chunk([64, 32, 16, 8], oom.step, 64.6)["chosen"] is None and oom.calls == [8]
    leaky = _FakeAllocator(leak=30.0)  # memory not freed between candidates: still only a PASSING candidate is chosen
    out = choose_cont_chunk([64, 32, 16, 8], leaky.step, 64.6)
    assert out["chosen"] in out["candidates"] and out["points"][str(out["chosen"])]["ok"]
    assert all(out["points"][str(c)]["ok"] for c in out["candidates"] if c <= out["chosen"])


@pytest.mark.parametrize("chunk", [0, 4])
def test_forked_caches_are_released_after_each_step_under_act_ckpt(chunk):
    """job-anx5y: with act_ckpt every forked cache stayed alive (a reference cycle through the autograd graph that gc
    cannot see): +234 MB per step on this tiny model before the fix. Live tensor bytes must not grow step over step."""
    import gc

    torch = pytest.importorskip("torch")
    from heads.backbone import tiny_hf_backbone
    from heads.layout import ByteEncoder, build_row
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE, Item
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    def live():
        gc.collect()
        return sum(o.numel() * o.element_size() for o in gc.get_objects() if torch.is_tensor(o))

    bb = tiny_hf_backbone("qwen3.5", seed=0, fork_mode="cache")
    req = {"state": "x " * 150, "questions": {"q": {"type": "choice", "instructions": "which?",
                                                     "criteria": {f"o{i:02d}": f"opt {i}" for i in range(20)}}}}
    row = build_row(req, "q", ByteEncoder(), template=TOY_TEMPLATE, layout="ordered")
    items = [Item(f"t/{i}", row, 3) for i in range(4)]
    cfg = TrainConfig(head="fork", layout="ordered", fork_mode="cache", lora_rank=4, batch_size=1, grad_accum=4,
                      loss=LossConfig(arm="clef"), act_ckpt=True, cont_chunk=chunk)
    tr = Trainer(DecisionModel(bb, "fork", {}, pad_id=0, seed=0), cfg, items, [], guard=SplitGuard.synthetic_only("t"))
    tr.train_step()
    first = live()
    for _ in range(3):
        tr.train_step()
    assert live() - first < 5e6  # bytes; before the fix: ~700 MB more after three steps
