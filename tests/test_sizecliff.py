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
TPL = {a: f"jrt-v3w-generic:qwen3.5-{a.replace('p', '.')}" for a in ARMS}
LEAN = "jrt-v3w-generic:qwen3.5-4b-lean"
JOB = (M / "first_cell_job.sh").read_text(encoding="utf-8")
CELLS = {a: json.loads((M / f"m2_first_cell_rwkv7_sizecliff_q35_{a}.json").read_text(encoding="utf-8")) for a in ARMS}
PARAMS = {a: json.loads((M / f"job_params_sizecliff_q35_{a}.json").read_text(encoding="utf-8")) for a in ARMS}
Q9 = json.loads((M / "m2_first_cell_rwkv7_q9af_v2_v3w.json").read_text(encoding="utf-8"))


def _tokdir(tpl: str) -> Path:
    return DEFAULT_TOKENIZERS / BACKBONES[TEMPLATE_BACKBONE[tpl]].tokenizer_dir


needs_tok = pytest.mark.skipif(not all((_tokdir(t) / "tokenizer.json").is_file() for t in [*TPL.values(), LEAN]),
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
    for t in [*TPL.values(), LEAN]:
        assert t in TEMPLATE_BACKBONE and len(PINNED_SPEC_SHA256[t]) == 64
    assert BACKBONES["qwen3.5-4b-lean"].lean and not any(BACKBONES[f"qwen3.5-{s}"].lean for s in ("0.8b", "2b", "4b"))
    assert PINNED_SPEC_SHA256["jrt-v3w-generic:qwen3.5"] == "2e4907ee4302d9d196052fc04c5618e88fc8058446644715345eaad93a3a2543"


@needs_tok
def test_bind_every_size_and_lean_train_rows_equal_readout():
    from heads.layout import build_row, resolve_template

    for t in [*TPL.values(), LEAN]:
        spec = bind_template(t, _tokdir(t))
        assert spec.identity()["sha256"] == PINNED_SPEC_SHA256[t] and len(spec.table) == 255
        for q in spec.questions(REQ):
            row = build_row(REQ, q["qid"], None, template=resolve_template(t), layout="g1v2")
            assert row.ids == q["prompt"] and row.option_ids == q["options"], (t, q["qid"])
            assert row.continuations == ([[x] for x in q["tokens"]] if q["coded"] else q["conts"])
            assert row.field_span[1] > row.field_span[0] and all(b > a for a, b in row.option_spans)
            assert row.state_span[1] > row.state_span[0]
    lean = bind_template(LEAN, _tokdir(LEAN))
    full = bind_template(TPL["4b"], _tokdir(TPL["4b"]))
    for ql, qf in zip(lean.questions(REQ), full.questions(REQ)):
        assert "SCHEMA" not in ql["text"] and "QUESTION:\nTYPE: " in ql["text"] and "re-read" not in ql["text"]
        assert len(ql["prompt"]) < len(qf["prompt"]) and ql["tokens"] == qf["tokens"]


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
               "execution_note", "qwen35", "v3w_note", "variant", "q9af", "census_note"}
    for a, c in CELLS.items():
        assert c["parent"] == "qwen35" and c["template"] == TPL[a] and "q9af" not in c
        for k in set(c) | set(Q9):
            if k not in allowed:
                assert c.get(k) == Q9.get(k), (a, k)  # = q9af_v2_v3w's D16 keys (data, attack/doc clean, LR, ...)
        assert c["n_train_records"] == 94786 and c["lora_rank"] == 64 and c["max_minutes"] == 100
        assert c["train_set"]["doc_clean"]["n_excluded"] == 14 and c["train_set"]["attack_clean"]["n_excluded"] == 5201
    for k in ("train_set", "n_train_records", "lora_rank", "max_minutes", "max_steps", "seed", "lr_lora", "head"):
        assert len({json.dumps(c[k], sort_keys=True) for c in CELLS.values()}) == 1, k
    hold = 0.0
    for a, p in PARAMS.items():
        assert p["mode"] == f"sizecliff_q35_{a}" and p["parent_arm"] == "none" and p["screen_parent"] == "none"
        assert p["stage"] == "screen" and p["screen_reference"] == "d1_v3w" and p["screen_reference_match"] == "family"
        assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
        assert costs.worst_case_usd(p) <= p["hard_cap_usd"]
        assert f"{costs.worst_case_usd(p):.2f}" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")
        assert not any(d["dest"].startswith("gdn/") for d in p["context"]["data"])
        assert {c["name"] for c in p["screen_comparators"]} == {"clef_flash", "clef_27b"}
        hold += p["max_duration_minutes"] * 0.063
    assert hold <= 45.0
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
    jk = S.jevkit_block(p, TPL["4b"])
    assert jk["lean_template"] == LEAN and jk["price_parsed"]["usd_per_m_input"] == 0.03
    with pytest.raises(S.ScreenError):  # comparator file missing / wrong sha -> refused
        S.locate_comparators(p, tmp_path, {"x"})
    rec = {"readout": {"template": TPL["4b"]}, "parent": {"source": "none"}}
    assert S.parent_predictions(rec, "unused", "none") == (None, {"source": "none", "readout_template": TPL["4b"]})
    d = {"schema": S.SCHEMA, "measured": True, "verdict": "PROMOTE", "readout": {"template": TPL["4b"],
                                                                              "reference_match": "family"},
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
    assert 'sha256sum -c "$CODE/$Q35_SHA"' in JOB and "qwen35_pin_check" in JOB
    assert JOB.index('elif [ "$PARENT" = qwen35 ]') < JOB.index("phase cell")
    assert 'PARENT_ARM=$(python -I -c' in JOB and '[ "$PARENT_ARM" = none ] && EVAL_PRICES=()' in JOB
    i, j = JOB.index('if [ "$PARENT_ARM" = none ]; then'), JOB.index("fi  # parent_arm in_job")
    assert i < JOB.index("arm_seed seed") < JOB.index("phase parent_$PSTAGE") < j < JOB.index("phase cell")
    for a in ARMS:
        assert PARAMS[a]["submit"]["command"].endswith(f"first_cell_job.sh sizecliff_q35_{a}")
        assert (ROOT / CELLS[a]["qwen35"]["snapshot_sha256_file"]).is_file()
        man = (ROOT / CELLS[a]["qwen35"]["snapshot_sha256_file"]).read_text(encoding="utf-8")
        assert f"{CELLS[a]['qwen35']['config_sha256']}  config.json" in man and "chat_template.jinja" in man
        assert ".safetensors" in man and "\r" not in man
