"""CPU verification of the A-PREREG-SIZECLIFF arms (modes sizecliff_q35_{0p8b,2b,4b}); never touches a GPU.

Per arm (cell config ``jobs/m2_first_cell/m2_first_cell_rwkv7_sizecliff_q35_<arm>.json``):
  draw      ``train.run_cell.load_cell_data`` on the real B1 100k-v2 files: pointer + manifest + content verified, the
            cohort excluded, the attack_clean (5,201) and doc_clean (14) pins re-checked, the 94,786-record draw; the
            drawn-id sha256 must be the SAME for every arm (one data draw)
  readout   train rows (``train.data.items_from_records`` with the arm's template: the rows run_cell trains on) ==
            the generic v3w readout's prompt ids / scored tokens (``scripts/check_q9af_v3w_equivalence.check_row``
            against ``V3WGenericBackend._questions`` on a model-free plugin over this size's REAL tokenizer files),
            on a seeded record sample + the rare strata (K > 16, dict instructions); the same on the 231 public
            JevBench items (native, and the 4B lean template); input tokens per decision on those items
  smoke     a tiny random Qwen3.5 stand-in WITH THIS ARM'S layer pattern (3 GDN : 1 attention) and the real 248,320
            vocabulary, saved as a text-only snapshot beside this size's tokenizer files, loaded by
            ``heads.backbone.load_qwen35_backbone`` (pin check on its config.json); LoRA groups == the modules present;
            ``train.run_cell.smoke_stage`` (cache fork == recompute fork: features + LoRA gradients, act_ckpt on) on
            real drawn rows; two optimiser steps of the cell's Trainer (loss arm, cosine schedule, fork head); then the
            served decider (``heads.decide_backend``) answers JevBench items through ``jevkit.systemone`` (native and,
            for the 4B, lean) with valid SystemOne answers

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_sizecliff_cpu.py --out receipts/2026-10-09-sizecliff/cpu.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
ARMS = ("0p8b", "2b", "4b")
TOKFILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")


def cfg_of(arm: str) -> dict:
    return json.loads((ROOT / f"jobs/m2_first_cell/m2_first_cell_rwkv7_sizecliff_q35_{arm}.json").read_text(
        encoding="utf-8"))


def tok_dir(cfg: dict) -> Path:
    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS, TEMPLATE_BACKBONE

    return DEFAULT_TOKENIZERS / BACKBONES[TEMPLATE_BACKBONE[cfg["template"]]].tokenizer_dir


def check_draw(cfg: dict) -> dict:
    from train.run_cell import load_cell_data

    t0 = time.time()
    train_recs, recs, tr_ids, ev_ids, ident = load_cell_data(
        cfg, str(DATA / "m2/build/sample-20000-seed20261006/records.jsonl.gz"), str(DATA / "m2/build/b1-draw100k-v2"))
    ts = ident.get("train_set") or {}
    return {"n_drawn": len(tr_ids), "n_eval_dev": len(ev_ids),
            "drawn_ids_sha256": hashlib.sha256("\n".join(sorted(tr_ids)).encode()).hexdigest(),
            "train_set_identity": {k: ts.get(k) for k in sorted(ts) if not isinstance(ts.get(k), (list,))},
            "seconds": round(time.time() - t0, 1), "_train_recs": train_recs, "_tr_ids": tr_ids}


def check_readout(cfg: dict, train_recs: dict, tr_ids: list, sample: int, seed: int) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_q9af_v3w_equivalence import check_row, model_free_plugin, stratum

    from decide.backends.v3w_generic import V3WGenericBackend
    from decide.v3w_generic import bind_template
    from heads.layout import hf_segment_encoder
    from train.data import items_from_records
    from train.guard import record_request

    d = tok_dir(cfg)
    spec = bind_template(cfg["template"], d)
    plug, tok = model_free_plugin(str(d))
    backend = V3WGenericBackend(plug, spec)
    enc = hf_segment_encoder(tok)
    rng = random.Random(seed)
    ids = sorted(tr_ids)
    rng.shuffle(ids)
    pick, big, dicts = ids[:sample], [], []
    for i in ids[sample: sample + 30000]:
        qs = train_recs[i]["request"]["questions"].values()
        if any(not isinstance(q.get("instructions"), str) for q in qs):
            dicts.append(i)
        elif any(len(q.get("criteria") or {}) > 16 for q in qs):
            big.append(i)
    pick += big[:200] + dicts[:200]
    strata, mism, n, refused = Counter(), [], 0, Counter()
    for i in pick:
        rec = train_recs[i]
        try:
            items = items_from_records([rec], enc, layout=cfg["layout"], template=cfg["template"])
        except ValueError as e:
            refused[type(e).__name__ + ": " + str(e)[:60]] += 1
            continue
        req = record_request(rec)
        qs = {q["qid"]: q for q in backend._questions(req)}
        for it in items:
            q = qs[it.id.rsplit("/", 1)[1]]
            n += 1
            for s in stratum(it.row, q, req):
                strata[s] += 1
            e = check_row(it, q)
            if e:
                mism.append({"id": it.id, "errors": e})
    return {"template": cfg["template"], "spec_identity": spec.identity(), "records_checked": len(pick), "rows": n,
            "mismatches": len(mism), "mismatch_examples": mism[:5], "strata": dict(strata),
            "refused_records": dict(refused), "ok": not mism and n >= 700 and all(
                strata[s] >= 20 for s in ("k<=16", "k17-255", "words", "dict_instructions"))}


def check_jevbench(template: str, tdir: Path) -> dict:
    import statistics as st

    from decide.v3w_generic import bind_template
    from heads.layout import build_row, resolve_template
    from jevkit.items import load_public
    from jevkit.systemone import item_request

    spec = bind_template(template, tdir)
    bad, toks = 0, []
    for it in load_public():
        req = item_request(it)
        q = spec.questions(req)[0]
        row = build_row(req, "q", None, template=resolve_template(template), layout="g1v2")
        want = [[t] for t in q["tokens"]] if q["coded"] else [list(c) for c in q["conts"]]
        if row.ids != q["prompt"] or [list(c) for c in row.continuations] != want or row.option_ids != q["options"]:
            bad += 1
        toks.append(len(q["prompt"]))
    return {"template": template, "items": len(toks), "mismatches": bad, "input_tokens_mean": round(st.mean(toks), 1),
            "input_tokens_p50": st.median(toks), "input_tokens_max": max(toks), "ok": bad == 0}


def tiny_snapshot(arm_cfg: dict, d: Path) -> dict:
    """A text-only tiny Qwen3.5 with the arm's 3:1 GDN:attention pattern (4 layers) and the real vocabulary."""
    import torch
    import transformers as tf

    real = json.loads((tok_dir(arm_cfg) / "config.json").read_text(encoding="utf-8"))["text_config"]
    torch.manual_seed(0)
    cfg = tf.Qwen3_5TextConfig(vocab_size=real["vocab_size"], hidden_size=32, intermediate_size=64,
                               num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                               linear_value_head_dim=8, layer_types=real["layer_types"][:4])
    model = tf.Qwen3_5ForCausalLM(cfg)
    model.save_pretrained(str(d))
    for f in TOKFILES:
        src = tok_dir(arm_cfg) / f
        if src.is_file():
            (d / f).write_bytes(src.read_bytes())
    return {"layer_types": cfg.layer_types, "config_sha256": hashlib.sha256((d / "config.json").read_bytes()).hexdigest()}


def check_smoke(cfg: dict, train_recs: dict, tr_ids: list, lean: str | None) -> dict:
    import torch

    from decide import Decider
    from decide.v3w_generic import bind_template
    from heads.backbone import group_matches, load_qwen35_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jevkit.answers import parse_answer
    from jevkit.items import load_public
    from jevkit.systemone import SystemOneAdapter
    from train.run_cell import cell_items, parent_encoder, smoke_stage, train_config
    from train.guard import SplitGuard
    from train.trainer import Trainer, with_execution

    out: dict = {}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "snap"
        d.mkdir()
        out["stand_in"] = tiny_snapshot(cfg, d)
        c = json.loads(json.dumps(cfg))
        c["qwen35"]["config_sha256"] = out["stand_in"]["config_sha256"]  # the stand-in's own config pin
        bb = load_qwen35_backbone(str(d), c["qwen35"], device="cpu", dtype="float32")
        found = group_matches(bb.model, bb.spec.lora_groups)
        out["lora_groups"] = {g: [len(found[g]), n] for g, _, n in bb.spec.lora_groups}
        out["lora_groups_ok"] = all(len(found[g]) == n for g, _, n in bb.spec.lora_groups)
        out["provenance"] = bb.provenance
        enc = parent_encoder("qwen35", str(d), c)
        root = ROOT / "harness" / "manifests"
        guard = SplitGuard.from_manifests([root / "dev-manifest.jsonl.gz", root / "test-manifest.jsonl.gz"])
        ids = sorted(tr_ids)[:400]
        items = [x for x in cell_items(train_recs, ids, c, enc, guard, 512, workers=1) if len(x.row.ids) <= 512]
        out["items"] = len(items)
        c["smoke"] = {**c["smoke"], "rows": 3, "max_k": 6, "max_t": 512, "lora_rank": 4}
        out["smoke_stage"] = smoke_stage(bb, items, c, cuda=False)
        model = DecisionModel(bb, c["head"], c.get("head_kwargs"), pad_id=0, seed=c["seed"])
        c2 = {**c, "grad_accum": 2, "lora_rank": 8, "warmup_steps": 1, "dev_card_every": 0, "best_ckpt": False,
              "collapse_guard": False}
        tr = Trainer(model, with_execution(train_config(c2, 2, None, False), c2), items[:16], [], guard=guard)
        losses = [float(tr.train_step()) for _ in range(2)]
        out["train_steps"] = {"losses": losses, "finite": all(torch.isfinite(torch.tensor(losses)).tolist())}
        arms = {"native": c["template"]}
        if lean:
            bind_template(lean, d)
            arms["lean"] = lean
        jev = {}
        its = load_public()[:6]
        for name, tpl in arms.items():
            ad = SystemOneAdapter(Decider(DecisionModelBackend(model, enc, template=tpl, layout="g1v2")), kind="decider")
            recs = [ad.answer_item(it) for it in its]
            jev[name] = {"status": dict(Counter(r["status"] for r in recs)),
                         "valid": sum(parse_answer(it, r["answer"]).valid for it, r in zip(its, recs) if r["status"] == "ok"),
                         "input_tokens": [(r["usage"] or {}).get("input_tokens") for r in recs]}
        out["jevkit_decide"] = jev
        out["ok"] = (out["lora_groups_ok"] and out["smoke_stage"]["pass"] and out["train_steps"]["finite"]
                     and all(v["status"].get("ok") == len(its) and v["valid"] == len(its) for v in jev.values()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep: dict = {"arms": {}}
    draws = set()
    for arm in a.arms.split(","):
        cfg = cfg_of(arm)
        r: dict = {"template": cfg["template"], "qwen35": cfg["qwen35"]}
        dr = check_draw(cfg)
        recs, ids = dr.pop("_train_recs"), dr.pop("_tr_ids")
        draws.add(dr["drawn_ids_sha256"])
        r["draw"] = dr
        r["readout"] = check_readout(cfg, recs, ids, a.sample, a.seed)
        r["jevbench_native"] = check_jevbench(cfg["template"], tok_dir(cfg))
        lean = json.loads((ROOT / f"jobs/m2_first_cell/job_params_sizecliff_q35_{arm}.json").read_text(
            encoding="utf-8")).get("jevkit", {}).get("lean_template")
        if lean:
            r["jevbench_lean"] = check_jevbench(lean, tok_dir(cfg))
        if not a.skip_smoke:
            r["smoke"] = check_smoke(cfg, recs, ids, lean)
        r["ok"] = (r["readout"]["ok"] and r["jevbench_native"]["ok"] and r.get("jevbench_lean", {"ok": True})["ok"]
                   and r.get("smoke", {"ok": True})["ok"] and dr["n_drawn"] == cfg["n_train_records"])
        rep["arms"][arm] = r
        print(json.dumps({arm: {"ok": r["ok"], "n_drawn": dr["n_drawn"], "rows": r["readout"]["rows"],
                                "mismatches": r["readout"]["mismatches"]}}), flush=True)
    rep["same_draw_every_arm"] = len(draws) == 1
    rep["ok"] = rep["same_draw_every_arm"] and all(r["ok"] for r in rep["arms"].values())
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes((s + "\n").encode("utf-8"))
    print(json.dumps({"ok": rep["ok"], "same_draw_every_arm": rep["same_draw_every_arm"]}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
