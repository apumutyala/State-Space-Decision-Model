"""CPU preflight of a first-cell training mode (lead 2026-10-12, after job-79fki exit 25): run_cell's PRE-TRAIN path on
the REAL data with the job's own functions, then one optimiser step and one dev card on a tiny random model whose
vocabulary is the real tokenizer's, so every stop rule up to the first step is exercised before any GPU spend.

  python -X utf8 -I scripts/cell_preflight.py --params jobs/m2_first_cell/job_params_<mode>.json --data /data \
      --tokenizer <Qwen3.5-4B tokenizer dir> --out <report.json>

Checks (each one is a run_cell exit when it fails on the GPU):
  * load_cell_data (pointer + manifest + content-map shas, cohort, attack / doc-overlap guards, exclude_files)  [27/28/34]
  * cell_items: train (train_layout), eval, dev-card slice (dev_card_records ids pin, id-disjoint)              [25]
  * min_train_items, assert_no_truncation (every drawn question survives the K / T filter): the DROPPED ids and
    their token lengths are listed                                                                             [25]
  * lm_label: every row's continuations are single tokens; labels within range; canonical meta present
  * a tiny lm_label model (real vocab): one Trainer step on the 16 shortest training rows (finite loss, grads) and
    one dev card on 16 slice rows
Not covered (GPU-only): the T ladder / measured T_max, memory, the clock. Exit 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("SSDM_DATA", a.data)
    p = json.loads((ROOT / a.params).read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    data = Path(a.data)
    rep: dict = {"params": a.params, "cell_config": p["cell_config"], "checks": {}, "fail": []}
    t0 = time.time()

    def fail(name, msg):
        rep["checks"][name] = {"pass": False, "detail": msg}
        rep["fail"].append(name)

    from train.guard import SplitGuard
    from train.run_cell import cell_items, dev_card_source, load_cell_data, parent_encoder, train_max_k

    enc = parent_encoder(cfg.get("parent", "qwen35"), a.tokenizer, cfg)
    guard = SplitGuard.from_manifests([ROOT / "harness/manifests/dev-manifest.jsonl.gz",
                                       ROOT / "harness/manifests/test-manifest.jsonl.gz"])
    rep["guard_note"] = ("harness manifests (the job uses the guard phase's readiness index; the in-job guard / heldout "
                         "phases already passed on this data)")
    records = data / "m2/records.jsonl.gz"
    try:
        train_recs, eval_recs, tr_ids, ev_ids, ident = load_cell_data(cfg, str(records), str(data / p["train_dir"]))
        rep["checks"]["load_cell_data"] = {"pass": True, "n_train": len(tr_ids), "n_eval": len(ev_ids)}
    except Exception as e:  # noqa: BLE001
        fail("load_cell_data", f"{type(e).__name__}: {e}")
        return finish(rep, a.out, t0)
    from jobs.m2_first_cell.cell_budget import check as cell_budget_check

    from jobs.m2_first_cell.screen import estimate as screen_estimate

    se = screen_estimate(p)  # the setup gate (exit 15): the screen projection must fit its cap
    if se["ok"]:
        rep["checks"]["screen_projection"] = {"pass": True, **se}
    else:
        fail("screen_projection", se)
    cb = cell_budget_check(p, cfg)  # job-wm8b5: the cell cap must cover the clock + the measured pre / post time
    if cb["pass"]:
        rep["checks"]["cell_budget"] = cb
    else:
        fail("cell_budget", cb)
    t_cap = max(cfg["t_ladder"])
    train_items = cell_items(train_recs, tr_ids, cfg, enc, guard, t_cap, workers=1, train=True)
    eval_items = cell_items(eval_recs, ev_ids, cfg, enc, guard, t_cap, workers=1, max_k=cfg["max_k"])
    try:
        dsrc = dev_card_source(cfg, str(data / p["train_dir"]))
        if dsrc is not None and set(dsrc[1]) & set(tr_ids):
            raise ValueError("dev_card_records ids overlap the training draw")
        card_items = eval_items if dsrc is None else cell_items(dsrc[0], dsrc[1], cfg, enc, guard, t_cap, workers=1)
        rep["checks"]["dev_card_source"] = {"pass": True, "n_card_items": len(card_items)}
    except Exception as e:  # noqa: BLE001
        fail("dev_card_source", f"{type(e).__name__}: {e}")
        card_items = []
    want = sum(len((train_recs[i].get("labels") or {})) for i in tr_ids)
    kept = {(x.id.rsplit("/", 1)[0], x.id.rsplit("/", 1)[1]) for x in train_items}
    dropped = sorted(f"{i}/{q}" for i in tr_ids for q in (train_recs[i].get("labels") or {}) if (i, q) not in kept)
    nt = {"want": want, "kept": len(train_items), "dropped_ids": dropped, "t_cap": t_cap,
          "max_row_tokens": max((len(x.row.ids) for x in train_items), default=0)}
    if dropped:  # lengths of the dropped rows under the TRAINING layout (why the filter dropped them)
        from heads.layout import build_row
        from train.guard import record_request

        lens = {}
        for d in dropped[:20]:
            rid, q = d.rsplit("/", 1)
            try:
                r = build_row(record_request(train_recs[rid]), q, enc, template=cfg["template"],
                              layout=cfg.get("train_layout") or cfg["layout"])
                lens[d] = {"prefix_tokens": len(r.ids), "n_options": r.n_options}
            except Exception as e:  # noqa: BLE001
                lens[d] = {"error": f"{type(e).__name__}: {e}"}
        nt["dropped_rows"] = lens
    if cfg.get("assert_no_truncation") and dropped:
        fail("assert_no_truncation", nt)
    else:
        rep["checks"]["assert_no_truncation"] = {"pass": True, **nt}
    mti = int(cfg.get("min_train_items") or 0)
    if len(train_items) < mti:
        fail("min_train_items", {"kept": len(train_items), "min_train_items": mti})
    else:
        rep["checks"]["min_train_items"] = {"pass": True, "kept": len(train_items), "min": mti}
    rep["checks"]["n_train_records"] = {"pass": len(tr_ids) == int(cfg["n_train_records"]), "draw": len(tr_ids),
                                        "config": cfg["n_train_records"], "max_steps_epoch": len(tr_ids) // 8,
                                        "config_max_steps": cfg["max_steps"]}
    if not rep["checks"]["n_train_records"]["pass"]:
        rep["fail"].append("n_train_records")
    # lm_label / canonical rows
    bad = [x.id for x in train_items + card_items
           if any(len(c) != 1 for c in x.row.continuations) or not (0 <= int(x.label) < x.row.n_options)]
    canon = sum(1 for x in train_items if (x.row.meta or {}).get("canon"))
    shuffled = sum(1 for x in train_items if (x.row.meta or {}).get("canon_layout") == cfg.get("train_layout"))
    ties = sum(1 for x in train_items + card_items if ((x.row.meta or {}).get("canon") or {}).get("tie_sets"))
    rep["checks"]["rows"] = {"pass": not bad, "bad_rows": bad[:20], "canon_rows": canon, "train_layout_rows": shuffled,
                             "rows_with_tie_sets": ties, "train_max_k": train_max_k(cfg),
                             "max_options": max((x.row.n_options for x in train_items), default=0)}
    if bad:
        rep["fail"].append("rows")
    if cfg.get("train_layout") and shuffled != len(train_items):
        fail("train_layout", {"rows": len(train_items), "in_train_layout": shuffled})
    # one optimiser step + one dev card on a tiny model with the real vocabulary
    try:
        rep["checks"]["tiny_step"] = tiny_step(cfg, enc, train_items, card_items, guard)
    except Exception as e:  # noqa: BLE001
        fail("tiny_step", f"{type(e).__name__}: {e}")
    return finish(rep, a.out, t0)


def tiny_step(cfg, enc, train_items, card_items, guard) -> dict:
    import torch
    import transformers as tf

    from heads.backbone import HiddenStateBackbone, qwen35_spec
    from heads.model import DecisionModel
    from train.run_cell import train_config
    from train.trainer import Trainer, with_execution

    from decide.shim_format import is_shim_template, spec_for_template

    if not is_shim_template(cfg["template"]):
        raise ValueError("tiny_step is wired for shim-template (lm_label) cells")
    vocab = len(spec_for_template(cfg["template"]).tok)
    torch.manual_seed(0)
    tcfg = tf.Qwen3_5TextConfig(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                num_attention_heads=2, num_key_value_heads=1, head_dim=8, linear_num_key_heads=2,
                                linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
                                layer_types=["linear_attention", "full_attention"], tie_word_embeddings=True)
    lm = tf.Qwen3_5ForCausalLM(tcfg).float().eval()
    bb = HiddenStateBackbone(lm.model, spec=qwen35_spec(tcfg, "tiny-preflight", "tiny"), fork_mode="cache")
    model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
    short = sorted(train_items, key=lambda x: len(x.row.ids))[:16]
    cards = sorted(card_items, key=lambda x: len(x.row.ids))[:16]
    tc = with_execution(train_config(cfg, 2, None, False), cfg)
    tc.act_ckpt = False
    tc.dev_card_every, tc.best_ckpt, tc.collapse_guard = 0, False, False
    tr = Trainer(model, tc, short, cards, guard=guard)
    loss = tr.train_step()
    from train.evalcard import run_card

    card = run_card(model, cards, tr.step) if cards else None
    ok = math.isfinite(loss) and (card is None or math.isfinite(card["raw"]["nll"]))
    return {"pass": bool(ok), "loss": loss, "card_nll": None if card is None else card["raw"]["nll"],
            "n_rows": len(short), "n_card": len(cards), "vocab": vocab}


def finish(rep: dict, out: str, t0: float) -> int:
    rep["pass"] = not rep["fail"]
    rep["seconds"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], "fail": rep["fail"],
                      "no_truncation": {k: v for k, v in (rep["checks"].get("assert_no_truncation") or {}).items()
                                        if k != "detail"}}, default=str)[:3000])
    return 0 if rep["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
