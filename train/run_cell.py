"""One m2 pilot cell on a real backbone (paid; GPU). Used by jobs/rwkv7 stage B (LoRA pilot with the cache fork head).

    python -m train.run_cell --config jobs/rwkv7/pilot_cell.json --model-dir <snapshot> --records <records.jsonl.gz> \
        --out <dir> --ckpt-dir <dir> --stage smoke|ladder|train|all

Stages (pre-registered in the config; exit codes in jobs/rwkv7/JOB_SPEC.md):
  smoke   cache fork == recompute fork on real rows (features and LoRA gradients), GPU, bf16      -> exit 21 on mismatch
  ladder  one optimiser step per T in t_ladder (max K rows) and the peak memory of each; the largest T under
          max_peak_gb becomes the training cap T_max                                             -> exit 22 if none fits
          measured rate: if even the fastest fitting rung's step time cannot reach 2 x pass-rule window steps
          within max_minutes                                                                       -> exit 26
  train   LoRA + fork head, CE arm, wall-clock capped, checkpoint/resume ($GMN_CHECKPOINT_DIR), eval cards on held-out
          pilot rows; non-finite loss -> exit 23; peak over max_peak_gb -> exit 24; after 10 steps, the measured mean
          step time cannot reach 2 x window steps in the time left -> exit 26 (a throughput finding, not pilot-fail)
Writes <out>/cell.json (merged per stage) and <out>/train_log.jsonl.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

EXIT = {"fork_mismatch": 21, "no_T_fits": 22, "nonfinite": 23, "memory": 24, "data": 25, "too_slow": 26}
CHECK_AFTER_STEPS = 10  # the in-training rate check uses the mean of the first steps actually run


# ----------------------------------------------------------------------------- data (CPU-testable)
def to_train_record(r: dict) -> dict:
    """m2 sample record (data/build.py: state/questions/labels at top level) -> train.data record format."""
    return {"id": r["id"], "request": {"model": "m2", "state": r["state"], "questions": r["questions"]},
            "labels": r["labels"]}


def split_ids(ids: list[str], n_train: int, n_eval: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic disjoint pilot split by sha256(seed:id) order."""
    order = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return order[n_eval: n_eval + n_train], order[:n_eval]


def keep_item(row, max_k: int, max_t: int) -> bool:
    return row.n_options <= max_k and len(row.ids) <= max_t and max(len(c) for c in row.continuations) <= 16


class RwkvSegmentEncoder:
    """Trusted text verbatim; untrusted text fails closed if it yields a special id (World tokenizer: only id 0)."""

    def __init__(self, tok, special: set[int]):
        self.tok, self.special = tok, set(special) | {0}

    def identity(self) -> dict:
        vocab = hashlib.sha256(json.dumps(sorted(self.tok.get_vocab().items())).encode()).hexdigest()
        return {"kind": "rwkv7-hf-trie", "vocab_sha256": vocab, "untrusted": "fail-closed on special ids"}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        from decide.untrusted import ControlTokenError

        ids = self.tok(text, add_special_tokens=False).input_ids
        if not trusted and self.special.intersection(ids):
            raise ControlTokenError("untrusted text encodes to a special token")
        return ids


def steps_needed(rule: dict) -> int:
    """The pass rule compares two disjoint windows of steps, so fewer than 2 x window steps can never pass."""
    return 2 * rule.get("window", 50)


def too_slow(step_s: float, minutes_left: float, need_steps: int) -> dict:
    """Measured stop condition (exit 26): at the measured seconds per optimiser step, can the remaining wall-clock
    budget still reach the steps the pass rule needs? A no is a throughput finding, not a pilot-fail."""
    affordable = 60 * minutes_left / max(step_s, 1e-9)
    return {"step_s": step_s, "minutes_left": minutes_left, "affordable_steps": affordable, "need_steps": need_steps,
            "too_slow": affordable < need_steps}


def pass_rule(history: list[dict], cards: list[dict], peak_gb: float, rule: dict) -> dict:
    """Pre-registered pilot verdict: finite, loss falls by >= loss_drop_frac_min (mean of last w vs first w steps),
    held-out NLL improves from the step-0 card to the last card, peak within budget."""
    w = rule.get("window", 50)
    losses = [h["loss"] for h in history]
    finite = all(math.isfinite(x) for x in losses) and bool(losses)
    first = sum(losses[:w]) / max(1, len(losses[:w])) if losses else float("nan")
    last = sum(losses[-w:]) / max(1, len(losses[-w:])) if losses else float("nan")
    drop = (first - last) / first if finite and first > 0 else float("nan")
    nll0 = cards[0]["raw"]["nll"] if cards else None
    nll1 = cards[-1]["raw"]["nll"] if len(cards) > 1 else None
    checks = {"finite": finite, "enough_steps": len(losses) >= 2 * w,
              "loss_drop": bool(finite and drop >= rule["loss_drop_frac_min"]),
              "eval_nll_improves": bool(nll0 is not None and nll1 is not None and nll1 < nll0),
              "memory": peak_gb <= rule["max_peak_gb"]}
    return {"pass": all(checks.values()), "checks": checks, "loss_first": first, "loss_last": last,
            "loss_drop_frac": drop, "eval_nll_first": nll0, "eval_nll_last": nll1, "peak_gb": peak_gb}


# ----------------------------------------------------------------------------- GPU stages
def _checkpoint_post():
    url = os.environ.get("GMN_METADATA_URL")
    if not url:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(url + "/v1/checkpoint", data=b"{}", method="POST",
                                                      headers={"Content-Type": "application/json"}), timeout=10)
    except Exception as e:  # noqa: BLE001  a 429 (rate floor) or 503 is not a failure; the last slot stands
        print(f"checkpoint declare: {e!r}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--stage", choices=["smoke", "ladder", "train", "all"], default="all")
    a = ap.parse_args(argv)

    import torch

    from heads.backbone import load_rwkv7_hf_backbone
    from heads.features import build_head_input
    from heads.model import DecisionModel
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
    from train.data import items_from_records
    from train.guard import SplitGuard
    from train.lora import apply_lora, remove_lora
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    cfg = json.load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell_path = out / "cell.json"
    mirror = Path(a.ckpt_dir) / "cell.json"
    src = cell_path if cell_path.exists() else mirror
    cell = json.loads(src.read_text()) if src.exists() else {"config": cfg}
    if cell.get("config") != cfg:
        raise SystemExit("cell.json was written under a different config; refusing to mix stages")

    def save():  # also mirrored into the checkpoint dir, so a resumed attempt skips finished stages
        for p in (cell_path, Path(a.ckpt_dir) / "cell.json"):
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(cell, indent=1))
            os.replace(tmp, p)

    def peak():
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def reset_peak():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    torch.manual_seed(cfg["seed"])
    tok = Rwkv7Tokenizer.from_pretrained(a.model_dir)
    enc = RwkvSegmentEncoder(tok, set(tok.all_special_ids))
    root = Path(__file__).resolve().parents[1] / "harness" / "manifests"
    guard = SplitGuard.from_manifests([root / "dev-manifest.jsonl.gz", root / "test-manifest.jsonl.gz"])
    with gzip.open(a.records, "rt", encoding="utf-8") as f:
        recs = {r["id"]: to_train_record(r) for r in map(json.loads, f)}
    tr_ids, ev_ids = split_ids(sorted(recs), cfg["n_train_records"], cfg["n_eval_records"], cfg["seed"])
    t_cap = max(cfg["t_ladder"])

    def items(ids, max_t):
        its = items_from_records([recs[i] for i in ids], enc, layout=cfg["layout"], template=cfg["template"],
                                 guard=guard)
        return [x for x in its if keep_item(x.row, cfg["max_k"], max_t)]

    train_items, eval_items = items(tr_ids, t_cap), items(ev_ids, t_cap)
    cell["data"] = {"records_sha256": hashlib.sha256(Path(a.records).read_bytes()).hexdigest(),
                    "n_train_items": len(train_items), "n_eval_items": len(eval_items),
                    "guard_manifests": guard.manifests, "max_k": cfg["max_k"]}
    save()
    if len(train_items) < cfg["min_train_items"]:
        print(f"too few items {len(train_items)}", flush=True)
        return EXIT["data"]

    bb = load_rwkv7_hf_backbone(a.model_dir, fork_mode="cache")
    cuda = torch.cuda.is_available()
    dev_type = "cuda" if cuda else "cpu"
    cell["weights_gb"] = torch.cuda.memory_allocated() / 1e9 if cuda else None
    stages = ["smoke", "ladder", "train"] if a.stage == "all" else [a.stage]

    if "smoke" in stages and "smoke" not in cell:
        sm = cfg["smoke"]
        rows = [x.row for x in train_items if x.row.n_options <= sm["max_k"] and len(x.row.ids) <= sm["max_t"]][: sm["rows"]]
        mods = apply_lora(bb.model, bb.spec.lora_groups, rank=sm["lora_rank"], generator=torch.Generator().manual_seed(5))
        for n, p in bb.model.named_parameters():
            if n.endswith("lora_B"):
                with torch.no_grad():
                    p.normal_(0.0, 0.02)
        res = {}
        for mode in ("cache", "recompute"):
            bb.fork_mode = mode
            bb.model.zero_grad(set_to_none=True)
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cuda):
                x = build_head_input(bb, rows, need_fork=True)
            x.fork.float().pow(2).mean().backward()
            g = torch.cat([p.grad.flatten().float() for n, p in bb.model.named_parameters() if n.endswith("lora_A")])
            res[mode] = (x.fork.detach().float(), g)
        fa, fb = res["cache"][0], res["recompute"][0]
        rel = float((fa - fb).abs().max() / fb.abs().max().clamp_min(1e-12))
        cos = float(torch.nn.functional.cosine_similarity(res["cache"][1], res["recompute"][1], dim=0))
        ok = rel <= sm["max_rel_feature_diff"] and cos >= sm["min_grad_cos"]
        cell["smoke"] = {"rows": len(rows), "rel_feature_diff": rel, "grad_cos": cos, "pass": ok, "peak_gb": peak(),
                         "lora_modules": {k: len(v) for k, v in mods.items()}, "pass_rule": sm}
        save()
        del res
        bb.model.zero_grad(set_to_none=True)
        remove_lora(bb.model)  # training starts from the pre-registered LoRA init, not the smoke one
        bb.fork_mode = "cache"
        reset_peak()
        if not ok:
            return EXIT["fork_mismatch"]

    def make_trainer(max_steps, ckpt_dir=None):
        model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
        model.to(bb.device)
        tc = TrainConfig(head=cfg["head"], head_kwargs=cfg.get("head_kwargs") or {}, layout=cfg["layout"],
                         fork_mode="cache", lora_rank=cfg["lora_rank"], loss=LossConfig(arm=cfg["loss_arm"]),
                         lr_head=cfg["lr_head"], lr_lora=cfg["lr_lora"], batch_size=cfg["batch_size"],
                         grad_accum=cfg["grad_accum"], max_steps=max_steps, bf16=cuda, grad_checkpointing=False,
                         seed=cfg["seed"], ckpt_dir=ckpt_dir, ckpt_every=cfg["ckpt_every"],
                         eval_every=cfg["eval_every"])
        return Trainer(model, tc, train_items, eval_items, guard=guard)

    if "ladder" in stages and "ladder" not in cell:
        lad = {}
        for T in cfg["t_ladder"]:
            its = sorted((x for x in train_items if len(x.row.ids) <= T), key=lambda x: (-x.row.n_options, -len(x.row.ids)))
            if not its:
                continue
            reset_peak()
            tr = None
            try:
                tr = make_trainer(1)
                tr.train_items = its[: cfg["batch_size"] * cfg["grad_accum"]]  # the worst rows (max K, then max T)
                t0 = time.time()
                loss = tr.train_step()
                if cuda:
                    torch.cuda.synchronize()
                lad[str(T)] = {"ok": math.isfinite(loss), "peak_gb": peak(), "step_s": time.time() - t0,
                               "row_len": len(its[0].row.ids), "k": its[0].row.n_options}
            except torch.OutOfMemoryError as e:
                lad[str(T)] = {"ok": False, "oom": str(e)[:200], "peak_gb": peak()}
            finally:
                del tr
                bb.model.zero_grad(set_to_none=True)
                remove_lora(bb.model)
                reset_peak()
        fits = [int(T) for T, r in lad.items() if r.get("ok") and r["peak_gb"] <= cfg["pass_rule"]["max_peak_gb"]]
        cell["ladder"] = {"points": lad, "T_max": max(fits) if fits else None}
        save()
        if not fits:
            return EXIT["no_T_fits"]
        # measured stop condition 1: even the fastest fitting rung's MEASURED step time (worst rows of that rung)
        # cannot reach the pass rule's step count inside max_minutes -> exit 26 before any training spend
        fastest = min(lad[str(T)]["step_s"] for T in fits)
        cell["ladder"]["rate_check"] = too_slow(fastest, cfg["max_minutes"], steps_needed(cfg["pass_rule"]))
        save()
        if cell["ladder"]["rate_check"]["too_slow"]:
            return EXIT["too_slow"]

    if "train" in stages:
        t_max = cell["ladder"]["T_max"]
        train_items[:] = [x for x in train_items if len(x.row.ids) <= t_max]
        eval_items[:] = [x for x in eval_items if len(x.row.ids) <= t_max]
        reset_peak()
        tr = make_trainer(cfg["max_steps"], a.ckpt_dir)
        resumed = tr.resume()
        if not resumed:
            tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(tr.model, eval_items, 0, a.out))
        t_start = time.time()
        t_end = t_start + 60 * cfg["max_minutes"]
        step0 = tr.step
        logf = open(out / "train_log.jsonl", "a")
        while tr.step < cfg["max_steps"] and time.time() < t_end:
            loss = tr.train_step()
            if tr.step - step0 == CHECK_AFTER_STEPS:  # measured stop condition 2: the real data mix's step time
                el = time.time() - t_start
                chk = too_slow(el / CHECK_AFTER_STEPS, (t_end - time.time()) / 60,
                               max(0, steps_needed(cfg["pass_rule"]) - tr.step))
                cell["train_rate_check"] = chk
                save()
                if chk["too_slow"]:
                    tr.save()
                    cell["train"] = {"stopped": "too_slow", "step": tr.step, **chk}
                    save()
                    return EXIT["too_slow"]
            logf.write(json.dumps({"step": tr.step, "loss": loss, "peak_gb": peak(), "t": time.time()}) + "\n")
            logf.flush()
            if not math.isfinite(loss):
                cell["train"] = {"stopped": "nonfinite", "step": tr.step}
                save()
                return EXIT["nonfinite"]
            if peak() > cfg["pass_rule"]["max_peak_gb"]:
                cell["train"] = {"stopped": "memory", "step": tr.step, "peak_gb": peak()}
                save()
                return EXIT["memory"]
            if tr.step % cfg["ckpt_every"] == 0:
                tr.save()
                _checkpoint_post()
            if tr.step % cfg["eval_every"] == 0:
                tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(tr.model, eval_items,
                                                                                           tr.step, a.out))
        if not tr.cards or tr.cards[-1]["step"] != tr.step:
            tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(tr.model, eval_items,
                                                                                       tr.step, a.out))
        tr.save()
        _checkpoint_post()
        cell["train"] = {"steps": tr.step, "resumed": resumed, "T_max": t_max, "cards": tr.cards,
                         "lineage": tr.lineage, "verdict": pass_rule(tr.history, tr.cards, peak(), cfg["pass_rule"])}
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
