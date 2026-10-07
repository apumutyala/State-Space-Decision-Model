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

EXIT = {"fork_mismatch": 21, "no_T_fits": 22, "nonfinite": 23, "memory": 24, "data": 25, "too_slow": 26,
        "guard": 27, "cohort": 28, "lineage": 29, "config": 35}
PARENTS = ("rwkv7", "q9af")


def parent_encoder(parent: str, model_dir: str):
    """The training-time segment encoder of each parent (the same one its decide backend serves with)."""
    if parent == "rwkv7":
        from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

        tok = Rwkv7Tokenizer.from_pretrained(model_dir)
        return RwkvSegmentEncoder(tok, set(tok.all_special_ids))
    if parent == "q9af":
        from transformers import AutoTokenizer

        from heads.layout import hf_segment_encoder

        return hf_segment_encoder(AutoTokenizer.from_pretrained(model_dir))
    raise ValueError(f"parent must be one of {PARENTS}, got {parent!r}")


def load_parent_backbone(parent: str, model_dir: str, cfg: dict, gdn_ckpt: str | None = None, device: str = "cuda"):
    if parent == "rwkv7":
        from heads.backbone import load_rwkv7_hf_backbone

        return load_rwkv7_hf_backbone(model_dir, device=device, fork_mode="cache", wkv=cfg.get("rwkv7_wkv"))
    if parent == "q9af":
        from heads.backbone import load_q9af_backbone

        q = cfg.get("q9af") or {}
        # hybrid (a partial export, attention layers kept) loads ONLY when the config says so; it is then labelled
        # kind "q9af-hybrid", attention_free False in the provenance (heads.backbone.hybrid_label)
        return load_q9af_backbone(model_dir, gdn_ckpt, q, device=device, fork_mode="cache",
                                  dtype="bfloat16" if str(device).startswith("cuda") else "float32",
                                  hybrid=bool(q.get("hybrid")))
    raise ValueError(f"parent must be one of {PARENTS}, got {parent!r}")
CHECK_AFTER_STEPS = 10  # the in-training rate check uses the mean of the first steps actually run


# ----------------------------------------------------------------------------- data (CPU-testable)
def host_provenance(r: dict) -> dict | None:
    """Explicit host provenance for an m2 record (``decide.provenance`` shape), for the controlsafe trust gate.

    A question's instructions are declared instruction-trusted ONLY when the builder put the item in ``state``
    (``aug.placement == "state"``) AND the instruction string is exactly one of the builder's own templates
    (``data.augment.INSTR``). Everything else (the passage under ``placement: instructions``, tool names,
    candidate dicts, option texts, the state) stays data: absent provenance is data (fail closed)."""
    if (r.get("aug") or {}).get("placement") != "state":
        return None
    from data.augment import INSTR
    from decide.provenance import PROVENANCE_SCHEMA, instructions_path

    templates = {t for v in INSTR.values() for t in v} | {"Answer the question."}
    segs = {}
    for qid, q in r["questions"].items():
        ins = q.get("instructions")
        if isinstance(ins, str) and ins in templates:
            segs[instructions_path(qid)] = [{"start": 0, "end": len(ins), "trust": "instruction",
                                             "origin": "host_template"}]
    return {"provenance": {"schema": PROVENANCE_SCHEMA, "default_trust": "data", "segments": segs}} if segs else None


def to_train_record(r: dict) -> dict:
    """m2 sample record (data/build.py: state/questions/labels at top level) -> train.data record format.
    Carries ``host_provenance`` under ``request["ssdm"]`` (read only by the trust gate's provenance policy;
    content hashes, guards and token ids are unchanged)."""
    req = {"model": "m2", "state": r["state"], "questions": r["questions"]}
    prov = host_provenance(r)
    if prov:
        req["ssdm"] = prov
    return {"id": r["id"], "request": req, "labels": r["labels"]}


def split_ids(ids: list[str], n_train: int, n_eval: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic disjoint pilot split by sha256(seed:id) order."""
    order = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return order[n_eval: n_eval + n_train], order[:n_eval]


def cohort_split(record_ids: list[str], cfg: dict, records: dict | None = None) -> tuple[list[str], list[str], dict]:
    """With ``cfg["cohort_sha256"]`` (the m2 job): train ids come ONLY from the PR #16 training pool (every record
    outside the frozen development cohort BY ID AND BY NORMALIZED CONTENT, so content twins of held-out records are
    never trained on) and eval ids are the cohort's ``dev`` role. With ``records`` (id -> record), every pool record's
    content is re-hashed and must equal the committed content map. Without a cohort: the legacy pilot split
    (``split_ids``). Raises ``train.cohort.CohortLeak`` / ``ValueError`` on any overlap, drift or cohort mismatch."""
    if not cfg.get("cohort_sha256"):
        tr, ev = split_ids(record_ids, cfg["n_train_records"], cfg["n_eval_records"], cfg["seed"])
        return tr, ev, {"cohort": None}
    from train.cohort import Cohort, read_content_map, record_content_hash

    co = Cohort(expected_sha256=cfg["cohort_sha256"])
    held, held_c = co.held_out(), co.held_out_content()
    content = read_content_map()
    unknown = [i for i in record_ids if i not in content]
    if unknown:
        raise ValueError(f"{len(unknown)} record(s) are not in the committed content map, e.g. {unknown[:3]}")
    pool = [i for i in record_ids if i not in held and content[i] not in held_c]
    tr, _ = split_ids(pool, cfg["n_train_records"], 0, cfg["seed"])
    if records is not None:
        drift = [i for i in tr if record_content_hash(records[i]) != content[i]]
        if drift:
            raise ValueError(f"{len(drift)} training record(s) differ in content from the content map, e.g. {drift[:3]}")
        co.assert_training_content(record_content_hash(records[i]) for i in tr)
    ev = co.ids("dev")
    missing = sorted(set(ev) - set(record_ids))
    if missing:
        raise ValueError(f"cohort dev records missing from the records file: {missing[:3]}")
    co.assert_training_ids(tr)
    return tr, ev, {"cohort": co.identity()}


def cell_items(recs: dict, ids, cfg: dict, enc, guard, max_t: int, workers: int | None = None) -> list:
    """Guarded rows of ``ids`` kept by the cell's K / T filter (``keep_item``). ``train.m2_gates`` rebuilds the
    training items with this exact function, so a trained checkpoint's lineage (data hash) can be re-verified.
    ``workers`` (default ``cfg["tokenize_workers"]``, else 1) tokenizes contiguous chunks in worker processes; the
    output is the same list in the same order as the serial path (``tests/test_m2_first_cell.py``)."""
    ids = list(ids)
    workers = int(workers if workers is not None else (cfg.get("tokenize_workers") or 1))
    workers = max(1, min(workers, os.cpu_count() or 1))
    if workers == 1 or len(ids) < 4 * workers:
        return _items_chunk([recs[i] for i in ids], cfg, enc, guard, max_t)
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    size = math.ceil(len(ids) / workers)
    chunks = [[recs[i] for i in ids[k: k + size]] for k in range(0, len(ids), size)]
    # spawn on every platform: never fork a process that may hold CUDA / torch threads (same path the CPU tests run)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        parts = list(ex.map(_items_chunk, chunks, [cfg] * len(chunks), [enc] * len(chunks), [guard] * len(chunks),
                            [max_t] * len(chunks)))
    return [x for p in parts for x in p]


def _items_chunk(records: list, cfg: dict, enc, guard, max_t: int) -> list:
    from train.data import items_from_records

    its = items_from_records(records, enc, layout=cfg["layout"], template=cfg["template"], guard=guard)
    return [x for x in its if keep_item(x.row, cfg["max_k"], max_t)]


def load_cell_data(cfg: dict, records_path: str, train_dir: str | None = None, cohort=None):
    """(train records by id, eval records by id, train ids, eval ids, data identity) for a cell config.

    Without ``cfg["train_set"]``: the PR #16 sample at ``records_path`` provides both (``cohort_split``). With it: train
    records come from the pointer-pinned external set in ``train_dir`` (``train.train_set.draw``: cohort excluded by id
    and content, manifest-verified, content re-hashed) and eval records (the cohort ``dev`` role) from ``records_path``.
    Raises ``ValueError`` (``CohortLeak`` / ``TrainSetMismatch`` subclass it) on any identity problem."""
    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        recs = {r["id"]: to_train_record(r) for r in map(json.loads, f)}
    if not cfg.get("train_set"):
        tr_ids, ev_ids, ident = cohort_split(sorted(recs), cfg, recs)
        return recs, recs, tr_ids, ev_ids, ident
    if not train_dir:
        raise ValueError("the cell config names a train_set: pass --train-dir with its pinned files")
    from train.cohort import Cohort
    from train.train_set import draw

    co = cohort if cohort is not None else Cohort(expected_sha256=cfg["cohort_sha256"])
    train_recs, tr_ids, tident = draw(cfg, train_dir, co)
    ev_ids = co.ids("dev")
    missing = sorted(set(ev_ids) - set(recs))
    if missing:
        raise ValueError(f"cohort dev records missing from the records file: {missing[:3]}")
    if set(tr_ids) & set(ev_ids):
        raise ValueError("training ids overlap the cohort dev role")
    return train_recs, recs, tr_ids, ev_ids, {"cohort": co.identity(), "train_set": tident}


def kd_fields(cfg: dict) -> dict:
    """``LossConfig`` KD fields from the cell's optional ``teacher`` block ({alpha, temperature, ...}). No block = {}:
    the loss config, and so the TrainConfig hash of every non-KD cell, is unchanged (``LossConfig.identity``)."""
    t = cfg.get("teacher")
    if not t:
        return {}
    return {"kd_alpha": float(t["alpha"]), "kd_temperature": float(t["temperature"])}


def attach_teacher(cfg: dict, items: list, teacher_dir, train_ids, expected: dict | None = None,
                   contamination: bool = True) -> dict | None:
    """KD cells (``cfg["teacher"]``): load the PINNED teacher (``train.teacher.load_teacher``: identity, file sha256,
    draw, contamination), require the cell's training draw to BE the teacher's draw and its file sha256 to be the one
    the config names, and attach per-item teacher probabilities (``train.teacher.attach``). None for a non-KD cell.
    Raises ValueError (``TeacherIdentityError`` subclasses it) on any mismatch; ``expected`` / ``contamination`` are for
    tests on synthetic teachers only."""
    t = cfg.get("teacher")
    if not t:
        if teacher_dir:
            raise ValueError("--teacher-dir given but the cell config has no teacher block")
        return None
    if not teacher_dir:
        raise ValueError("the cell config sets 'teacher' (KD): pass --teacher-dir with the pinned teacher export")
    from train.teacher import attach, ids_sha256, load_teacher, pinned

    exp = dict(expected) if expected is not None else pinned()
    if t.get("teacher_sha256") != exp["sha256"]:
        raise ValueError(f"teacher.teacher_sha256 {str(t.get('teacher_sha256'))[:12]} != the pinned teacher "
                         f"{exp['sha256'][:12]}")
    if ids_sha256(list(train_ids)) != exp["drawn_ids_sha256"]:
        raise ValueError("the cell's training draw is not the teacher's draw (drawn ids sha256 differ)")
    ts = load_teacher(teacher_dir, exp, contamination=contamination)
    cov = attach(items, ts)
    if not cov["with_teacher"]:
        raise ValueError("no training item received a teacher distribution")
    return {**cov, "teacher_sha256": exp["sha256"], "drawn_ids_sha256": exp["drawn_ids_sha256"],
            "alpha": float(t["alpha"]), "temperature": float(t["temperature"])}


def train_config(cfg: dict, max_steps: int, ckpt_dir=None, bf16: bool = False):
    """The cell's ``TrainConfig`` (its hash is checked when a checkpoint is loaded)."""
    from train.losses import LossConfig
    from train.trainer import TrainConfig

    return TrainConfig(head=cfg["head"], head_kwargs=cfg.get("head_kwargs") or {}, layout=cfg["layout"],
                       fork_mode="cache", lora_rank=cfg["lora_rank"],
                       loss=LossConfig(arm=cfg["loss_arm"], **kd_fields(cfg)),
                       lr_head=cfg["lr_head"], lr_lora=cfg["lr_lora"], batch_size=cfg["batch_size"],
                       warmup_steps=int(cfg.get("warmup_steps") or 0),
                       grad_accum=cfg["grad_accum"], max_steps=max_steps, bf16=bf16, grad_checkpointing=False,
                       seed=cfg["seed"], ckpt_dir=ckpt_dir, ckpt_every=cfg["ckpt_every"],
                       eval_every=cfg["eval_every"], controlsafe=cfg.get("controlsafe") or {})


def validate_head(cfg: dict) -> None:
    """Head-specific config checks (exit 35). G16 needs the controlsafe trust gate with the D stream readout."""
    if cfg.get("head") != "g16":
        return
    cs = cfg.get("controlsafe") or {}
    if not cs.get("trust_gate"):
        raise ValueError("head 'g16' needs controlsafe.trust_gate (G16 reads the G15 protected stream)")
    if cs.get("readout", "decision") != "decision":
        raise ValueError("head 'g16' needs controlsafe.readout 'decision' (the evidence channel reads the D stream)")
    c = (cfg.get("head_kwargs") or {}).get("c", 1.0)
    if not (isinstance(c, (int, float)) and c > 0):
        raise ValueError(f"head_kwargs.c (evidence bound) must be > 0, got {c!r}")


def keep_item(row, max_k: int, max_t: int) -> bool:
    return row.n_options <= max_k and len(row.ids) <= max_t and max(len(c) for c in row.continuations) <= 16


class RwkvSegmentEncoder:
    """Trusted text verbatim; untrusted text fails closed if it yields a special id (World tokenizer: only id 0)."""

    def __init__(self, tok, special: set[int]):
        self.tok, self.special = tok, set(special) | {0}
        self._ident = None

    def identity(self) -> dict:
        # memoised: hashing the 65,536-entry vocabulary took ~32 ms and ran once per ROW (row.meta["encoder"]), about
        # half of the training-set tokenization time; the vocabulary is immutable, so the value is unchanged
        if self._ident is None:
            vocab = hashlib.sha256(json.dumps(sorted(self.tok.get_vocab().items())).encode()).hexdigest()
            self._ident = {"kind": "rwkv7-hf-trie", "vocab_sha256": vocab, "untrusted": "fail-closed on special ids"}
        return dict(self._ident)

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        from decide.untrusted import ControlTokenError

        ids = self.tok(text, add_special_tokens=False).input_ids
        if not trusted and self.special.intersection(ids):
            raise ControlTokenError("untrusted text encodes to a special token")
        return ids


def train_seconds_used(cell: dict, resumed: bool) -> float:
    """Training wall-clock already spent by the checkpointed steps (``cell["train_clock"]``, written with every
    checkpoint and mirrored in the checkpoint dir). A fresh start uses 0; a resume subtracts it from ``max_minutes``, so
    restarts can never train past the pre-registered budget. Time after the last checkpoint is not counted: those steps
    are lost and re-run."""
    if not resumed:
        return 0.0
    return float((cell.get("train_clock") or {}).get("used_s") or 0.0)


def steps_needed(rule: dict, min_steps: int | None = None) -> int:
    """The pass rule compares two disjoint windows of steps, so fewer than 2 x window steps can never pass. A cell may
    pre-register a larger ``min_steps`` (the training it needs to be a meaningful measurement); the exit-26 checks then
    stop before spending on a run that cannot reach it."""
    return max(2 * rule.get("window", 50), int(min_steps or 0))


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


def smoke_stage(bb, train_items, cfg: dict, cuda: bool, peak=lambda: 0.0) -> dict:
    """Cache-fork smoke: features and LoRA gradients of the cache fork equal the recompute fork on a few short rows
    (``cfg["smoke"]``). Leaves the backbone as it found it (no LoRA, bridges zeroed, ``fork_mode`` "cache").

    Runs under the cell's per-block activation checkpointing (``cfg["act_ckpt"]``), exactly as training does (job-uhxb5,
    LESSONS L-33): the recompute fork pushes every (row, option) continuation through the backbone in one batch, and
    without checkpointing its activations do not fit beside 13.3B bf16 weights on 80 GB (CUDA OOM at 77 GiB). The
    checkpointed gradients equal the plain ones (tests/test_cell_perf.py), so the comparison is unchanged; it now also
    covers the checkpointed cache fork that training actually runs."""
    import torch

    from heads import act_ckpt
    from heads.features import build_head_input
    from train.lora import apply_lora, remove_lora

    use_ckpt = bool(cfg.get("act_ckpt"))
    was_ckpt = act_ckpt.is_enabled(bb.model)
    if use_ckpt:
        act_ckpt.enable_block_checkpointing(bb.model)  # raises for an unsupported backbone, as the Trainer would
    dev_type = "cuda" if cuda else "cpu"
    sm = cfg["smoke"]
    rows = [x.row for x in train_items if x.row.n_options <= sm["max_k"] and len(x.row.ids) <= sm["max_t"]][: sm["rows"]]
    mods = apply_lora(bb.model, bb.spec.lora_groups, rank=sm["lora_rank"], generator=torch.Generator().manual_seed(5))
    for n, p in bb.model.named_parameters():
        if n.endswith("lora_B"):
            with torch.no_grad():
                p.normal_(0.0, 0.02)
    # trust-gated backbone: the P -> D bridges live outside bb.model; randomise them too so the smoke exercises
    # the bridge injection on the cache fork (prefix / continuation alignment), and re-zero them afterwards
    bridges_b = [p for n, p in getattr(bb, "bridges", torch.nn.Module()).named_parameters() if n.endswith("lora_B")]
    with torch.no_grad():
        for p in bridges_b:
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
        del x, g  # the cache pass's tensors are not kept alive through the recompute pass
    fa, fb = res["cache"][0], res["recompute"][0]
    rel = float((fa - fb).abs().max() / fb.abs().max().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(res["cache"][1], res["recompute"][1], dim=0))
    ok = rel <= sm["max_rel_feature_diff"] and cos >= sm["min_grad_cos"]
    out = {"rows": len(rows), "rel_feature_diff": rel, "grad_cos": cos, "pass": ok, "peak_gb": peak(),
           "lora_modules": {k: len(v) for k, v in mods.items()}, "pass_rule": sm, "act_ckpt": use_ckpt}
    del res
    bb.model.zero_grad(set_to_none=True)
    remove_lora(bb.model)  # training starts from the pre-registered LoRA init, not the smoke one
    with torch.no_grad():
        for p in bridges_b:  # bridges back to their zero init (remove_lora does not touch them)
            p.zero_()
        for p in getattr(bb, "bridges", torch.nn.Module()).parameters():
            p.grad = None
    bb.fork_mode = "cache"
    if use_ckpt and not was_ckpt:  # the Trainer applies (or removes) its own execution flags afterwards
        act_ckpt.disable_block_checkpointing(bb.model)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gdn-ckpt", default=None, help="parent q9af: the converted all-GDN checkpoint from convert/ "
                    "(its sha256 must equal the cell config's q9af.converted_sha256)")
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--stage", choices=["smoke", "ladder", "train", "all"], default="all")
    ap.add_argument("--guard-readiness", default=None,
                    help="readiness.json from jobs.m2_pilot.guard_ready (required when the config sets "
                         "require_guard_readiness): the guard then includes the verified content-hash index")
    ap.add_argument("--train-dir", default=None, help="config train_set: dir with the pointer-pinned training-set files "
                    "(train.train_set); --records then supplies only the cohort (eval) records")
    ap.add_argument("--device", default="cuda", help="cuda (paid runs); cpu only for the tiny-model dry run")
    ap.add_argument("--teacher-dir", default=None, help="KD cells (config 'teacher'): the pinned teacher export "
                    "(train.teacher.load_teacher); refused for a cell without a teacher block")
    a = ap.parse_args(argv)

    import torch

    from heads.backbone import LineageError
    from heads.model import DecisionModel
    from train.guard import SplitGuard
    from train.lora import remove_lora
    from train.trainer import Trainer

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

    # CUDA paths (bf16 autocast, peak memory) follow --device, not merely an installed CUDA build: a CPU run on a
    # CUDA-capable host (CUDA_VISIBLE_DEVICES="" can still report is_available) must not autocast to a cuda device
    on_cuda = torch.cuda.is_available() and str(a.device).startswith("cuda")

    def peak():
        return torch.cuda.max_memory_allocated() / 1e9 if on_cuda else 0.0

    def reset_peak():
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    torch.manual_seed(cfg["seed"])
    parent = cfg.get("parent", "rwkv7")
    try:
        validate_head(cfg)
        if parent == "q9af" or cfg.get("teacher") or cfg.get("head_from_proxy"):
            from jobs.q9af_cell import hooks  # typed KD-teacher / proxy-head hooks (fail closed until wired)

            cfg_eff = hooks.apply(cfg)
            if cfg_eff != cfg:
                raise ValueError("hooks changed the config: resolve head_from_proxy into the committed config first")
    except ValueError as e:
        print(f"config: {e}", flush=True)
        return EXIT["config"]
    if parent == "q9af":  # pins are checked before tokenizer or weights are touched
        try:
            from heads.backbone import q9af_pin_check

            if not a.gdn_ckpt:
                raise LineageError("parent q9af needs --gdn-ckpt")
            q9af_pin_check(cfg.get("q9af") or {}, a.gdn_ckpt, a.model_dir)
        except (LineageError, OSError) as e:
            print(f"q9af lineage: {e}", flush=True)
            return EXIT["lineage"]
    enc = parent_encoder(parent, a.model_dir)
    from train.g16_card import make_hook as g16_hook

    # G16 cells: certify on the safety_eval DEV requests ONCE, after the training loop and the final card (outside
    # max_minutes), so the G16 arm and its non-G16 pair train under the same budget (step parity); wall-clock logged
    g16 = g16_hook(cfg, enc)
    root = Path(__file__).resolve().parents[1] / "harness" / "manifests"
    if a.guard_readiness:
        from jobs.m2_pilot.guard_ready import GuardNotReady, load_ready_guard

        try:
            guard = load_ready_guard(a.guard_readiness)
        except GuardNotReady as e:
            print(f"guard not ready: {e}", flush=True)
            return EXIT["guard"]
    elif cfg.get("require_guard_readiness"):
        print("config requires --guard-readiness (verified content-hash index); refusing to train", flush=True)
        return EXIT["guard"]
    else:
        guard = SplitGuard.from_manifests([root / "dev-manifest.jsonl.gz", root / "test-manifest.jsonl.gz"])
    try:
        train_recs, eval_recs, tr_ids, ev_ids, cohort_ident = load_cell_data(cfg, a.records, a.train_dir)
    except ValueError as e:  # CohortLeak / TrainSetMismatch subclass ValueError
        print(f"cohort: {e}", flush=True)
        return EXIT["cohort"]
    t_cap = max(cfg["t_ladder"])
    t_tok = time.time()
    train_items = cell_items(train_recs, tr_ids, cfg, enc, guard, t_cap)
    eval_items = cell_items(eval_recs, ev_ids, cfg, enc, guard, t_cap)
    try:
        kd_cov = attach_teacher(cfg, train_items, a.teacher_dir, tr_ids)
    except ValueError as e:  # TeacherIdentityError / TeacherContamination subclass ValueError
        print(f"teacher: {e}", flush=True)
        return EXIT["config"]
    cell["data"] = {"records_sha256": hashlib.sha256(Path(a.records).read_bytes()).hexdigest(),
                    "n_train_records": len(tr_ids), "tokenize_s": round(time.time() - t_tok, 1),
                    "n_train_items": len(train_items), "n_eval_items": len(eval_items),
                    "guard_manifests": guard.manifests, "guard": guard.identity(), "max_k": cfg["max_k"],
                    **cohort_ident, **({"teacher": kd_cov} if kd_cov else {})}
    save()
    if len(train_items) < cfg["min_train_items"]:
        print(f"too few items {len(train_items)}", flush=True)
        return EXIT["data"]

    bb = load_parent_backbone(parent, a.model_dir, cfg, a.gdn_ckpt, device=a.device)
    if (cfg.get("controlsafe") or {}).get("trust_gate"):  # optional G15 arm (controlsafe/CONTROLSAFE.md)
        from controlsafe.trust_gate import wrap as trust_gate_wrap

        bb = trust_gate_wrap(bb, {**cfg["controlsafe"], "seed": cfg["seed"]})
    cuda = on_cuda
    cell["weights_gb"] = torch.cuda.memory_allocated() / 1e9 if cuda else None
    stages = ["smoke", "ladder", "train"] if a.stage == "all" else [a.stage]

    if "smoke" in stages and "smoke" not in cell:
        cell["smoke"] = smoke_stage(bb, train_items, cfg, cuda, peak)
        save()
        reset_peak()
        if not cell["smoke"]["pass"]:
            return EXIT["fork_mismatch"]

    def make_trainer(max_steps, ckpt_dir=None):
        model = DecisionModel(bb, cfg["head"], cfg.get("head_kwargs"), pad_id=0, seed=cfg["seed"])
        model.to(bb.device)
        from train.trainer import with_execution  # act_ckpt / row_batch / row_tokens of the cell (PR #86 hook)

        return Trainer(model, with_execution(train_config(cfg, max_steps, ckpt_dir, cuda), cfg), train_items,
                       eval_items, guard=guard)

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
        cell["ladder"]["rate_check"] = too_slow(fastest, cfg["max_minutes"], steps_needed(cfg["pass_rule"], cfg.get("min_steps")))
        save()
        if cell["ladder"]["rate_check"]["too_slow"]:
            return EXIT["too_slow"]

    if "train" in stages:
        t_max = cell["ladder"]["T_max"]
        train_items[:] = [x for x in train_items if len(x.row.ids) <= t_max]
        eval_items[:] = [x for x in eval_items if len(x.row.ids) <= t_max]
        reset_peak()
        tr = make_trainer(cfg["max_steps"], a.ckpt_dir)
        keep = bool(cfg.get("pair"))  # paired cells write units-step*.json for train.g16_card.capability_cost
        resumed = tr.resume()
        if not resumed:
            tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(tr.model, eval_items, 0, a.out,
                                                                                       keep_units=keep))
            tr.cards[-1].pop("units", None)
        t_start = time.time()
        # the max_minutes budget is cumulative over attempts: a restart resumes with what the checkpointed steps used
        used_before = train_seconds_used(cell, resumed)
        t_end = t_start + 60 * cfg["max_minutes"] - used_before
        cell["train_clock_resume"] = {"resumed": resumed, "step": tr.step, "used_s_before": used_before,
                                      "left_s": max(0.0, t_end - t_start)}
        save()
        step0 = tr.step
        logf = open(out / "train_log.jsonl", "a")
        while tr.step < cfg["max_steps"] and time.time() < t_end:
            loss = tr.train_step()
            if tr.step - step0 == CHECK_AFTER_STEPS:  # measured stop condition 2: the real data mix's step time
                el = time.time() - t_start
                chk = too_slow(el / CHECK_AFTER_STEPS, (t_end - time.time()) / 60,
                               max(0, steps_needed(cfg["pass_rule"], cfg.get("min_steps")) - tr.step))
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
                cell["train_clock"] = {"step": tr.step, "used_s": used_before + time.time() - t_start}
                save()
                _checkpoint_post()
            if tr.step % cfg["eval_every"] == 0:
                tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(tr.model, eval_items,
                                                                                           tr.step, a.out,
                                                                                           keep_units=keep))
                tr.cards[-1].pop("units", None)
        if not tr.cards or tr.cards[-1]["step"] != tr.step:
            # paired cells (cfg["pair"]) keep per-item units for train.g16_card.capability_cost
            tr.cards.append(__import__("train.evalcard", fromlist=["run_card"]).run_card(
                tr.model, eval_items, tr.step, a.out, keep_units=keep))
            tr.cards[-1].pop("units", None)
        tr.save()
        cell["train_clock"] = {"step": tr.step, "used_s": used_before + time.time() - t_start}
        _checkpoint_post()
        cell["train"] = {"steps": tr.step, "resumed": resumed, "T_max": t_max, "cards": tr.cards,
                         "lineage": tr.lineage, "verdict": pass_rule(tr.history, tr.cards, peak(), cfg["pass_rule"])}
        save()
        if g16 is not None:  # after the checkpoint and verdict are saved: a timeout here loses only this record
            t_g = time.time()
            cell["train"]["g16_safety_eval"] = {"step": tr.step, **g16(tr.model), "seconds": round(time.time() - t_g, 1)}
            save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
