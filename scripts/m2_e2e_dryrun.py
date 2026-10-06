"""m2 end-to-end dry run on CPU with tiny random backbones (pipeline qualification, NOT a result).

    python -X utf8 -m scripts.m2_e2e_dryrun --out <dir> [--profile smoke|full]

Proves the merged m2 pipeline runs end to end before any paid run:

1. data      a small slice of the PR #16 training records (``train.data.load_m2_records``; every line
             verified against the committed manifest), carved into train / calibration / eval by id hash.
             The ``SplitGuard`` is built from the m0 harness DEV + TEST manifests and must refuse forged
             Index rows (``_evaluation`` metadata, a DEV run_id, a TEST group_id, Index ``source``).
             Without the (outside-git) record file, synthetic records of the same ``ssdm-train-v0``
             shape are used and the receipt says so.
2. train     integrator, cache fork, snapshot and Head B x {ce, clef} loss arms with LoRA on two tiny
             random backbones: the functional tiny recurrent LM and a tiny all-GDN Qwen3.5 (Q9-AF
             layout). Each run checkpoints at step k, is resumed by a FRESH model + trainer (lineage and
             config checks), and continues to step 2k. Negative checks: a resume with different training
             data or a different config must be refused; a resumed run must equal an uninterrupted one.
3. calibrate ``FeatureTemperature`` fit on the calibration split (frozen logits); a card refuses a
             temperature fit on its own eval items.
4. eval card per checkpoint (parent = step k, child = step 2k): decision metrics raw/tempered;
             harness Index-format scoring of SYNTHETIC DEV-shaped rows (never real DEV rows) through the
             decide adapter; safety_eval clean + D1-A steer/hijack on synthetic dev-split requests via
             ``safety_eval.decide_bridge.DecideBridge`` over ``heads.decide_backend``; G4 exact certificate
             and G5 LTT selective bound on the tiny model's (stand-in) outputs.
5. promote   ``train.promotion.verdict`` (SAFETY_SPEC G12 + G13 noninferiority vs the parent).
6. receipt   ``<out>/receipt.json``.

Tiny random models make every metric meaningless as a measurement; the receipt only proves the code
paths connect, run, and fail closed. Hardware: CPU only. CUDA_VISIBLE_DEVICES is blanked before torch loads,
the run refuses if CUDA is still visible, and nothing is ever moved to a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, before torch is imported

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RECORDS = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data")) / \
    "m2" / "build" / "sample-20000-seed20261006" / "records.jsonl.gz"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "sample-20000-seed20261006.manifest.jsonl.gz"
INDEX_MANIFESTS = [ROOT / "harness" / "manifests" / "dev-manifest.jsonl.gz",
                   ROOT / "harness" / "manifests" / "test-manifest.jsonl.gz"]

HEADS = ("integrator", "fork", "snapshot", "head_b")
HEAD_KW = {"integrator": {"channels": 8, "t_chunk": 64}, "fork": {"hidden": 16},
           "snapshot": {"channels": 8, "max_snapshots": 4}, "head_b": {"d_state": 8, "hidden": 16, "t_chunk": 64}}
ARMS = ("ce", "clef")
BACKBONES = ("tiny", "q9af")

PROFILES = {
    # Every profile trains + checkpoints + resumes every (backbone, head, arm) and gives every checkpoint a
    # decision card with temperature. ``full_card`` = the (backbone, head, arm) runs that also get the full card
    # (harness, steer, G4/G5) and the promotion verdict; None = all runs.
    # ci: the pytest profile (< 3 min): one full-card run per backbone, one per loss arm.
    "ci": {"n_records": 36, "steps": 2, "batch": 2,
           "full_card": {("tiny", "integrator", "ce"), ("q9af", "fork", "clef")},
           "n_route": 2, "n_pay": 1, "n_g4": 2, "n_boot": 200, "index_groups": 3},
    "smoke": {"n_records": 36, "steps": 2, "batch": 2,
              "full_card": {(b, h, a) for b, h in (("tiny", "integrator"), ("q9af", "fork")) for a in ARMS},
              "n_route": 2, "n_pay": 1, "n_g4": 3, "n_boot": 200, "index_groups": 3},
    "full": {"n_records": 60, "steps": 3, "batch": 2, "full_card": None, "n_route": 3, "n_pay": 2, "n_g4": 6,
             "n_boot": 500, "index_groups": 4},
}


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_identity() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return {"commit": sha, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"commit": "unknown", "error": str(exc)}


def ensure_kit() -> dict:
    """The harness needs the official kit (``decision_index``); add the pinned local clone if needed."""
    try:
        import decision_index  # noqa: F401
    except ImportError:
        from harness import config as hc

        if not (hc.KIT_CLONE / "decision_index").is_dir():
            raise
        sys.path.insert(0, str(hc.KIT_CLONE))
        import decision_index  # noqa: F401
    from harness import config as hc

    return {"kit_path": str(Path(sys.modules["decision_index"].__file__).parent), "pinned_commit": hc.KIT_COMMIT}


# ---------------------------------------------------------------------------------------------- data
def _rec_size(rec: dict) -> int:
    return len(json.dumps({"s": rec["state"], "q": rec["questions"]}, ensure_ascii=False))


def _short(rec: dict, max_chars: int = 700, max_k: int = 6) -> bool:
    if _rec_size(rec) > max_chars:
        return False
    for q in rec["questions"].values():
        k = 2 if q["type"] == "noul" else len(q.get("criteria") or [])
        if k > max_k:
            return False
    return True


def synthetic_m2_records(n: int, seed: int = 0) -> list[dict]:
    """PR #16-shaped (``ssdm-train-v0``: top-level state/questions) synthetic records. Fallback only."""
    from train.data import toy_records

    out = []
    for r in toy_records(n, seed=seed):
        out.append({"id": f"m2synth:{r['id']}", "schema": "ssdm-train-v0", "source": "synthetic_e2e",
                    "family": "toy", "source_split": "train", "state": {"note": r["request"]["state"]},
                    "questions": r["request"]["questions"], "labels": r["labels"], "untrusted_fields": []})
    return out


def load_slice(records: Path, manifest: Path, n: int) -> tuple[list[dict], dict]:
    from train.data import load_m2_records

    if not records.exists():
        recs = synthetic_m2_records(n)
        return recs, {"kind": "synthetic-fallback (PR #16 record file not present)", "n_selected": len(recs),
                      "selected_ids_sha256": _sha([r["id"] for r in recs])}
    per_source: dict[str, int] = {}
    cap = max(2, n // 8)

    def keep(rec):  # short rows only (CPU), round-robin-ish diversity across sources
        if not _short(rec) or per_source.get(rec["source"], 0) >= cap:
            return False
        per_source[rec["source"]] = per_source.get(rec["source"], 0) + 1
        return True

    recs, ident = load_m2_records(records, manifest, keep=keep, limit=n)
    ident.update(kind="PR #16 sample (manifest-verified)", per_source=dict(sorted(per_source.items())),
                 manifest=str(manifest.relative_to(ROOT)),
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    return recs, ident


def split_of(rid: str) -> str:
    b = int(hashlib.sha256(f"m2-e2e|{rid}".encode()).hexdigest(), 16) % 10
    return "train" if b < 6 else "calib" if b < 8 else "eval"


def guard_probes(guard) -> dict:
    """Forged Index rows the guard must refuse (ids only; no DEV/TEST row content is read)."""
    from train.guard import ForbiddenRow

    base = {"state": {"x": "probe"}, "questions": {"q": {"type": "noul", "instructions": "probe?"}},
            "labels": {"q": "true"}, "schema": "ssdm-train-v0", "source": "probe"}
    dev_id = next(i for i in sorted(guard.ids) if ":" in i)
    test_group = next(i for i in sorted(guard.ids) if i != dev_id)
    probes = {
        "evaluation_metadata": {**base, "id": "probe-1", "_evaluation": {"catalog_id": 32}},
        "index_source": {**base, "id": "probe-2", "source": "decision-index"},
        "manifest_run_id": {**base, "id": "probe-3", "run_id": dev_id},
        "manifest_group_id": {**base, "id": "probe-4", "group_id": test_group},
        "manifest_id_as_record_id": {**base, "id": dev_id},
    }
    out = {}
    for name, rec in probes.items():
        try:
            guard.check(rec)
            out[name] = "ADMITTED"
        except ForbiddenRow as exc:
            out[name] = f"refused: {str(exc)[:80]}"
    return out


# ---------------------------------------------------------------------------------------------- eval inputs
def synthetic_dev_rows(n_groups: int) -> list[dict]:
    """Harness/kit-format rows shaped like Index DEV rows (MuSR field-accuracy, catalog 32; ACOS per-review
    F1, catalog 38). Entirely synthetic: ids are prefixed ``SYNTH`` and checked against the manifests."""
    from decision_index.suite.io import dumps

    rows = []
    colors = ["red", "green", "blue", "amber"]
    for g in range(n_groups):
        for i in range(2):
            gold = colors[(g + i) % 4]
            q = {"type": "choice", "instructions": "Which colour is named in the note?",
                 "criteria": {c: f"the colour {c}" for c in colors}}
            st = {"note": f"case {g}.{i}: the parcel is {gold}"}
            rows.append(_kit_row(32, f"SYNTH-32-{g}", f"32:SYNTH:{g}:{i}", st, {"answer": q}, {"answer": gold}, dumps))
        for i in range(1 + g % 3):  # unequal groups / label mixes, so cluster resamples differ
            gold = "yes" if (3 * g + i) % 4 in (0, 3) else "no"
            q = {"type": "choice", "instructions": "Does the review mention price?",
                 "criteria": {"yes": "price is mentioned", "no": "price is not mentioned"}}
            st = {"review": f"review {g}.{i}: " + ("too expensive" if gold == "yes" else "lovely colour")}
            rows.append(_kit_row(38, f"SYNTH-38-{g}", f"38:SYNTH:{g}:{i}", st, {"q": q}, {"q": gold}, dumps))
    return rows


def _kit_row(n, gid, rid, state, questions, expected, dumps):
    payload = {"state": state, "questions": questions}
    psha = hashlib.sha256(dumps(payload).encode()).hexdigest()
    return {"_evaluation": {"run_id": rid, "catalog_id": n, "dataset": "SYNTH", "track": "synthetic", "group_id": gid,
                            "payload_sha256": psha, "proxy_tokens": len(dumps(payload)) // 4},
            "state": state, "questions": questions, "expected": expected, "scoring": {}, "metadata": {"synthetic": True}}


def index_predictor(decider):
    from decision_index.engines.base import validate

    def predict(rows):
        res = {}
        for r in rows:
            e = r["_evaluation"]
            t = time.perf_counter()
            resp = decider.decide({"model": "ssdm-e2e", "state": r["state"], "questions": r["questions"]})
            status = (resp.get("ssdm") or {}).get("status", "ok") if "answers" in resp else "error"
            out = {**e, "status": status, "total_wall_ms": (time.perf_counter() - t) * 1000}
            if status == "ok":
                response = {"answers": resp["answers"]}
                validate(r["questions"], response)
                out["response"] = response
            else:
                out["error"] = json.dumps((resp.get("ssdm") or {}).get("error"))
            res[e["run_id"]] = out
        return res

    return predict


# ---------------------------------------------------------------------------------------------- models
def make_backbone(kind: str, seed: int):
    from heads.backbone import tiny_backbone, tiny_hf_backbone

    if kind == "tiny":
        return tiny_backbone(seed=seed, hidden=16, layers=2, fork_mode="cache")
    if kind == "q9af":
        return tiny_hf_backbone("q9af", seed=seed, hidden=32, fork_mode="cache")
    raise ValueError(kind)


def make_model(kind: str, head: str, seed: int):
    from heads import DecisionModel

    return DecisionModel(make_backbone(kind, seed), head, head_kwargs=HEAD_KW[head], seed=seed)


def make_cfg(head: str, arm: str, ckpt_dir: Path, steps: int, batch: int, **over):
    from train.losses import LossConfig
    from train.trainer import TrainConfig

    kw = dict(head=head, head_kwargs=HEAD_KW[head], fork_mode="cache", lora_rank=4, loss=LossConfig(arm=arm),
              lr_head=1e-2, lr_lora=1e-3, batch_size=batch, max_steps=2 * steps, ckpt_dir=str(ckpt_dir),
              ckpt_every=steps, seed=0)
    kw.update(over)
    return TrainConfig(**kw)


def trainer_at(kind, head, cfg, items, guard, ckpt: Path | None):
    """Fresh model + trainer, optionally restored from ``ckpt`` (lineage/config checked)."""
    from train.trainer import Trainer

    tr = Trainer(make_model(kind, head, cfg.seed), cfg, items, guard=guard)
    if ckpt is not None:
        tr.load(ckpt)
    return tr


def max_abs_diff(a: dict, b: dict) -> float:
    import torch

    if set(a) != set(b):
        return float("inf")
    return max(float((a[k] - b[k]).abs().max()) for k in a) if a else 0.0


# ---------------------------------------------------------------------------------------------- main
def run(out: Path, profile: str = "smoke", records: Path = DEFAULT_RECORDS, manifest: Path = DEFAULT_MANIFEST,
        backbones=BACKBONES, heads=HEADS, arms=ARMS) -> dict:
    t_start = time.perf_counter()
    import numpy as np
    import torch

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; this dry run is CPU-only by design")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    P = PROFILES[profile]
    out.mkdir(parents=True, exist_ok=True)
    kit = ensure_kit()

    from decide import Decider
    from harness import bootstrap as HB
    from harness import score as HS
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from safety_eval import attacks, certify, selective
    from safety_eval import data as sdata
    from safety_eval.decide_bridge import DecideBridge
    from train import promotion
    from train.calibration import FeatureTemperature, features
    from train.data import TOY_TEMPLATE, items_from_records, split_hash
    from train.evalcard import predict, run_card
    from train.guard import SplitGuard

    receipt: dict[str, Any] = {
        "kind": "m2 end-to-end CPU dry run (tiny random models; pipeline qualification only, NOT a result)",
        "git": git_identity(), "profile": profile, "profile_config": {k: (sorted(v) if isinstance(v, set) else v)
                                                                      for k, v in P.items()},
        "command": " ".join([Path(sys.executable).name, "-X utf8 -m scripts.m2_e2e_dryrun"] + sys.argv[1:]),
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "numpy": np.__version__,
                "platform": platform.platform(), "hardware": "CPU only, $0", "cuda_visible": False,
                "threads": torch.get_num_threads(), **kit},
        "stages": {}, "bugs_guarded": [],
    }
    st = receipt["stages"]

    # ---------------------------------------------------------------- 1. data + guard
    t = time.perf_counter()
    guard = SplitGuard.from_manifests(INDEX_MANIFESTS)
    recs, data_ident = load_slice(records, manifest, P["n_records"])
    enc = ByteEncoder()
    splits = {"train": [], "calib": [], "eval": []}
    for r in recs:
        splits[split_of(r["id"])].append(r)
    items = {k: items_from_records(v, enc, template=TOY_TEMPLATE, guard=guard) for k, v in splits.items()}
    for k in items:
        if not items[k]:
            raise RuntimeError(f"empty {k} split; raise n_records")
    probes = guard_probes(guard)
    st["data"] = {"records": data_ident, "guard": guard.identity(), "guard_probes": probes,
                  "guard_refuses_all_probes": all(v.startswith("refused") for v in probes.values()),
                  "splits": {k: {"n_records": len(splits[k]), "n_items": len(items[k]),
                                 "items_sha256": split_hash(items[k]),
                                 "max_tokens": max(len(i.row.ids) for i in items[k])} for k in items},
                  "template": TOY_TEMPLATE.identity(), "encoder": enc.identity(),
                  "seconds": round(time.perf_counter() - t, 2)}
    if not st["data"]["guard_refuses_all_probes"]:
        raise AssertionError(f"SplitGuard admitted a forged Index row: {probes}")

    # synthetic eval inputs shared by every card (development cohort stand-ins)
    dev_rows = synthetic_dev_rows(P["index_groups"])
    clash = [r["_evaluation"]["run_id"] for r in dev_rows if r["_evaluation"]["run_id"] in guard.ids]
    if clash:
        raise AssertionError(f"synthetic DEV-shaped ids collide with Index manifests: {clash}")
    steer_reqs = sdata.build(P["n_route"], P["n_pay"], split="dev")
    steer_variants = attacks.build_variants(steer_reqs, hijack=[], d1=None)  # clean + D1-A (D1-B held out)
    cohort = {"index_format_rows": {"kind": "synthetic DEV-shaped (catalogs 32, 38)", "n": len(dev_rows),
                                    "sha256": _sha(dev_rows)},
              "steer": {"split": "dev", **sdata.manifest(steer_reqs), "n_variants": len(steer_variants)},
              "decision_eval_items": {"n": len(items["eval"]), "sha256": split_hash(items["eval"])}}
    st["cohort"] = cohort

    # ---------------------------------------------------------------- 2-5 per configuration
    runs = []
    for kind in backbones:
        for head in heads:
            for arm in arms:
                runs.append(_one(kind, head, arm, out, P, items, guard, enc, cohort, dev_rows, steer_variants,
                                 full=P["full_card"] is None or (kind, head, arm) in P["full_card"],
                                 negative_checks=(head == heads[0] and arm == arms[0])))
                print(json.dumps({k: runs[-1].get(k) for k in ("backbone", "head", "arm", "seconds", "verdict_accept")}),
                      flush=True)
    st["runs"] = runs

    # ---------------------------------------------------------------- summary + receipt
    full = [r for r in runs if r.get("full_card")]
    summary = {
        "n_runs": len(runs), "n_full_cards": 2 * len(full),
        "all_losses_finite": all(r["losses_finite"] for r in runs),
        "all_resumes_ok": all(r["resume"]["ok"] for r in runs),
        "resume_equals_uninterrupted": [r["resume"].get("uninterrupted_max_abs_diff") for r in runs
                                        if "uninterrupted_max_abs_diff" in r["resume"]],
        "lineage_negative_checks": [r["resume"].get("negative") for r in runs if r["resume"].get("negative")],
        "guard_refuses_all_probes": st["data"]["guard_refuses_all_probes"],
        "steer_complete": all(r["cards"]["child"]["steer_complete"] for r in full),
        "index_scored": all(r["cards"]["child"]["index_status"] == "ok" for r in full),
        "promotion_verdicts": {f"{r['backbone']}/{r['head']}/{r['arm']}": {"accept": r["verdict"]["accept"],
                                                                         "failed": r["verdict"]["failed"]}
                               for r in full},
        "seconds_total": round(time.perf_counter() - t_start, 1),
    }
    receipt["summary"] = summary
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=float), encoding="utf-8", newline="\n")
    return receipt


def _one(kind, head, arm, out, P, items, guard, enc, cohort, dev_rows, steer_variants, full, negative_checks) -> dict:
    import numpy as np
    import torch

    from decide import Decider
    from harness import bootstrap as HB
    from harness import score as HS
    from heads.decide_backend import DecisionModelBackend
    from safety_eval import certify, selective
    from safety_eval.decide_bridge import DecideBridge
    from safety_eval.request import host_decision
    from safety_eval.data import QKEY, strip_meta
    from train import promotion
    from train.calibration import FeatureTemperature, features
    from train.data import TOY_TEMPLATE
    from train.evalcard import predict, run_card
    from train.trainer import Trainer

    t0 = time.perf_counter()
    k = P["steps"]
    rdir = out / f"{kind}-{head}-{arm}"
    cfg = make_cfg(head, arm, rdir / "ckpt", k, P["batch"])
    rec: dict[str, Any] = {"backbone": kind, "head": head, "arm": arm, "config_hash": cfg.hash(),
                           "config": cfg.to_json(), "full_card": full}
    train_items = items["train"]

    # ---- train k steps, checkpoint (parent)
    tr = Trainer(make_model(kind, head, cfg.seed), cfg, train_items, guard=guard)
    rec["lora_modules"] = len(tr.lora_modules)
    rec["trainable_params"] = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
    tr.train(k)
    parent_ckpt = rdir / "ckpt" / f"ckpt-step{k:07d}.pt"
    # ---- resume with a FRESH model + trainer, continue to 2k (child)
    tr2 = Trainer(make_model(kind, head, cfg.seed), cfg, train_items, guard=guard)
    resumed = tr2.resume()
    rs: dict[str, Any] = {"resumed_from_step": tr2.step if resumed else None}
    tr2.train()
    child_ckpt = rdir / "ckpt" / f"ckpt-step{2 * k:07d}.pt"
    rs["ok"] = bool(resumed and rs["resumed_from_step"] == k and tr2.step == 2 * k and child_ckpt.exists()
                    and parent_ckpt.exists())
    rec["history"] = tr2.history
    rec["losses_finite"] = all(np.isfinite(h["loss"]) for h in tr2.history)
    rec["lineage"] = tr2.lineage
    if negative_checks:
        # an uninterrupted 2k-step run must equal train(k); save; fresh resume; train(k)
        ref = Trainer(make_model(kind, head, cfg.seed), make_cfg(head, arm, rdir / "ckpt-ref", k, P["batch"],
                                                                 ckpt_every=0), train_items, guard=guard)
        ref.train()
        rs["uninterrupted_max_abs_diff"] = max_abs_diff(ref.trainable_state(), tr2.trainable_state())
        neg = {}
        try:  # different training data -> lineage refusal
            trainer_at(kind, head, cfg, list(reversed(train_items)), guard, parent_ckpt)
            neg["different_data"] = "ACCEPTED"
        except ValueError as exc:
            neg["different_data"] = "refused" if "lineage" in str(exc) else f"refused (other): {exc}"
        try:  # different (non-operational) config -> config refusal
            trainer_at(kind, head, make_cfg(head, arm, rdir / "ckpt", k, P["batch"], lr_head=2e-2), train_items,
                       guard, parent_ckpt)
            neg["different_config"] = "ACCEPTED"
        except ValueError as exc:
            neg["different_config"] = "refused" if "config" in str(exc) else f"refused (other): {exc}"
        rs["negative"] = neg
        rs["ok"] = rs["ok"] and rs["uninterrupted_max_abs_diff"] == 0.0 and all(v == "refused" for v in neg.values())
    rec["resume"] = rs
    child = tr2.model
    parent = trainer_at(kind, head, cfg, train_items, guard, parent_ckpt).model

    # ---- 3. temperature on the calibration split (child logits frozen)
    lp, labels, ev, ks = predict(child, items["calib"])
    temp = FeatureTemperature.fit(lp, torch.tensor(labels), features(ev, ks), [i.id for i in items["calib"]])
    rec["temperature"] = {**temp.identity(), "fit_report": temp.fit_report}
    if negative_checks:
        bad = FeatureTemperature.fit(*_pl(predict(child, items["eval"])), [i.id for i in items["eval"]])
        try:
            run_card(child, items["eval"], 0, temperature=bad)
            rec["temperature"]["refuses_eval_fit"] = False
        except ValueError:
            rec["temperature"]["refuses_eval_fit"] = True

    # ---- 4. eval cards (parent = step k, child = step 2k)
    cards, fields, units = {}, {}, {}
    for name, model, step in (("parent", parent, k), ("child", child, 2 * k)):
        kw: dict[str, Any] = {}
        decider = None
        if full:
            backend = DecisionModelBackend(model, enc, template=TOY_TEMPLATE, layout=cfg.layout,
                                           checkpoint={"path": str((parent_ckpt if name == "parent" else child_ckpt)
                                                                   .relative_to(out)), "step": step})
            decider = Decider(backend)
            kw = dict(index_rows=dev_rows, index_predict=index_predictor(decider), steer_variants=steer_variants,
                      steer_decide=DecideBridge(decider), keep_units=True, n_boot=P["n_boot"])
        card = run_card(model, items["eval"], step, out_dir=rdir / "cards", temperature=temp, cohort=cohort, **kw)
        c = {"raw": _dm(card["raw"]), "tempered": _dm(card["tempered"]), "index_status": card["index_score"]["status"]}
        if full:
            c["index_scores"] = card["index_score"]["scores"]
            c["index_statuses"] = card["index_score"]["statuses"]
            sm = card["steer"]["summary"]
            fam = sm["families"].get("d1_A") or sm["families"].get("d1") or {}
            c["steer_status_counts"] = sm["status_counts"]
            c["steer_complete"] = bool(fam.get("complete"))
            c["steer_d1a"] = {kk: (fam.get(kk) or {}).get("point") if isinstance(fam.get(kk), dict) else fam.get(kk)
                              for kk in ("n_units", "steer_rate", "hijack_rate", "alarm_rate", "violation_rate",
                                         "clean_acc")}
            units[name] = card["units"]
            # G4 exact certificate (k = 1) on the first base requests' clean + D1-A variants
            base = sorted({v["request_id"].split("::")[0] for v in steer_variants})[: P["n_g4"]]
            g4v = [v for v in steer_variants if v["request_id"].split("::")[0] in base]
            bridge = DecideBridge(decider)
            g4rows = certify.evaluate(bridge, g4v, k=1, arm="exact", max_decide_calls=256)
            gm = certify.g4_metrics(g4rows)
            c["g4_exact_k1"] = {"n_variants": len(g4v), "guarantee_checked_units": certify.check_guarantee(g4rows),  # raises on a violation
                                **{kind_: {m: gm[kind_][m] for m in ("escalation_rate", "auto_execute_rate",
                                                                     "certified_but_wrong_rate", "decide_calls")}
                                   for kind_ in ("clean", "inserted") if kind_ in gm}}
            # G5 LTT selective bound on the tiny model's (stand-in) executed confidences: decision eval items
            probs_t = card["units"]["decision_probs_tempered"]
            sc = [max(p) for p in probs_t]
            wr = [int(np.argmax(p)) != it.label for p, it in zip(probs_t, items["eval"])]
            cl = [it.id.rsplit("/", 1)[0] for it in items["eval"]]
            g5 = selective.ltt_selective(sc, wr, alpha=0.2, delta=0.1, clusters=cl)
            c["g5_ltt"] = {kk: g5.get(kk) for kk in ("n_s", "mu_s", "feasibility", "lam_hat", "chosen")}
            fields[name] = promotion.decision_fields(items["eval"], card["units"]["decision_probs_tempered"])
        cards[name] = c
    rec["cards"] = cards

    # ---- 5. G12/G13 promotion rule (child vs parent)
    if full:
        cap = HB.compare(HS.decompose(dev_rows, units["child"]["index_results"]),
                         HS.decompose(dev_rows, units["parent"]["index_results"]), B=P["n_boot"], seed=0)
        dec = promotion.paired_decision(fields["child"], fields["parent"], n_boot=P["n_boot"])
        v = promotion.verdict(decision=dec, steer_child_rows=units["child"]["steer_rows"],
                              steer_parent_rows=units["parent"]["steer_rows"], capability=cap, n_boot=P["n_boot"])
        # self-comparison sanity: a checkpoint vs itself must not be rejected for regression
        self_v = promotion.verdict(decision=promotion.paired_decision(fields["parent"], fields["parent"],
                                                                      n_boot=P["n_boot"]),
                                   steer_child_rows=units["parent"]["steer_rows"],
                                   steer_parent_rows=units["parent"]["steer_rows"],
                                   capability=HB.compare(HS.decompose(dev_rows, units["parent"]["index_results"]),
                                                         HS.decompose(dev_rows, units["parent"]["index_results"]),
                                                         B=P["n_boot"], seed=0), n_boot=P["n_boot"])
        rec["verdict"] = {"accept": v["accept"], "failed": v["failed"],
                          "criteria": [{kk: c.get(kk) for kk in ("criterion", "pass", "evaluable", "reason")}
                                       for c in v["criteria"]],
                          "capability_diff_index": cap["diff"]["index"], "margins": v["margins"]}
        rec["verdict_self"] = {"accept": self_v["accept"], "failed": self_v["failed"]}
        rec["verdict_accept"] = v["accept"]
        (rdir / "verdict.json").write_text(json.dumps(v, indent=1, default=float), encoding="utf-8")
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def _pl(pred):
    import torch

    from train.calibration import features

    lp, labels, ev, ks = pred
    return lp, torch.tensor(labels), features(ev, ks)


def _probs_t(model, items, temp):
    from train.calibration import features
    from train.evalcard import predict

    lp, _, ev, ks = predict(model, items)
    return [p[:k] for p, k in zip(temp.apply(lp, features(ev, ks)).exp().tolist(), ks)]


def _dm(m: dict) -> dict:
    return {k: m.get(k) for k in ("n", "accuracy", "nll", "brier", "ece_binned15", "ece_debiased", "mean_conf") if k in m}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--heads", default=",".join(HEADS))
    a = ap.parse_args(argv)
    rec = run(Path(a.out), a.profile, Path(a.records), Path(a.manifest), backbones=tuple(a.backbones.split(",")),
              heads=tuple(a.heads.split(",")))
    print(json.dumps(rec["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
