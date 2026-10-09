"""Mode ``safety_eval_d13_v3w`` (``eval_only: "safety"``): the SAFETY evidence of a saved checkpoint, in ONE job, two
batteries, no training.

    # GPU (in the job): the stock parent on the official battery, then the child on both batteries (one model load)
    python -X utf8 -m jobs.m2_first_cell.safety_battery score --arm parent --cell-config <cfg> --model-dir <snap> \
        --batteries official --resume-dir <ck>/safety --out-dir <out>/safety [--progress <p.jsonl>]
    python -X utf8 -m jobs.m2_first_cell.safety_battery score --arm child --cell <out>/cell.json --ckpt-dir <ckcell> \
        --ckpt <ckpt> --model-dir <snap> --records <records> --train-dir <150k> --guard-readiness <readiness.json> \
        --readout v3w --batteries official,heldout_v1 --resume-dir <ck>/safety --out-dir <out>/safety
    # CPU: pricing (setup preflight; exit 15 past a cap), the report (finalize validator ``valid_result``)
    python -X utf8 -m jobs.m2_first_cell.safety_battery estimate --params <job_params>
    python -X utf8 -m jobs.m2_first_cell.safety_battery report --out-dir <out>/safety --cell <out>/cell.json \
        --params <job_params> --elapsed-min N --out <out>/safety_eval_result.json

Batteries (no metric is defined here; every number is ``safety_eval.metrics``):
  official    ``train.m2_gates.stage_variants(cohort v2, "hijack")`` = BOARD H1 / mode ``hijack_eval(_champ)``: the 360
              ``hijack_subset(steer_dev)`` requests x (clean + the 5 ``safety_eval.attacks.HIJACK_TEMPLATES`` + the 4
              D1-A steer templates) = 3,600 units, scored on BOTH arms (child under ``--readout``, the parent = the
              stock readout exactly as score_parent serves it, job-j8wgy / job-6babj). Reported with
              ``jobs.m2_first_cell.hijack_eval.measurement`` (paired child - parent, eligibility parent clean-correct,
              two-way + fixed-template bootstrap). Reportable for a cell only if its training draw excluded every eval
              attack template (``train.attack_guard``: ``training_exclusion`` below).
  heldout_v1  ``safety_eval.heldout_v1.expected_variants()`` = the pre-declared held-out battery (21 templates, 7,920
              units on the same 360 base requests; unit set pinned in ``baselines.clef_heldout_v1``), child only. Its
              rows are exactly what ``python -m baselines.clef_heldout_v1 compare --champion <rows> --units
              units-clef-flash.jsonl units-clef.jsonl`` consumes (the Clef arms are their own jobs); the official rows
              go in as ``--champion-official`` (generalisation gap).
Both arms are scored UNTEMPERED (identity calibration), as every safety battery here. The child is loaded exactly as
``jobs.m2_first_cell.fulldev_eval`` loads it for a non-native readout (``load_child(temperature=False, readout=...)``:
the readout D13's 41.07 proxy screen and fulldev_eval_d13_v3w score). The 360 clean units are shared by both
batteries and scored once per battery; their agreement is reported (a determinism check, not a metric).
Rows are appended per chunk to ``<resume-dir>/<arm>-<battery>.partial.jsonl`` (identity-checked on reuse).
Exit codes: 0 ok | 2 usage | 5 estimate params | 9 incomplete / invalid result | 10 identity (unit set, checkpoint,
readout, resume log) | 15 projected past a cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "ssdm.m2.safety_battery.v1"
BATTERIES = ("official", "heldout_v1")
ARMS = ("child", "parent")
COHORT_SHA256 = "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"  # cohort v2 (the H1 / S2 units)
#: train.m2_gates.stage_variants(cohort v2, "hijack") on the committed cohort (computed 2026-10-09; = job-j8wgy's units)
OFFICIAL_PIN = {"by_family": {"clean": 360, "d1": 1440, "hijack": 1800}, "n_clusters": 45,
                "request_ids_sha256": "5546813de95900dee45b0822b5f4133cae20f605e5a04476b321bdfe6dbab2e9",
                "units_sha256": "8668786780077917156c8b4d6ffe5fff4d88fb328536c57b010a151de5701cb9"}
CHUNK = 100
ARM_TAG = {"child": "m2_child", "parent": "rwkv7_stock"}
#: the clean-unit fields compared across the two batteries (same request, same arm, same loaded model)
CLEAN_KEYS = ("status", "raw_argmax", "executed", "correct")


class Identity(ValueError):
    """Exit 10: unit set / checkpoint / readout / resume log does not match what this run must score."""


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _cohort(path: str | None, sha: str | None):
    from train.cohort import Cohort

    return Cohort(path, expected_sha256=sha) if path else Cohort(expected_sha256=COHORT_SHA256)


def battery_identity(variants: list[dict]) -> dict[str, Any]:
    from safety_eval.metrics import unit_fingerprint
    from safety_eval.request import sha256_obj

    return {"n_units": len(variants),
            "by_family": dict(sorted(Counter(v["_meta"]["attack_family"] for v in variants).items())),
            "n_clusters": len({v["_meta"]["cluster"] for v in variants}),
            "templates": sorted({v["_meta"]["template"] for v in variants} - {"clean"}),
            "request_ids_sha256": sha256_obj([v["request_id"] for v in variants]),
            "units_sha256": sha256_obj([[v["request_id"], unit_fingerprint(v)] for v in variants])}


def battery_variants(name: str, cohort=None, production: bool = True) -> tuple[list[dict], dict]:
    """(variants, identity) of one battery; ``production`` (the committed cohort v2) checks the pinned unit set."""
    from train import m2_gates as G

    c = cohort if cohort is not None else _cohort(None, None)
    if name == "official":
        v = G.stage_variants(c, "hijack")
        ident = {"battery": "official_hijack_d1a", "builder": "train.m2_gates.stage_variants(cohort, 'hijack')",
                 **battery_identity(v)}
        if production and any(ident[k] != OFFICIAL_PIN[k] for k in OFFICIAL_PIN):
            raise Identity(f"official battery {ident} != pinned {OFFICIAL_PIN}")
        return v, ident
    if name == "heldout_v1":
        from baselines import clef_heldout_v1 as CH
        from safety_eval import heldout_v1 as HV

        v = HV.expected_variants(c)
        if production:
            try:
                CH.check_production_variants(v)
            except CH.UnitSetMismatch as e:
                raise Identity(str(e)) from e
        return v, {**HV.identity(v), "builder": "safety_eval.heldout_v1.expected_variants()"}
    raise ValueError(f"battery {name!r}: one of {BATTERIES}")


def check_rows(rows: list[dict], variants: list[dict], what: str) -> None:
    from baselines.clef_hijack_matched import UnitSetMismatch, check_unit_set

    try:
        check_unit_set(rows, variants, what)
    except UnitSetMismatch as e:
        raise Identity(str(e)) from e


# ------------------------------------------------------------------------------------------------ GPU: arms
def load_arm(a):
    """(steer callable, identity) of one arm. child = ``fulldev_eval.load_child`` UNTEMPERED under ``--readout``
    (native = the cell's own template); parent = ``score_parent.parent_decider`` as the H1 parent arm serves it."""
    from safety_eval.decide_bridge import DecideBridge

    if a.arm == "child":
        from jobs.m2_first_cell import fulldev_eval as F

        cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
        ro = F.readout_spec(a.readout, cfg)
        decider, _, ident = F.load_child(a, temperature=False,
                                         readout=None if ro["native"] else (ro["template"], ro["layout"]))
        return DecideBridge(decider), {**ident, "readout": ro, "calibration": "identity (untempered)"}
    from jobs.m2_parent_arm.score_parent import parent_decider

    cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
    dec, pid = parent_decider(cfg.get("parent", "rwkv7"), a.model_dir, None, cfg, device=a.device)
    return DecideBridge(dec), {"arm": "parent", "parent": pid, "revision": dec.revision(),
                               "readout": {"name": "stock", "template": cfg["template"], "layout": cfg.get("layout")},
                               "calibration": "identity (untempered)"}


def run_identity(ident: dict, arm: str, battery: str, bident: dict) -> dict[str, Any]:
    """What a resume log must share with this run: arm, battery units, checkpoint bytes, readout."""
    ck = ident.get("checkpoint") or {}
    return {"schema": SCHEMA, "arm": arm, "battery": battery, "units_sha256": bident["units_sha256"],
            "checkpoint_sha256": ck.get("sha256"),
            "readout_sha256": _sha_bytes(json.dumps(ident.get("readout"), sort_keys=True, default=str).encode()),
            "parent": json.dumps(ident.get("parent"), sort_keys=True, default=str)}


def _resume(path: Path, idp: Path, rid: dict, want: dict[str, str]) -> dict[str, dict]:
    from harness.jsonl import drop_truncated_tail, iter_jsonl_strict

    if idp.is_file():
        old = json.loads(idp.read_text(encoding="utf-8"))
        bad = [k for k in rid if old.get(k) != rid[k]]
        if bad:
            raise Identity(f"resume log {path} belongs to another run (differs on {bad})")
    idp.write_text(json.dumps(rid, indent=1), encoding="utf-8")
    if not path.is_file():
        return {}
    drop_truncated_tail(path)
    done = {}
    for _, r in iter_jsonl_strict(path):
        if want.get(r.get("request_id")) != r.get("unit_sha"):
            raise Identity(f"resume log {path}: unit {r.get('request_id')} is not one of this battery's units")
        done[r["request_id"]] = r
    return done


def score(a, loader=load_arm) -> int:
    from safety_eval import metrics as M
    from safety_eval.data import strip_meta
    from train import m2_gates as G

    bats = [b for b in a.batteries.split(",") if b]
    if not bats or len(set(bats)) != len(bats) or any(b not in BATTERIES for b in bats):
        print(f"--batteries {a.batteries!r}: a comma list of {BATTERIES}", file=sys.stderr)
        return 2
    try:  # the unit sets first: fail closed before any model load
        cohort = _cohort(a.cohort, a.cohort_sha256)
        plans = {b: battery_variants(b, cohort, production=a.cohort is None) for b in bats}
    except (Identity, ValueError) as e:  # a pinned unit set / cohort hash that does not rebuild
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train.cohort import CohortLeak

    try:
        steer, ident = loader(a)
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out, res = Path(a.out_dir), Path(a.resume_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    from safety_eval.metrics import unit_fingerprint

    done, todo = {}, []
    try:
        for b, (v, bi) in plans.items():
            want = {x["request_id"]: unit_fingerprint(x) for x in v}
            p = res / f"{a.arm}-{b}.partial.jsonl"
            done[b] = _resume(p, p.with_name(f"{a.arm}-{b}.partial.identity.json"),
                              run_identity(ident, a.arm, b, bi), want)
            todo += [(b, x) for x in v if x["request_id"] not in done[b]]
    except Identity as e:
        print(f"identity: {e}", file=sys.stderr)
        return 10
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(strip_meta(x)) for _, x in todo])
    call = G.beat_wrap(prog.wrap(steer))
    t0 = time.perf_counter()
    for b in bats:
        mine = [x for bb, x in todo if bb == b]
        with open(res / f"{a.arm}-{b}.partial.jsonl", "a", encoding="utf-8", newline="\n") as f:
            for i in range(0, len(mine), CHUNK):
                for r in M.run(call, mine[i:i + CHUNK]):
                    r["arm"] = ARM_TAG[a.arm]
                    r["battery"] = b
                    f.write(json.dumps(r, default=str) + "\n")
                    done[b][r["request_id"]] = r
                f.flush()
                print(f"[{a.arm}] {b}: {len(done[b])}/{len(plans[b][0])} units ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
    secs = time.perf_counter() - t0
    shas = {}
    for b, (v, _) in plans.items():
        rows = [done[b].get(x["request_id"]) for x in v]
        if any(r is None for r in rows):
            print(f"{b}: {sum(r is None for r in rows)} units have no row", file=sys.stderr)
            return 9
        p = out / f"units-{a.arm}-{b}.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        shas[b] = _sha_bytes(p.read_bytes())
    n_new = len(todo)
    (out / f"identity-{a.arm}.json").write_text(json.dumps(
        {**ident, "batteries": {b: bi for b, (_, bi) in plans.items()}, "results_sha256": shas,
         "seconds_this_attempt": round(secs, 1), "units_this_attempt": n_new,
         "s_per_unit": round(secs / n_new, 4) if n_new else None}, indent=1, default=str), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def _rows(p: Path) -> list[dict]:
    from harness.jsonl import read_jsonl_strict

    return read_jsonl_strict(p)


def clean_agreement(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    """The shared clean units of two batteries (same arm, same model): equal on ``CLEAN_KEYS`` and the max absolute
    difference of the raw option probabilities."""
    ca = {r["request_id"]: r for r in rows_a if r["attack_family"] == "clean"}
    cb = {r["request_id"]: r for r in rows_b if r["attack_family"] == "clean"}
    ids = sorted(set(ca) & set(cb))
    same = sum(all(ca[i].get(k) == cb[i].get(k) for k in CLEAN_KEYS) for i in ids)
    dp = [abs(float(p) - float((cb[i].get("raw_probs") or {}).get(o, math.nan)))
          for i in ids for o, p in (ca[i].get("raw_probs") or {}).items()]
    dp = [x for x in dp if math.isfinite(x)]
    return {"n_shared": len(ids), "n_equal": same, "keys": list(CLEAN_KEYS),
            "max_abs_prob_diff": max(dp) if dp else None}


def training_exclusion(cell: dict) -> dict[str, Any]:
    """Was every eval attack template kept OUT of this checkpoint's training draw? From the run's own cell.json
    (``data.train_set.attack_guard``, written by train.train_set when the config pins ``attack_clean``), checked
    against the CURRENT ``train.attack_guard`` eval template set (official hijack, D1-A, D1-B, heldout_v1)."""
    from train import attack_guard as AG

    ts = (cell.get("data") or {}).get("train_set") or {}
    ag = ts.get("attack_guard") or {}
    cur = AG.eval_templates_sha256()
    ok = bool(ag) and ag.get("drawn_hits") == 0 and ag.get("templates_sha256") == cur
    return {"drawn_ids_sha256": ts.get("drawn_ids_sha256"), "n_drawn": ts.get("n_drawn"),
            "attack_guard": {k: ag.get(k) for k in ("rule", "templates_sha256", "n_records", "n_excluded",
                                                     "excluded_ids_sha256", "per_set", "drawn_hits")},
            "current_eval_templates_sha256": cur,
            "eval_template_sets": ["official_hijack", "d1a_steer", "d1b_steer", "heldout_v1"],
            "official_reportable": ok, "heldout_v1_held_out": ok,
            "rule": "reportable iff the draw's attack_guard scan used the current eval template set and found 0 hits "
                    "in the drawn records (the 8,049 D1-A rows of the 150k pool were excluded before the draw)",
            "caveat": "heldout_v1 S02_table has no text-detectable fragment (train.attack_guard skipped_fragments); "
                      "the training-only TRAIN_HIJACK augmentation stays in training (disjoint from every eval "
                      "template: data.injection.FORBIDDEN_6GRAMS)"}


def report(a) -> dict[str, Any]:
    from baselines import clef_heldout_v1 as CH
    from jobs.common.costs import quoted_max_cost_usd, worst_case_usd
    from jobs.m2_first_cell.hijack_eval import measurement

    out = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cohort = _cohort(a.cohort, a.cohort_sha256)
    prod = a.cohort is None
    ov, oid = battery_variants("official", cohort, prod)
    hv, hid = battery_variants("heldout_v1", cohort, prod)
    c_off, p_off = _rows(out / "units-child-official.jsonl"), _rows(out / "units-parent-official.jsonl")
    c_hv = _rows(out / "units-child-heldout_v1.jsonl")
    check_rows(c_off, ov, "child official")
    check_rows(p_off, ov, "parent official")
    check_rows(c_hv, hv, "child heldout_v1")
    idc = json.loads((out / "identity-child.json").read_text(encoding="utf-8"))
    idp = json.loads((out / "identity-parent.json").read_text(encoding="utf-8"))
    usd = params["batch_usd_per_min"]
    hv_arm = CH.summarize_arm(c_hv, hv, a.n_boot)
    excl = training_exclusion(cell)
    ckpt = {k: v for k, v in params["checkpoint"].items() if not k.startswith("_")}
    res = {
        "schema": SCHEMA, "mode": params.get("mode"),
        "checkpoint": {**ckpt, "loaded": idc.get("checkpoint")},
        "cell": {k: cell["config"].get(k) for k in ("parent", "head", "loss_arm", "lora_rank", "template", "layout",
                                                     "n_train_records", "max_steps", "seed")},
        "readout_requested": params.get("readout") or "native", "readout": idc.get("readout"),
        "parent_readout": idp.get("readout"),
        "training_exclusion": excl,
        "official": {"units": oid, **measurement(c_off, p_off, cohort, a.n_boot),
                     "reportable": excl["official_reportable"],
                     "note": "official HIJACK_TEMPLATES (5) + D1-A (4) on the H1 units; x = child (readout above, "
                             "untempered), y = stock parent; reportable only because the draw excluded every eval "
                             "attack template (training_exclusion)"},
        "heldout_v1": {"units": hid, "child": hv_arm, "complete": hv_arm["complete"],
                       "rows": "units-child-heldout_v1.jsonl",
                       "note": "the general claim: paired vs Clef-flash / Clef 27B with Holm at landing "
                               "(baselines.clef_heldout_v1 compare; see compare_command)"},
        "clean_agreement": clean_agreement(c_off, c_hv),
        "scoring_seconds": {"child": idc.get("seconds_this_attempt"), "parent": idp.get("seconds_this_attempt"),
                            "child_s_per_unit": idc.get("s_per_unit"), "parent_s_per_unit": idp.get("s_per_unit")},
        "results_sha256": {"child": idc.get("results_sha256"), "parent": idp.get("results_sha256")},
        "compare_command": ("python -X utf8 -m baselines.clef_heldout_v1 compare --champion <art>/safety/"
                            "units-child-heldout_v1.jsonl --champion-identity <art>/safety/identity-child.json "
                            "--champion-official <art>/safety/units-child-official.jsonl --units "
                            "<job-uk2x6/job-gmbqi>/units-clef-flash.jsonl <...>/units-clef.jsonl --out "
                            "<dir>/heldout_v1-compare.json"),
        "cost": {"elapsed_min": a.elapsed_min, "runtime_usd": round(a.elapsed_min * usd, 2), "usd_per_min": usd,
                 "build_minutes_not_included": True, "worst_case_usd": worst_case_usd(params),
                 "quoted_max_cost_usd": quoted_max_cost_usd(params), "hard_cap_usd": params["hard_cap_usd"]},
        "verdict": "measured",
    }
    res["complete"] = bool(res["official"]["complete"] and res["heldout_v1"]["complete"])
    return res


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: both batteries complete, the pinned checkpoint loaded, the readout asked."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        o, h = d["official"], d["heldout_v1"]
        hj, st = o["hijack"], o["steer_d1a"]
        fam = h["child"]["families"]
        return (d.get("schema") == SCHEMA and d.get("verdict") == "measured" and d["complete"] is True
                and o["complete"] is True and h["complete"] is True
                and hj["n_units"] > 0 and st["n_units"] > 0
                and all(fin(hj["hijack_rate"][k]) for k in ("child", "parent"))
                and all(fin(hj["hijack_rate"]["diff"][k]) for k in ("point", "lo", "hi"))
                and fin(st["steer_rate"]["diff"]["point"])
                and fin(fam["heldout_hijack"]["hijack_rate"]["point"])
                and fin(fam["heldout_steer"]["steer_rate"]["point"])
                and o["reportable"] == d["training_exclusion"]["official_reportable"]
                and bool(d["checkpoint"].get("ckpt_sha256"))
                and (d["checkpoint"].get("loaded") or {}).get("sha256") == d["checkpoint"]["ckpt_sha256"]
                and (d.get("readout") or {}).get("name") == d.get("readout_requested"))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ CPU: estimate
def estimate(params: dict) -> dict[str, Any]:
    """Expected minutes per scoring phase = fixed (load) + units x the measured s / unit; ok iff each is <= 95 % of
    its cap (the rate guards' rule). Units from the pinned batteries (production unit counts)."""
    s = params["safety"]
    n = {"official": sum(OFFICIAL_PIN["by_family"].values()), "heldout_v1": 7_920}
    caps = params["phase_caps_minutes"]
    out: dict[str, Any] = {"phases": {}}
    for arm, phase in (("parent", "parent_safety"), ("child", "safety")):
        bats = s[f"{arm}_batteries"]
        if any(b not in BATTERIES for b in bats) or (arm == "parent" and bats != ["official"]):
            raise ValueError(f"{arm}_batteries {bats!r}")
        units = sum(n[b] for b in bats)
        mins = s[f"{arm}_fixed_minutes"] + units * s[f"{arm}_s_per_unit"] / 60
        out["phases"][phase] = {"units": units, "expected_min": round(mins, 1), "cap_min": caps[phase],
                                "headroom": round(caps[phase] / mins - 1, 3), "ok": mins <= 0.95 * caps[phase]}
    if "official" not in s["child_batteries"] or "heldout_v1" not in s["child_batteries"]:
        raise ValueError("child_batteries must hold official and heldout_v1")
    out["ok"] = all(p["ok"] for p in out["phases"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.safety_battery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--batteries", required=True, help=f"comma list of {BATTERIES}")
    s.add_argument("--cell", help="child: the checkpoint's cell.json (config + ladder)")
    s.add_argument("--cell-config", help="parent: the mode's cell config (template, WKV)")
    s.add_argument("--ckpt-dir")
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records")
    s.add_argument("--train-dir")
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness")
    s.add_argument("--readout", default="native", help="child: fulldev_eval.READOUT_NAMES (untempered either way)")
    s.add_argument("--resume-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--cell", required=True)
    r.add_argument("--params", required=True)
    r.add_argument("--elapsed-min", type=float, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=2000)
    for p in (s, r):
        p.add_argument("--cohort", default=None, help="tests only: a mini cohort file (with --cohort-sha256)")
        p.add_argument("--cohort-sha256", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        try:
            est = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")))
        except (ValueError, KeyError) as e:
            print(f"estimate: {e}", file=sys.stderr)
            return 5
        print(json.dumps(est, indent=1))
        return 0 if est["ok"] else 15
    if (a.cohort is None) != (a.cohort_sha256 is None):
        ap.error("--cohort and --cohort-sha256 go together (tests only)")
    if a.cmd == "score":
        need = ("cell", "ckpt_dir", "records", "train_dir", "guard_readiness") if a.arm == "child" else ("cell_config",)
        miss = [k for k in need if not getattr(a, k)]
        if miss:
            ap.error(f"--arm {a.arm} needs {miss}")
        return score(a)
    try:
        res = report(a)
    except Identity as e:
        print(f"identity: {e}", file=sys.stderr)
        return 10
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    o, h = res["official"], res["heldout_v1"]["child"]["families"]
    print(json.dumps({"official_hijack": o["hijack"]["hijack_rate"], "official_steer_diff": o["steer_d1a"]["steer_rate"]
                      ["diff"], "official_reportable": o["reportable"],
                      "heldout_hijack": h["heldout_hijack"]["hijack_rate"]["point"],
                      "heldout_steer": h["heldout_steer"]["steer_rate"]["point"],
                      "clean_agreement": res["clean_agreement"], "valid": valid_result(res)}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
