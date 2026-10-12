"""Evidence lane: ONE rescore + guarantee report for a landed SSDM cell (m2_first_cell screen artifact or a lead landing
dir that holds the same files). Everything is recomputed from the raw per-item files; numbers in screen.json /
perturb_report.json are only diffed against. Artifact = untrusted data: run with ``python -I`` from this checkout.

  python -I -X utf8 scripts/ev_cell_rescore.py --art <extracted artifact dir> --control stock_shim \
      --primary native_shimcal --out <dir> [--price <spec | price.json>] [--expect-steps N] [--slice-ids-sha SHA] \
      [--target NAME=<preds.jsonl>::<price spec | price.json>]... [--holm-arm ARM]... [--holm-ext NAME=P]... \
      [--g1 NAME=<dir>]... [--g1-max-dp 0] [--flip8 <dir>] [--flip-baseline NAME=<base.jsonl>,<pert.jsonl>] \
      [--flip-baseline-receipt <flip_ci_8op.json>::<model>] [--mtimes <json>] [--identity-child <json>] \
      [--groups "A=f1+f2;B=..."] [--B 1000] [--seed 16] [--score-inv <dir> [--score-inv-max-dp 0]]

Sections of the report (<out>/cell_rescore.json + <out>/CELL_RESCORE.md):
  (1) PRIMARY  paired delta Composite A (primary - in-job control; B=1000 seed 16, strata of the control table, S/K
               held at each arm's measured value) with 95% CI, one-sided bootstrap p, Holm levels over --holm-arm /
               --holm-ext; the diff of every number against the in-job screen.json (scripts/ev_rescore_jevkit.py).
  (2) SECONDARY per EV-PREREG-SECONDARY-AXES (per-axis CIs, top-1 + exact McNemar, choice NLL, I<50 fraction,
               Composite-A quantiles), per-type cc / top-1, noul 0.2-0.8 band.
  (3) TARGET   primary - existing preds of a target system (e.g. job-zbtbj's H2O), paired on ids; S and K held at
               each system's OWN value (cross-job: S measured in another job / hardware run -> caveat column).
  (4) INTEGRITY 231 ids, no duplicates, choice option order = criteria order (loader contract), 0 truncation, the
               expected trained steps, the scored checkpoint = the selected one.
  (5) SELECTION cell.json dev cards (argmin NLL = best), the step-0 floor from slice_floor_rows.jsonl (paired CI +
               exact McNemar), slice ids sha, timestamps of the selection files vs the first public pred.
  (6) G1       flips recomputed from the perturbed preds for option_reverse / option_shuffle / key_relabel: every
               op n > 0, 0 invalid, the ON-DISK perturbed.jsonl must list each row in the regenerated in-memory order
               and differ from base order (file-order check; a sort_keys dump makes the check vacuous -> FAIL);
               PASS only with 0 flips and max|dp| == 0 (bitwise), or <= --g1-max-dp flagged NON-BITWISE.
  (6b) SCORE-INV  (optional, --score-inv <dir>) score-item invariance, SEPARATE from G1: scripts/ev_score_flip.py on
               the requests of scripts/ev_score_perturb.py (score_reverse / score_shuffle / score_relabel /
               score_permute, level-as-content rendering) + the head's preds. Reported alongside G1; it never changes
               the G1 section or its verdict, and its FAIL is a warning (exit code unchanged); read ``score_inv_pass``.
  (7) FLIPS    the 8-op battery (whatever ops the --flip8 dir holds) with bootstrap CIs over base items, paired vs a
               supplied baseline on the same resamples, plus a reported baseline (stock shim 7.55% [5.71, 9.67]).
  (8) TRANSFER per family: slice delta (selected - step 0) next to public delta (primary - control), flags.
Exit 0 = no FLAG; 3 = at least one FLAG (read ``flags``); 2 = inputs missing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import items as IT  # noqa: E402
from jevkit import perturb as P  # noqa: E402
from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402
from scripts import ev_rescore_jevkit as R  # noqa: E402
from scripts import ev_score_flip as SF  # noqa: E402

TYPES = ("choice", "noul", "score")
G1_OPS = ("option_reverse", "option_shuffle", "key_relabel")
QS = (0.5, 1, 1.25, 2.5, 5, 25, 50, 75, 95, 97.5, 98.75, 99, 99.5)
CROSS_JOB = ("cross-job: the target's S comes from its own job / hardware run and K from its own declared price; "
             "only I and C are paired on the same items")


# ----------------------------------------------------------------------------------------------- small helpers
def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rj(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def pct(v, qs=(2.5, 97.5)):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return [float(np.percentile(v, q)) for q in qs] if v else None


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2.0 ** n))


def utc(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def top1(t, items) -> np.ndarray:
    """argmax top-1: choice / score top label == gold; noul P(yes) >= 0.5 <-> gold yes (abstention not penalised)."""
    out = np.zeros(len(t.ids), dtype=bool)
    for i, it in enumerate(items):
        out[i] = (bool(t.valid[i]) and ((t.pyes[i] >= 0.5) == bool(t.yes[i]))) if it.type == "noul" \
            else bool(t.top_correct[i])
    return out


def type_cc(t, idx=None) -> dict:
    pts = JS.aggregate(t, idx, detail=True)["intelligence"].get("per_type_split") or {}
    return {k.split("|")[1]: v["cc_unclipped"] for k, v in pts.items() if k.startswith("open|")}


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down over a family of one-sided p values: rank k (0-based, ascending p) is tested at alpha / (m - k);
    testing stops at the first failure. Returns per name {p, rank, level, pass_p, tested}."""
    order = sorted(ps, key=lambda n: (ps[n], n))
    m, out, stop = len(order), {}, False
    for k, n in enumerate(order):
        lvl = alpha / (m - k)
        ok = (not stop) and ps[n] < lvl
        out[n] = {"p": ps[n], "rank": k + 1, "level": lvl, "tested": not stop, "pass_p": bool(ok)}
        stop = stop or not ok
    return out


def find_art_file(art: Path, rel: str) -> Path | None:
    return R.find_one(art, rel)


# ----------------------------------------------------------------------------------------------- (6)/(7) flips
def load_pert_file(path: Path) -> list[dict]:
    return rj(path)


def to_perturbed(rows: list[dict]) -> list[P.Perturbed]:
    out = []
    for d in rows:
        d = dict(d)
        m = d.pop("perturbation")
        out.append(P.Perturbed(IT.Item.from_dict(d), m["op"], m["seed"], m["base_id"], m["label_map"]))
    return out


def regenerate(items, pert_ids) -> list[P.Perturbed]:
    """The in-memory perturbations jevkit would make for these ids (``<base>~<op>~<seed>``), in that order."""
    by = {it.id: it for it in items}
    out = []
    for pid in pert_ids:
        base, op, seed = pid.rsplit("~", 2)
        p = P.OPERATORS[op](by[base], int(seed))
        if p is None or p.item.id != pid:
            raise ValueError(f"{pid}: jevkit does not generate this perturbation")
        out.append(p)
    return out


def _base_order(question: dict, label_map: dict) -> list[str]:
    inv = {str(v): str(k) for k, v in (label_map or {}).items()}
    crit = question.get("criteria") or {}
    return [inv.get(str(o), str(o)) for o in (crit if isinstance(crit, dict) else range(len(crit)))]


def file_order_check(items, rows: list[dict]) -> dict:
    """Per operator, from the ON-DISK rows: ``matches_regenerated`` = the row lists its options (keys, in order) exactly
    as jevkit's in-memory perturbation does; ``changed_vs_base`` = the row's option order mapped back to base ids differs
    from the base item's (reorder ops) or its keys were renamed (key_relabel). A sort_keys dump restores base order:
    0 changed, the reorder check is vacuous."""
    by = {it.id: it for it in items}
    regen = {p.item.id: p for p in regenerate(items, [r["id"] for r in rows])}
    out: dict[str, dict] = {}
    for r in rows:
        m = r["perturbation"]
        o = out.setdefault(m["op"], {"n": 0, "matches_regenerated": 0, "changed_vs_base": 0, "not_matching_ids": []})
        o["n"] += 1
        q = r.get("question") or {}
        want = regen[r["id"]].item.question.get("criteria") or {}
        same = list(q.get("criteria") or {}) == list(want)
        o["matches_regenerated"] += int(same)
        if not same and len(o["not_matching_ids"]) < 10:
            o["not_matching_ids"].append(r["id"])
        b = by[m["base_id"]]
        lm = m.get("label_map") or {}
        if m["op"] == "key_relabel":
            changed = any(str(k) != str(v) for k, v in lm.items())
        else:
            changed = _base_order(q, lm) != _base_order(b.question, {})
        o["changed_vs_base"] += int(changed)
    return out


def flip_outcomes(items, perturbed, base_ans, pert_ans) -> list[tuple]:
    """(base_index, op, flip in {0,1} | None = invalid) pair by pair: jevkit.perturb.flip_report's rule."""
    base = {it.id: it for it in items}
    idx = {it.id: i for i, it in enumerate(items)}
    rows = []
    for p in perturbed:
        b = base[p.base_id]
        db, dp_ = P._dist(b, base_ans.get(b.id)), P._dist(p.item, pert_ans.get(p.item.id))
        if db is None or dp_ is None:
            rows.append((idx[b.id], p.op, None))
            continue
        back = {lab: dp_[p.label_map[lab]] for lab in b.labels}
        rows.append((idx[b.id], p.op, int(P._decision(b, db) != P._decision(b, back))))
    return rows


def flip_rates(rows, ops, weights=None) -> dict:
    out = {}
    for op in ops:
        n = f = 0.0
        for bi, o, fl in rows:
            if o != op or fl is None:
                continue
            w = 1.0 if weights is None else weights[bi]
            n += w
            f += w * fl
        out[op] = (f / n) if n else None
    vals = [v for v in out.values() if v is not None]
    out["_average"] = (sum(vals) / len(vals)) if vals else None
    return out


def flip_boot(models: dict[str, list], ops: list[str], n_items: int, B: int, seed: int, ref: str | None) -> dict:
    """Bootstrap over BASE items (all perturbed variants of a drawn item come along), same resamples for every model;
    paired deltas vs ``ref``. Mirrors claude/ev-flip-rate receipts/2026-10-10-ev-flip-rate/flip_ci.py."""
    rng = np.random.default_rng(seed)
    keys = list(ops) + ["_average"]
    draws = {m: {k: [] for k in keys} for m in models}
    deltas = {m: {k: [] for k in keys} for m in models if ref and m != ref}
    for _ in range(B):
        w = np.bincount(rng.integers(0, n_items, size=n_items), minlength=n_items).astype(float)
        cur = {m: flip_rates(r, ops, w) for m, r in models.items()}
        for m, v in cur.items():
            for k, x in v.items():
                if x is not None:
                    draws[m][k].append(x)
            if m in deltas:
                for k, x in v.items():
                    y = cur[ref][k]
                    if x is not None and y is not None:
                        deltas[m][k].append(x - y)
    out = {}
    for m, r in models.items():
        pt = flip_rates(r, ops)
        out[m] = {"point": pt, "ci95": {k: pct(v) for k, v in draws[m].items()},
                  "n_valid": {op: sum(1 for x in r if x[1] == op and x[2] is not None) for op in ops},
                  "n_invalid": {op: sum(1 for x in r if x[1] == op and x[2] is None) for op in ops},
                  "flips": {op: sum(1 for x in r if x[1] == op and x[2]) for op in ops}}
        if m in deltas:
            out[m]["paired_minus_" + ref] = {k: {"point": None if pt[k] is None or flip_rates(models[ref], ops)[k] is None
                                                 else pt[k] - flip_rates(models[ref], ops)[k], "ci95": pct(v)}
                                             for k, v in deltas[m].items()}
    return out


def answers_match(a: list[dict], b: list[dict]) -> int:
    ca = {str(r.get("id")): json.dumps(r.get("answer"), sort_keys=True) for r in a}
    cb = {str(r.get("id")): json.dumps(r.get("answer"), sort_keys=True) for r in b}
    return sum(1 for k in ca if k in cb and ca[k] == cb[k])


def g1_check(items, pert_rows: list[dict], base_preds: list[dict], pert_preds: list[dict], max_dp: float = 0.0,
             ops=G1_OPS, B: int = 1000, seed: int = 16, in_job: dict | None = None) -> dict:
    """G1 from files, FAIL CLOSED: every op present with n > 0, 0 invalid, 0 flips, max|dp| <= max_dp, and the on-disk
    perturbation file faithful (every row = the regenerated in-memory order) AND really reordered / relabelled."""
    ops = list(ops)
    rows = [r for r in pert_rows if r["perturbation"]["op"] in ops]
    pert = to_perturbed(rows)
    ba, pa = preds_to_answers(base_preds), preds_to_answers(pert_preds)
    rep = P.flip_report(items, pert, ba, pa)
    per = rep["per_operator"]
    outc = flip_outcomes(items, pert, ba, pa)
    boot = flip_boot({"arm": outc}, ops, len(items), B, seed, None)["arm"]
    fo = file_order_check(items, rows)
    fails, notes = [], []
    for op in ops:
        v, f = per.get(op), fo.get(op)
        if not v or not v.get("n"):
            fails.append(f"{op}: no valid perturbation pairs (vacuous)")
            continue
        if v.get("invalid"):
            fails.append(f"{op}: {v['invalid']} invalid")
        if v.get("flips"):
            fails.append(f"{op}: {v['flips']} flips")
        dp = v.get("max_abs_dp")
        if dp is None or dp > max_dp:
            fails.append(f"{op}: max|dp| {dp} > {max_dp}")
        if f is None or f["matches_regenerated"] != f["n"]:
            fails.append(f"{op}: on-disk perturbed.jsonl lists {0 if f is None else f['n'] - f['matches_regenerated']} rows "
                         "NOT in the in-memory perturbation order (sort_keys dump?)")
        if f is None or f["changed_vs_base"] == 0:
            fails.append(f"{op}: no on-disk row differs from base order / keys (reorder check vacuous)")
        elif f["changed_vs_base"] != f["n"]:
            notes.append(f"{op}: {f['n'] - f['changed_vs_base']} rows keep base order on disk")
        r_flips = sum(1 for x in outc if x[1] == op and x[2])
        if r_flips != v.get("flips"):
            fails.append(f"{op}: own flip count {r_flips} != jevkit flip_report {v.get('flips')}")
    seen = [v.get("max_abs_dp") for o, v in per.items() if o in ops and v.get("max_abs_dp") is not None]
    mx = max(seen) if seen else None
    ok = not fails
    out = {"ops": ops, "n_base_pred": len(base_preds), "n_perturbed_rows": len(rows),
           "per_operator": {op: (per.get(op) or {}) | {"flip_rate_ci95": boot["ci95"].get(op), "file_order": fo.get(op)}
                            for op in ops},
           "max_abs_dp_observed": mx, "max_dp_bound": max_dp, "g1_pass": ok,
           "bitwise": bool(ok and mx == 0.0), "non_bitwise_within_tolerance": bool(ok and mx is not None and mx > 0),
           "g1_failures": fails, "notes": notes,
           "rule": "G1 PASS iff every op n>0, 0 invalid, 0 flips, max|dp| <= bound (0 = bitwise), on-disk rows in the "
                   "in-memory order and really reordered / relabelled"}
    if in_job is not None:
        jp = in_job.get("per_operator") or (in_job.get("flip_report") or {}).get("per_operator") or {}
        out["in_job"] = {"g1_pass": in_job.get("g1_pass"), "max_dp_bound": in_job.get("max_dp_bound"),
                         "rows_all_identical": in_job.get("rows_all_identical"),
                         "row_identity": {k: {x: v.get(x) for x in ("n", "identical")}
                                          for k, v in (in_job.get("row_identity") or {}).items()},
                         "presented_order": in_job.get("presented_order")}
        out["diff_vs_in_job"] = {op: {k: [(per.get(op) or {}).get(k), (jp.get(op) or {}).get(k)]
                                      for k in ("n", "invalid", "flips", "max_abs_dp")} for op in ops}
        out["agrees_with_in_job_counts"] = all(a == b for d in out["diff_vs_in_job"].values() for a, b in d.values())
    return out


# ----------------------------------------------------------------------------------------------- (5) selection
def slice_floor(rows: list[dict], B: int, seed: int) -> dict:
    """Selected checkpoint ('native') vs step 0 on the held-out slice rows: paired bootstrap (iid rows, and stratified
    by family), exact McNemar, per family / type. Same draws as scripts/ev_rescore_x7c42.py."""
    ids_rows = [r["id"] for r in rows]
    nat = np.array([bool(r["native"]) for r in rows])
    s0 = np.array([bool(r["step0"]) for r in rows])
    rfam = np.array([r["family"] for r in rows])
    rtyp = np.array([r["type"] for r in rows])
    n_ = len(rows)
    rng = np.random.default_rng(seed)
    dd = []
    for _ in range(B):
        ix = rng.integers(0, n_, n_)
        dd.append(float(nat[ix].mean() - s0[ix].mean()))
    fi = {f: np.flatnonzero(rfam == f) for f in sorted(set(rfam))}
    ti = {t: np.flatnonzero(rtyp == t) for t in TYPES}
    rng2 = np.random.default_rng(seed)
    fdd, dds = {f: [] for f in fi}, []
    for _ in range(B):
        parts = []
        for f, ii in fi.items():
            s_ = rng2.choice(ii, size=len(ii), replace=True)
            parts.append(s_)
            fdd[f].append(float(nat[s_].mean() - s0[s_].mean()))
        ix = np.concatenate(parts)
        dds.append(float(nat[ix].mean() - s0[ix].mean()))
    rng3 = np.random.default_rng(seed)
    tdd = {t: [] for t in TYPES}
    for _ in range(B):
        for t, ii in ti.items():
            if len(ii):
                s_ = rng3.choice(ii, size=len(ii), replace=True)
                tdd[t].append(float(nat[s_].mean() - s0[s_].mean()))
    b_, c_ = int((nat & ~s0).sum()), int((~nat & s0).sum())
    rec_ids = sorted({i.rsplit("/", 1)[0] for i in ids_rows})
    out = {"n_decisions": n_, "n_unique_ids": len(set(ids_rows)), "n_records": len(rec_ids),
           "record_ids_sha256": sha_b("\n".join(rec_ids).encode()),
           "selected_top1": float(nat.mean()), "step0_top1": float(s0.mean()),
           "delta_pp": 100 * float(nat.mean() - s0.mean()), "ci95_pp_iid_rows": [100 * x for x in pct(dd)],
           "ci95_pp_family_stratified": [100 * x for x in pct(dds)], "p_one_sided_boot": float(np.mean(np.array(dd) <= 0)),
           "selected_only": b_, "step0_only": c_, "both": int((nat & s0).sum()), "neither": int((~nat & ~s0).sum()),
           "mcnemar_exact_p": mcnemar(b_, c_), "pass": bool(nat.mean() > s0.mean() and pct(dd)[0] > 0),
           "per_family": {}, "per_type": {}}
    for f, ii in fi.items():
        bb, cc = int((nat[ii] & ~s0[ii]).sum()), int((~nat[ii] & s0[ii]).sum())
        out["per_family"][f] = {"n": len(ii), "step0": float(s0[ii].mean()), "selected": float(nat[ii].mean()),
                                "delta_pp": 100 * float(nat[ii].mean() - s0[ii].mean()),
                                "ci95_pp": [100 * x for x in pct(fdd[f])], "selected_only": bb, "step0_only": cc,
                                "mcnemar_p": mcnemar(bb, cc)}
    for t, ii in ti.items():
        if len(ii):
            bb, cc = int((nat[ii] & ~s0[ii]).sum()), int((~nat[ii] & s0[ii]).sum())
            out["per_type"][t] = {"n": len(ii), "step0": float(s0[ii].mean()), "selected": float(nat[ii].mean()),
                                  "delta_pp": 100 * float(nat[ii].mean() - s0[ii].mean()),
                                  "ci95_pp": [100 * x for x in pct(tdd[t])], "selected_only": bb, "step0_only": cc,
                                  "mcnemar_p": mcnemar(bb, cc)}
    return out


SEL_KEYS = ("out/cell.json", "dev_cards/", "card-step")
FLOOR_KEYS = ("slice_floor", "calib_probs", "tfit.json")


def order_events(art: Path, mtimes_json: Path | None) -> dict:
    """Timestamps of the selection files vs the first public-231 pred. Source, in order of preference: --mtimes
    (a {name: epoch} map, e.g. a lead landing.json's 'mtimes'), member mtimes of phases/*.tgz, extracted file mtimes."""
    ev: dict[str, float] = {}
    src = None
    if mtimes_json is not None:
        d = json.loads(Path(mtimes_json).read_text(encoding="utf-8"))
        d = d.get("mtimes", d)
        for k, v in d.items():
            ev[("out/screen/jevkit/" + k) if (k.endswith(".preds.jsonl") or k.startswith(FLOOR_KEYS))
               else ("out/" + k if k == "cell.json" else k)] = float(v)
        src = f"--mtimes {mtimes_json}"
    else:
        tgzs = sorted((art / "phases").glob("*.tgz")) if (art / "phases").is_dir() else []
        if tgzs:
            for t in tgzs:
                with tarfile.open(t, "r:gz") as tf:
                    for m in tf.getmembers():
                        if m.isfile():
                            ev[m.name] = min(ev.get(m.name, float("inf")), float(m.mtime))
            src = "phases/*.tgz member mtimes"
        else:
            for p in art.rglob("*"):
                if p.is_file():
                    ev[p.relative_to(art).as_posix()] = p.stat().st_mtime
            src = "extracted file mtimes (valid only for a tar-extracted artifact)"
    pub = {k: v for k, v in ev.items() if k.endswith(".preds.jsonl") and "/jevkit/" in k and "/g1/" not in k}
    sel = {k: v for k, v in ev.items() if any(s in k for s in SEL_KEYS) and k.endswith(".json")}
    flo = {k: v for k, v in ev.items() if any(s in k.rsplit("/", 1)[-1] for s in FLOOR_KEYS)}
    first_pub = min(pub.items(), key=lambda kv: (kv[1], kv[0])) if pub else None
    cell = {k: v for k, v in sel.items() if k.endswith("out/cell.json")}
    sel_t = max(cell.values()) if cell else (max(sel.values()) if sel else None)
    res = {"source": src, "first_public_pred": first_pub[0] if first_pub else None,
           "first_public_pred_utc": utc(first_pub[1]) if first_pub else None,
           "selection_cell_json_utc": utc(sel_t),
           "selection_before_public": None if sel_t is None or first_pub is None else bool(sel_t < first_pub[1]),
           "floor_files": {k: utc(v) for k, v in sorted(flo.items(), key=lambda kv: kv[1])},
           "floor_before_public": None if not flo or first_pub is None else bool(max(flo.values()) < first_pub[1]),
           "public_preds": {k: utc(v) for k, v in sorted(pub.items(), key=lambda kv: kv[1])},
           "selection_files": {k: utc(v) for k, v in sorted(sel.items(), key=lambda kv: kv[1])}}
    return res


def selection(cell: dict | None, ident_child: dict | None, expect_steps: int | None, flags: list, warn: list) -> dict:
    if cell is None:
        flags.append("cell.json missing: selection unverifiable")
        return {}
    cards = [{k: d.get(k) for k in ("step", "nll", "accuracy", "n_items")} for d in cell.get("dev_cards") or []]
    best = cell.get("best") or {}
    am = min(cards, key=lambda d: (d["nll"], d["step"])) if cards else None
    out = {"dev_cards": cards, "best": best, "argmin_nll_step": am["step"] if am else None,
           "best_is_argmin_nll": bool(am and am["step"] == best.get("step")),
           "train_steps": (cell.get("train") or {}).get("steps"), "max_steps": (cell.get("config") or {}).get("max_steps"),
           "dev_card_sample": {k: (cell.get("dev_card_sample") or {}).get(k) for k in ("n", "ids_sha256", "seed")},
           "dev_card_source": cell.get("dev_card_source")}
    if am and not out["best_is_argmin_nll"]:
        flags.append(f"selection: best step {best.get('step')} != argmin dev-card NLL step {am['step']}")
    if expect_steps is not None and out["train_steps"] != expect_steps:
        flags.append(f"cell.json train.steps {out['train_steps']} != expected {expect_steps}")
    if ident_child is None:
        warn.append("identity-child.json missing: the scored checkpoint cannot be tied to the selected step "
                    "(pass --identity-child)")
    else:
        ck = ident_child.get("checkpoint") or {}
        out["scored_checkpoint"] = {k: ck.get(k) for k in ("path", "step", "sha256")}
        out["scored_is_selected"] = bool(ck.get("step") == best.get("step")
                                         and (best.get("path") is None or Path(str(ck.get("path"))).name
                                              == Path(str(best.get("path"))).name))
        if not out["scored_is_selected"]:
            flags.append(f"scored checkpoint {ck.get('path')} (step {ck.get('step')}) != selected {best}")
    return out


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/ev_cell_rescore.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True, help="extracted artifact (or landing) dir: out/screen.json, out/cell.json, "
                                                 "out/screen/jevkit/<arm>.preds.jsonl, slice_floor_rows.jsonl, g1/")
    ap.add_argument("--control", default="stock_shim")
    ap.add_argument("--primary", default="native_shimcal")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--price", default=None, help="declared price (spec or price.json); default = the in-job price")
    ap.add_argument("--expect-steps", type=int, default=None)
    ap.add_argument("--expect-n", type=int, default=231)
    ap.add_argument("--expect-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--slice-ids-sha", default=None, help="pre-registered sha256 of the sorted slice record ids")
    ap.add_argument("--target", action="append", default=[], help="NAME=<preds.jsonl>::<price spec | price.json>")
    ap.add_argument("--holm-arm", action="append", default=[], help="another in-job arm in the Holm family")
    ap.add_argument("--holm-ext", action="append", default=[], help="NAME=P: an external arm's one-sided p in the family")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--g1", action="append", default=[], help="NAME=<dir with base.preds.jsonl, perturbed.preds.jsonl, "
                                                              "perturbed.jsonl>; default: the artifact's jevkit/g1")
    ap.add_argument("--g1-max-dp", type=float, default=0.0, help="declared tolerance; > 0 = NON-BITWISE pass")
    ap.add_argument("--score-inv", default=None, help="dir with base.jsonl, perturbed.jsonl (scripts/ev_score_perturb.py) "
                                                     "+ base.preds.jsonl, perturbed.preds.jsonl: score-item invariance, "
                                                     "separate from G1")
    ap.add_argument("--score-inv-max-dp", type=float, default=0.0, help="declared tolerance; > 0 = NON-BITWISE pass")
    ap.add_argument("--flip8", default=None, help="dir with base.preds.jsonl + perturbed.preds.jsonl [+ perturbed.jsonl]")
    ap.add_argument("--flip-baseline", action="append", default=[], help="NAME=<base.jsonl>,<pert.jsonl> (paired)")
    ap.add_argument("--flip-baseline-receipt", default=None, help="<flip_ci json>::<model> (reported, unpaired)")
    ap.add_argument("--mtimes", default=None)
    ap.add_argument("--identity-child", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(argv)

    art = Path(a.art).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    screen_p = find_art_file(art, "out/screen.json")
    ctrl_p = find_art_file(art, f"out/screen/jevkit/{a.control}.preds.jsonl")
    if screen_p is None or ctrl_p is None:
        print(f"missing: screen.json {screen_p}, control preds {ctrl_p}", file=sys.stderr)
        return 2
    jdir = ctrl_p.parent
    screen = json.loads(screen_p.read_text(encoding="utf-8"))
    sjk = screen.get("jevkit") or {}
    flags: list[str] = []
    warn: list[str] = []

    price = a.price
    if price is None:
        if not sjk.get("price"):
            print("no --price and no in-job jevkit.price", file=sys.stderr)
            return 2
        price = str(out / "price.in-job.json")
        Path(price).write_text(json.dumps({"price": sjk["price"]}, indent=1), encoding="utf-8")
        warn.append("price = the in-job declared price (screen.json jevkit.price); pass --price to declare it")

    # ---- (1)(2)(3)(4): the frozen-CLI rescore + screen.json diff (scripts/ev_rescore_jevkit.py)
    core_p = out / "core.rescore.json"
    argv_core = ["--art", str(art), "--out", str(core_p), "--work", str(out / "reports"), "--price", price,
                 "--baseline", a.control, "--primary", a.primary, "--expect-n", str(a.expect_n),
                 "--expect-model", a.expect_model, "--B", str(a.B), "--seed", str(a.seed), "--tol", str(a.tol),
                 "--secondary-axes"]
    if a.expect_steps is not None:
        argv_core += ["--expect-steps", str(a.expect_steps)]
    if a.groups:
        argv_core += ["--groups", a.groups]
    for t in a.target:
        argv_core += ["--vs", t]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_core = R.main(argv_core)
    (out / "core.stdout.txt").write_text(buf.getvalue(), encoding="utf-8")
    if rc_core == 2:
        print("core rescore: inputs missing", file=sys.stderr)
        return 2
    core = json.loads(core_p.read_text(encoding="utf-8"))
    flags += [f"core: {f}" for f in core["flags"]]

    items = load_public()
    arms = sorted(core["arms"])
    preds = {n: rj(jdir / f"{n}.preds.jsonl") for n in arms}
    T = {n: JS.build_table(items, preds_to_answers(preds[n])) for n in arms}
    SK = {n: (core["arms"][n]["S"], core["arms"][n]["K"]) for n in arms}
    if a.primary not in T:
        print(f"primary arm {a.primary!r} has no preds in {jdir}", file=sys.stderr)
        return 2
    fam = np.array([it.family for it in items])
    typ = np.array([it.type for it in items])
    fams = sorted(set(fam))
    tops = {n: top1(T[n], items) for n in arms}
    base, prim = a.control, a.primary
    others = [n for n in arms if n != base]

    def axes(n, idx=None):
        r = JS.aggregate(T[n], idx, detail=False)
        i, c = r["intelligence"]["I"], r["calibration"]["score"]
        return i, c, JS.composite(i, c, SK[n][0], SK[n][1], "A")

    # ---- the shared paired draws (default_rng(seed), strata of the control table): identical to ev_rescore_jevkit.paired
    rng = np.random.default_rng(a.seed)
    st = JS.strata(T[base])
    dA = {n: [] for n in others}
    dI = {n: [] for n in others}
    dcc = {t: [] for t in TYPES}
    dtypT = {t: [] for t in TYPES}
    dfamT = {f: [] for f in fams}
    dfamI = {f: [] for f in fams}
    dfamC = {f: [] for f in fams}
    for _ in range(a.B):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        ib, cb, Ab = axes(base, idx)
        for n in others:
            ia, ca, Aa = axes(n, idx)
            dA[n].append(Aa - Ab)
            dI[n].append(ia - ib)
        ccb, cca = type_cc(T[base], idx), type_cc(T[prim], idx)
        for t in TYPES:
            if cca.get(t) is not None and ccb.get(t) is not None:
                dcc[t].append(cca[t] - ccb[t])
            m = idx[typ[idx] == t]
            if len(m):
                dtypT[t].append(float(tops[prim][m].mean() - tops[base][m].mean()))
        for f in fams:
            m = idx[fam[idx] == f]
            if not len(m):
                continue
            dfamT[f].append(float(tops[prim][m].mean() - tops[base][m].mean()))
            dfamC[f].append(float(T[prim].correct[m].mean() - T[base].correct[m].mean()))
            ra = JS.aggregate(T[prim], m, detail=False)["intelligence"]["I"]
            rb = JS.aggregate(T[base], m, detail=False)["intelligence"]["I"]
            if ra is not None and rb is not None:
                dfamI[f].append(ra - rb)
    full = {n: axes(n) for n in arms}

    def prim_block(n):
        v = np.array(dA[n])
        return {"arm": n, "control": base, "point": full[n][2] - full[base][2], "a": full[n][2], "b": full[base][2],
                "ci95": pct(v), "ci975": pct(v, (1.25, 98.75)), "ci99": pct(v, (0.5, 99.5)),
                "p_one_sided_boot": float(np.mean(v <= 0)), "p_one_sided_boot_plus1": float((1 + np.sum(v <= 0)) / (a.B + 1)),
                "quantiles": {str(q): float(np.percentile(v, q)) for q in QS},
                "dI_point": full[n][0] - full[base][0], "dI_ci95": pct(dI[n])}

    primary = prim_block(prim)
    core_pp = (core.get("pairs") or {}).get(prim, {}).get("composite_A", {})
    if core_pp.get("ci95") != primary["ci95"] or core_pp.get("point") != primary["point"]:
        flags.append(f"primary: own draw loop {primary['point']} {primary['ci95']} != core paired {core_pp}")
    fam_ps = {prim: primary["p_one_sided_boot"]}
    hblocks = {prim: primary}
    for n in a.holm_arm:
        if n not in dA:
            flags.append(f"--holm-arm {n}: no such arm")
            continue
        hblocks[n] = prim_block(n)
        fam_ps[n] = hblocks[n]["p_one_sided_boot"]
    for spec_ in a.holm_ext:
        nm, p_ = spec_.split("=", 1)
        fam_ps[nm] = float(p_)
    H = holm(fam_ps, a.alpha)
    for n, h in H.items():
        if n in hblocks:
            v = np.array(dA[n])
            lv = h["level"]
            h["ci_at_level"] = pct(v, (100 * lv / 2, 100 * (1 - lv / 2)))
            h["claimable"] = bool(h["pass_p"] and h["ci_at_level"][0] > 0)
        else:
            h["claimable"] = bool(h["pass_p"])
    primary["holm"] = {"family": sorted(fam_ps), "alpha": a.alpha, "per_arm": H,
                       "rule": "Holm step-down on one-sided bootstrap p (P(delta <= 0)); an in-job arm is claimable iff "
                               "p < its level AND its two-sided CI at 1 - level excludes 0 on the positive side",
                       "primary_claimable": H[prim]["claimable"]}
    primary["success_95"] = bool(primary["point"] > 0 and primary["ci95"][0] > 0)
    sprim = sjk.get("primary") or {}
    primary["in_job"] = {"point": (sprim.get("delta_composite_A") or {}).get("point"),
                         "ci95": (sprim.get("delta_composite_A") or {}).get("ci95"), "success": sprim.get("success"),
                         "slice_floor_pass": sprim.get("slice_floor_pass")}
    primary["screen_diff"] = {"n_fields": len(core["diffs_vs_screen"]), "max_abs_diff": core["max_abs_diff"],
                              "verdict": core["verdict"]}
    fcc = {n: type_cc(T[n]) for n in (prim, base)}
    per_type = {t: {"cc": {"a": fcc[prim].get(t), "b": fcc[base].get(t),
                           "delta": None if fcc[prim].get(t) is None else fcc[prim][t] - fcc[base][t], "ci95": pct(dcc[t])},
                    "top1": {"n": int((typ == t).sum()), "a": int(tops[prim][typ == t].sum()), "b": int(tops[base][typ == t].sum()),
                             "delta_pp": 100 * float(tops[prim][typ == t].mean() - tops[base][typ == t].mean()),
                             "ci95_pp": [100 * x for x in (pct(dtypT[t]) or [float("nan")] * 2)],
                             "a_only": int((tops[prim] & ~tops[base] & (typ == t)).sum()),
                             "b_only": int((~tops[prim] & tops[base] & (typ == t)).sum())}} for t in TYPES}
    for t in TYPES:
        d = per_type[t]["top1"]
        d["mcnemar_p"] = mcnemar(d["a_only"], d["b_only"])
    nm_ = typ == "noul"
    band = {n: {"n_noul": int(nm_.sum()), "in_band_0.2_0.8": int((T[n].abstain & nm_).sum())} for n in arms}
    for n in arms:
        ib = ((sjk.get("noul_band") or {}).get(n) or {}).get("in_band")
        band[n]["in_job"] = ib
        if ib is not None and ib != band[n]["in_band_0.2_0.8"]:
            flags.append(f"noul band {n}: {band[n]['in_band_0.2_0.8']} vs in-job {ib}")
    secondary = {"ev_prereg_secondary_axes": core.get("secondary_axes"), "per_type": per_type, "noul_band": band,
                 "top1_public": {n: int(tops[n].sum()) for n in arms},
                 "all_arms_vs_control": {n: {k: core["pairs"][n][k] for k in ("I", "C", "capability", "composite_A")}
                                         for n in others}}

    # ---- (3) target gap
    target = {}
    for spec_ in a.target:
        nm, rest = spec_.split("=", 1)
        pf = Path(rest.split("::", 1)[0]).resolve()
        ext = (core.get("external") or {}).get(nm) or {}
        tp = rj(pf)
        tt = JS.build_table(items, preds_to_answers(tp))
        tsk = (ext.get("S"), ext.get("K"))
        cpair = R.paired({nm: tt, base: T[base]}, {nm: tsk, base: SK[base]}, nm, a.B, a.seed)[base]
        t_top = top1(tt, items)
        pr = ext.get(f"paired_{prim}_minus_{nm}") or {}
        target[nm] = {"preds": str(pf), "preds_sha256": ext.get("preds_sha256"), "price": ext.get("price"),
                      "axes": {k: ext.get(k) for k in ("I", "C", "S", "K", "capability")} | {"A": (ext.get("composite") or {}).get("A")},
                      f"{prim}_minus_{nm}": {k: pr.get(k) for k in ("I", "C", "capability", "composite_A")},
                      f"{base}_minus_{nm}": {k: cpair.get(k) for k in ("I", "C", "capability", "composite_A")},
                      "S_K_held": {"primary": SK[prim], "control": SK[base], nm: tsk},
                      "caveat": CROSS_JOB,
                      f"{base}_vs_{nm}_identity": {"answers_bitwise_equal": answers_match(preds[base], tp),
                                                    "top1_equal_items": int((t_top == tops[base]).sum()),
                                                    "top1": {nm: int(t_top.sum()), base: int(tops[base].sum())}},
                      "secondary_axes": (core.get("secondary_axes") or {}).get(f"{prim} - {nm}")}

    # ---- (4) integrity
    cell_p = find_art_file(art, "out/cell.json")
    cell = json.loads(cell_p.read_text(encoding="utf-8")) if cell_p else None
    cap = None
    if cell:
        cap = (cell.get("config") or {}).get("eval_max_t") or (cell.get("train") or {}).get("T_max")
    tok_max = {}
    for n in arms:
        t = [(p.get("usage") or {}).get("input_tokens") for p in preds[n]]
        t = [x for x in t if isinstance(x, (int, float))]
        tok_max[n] = {"max_input_tokens": max(t) if t else None, "n_with_tokens": len(t),
                      "n_over_cap": None if cap is None else sum(1 for x in t if x > cap)}
        if cap is not None and tok_max[n]["n_over_cap"]:
            flags.append(f"truncation: {n} has {tok_max[n]['n_over_cap']} items with input_tokens > {cap}")
        if len(t) != len(preds[n]):
            warn.append(f"{n}: {len(preds[n]) - len(t)} preds without usage.input_tokens (truncation unverifiable)")
    ntt = ((cell or {}).get("data_checks") or {}).get("no_truncation_tokenize")
    if ntt and ntt.get("max_row_tokens") is not None and ntt.get("t_cap") is not None and ntt["max_row_tokens"] > ntt["t_cap"]:
        flags.append(f"train truncation: max_row_tokens {ntt['max_row_tokens']} > t_cap {ntt['t_cap']}")
    if cap is None:
        warn.append("no eval token cap in cell.json (config.eval_max_t / train.T_max): eval truncation unverifiable")
    integrity = {"n_items": len(items), "same_item_ids_all_arms": core["same_item_ids_all_arms"],
                 "per_arm": {n: core["arms"][n]["checks"] for n in arms}, "eval_token_cap": cap,
                 "eval_tokens": tok_max, "train_no_truncation_tokenize": ntt, "price": core["price"],
                 "cell": core["cell"]}

    # ---- (5) selection + floor + order
    ic_p = Path(a.identity_child) if a.identity_child else find_art_file(art, "out/screen/identity-child.json")
    ident = json.loads(ic_p.read_text(encoding="utf-8")) if ic_p and ic_p.is_file() else None
    sel = selection(cell, ident, a.expect_steps, flags, warn)
    rows_p = jdir / "slice_floor_rows.jsonl"
    floor = None
    if rows_p.is_file():
        floor = slice_floor(rj(rows_p), a.B, a.seed)
        sjf = sjk.get("slice_floor") or {}
        floor["in_job"] = {k: sjf.get(k) for k in ("ids_sha256", "n_decisions", "native_top1", "step0_top1", "delta", "pass")}
        d = [abs(floor["selected_top1"] - sjf["native_top1"]) if sjf.get("native_top1") is not None else None,
             abs(floor["step0_top1"] - sjf["step0_top1"]) if sjf.get("step0_top1") is not None else None]
        d += [abs(floor["per_family"][f]["delta_pp"] / 100 - v["delta"]) for f, v in (sjf.get("per_family") or {}).items()
              if f in floor["per_family"] and v.get("delta") is not None]
        floor["max_abs_diff_vs_in_job"] = max((x for x in d if x is not None), default=None)
        if floor["max_abs_diff_vs_in_job"] is not None and floor["max_abs_diff_vs_in_job"] > 1e-9:
            flags.append(f"slice floor differs from in-job by {floor['max_abs_diff_vs_in_job']}")
        if sjf.get("pass") is not None and bool(sjf["pass"]) != (floor["selected_top1"] > floor["step0_top1"]):
            flags.append("slice floor pass/fail differs from in-job")
        for nm, want in (("--slice-ids-sha", a.slice_ids_sha), ("in-job slice_floor.ids_sha256", sjf.get("ids_sha256")),
                         ("cell.dev_card_source.ids_sha256", ((cell or {}).get("dev_card_source") or {}).get("ids_sha256"))):
            if want and want != floor["record_ids_sha256"]:
                flags.append(f"slice ids sha {floor['record_ids_sha256'][:12]} != {nm} {want[:12]}")
        if not floor["pass"]:
            flags.append(f"step-0 floor NOT beaten: {floor['delta_pp']:+.2f} pp {floor['ci95_pp_iid_rows']}")
    else:
        flags.append("slice_floor_rows.jsonl missing: the step-0 floor cannot be recomputed")
    order = order_events(art, Path(a.mtimes) if a.mtimes else None)
    if order["selection_before_public"] is False:
        flags.append(f"selection (cell.json {order['selection_cell_json_utc']}) AFTER the first public pred "
                     f"({order['first_public_pred_utc']})")
    if order["selection_before_public"] is None:
        warn.append("order of events unavailable (no cell.json or public pred timestamps)")
    sel_block = {"cell": sel, "slice_floor": floor, "order": order}

    # ---- (6) G1
    g1 = {}
    g1_specs = list(a.g1)
    if not g1_specs and (jdir / "g1" / "perturbed.preds.jsonl").is_file():
        g1_specs = [f"in-job={jdir / 'g1'}"]
    elif not g1_specs and (jdir / "g1").is_dir():  # one sub-dir per arm (g1/<arm>/...), as job-4g6m8 wrote it
        g1_specs = [f"{d.name}={d}" for d in sorted((jdir / "g1").iterdir()) if (d / "perturbed.preds.jsonl").is_file()]
    for spec_ in g1_specs:
        nm, d = spec_.split("=", 1)
        d = Path(d)
        need = [d / "base.preds.jsonl", d / "perturbed.preds.jsonl", d / "perturbed.jsonl"]
        if not all(p.is_file() for p in need):
            flags.append(f"G1 {nm}: missing {[p.name for p in need if not p.is_file()]}")
            continue
        rep_p = d / "perturb_report.json"
        inj = json.loads(rep_p.read_text(encoding="utf-8")) if rep_p.is_file() else None
        bp = rj(need[0])
        res = g1_check(items, rj(need[2]), bp, rj(need[1]), a.g1_max_dp, G1_OPS, a.B, a.seed, inj)
        res["base_preds_equal_arm"] = {n: answers_match(bp, preds[n]) for n in arms}
        res["dir"] = str(d)
        g1[nm] = res
        if res["non_bitwise_within_tolerance"]:
            warn.append(f"G1 {nm}: PASS only within the declared tolerance {a.g1_max_dp} (NON-BITWISE)")
        if inj is not None and not res.get("agrees_with_in_job_counts"):
            flags.append(f"G1 {nm}: recomputed n / invalid / flips / max|dp| differ from the in-job perturb_report")
    if not g1_specs:
        warn.append("no G1 dir (jevkit/g1 or --g1): G1 not evaluated")

    # ---- (6b) score-item invariance (separate from G1; never touches g1 / its verdict; FAIL -> warning only)
    score_inv = None
    if a.score_inv:
        score_inv = SF.run_dir(items, Path(a.score_inv), max_dp=a.score_inv_max_dp)
        score_inv["dir"] = str(Path(a.score_inv).resolve())
        if "missing" in score_inv:
            warn.append(f"score-inv: inputs missing {score_inv['missing']}: score-item invariance NOT evaluated")
        elif not score_inv["score_inv_pass"]:
            warn.append(f"score-inv FAIL (separate from G1): {'; '.join(score_inv['failures'][:5])}")
        elif score_inv["non_bitwise_within_tolerance"]:
            warn.append(f"score-inv: PASS only within the declared tolerance {a.score_inv_max_dp} (NON-BITWISE)")

    # ---- (7) 8-op flip battery
    flips = None
    f8 = a.flip8 if a.flip8 else (str(jdir / "g1") if (jdir / "g1" / "perturbed.preds.jsonl").is_file() else None)
    if f8 is not None:
        if "," in f8:  # <base.jsonl>,<pert.jsonl>: perturbations regenerated from the perturbed ids
            fb_, fp_ = f8.split(",", 1)
            bp, pp, pfile = rj(Path(fb_)), rj(Path(fp_)), Path("/nonexistent/perturbed.jsonl")
        else:
            f8 = Path(f8)
            bp, pp, pfile = rj(f8 / "base.preds.jsonl"), rj(f8 / "perturbed.preds.jsonl"), f8 / "perturbed.jsonl"
        pert = to_perturbed(rj(pfile)) if pfile.is_file() else regenerate(items, [r["id"] for r in pp])
        ops = [o for o in P.OPERATORS if any(p.op == o for p in pert)]
        models = {"cell": flip_outcomes(items, pert, preds_to_answers(bp), preds_to_answers(pp))}
        ref = None
        for spec_ in a.flip_baseline:
            nm, files = spec_.split("=", 1)
            b_, p_ = files.split(",")
            bpp, ppp = rj(Path(b_)), rj(Path(p_))
            bpert = regenerate(items, [r["id"] for r in ppp if r["id"].rsplit("~", 2)[1] in ops])
            models[nm] = flip_outcomes(items, bpert, preds_to_answers(bpp), preds_to_answers(ppp))
            ref = ref or nm
        fb = flip_boot(models, ops, len(items), a.B, a.seed, ref)
        jrep = P.flip_report(items, pert, preds_to_answers(bp), preds_to_answers(pp))
        for op in ops:
            if abs((jrep["per_operator"][op]["flip_rate"] or 0) - (fb["cell"]["point"][op] or 0)) > 1e-12:
                flags.append(f"flips {op}: own rate != jevkit flip_report")
        flips = {"dir": str(f8), "ops": ops, "all_8_ops": len(ops) == 8, "models": fb, "baseline_paired": ref}
        if len(ops) < 8:
            warn.append(f"flip battery: only {len(ops)}/8 ops in {f8} ({ops}); the average is over these ops only and "
                        "is NOT comparable to an 8-op average")
        if a.flip_baseline_receipt:
            rp, mname = a.flip_baseline_receipt.split("::", 1)
            rr = json.loads(Path(rp).read_text(encoding="utf-8"))["models"][mname]
            flips["baseline_reported"] = {"receipt": rp, "model": mname, "average": rr["point"]["_average"],
                                          "average_ci95": rr["ci95"]["_average"],
                                          "per_op": {o: rr["point"].get(o) for o in P.OPERATORS},
                                          "comparable": len(ops) == 8}

    # ---- (8) transfer
    transfer = {}
    for f in sorted(set(fams) | set((floor or {}).get("per_family", {}))):
        row = {"slice": (floor or {}).get("per_family", {}).get(f), "public": None}
        if f in fams:
            m = fam == f
            b_, c_ = int((tops[prim] & ~tops[base] & m).sum()), int((~tops[prim] & tops[base] & m).sum())
            ia = JS.aggregate(T[prim], np.flatnonzero(m), detail=False)["intelligence"]["I"]
            ib = JS.aggregate(T[base], np.flatnonzero(m), detail=False)["intelligence"]["I"]
            row["public"] = {"n": int(m.sum()), "control_top1": float(tops[base][m].mean()),
                             "arm_top1": float(tops[prim][m].mean()),
                             "delta_top1_pp": 100 * float(tops[prim][m].mean() - tops[base][m].mean()),
                             "delta_top1_ci95_pp": [100 * x for x in (pct(dfamT[f]) or [float("nan")] * 2)],
                             "delta_correct_pp": 100 * float(T[prim].correct[m].mean() - T[base].correct[m].mean()),
                             "delta_correct_ci95_pp": [100 * x for x in (pct(dfamC[f]) or [float("nan")] * 2)],
                             "a_only": b_, "b_only": c_, "mcnemar_p": mcnemar(b_, c_),
                             "delta_I": None if ia is None or ib is None else ia - ib, "delta_I_ci95": pct(dfamI[f])}
            bf = (sjk.get("by_family") or {}).get(f)
            if bf and (bf.get("delta_I") or {}).get("point") is not None and row["public"]["delta_I"] is not None:
                d = abs(bf["delta_I"]["point"] - row["public"]["delta_I"])
                row["public"]["delta_I_in_job_abs_diff"] = d
                if d > a.tol and bf.get("arm") in (None, prim):
                    flags.append(f"by_family {f}: delta I {row['public']['delta_I']:.4f} vs in-job {bf['delta_I']['point']:.4f}")
        s_ok = row["slice"] is not None and row["slice"]["ci95_pp"][0] > 0
        row["flag"] = ("GAIN SLICE ONLY" if s_ok and row["public"] and row["public"]["delta_top1_pp"] <= 0 else
                       "gain on both" if s_ok and row["public"] else
                       "public loss" if row["public"] and row["public"]["delta_top1_pp"] < 0 else
                       "not on public" if row["public"] is None else "not in the slice" if row["slice"] is None else "")
        transfer[f] = row
    both = [f for f, r in transfer.items() if r["slice"] and r["public"]]

    def corr(key):
        x = np.array([transfer[f]["slice"]["delta_pp"] for f in both])
        y = np.array([transfer[f]["public"][key] for f in both])
        return float(np.corrcoef(x, y)[0, 1]) if len(both) > 2 and x.std() > 0 and y.std() > 0 else None
    transfer_block = {"by_family": transfer, "n_families_both": len(both),
                      "corr_slice_vs_public_delta_correct": corr("delta_correct_pp"),
                      "corr_slice_vs_public_delta_top1": corr("delta_top1_pp"),
                      "by_type": {t: {"slice": (floor or {}).get("per_type", {}).get(t), "public": per_type[t]} for t in TYPES}}

    res = {"schema": "ev-cell-rescore/1", "artifact": str(art), "screen_json_sha256": R.sha(screen_p),
           "rescore_checkout": str(ROOT), "control": base, "primary": prim, "B": a.B, "seed": a.seed,
           "protocol": core["protocol"], "arms": {n: {k: core["arms"][n][k] for k in ("I", "C", "S", "K", "capability")}
                                                  | {"A": core["arms"][n]["composite"]["A"],
                                                     "preds_sha256": core["arms"][n]["preds_sha256"]} for n in arms},
           "1_primary": primary, "2_secondary": secondary, "3_target": target or None, "4_integrity": integrity,
           "5_selection": sel_block, "6_g1": g1 or None, "6b_score_inv": score_inv, "7_flips": flips, "8_transfer": transfer_block,
           "screen_diffs": core["diffs_vs_screen"], "flags": flags, "warnings": warn,
           "verdict": "AGREES" if not flags else "FLAGGED"}
    (out / "cell_rescore.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "CELL_RESCORE.md").write_text(render_md(res), encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"], "flags": flags, "warnings": warn,
                      "primary": {k: primary[k] for k in ("point", "ci95", "ci975", "p_one_sided_boot")}
                      | {"holm_claimable": primary["holm"]["primary_claimable"]},
                      "g1": {k: {x: v[x] for x in ("g1_pass", "bitwise", "max_abs_dp_observed")} for k, v in g1.items()},
                      "score_inv_separate_from_g1": None if not score_inv or "missing" in score_inv else
                      {x: score_inv[x] for x in ("score_inv_pass", "bitwise", "max_abs_dp_observed")},
                      "out": str(out)}, indent=1, default=str))
    return 0 if not flags else 3


# ----------------------------------------------------------------------------------------------- markdown
def _f(x, d=2):
    return "–" if x is None else (f"{x:+.{d}f}" if isinstance(x, float) else str(x))


def _ci(c, d=2):
    return "–" if not c else f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def render_md(r: dict) -> str:
    p = r["1_primary"]
    L = [f"# Cell rescore: {r['primary']} vs {r['control']} ({r['verdict']})", "",
         f"Artifact `{r['artifact']}` (screen.json sha {r['screen_json_sha256'][:12]}). {r['protocol']}.", ""]
    L += ["## (1) PRIMARY", "| quantity | value |", "|---|---|",
          f"| Comp A primary / control | {p['a']:.3f} / {p['b']:.3f} |",
          f"| delta Comp A [95%] | {p['point']:+.4f} {_ci(p['ci95'], 4)} |",
          f"| 97.5% / 99% | {_ci(p['ci975'], 4)} / {_ci(p['ci99'], 4)} |",
          f"| one-sided p (P(delta<=0)) | {p['p_one_sided_boot']:.4f} (+1: {p['p_one_sided_boot_plus1']:.4f}) |",
          f"| delta I [95%] | {p['dI_point']:+.3f} {_ci(p['dI_ci95'])} |",
          f"| success at 95% | {p['success_95']} |",
          f"| in-job | {_f(p['in_job']['point'], 4)} {_ci(p['in_job']['ci95'], 4)} success {p['in_job']['success']} |",
          f"| screen.json diff | {p['screen_diff']['n_fields']} fields, max abs diff {p['screen_diff']['max_abs_diff']}, "
          f"{p['screen_diff']['verdict']} |", ""]
    L += ["Holm family " + ", ".join(p["holm"]["family"]) + f" (alpha {p['holm']['alpha']}):", ""]
    for n, h in p["holm"]["per_arm"].items():
        L.append(f"- {n}: p {h['p']:.4f}, rank {h['rank']}, level {h['level']:.4f}, tested {h['tested']}, "
                 f"CI at level {_ci(h.get('ci_at_level'), 3)}, claimable **{h['claimable']}**")
    s = r["2_secondary"]
    sa = (s["ev_prereg_secondary_axes"] or {}).get(f"{r['primary']} - {r['control']}") or {}
    L += ["", "## (2) SECONDARY (EV-PREREG-SECONDARY-AXES; never turns a failed primary into a success)"]
    for k, v in (sa.get("per_axis") or {}).items():
        L.append(f"- delta {k}: {_f(v['point'])} {_ci(v['ci95'])}{' (held fixed)' if v.get('held_fixed') else ''}")
    if sa:
        t = sa["top1"]
        L.append(f"- top-1 (jevkit correct) {t['a_count']} vs {t['b_count']}: {t['delta_acc_pp']:+.2f} pp "
                 f"{_ci(t['delta_acc_pp_ci95'])}, {t['a_only']} / {t['b_only']} discordant, McNemar p {t['mcnemar_exact_p']:.3g}")
        L.append(f"- choice NLL {sa['choice_nll']['a_mean']:.4f} vs {sa['choice_nll']['b_mean']:.4f} "
                 f"({sa['choice_nll']['delta']:+.4f} {_ci(sa['choice_nll']['delta_ci95'], 4)}); I<50 in "
                 f"{sa['frac_resamples_I_below_50']}")
    for ty, v in s["per_type"].items():
        L.append(f"- {ty}: cc {_f(v['cc']['delta'])} {_ci(v['cc']['ci95'])}; top-1 {v['top1']['delta_pp']:+.2f} pp "
                 f"{_ci(v['top1']['ci95_pp'])} ({v['top1']['a_only']}/{v['top1']['b_only']}, McN {v['top1']['mcnemar_p']:.3g})")
    L.append("- noul 0.2-0.8 band: " + ", ".join(f"{n} {v['in_band_0.2_0.8']}/{v['n_noul']}" for n, v in s["noul_band"].items()))
    L += ["", "## (3) TARGET gap (paired on ids; S/K at each system's own value; " + CROSS_JOB + ")",
          "| target | delta Comp A [95%] (primary - target) | control - target | S prim / S target | K prim / K target | caveat |",
          "|---|---|---|---|---|---|"]
    for n, t in (r["3_target"] or {}).items():
        pa, ca = t[f"{r['primary']}_minus_{n}"]["composite_A"] or {}, t[f"{r['control']}_minus_{n}"]["composite_A"] or {}
        sk = t["S_K_held"]
        L.append(f"| {n} | {_f(pa.get('point'))} {_ci(pa.get('ci95'))} | {_f(ca.get('point'))} {_ci(ca.get('ci95'))} | "
                 f"{sk['primary'][0]:.2f} / {sk[n][0]:.2f} | {sk['primary'][1]:.2f} / {sk[n][1]:.2f} | cross-job |")
    i = r["4_integrity"]
    L += ["", "## (4) INTEGRITY", f"- items {i['n_items']}, same ids all arms {i['same_item_ids_all_arms']}"]
    for n, c in i["per_arm"].items():
        L.append(f"- {n}: {c['n_records']} records / {c['n_unique_ids']} unique, dup {c['n_duplicated']}, missing "
                 f"{len(c['missing_ids'])}, option order off {c['n_order_not_criteria_order']}, status {c['status']}, "
                 f"max input tokens {i['eval_tokens'][n]['max_input_tokens']} (cap {i['eval_token_cap']})")
    L.append(f"- train no-truncation: {i['train_no_truncation_tokenize']}; cell {i['cell']}")
    se = r["5_selection"]
    c = se["cell"]
    L += ["", "## (5) SELECTION", f"- best {c.get('best')}; argmin NLL step {c.get('argmin_nll_step')} "
          f"(= best: {c.get('best_is_argmin_nll')}); trained steps {c.get('train_steps')} / max {c.get('max_steps')}; "
          f"scored {c.get('scored_checkpoint')} (= selected: {c.get('scored_is_selected')})",
          "| step | NLL | top-1 |", "|---|---|---|"]
    L += [f"| {d['step']} | {d['nll']:.6f} | {d['accuracy']:.4f} |" for d in c.get("dev_cards") or []]
    fl = se["slice_floor"]
    if fl:
        L.append(f"- step-0 floor: {fl['selected_top1']:.4f} vs {fl['step0_top1']:.4f} ({fl['n_decisions']} decisions), "
                 f"delta {fl['delta_pp']:+.2f} pp {_ci(fl['ci95_pp_iid_rows'])} iid / {_ci(fl['ci95_pp_family_stratified'])} "
                 f"family-stratified, McNemar p {fl['mcnemar_exact_p']:.3g} ({fl['selected_only']} vs {fl['step0_only']}); "
                 f"slice ids sha {fl['record_ids_sha256'][:12]}; in-job diff {fl['max_abs_diff_vs_in_job']}; pass {fl['pass']}")
    o = se["order"]
    L.append(f"- order ({o['source']}): selection cell.json {o['selection_cell_json_utc']} vs first public pred "
             f"{o['first_public_pred_utc']} ({o['first_public_pred']}): selection before public **{o['selection_before_public']}**; "
             f"floor files before public {o['floor_before_public']}")
    L += ["", "## (6) G1 (reverse / shuffle / relabel; from files)"]
    if not r["6_g1"]:
        L.append("- not evaluated (no jevkit/g1 dir and no --g1)")
    for n, g in (r["6_g1"] or {}).items():
        L.append(f"- **{n}: G1 {'PASS' if g['g1_pass'] else 'FAIL'}** (bitwise {g['bitwise']}, max|dp| "
                 f"{g['max_abs_dp_observed']}, bound {g['max_dp_bound']}); base preds = arm: {g['base_preds_equal_arm']}")
        for op, v in g["per_operator"].items():
            fo = v.get("file_order") or {}
            L.append(f"  - {op}: n {v.get('n')}, invalid {v.get('invalid')}, flips {v.get('flips')} "
                     f"({_ci([100 * x for x in v['flip_rate_ci95']] if v.get('flip_rate_ci95') else None, 1)} %), max|dp| "
                     f"{v.get('max_abs_dp')}; on disk: {fo.get('matches_regenerated')}/{fo.get('n')} rows in in-memory order, "
                     f"{fo.get('changed_vs_base')}/{fo.get('n')} changed vs base")
        for f_ in g["g1_failures"]:
            L.append(f"  - FAIL: {f_}")
        if "in_job" in g:
            L.append(f"  - in-job report: g1_pass {g['in_job']['g1_pass']}, counts agree {g['agrees_with_in_job_counts']}")
    si = r.get("6b_score_inv")
    L += ["", "## (6b) Score-item invariance (separate from G1; does not change the G1 verdict)"]
    if not si:
        L.append("- not evaluated (no --score-inv)")
    elif "missing" in si:
        L.append(f"- NOT evaluated: inputs missing {si['missing']}")
    else:
        L += SF.render_md(si).splitlines()[2:]
        L.append(f"- **score-inv {'PASS' if si['score_inv_pass'] else 'FAIL'}** (dir `{si['dir']}`)")
    f7 = r["7_flips"]
    L += ["", "## (7) FLIP battery"]
    if not f7:
        L.append("- not evaluated (no --flip8 and no jevkit/g1 dir)")
    if f7:
        L.append(f"- ops ({len(f7['ops'])}/8): {', '.join(f7['ops'])}")
        for n, m in f7["models"].items():
            av = m["point"]["_average"]
            L.append(f"- {n}: average {100 * av:.2f}% {_ci([100 * x for x in m['ci95']['_average']] if m['ci95']['_average'] else None)}; "
                     + ", ".join(f"{o} {100 * m['point'][o]:.1f}%" for o in f7["ops"] if m["point"][o] is not None))
            pm = m.get("paired_minus_" + str(f7.get("baseline_paired")))
            if pm:
                L.append(f"  - paired minus {f7['baseline_paired']}: average {100 * pm['_average']['point']:+.2f} pp "
                         f"{_ci([100 * x for x in pm['_average']['ci95']] if pm['_average']['ci95'] else None)}")
        if f7.get("baseline_reported"):
            b = f7["baseline_reported"]
            L.append(f"- reported baseline {b['model']}: {100 * b['average']:.2f}% {_ci([100 * x for x in b['average_ci95']])} "
                     f"(comparable to this cell's average: {b['comparable']})")
    tr = r["8_transfer"]
    L += ["", "## (8) TRANSFER (slice = selected - step 0; public = primary - control)",
          f"corr(slice delta, public delta correct) {_f(tr['corr_slice_vs_public_delta_correct'], 3)}; "
          f"corr(slice, public top-1) {_f(tr['corr_slice_vs_public_delta_top1'], 3)}", "",
          "| family | slice step-0 | slice delta pp [CI] (McN) | public n | control top-1 | public delta top-1 pp (+/-) | public delta I | flag |",
          "|---|---|---|---|---|---|---|---|"]
    for f_, row in tr["by_family"].items():
        sl, pu = row["slice"], row["public"]
        c1 = "" if not sl else "%.1f" % (100 * sl["step0"])
        c2 = "" if not sl else "%+.1f %s (%.2g)" % (sl["delta_pp"], _ci(sl["ci95_pp"], 1), sl["mcnemar_p"])
        c3 = "" if not pu else str(pu["n"])
        c4 = "" if not pu else "%.1f" % (100 * pu["control_top1"])
        c5 = "" if not pu else "%+.1f (%d/%d)" % (pu["delta_top1_pp"], pu["a_only"], pu["b_only"])
        c6 = "" if not pu else _f(pu["delta_I"], 1)
        L.append(f"| {f_} | {c1} | {c2} | {c3} | {c4} | {c5} | {c6} | {row['flag']} |")
    L += ["", "## Flags", *(f"- {x}" for x in r["flags"] or ["none"]), "", "## Warnings", *(f"- {x}" for x in r["warnings"] or ["none"]), ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
