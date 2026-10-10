"""Independent jevkit rescore of a landed m2_first_cell screen artifact (evidence lane; A-PREREG-JEV1-4B and kin).

Recomputes every JevBench number of the job FROM THE RAW PER-ITEM PREDICTIONS (``out/screen/jevkit/<arm>.preds.jsonl``)
and never reads a number out of ``screen.json`` except to diff against it:

  * per arm: ``python -m jevkit score --public --pred <arm>.preds.jsonl --price <declared price>`` (the frozen CLI,
    run as a subprocess from THIS checkout) -> I, C, S, K, Capability, Composite A/B/C, CIs; tokens / decision from the
    preds' ``usage.input_tokens``;
  * paired deltas: every arm minus ``--baseline`` (default ``stock_shim``) on the SAME stratified resamples
    (jevkit.scoring.strata / aggregate / capability / composite; B 1000, seed 16; Speed and Cost held at each arm's
    measured axes), with a second, jevkit-native path for the Composite-A difference
    (``jevkit.scoring.bootstrap({arm, baseline})['_pairs']``) as a cross-check of this script's own loop;
  * diffs against the in-job ``screen.json`` (``jevkit.arms.*.summary``, ``vs_stock_shim``, ``primary``,
    ``lean_minus_native``): any |delta| > ``--tol`` (default 0.05 points) is a FLAG;
  * integrity checks: every arm has exactly the pinned public item ids (231), no id twice, the same id set across
    arms, status counts; choice answers' probability keys = the item's labels, in the request's criteria order (the
    jev1 loader / shim contract: options are lettered in criteria order, never re-sorted); the declared price class
    (kind + model, e.g. Qwen/Qwen3.5-4B) equals the in-job one; the latency source (pred latency_s on the open
    standard+judge items, n, every pred timed); cell.json trained steps (``--expect-steps``) and the scored checkpoint.

Usage (artifact = untrusted data: run with -I from a scripts dir outside it, the repo root on sys.path explicitly):
  python -I -X utf8 scripts/ev_rescore_jevkit.py --art <extracted artifact root> --out rescore.json \
      [--baseline stock_shim] [--primary native] [--expect-steps 1121] [--price "<price spec>"]
Exit 0 = rescored with no FLAG; 3 = at least one FLAG (read ``flags``); 2 = inputs missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.meter import parse_price  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402

# A-PREREG-JEV1-4B's declared price (job_params_jev1_q35_4b.json @5e159437, same as the size-cliff 4B arm)
DEFAULT_PRICE = ("kind=size_class_estimate,in=0.03,out=0,source=declared reference price for Qwen3.5-4B "
                 "(A-PREREG-SIZECLIFF operator brief 2026-10-09; size-class reference: no list tariff),"
                 "as_of=2026-10-09,model=Qwen/Qwen3.5-4B")
TIMED_SOURCE = "pred latency_s on open standard+judge items"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_one(root: Path, rel: str) -> Path | None:
    p = root / rel
    if p.is_file():
        return p
    hits = sorted(root.rglob(Path(rel).name))
    hits = [h for h in hits if h.as_posix().endswith(rel)]
    return hits[0] if hits else None


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def jevkit_cli(pred: Path, price: str, out: Path) -> dict:
    """The frozen CLI, as a subprocess of THIS checkout (cwd = repo root, never the artifact dir)."""
    cmd = [sys.executable, "-X", "utf8", "-m", "jevkit", "score", "--public", "--pred", str(pred), "--price", price,
           "--out", str(out)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"jevkit score {pred.name} failed ({r.returncode}): {r.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def integrity(name: str, preds: list[dict], items: list, flags: list) -> dict:
    by_id = {it.id: it for it in items}
    ids = [str(p.get("id")) for p in preds]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    unknown = sorted(set(ids) - set(by_id))
    missing = sorted(set(by_id) - set(ids))
    status: dict = {}
    for p in preds:
        status[p.get("status")] = status.get(p.get("status"), 0) + 1
    order_bad, keyset_bad, n_choice_probs = [], [], 0
    no_latency = 0
    for p in preds:
        it = by_id.get(str(p.get("id")))
        if p.get("latency_s") is None:
            no_latency += 1
        a = p.get("answer")
        if it is None or not isinstance(a, dict) or it.type != "choice":
            continue
        probs = a.get("probabilities")
        if not isinstance(probs, dict):
            continue
        n_choice_probs += 1
        if set(map(str, probs)) != set(map(str, it.labels)):
            keyset_bad.append(it.id)
        elif list(map(str, probs)) != list(map(str, it.question.get("criteria") or {})):
            order_bad.append(it.id)
    out = {"n_records": len(preds), "n_unique_ids": len(set(ids)), "duplicated_ids": dup[:20], "n_duplicated": len(dup),
           "unknown_ids": unknown[:20], "missing_ids": missing[:20], "status": status,
           "choice_with_probs": n_choice_probs, "choice_keyset_mismatch": keyset_bad[:20],
           "choice_order_not_criteria_order": order_bad[:20], "n_order_not_criteria_order": len(order_bad),
           "preds_without_latency": no_latency}
    if dup:
        flags.append(f"{name}: {len(dup)} item ids duplicated in the preds (preds_to_answers keeps the LAST)")
    if unknown or missing or len(preds) != len(items):
        flags.append(f"{name}: id set != the {len(items)} pinned public items (unknown {len(unknown)}, missing "
                     f"{len(missing)}, records {len(preds)})")
    if keyset_bad:
        flags.append(f"{name}: {len(keyset_bad)} choice answers whose probability keys != the item labels")
    if order_bad:
        flags.append(f"{name}: {len(order_bad)} choice answers whose probability keys are not in criteria order "
                     "(options re-sorted vs the jev1 loader / shim contract)")
    if set(status) - {"ok"}:
        flags.append(f"{name}: non-ok statuses {status} (scored as invalid answers)")
    return out


def tokens(preds: list[dict]) -> dict:
    t = [(p.get("usage") or {}).get("input_tokens") for p in preds]
    t = [x for x in t if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return {"mean": float(np.mean(t)) if t else None, "p50": float(np.median(t)) if t else None, "n": len(t)}


def paired(tables: dict, axes: dict, base: str, B: int, seed: int) -> dict:
    """Every arm minus ``base`` on one shared set of stratified resamples (strata of the base table)."""
    def ax(nm, idx=None):
        r = JS.aggregate(tables[nm], idx, detail=False)
        i, c = r["intelligence"]["I"], r["calibration"]["score"]
        s, k = axes[nm]
        return {"I": i, "C": c, "capability": JS.capability(i, c), "composite_A": JS.composite(i, c, s, k, "A")}

    arms = [n for n in tables if n != base]
    full = {n: ax(n) for n in tables}
    rng = np.random.default_rng(seed)
    st = JS.strata(tables[base])
    draws = {n: {k: [] for k in full[base]} for n in arms}
    for _ in range(B):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        xb = ax(base, idx)
        for n in arms:
            xa = ax(n, idx)
            for k in xa:
                if xa[k] is not None and xb[k] is not None:
                    draws[n][k].append(xa[k] - xb[k])
    out = {}
    for n in arms:
        out[n] = {"B": B, "seed": seed, "a": n, "b": base}
        for k in full[base]:
            v = draws[n][k]
            pt = None if full[n][k] is None or full[base][k] is None else full[n][k] - full[base][k]
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            out[n][k] = {"point": pt, "ci95": ci, "a": full[n][k], "b": full[base][k],
                         "ci_excludes_0": None if ci is None else bool(ci[0] > 0 or ci[1] < 0)}
        # second path: jevkit's own paired bootstrap (frozen scorer) for the Composite-A difference
        jb = JS.bootstrap({n: tables[n], base: tables[base]}, {n: axes[n], base: axes[base]}, b=B, seed=seed)
        out[n]["jevkit_bootstrap_composite_A_diff_ci95"] = jb["_pairs"][f"{n} - {base}"]["composite_A_diff_ci95"]
    return out


def cmp(flags: list, diffs: list, where: str, mine, theirs, tol: float) -> None:
    if isinstance(mine, (list, tuple)) and isinstance(theirs, (list, tuple)):
        for j, (m, t) in enumerate(zip(mine, theirs)):
            cmp(flags, diffs, f"{where}[{j}]", m, t, tol)
        return
    if mine is None and theirs is None:
        return
    if mine is None or theirs is None:
        diffs.append({"where": where, "rescore": mine, "screen": theirs, "abs_diff": None})
        flags.append(f"diff {where}: rescore {mine} vs screen {theirs} (one side missing)")
        return
    d = abs(float(mine) - float(theirs))
    diffs.append({"where": where, "rescore": mine, "screen": theirs, "abs_diff": d})
    if d > tol:
        flags.append(f"diff {where}: rescore {mine:.4f} vs screen {theirs:.4f} (|d| {d:.4f} > {tol})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/ev_rescore_jevkit.py")
    ap.add_argument("--art", required=True, help="extracted artifact root (holds out/screen.json, out/screen/jevkit/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="where the per-arm jevkit reports go (default: <out> stem dir)")
    ap.add_argument("--price", default=DEFAULT_PRICE)
    ap.add_argument("--baseline", default="stock_shim")
    ap.add_argument("--primary", default="native")
    ap.add_argument("--expect-steps", type=int, default=None)
    ap.add_argument("--expect-n", type=int, default=231)
    ap.add_argument("--expect-model", default="Qwen/Qwen3.5-4B", help="the declared price class's model")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(argv)

    art = Path(a.art).resolve()
    screen_p = find_one(art, "out/screen.json")
    jdir = (find_one(art, "out/screen/jevkit/identity.json") or Path("/nonexistent")).parent
    if screen_p is None or not jdir.is_dir():
        print(f"missing: screen.json {screen_p}, jevkit dir {jdir}", file=sys.stderr)
        return 2
    work = Path(a.work) if a.work else Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    flags: list[str] = []
    diffs: list[dict] = []
    screen = json.loads(screen_p.read_text(encoding="utf-8"))
    sjk = screen.get("jevkit") or {}
    items = load_public()
    if len(items) != a.expect_n:
        flags.append(f"pinned public items: {len(items)} != {a.expect_n}")

    # ---- price class
    mine_price = parse_price(a.price)
    price_chk = {"declared": a.price, "parsed_model": mine_price.model, "parsed_kind": mine_price.kind,
                 "screen_price": sjk.get("price")}
    if (mine_price.model or "") != a.expect_model:
        flags.append(f"price class model {mine_price.model!r} != expected {a.expect_model!r}")
    sp = sjk.get("price") or {}
    for k_m, k_s in (("model", "model"), ("kind", "kind"), ("usd_per_m_input", "usd_per_m_input"),
                     ("usd_per_m_output", "usd_per_m_output")):
        if getattr(mine_price, k_m, None) != sp.get(k_s):
            flags.append(f"price {k_m}: declared {getattr(mine_price, k_m, None)!r} != in-job {sp.get(k_s)!r}")

    # ---- arms: raw preds -> jevkit CLI
    arm_files = sorted(jdir.glob("*.preds.jsonl"))
    arms, tables, axes, preds_by = {}, {}, {}, {}
    for pf in arm_files:
        name = pf.name[: -len(".preds.jsonl")]
        preds = read_jsonl(pf)
        preds_by[name] = preds
        chk = integrity(name, preds, items, flags)
        rep = jevkit_cli(pf, a.price, work / f"{name}.rescore.jevkit.json")
        spd = rep.get("speed") or {}
        if spd.get("source") != TIMED_SOURCE:
            flags.append(f"{name}: latency source {spd.get('source')!r} (not the open standard+judge timed set)")
        arms[name] = {"preds": pf.name, "preds_sha256": sha(pf), "checks": chk,
                      "I": rep["axes"]["intelligence"], "C": rep["axes"]["calibration"], "S": rep["axes"]["speed"],
                      "K": rep["axes"]["cost"], "capability": rep["capability"], "composite": rep["scores"],
                      "I_ci95": (rep.get("ci95") or {}).get("I"), "C_ci95": (rep.get("ci95") or {}).get("C"),
                      "composite_ci95": (rep.get("ci95") or {}).get("composite"),
                      "intelligence_basis": rep["intelligence"].get("basis"),
                      "usd_per_1000": (rep.get("cost") or {}).get("usd_per_1000"),
                      "tokens_per_decision": tokens(preds),
                      "latency": {k: spd.get(k) for k in ("source", "n", "p50_s_raw", "p95_s_raw", "p50_s_adjusted",
                                                          "adjustment", "score")},
                      "report_sha256": sha(work / f"{name}.rescore.jevkit.json")}
        tables[name] = JS.build_table(items, preds_to_answers(preds))
        axes[name] = (rep["axes"]["speed"], rep["axes"]["cost"])
        # diff against the in-job summary of this arm
        ss = ((sjk.get("arms") or {}).get(name) or {}).get("summary")
        if ss is None:
            flags.append(f"{name}: arm present in the preds but not in screen.json jevkit.arms")
        else:
            for k in ("I", "C", "S", "K", "capability"):
                cmp(flags, diffs, f"arms.{name}.{k}", arms[name][k], ss.get(k), a.tol)
            for o in ("A", "B", "C"):
                cmp(flags, diffs, f"arms.{name}.composite.{o}", arms[name]["composite"].get(o),
                    (ss.get("composite") or {}).get(o), a.tol)
            cmp(flags, diffs, f"arms.{name}.I_ci95", arms[name]["I_ci95"], ss.get("I_ci95"), a.tol)
            cmp(flags, diffs, f"arms.{name}.C_ci95", arms[name]["C_ci95"], ss.get("C_ci95"), a.tol)
            cmp(flags, diffs, f"arms.{name}.tokens_mean", arms[name]["tokens_per_decision"]["mean"],
                (ss.get("input_tokens_per_decision") or {}).get("mean"), a.tol)
    for nm in (sjk.get("arms") or {}):
        if nm not in arms:
            flags.append(f"screen.json reports arm {nm!r} but its preds file is missing")
    id_sets = {n: tuple(sorted(str(p.get("id")) for p in pr)) for n, pr in preds_by.items()}
    same_ids = len(set(id_sets.values())) == 1
    if not same_ids:
        flags.append("arms are not scored on the same item id set")
    if any(t.ids != tables[next(iter(tables))].ids for t in tables.values()):
        flags.append("item tables are not in the same order (paired bootstrap invalid)")

    # ---- paired deltas vs the baseline
    out_pairs, primary = None, None
    if a.baseline in tables:
        out_pairs = paired(tables, axes, a.baseline, a.B, a.seed)
        for n, d in out_pairs.items():
            d["tokens_mean_delta"] = (None if arms[n]["tokens_per_decision"]["mean"] is None else
                                      arms[n]["tokens_per_decision"]["mean"]
                                      - arms[a.baseline]["tokens_per_decision"]["mean"])
            cmp(flags, diffs, f"pairs.{n}.composite_A.ci95 (own loop vs jevkit.scoring.bootstrap)",
                d["composite_A"]["ci95"], d["jevkit_bootstrap_composite_A_diff_ci95"], 1e-9)
            sv = (sjk.get(f"vs_{a.baseline}") or {}).get(n)
            if sv is not None:
                for k in ("I", "C", "capability", "composite_A"):
                    cmp(flags, diffs, f"vs_{a.baseline}.{n}.{k}.point", d[k]["point"], (sv.get(k) or {}).get("point"),
                        a.tol)
                    cmp(flags, diffs, f"vs_{a.baseline}.{n}.{k}.ci95", d[k]["ci95"], (sv.get(k) or {}).get("ci95"),
                        a.tol)
            elif a.baseline == "stock_shim":
                flags.append(f"screen.json has no vs_stock_shim.{n}")
        p = out_pairs.get(a.primary)
        if p is not None:
            lo = p["composite_A"]["ci95"][0] if p["composite_A"]["ci95"] else None
            hi = p["composite_A"]["ci95"][1] if p["composite_A"]["ci95"] else None
            primary = {"arm": a.primary, "baseline": a.baseline, "delta_composite_A": p["composite_A"],
                       "rule": "A-PREREG-JEV1-4B: SUCCESS iff delta Composite A > 0 with its 95% CI excluding 0",
                       "success": bool(p["composite_A"]["point"] is not None and p["composite_A"]["point"] > 0
                                       and lo is not None and lo > 0),
                       "secondary": {k: p[k] for k in ("I", "C", "capability")}
                       | {"tokens_mean_delta": p["tokens_mean_delta"]},
                       "ci_hi": hi}
            sprim = sjk.get("primary")
            if sprim is not None:
                cmp(flags, diffs, "primary.delta_composite_A.point", p["composite_A"]["point"],
                    (sprim.get("delta_composite_A") or {}).get("point"), a.tol)
                if bool(sprim.get("success")) != primary["success"]:
                    flags.append(f"primary verdict differs: rescore success={primary['success']} vs screen "
                                 f"{sprim.get('success')}")
            elif a.baseline == "stock_shim":
                flags.append("screen.json has no jevkit.primary block")
    else:
        flags.append(f"baseline arm {a.baseline!r} has no preds file: no paired primary")

    # lean - native (the size-cliff block), when the job wrote it: same protocol, own loop
    if "lean_minus_native" in sjk and {"lean", "native"} <= set(tables):
        ln = paired({"native": tables["native"], "lean": tables["lean"]},
                    {"native": axes["native"], "lean": axes["lean"]}, "native", a.B, a.seed)["lean"]
        s = sjk["lean_minus_native"]
        cmp(flags, diffs, "lean_minus_native.I", ln["I"]["point"], s.get("I"), a.tol)
        cmp(flags, diffs, "lean_minus_native.I_ci95", ln["I"]["ci95"], s.get("I_ci95"), a.tol)
        cmp(flags, diffs, "lean_minus_native.C", ln["C"]["point"], s.get("C"), a.tol)
        cmp(flags, diffs, "lean_minus_native.C_ci95", ln["C"]["ci95"], s.get("C_ci95"), a.tol)
        out_pairs = (out_pairs or {}) | {"lean_minus_native": ln}

    # ---- cell / checkpoint
    cell_chk = {}
    cell_p = find_one(art, "out/cell.json")
    if cell_p is not None:
        cell = json.loads(cell_p.read_text(encoding="utf-8"))
        cell_chk = {"train_steps": (cell.get("train") or {}).get("steps"),
                    "max_steps": (cell.get("config") or {}).get("max_steps"), "best": cell.get("best")}
        if a.expect_steps is not None and cell_chk["train_steps"] != a.expect_steps:
            flags.append(f"cell.json train.steps {cell_chk['train_steps']} != expected {a.expect_steps}")
    elif a.expect_steps is not None:
        flags.append("cell.json missing")
    idc = find_one(art, "out/screen/identity-child.json")
    if idc is not None:
        ck = (json.loads(idc.read_text(encoding="utf-8")).get("checkpoint") or {})
        cell_chk["scored_checkpoint"] = {k: ck.get(k) for k in ("path", "step", "sha256") if k in ck}
    jid = json.loads((jdir / "identity.json").read_text(encoding="utf-8"))
    cell_chk["stock_arm_identity"] = {k: v for k, v in (jid.get("stock_shim") or {}).items()
                                      if k in ("arm", "shim", "lora", "lm_head", "n", "status", "temperature")}

    res = {"schema": "ev-rescore-jevkit/1", "artifact": str(art), "screen_json": str(screen_p),
           "screen_json_sha256": sha(screen_p), "rescore_checkout": str(ROOT),
           "protocol": f"jevkit CLI per arm (--public, declared price); paired stratified bootstrap B={a.B} seed "
                       f"{a.seed} (strata of the baseline table), Speed/Cost held fixed; tol {a.tol}",
           "n_items": len(items), "same_item_ids_all_arms": same_ids, "price": price_chk, "arms": arms,
           "pairs": out_pairs, "primary": primary, "cell": cell_chk, "diffs_vs_screen": diffs,
           "max_abs_diff": max((d["abs_diff"] for d in diffs if d["abs_diff"] is not None), default=None),
           "flags": flags, "verdict": "AGREES" if not flags else "FLAGGED"}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"], "flags": flags, "max_abs_diff": res["max_abs_diff"],
                      "primary": primary, "arms": {n: {k: arms[n][k] for k in ("I", "C", "S", "K", "capability")}
                                                   | {"A": arms[n]["composite"]["A"]} for n in arms}},
                     indent=1, default=str))
    return 0 if not flags else 3


if __name__ == "__main__":
    raise SystemExit(main())
