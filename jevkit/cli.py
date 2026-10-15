"""``python -m jevkit <command>``.

score       --pred P.jsonl (--items I.jsonl ... | --public) [--price SPEC [--reference-price SPEC] |
            --usd-per-1000 X --price-note TEXT] [--api] [--p50 S --p95 S] [--exclude-from-cost IDS]
            [--B 1000] [--seed 16] [--g-med 2.5734] [--out report.json]
perturb     make  (--items I.jsonl | --public) --out P.jsonl [--ops a,b] [--seeds 0,1]
            score (--items I.jsonl | --public) --perturbed P.jsonl --pred B.jsonl --pred-perturbed Q.jsonl [--out R]
guarantees  conformal (--items | --public) --pred P.jsonl [--cal-frac 0.5] [--alpha .1 --epsilon .05 --delta .05]
            injection --cal-items I --cal-pred P --test T.jsonl [--alpha .1]
            injection --mode matched --cal-intervals CAL.jsonl --test TEST.jsonl (g16sym records)
board       --api api_v161.json [--out R.json]           reproduce the published board from its sub-fields
items       [--data-dir D] [--fingerprints F.json]       verify the pinned public items (offline)
fetch       [--dest D]                                   download + verify the pinned public items (network)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import board, guarantees, meter, perturb, scoring, spec, systemone
from . import items as IT


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(obj: Any, out: str | None) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    if out:
        Path(out).write_bytes((text + "\n").encode("utf-8"))  # LF on every OS (reports land in byte-exact receipts)
    else:
        sys.stdout.write(text + "\n")


def _load_items(a: argparse.Namespace) -> list[IT.Item]:
    out: list[IT.Item] = []
    if getattr(a, "public", False):
        out += IT.load_public(getattr(a, "data_dir", None))
    for p in getattr(a, "items", None) or []:
        out += IT.load_items(p, tier=getattr(a, "tier", None))
    if not out:
        raise SystemExit("no items: pass --items or --public")
    ids = [it.id for it in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids across --items files")
    return out


def _add_items(p: argparse.ArgumentParser) -> None:
    p.add_argument("--items", action="append", help="item JSONL (repeatable)")
    p.add_argument("--public", action="store_true", help="the pinned JevBench public items (eval-only)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tier", default=None, help="tier for items without a tier field")


# ------------------------------------------------------------------------------------------- score
def cmd_score(a: argparse.Namespace) -> int:
    items = _load_items(a)
    preds = _read_jsonl(a.pred)
    answers = systemone.preds_to_answers(preds)
    unknown = set(answers) - {it.id for it in items}
    by_id = {it.id: it for it in items}
    self_hosted = not a.api
    notes = []
    if a.p50 is not None and a.p95 is not None:
        sp = scoring.speed(a.p50, a.p95, self_hosted=self_hosted)
        sp["source"] = "given --p50/--p95 (raw)"
    else:
        timed = {it.id for it in meter.timed_subset(items)}
        lat = [p.get("latency_s") for p in preds if p.get("id") in timed and p.get("latency_s") is not None]
        src = "pred latency_s on open standard+judge items"
        if not lat:
            lat = [p.get("latency_s") for p in preds if p.get("latency_s") is not None]
            src = "pred latency_s on all items (no open standard/judge item was timed)"
            notes.append("Speed timed on a different item set than the board's open standard+judge set")
        sp = meter.latency_summary(lat, self_hosted=self_hosted) if lat else scoring.speed(None, None)
        sp["source"] = src if lat else "no latency"
    excl: set[str] = set()
    if a.exclude_from_cost:
        d = json.loads(Path(a.exclude_from_cost).read_text(encoding="utf-8"))
        excl = set(d["ids"] if isinstance(d, dict) else d)
    if a.usd_per_1000 is not None:
        if not a.price_note:
            raise SystemExit("--usd-per-1000 needs --price-note (the price source is a recorded input)")
        co = {"usd_per_1000": a.usd_per_1000, "score": scoring.cost_score(a.usd_per_1000), "price_rule": "given",
              "price_note": a.price_note}
    elif a.price:
        ref = meter.parse_price(a.reference_price) if a.reference_price else None
        co = meter.cost_summary(meter.records_from_preds(preds), meter.parse_price(a.price), ref, excl)
    else:
        co = {"usd_per_1000": None, "score": None, "price_rule": "no price given: Cost and Composite are null"}
    rep = scoring.score_system(items, answers, speed_axis=sp, cost_axis=co, g_med=a.g_med, b=a.B, seed=a.seed)
    rep["inputs"] = {"pred": a.pred, "pred_sha256": IT.sha256_file(a.pred), "items": a.items, "public": a.public,
                     "n_pred_records": len(preds), "n_pred_unknown_ids": len(unknown),
                     "n_items_without_pred": sum(1 for i in by_id if i not in answers),
                     "lane": "api" if a.api else "self-hosted (x2 + 0.15 s)"}
    rep["spec"] = "JevBench v1.6.1 (reverse-derived; jevkit/README.md)"
    rep["approximations"] = notes + _approximations(rep)
    _write(rep, a.out)
    return 0


def _approximations(rep: dict) -> list[str]:
    out = []
    basis = rep["intelligence"].get("basis", "")
    if basis != "open+sealed":
        out.append(f"Intelligence basis: {basis}. The board's I averages open and sealed (1,200 unpublished items) and "
                   "applies the gap penalty; a local run cannot reproduce the sealed half.")
    out.append("Latency is in-process wall time (no evaluator network hop); the self-hosted x2 + 0.15 s adjustment "
               "is the board's own assumption.")
    out.append("Bootstrap is over the scored items only (B and seed as published), so CIs are local, not the board's.")
    return out


# ------------------------------------------------------------------------------------------- perturb
def cmd_perturb(a: argparse.Namespace) -> int:
    items = _load_items(a)
    ops = a.ops.split(",") if a.ops else None
    if ops:
        bad = [o for o in ops if o not in perturb.OPERATORS]
        if bad:
            raise SystemExit(f"unknown operators {bad}; known {list(perturb.OPERATORS)}")
    if a.mode == "make":
        seeds = [int(x) for x in a.seeds.split(",")]
        ps = perturb.make_perturbations(items, ops, seeds)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for p in ps)
        counts: dict[str, int] = {}
        for p in ps:
            counts[p.op] = counts.get(p.op, 0) + 1
        _write({"n_base": len(items), "n_perturbed": len(ps), "per_operator": counts, "out": a.out}, None)
        return 0
    ps = []
    for d in _read_jsonl(a.perturbed):
        meta = d.pop("perturbation")
        ps.append(perturb.Perturbed(IT.Item.from_dict(d), meta["op"], meta["seed"], meta["base_id"], meta["label_map"]))
    if ops:
        ps = [p for p in ps if p.op in ops]
    rep = perturb.flip_report(items, ps, systemone.preds_to_answers(_read_jsonl(a.pred)),
                              systemone.preds_to_answers(_read_jsonl(a.pred_perturbed)))
    _write(rep, a.out)
    return 0


# ------------------------------------------------------------------------------------------- guarantees
def _split(items: list[IT.Item], frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministic hash split (no item is chosen on the basis of any answer)."""
    cal, test = set(), set()
    for it in items:
        h = int(hashlib.sha256(f"{seed}|{it.id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (cal if h < frac else test).add(it.id)
    return cal, test


def cmd_guarantees(a: argparse.Namespace) -> int:
    if a.mode == "conformal":
        items = _load_items(a)
        sc = guarantees.scored_from_answers(items, systemone.preds_to_answers(_read_jsonl(a.pred)))
        cal_ids, _ = _split(items, a.cal_frac, a.seed)
        cal = [r for r in sc if r.id in cal_ids]
        test = [r for r in sc if r.id not in cal_ids]
        # three disjoint, answer-independent sets: test (held out), and calibration cut again into FIT / LTT halves
        rep = guarantees.conformal_report(cal, test, a.alpha, a.epsilon, a.delta, mondrian=not a.pooled, method=a.ltt,
                                          split_seed=a.seed, fit_frac=a.fit_frac)
        rep["split"] = {"cal_frac": a.cal_frac, "seed": a.seed, "n_cal": len(cal), "n_test": len(test),
                        "test_ids_sha256": guarantees.ids_sha256(r.id for r in test)}
    elif a.inj_mode == "matched":
        # calibrate and test on the same trusted-only bound s_lo (ssdm.g16sym.export.v1 records; cal needs gold)
        if not (a.cal_intervals and a.test):
            raise SystemExit("--mode matched needs --cal-intervals CAL.jsonl and --test TEST.jsonl (g16sym records)")
        rep = guarantees.injection_robust_report(_read_jsonl(a.cal_intervals), _read_jsonl(a.test), a.alpha,
                                                 mode="matched", epsilon=a.epsilon, delta=a.delta,
                                                 split_seed=a.seed, fit_frac=a.fit_frac)
    else:
        cal_items = IT.load_items(a.cal_items, tier=a.tier)
        cal = guarantees.scored_from_answers(cal_items, systemone.preds_to_answers(_read_jsonl(a.cal_pred)))
        rep = guarantees.injection_robust_report(cal, _read_jsonl(a.test), a.alpha, mondrian=not a.pooled)
    _write(rep, a.out)
    return 0


# ------------------------------------------------------------------------------------------- board / items / fetch
def cmd_board(a: argparse.Namespace) -> int:
    rep = board.verify_file(a.api, a.tol)
    if a.out:
        _write(rep, a.out)
    sys.stdout.write("\n".join(board.summary_lines(rep)) + "\n")
    scored = [k for k in rep["checks"] if not k.startswith("view[")]
    return 0 if all(not rep["checks"][k]["mismatches"] for k in scored) else 1


def cmd_items(a: argparse.Namespace) -> int:
    pin = IT.read_pin()
    shas = IT.verify_public(a.data_dir, pin)
    its = IT.load_public(a.data_dir, pin)
    rep = {"commit": pin["commit"], "sha256": shas, "n": len(its),
           "by_tier_type": {f"{t}|{y}": sum(1 for i in its if i.tier == t and i.type == y)
                            for t in spec.TIERS for y in spec.TYPES if any(i.tier == t and i.type == y for i in its)}}
    if a.fingerprints:
        _write(IT.fingerprint_manifest(its), a.fingerprints)
        rep["fingerprints"] = a.fingerprints
    _write(rep, None)
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    _write(IT.fetch_public(a.dest), None)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jevkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    _add_items(s)
    s.add_argument("--pred", required=True)
    s.add_argument("--price", help="price spec or JSON file (meter.parse_price)")
    s.add_argument("--reference-price", help="base-model reference price (the board's floor)")
    s.add_argument("--usd-per-1000", type=float)
    s.add_argument("--price-note")
    s.add_argument("--exclude-from-cost", help='JSON file, {"ids": [...]} or a list: item ids kept out of the cost set')
    s.add_argument("--api", action="store_true", help="hosted API lane: no x2 + 0.15 s latency adjustment")
    s.add_argument("--p50", type=float)
    s.add_argument("--p95", type=float)
    s.add_argument("--B", type=int, default=spec.BOOTSTRAP_B)
    s.add_argument("--seed", type=int, default=spec.BOOTSTRAP_SEED)
    s.add_argument("--g-med", type=float, default=spec.G_MED_V161)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)

    p = sub.add_parser("perturb")
    p.add_argument("mode", choices=("make", "score"))
    _add_items(p)
    p.add_argument("--out")
    p.add_argument("--ops")
    p.add_argument("--seeds", default="0")
    p.add_argument("--perturbed")
    p.add_argument("--pred")
    p.add_argument("--pred-perturbed")
    p.set_defaults(fn=cmd_perturb)

    g = sub.add_parser("guarantees")
    g.add_argument("mode", choices=("conformal", "injection"))
    _add_items(g)
    g.add_argument("--pred")
    g.add_argument("--cal-frac", type=float, default=0.5, help="share of items in calibration (rest = test)")
    g.add_argument("--fit-frac", type=float, default=0.5,
                   help="share of the calibration decisions that fit q_hat; the disjoint rest runs LTT")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--alpha", type=float, default=0.1)
    g.add_argument("--epsilon", type=float, default=0.05)
    g.add_argument("--delta", type=float, default=0.05)
    g.add_argument("--pooled", action="store_true", help="one quantile for all types (default: Mondrian per type)")
    g.add_argument("--ltt", choices=("bonferroni", "fixed_sequence"), default="bonferroni")
    g.add_argument("--cal-items")
    g.add_argument("--cal-pred")
    g.add_argument("--test")
    g.add_argument("--mode", dest="inj_mode", choices=("clean", "matched"), default="clean",
                   help="injection only: clean = q_hat from clean calibration predictions (default); matched = "
                        "q_hat from s_lo of ssdm.g16sym.export.v1 calibration records (--cal-intervals)")
    g.add_argument("--cal-intervals", help="matched mode: calibration JSONL of g16sym export records with gold")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_guarantees)

    b = sub.add_parser("board")
    b.add_argument("--api", required=True)
    b.add_argument("--tol", type=float, default=board.TOL)
    b.add_argument("--out")
    b.set_defaults(fn=cmd_board)

    i = sub.add_parser("items")
    i.add_argument("--data-dir")
    i.add_argument("--fingerprints")
    i.set_defaults(fn=cmd_items)

    f = sub.add_parser("fetch")
    f.add_argument("--dest")
    f.set_defaults(fn=cmd_fetch)

    a = ap.parse_args(argv)
    if a.cmd == "perturb" and a.mode == "make" and not a.out:
        ap.error("perturb make needs --out")
    if a.cmd == "perturb" and a.mode == "score" and not (a.perturbed and a.pred and a.pred_perturbed):
        ap.error("perturb score needs --perturbed, --pred and --pred-perturbed")
    return a.fn(a)
