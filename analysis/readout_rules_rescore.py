"""Offline re-score of saved predictions under the opt-in readout rules r1 (decide.readout_rules), DIAGNOSTIC; CPU only.

Run: python -X utf8 -I analysis/readout_rules_rescore.py <repo_root> <out.json> <arm>=<results.jsonl> [...]
     [--proxy600 (default) | --all-dev | --dev-minus-proxy600]

Modes (row selections); every output section carries its label and ids_sha256 (sha256 of the sorted run_ids joined by
newlines, as fulldev_eval.ids_sha256 and the proxy files):
  --proxy600            the 644 proxy600 ids (sha 7d42c398...)              label "diagnostic (R1+R2 only)"
  --all-dev             full DEV, 12,104 rows (includes the proxy rows)     label "diagnostic (R1+R2 only)"
  --dev-minus-proxy600  full DEV minus the 644 proxy600 ids (n 11,460), and the variant excl_overlength51 that also
                        drops the 51 ids of jobs/clef_baseline/overlength_rows.json      label "claim (R1+R2 only)"
Only --dev-minus-proxy600 emits a "claim" label: the rules were found on proxy600, so its rows are excluded.

Each arm's ok rows get ``decide.readout_rules.apply_rules`` (R1 BFCL exactly-one, R2 SATA top-m) on the request the
engine saw ({state, questions}) and the wire answers' own probabilities (``response``: the untempered arm). Rows that
carry a ``served_response`` (tempered) are also re-scored with the yes-selection taken from the TEMPERED probabilities
(the served path), as the sensitivity-only variant "sensitivity_not_claim_rules_from_served_probs" (the "headline" is always the untempered after_minus_before). R3 (the input cap) is NOT measured offline:
input_tokens_exceeded rows need a GPU re-run, so they stay as they were (not ok = no credit). Scored with the frozen
harness.score.decompose / score.point and paired with harness.bootstrap.compare (case clusters, B=2000, seed 0), as the
m2 screen pairs arms. Nothing here is fitted.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from harness import config as _config  # noqa: E402

try:
    import decision_index  # noqa: F401
except ImportError:
    sys.path.insert(1, str(_config.KIT_CLONE))
from decide.readout_rules import RULES_ID, apply_rules  # noqa: E402
from harness import bootstrap, score  # noqa: E402
from harness.cli import load_slice  # noqa: E402

EDITION = "0.2.1"
B = 2000


def read_pred(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            out[j["run_id"]] = j
    return out


def with_rules(rows, res, from_served=False):
    """Rules applied to every ok row's ``response`` answers; ``from_served``: the yes-selection uses the row's
    ``served_response`` probabilities (tempered) instead of its own (rows without one: their own)."""
    out, per_rule, rows_changed = dict(res), {}, 0
    for r in rows:
        rid = r["_evaluation"]["run_id"]
        x = res.get(rid)
        if x is None or x.get("status") != "ok":
            continue
        y = copy.deepcopy(x)
        probs = None
        if from_served and (x.get("served_response") or {}).get("answers"):
            probs = {k: a["probabilities"] for k, a in x["served_response"]["answers"].items()}
        d = apply_rules({"state": r["state"], "questions": r["questions"]}, y["response"]["answers"], probs)
        if d["rule"]:
            s = per_rule.setdefault(d["rule"], {"rows": 0, "rows_changed": 0, "fields_changed": 0})
            s["rows"] += 1
            s["rows_changed"] += bool(d["changed"])
            s["fields_changed"] += len(d["changed"])
            rows_changed += bool(d["changed"])
        out[rid] = y
    return out, per_rule


def ci(d):
    return {k: round(d[k], 3) for k in ("point", "lo", "hi") if k in d} | {"excludes_zero": d.get("excludes_zero")}


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def section(rows, arms, label):
    ids = {r["_evaluation"]["run_id"] for r in rows}
    out = {"label": label, "n_rows": len(rows), "ids_sha256": ids_sha256(ids), "arms": {}}
    before, after = {}, {}
    for name, all_res in arms.items():
        res = {k: v for k, v in all_res.items() if k in ids}
        assert set(res) == ids, f"{name}: {len(ids - set(res))} rows missing"
        res2, per_rule = with_rules(rows, res)
        before[name], after[name] = res, res2
        db, da = score.decompose(rows, res, edition=EDITION), score.decompose(rows, res2, edition=EDITION)
        c = bootstrap.compare(da, db, B=B, seed=0)
        arm = {"index_before": round(score.point(db)["index"], 3), "index_after": round(score.point(da)["index"], 3),
               "after_minus_before": ci(c["diff"]["index"]),
               "benchmarks_changed": {n: ci(v) for n, v in c["diff"]["benchmarks"].items() if abs(v["point"]) > 1e-9},
               "rules": per_rule, "non_ok_rows": sum(1 for v in res.values() if v.get("status") != "ok"),
               "rows_with_served_response": sum(1 for v in res.values() if v.get("served_response"))}
        arm["headline"] = {"after_minus_before": arm["after_minus_before"],
                           "basis": "untempered wire probabilities (the pre-registered number)"}
        if arm["rows_with_served_response"]:
            res3, pr3 = with_rules(rows, res, from_served=True)
            d3 = score.decompose(rows, res3, edition=EDITION)
            c3 = bootstrap.compare(d3, db, B=B, seed=0)
            # sensitivity only, never the claim/headline: rules re-applied on the served (tempered) probabilities
            arm["sensitivity_not_claim_rules_from_served_probs"] = {"index_after": round(score.point(d3)["index"], 3),
                                              "after_minus_before": ci(c3["diff"]["index"]), "rules": pr3}
        out["arms"][name] = arm
        print(label, len(rows), name, json.dumps(arm), flush=True)
    names = list(arms)
    if len(names) == 2:
        a, b = names
        for tag, X in (("before", before), ("after", after)):
            c = bootstrap.compare(score.decompose(rows, X[a], edition=EDITION),
                                  score.decompose(rows, X[b], edition=EDITION), B=B, seed=0)
            out[f"{a}_minus_{b}_{tag}"] = ci(c["diff"]["index"])
            print(f"  {a}-{b} {tag}", out[f"{a}_minus_{b}_{tag}"], flush=True)
    return out


def main():
    from jobs.common.dev_proxy import load_proxy

    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    paths = dict(a.split("=", 1) for a in args)
    arms = {n: read_pred(p) for n, p in paths.items()}
    dev = load_slice("dev", EDITION)
    assert len(dev) == 12_104, len(dev)
    prox = load_proxy("proxy600")
    pids = set(prox["run_ids"])
    assert len(pids) == 644 and prox["ids_sha256"].startswith("7d42c398")
    assert ids_sha256(pids) == prox["ids_sha256"]  # same id-hash convention as the proxy file
    diag, claim = "diagnostic (R1+R2 only)", "claim (R1+R2 only)"
    out = {"rules": RULES_ID, "measured": "R1 (BFCL exactly-one) + R2 (SATA top-m) only; R3 (input cap) not measured "
                                          "offline: input_tokens_exceeded rows need a GPU re-run",
           "edition": EDITION, "scorer": "harness.score.decompose + bootstrap.compare (B=%d, seed 0)" % B,
           "proxy600_ids_sha256": prox["ids_sha256"],
           "arm_files": {n: {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                         for n, p in paths.items()},
           "sections": {}}
    if "--dev-minus-proxy600" in sys.argv:
        from jobs.m2_first_cell.fulldev_eval import overlength_ids

        ol = overlength_ids()
        assert len(ol) == 51
        rows = [r for r in dev if r["_evaluation"]["run_id"] not in pids]
        assert len(rows) == 11_460, len(rows)
        out["sections"]["dev_minus_proxy600"] = section(rows, arms, claim)
        rows_x = [r for r in rows if r["_evaluation"]["run_id"] not in ol]
        sx = section(rows_x, arms, claim)
        sx["overlength51_dropped"] = len(rows) - len(rows_x)
        out["sections"]["dev_minus_proxy600_excl_overlength51"] = sx
    elif "--all-dev" in sys.argv:
        out["sections"]["all_dev"] = section(dev, arms, diag)
    else:
        out["sections"]["proxy600"] = section([r for r in dev if r["_evaluation"]["run_id"] in pids], arms, diag)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
