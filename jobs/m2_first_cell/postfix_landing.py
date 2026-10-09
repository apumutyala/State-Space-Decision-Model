"""CPU landing step for mode ``readout_ab_d1pf``: the ordinal-fix (claude/anti-collapse 6be1c8f) effect on D1 + v3w.

Before the fix ``heads.features.build_head_input`` gave every layout-g1v2 row e = 0 (option embeddings), so job-39gnn's
v3w readout of D1 (the champion, 38.67) fed typed.ordinal.gap zeros on score questions, although D1 (trained under
layout ``ordered``) learned that weight on live e. The post-fix job re-reads the same checkpoint on the same 2,008 rows;
this step pairs it with the PRE-fix v3w results (job-39gnn art-zskmx, every byte pinned in the params'
``postfix_landing`` block) and reports:

* ``v1_control``: post-fix v1 per-row (status + response) == job-39gnn's v1, and (``--d1``) == D1's in-job index arm
  (art-bm3e6 x_evals/out/units-child-index.json). Layout ``ordered`` is untouched by the fix: any difference fails.
* ``v3w``: Index post - pre (paired cluster bootstrap, same replicates) and the per-question-type split: a hybrid
  results set takes the post-fix answers on SCORE questions only (pre-fix elsewhere), so
  Index(hybrid) - Index(pre) = the score-question share and Index(post) - Index(hybrid) = the choice/noul share
  (the other order is reported too: the interaction is their difference). Plus per-type answer-change counts.
* ``v3w_refit``: the post-fix per-readout refit (results_refit) Index, for the record (the champion number is untempered).

    python -X utf8 -m jobs.m2_first_cell.postfix_landing --post <out>/readout_ab --pre-tar art-zskmx.tar \
        [--d1 units-child-index.json] --out postfix_landing.json
Exit 0 = report written and the v1 control held; 2 = a pin or the v1 control failed (report still written).
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "jobs/m2_first_cell/job_params_readout_ab_d1pf.json"
V3W = "out/readout_ab/child.v3w.results.jsonl"
V1 = "out/readout_ab/child.v1.results.jsonl"
#: per-row fields that legitimately differ between two runs of the same code (timing)
TIMING_FIELDS = ("total_wall_ms",)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pre_members(tar_path: Path, pins: dict) -> dict[str, bytes]:
    """{member: bytes} for the pinned members of the pre-fix artifact; refuses any pin mismatch."""
    raw = tar_path.read_bytes()
    if sha256(raw) != pins["pre_artifact_sha256"]:
        raise SystemExit(f"{tar_path}: sha256 {sha256(raw)} != pinned {pins['pre_artifact_sha256']}")
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        phase = t.extractfile(pins["pre_phase_member"]).read()
    if sha256(phase) != pins["pre_phase_sha256"]:
        raise SystemExit(f"{pins['pre_phase_member']}: sha256 mismatch")
    out = {}
    with tarfile.open(fileobj=io.BytesIO(phase), mode="r:gz") as t:
        for name, want in pins["pre_members_sha256"].items():
            b = t.extractfile(name).read()
            if sha256(b) != want:
                raise SystemExit(f"{name}: sha256 {sha256(b)} != pinned {want}")
            out[name] = b
    return out


def read_rows(b: bytes) -> dict[str, dict]:
    out = {}
    for ln in b.decode("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["run_id"] in out:
                raise SystemExit(f"duplicate run_id {r['run_id']}")
            out[r["run_id"]] = r
    return out


def same_rows(a: dict, b: dict, ids) -> dict:
    """Per-row identity on everything but timing (status, response, error, ids ...)."""
    diff = collections.Counter()
    bad = []
    for k in ids:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            diff["missing"] += 1
            bad.append(k)
            continue
        fs = {f for f in set(x) | set(y) if f not in TIMING_FIELDS and x.get(f) != y.get(f)}
        for f in fs:
            diff[f] += 1
        if fs:
            bad.append(k)
    return {"rows": len(ids), "rows_identical": len(ids) - len(bad), "field_diffs": dict(diff), "first_bad": bad[:5]}


def qtypes_of(rows) -> dict[str, dict[str, str]]:
    return {r["_evaluation"]["run_id"]: {q: v["type"] for q, v in r["questions"].items()} for r in rows}


def hybrid(pre: dict, post: dict, qtypes: dict, take: set[str]) -> dict:
    """pre-fix rows with the post-fix answers on questions whose request type is in ``take``."""
    out = {}
    for k, r in pre.items():
        h = copy.deepcopy(r)
        p = post[k]
        if r.get("status") == "ok" and p.get("status") == "ok":
            for q, t in qtypes[k].items():
                if t in take and q in p["response"]["answers"]:
                    h["response"]["answers"][q] = p["response"]["answers"][q]
        out[k] = h
    return out


def answer_changes(pre: dict, post: dict, qtypes: dict) -> dict:
    """Per request question type: questions compared, argmax changes, any probability change, max |dp|."""
    rep = collections.defaultdict(lambda: {"questions": 0, "argmax_changed": 0, "probs_changed": 0, "max_abs_dp": 0.0})
    for k, a in pre.items():
        b = post[k]
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        for q, t in qtypes[k].items():
            x, y = a["response"]["answers"].get(q), b["response"]["answers"].get(q)
            if x is None or y is None:
                continue
            s = rep[t]
            s["questions"] += 1
            if x.get("type") == "choice" and y.get("type") == "choice":
                s["argmax_changed"] += x["choice"] != y["choice"]
                dps = [abs(x["probabilities"].get(c, 0) - y["probabilities"].get(c, 0))
                       for c in set(x["probabilities"]) | set(y["probabilities"])]
            elif x.get("type") == "noul" and y.get("type") == "noul":
                s["argmax_changed"] += (x["noul"] >= 0.5) != (y["noul"] >= 0.5)
                dps = [abs(x["noul"] - y["noul"])]
            else:
                s["argmax_changed"] += 1
                dps = [1.0]
            s["probs_changed"] += max(dps) > 0
            s["max_abs_dp"] = round(max(s["max_abs_dp"], max(dps)), 6)
    return dict(rep)


def index_report(rows, systems: dict[str, dict], B=2000, seed=0) -> dict:
    """Index point + CI per system and the paired differences, all from ONE set of bootstrap replicates."""
    import numpy as np
    from decision_index.scoring.report import compact
    from harness import bootstrap, score as S

    names = list(systems)
    benches = []
    for n in names:
        res = {k: compact(v) for k, v in systems[n].items()}
        S.require_results(rows, res, False, what=n)
        benches.append(S.decompose(rows, res))
    reps = bootstrap.replicate_skills(benches, B=B, seed=seed)
    idx = {n: S.index_from_skills({b: v[1] for b, v in r.items()}, True, None) for n, r in zip(names, reps)}

    def ci(x):
        lo, hi = np.quantile(x[1:], [0.025, 0.975])
        return {"point": round(float(x[0]), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}

    return {"index": {n: ci(v) for n, v in idx.items()}, "_raw": idx, "_ci": ci}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True, help="the post-fix job's out/readout_ab directory")
    ap.add_argument("--pre-tar", required=True, help="job-39gnn artifact art-zskmx (tar)")
    ap.add_argument("--d1", help="D1's in-job index arm (art-bm3e6 x_evals/out/units-child-index.json), optional")
    ap.add_argument("--params", default=str(PARAMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)

    from jobs.common import dev_subset as DS

    pins = json.loads(Path(a.params).read_text(encoding="utf-8"))["postfix_landing"]
    pre_b = pre_members(Path(a.pre_tar), pins)
    pre_v3, pre_v1 = read_rows(pre_b[V3W]), read_rows(pre_b[V1])
    post_dir = Path(a.post)
    post_v3 = read_rows((post_dir / "child.v3w.results.jsonl").read_bytes())
    post_v1 = read_rows((post_dir / "child.v1.results.jsonl").read_bytes())
    refit_p = post_dir / "child.v3w.results_refit.jsonl"
    post_rf = read_rows(refit_p.read_bytes()) if refit_p.exists() else None
    sub = DS.load_subset()
    ids = list(sub["run_ids"])
    rows = DS._rows(sub)
    qt = qtypes_of(rows)
    rep = {"schema": "ssdm.m2.readout_ab.postfix_landing.v1", "pins": {k: v for k, v in pins.items() if k != "_note"},
           "post_files_sha256": {p.name: sha256(p.read_bytes()) for p in sorted(post_dir.glob("child.*.results*.jsonl"))}}
    ok = True
    for n, d in (("pre_v1", pre_v1), ("pre_v3w", pre_v3), ("post_v1", post_v1), ("post_v3w", post_v3)):
        if set(d) != set(ids):
            raise SystemExit(f"{n}: run ids != the 2,008-row subset")
    # ---- v1 control: layout ordered is untouched by the fix
    ctl = {"post_vs_pre_job39gnn": same_rows(post_v1, pre_v1, ids)}
    if a.d1:
        d1 = json.loads(Path(a.d1).read_text(encoding="utf-8"))["index_results"]
        ctl["post_vs_d1_in_job"] = same_rows(post_v1, d1, ids)
    ctl["pass"] = all(v["rows_identical"] == v["rows"] for v in ctl.values() if isinstance(v, dict))
    ok &= ctl["pass"]
    rep["v1_control"] = ctl
    # ---- v3w: post - pre, split by request question type
    counts = collections.Counter(t for k in ids for t in qt[k].values())
    rep["question_types"] = dict(counts)
    rep["v3w_answer_changes"] = answer_changes(pre_v3, post_v3, qt)
    other = {t for t in counts if t != "score"}
    systems = {"pre": pre_v3, "post": post_v3,
               "pre+score_post": hybrid(pre_v3, post_v3, qt, {"score"}),
               "pre+other_post": hybrid(pre_v3, post_v3, qt, other),
               "post_v1": post_v1}
    if post_rf is not None:
        systems["post_refit"] = post_rf
    ir = index_report(rows, systems, B=a.B)
    raw, ci = ir["_raw"], ir["_ci"]
    rep["index"] = ir["index"]
    rep["v3w_delta"] = {
        "post_minus_pre": ci(raw["post"] - raw["pre"]),
        "score_questions": ci(raw["pre+score_post"] - raw["pre"]),
        "choice_noul": ci(raw["post"] - raw["pre+score_post"]),
        "score_questions_other_order": ci(raw["post"] - raw["pre+other_post"]),
        "choice_noul_other_order": ci(raw["pre+other_post"] - raw["pre"]),
        "post_minus_v1_control": ci(raw["post"] - raw["post_v1"]),
    }
    if post_rf is not None:
        rep["v3w_delta"]["post_refit_minus_pre"] = ci(raw["post_refit"] - raw["pre"])
    pre_pt = rep["index"]["pre"]["point"]
    rep["pre_matches_pinned_champion"] = abs(pre_pt - pins["pre_v3w_index"]) < 1e-3
    ok &= rep["pre_matches_pinned_champion"]
    rep["pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], "v1_control": ctl["pass"], "index": rep["index"], "v3w_delta": rep["v3w_delta"]},
                     indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
