"""Document-level train/eval check (coordinator 2026-10-09, after the NLI4CT premise finding of the 6-gram audit,
claude/b-gap-breakdown @f689a4b analysis/NGRAM_TEST_AUDIT_2026-10-09.md).

For every training row: its word 6-grams (the audit's normalisation and hashing, ``analysis/ngram_test_audit.py`` at
f689a4b, loaded from a pinned copy; every field: state, instructions, options, since a premise can sit in any of them)
are looked up in the audit's eval index (DEV ∪ TEST, ``eval.npz`` + ``eval-meta.json``, sha256-pinned). Boilerplate is
the audit's rule: eval df > 1 % of eval rows AND train df > 1 % of this draw's rows (computed in a first pass).
Postings of grams in more than 500 eval rows are skipped (as the audit). For each eval row with a STATE (document),
coverage = shared non-boilerplate state grams / the eval row's state grams. A row FAILS when some eval row's document
is covered >= 50 % with at least MIN_SHARED grams shared, unless the (training source, eval family) pair is public
reference text shared by design (ALLOWED: Wikipedia evidence / premises), which is reported, not failed.

    python -X utf8 -I -c "...data.doc_check.main([...])" --records <records.jsonl.gz> --audit-py <ngram_test_audit.py>
        --eval-dir <dir with eval.npz + eval-meta.json> --out <json>      # exit 1 on any non-allowed hit
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

AUDIT_PY_SHA256 = None  # filled from the pinned copy at run time and reported
EVAL_NPZ_SHA256 = "8477e924d9b3426b1a6b9be061ef1a19032918780bb32d6c74a4b790a5c28a45"
EVAL_META_SHA256 = "7e31aa20846b5a0bf34d20ee14d4cbe2cfe0dcfdc3521f8084a10924bb057823"
COVERAGE = 0.5
MIN_SHARED = 3
POST_CAP = 500
#: public reference text shared by design: (train source, eval family substring) pairs; every allowed hit is counted and
#: reported. Coordinator 2026-10-09 (v2d): ONLY the ANLI premise -> HoVer Wikipedia evidence pair stays allowed (one v2c
#: row); every other v2c hit (NLI4CT, glaive/When2Call -> BFCL tool schemas, HoVer/ESCI) is dropped, not allowlisted.
ALLOWED = [("anli", "hover")]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load_audit(path: str):
    spec = importlib.util.spec_from_file_location("ngram_test_audit_f689a4b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _allowed(src: str, eval_fam: str) -> bool:
    s, e = (src or "").lower(), (eval_fam or "").lower()
    return any(a in s and b in e for a, b in ALLOWED)


def check(records: str, audit_py: str, eval_dir: str) -> dict:
    import numpy as np
    A = _load_audit(audit_py)
    ed = Path(eval_dir)
    for f, want in (("eval.npz", EVAL_NPZ_SHA256), ("eval-meta.json", EVAL_META_SHA256)):
        got = _sha(ed / f)
        if got != want:
            raise SystemExit(f"{f}: sha256 {got} != pinned {want}")
    z = np.load(ed / "eval.npz")
    EH, EDF, EFB, POST, PSTART = z["h"], z["df"], z["fb"], z["post"], z["pstart"]
    meta = json.loads((ed / "eval-meta.json").read_text(encoding="utf-8"))
    n_eval = len(meta)
    ev_s = np.array([m[4] for m in meta], np.int64)
    eval_thr = 0.01 * n_eval

    def row_grams(r):
        f = A.row_fields(r)
        hs = [A.ngram_hashes(f[k])[0] for k in ("S", "Q", "O")]
        h = np.unique(np.concatenate(hs)) if hs else np.empty(0, np.uint64)
        pos = np.searchsorted(EH, h)
        pos[pos >= len(EH)] = 0
        return pos[EH[pos] == h]

    # pass 1: train df of the frequent eval grams -> boilerplate
    n_rows = 0
    tdf = collections.Counter()
    for r in A.read_jsonl_gz(records):
        n_rows += 1
        p = row_grams(r)
        for x in p[EDF[p] > eval_thr].tolist():
            tdf[x] += 1
    boiler = {x for x, c in tdf.items() if c > 0.01 * n_rows}
    # pass 2: per-row best eval-document coverage
    fails, allowed = [], []
    by_pair = collections.Counter()
    allowed_pair = collections.Counter()
    hist = collections.Counter()
    for r in A.read_jsonl_gz(records):
        p = row_grams(r)
        if not len(p):
            continue
        p = p[(EFB[p] & 1) > 0]  # grams that occur in some eval STATE
        p = p[(EDF[p] <= POST_CAP)]
        if boiler:
            p = np.array([x for x in p.tolist() if x not in boiler], np.int64)
        if not len(p):
            continue
        cnt = np.bincount(np.concatenate([POST[PSTART[x]:PSTART[x] + EDF[x]] for x in p.tolist()]), minlength=n_eval)
        cov = np.where(ev_s > 0, cnt / np.maximum(ev_s, 1), 0.0)
        b = int(np.argmax(cov))
        c = float(min(1.0, cov[b]))
        hist[">=0.5" if c >= 0.5 else ">=0.2" if c >= 0.2 else ">0"] += 1
        if c >= COVERAGE and int(cnt[b]) >= MIN_SHARED:
            d = {"id": r["id"], "source": r.get("source"), "family": r.get("family"), "eval": meta[b][1],
                 "eval_split": meta[b][0], "eval_family": meta[b][2], "eval_state_grams": int(ev_s[b]),
                 "shared": int(cnt[b]), "coverage": round(c, 4)}
            key = f"{r.get('source')}->{meta[b][2]}"
            if _allowed(r.get("source"), meta[b][2]):
                allowed.append(d)
                allowed_pair[key] += 1
            else:
                fails.append(d)
                by_pair[key] += 1
    return {"records": records, "n_rows": n_rows, "rule": {"coverage": COVERAGE, "min_shared_grams": MIN_SHARED,
            "post_cap": POST_CAP, "boilerplate": "eval df > 1% AND draw df > 1% (audit rule)",
            "fields": "train S+Q+O grams vs eval STATE grams", "allowed_pairs": ALLOWED},
            "eval": {"n_rows": n_eval, "eval_npz_sha256": EVAL_NPZ_SHA256, "eval_meta_sha256": EVAL_META_SHA256,
                     "audit_py_sha256": _sha(Path(audit_py)), "source": "claude/b-gap-breakdown @f689a4b"},
            "n_boilerplate_grams": len(boiler), "rows_by_best_doc_coverage": dict(hist),
            "n_fail": len(fails), "fail_by_pair": dict(by_pair.most_common()), "fails": fails[:2000],
            "n_allowed": len(allowed), "allowed_by_pair": dict(allowed_pair.most_common()), "allowed": allowed[:500],
            "pass": not fails}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="data.doc_check")
    ap.add_argument("--records", required=True)
    ap.add_argument("--audit-py", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = check(a.records, a.audit_py, a.eval_dir)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "n_boilerplate_grams", "rows_by_best_doc_coverage", "n_fail",
                                          "fail_by_pair", "n_allowed", "allowed_by_pair", "pass")}, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
