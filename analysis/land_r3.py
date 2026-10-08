"""Land R3 (job JOB=readout_g1v3: stock RWKV7-G1k-13.3B, eight readout variants, one model load) in one command.

    python -I -X utf8 analysis/land_r3.py --art <downloaded art-*.tar | dir holding it | extracted artifact dir> \
        --job job-fgu2c [--job-json saved_job_get.json] [--out report.json] [--tsv ledger_rows.tsv]

Prints, in order:
  1. integrity: artifact tar sha256/size vs ``autor job get`` (or --job-json), MANIFEST.jsonl phase-file sha256/bytes,
     any SHA256SUMS files, the job's own score.json/ece.json vs the recomputed ones, identical 2,008-row sets in every
     variant; reproducibility: the v2 arm vs job-zhyvg per row (max |dp|, choice flips, Index diff) and v1 vs job-uhxb5;
  2. per variant: Index [CI] + ECE, paired diff vs v2 (control), vs v1, and stepwise along the ladder;
  3. pre-declared arithmetic-mean ensembles v1 (+) {v2f, v3, v3w, v3r} (plus the reference v1 (+) v2), paired vs v1,
     vs the better member and vs the best single variant;
  4. area diffs vs v2, ToolRet/BFCL/BRIGHT/VAST/FinEntity/ForecastBench, >16-option accuracy, dominant-label share
     on When2Call/WinoGrande/HoVer;
  5. ledger-ready TSV rows (keep iff the paired CI vs v2 has lo > 0) + the recommendation paragraph;
  6. the winner paired vs Clef 27B and Clef-flash on the subset (analysis.paired_vs_clef --rows subset).

Nothing is re-implemented: rows come from jobs.common.dev_subset (the pre-registered subset), results through the
strict reader (harness.cli.load_results), points/CIs through harness.score.decompose + harness.bootstrap
(replicate_skills/summarize_one exactly as ``dev_subset score``; compare exactly as ``dev_subset compare``, B=2000 seed
0), ECE through harness.calibration (as ``dev_subset ece``). The ensemble rule is the fixed one already used for the
28.875 v1 (+) v2 reproduction (mean of the two probability vectors, renormalised; noul mean; non-ok -> the ok arm).
CPU only; the artifact is untrusted data (run with python -I; tar members are extracted with the "data" filter).
Exit 0 when no integrity/reproducibility problem was found, else 1 (the report is printed either way).

Self-test (2026-10-08, synthetic artifact in the fgu2c layout holding job-zhyvg's v2 results as "v2" and job-uhxb5's
parent-arm units as "v1", --expect v2,v1): v2 27.31 [24.53, 30.03], v1 24.85 [22.35, 27.71], v1 (+) v2 28.87
[26.13, 31.72] (vs v1 +4.03 [+1.82, +6.09], vs v2 +1.56 [+0.19, +3.14]), v2 vs Clef -34.83, vs Clef-flash -28.47 --
all equal to the landed receipts.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from analysis import paired_vs_clef as P  # noqa: E402  (also puts the kit on sys.path)
from harness import bootstrap, calibration, cli, mde, score  # noqa: E402
from harness.jsonl import read_results_strict  # noqa: E402
from jobs.common import dev_subset as D  # noqa: E402

HOME = Path(os.environ.get("USERPROFILE") or Path.home())
VARIANTS = ("v2", "v2f", "v3", "v3w", "v3r", "v3p", "v1", "v3c")  # job priority order
FAMILY = {"v2": "rwkv7-hf-g1-v2", "v2f": "rwkv7-hf-g1-v2f", "v3": "rwkv7-hf-g1-v3", "v3w": "rwkv7-hf-g1-v3w",
          "v3r": "rwkv7-hf-g1-v3r", "v3p": "rwkv7-hf-g1-v3p", "v1": "rwkv7-hf", "v3c": "rwkv7-hf-g1-v3c"}
LADDER = (("v2f", "v2"), ("v3", "v2f"), ("v3w", "v3"), ("v3r", "v3"), ("v3p", "v3"), ("v3c", "v3"))
ENSEMBLES = (("v1", "v2f"), ("v1", "v3"), ("v1", "v3w"), ("v1", "v3r"))  # pre-declared
REF_ENSEMBLE = ("v1", "v2")  # the reproduced 28.875 [26.131, 31.725]
FAMILIES = ("ToolRet", "BFCL", "BRIGHT", "VAST", "FinEntity", "ForecastBench")
DOMINANT = ("When2Call", "WinoGrande", "HoVer")
AREAS = ("knowledge", "language", "retrieval", "tools", "arts")
DEFAULTS = {
    "zhyvg_results": HOME / "r1_zhyvg_land/x/out/index/stock/results.jsonl",
    "zhyvg_tgz": HOME / "r1_zhyvg_land/art/phases/final.tgz",
    "uhxb5_units": HOME / "m2fc_uhxb5_out/x_cell/out/parent_arm/units-index.json",
    "clef": HOME / "clef_d2u54_art/r/clef-index-subset/index/decide/results.jsonl",
    "clef_flash": HOME / "research/CHLabs/review-artifacts/rwkv7-baseline-preserve-20261007/clef-flash-job-sgihj-results.jsonl",
}
AUTOR = ["wsl.exe", "-d", "bi-ubuntu", "--", "/root/.local/bin/autor", "--workspace", "default"]


# ----------------------------------------------------------------------------------------------- helpers
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(c, sign=False) -> str:
    f = "{:+.2f}" if sign else "{:.2f}"
    return f"{f.format(c['point'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def _safe_extract(tar_path: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        names = t.getnames()
        t.extractall(dest, filter="data")  # refuses absolute paths, .., links out of dest, devices
    return names


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.problems: list[str] = []

    def p(self, s=""):
        self.lines.append(s)
        print(s, flush=True)

    def bad(self, s):
        self.problems.append(s)
        self.p(f"  !! {s}")


# ----------------------------------------------------------------------------------------------- 1. integrity
def job_meta(job: str | None, job_json: str | None) -> dict | None:
    if job_json:
        return json.loads(Path(job_json).read_text(encoding="utf-8-sig"))
    if not job or job == "?":
        return None
    try:
        out = subprocess.run(AUTOR + ["job", "get", job, "--json"], capture_output=True, timeout=120, check=False)
        return json.loads(out.stdout.decode("utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        print(f"autor job get {job} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def locate(art: Path, work: Path, rep: Report) -> dict:
    """-> {tar, art_dir, out_dir}: the artifact tar (if any), the extracted artifact root, the final phase out/ dir."""
    tar = None
    if art.is_file():
        tar = art
    elif art.is_dir() and not (art / "MANIFEST.jsonl").is_file():
        tars = sorted(art.glob("*.tar"))
        if len(tars) != 1:
            raise SystemExit(f"{art}: expected one artifact .tar or an extracted dir with MANIFEST.jsonl; found {tars}")
        tar = tars[0]
    if tar is not None:
        art_dir = work / "art"
        _safe_extract(tar, art_dir)
    else:
        art_dir = art
    return {"tar": tar, "art_dir": art_dir}


def integrity(loc: dict, meta: dict | None, work: Path, rep: Report) -> dict:
    out = {"tar": None, "manifest": [], "sha256sums": [], "job": None}
    tar, art_dir = loc["tar"], loc["art_dir"]
    rep.p("## 1. Integrity")
    if tar is not None:
        s, n = sha256_file(tar), tar.stat().st_size
        out["tar"] = {"path": str(tar), "sha256": s, "bytes": n}
        rep.p(f"artifact tar {tar.name}: sha256 {s} ({n:,} bytes)")
    else:
        rep.p("artifact: extracted directory given (no tar): the artifact sha256 cannot be checked against the job")
    if meta:
        a = meta.get("artifact") or {}
        rc = meta.get("receipt") or {}
        out["job"] = {"job_id": meta.get("job_id"), "status": meta.get("status"), "artifact_id": a.get("artifact_id"),
                      "artifact_sha256": a.get("sha256"), "artifact_bytes": a.get("size_bytes"),
                      "cost_usd": (rc.get("charged_usd_micros") or 0) / 1e6 if rc else None,
                      "billed_seconds": rc.get("billed_seconds"), "image_digest": rc.get("image_digest"),
                      "result_ok": (meta.get("result") or {}).get("ok"),
                      "success_verdict": (meta.get("checks") or {}).get("success_verdict")}
        j = out["job"]
        rep.p(f"job {j['job_id']}: status {j['status']}, success check {j['success_verdict']}, result ok "
              f"{j['result_ok']}, artifact {j['artifact_id']} sha256 {j['artifact_sha256']}, cost ${j['cost_usd']}")
        if j["status"] != "succeeded":
            rep.bad(f"job status is {j['status']!r}, not 'succeeded'")
        if tar is not None:
            if j["artifact_sha256"] != out["tar"]["sha256"]:
                rep.bad(f"artifact sha256 MISMATCH: tar {out['tar']['sha256']} vs job {j['artifact_sha256']}")
            elif j["artifact_bytes"] not in (None, out["tar"]["bytes"]):
                rep.bad(f"artifact size MISMATCH: tar {out['tar']['bytes']} vs job {j['artifact_bytes']}")
            else:
                rep.p("  artifact sha256 + size == autor job get: OK")
    else:
        rep.bad("no job metadata (pass --job job-XXXXX or --job-json): artifact sha256 not checked")
    mf = art_dir / "MANIFEST.jsonl"
    if not mf.is_file():
        rep.bad(f"no MANIFEST.jsonl in {art_dir}")
    else:
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            f = art_dir / e["file"]
            got = sha256_file(f) if f.is_file() else None
            ok = bool(e.get("ok")) and got == e.get("sha256") and f.is_file() and f.stat().st_size == e.get("bytes")
            out["manifest"].append({**e, "sha256_got": got, "verified": ok})
            rep.p(f"  MANIFEST {e['phase']:<6} {e['file']}: {'OK' if ok else 'FAIL'}")
            if not ok:
                rep.bad(f"MANIFEST entry {e['file']} does not verify (ok={e.get('ok')}, sha {got} vs {e.get('sha256')})")
    markers = sorted(p.name for p in art_dir.glob("RWKV7_*_DONE"))
    rep.p(f"  markers: {markers or 'NONE'}")
    if "RWKV7_READOUT_G1V3_DONE" not in markers:
        rep.bad("no RWKV7_READOUT_G1V3_DONE marker (run did not finish cleanly)")
    # the final phase tgz holds out/ + logs/ at the end of the run
    final = None
    for name in ("final", "score", "index"):
        cand = [m for m in out["manifest"] if m["phase"] == name and m["verified"]]
        if cand:
            final = art_dir / cand[-1]["file"]
            break
    if final is None:
        cands = sorted((art_dir / "phases").glob("*.tgz"))
        final = cands[-1] if cands else None
        if final is None:
            raise SystemExit(f"{art_dir}: no phase tgz to read")
        rep.bad(f"no verified final/score/index phase in MANIFEST; reading {final.name} unverified")
    x = work / "x"
    _safe_extract(final, x)
    out["phase_read"] = final.name
    rep.p(f"  reading phase {final.name} -> {x}")
    for sums in sorted([*art_dir.rglob("SHA256SUMS*"), *x.rglob("SHA256SUMS*")]):
        n_ok = n_bad = 0
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            h, name = parts[0], parts[1].lstrip("*")
            f = (sums.parent / name)
            if f.is_file() and sha256_file(f) == h:
                n_ok += 1
            else:
                n_bad += 1
        out["sha256sums"].append({"file": str(sums), "ok": n_ok, "bad": n_bad})
        rep.p(f"  {sums.name} ({sums.parent.name}): {n_ok} ok, {n_bad} bad")
        if n_bad:
            rep.bad(f"{sums}: {n_bad} entries fail")
    if not out["sha256sums"]:
        rep.p("  SHA256SUMS: none in this artifact (the MANIFEST.jsonl phase hashes are the export receipts)")
    loc["out_dir"] = x / "out"
    return out


# ----------------------------------------------------------------------------------------------- arms
class Arms:
    """Rows + per-arm results/decompositions, each loaded once."""

    def __init__(self, B, seed):
        self.B, self.seed = B, seed
        self.subset = D.load_subset()
        self.rows = D._rows(self.subset)
        self.ids = list(self.subset["run_ids"])
        self.paths, self.raw, self.res, self.benches, self.score, self.ece = {}, {}, {}, {}, {}, {}

    def add(self, name, path):
        path = Path(path)
        raw = read_results_strict(path, self.ids)
        extra = sorted(set(raw) - set(self.ids))
        if extra:
            raise SystemExit(f"{name}: {path} has {len(extra)} rows outside the subset (e.g. {extra[:3]})")
        res = cli.load_results(str(path), self.ids, False)
        score.require_results(self.rows, res, False, name)
        self.paths[name], self.raw[name], self.res[name] = path, raw, res
        self.benches[name] = score.decompose(self.rows, res, edition=None)
        # == dev_subset.cmd_score
        rep = bootstrap.replicate_skills([self.benches[name]], B=self.B, seed=self.seed)[0]
        self.score[name] = bootstrap.summarize_one(rep, edition=None)
        # == dev_subset.cmd_ece
        self.ece[name] = calibration.report(calibration.fields(self.rows, res))
        return self

    def compare(self, a, b):
        # == dev_subset.cmd_compare
        out = bootstrap.compare(self.benches[a], self.benches[b], B=self.B, seed=self.seed, allow_unaligned=False)
        out["mde"] = mde.report(out["diff"]["index"]["se"], sum(len(x.clusters) for x in self.benches[a].values()))
        return out

    def idx(self, name):
        return self.score[name]["index"]

    def ece_s(self, name):
        e = self.ece[name]
        return {"board_protocol_ece": e["board_protocol"].get("ece"), "overall_ece_10bin": e["overall"].get("ece_10bin"),
                "overall_smece": e["overall"].get("smece")}


def units_to_jsonl(units_json: Path, dest: Path) -> Path:
    """job-uhxb5 units-index.json (index_results {run_id: record}) -> a results.jsonl (records unchanged)."""
    U = json.loads(units_json.read_text(encoding="utf-8"))["index_results"]
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(U):
            r = U[rid]
            assert r.get("run_id") == rid, rid
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dest


def answers_sha(raw: dict) -> str:
    h = hashlib.sha256()
    for rid in sorted(raw):
        r = raw[rid]
        h.update(json.dumps([rid, r.get("status"), (r.get("response") or {}).get("answers")], sort_keys=True).encode())
    return h.hexdigest()


def row_diff(ra: dict, rb: dict) -> dict:
    """Per-row prediction diff of two raw results maps (same ids): max |dp| over every option probability / noul."""
    maxdp, where = 0.0, None
    n_rows_diff = n_status = n_flip = n_q = n_keys = 0
    for rid in sorted(ra):
        a, b = ra[rid], rb[rid]
        if a.get("status") != b.get("status"):
            n_status += 1
            n_rows_diff += 1
            continue
        aa = ((a.get("response") or {}).get("answers")) or {}
        bb = ((b.get("response") or {}).get("answers")) or {}
        if aa != bb:
            n_rows_diff += 1
        if set(aa) != set(bb):
            n_keys += 1
        for q in set(aa) & set(bb):
            x, y = aa[q], bb[q]
            n_q += 1
            if x.get("type") == "choice" and y.get("type") == "choice":
                if x.get("choice") != y.get("choice"):
                    n_flip += 1
                px, py = x.get("probabilities") or {}, y.get("probabilities") or {}
                for k in set(px) | set(py):
                    d = abs(float(px.get(k, 0.0)) - float(py.get(k, 0.0)))
                    if d > maxdp:
                        maxdp, where = d, f"{rid}/{q}/{k}"
            elif x.get("type") == "noul" and y.get("type") == "noul":
                d = abs(float(x["noul"]) - float(y["noul"]))
                if d > maxdp:
                    maxdp, where = d, f"{rid}/{q}"
            elif x != y:
                n_flip += 1
    return {"max_abs_dp": maxdp, "argmax": where, "rows_differ": n_rows_diff, "status_differ": n_status,
            "question_keys_differ": n_keys, "questions": n_q, "choice_flips": n_flip,
            "identical": n_rows_diff == 0}


# ----------------------------------------------------------------------------------------------- ensembles
def ensemble_arith(r1: dict, r2: dict, dest: Path) -> dict:
    """The fixed rule (as the 28.875 reproduction): mean of the two probability vectors renormalised, noul mean,
    a non-ok arm falls back to the ok one; ties -> first option in r2's order. r2's row metadata is kept."""
    stats = collections.Counter()

    def comb(p1, p2):
        assert set(p1) == set(p2), (sorted(p1), sorted(p2))
        keys = list(p2)
        m = {k: 0.5 * (p1[k] + p2[k]) for k in keys}
        s = sum(m.values())
        if s <= 0:
            stats["zero_sum"] += 1
            m, s = {k: 1 / len(keys) for k in keys}, 1
        return {k: v / s for k, v in m.items()}

    assert set(r1) == set(r2)
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(r2):
            a, b = r1[rid], r2[rid]
            out = {k: v for k, v in b.items() if k not in ("payload", "raw_output", "response")}
            if a["status"] != "ok" or b["status"] != "ok":
                stats["fallback_single"] += 1
                src = b if b["status"] == "ok" else a
                out["status"] = src["status"]
                out["response"] = src.get("response")
                if "error" in src:
                    out["error"] = src["error"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            a1, a2 = a["response"]["answers"], b["response"]["answers"]
            assert set(a1) == set(a2), rid
            ans = {}
            for q, x2 in a2.items():
                x1 = a1[q]
                assert x1["type"] == x2["type"], (rid, q)
                if x2["type"] == "choice":
                    p = comb(x1["probabilities"], x2["probabilities"])
                    best = max(p.values())
                    ch = [k for k in p if p[k] == best]
                    stats["ties"] += len(ch) > 1
                    ans[q] = {"type": "choice", "choice": ch[0], "confidence": round(best, 4),
                              "probabilities": {k: round(v, 4) for k, v in p.items()}}
                    stats["choice"] += 1
                    stats["choice_changed_vs_one"] += ch[0] != x1["choice"] or ch[0] != x2["choice"]
                elif x2["type"] == "noul":
                    ans[q] = {"type": "noul", "noul": 0.5 * (x1["noul"] + x2["noul"])}
                    stats["noul"] += 1
                else:
                    raise ValueError(f"unknown answer type {x2['type']}")
            resp = dict(b["response"])
            resp["answers"], resp["model"] = ans, "ens-arith"
            out["status"], out["response"] = "ok", resp
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return dict(stats)


# ----------------------------------------------------------------------------------------------- 4. slices
def catalog_names(rows) -> dict:
    return {str(r["_evaluation"]["catalog_id"]): r["_evaluation"]["dataset"] for r in rows}


def find_cat(names: dict, prefix: str) -> str | None:
    hits = [c for c, n in names.items() if n.startswith(prefix)]
    return hits[0] if len(hits) == 1 else None


def gold_choice_qs(row):
    for k, q in sorted(row["questions"].items()):
        if q.get("type") != "choice" or not isinstance(q.get("criteria"), dict):
            continue
        ids = sorted(str(o) for o in q["criteria"])
        gold = (row.get("expected") or {}).get(k)
        if isinstance(gold, str) and gold in ids and len(ids) >= 2:
            yield k, ids, gold


def _choice(raw, rid, key):
    r = raw.get(rid) or {}
    if r.get("status") != "ok":
        return None
    c = (((r.get("response") or {}).get("answers") or {}).get(key) or {}).get("choice")
    return None if c is None else str(c)


def many_option_acc(rows, raw, k_min=17) -> dict:
    n = c = 0
    by = collections.Counter()
    for row in rows:
        rid = row["_evaluation"]["run_id"]
        for key, ids, gold in gold_choice_qs(row):
            if len(ids) < k_min:
                continue
            n += 1
            ok = _choice(raw, rid, key) == gold
            c += ok
            by[row["_evaluation"]["dataset"]] += 1
    return {"n_questions": n, "accuracy": c / n if n else float("nan"), "by_benchmark_n": dict(by)}


def dominant_label(rows, raw, cat) -> dict:
    chosen, gold, n, acc, unans = collections.Counter(), collections.Counter(), 0, 0, 0
    for row in rows:
        if str(row["_evaluation"]["catalog_id"]) != cat:
            continue
        rid = row["_evaluation"]["run_id"]
        for key, ids, g in gold_choice_qs(row):
            gold[g] += 1
            ch = _choice(raw, rid, key)
            if ch is None:
                unans += 1
                continue
            n += 1
            chosen[ch] += 1
            acc += ch == g
    if not n:
        return {"n": 0}
    lab, cnt = chosen.most_common(1)[0]
    glab, gcnt = gold.most_common(1)[0]
    ng = sum(gold.values())
    return {"n": n, "unanswered": unans, "dominant_label": lab, "dominant_share": cnt / n,
            "gold_share_of_that_label": gold.get(lab, 0) / ng, "gold_majority_label": glab,
            "gold_majority_share": gcnt / ng, "accuracy": acc / n}


# ----------------------------------------------------------------------------------------------- main
def land(a) -> dict:
    rep = Report()
    art = Path(a.art).resolve()
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix=f"{art.stem}-land_r3-", dir=art.parent))
    work.mkdir(parents=True, exist_ok=True)
    rep.p(f"# land_r3: {art}  (work dir {work}; B={a.B}, seed={a.seed})")
    meta = job_meta(a.job, a.job_json)
    loc = locate(art, work, rep)
    integ = integrity(loc, meta, work, rep)
    out_dir = loc["out_dir"]
    idx_dir = out_dir / "index"
    report = {"art": str(art), "work": str(work), "integrity": integ, "B": a.B, "seed": a.seed}

    arms = Arms(a.B, a.seed)
    names = catalog_names(arms.rows)
    present = []
    for v in VARIANTS:
        p = idx_dir / v / "results.jsonl"
        if not p.is_file():
            rep.p(f"  variant {v}: no results.jsonl (not run / cut short)")
            continue
        try:
            arms.add(v, p)
            present.append(v)
        except Exception as e:  # noqa: BLE001  (incomplete / malformed: reported, not scored)
            rep.bad(f"variant {v}: results unusable: {type(e).__name__}: {e}")
    missing = [v for v in VARIANTS if v not in present]
    rep.p(f"  variants scored: {present}; missing: {missing or 'none'}")
    for v in missing:
        if v in a.expect:
            rep.bad(f"expected variant {v} is missing or unusable")
    # identical row sets (strict reader already asserted == subset ids per arm; check pairwise anyway)
    sets = {v: frozenset(arms.raw[v]) for v in present}
    same = len(set(sets.values())) <= 1 and all(len(s) == len(arms.ids) for s in sets.values())
    rep.p(f"  row sets: {len(arms.ids)} subset ids; every variant answers exactly that set: {'OK' if same else 'FAIL'}")
    if not same:
        rep.bad(f"row sets differ: { {v: len(s) for v, s in sets.items()} }")
    status = {v: dict(collections.Counter(r.get("status") for r in arms.raw[v].values())) for v in present}
    rep.p(f"  status counts: {json.dumps(status)}")
    # the job's own score/ece files vs recomputed
    chk = {}
    for v in present:
        c = {}
        sp, ep = idx_dir / v / "score.json", idx_dir / v / "ece.json"
        if sp.is_file():
            js = json.loads(sp.read_text(encoding="utf-8"))
            theirs = js["slice_index"]["index"]["point"]
            c["score"] = {"job": theirs, "recomputed": arms.idx(v)["point"],
                          "ok": abs(theirs - arms.idx(v)["point"]) < 1e-6 and js.get("rows") == len(arms.ids)
                          and js.get("subset_ids_sha256") == arms.subset["ids_sha256"]}
        else:
            c["score"] = {"job": None, "ok": False}
        if ep.is_file():
            je = json.loads(ep.read_text(encoding="utf-8"))
            t, m = je["board_protocol"].get("ece"), arms.ece[v]["board_protocol"].get("ece")
            c["ece"] = {"job": t, "recomputed": m, "ok": t is not None and m is not None and abs(t - m) < 1e-9}
        else:
            c["ece"] = {"job": None, "ok": False}
        chk[v] = c
        if not (c["score"]["ok"] and c["ece"]["ok"]):
            rep.bad(f"variant {v}: job score/ece file disagrees with the recomputation or is missing: {c}")
    rep.p(f"  job score.json/ece.json == recomputed: "
          f"{'OK' if all(c['score']['ok'] and c['ece']['ok'] for c in chk.values()) else 'FAIL'}")
    report["variants_present"], report["status_counts"], report["job_file_check"] = present, status, chk

    # reproducibility
    repro = {}
    rep.p("\n### Reproducibility")
    refs = []
    zr = Path(a.zhyvg_results)
    if not zr.is_file() and Path(a.zhyvg_tgz).is_file():
        _safe_extract(Path(a.zhyvg_tgz), work / "zhyvg")
        zr = work / "zhyvg/out/index/stock/results.jsonl"
    if "v2" in present and zr.is_file():
        refs.append(("v2", "zhyvg", zr))
    if "v1" in present and Path(a.uhxb5_units).is_file():
        refs.append(("v1", "uhxb5", units_to_jsonl(Path(a.uhxb5_units), work / "uhxb5-v1.jsonl")))
    for v, ref, path in refs:
        arms.add(ref, path)
        d = row_diff(arms.raw[v], arms.raw[ref])
        c = arms.compare(v, ref)
        d.update({"index_arm": arms.idx(v)["point"], "index_ref": arms.idx(ref)["point"],
                  "index_diff": arms.idx(v)["point"] - arms.idx(ref)["point"], "paired": c["diff"]["index"],
                  "answers_sha256_arm": answers_sha(arms.raw[v]), "answers_sha256_ref": answers_sha(arms.raw[ref]),
                  "ref_path": str(path), "ref_sha256": sha256_file(path)})
        repro[f"{v}_vs_{ref}"] = d
        verdict = "IDENTICAL" if d["identical"] else "DIFFERS"
        rep.p(f"{v} arm vs job-{ref}: {verdict}; max |dp| {d['max_abs_dp']:.3g} (at {d['argmax']}); rows differ "
              f"{d['rows_differ']}, status differ {d['status_differ']}, choice flips {d['choice_flips']}/{d['questions']}; "
              f"Index {d['index_arm']:.4f} vs {d['index_ref']:.4f} (diff {d['index_diff']:+.4f}; paired "
              f"{ci(d['paired'], True)})")
        if v == "v2" and not d["identical"]:
            rep.bad("v2 control is NOT byte-identical to job-zhyvg in its answers (the control was declared identical)")
    if "v2" in present and not any(r[0] == "v2" for r in refs):
        rep.bad("job-zhyvg reference results not found: v2 reproducibility not checked")
    report["reproducibility"] = repro

    # 2. per variant
    rep.p("\n## 2. Per variant (frozen scorer; paired = harness.bootstrap.compare, B=%d seed %d)" % (a.B, a.seed))
    rep.p("| variant | family | Index [95% CI] | board ECE | ECE 10-bin | vs v2 (control) | p<=0 | vs v1 | p<=0 |")
    rep.p("|---|---|---|---|---|---|---|---|---|")
    per = {}
    for v in present:
        e = arms.ece_s(v)
        vs2 = arms.compare(v, "v2") if "v2" in present and v != "v2" else None
        vs1 = arms.compare(v, "v1") if "v1" in present and v != "v1" else None
        per[v] = {"family": FAMILY[v], "index": arms.idx(v), "areas": arms.score[v]["areas"],
                  "benchmarks": arms.score[v]["benchmarks"], "ece": e,
                  "vs_v2": vs2 and {"diff": vs2["diff"], "mde": vs2["mde"]},
                  "vs_v1": vs1 and {"diff": vs1["diff"], "mde": vs1["mde"]}}
        f2 = (ci(vs2["diff"]["index"], True), f"{vs2['diff']['index']['p_le_zero']:.3f}") if vs2 else ("control", "")
        f1 = (ci(vs1["diff"]["index"], True), f"{vs1['diff']['index']['p_le_zero']:.3f}") if vs1 else ("-", "")
        rep.p(f"| {v} | {FAMILY[v]} | {ci(arms.idx(v))} | {e['board_protocol_ece']:.4f} | {e['overall_ece_10bin']:.4f} "
              f"| {f2[0]} | {f2[1]} | {f1[0]} | {f1[1]} |")
    rep.p("\nLadder (stepwise paired diffs):")
    ladder = {}
    for x, y in LADDER:
        if x in present and y in present:
            c = arms.compare(x, y)
            ladder[f"{x}-{y}"] = c["diff"]
            rep.p(f"  {x} - {y}: {ci(c['diff']['index'], True)}  p<=0 {c['diff']['index']['p_le_zero']:.3f}")
        else:
            rep.p(f"  {x} - {y}: n/a (missing arm)")
    report["per_variant"], report["ladder"] = per, ladder

    # 3. ensembles
    rep.p("\n## 3. Pre-declared ensembles (arithmetic mean, fixed rule, no tuning)")
    singles = {v: arms.idx(v)["point"] for v in present if v not in ("zhyvg", "uhxb5")}
    best_single = max(singles, key=singles.get) if singles else None
    ens = {}
    todo = [e for e in ENSEMBLES] + [REF_ENSEMBLE]
    rep.p(f"| ensemble | Index [95% CI] | board ECE | vs v1 | vs better member | vs best single ({best_single}) | vs v2 |")
    rep.p("|---|---|---|---|---|---|---|")
    for m1, m2 in todo:
        label = f"{m1}+{m2}" + (" (reference)" if (m1, m2) == REF_ENSEMBLE else "")
        if m1 not in present or m2 not in present:
            rep.p(f"| {label} | n/a (missing arm) | | | | | |")
            continue
        name = f"ens_{m1}_{m2}"
        try:
            st = ensemble_arith(arms.raw[m1], arms.raw[m2], work / f"{name}.jsonl")
            arms.add(name, work / f"{name}.jsonl")
        except (AssertionError, ValueError) as e:
            rep.bad(f"ensemble {label}: rule not applicable ({type(e).__name__}: {e})")
            continue
        better = m1 if singles[m1] >= singles[m2] else m2
        cv1, cb, cbs = arms.compare(name, "v1"), arms.compare(name, better), arms.compare(name, best_single)
        cv2 = arms.compare(name, "v2") if "v2" in present else None
        ens[label] = {"members": [m1, m2], "stats": st, "index": arms.idx(name), "ece": arms.ece_s(name),
                      "areas": arms.score[name]["areas"], "benchmarks": arms.score[name]["benchmarks"],
                      "vs_v1": cv1["diff"], "vs_better_member": {"member": better, **cb["diff"]},
                      "vs_best_single": {"member": best_single, **cbs["diff"]}, "vs_v2": cv2 and cv2["diff"],
                      "path": str(work / f"{name}.jsonl")}
        rep.p(f"| {label} | {ci(arms.idx(name))} | {arms.ece_s(name)['board_protocol_ece']:.4f} | "
              f"{ci(cv1['diff']['index'], True)} | {better}: {ci(cb['diff']['index'], True)} | "
              f"{ci(cbs['diff']['index'], True)} | {ci(cv2['diff']['index'], True) if cv2 else '-'} |")
    report["ensembles"] = ens

    # 4. slices
    rep.p("\n## 4. Areas, families, >16-option accuracy, dominant-label share")
    rep.p("Area points (diff vs v2):")
    rep.p("| variant | " + " | ".join(AREAS) + " |")
    rep.p("|---|" + "---|" * len(AREAS))
    for v in present:
        cells = []
        for ar in AREAS:
            pt = arms.score[v]["areas"].get(ar, {}).get("point", float("nan"))
            dd = per[v]["vs_v2"]["diff"]["areas"].get(ar) if per[v]["vs_v2"] else None
            cells.append(f"{pt:.2f}" + (f" ({ci(dd, True)})" if dd else ""))
        rep.p(f"| {v} | " + " | ".join(cells) + " |")
    fam = {f: find_cat(names, f) for f in FAMILIES}
    rep.p("\nFamily benchmarks (point [CI]; diff vs v2):")
    rep.p("| variant | " + " | ".join(f"{f} (#{fam[f]})" for f in FAMILIES) + " |")
    rep.p("|---|" + "---|" * len(FAMILIES))
    fams = {}
    for v in present:
        cells, fams[v] = [], {}
        for f in FAMILIES:
            c = fam[f]
            b = arms.score[v]["benchmarks"].get(c) if c else None
            dd = per[v]["vs_v2"]["diff"]["benchmarks"].get(c) if (c and per[v]["vs_v2"]) else None
            fams[v][f] = {"catalog_id": c, "score": b, "diff_vs_v2": dd}
            cells.append((ci(b) if b else "n/a") + (f" ({dd['point']:+.2f})" if dd else ""))
        rep.p(f"| {v} | " + " | ".join(cells) + " |")
    dom_cat = {d: find_cat(names, d) for d in DOMINANT}
    rep.p("\n>16-option accuracy and dominant predicted label share (gold share of that label; gold majority share):")
    rep.p("| variant | >16-opt acc (n) | " + " | ".join(DOMINANT) + " |")
    rep.p("|---|---|" + "---|" * len(DOMINANT))
    slices = {}
    ens_names = {lab: f"ens_{e['members'][0]}_{e['members'][1]}" for lab, e in ens.items()}
    for v in present + list(ens_names):
        key = ens_names.get(v, v)
        mo = many_option_acc(arms.rows, arms.raw[key])
        dl = {d: dominant_label(arms.rows, arms.raw[key], dom_cat[d]) if dom_cat[d] else {"n": 0} for d in DOMINANT}
        slices[v] = {"many_option": mo, "dominant": dl}
        cells = [f"{x['dominant_label']} {x['dominant_share']:.2f} (gold {x['gold_share_of_that_label']:.2f}; "
                 f"maj {x['gold_majority_share']:.2f})" if x.get("n") else "n/a" for x in dl.values()]
        rep.p(f"| {v} | {mo['accuracy']:.3f} ({mo['n_questions']}) | " + " | ".join(cells) + " |")
    report["families"], report["slices"] = fams, slices

    # 5. ledger + recommendation
    rep.p("\n## 5. Ledger rows (TSV: utc agent job sha recipe index lo hi cost status receipt note)")
    j = integ.get("job") or {}
    cost = j.get("cost_usd")
    nvar = max(1, len(present))
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    receipt = j.get("artifact_id") or "?"
    job = j.get("job_id") or a.job or "?"
    tsv = []
    for v in present:
        i = arms.idx(v)
        d2 = per[v]["vs_v2"]["diff"]["index"] if per[v]["vs_v2"] else None
        keep = bool(d2 and d2["lo"] > 0)
        note = (f"control (= job-zhyvg: {'identical' if repro.get('v2_vs_zhyvg', {}).get('identical') else 'NOT identical/unchecked'})"
                if v == "v2" else f"paired vs v2 {ci(d2, True)} p<=0 {d2['p_le_zero']:.3f}" if d2 else "no v2 control")
        if per[v]["vs_v1"]:
            note += f"; vs v1 {ci(per[v]['vs_v1']['diff']['index'], True)}"
        note += f"; board ECE {per[v]['ece']['board_protocol_ece']:.4f}; non-ok {sum(n for s, n in status[v].items() if s != 'ok')}/{len(arms.ids)}"
        note += f"; cost = 1/{nvar} share of the one-load job ${cost:.2f}" if cost is not None else "; cost unknown"
        c = f"{cost / nvar:.2f}" if cost is not None else "?"
        tsv.append("\t".join([utc, a.agent, job, a.code_sha, f"stock rwkv7 readout {v} ({FAMILY[v]}), DEV subset 2008",
                              f"{i['point']:.2f}", f"{i['lo']:.2f}", f"{i['hi']:.2f}", c,
                              "keep" if keep else "discard", receipt, note]))
    for lab, e in ens.items():
        d2 = e["vs_v2"]["index"] if e["vs_v2"] else None
        keep = bool(d2 and d2["lo"] > 0)
        note = (f"CPU post-hoc arithmetic-mean ensemble (two templates: not a training-cell readout); vs v1 "
                f"{ci(e['vs_v1']['index'], True)}; vs {e['vs_better_member']['member']} "
                f"{ci(e['vs_better_member']['index'], True)}"
                + (f"; vs v2 {ci(d2, True)}" if d2 and e["vs_better_member"]["member"] != "v2" else ""))
        tsv.append("\t".join([utc, a.agent, job, a.code_sha, f"ensemble arith {lab}", f"{e['index']['point']:.2f}",
                              f"{e['index']['lo']:.2f}", f"{e['index']['hi']:.2f}", "0.00",
                              "keep" if keep else "discard", receipt, note]))
    for t in tsv:
        rep.p(t)
    report["ledger_tsv"] = tsv
    if a.tsv:
        Path(a.tsv).write_text("\n".join(tsv) + "\n", encoding="utf-8", newline="\n")

    # winner: single template only
    cand = [v for v in present if v != "v2" and per[v]["vs_v2"] and per[v]["vs_v2"]["diff"]["index"]["lo"] > 0]
    if cand:
        winner = max(cand, key=lambda v: arms.idx(v)["point"])
        why = f"its paired CI vs the v2 control excludes zero ({ci(per[winner]['vs_v2']['diff']['index'], True)})"
    else:
        winner = "v2" if "v2" in present else best_single
        why = ("no variant's paired CI vs the v2 control has lo > 0, so the control readout stays"
               if "v2" in present else "no v2 control in this artifact; highest point")
    best_pt = max(present, key=lambda v: arms.idx(v)["point"]) if present else None
    best_ens = max(ens, key=lambda k: ens[k]["index"]["point"]) if ens else None
    para = (f"Recommendation: use {winner} ({FAMILY.get(winner, '?')}) as the readout for training cells: {why}. "
            f"{winner} scores {ci(arms.idx(winner))} on the 2,008-row DEV subset (board ECE "
            f"{arms.ece_s(winner)['board_protocol_ece']:.4f}). Highest point single variant: {best_pt} "
            f"{ci(arms.idx(best_pt))}" + (f" (not separated from v2: {ci(per[best_pt]['vs_v2']['diff']['index'], True)})"
                                          if best_pt != winner and per[best_pt]["vs_v2"] else "") + ". "
            + (f"Best ensemble {best_ens} reaches {ci(ens[best_ens]['index'])} (vs {ens[best_ens]['vs_best_single']['member']} "
               f"{ci(ens[best_ens]['vs_best_single']['index'], True)}), but an ensemble needs two templates per row "
               f"(two readouts per training example), so it is reported, not recommended, for cells. " if best_ens else "")
            + ("Integrity problems above must be resolved first." if rep.problems else
               "All integrity and reproducibility checks above passed."))
    rep.p("\n" + para)
    report["winner"], report["recommendation"] = winner, para

    # 6. paired vs Clef
    rep.p(f"\n## 6. Winner ({winner}) paired vs Clef 27B and Clef-flash (DEV subset)")
    if Path(a.clef).is_file() and Path(a.clef_flash).is_file() and winner in arms.paths:
        try:
            r = P.run(a.clef, a.clef_flash, str(arms.paths[winner]), "subset", "0.2.1", a.B, a.seed)
            report["vs_clef"] = P.brief(r)
            report["vs_clef_full"] = r
            rep.p(P.table(r))
        except Exception as e:  # noqa: BLE001
            rep.bad(f"paired_vs_clef failed: {type(e).__name__}: {e}")
    else:
        rep.bad(f"Clef inputs missing ({a.clef} / {a.clef_flash}): pairing skipped")

    rep.p(f"\nPROBLEMS: {len(rep.problems)}" + ("" if not rep.problems else "\n  - " + "\n  - ".join(rep.problems)))
    report["problems"] = rep.problems
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 analysis/land_r3.py", description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True, help="artifact tar, a dir holding exactly one .tar, or an extracted artifact dir")
    ap.add_argument("--job", default="job-fgu2c", help="job id for `autor job get` (artifact sha256, cost, status)")
    ap.add_argument("--job-json", help="saved `autor job get --json` output instead of calling autor")
    ap.add_argument("--work", help="scratch dir for extraction + ensemble files (default: a fresh dir next to --art)")
    ap.add_argument("--zhyvg-results", default=str(DEFAULTS["zhyvg_results"]))
    ap.add_argument("--zhyvg-tgz", default=str(DEFAULTS["zhyvg_tgz"]), help="used if --zhyvg-results is absent")
    ap.add_argument("--uhxb5-units", default=str(DEFAULTS["uhxb5_units"]))
    ap.add_argument("--clef", default=str(DEFAULTS["clef"]))
    ap.add_argument("--clef-flash", default=str(DEFAULTS["clef_flash"]))
    ap.add_argument("--agent", default="agent-b")
    ap.add_argument("--code-sha", default="0290fa8", help="job code commit for the ledger sha column")
    ap.add_argument("--expect", default=",".join(VARIANTS), type=lambda s: [x for x in s.split(",") if x],
                    help="variants that must be present (default: all eight)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="full JSON report")
    ap.add_argument("--tsv", help="also write the ledger rows here")
    a = ap.parse_args(argv)
    r = land(a)
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
