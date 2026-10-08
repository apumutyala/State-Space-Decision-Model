"""Agent B: independent rescore of a ledger row from the job's RAW per-row predictions, with the FROZEN scorer.

    CUDA_VISIBLE_DEVICES= python -X utf8 -I analysis/rescore_keep.py --artifact <downloaded job artifact dir> --job job-xxxxx \
        [--ledger origin/state | <ledger.tsv>] [--claim INDEX,LO,HI] [--predictions SUBSTR] [--job-score SUBSTR[:dotted.key]] \
        [--rows auto|subset|full] [--edition 0.2.1|0.3] [--omit-gsm8k03] [--tol 0.01]

What it does (AGENTS.md: "B also rescores A's keep results from raw predictions; a mismatch marks the row invalid"):

1. Walks the artifact dir as UNTRUSTED DATA: plain files plus the members of every .tar/.tgz/.tar.gz in it (nested archives
   too, read in memory; nothing is extracted, links are skipped). Run it with ``python -X utf8 -I`` so nothing in the artifact or
   the cwd can shadow an import.
2. Finds the raw per-row predictions: ``results.jsonl`` files (harness/clef layout: ``index/<engine>/results.jsonl``) and
   m2 unit caches (``units[-child]-{index,g13}.json``, key ``index_results``). Identical copies (same sha256) collapse. If
   more than one distinct prediction set covers the expected rows, ``--predictions`` must pick one (it is never chosen by
   which one matches the claim).
3. Checks the row set: the predicted ``run_id`` set must EQUAL the expected eval manifest's id set: the pre-registered
   2,008-row m1 DEV subset (``jobs/common/dev_subset_m1.json``, ids + DEV-manifest hash checked by ``load_subset``) or the
   full DEV slice (``harness/manifests/dev-manifest.jsonl.gz``) for the edition. No missing, no extra rows.
4. Recomputes the Index by calling the official entry points unchanged: ``jobs.common.dev_subset.cmd_score`` (subset) or
   ``harness.cli.cmd_score`` (full DEV); both run ``harness.jsonl`` (strict reader) -> ``harness.score.require_results``
   -> ``harness.score.decompose`` -> ``harness.bootstrap.replicate_skills`` / ``summarize_one``. Nothing in the scorer is
   re-implemented here. Unit caches are transcoded byte-for-byte-equivalently into a temporary results.jsonl first
   (one JSON object per line, ``run_id`` checked against the cache key).
5. Integrity: every ``SHA256SUMS`` and m2 ``MANIFEST.jsonl`` found is verified against the files beside it (absent files
   are reported, a present file with a different hash fails); ``job.json``'s artifact sha256 is checked against the
   downloaded ``<artifact_id>.tar`` when both are there.
6. Compares point/lo/hi with the job's own score json AND with the ledger claim, each within ``--tol`` (0.01).
   Prints PASS or MISMATCH (with reasons) and a ready-to-append ``results/ledger.tsv`` line
   (utc agent job sha recipe index lo hi cost status receipt note): status ``invalid`` on a mismatch.

Exit: 0 PASS, 1 MISMATCH, 2 the environment cannot score (kit / DEV rows missing): no ledger line then.
CPU only; the frozen paths (harness/score.py, bootstrap.py, calibration.py, slicing.py, editions_data/, manifests/,
train/promotion.py) are only imported, never touched.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # python -I drops the script dir; the repo root is trusted code, the artifact is not
    sys.path.insert(0, str(ROOT))

LEDGER_COLS = ("utc", "agent", "job", "sha", "recipe", "index", "lo", "hi", "cost", "status", "receipt", "note")
ARCHIVE_SUFFIXES = (".tar", ".tgz", ".tar.gz")
MAX_MEMBER_BYTES = 4 << 30
UNITS_RE = re.compile(r"(^|/)units-(child-)?(index|g13)\.json$")
NOTE_CLAIM_RE = re.compile(r"Index\s+(-?\d+(?:\.\d+)?)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


# ----------------------------------------------------------------------------- untrusted artifact as a virtual tree
class Member:
    """One file of the artifact: on disk or inside an archive. ``label`` = '<disk path>' or '<archive>!<member>[!...]'."""

    def __init__(self, label: str, ns: str, rel: str, size: int, reader: Callable[[], bytes]):
        self.label, self.ns, self.rel, self.size, self._reader = label, ns, rel, size, reader
        self._sha = None

    def read(self) -> bytes:
        if self.size > MAX_MEMBER_BYTES:
            raise ValueError(f"{self.label}: {self.size} bytes exceeds the {MAX_MEMBER_BYTES}-byte read cap")
        return self._reader()

    def sha256(self) -> str:
        if self._sha is None:
            self._sha = hashlib.sha256(self.read()).hexdigest()
        return self._sha

    @property
    def name(self) -> str:
        return self.rel.rsplit("/", 1)[-1]


def _norm(p: str) -> str:
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def _tar_members(tf: tarfile.TarFile, label: str, depth: int, out: list, notes: list):
    for ti in tf.getmembers():
        if not ti.isreg():  # dirs, symlinks, hardlinks, devices: never followed
            continue
        rel = _norm(ti.name)
        if not rel or ".." in rel.split("/"):
            notes.append(f"skipped unsafe member name {ti.name!r} in {label}")
            continue
        lab = f"{label}!{rel}"
        rd = (lambda t=tf, i=ti: t.extractfile(i).read())
        m = Member(lab, label, rel, ti.size, rd)
        out.append(m)
        if _is_archive(rel) and depth < 3:
            try:
                inner = tarfile.open(fileobj=io.BytesIO(m.read()), mode="r:*")
                _tar_members(inner, lab, depth + 1, out, notes)
            except (tarfile.TarError, OSError, EOFError) as e:
                notes.append(f"unreadable nested archive {lab}: {e}")


def walk_artifact(root: Path) -> tuple[list[Member], list[str]]:
    out: list[Member] = []
    notes: list[str] = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns.sort()
        for fn in sorted(fns):
            p = Path(dp) / fn
            if p.is_symlink() or not p.is_file():
                continue
            rel = _norm(str(p.relative_to(root)))
            parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
            m = Member(rel, f"<dir>/{parent}", fn, p.stat().st_size, (lambda q=p: q.read_bytes()))
            out.append(m)
            if _is_archive(fn):
                try:
                    tf = tarfile.open(p, mode="r:*")
                    _tar_members(tf, rel, 1, out, notes)
                except (tarfile.TarError, OSError, EOFError) as e:
                    notes.append(f"unreadable archive {rel}: {e}")
    return out, notes


def _dir_of(m: Member) -> tuple[str, str]:
    """(namespace key, directory) so siblings can be resolved: for disk files the namespace is the disk root."""
    if "!" in m.label:
        ns = m.ns
        d = m.rel.rsplit("/", 1)[0] if "/" in m.rel else ""
    else:
        ns = "<dir>"
        d = m.label.rsplit("/", 1)[0] if "/" in m.label else ""
    return ns, d


def _index_by_path(members: list[Member]) -> dict[tuple[str, str], Member]:
    out = {}
    for m in members:
        ns = m.ns if "!" in m.label else "<dir>"
        rel = m.rel if "!" in m.label else m.label
        out[(ns, rel)] = m
    return out


def _join(d: str, rel: str) -> str:
    parts = [x for x in (d + "/" + _norm(rel)).split("/") if x and x != "."]
    st: list[str] = []
    for x in parts:
        if x == "..":
            if st:
                st.pop()
        else:
            st.append(x)
    return "/".join(st)


# ----------------------------------------------------------------------------- integrity
def _grown_log(m: Member, want: str) -> bool:
    """A ``*.log`` whose listed hash is that of a newline-terminated PREFIX of the present file: an append-only log that
    grew after SHA256SUMS was written (jobs/clef_baseline/run_job.sh writes SHA256SUMS, then logs the final stage).
    Nothing that was hashed changed; reported as a note, not a failure. Never applies to any other file."""
    if not m.name.endswith(".log"):
        return False
    b = m.read()
    i = b.find(b"\n")
    while i != -1:
        if hashlib.sha256(b[: i + 1]).hexdigest() == want:
            return True
        i = b.find(b"\n", i + 1)
    return False


def check_integrity(members: list[Member], root: Path) -> dict[str, Any]:
    by_path = _index_by_path(members)
    rep: dict[str, Any] = {"sha256sums": [], "manifests": [], "job_json": None, "failures": [], "notes": [],
                           "verified": set()}
    for m in members:
        if m.name == "SHA256SUMS":
            ns, d = _dir_of(m)
            ent = {"file": m.label, "ok": 0, "absent": 0, "bad": [], "grown_logs": []}
            for line in m.read().decode("utf-8", "replace").splitlines():
                mm = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line.strip())
                if not mm:
                    continue
                key = (ns, _join(d, mm.group(2)))
                tgt = by_path.get(key)
                want = mm.group(1).lower()
                if tgt is None:
                    ent["absent"] += 1
                elif tgt.sha256() != want:
                    if _grown_log(tgt, want):
                        ent["grown_logs"].append(mm.group(2))
                    else:
                        ent["bad"].append(mm.group(2))
                else:
                    ent["ok"] += 1
                    rep["verified"].add(key)
            if ent["grown_logs"]:
                rep["notes"].append(f"{m.label}: {ent['grown_logs']} grew after SHA256SUMS (listed hash = a line "
                                    "prefix of the present log)")
            if ent["bad"]:
                rep["failures"].append(f"{m.label}: sha256 mismatch for {ent['bad'][:5]}")
            rep["sha256sums"].append(ent)
        elif m.name == "MANIFEST.jsonl":
            ns, d = _dir_of(m)
            ent = {"file": m.label, "ok": 0, "absent": 0, "bad": []}
            for line in m.read().decode("utf-8", "replace").split("\n"):
                try:
                    r = json.loads(line) if line.strip() else None
                except json.JSONDecodeError:
                    r = None
                if not isinstance(r, dict) or not isinstance(r.get("file"), str) or not isinstance(r.get("sha256"), str):
                    continue
                key = (ns, _join(d, r["file"]))
                tgt = by_path.get(key)
                if tgt is None:
                    ent["absent"] += 1
                elif tgt.sha256() != r["sha256"].lower():
                    ent["bad"].append(r["file"])
                else:
                    ent["ok"] += 1
                    rep["verified"].add(key)
            if ent["bad"]:
                rep["failures"].append(f"{m.label}: sha256 mismatch for {ent['bad'][:5]}")
            rep["manifests"].append(ent)
    jj = root / "job.json"
    if jj.is_file():
        try:
            art = (json.loads(jj.read_text(encoding="utf-8")) or {}).get("artifact") or {}
            aid, want = art.get("artifact_id"), (art.get("sha256") or "").lower()
            tar = root / f"{aid}.tar" if isinstance(aid, str) and re.fullmatch(r"[\w.-]+", aid) else None
            if tar is not None and tar.is_file() and want:
                got = hashlib.sha256(tar.read_bytes()).hexdigest()
                rep["job_json"] = {"artifact_id": aid, "ok": got == want}
                if got != want:
                    rep["failures"].append(f"job.json artifact sha256 {want[:12]} != {tar.name} {got[:12]}")
            else:
                rep["job_json"] = {"artifact_id": aid, "ok": None, "note": "no downloaded tar to check"}
        except (ValueError, OSError) as e:
            rep["job_json"] = {"error": str(e)}
    return rep


# ----------------------------------------------------------------------------- predictions
def _candidates(members: list[Member]) -> list[Member]:
    out = []
    for m in members:
        r = m.rel if "!" in m.label else m.label
        if m.name in ("results.jsonl", "results.jsonl.gz") or m.name.endswith(".results.jsonl") or UNITS_RE.search(r):
            out.append(m)
    return out


def _pred_ids(m: Member) -> tuple[set[str] | None, str | None]:
    """Cheap id set for selection only (the official strict reader re-reads the file when scoring)."""
    try:
        raw = m.read()
        if m.name.endswith(".gz"):
            import gzip
            raw = gzip.decompress(raw)
        if m.name.endswith(".json"):
            res = json.loads(raw.decode("utf-8")).get("index_results")
            if not isinstance(res, dict) or not res:
                return None, "no index_results"
            return set(res), None
        ids = set()
        for line in raw.decode("utf-8").split("\n"):
            if line.strip():
                ids.add(json.loads(line).get("run_id"))
        return ids, None
    except (ValueError, UnicodeDecodeError, AttributeError, OSError, EOFError) as e:
        return None, f"{type(e).__name__}: {e}"


def materialize(m: Member, workdir: Path) -> tuple[Path, list[str]]:
    """Path to a results.jsonl the official loader can read. results files are copied byte-identical; unit caches are
    transcoded (one object per line, sorted by run_id; a value whose run_id differs from its key is a problem)."""
    problems: list[str] = []
    raw = m.read()
    if m.name.endswith(".json"):
        res = json.loads(raw.decode("utf-8"))["index_results"]
        p = workdir / "results.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for k in sorted(res):
                v = res[k]
                if not isinstance(v, dict):
                    problems.append(f"index_results[{k!r}] is not an object")
                    continue
                if v.get("run_id") != k:
                    problems.append(f"index_results key {k!r} carries run_id {v.get('run_id')!r}")
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        return p, problems
    p = workdir / ("results.jsonl.gz" if m.name.endswith(".gz") else "results.jsonl")
    p.write_bytes(raw)
    return p, problems


# ----------------------------------------------------------------------------- job's own score json
def _dig(d: Any, path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _ci3(x: Any) -> dict | None:
    if isinstance(x, dict) and all(isinstance(x.get(k), (int, float)) for k in ("point", "lo", "hi")):
        return {k: float(x[k]) for k in ("point", "lo", "hi")}
    return None


def find_job_score(members: list[Member], pred: Member, override: str | None) -> tuple[dict | None, str, dict]:
    """(ci, where, extra) for the job's own reported score of THIS prediction set."""
    by_path = _index_by_path(members)

    def load(m):
        try:
            return json.loads(m.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    if override:
        sub, _, key = override.partition(":")
        hits = [m for m in members if sub in m.label and m.name.endswith(".json")]
        for m in hits:
            d = load(m)
            for k in ([key] if key else ["slice_index.index", "index", "scores.parent.index", "headline.value.index"]):
                ci = _ci3(_dig(d, k))
                if ci:
                    return ci, f"{m.label}:{k}", {"edition": _dig(d, "slice_index.edition"), "json": d}
        return None, f"--job-score {override!r}: no file/key with point/lo/hi", {}
    ns, d = _dir_of(pred)
    rel_pred = pred.rel if "!" in pred.label else pred.label
    tries: list[tuple[str, str, str]] = []  # (ns, path, key)
    if pred.name.startswith("results.jsonl"):
        parent = d.rsplit("/", 1)
        if len(parent) == 2:
            tries.append((ns, f"{parent[0]}/score_{parent[1]}.json", "slice_index.index"))
        elif d:
            tries.append((ns, f"score_{d}.json", "slice_index.index"))
    m_units = UNITS_RE.search(rel_pred)
    if m_units:
        arm = "child" if m_units.group(2) else "parent"
        for m in members:
            if m.name == "land_summary.json":
                tries.append((_dir_of(m)[0], m.rel if "!" in m.label else m.label, f"scores.{arm}.index"))
            elif m.name == "first_cell_result.json":
                key = "index_dev_subset." + ("child" if arm == "child" else "parent_stock_readout") + ".index"
                tries.append((_dir_of(m)[0], m.rel if "!" in m.label else m.label, key))
            elif m.name == "report.json":
                tries.append((_dir_of(m)[0], m.rel if "!" in m.label else m.label, f"verdicts.index.scores.{arm}.index"))
    for tns, tpath, key in tries:
        m = by_path.get((tns, tpath))
        if m is None:
            continue
        dd = load(m)
        ci = _ci3(_dig(dd, key))
        if ci:
            ed = _dig(dd, "slice_index.edition") or _dig(dd, key.rsplit(".", 1)[0] + ".edition")
            return ci, f"{m.label}:{key}", {"edition": ed, "B": _dig(dd, "scores.B"), "seed": _dig(dd, "scores.seed")}
    return None, "no job score json found for these predictions (pass --job-score)", {}


# ----------------------------------------------------------------------------- ledger
def read_ledger(src: str) -> list[dict[str, str]]:
    if Path(src).is_file():
        text = Path(src).read_text(encoding="utf-8")
    else:
        text = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{src}:results/ledger.tsv"], text=True,
                                       encoding="utf-8")
    lines = [x for x in text.split("\n") if x.strip()]
    hdr = lines[0].split("\t")
    return [dict(zip(hdr, x.split("\t"))) for x in lines[1:]]


def ledger_claim(rows: list[dict[str, str]], job: str) -> tuple[dict | None, dict | None, str]:
    mine = [r for r in rows if r.get("job") == job]
    if not mine:
        return None, None, f"no ledger row for {job}"
    keep = [r for r in mine if r.get("status") == "keep"]
    row = (keep or mine)[-1]
    try:
        if all((row.get(k) or "").strip() for k in ("index", "lo", "hi")):
            return row, {"point": float(row["index"]), "lo": float(row["lo"]), "hi": float(row["hi"])}, "ledger columns"
    except ValueError:
        pass
    mm = NOTE_CLAIM_RE.search(row.get("note") or "")
    if mm:
        return row, dict(zip(("point", "lo", "hi"), map(float, mm.groups()))), "ledger note (index/lo/hi columns empty)"
    return row, None, "ledger row has no index/lo/hi"


def _tsv(x: Any) -> str:
    return re.sub(r"[\t\r\n]+", " ", "" if x is None else str(x)).strip()


# ----------------------------------------------------------------------------- official scoring
def official_expected_rows(mode: str, edition, gsm8k03: bool):
    from harness.cli import load_slice
    from jobs.common import dev_subset as DS

    if mode == "subset":
        subset = DS.load_subset()  # checks ids_sha256 and that it was built from the committed DEV manifest
        return DS._rows(subset, edition, gsm8k03), subset  # noqa: SLF001 - the official subset row loader
    return load_slice("dev", edition, gsm8k03), None


def official_score(mode: str, results_path: Path, edition, gsm8k03: bool, B: int, seed: int, subset) -> dict:
    ns = argparse.Namespace(results=str(results_path), B=B, seed=seed, edition=edition, omit_gsm8k03=not gsm8k03,
                            allow_missing=False, allow_partial=False, slice="dev")
    if mode == "subset":
        from jobs.common import dev_subset as DS

        return DS.cmd_score(ns, subset)
    from harness import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.cmd_score(ns)
    return json.loads(buf.getvalue())


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rescore_keep", description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifact", required=True, help="downloaded job artifact dir (untrusted)")
    ap.add_argument("--job", required=True)
    ap.add_argument("--ledger", default="origin/state", help="git ref holding results/ledger.tsv, or a ledger.tsv path")
    ap.add_argument("--claim", help="INDEX,LO,HI: overrides the ledger row's columns (e.g. a claim made in prose)")
    ap.add_argument("--predictions", help="substring or glob of the prediction file label to score (required if ambiguous)")
    ap.add_argument("--job-score", help="SUBSTR[:dotted.key] of the job's own score json (default: auto)")
    ap.add_argument("--rows", default="auto", choices=["auto", "subset", "full"])
    ap.add_argument("--edition", default=None, choices=["0.2.1", "0.3"], help="default: the job score's edition, else 0.2.1")
    ap.add_argument("--omit-gsm8k03", action="store_true")
    ap.add_argument("--B", type=int, default=None, help="default: the job score's B, else 2000")
    ap.add_argument("--seed", type=int, default=None, help="default: the job score's seed, else 0")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--agent", default="claude")
    ap.add_argument("--workdir", help="keep the materialised results.jsonl here (default: a temporary dir)")
    ap.add_argument("--json", action="store_true", help="also print the full report as JSON")
    a = ap.parse_args(argv)

    root = Path(a.artifact).resolve()
    if not root.is_dir():
        print(f"--artifact {root} is not a directory", file=sys.stderr)
        return 2
    try:
        import numpy  # noqa: F401
        import decision_index  # noqa: F401
        from harness import config
    except ImportError as e:
        print(f"cannot score here: {e} (needs numpy + the pinned decision_index kit)", file=sys.stderr)
        return 2

    reasons: list[str] = []
    rep: dict[str, Any] = {"job": a.job, "artifact": str(root), "kit_commit": config.KIT_COMMIT}

    # ledger claim
    try:
        lrows = read_ledger(a.ledger)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"cannot read ledger {a.ledger}: {e}", file=sys.stderr)
        return 2
    lrow, claim, claim_src = ledger_claim(lrows, a.job)
    if a.claim:
        try:
            p, lo, hi = (float(x) for x in a.claim.split(","))
            claim, claim_src = {"point": p, "lo": lo, "hi": hi}, "--claim"
        except ValueError:
            print("--claim must be INDEX,LO,HI", file=sys.stderr)
            return 2
    rep["ledger_row"], rep["claim"], rep["claim_source"] = lrow, claim, claim_src
    if claim is None:
        reasons.append(f"no claim to check ({claim_src})")

    members, notes = walk_artifact(root)
    rep["walk_notes"] = notes
    integ = check_integrity(members, root)
    rep["integrity"] = {k: (len(v) if k == "verified" else v) for k, v in integ.items()}
    reasons += integ["failures"]

    cands = _candidates(members)
    if a.predictions:
        cands = [m for m in cands if a.predictions in m.label or fnmatch.fnmatch(m.label, a.predictions)]
    distinct: dict[str, list[Member]] = {}
    for m in cands:
        try:
            distinct.setdefault(m.sha256(), []).append(m)
        except (OSError, tarfile.TarError, ValueError) as e:
            notes.append(f"unreadable candidate {m.label}: {e}")

    # expected row sets (official loaders)
    try:
        sub_rows, subset = official_expected_rows("subset", a.edition, not a.omit_gsm8k03)
        full_rows, _ = official_expected_rows("full", a.edition, not a.omit_gsm8k03)
    except SystemExit as e:
        print(f"cannot load the expected DEV rows: {e}", file=sys.stderr)
        return 2
    expected = {"subset": {r["_evaluation"]["run_id"] for r in sub_rows},
                "full": {r["_evaluation"]["run_id"] for r in full_rows}}
    rep["expected_rows"] = {k: len(v) for k, v in expected.items()}

    listing = []
    covering = []
    for sha, ms in distinct.items():
        ids, err = _pred_ids(ms[0])
        mode = next((k for k in ("subset", "full") if ids is not None and ids == expected[k]), None)
        listing.append({"sha256": sha, "labels": [m.label for m in ms], "rows": len(ids) if ids else None,
                        "row_set": mode, "error": err})
        if mode and (a.rows in ("auto", mode)):
            covering.append((sha, ms, mode))
    rep["prediction_candidates"] = listing

    pred = None
    if a.predictions and len(distinct) == 1:
        sha, ms = next(iter(distinct.items()))
        ids, _ = _pred_ids(ms[0])
        mode = next((k for k in ("subset", "full") if ids is not None and ids == expected[k]), None)
        pred, mode_used = ms[0], mode or (a.rows if a.rows != "auto" else None)
        if mode is None:
            want = expected.get(a.rows) if a.rows != "auto" else expected["subset"]
            got = ids or set()
            reasons.append(f"row set of {pred.label} equals neither expected manifest "
                           f"({len(got)} rows; vs {a.rows if a.rows != 'auto' else 'subset'}: "
                           f"{len(want - got)} missing, {len(got - want)} extra)")
    elif len(covering) == 1:
        sha, ms, mode_used = covering[0]
        pred = ms[0]
    elif not distinct:
        reasons.append("no raw per-row predictions found in the artifact")
    elif len(covering) > 1:
        print("several distinct prediction sets cover the expected rows; pick one with --predictions:", file=sys.stderr)
        for c in listing:
            if c["row_set"]:
                print(f"  {c['sha256'][:12]} rows={c['rows']} {c['labels'][0]}", file=sys.stderr)
        return 2
    else:
        reasons.append("no prediction set covers the expected row set exactly: "
                       + "; ".join(f"{c['labels'][0]} rows={c['rows']}" for c in listing))

    recomputed = job_ci = None
    if pred is not None:
        copies = distinct[pred.sha256()]
        keys = set()
        for m in copies:  # the file itself, or any archive that contains it (a verified phase .tgz covers its members)
            parts = m.label.split("!")
            for k in range(1, len(parts) + 1):
                keys.add(("!".join(parts[:k - 1]), parts[k - 1]) if k > 1 else ("<dir>", parts[0]))
        rep["predictions"] = {"label": pred.label, "sha256": pred.sha256(), "row_set": mode_used,
                              "copies": [m.label for m in copies],
                              "checksum_verified": bool(keys & integ["verified"])}
        job_ci, where, extra = find_job_score(members, pred, a.job_score)
        rep["job_score"] = {"ci": job_ci, "where": where}
        if job_ci is None:
            reasons.append(where)
        ed = a.edition or extra.get("edition") or config.EDITION
        if a.edition and extra.get("edition") and config.edition_id(extra["edition"]) != config.edition_id(a.edition):
            reasons.append(f"--edition {a.edition} != job score edition {extra['edition']}")
        B = a.B if a.B is not None else (extra.get("B") if isinstance(extra.get("B"), int) else 2000)
        seed = a.seed if a.seed is not None else (extra.get("seed") if isinstance(extra.get("seed"), int) else 0)
        if config.edition_id(ed) != config.edition_id(a.edition) and mode_used:  # re-derive the expected set under the job's edition if it differs
            try:
                exp_rows, _ = official_expected_rows(mode_used, ed, not a.omit_gsm8k03)
                ids, _ = _pred_ids(pred)
                if ids != {r["_evaluation"]["run_id"] for r in exp_rows}:
                    reasons.append(f"row set != expected {mode_used} manifest under edition {ed}")
            except SystemExit as e:
                reasons.append(f"expected rows under edition {ed}: {e}")
        rep["scoring"] = {"entry": "jobs.common.dev_subset.cmd_score" if mode_used == "subset" else "harness.cli.cmd_score",
                          "edition": config.edition_id(ed), "B": B, "seed": seed}
        if mode_used:
            ctx = (contextlib.nullcontext(a.workdir) if a.workdir else tempfile.TemporaryDirectory(prefix="rescore-"))
            with ctx as wd:
                Path(wd).mkdir(parents=True, exist_ok=True)
                try:
                    path, probs = materialize(pred, Path(wd))
                    reasons += probs
                    out = official_score(mode_used, path, ed, not a.omit_gsm8k03, B, seed, subset)
                    recomputed = _ci3(_dig(out, "slice_index.index"))
                    rep["official_output"] = {"rows": out.get("rows"), "subset_ids_sha256": out.get("subset_ids_sha256"),
                                              "index": _dig(out, "slice_index.index"), "areas":
                                              {k: v.get("point") for k, v in (_dig(out, "slice_index.areas") or {}).items()}}
                except SystemExit as e:  # the official CLIs exit on a bad file / row mismatch
                    reasons.append(f"official scorer refused: {e}")
                except Exception as e:  # noqa: BLE001 - any scorer failure on this file is a finding, never a pass
                    reasons.append(f"official scorer failed: {type(e).__name__}: {e}")
            # the artifact's own declared subset hash, if it ships one, must be the pre-registered one
            if mode_used == "subset":
                for m in members:
                    if m.name in ("index-subset.json", "rows.json"):
                        try:
                            h = json.loads(m.read().decode("utf-8")).get("subset_ids_sha256")
                        except (ValueError, UnicodeDecodeError, AttributeError):
                            h = None
                        if h and h != subset["ids_sha256"]:
                            reasons.append(f"{m.label} declares subset {h[:12]} != pre-registered {subset['ids_sha256'][:12]}")

    rep["recomputed"] = recomputed
    if pred is not None and recomputed is None and not any("official scorer" in r for r in reasons):
        reasons.append("official scorer produced no index")

    def cmp(name, other):
        if recomputed is None or other is None:
            return None
        d = {k: abs(recomputed[k] - other[k]) for k in ("point", "lo", "hi")}
        bad = [k for k, v in d.items() if v > a.tol]
        if bad:
            reasons.append(f"recomputed vs {name}: " + ", ".join(f"{k} {recomputed[k]:.4f} vs {other[k]:.4f}" for k in bad))
        return {k: round(v, 6) for k, v in d.items()}

    rep["abs_diff_vs_job_score"] = cmp("job score json", job_ci)
    rep["abs_diff_vs_claim"] = cmp("ledger claim", claim)
    verdict = "PASS" if not reasons else "MISMATCH"
    rep["verdict"], rep["reasons"] = verdict, reasons

    lr = lrow or {}
    r3 = recomputed or {}
    psha = rep.get("predictions", {}).get("sha256", "")[:12]
    if verdict == "PASS":
        note = (f"B rescore PASS from raw predictions ({rep['predictions']['row_set']} rows, sha256 {psha}); "
                f"official {rep['scoring']['entry']} {r3['point']:.2f} [{r3['lo']:.2f},{r3['hi']:.2f}] "
                f"= job score and claim ({claim_src}) within {a.tol}")
        status = lr.get("status", "")  # a confirmation keeps the row's status; no ledger row -> left blank
    else:
        note = f"B rescore MISMATCH of {a.job} (preds sha256 {psha or 'n/a'}): " + "; ".join(reasons)
        status = "invalid"
    line = [datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), a.agent, a.job, lr.get("sha", ""),
            lr.get("recipe", ""), f"{r3['point']:.2f}" if r3 else "", f"{r3['lo']:.2f}" if r3 else "",
            f"{r3['hi']:.2f}" if r3 else "", "0", status, lr.get("receipt", ""), note]
    rep["ledger_line"] = "\t".join(_tsv(x) for x in line)

    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    print(f"{verdict}  {a.job}")
    if rep.get("predictions"):
        print(f"  predictions : {rep['predictions']['label']}  ({rep['predictions']['row_set']}, sha256 {psha}, "
              f"{len(rep['predictions']['copies'])} identical copies)")
    if recomputed:
        print(f"  recomputed  : {recomputed['point']:.4f} [{recomputed['lo']:.4f}, {recomputed['hi']:.4f}]  "
              f"via {rep['scoring']['entry']} (edition {rep['scoring']['edition']}, B={rep['scoring']['B']}, "
              f"seed={rep['scoring']['seed']})")
    if job_ci:
        print(f"  job score   : {job_ci['point']:.4f} [{job_ci['lo']:.4f}, {job_ci['hi']:.4f}]  ({rep['job_score']['where']})")
    if claim:
        print(f"  claim       : {claim['point']:.2f} [{claim['lo']:.2f}, {claim['hi']:.2f}]  ({claim_src})")
    ints = integ["sha256sums"] + integ["manifests"]
    if ints or integ["job_json"]:
        print("  integrity   : " + "; ".join(f"{e['file']} ok={e['ok']} absent={e['absent']} bad={len(e['bad'])}"
                                             for e in ints) + (f"; job.json {integ['job_json']}" if integ["job_json"] else ""))
    if rep.get("predictions"):
        print(f"  pred hash   : {'verified by SHA256SUMS/MANIFEST' if rep['predictions']['checksum_verified'] else 'not covered by any checksum file in the artifact'}")
    for n in integ["notes"]:
        print(f"  note        : {n}")
    for r in reasons:
        print(f"  reason      : {r}")
    print("ledger line (append to results/ledger.tsv on the state branch):")
    print(rep["ledger_line"])
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
