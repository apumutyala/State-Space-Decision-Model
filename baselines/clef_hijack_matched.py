"""MATCHED hijack + D1-A steer comparison: Clef 27B and Clef-flash vs our trained cell D1 (JOB_SPEC section 13).

D1 was measured by job-j8wgy (jobs/m2_first_cell mode hijack_eval @892bc6c, artifact art-4naug): hijack 0.0 % vs the
stock RWKV7 parent 34.9 % on the five ``safety_eval.attacks.HIJACK_TEMPLATES`` and D1-A attacked accuracy 99.2 % vs
66.1 %. This module scores EXACTLY the same units on both Clef releases and pairs them with D1's per-unit rows:

  units   ``train.m2_gates.stage_variants(Cohort(v2 pin), "hijack")`` = ``hijack_subset(steer_dev)`` (8 requests x 45
          clusters, salted hash) through ``safety_eval.attacks.build_variants(hijack=all 5, d1=D1-A, final=False)``:
          360 clean + 1,800 hijack + 1,440 D1-A = 3,600 units. Same builder, cohort, subset salt and order as job-j8wgy.
  decider ``baselines.clef_backend`` through ``decide.Decider`` + ``safety_eval.decide_bridge`` (the clef_ssdm arm of
          ``baselines/run_clef_safety``: untrusted text through decide/untrusted.py, like D1), FAST GDN path
          (fla / causal-conv1d kernels; accuracy-neutral per job-zfexa / job-t5yg2: reference-vs-fast argmax agreement
          99.50 % (Clef 27B, 2,008 rows) and 99.80 % (Clef-flash, 12,104 rows), Index delta -0.06).
  metrics ``safety_eval.metrics.run`` per unit; ``metrics.compare_arms`` paired, two-way (cluster x template) and
          fixed-template cluster bootstrap, B = 2000 (as ``train.m2_gates.hijack_measurement``).

Eligibility (pre-declared, JOB_SPEC 13.3). PRIMARY for the cross-model pair Clef - D1: ``both`` (the unit's clean
request is clean-correct in BOTH models), because D1's own "parent clean-correct" population belongs to a third model.
Also reported: each model's own ``reference_y`` numbers (own clean-correct; ``metrics.summarise``), and the
``reference_y`` pair with D1 as the reference (D1 clean-correct) for continuity with job-j8wgy. hijack_rate, alarm_rate
and attacked_acc are over every observed unit (compare_arms); only steer_rate is restricted to the eligible units.

Fail closed: D1's rows, the RWKV7 parent rows and each Clef run must cover EXACTLY the expected unit ids with equal
``unit_sha`` (exit 21); the artifact must match its sha256 pin (exit 10); D1's re-computed rates must equal the
declared job-j8wgy numbers (exit 21).

    python -X utf8 -m baselines.clef_hijack_matched fetch-d1 --out <dir>/d1         # $D1_REF_URL (art-4naug)
    python -X utf8 -m baselines.clef_hijack_matched score --model-dir <snap> --release clef --out <dir>
    python -X utf8 -m baselines.clef_hijack_matched compare --d1-dir <dir>/d1 --units <dir>/units-clef-flash.jsonl \
        <dir>/units-clef.jsonl --out <dir>/summary.json
Exit codes: 0 ok | 2 usage | 10 reference fetch / sha256 / layout | 21 unit-set or declared-number mismatch |
30 model load | 31 scoring failed | 33 compare failed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

#: the committed cohort (train/cohorts/m2-dev-cohort-v2.jsonl.gz), as jobs/m2_first_cell/job_params_hijack_eval.json
COHORT_SHA256 = "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
#: sha256_obj of the 3,600 expected request ids in builder order (computed from the committed cohort, 2026-10-08)
VARIANTS_SHA256 = "5546813de95900dee45b0822b5f4133cae20f605e5a04476b321bdfe6dbab2e9"
N_UNITS = {"clean": 360, "hijack": 1800, "d1": 1440}
N_CLUSTERS = 45
#: job-j8wgy's output artifact (tar of /output; autor get /artifacts/art-4naug: sha256 + size)
D1_ARTIFACT = {"artifact": "art-4naug", "job": "job-j8wgy", "commit": "892bc6c",
               "sha256": "42c57b6152dadd8f5b916a5d36444ad1a5ab5514b2dd67981a531d1a9da69130", "size_bytes": 11294720,
               "url_env": "D1_REF_URL"}
#: members inside a phase tarball of the artifact (jobs/common/lib.sh export_phase tars WORK/out + WORK/logs)
D1_MEMBERS = {"child": "out/units-child-hijack.json", "parent": "out/parent_arm_hijack/units-hijack.json",
              "result": "out/hijack_eval_result.json"}
#: job-j8wgy as recorded in state results/ledger.tsv (2026-10-08T18:10Z), two-way compare_arms x = D1, y = parent;
#: rounded to 0.1 pp there, so the check tolerance is half a unit of the last digit
D1_DECLARED = {"hijack": {"hijack_rate": (0.0, 0.349), "attacked_acc": (0.997, 0.909)},
               "d1_A": {"attacked_acc": (0.992, 0.661)}}
D1_DECLARED_TOL = 0.0005 + 1e-9
FAMILIES = ("hijack", "d1_A")
BOOTSTRAPS = ("two_way", "fixed_template")
REPORTED = ("hijack_rate", "attacked_acc", "steer_rate", "alarm_rate", "refusal_rate")
RELEASES = ("clef-flash", "clef")
CHUNK = 200


class UnitSetMismatch(RuntimeError):
    """Exit 21: an arm does not cover exactly the expected units (ids or unit_sha), or D1 does not reproduce."""


class ReferenceError_(RuntimeError):
    """Exit 10: the D1 reference artifact is missing, has the wrong sha256, or lacks the expected members."""


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ expected units
def expected_variants(cohort_path: str | None = None, cohort_sha256: str = COHORT_SHA256) -> list[dict]:
    """The hijack stage's variants, built exactly as job-j8wgy built them."""
    from train import m2_gates as G
    from train.cohort import Cohort

    cohort = Cohort(cohort_path, expected_sha256=cohort_sha256) if cohort_path else Cohort(expected_sha256=cohort_sha256)
    return G.stage_variants(cohort, "hijack")


def variants_identity(variants: list[dict]) -> dict[str, Any]:
    from collections import Counter

    from safety_eval.request import sha256_obj

    return {"n_units": len(variants), "by_family": dict(Counter(v["_meta"]["attack_family"] for v in variants)),
            "n_clusters": len({v["_meta"]["cluster"] for v in variants}),
            "request_ids_sha256": sha256_obj([v["request_id"] for v in variants])}


def check_production_variants(variants: list[dict]) -> dict[str, Any]:
    """The committed cohort must give the pinned 3,600-unit set (fail closed, exit 21)."""
    ident = variants_identity(variants)
    if ident["by_family"] != N_UNITS or ident["n_clusters"] != N_CLUSTERS or \
            ident["request_ids_sha256"] != VARIANTS_SHA256:
        raise UnitSetMismatch(f"hijack stage variants {ident} != pinned {N_UNITS} / {N_CLUSTERS} / {VARIANTS_SHA256}")
    return ident


def check_unit_set(rows: list[dict], variants: list[dict], arm: str) -> None:
    """``rows`` must be EXACTLY one row per expected unit, ids and unit_sha equal (no extras, no gaps, no dups)."""
    from safety_eval.metrics import unit_fingerprint

    want = {v["request_id"]: unit_fingerprint(v) for v in variants}
    ids = [r.get("request_id") for r in rows]
    if len(ids) != len(set(ids)):
        raise UnitSetMismatch(f"{arm}: duplicate unit ids ({len(ids) - len(set(ids))})")
    got = set(ids)
    if got != set(want):
        miss, extra = sorted(set(want) - got), sorted(got - set(want))
        raise UnitSetMismatch(f"{arm}: unit ids differ from the expected set: {len(miss)} missing {miss[:3]}, "
                              f"{len(extra)} unexpected {extra[:3]}")
    bad = [r["request_id"] for r in rows if r.get("unit_sha") != want[r["request_id"]]]
    if bad:
        raise UnitSetMismatch(f"{arm}: {len(bad)} units scored different content (unit_sha) e.g. {bad[:3]}")


# ------------------------------------------------------------------------------------------------ D1 reference
def _phase_order(names: list[str]) -> list[str]:
    ph = [n for n in names if n.rstrip("/").split("/")[-2:-1] == ["phases"] and n.endswith(".tgz")]
    pri = {"final.tgz": 0, "report.tgz": 1, "evals.tgz": 2}
    return sorted(ph, key=lambda n: (pri.get(n.rsplit("/", 1)[-1], 9), -ph.index(n)))


def _members_from(t: tarfile.TarFile) -> dict[str, bytes] | None:
    by = {m.name.removeprefix("./"): m for m in t.getmembers() if m.isfile()}
    found = {}
    for k, suffix in D1_MEMBERS.items():
        hit = [n for n in by if n == suffix or n.endswith("/" + suffix)]
        if hit:
            found[k] = t.extractfile(by[sorted(hit, key=len)[0]]).read()
    return found if {"child", "parent"} <= set(found) else None


def d1_members(artifact: Path) -> tuple[dict[str, bytes], str]:
    """(member bytes, where) from the artifact: the members directly, else the newest phase tarball holding them."""
    try:
        with tarfile.open(artifact, "r:*") as t:
            direct = _members_from(t)
            if direct:
                return direct, "artifact root"
            names = [m.name.removeprefix("./") for m in t.getmembers() if m.isfile()]
            for n in _phase_order(names):
                inner = t.extractfile(next(m for m in t.getmembers() if m.name.removeprefix("./") == n)).read()
                with tarfile.open(fileobj=io.BytesIO(inner), mode="r:*") as ti:
                    got = _members_from(ti)
                if got:
                    return got, n
    except tarfile.TarError as e:
        raise ReferenceError_(f"D1 artifact is not a readable tar: {e}") from e
    raise ReferenceError_(f"D1 artifact has no {D1_MEMBERS['child']} + {D1_MEMBERS['parent']} (root or phases/*.tgz)")


def fetch_d1(out: Path, url: str | None = None, artifact: Path | None = None) -> dict[str, Any]:
    """Download (or take) art-4naug, verify its sha256 pin, and write child / parent / result json into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    if artifact is None:
        url = url or os.environ.get(D1_ARTIFACT["url_env"])
        if not url or url.startswith("<"):
            raise ReferenceError_(f"${D1_ARTIFACT['url_env']} is not set (get_artifact {D1_ARTIFACT['artifact']} URL)")
        if not url.startswith(("https://", "file://")):
            raise ReferenceError_("the D1 artifact URL must be https:// (get_artifact) or file:// (tests)")
        artifact = out / "art-4naug.tar"
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(artifact, "wb") as f:  # noqa: S310 (scheme checked)
                for b in iter(lambda: r.read(1 << 20), b""):
                    h.update(b)
                    f.write(b)
        except OSError as e:
            raise ReferenceError_(f"D1 artifact download failed: {type(e).__name__}: {e}") from e
        got = h.hexdigest()
    else:
        got = _sha_bytes(Path(artifact).read_bytes())
    if got != D1_ARTIFACT["sha256"]:
        raise ReferenceError_(f"D1 artifact sha256 {got} != pin {D1_ARTIFACT['sha256']}")
    mem, where = d1_members(Path(artifact))
    files = {}
    for k, b in mem.items():
        (out / f"d1-{k}.json").write_bytes(b)
        files[k] = {"member": D1_MEMBERS[k], "sha256": _sha_bytes(b), "bytes": len(b)}
    ident = {**D1_ARTIFACT, "artifact_sha256_verified": got, "members_from": where, "files": files}
    (out / "d1-identity.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")
    return ident


def load_d1(d1_dir: Path) -> tuple[list[dict], list[dict], dict | None, dict]:
    d = Path(d1_dir)
    child = json.loads((d / "d1-child.json").read_text(encoding="utf-8"))["steer_rows"]
    parent = json.loads((d / "d1-parent.json").read_text(encoding="utf-8"))["steer_rows"]
    res = json.loads((d / "d1-result.json").read_text(encoding="utf-8")) if (d / "d1-result.json").exists() else None
    ident = json.loads((d / "d1-identity.json").read_text(encoding="utf-8")) if (d / "d1-identity.json").exists() else {}
    return child, parent, res, ident


def d1_reproduction(child: list[dict], parent: list[dict], variants: list[dict], n_boot: int,
                    declared: dict | None = D1_DECLARED, result: dict | None = None) -> dict[str, Any]:
    """Re-compute job-j8wgy's measurement from its rows (``train.m2_gates.hijack_measurement``); the rates must equal
    the declared numbers (and the artifact's hijack_eval_result.json when shipped), else exit 21."""
    from train import m2_gates as G

    m = G.hijack_measurement(child, parent, variants, n_boot)
    tw = {f: m["families"][f]["two_way"] for f in FAMILIES}
    got = {f: {k: (tw[f][k]["x"], tw[f][k]["y"]) for k in REPORTED} for f in FAMILIES}
    bad = []
    if not m["complete"]:
        bad.append("D1 / parent rows incomplete")
    for f, ks in (declared or {}).items():
        for k, (x, y) in ks.items():
            gx, gy = got[f][k]
            if abs(gx - x) > D1_DECLARED_TOL or abs(gy - y) > D1_DECLARED_TOL:
                bad.append(f"{f}.{k}: re-computed ({gx:.4f}, {gy:.4f}) != declared ({x}, {y})")
    if result is not None:
        rh = result.get("hijack", {}).get("hijack_rate", {})
        for k, i in (("child", 0), ("parent", 1)):
            if k in rh and abs(rh[k] - got["hijack"]["hijack_rate"][i]) > 1e-12:
                bad.append(f"hijack_eval_result.json hijack_rate.{k} {rh[k]} != re-computed {got['hijack']['hijack_rate'][i]}")
    if bad:
        raise UnitSetMismatch("D1 does not reproduce job-j8wgy: " + "; ".join(bad))
    return {"complete": m["complete"], "rates_two_way": got, "declared": declared,
            "hijack_diff_two_way": tw["hijack"]["hijack_rate"]["diff"],
            "steer_d1a_diff_fixed_template": m["families"]["d1_A"]["fixed_template"]["steer_rate"]["diff"]}


# ------------------------------------------------------------------------------------------------ Clef scoring
def score(model_dir: str, release: str, out: Path, *, device: str = "cuda", dtype: str = "bfloat16",
          verify: bool = True, cohort_path: str | None = None, cohort_sha256: str = COHORT_SHA256,
          log=print) -> dict[str, Any]:
    """Score the expected units on one Clef release (fast GDN path); rows appended to units-<release>.jsonl in
    chunks (the live rate guard counts lines); identity in units-<release>.identity.json."""
    from baselines import clef_backend as CB
    from safety_eval import metrics
    from safety_eval.decide_bridge import DecideBridge

    variants = expected_variants(cohort_path, cohort_sha256)
    ident_v = check_production_variants(variants) if cohort_path is None else variants_identity(variants)
    out.mkdir(parents=True, exist_ok=True)
    res = out / f"units-{release}.jsonl"
    t_load = time.time()
    backend, _model, _proc = CB.load_clef(model_dir, release, encoding="ssdm", device=device, dtype=dtype,
                                          verify=verify)
    ks = CB.kernel_stack()
    if os.environ.get("CLEF_DRY_RUN") != "1" and not ks.get("import:fla"):
        raise RuntimeError("fast GDN path requested but fla does not import (transformers would fall back silently)")
    bridge = DecideBridge(CB.build_decider(backend), model=release)
    t_load = time.time() - t_load
    done = 0
    t0 = time.time()
    with res.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(0, len(variants), CHUNK):
            rows = metrics.run(bridge, variants[i:i + CHUNK])
            for r in rows:
                r["arm"] = f"{release}_ssdm"
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            done += len(rows)
            log(f"[{release}] {done}/{len(variants)} units ({time.time() - t0:.0f}s)")
    secs = time.time() - t0
    ident = {"release": release, "revision": backend.revision(), "files_sha256": backend.files_sha256,
             "kernel_stack": ks, "gdn_path": "fast", "encoding": "ssdm", "device": device, "dtype": dtype,
             "variants": ident_v, "cohort_sha256": cohort_sha256, "seconds_scoring": round(secs, 1),
             "seconds_load": round(t_load, 1), "s_per_unit": round(secs / max(len(variants), 1), 4),
             "results_sha256": _sha_bytes(res.read_bytes())}
    (out / f"units-{release}.identity.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    return ident


def read_units(path: Path) -> list[dict]:
    from harness.jsonl import read_jsonl_strict

    return read_jsonl_strict(Path(path))


# ------------------------------------------------------------------------------------------------ comparison
def _pick(c: dict) -> dict[str, Any]:
    keep = {k: c[k] for k in ("n_units", "eligibility", "n_eligible", "bootstrap", "complete", "n_error",
                              "n_missing", "clean_acc_x", "clean_acc_y", "steer_discordance") if k in c}
    for k in REPORTED:
        keep[k] = c[k]
    if "steer_rate_two_way" in c:
        keep["steer_rate_two_way"] = c["steer_rate_two_way"]
    return keep


def _own(rows: list[dict], variants: list[dict], n_boot: int) -> dict[str, Any]:
    """A model's own numbers (own clean-correct eligibility = ``reference_y`` on itself), ``metrics.summarise``."""
    from safety_eval import metrics as M

    s = M.summarise(rows, n_boot, expected=variants)
    out = {"clean": {k: s["clean"][k] for k in ("n_units", "accuracy", "false_alarm_rate", "refusal_rate")},
           "status_counts": s["status_counts"]}
    for f in FAMILIES:
        fs = s["families"][f]
        out[f] = {k: fs[k] for k in ("n_units", "complete", "outcomes", "hijack_rate", "attacked_acc", "clean_acc",
                                     "steer_rate", "alarm_rate", "refusal_rate", "steer_bestofK", "hijack_bestofK")}
    return out


def compare(d1_dir: Path, units: dict[str, Path], n_boot: int = 2000, cohort_path: str | None = None,
            cohort_sha256: str = COHORT_SHA256, declared: dict | None = D1_DECLARED) -> dict[str, Any]:
    from safety_eval import metrics as M
    from train import m2_gates as G
    from train.promotion import _family_ids

    variants = expected_variants(cohort_path, cohort_sha256)
    ident_v = check_production_variants(variants) if cohort_path is None else variants_identity(variants)
    child, parent, res, d1_ident = load_d1(d1_dir)
    check_unit_set(child, variants, "D1 (job-j8wgy child)")
    check_unit_set(parent, variants, "RWKV7 parent (job-j8wgy)")
    arms = {"d1": child, "rwkv7_parent": parent}
    for rel, p in units.items():
        rows = read_units(p)
        check_unit_set(rows, variants, rel)
        arms[rel] = rows
    out: dict[str, Any] = {
        "schema": "ssdm.clef.hijack_matched.v1", "units": ident_v, "d1_reference": d1_ident, "n_boot": n_boot,
        "eligibility": {"primary": "both", "definition": "Clef - D1 pairs: units whose clean request is clean-correct "
                                                          "in BOTH models (compare_arms eligibility='both'); steer_rate "
                                                          "only; hijack / alarm / attacked_acc over every observed unit",
                        "also": ["own reference_y per model (metrics.summarise)",
                                 "reference_y with D1 as the reference (D1 clean-correct)"]},
        "d1_reproduction": d1_reproduction(child, parent, variants, n_boot, declared, res),
        "per_model": {a: _own(r, variants, n_boot) for a, r in arms.items()},
        "paired_vs_d1": {}}
    for rel in units:
        blk = {}
        for f in FAMILIES:
            ids = _family_ids(variants, f)
            fb = {}
            for elig in ("both", "reference_y"):
                for bs in BOOTSTRAPS:
                    c = M.compare_arms(M.with_expected(arms[rel], variants), M.with_expected(child, variants), f,
                                       n_boot=n_boot, eligibility=elig, expected_ids=ids, bootstrap=bs)
                    fb[f"{elig}|{bs}"] = _pick(c)
            fb["note"] = (f"x = {rel}, y = D1; diff = {rel} - D1 (negative = {rel} safer / less accurate); "
                          "reference_y = D1 clean-correct")
            blk[f] = fb
        out["paired_vs_d1"][rel] = blk
    out["complete"] = all(b[f][f"{e}|{bs}"]["complete"] for b in out["paired_vs_d1"].values() for f in FAMILIES
                          for e in ("both", "reference_y") for bs in BOOTSTRAPS)
    out["headline"] = {rel: {f: {k: {"model": b[f]["both|two_way"][k]["x"], "d1": b[f]["both|two_way"][k]["y"],
                                      "diff": b[f]["both|two_way"][k]["diff"]}
                                 for k in REPORTED} | {"n_eligible_both": b[f]["both|two_way"]["n_eligible"],
                                                       "clean_acc_model": b[f]["both|two_way"]["clean_acc_x"],
                                                       "clean_acc_d1": b[f]["both|two_way"]["clean_acc_y"]}
                            for f in FAMILIES}
                       for rel, b in out["paired_vs_d1"].items()}
    _ = G  # the unit set is train.m2_gates' hijack stage (imported for the builder above)
    return out


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m baselines.clef_hijack_matched")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-d1")
    f.add_argument("--out", required=True)
    f.add_argument("--artifact", default=None, help="a local copy of art-4naug (sha256-checked) instead of the URL")
    s = sub.add_parser("score")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--release", choices=RELEASES, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--no-verify", action="store_true", help="skip weight sha256 (dry runs only)")
    c = sub.add_parser("compare")
    c.add_argument("--d1-dir", required=True)
    c.add_argument("--units", nargs="+", required=True, help="units-<release>.jsonl files")
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=2000)
    for p in (s, c):
        p.add_argument("--cohort", default=None, help="tests only: a mini cohort file (with --cohort-sha256)")
        p.add_argument("--cohort-sha256", default=COHORT_SHA256)
    c.add_argument("--skip-declared-check", action="store_true", help="tests only (mini cohort): no job-j8wgy numbers")
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        if a.cmd == "fetch-d1":
            ident = fetch_d1(Path(a.out), artifact=Path(a.artifact) if a.artifact else None)
            # fail closed BEFORE any weight download / GPU phase: D1 + parent cover exactly the pinned unit set and
            # D1's re-computed rates equal job-j8wgy's declared numbers (point rates; the bootstrap size is irrelevant)
            variants = expected_variants()
            check_production_variants(variants)
            child, parent, res, _ = load_d1(Path(a.out))
            check_unit_set(child, variants, "D1 (job-j8wgy child)")
            check_unit_set(parent, variants, "RWKV7 parent (job-j8wgy)")
            rep = d1_reproduction(child, parent, variants, 200, D1_DECLARED, res)
            (Path(a.out) / "d1-precheck.json").write_text(json.dumps({"n_units": len(variants), **rep}, indent=1,
                                                                     default=float), encoding="utf-8")
            log(json.dumps({"d1": ident["artifact"], "members_from": ident["members_from"], "n_units": len(variants),
                            "rates_two_way": rep["rates_two_way"]}, default=float))
            return 0
        if a.cmd == "score":
            try:
                ident = score(a.model_dir, a.release, Path(a.out), device=a.device, dtype=a.dtype,
                              verify=not a.no_verify, cohort_path=a.cohort, cohort_sha256=a.cohort_sha256, log=log)
            except (UnitSetMismatch, ReferenceError_):
                raise
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                log(f"SCORE FAILED: {type(e).__name__}: {e}")
                return 31
            log(json.dumps({"release": a.release, "s_per_unit": ident["s_per_unit"]}))
            return 0
        units = {}
        for p in a.units:
            rel = Path(p).name.removeprefix("units-").removesuffix(".jsonl")
            if rel not in RELEASES or rel in units:
                log(f"--units {p}: expected units-<{'|'.join(RELEASES)}>.jsonl, once each")
                return 2
            units[rel] = Path(p)
        if a.skip_declared_check and a.cohort is None:
            log("--skip-declared-check is for a mini test cohort only")
            return 2
        try:
            out = compare(Path(a.d1_dir), units, a.n_boot, a.cohort, a.cohort_sha256,
                          None if a.skip_declared_check else D1_DECLARED)
        except (UnitSetMismatch, ReferenceError_):
            raise
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            log(f"COMPARE FAILED: {type(e).__name__}: {e}")
            return 33
        Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        log(json.dumps({"complete": out["complete"], "headline": {
            r: {f: {k: out["headline"][r][f][k]["model"] for k in ("hijack_rate", "attacked_acc", "steer_rate")}
                for f in FAMILIES} for r in units}}, default=float))
        return 0 if out["complete"] else 21
    except UnitSetMismatch as e:
        log(f"UNIT SET / REPRODUCTION MISMATCH (fail closed): {e}")
        return 21
    except ReferenceError_ as e:
        log(f"D1 REFERENCE: {e}")
        return 10


if __name__ == "__main__":
    sys.exit(main())
