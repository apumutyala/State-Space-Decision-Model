"""Seed a NEW first-cell job's parent-arm cache from an earlier job's exported, PINNED parent arm (LESSONS L-34).

    python -X utf8 -m jobs.m2_first_cell.arm_seed seed  --src $DATA/parent_arm_seed --dst $CK/parent_arm \
        --params jobs/m2_first_cell/job_params.json --cell <cell config> --out arm-seed.json
    python -X utf8 -m jobs.m2_first_cell.arm_seed check --src <dir> --params ... --cell ...   (build_context.sh)

Why: the parent arm (stock RWKV7 scored on G12 dev + the Index DEV subset, ~68 GPU minutes) depends only on the parent
snapshot, the WKV implementation, the cohort, the unit spec and the Index rows -- not on any training. A platform
restart keeps $GMN_CHECKPOINT_DIR, so the runner's sha-verified cache already skips it; a RESUBMITTED job starts with
an empty checkpoint slot and would pay for it again (job-uhxb5 died in the cell phase after both parent stages had
exported). This module lets the build context ship that export as data (``data/parent_arm_seed/``) and installs it
into the empty cache only when ALL of these hold, else exit 10 before any GPU work:

* ``arm.json`` sha256 == ``job_params.json`` ``parent_arm_seed.arm_sha256`` (no pin -> a shipped seed is refused);
* both stages (g12, index) present, each units file's sha256 == its ``arm.json`` entry;
* each stage's parent identity == ``train.m2_gates.expected_parent_identity`` for the cell (repo, revision, WKV),
  its cohort sha256 == the cell's, its unit spec == ``train.m2_gates.unit_spec(cell)``.

The copy is atomic: files go to a private temp dir, the copy is checked, then renamed to ``--dst`` (a failed check
never leaves a cache a restart could reuse). The runner then runs ``score_parent --verify-cache`` BEFORE the cell
phase: the full units identity (items, variants, Index rows) through ``train.m2_gates.read_arm_cache``, the guard, the
parent decider's revision on this box and a short re-score probe (exit 10 on any mismatch, before training).
``arm.json``, the two units files and a pinned ``seed-provenance.json`` are installed together: the earlier box's
rate reports are NOT, so
``jobs.m2_first_cell.reprice`` takes its factor from the probe measured on THIS box (``--probe``); the cell's
measured-rate stop (exit 26) and the evals' live rate guard still run. A cache already in
``--dst`` (a restart of this job) is left untouched, but a seeded cache returns ``verify_seed`` on every restart,
including older installs without provenance when the manifest matches the pin. The runner
also verifies ordinary cached arms before training and always exports their comparator files.
Absent both a cache and ``--src``: nothing to do (the job scores the parent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

STAGES = ("g12", "index")
PROVENANCE = "seed-provenance.json"


class SeedError(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pinned_sha(params: dict) -> str | None:
    pin = params.get("parent_arm_seed") or {}
    return pin.get("arm_sha256") or None


def check(src: str | Path, arm_sha256: str | None, cell: dict) -> dict:
    """Verify a seed dir against the pin and the cell; returns {stage: units sha256}. Raises SeedError."""
    from train.m2_gates import ARM_SCHEMA, expected_parent_identity, unit_spec

    src = Path(src)
    man_p = src / "arm.json"
    if not man_p.is_file():
        raise SeedError(f"no arm.json in {src}")
    if not arm_sha256:
        raise SeedError("a parent arm seed is shipped but job_params parent_arm_seed.arm_sha256 is not pinned")
    got = _sha(man_p)
    if got != arm_sha256:
        raise SeedError(f"seed arm.json sha256 {got} != pinned {arm_sha256}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man.get("schema") != ARM_SCHEMA or set(man.get("stages", {})) != set(STAGES):
        raise SeedError(f"seed arm.json schema/stages {man.get('schema')} {sorted(man.get('stages', {}))}")
    want_parent = expected_parent_identity(cell.get("parent", "rwkv7"), wkv=cell.get("rwkv7_wkv"))
    spec = unit_spec(cell)
    out = {}
    for st in STAGES:
        e = man["stages"][st]
        f = src / e["file"]
        if Path(e["file"]).name != e["file"] or not f.is_file():
            raise SeedError(f"seed {st}: units file {e['file']!r} missing")
        if _sha(f) != e["sha256"]:
            raise SeedError(f"seed {st}: {e['file']} sha256 != its arm.json entry")
        parent = (e.get("arm") or {}).get("parent") or {}
        bad = {k: parent.get(k) for k, v in want_parent.items() if parent.get(k) != v}
        if bad:
            raise SeedError(f"seed {st}: parent identity {bad} != {want_parent}")
        ui = e.get("units_identity") or {}
        if ui.get("stage") != st or ui.get("cohort_sha256") != cell.get("cohort_sha256"):
            raise SeedError(f"seed {st}: units stage/cohort {ui.get('stage')}/{ui.get('cohort_sha256')} != the cell's")
        if ui.get("unit_spec") != spec:
            raise SeedError(f"seed {st}: unit spec {ui.get('unit_spec')} != {spec}")
        out[st] = e["sha256"]
    return out


def seed(src: str | Path, dst: str | Path, arm_sha256: str | None, cell: dict) -> dict:
    src, dst = Path(src), Path(dst)
    if (dst / "arm.json").exists():
        # The provenance lives INSIDE the atomic install, so a restart cannot lose it between installation and a
        # shell-side marker write. Also recognize older installs interrupted before their external marker existed.
        meta = dst / PROVENANCE
        is_seed = meta.exists() or bool(arm_sha256 and _sha(dst / "arm.json") == arm_sha256)
        if is_seed:
            if meta.exists():
                try:
                    saved_pin = json.loads(meta.read_text(encoding="utf-8"))["arm_sha256"]
                except (ValueError, KeyError) as e:
                    raise SeedError(f"invalid seed provenance in {meta}: {e}") from e
                if saved_pin != arm_sha256:
                    raise SeedError(f"seed provenance pin {saved_pin} != job_params pin {arm_sha256}")
            units = check(dst, arm_sha256, cell)
            return {"seeded": False, "verify_seed": True, "arm_sha256": arm_sha256, "units_sha256": units,
                    "reason": "seeded cache restart: re-verify full identity and probe before training"}
        return {"seeded": False, "verify_seed": False,
                "reason": "parent arm cache already present (restart of this job): left as is"}
    if not src.exists():
        return {"seeded": False, "reason": f"no seed shipped ({src}): the job scores the parent"}
    check(src, arm_sha256, cell)
    if dst.exists() and any(dst.iterdir()):
        raise SeedError(f"{dst} holds files but no arm.json: refusing to seed over a partial cache")
    man = json.loads((src / "arm.json").read_text(encoding="utf-8"))
    files = ["arm.json"] + [man["stages"][st]["file"] for st in STAGES]
    # atomic (PR #147 review): copy into a private temp dir, verify THE COPY, then rename it into place. A copy that
    # fails its check never becomes $CK/parent_arm, so a restart can never pick up an unverified cache.
    tmp = dst.parent / f".{dst.name}.seed-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)  # our own leftover from an interrupted attempt
    tmp.mkdir(parents=True)
    try:
        for f in files:
            shutil.copyfile(src / f, tmp / f)
        units = check(tmp, arm_sha256, cell)
        (tmp / PROVENANCE).write_text(json.dumps({"arm_sha256": arm_sha256}), encoding="utf-8", newline="\n")
        if dst.exists():
            dst.rmdir()  # empty (checked above)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"seeded": True, "verify_seed": True, "arm_sha256": arm_sha256, "units_sha256": units,
            "files": [*files, PROVENANCE],
            "rate_reports": "not installed: reprice uses the seed_verify probe factor (score_parent --verify-cache)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "check"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--params", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    pin = pinned_sha(params)
    try:
        if a.cmd == "check":
            rec = {"ok": True, "units_sha256": check(a.src, pin, cell), "arm_sha256": pin}
        else:
            if not a.dst:
                ap.error("seed needs --dst")
            rec = seed(a.src, a.dst, pin, cell)
    except SeedError as e:
        print(f"parent arm seed refused: {e}", flush=True)
        return 10
    text = json.dumps(rec, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
