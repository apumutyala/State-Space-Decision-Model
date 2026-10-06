"""Training-box readiness for the m2 guard: rebuild the content-hash index from the pinned official suite and refuse
to train unless it is the pinned index.

    python -X utf8 -m jobs.m2_pilot.guard_ready --suite-dir /data/suite-0.2 --out-dir $WORK/guard [--kit-dir /opt/decision-index]
    python -X utf8 -m jobs.m2_pilot.guard_ready --hf --out-dir ...     # only once guard_pin.json pins hf_suite.revision

Checks, in order (each recorded in ``readiness.json``; the first failing class sets the exit code):
  30  kit / suite identity: ``decision_index.__version__`` and the kit's own edition hashes equal the pin; the kit
      clone (``--kit-dir``) is at the pinned commit; each suite row file's UNCOMPRESSED sha256 equals the pin; the
      exclusions file (if present) equals the pin
  32  the committed harness dev/test manifests' uncompressed sha256 equal the pin
  31  the rebuilt content index's sha256 / size equal the pin. A different index is RECORDED
      (``observed`` + ``new_sha256_recorded``) and training is refused: a new index means a new suite or builder,
      which needs review, not a silent swap.
  0   ready: ``readiness.json`` has ``"ready": true`` and names the index file.

``load_ready_guard(readiness.json)`` is what training code calls: it re-hashes the index file named there, checks it
against the pin again and returns the ``SplitGuard`` built from the harness manifests + that index. Without a ready
file it raises, so a run cannot start on an unverified guard.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PIN = Path(__file__).resolve().parent / "guard_pin.json"
EXIT = {"ready": 0, "suite_identity": 30, "index_mismatch": 31, "manifests": 32, "usage": 2}


class GuardNotReady(RuntimeError):
    pass


def _sha_file(p: Path, gunzip: bool = False) -> str:
    opener = gzip.open if gunzip else open
    h = hashlib.sha256()
    with opener(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pin(path: Path = PIN) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_kit(pin: dict, kit_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import decision_index
        from decision_index import editions
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"decision_index not importable: {exc!r}"}
    out["version"] = getattr(decision_index, "__version__", None)
    e = editions.get(pin["kit"]["edition"])
    want = pin["suite_files_sha256_uncompressed"]
    out["edition_hashes"] = {"rows_sha256": e.get("rows_sha256"), "added_sha256": e.get("added_sha256"),
                             "exclusions_sha256": e.get("exclusions_sha256")}
    ok = (out["version"] == pin["kit"]["version"]
          and e.get("rows_sha256") == want["selected-rows.jsonl.gz"]
          and e.get("added_sha256") == want["added-rows.jsonl.gz"]
          and e.get("exclusions_sha256") == pin["suite_exclusions_sha256"])
    if kit_dir is not None:
        try:
            head = subprocess.run(["git", "-C", str(kit_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            head = f"unreadable: {exc!r}"
        out["kit_commit"] = head
        ok = ok and head == pin["kit"]["commit"]
    else:
        out["kit_commit"] = "not checked (no --kit-dir)"
    out["ok"] = ok
    return out


def check_suite(pin: dict, suite_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": str(suite_dir), "files": {}}
    ok = True
    for name, want in pin["suite_files_sha256_uncompressed"].items():
        p = suite_dir / name
        got = _sha_file(p, gunzip=True) if p.exists() else None
        out["files"][name] = {"sha256_uncompressed": got, "match": got == want}
        ok = ok and got == want
    ex = suite_dir / "excluded-questions.json"
    if ex.exists():
        got = _sha_file(ex)
        out["exclusions"] = {"sha256": got, "match": got == pin["suite_exclusions_sha256"]}
        ok = ok and got == pin["suite_exclusions_sha256"]
    else:
        # The index is built from rows named in the harness manifests; exclusions do not touch them (the pinned index
        # was reproduced with and without the file). Recorded, not fatal.
        out["exclusions"] = "absent (index unaffected: reproduced identically with and without exclusions)"
    out["ok"] = ok
    return out


def check_manifests(pin: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel, want in pin["harness_manifests_sha256"].items():
        got = _sha_file(ROOT / rel, gunzip=True)
        out[rel] = {"sha256_uncompressed": got, "match": got == want}
    out["ok"] = all(v["match"] for v in out.values() if isinstance(v, dict))
    return out


def fetch_hf_suite(pin: dict, dest: Path) -> Path:
    hf = pin["hf_suite"]
    if not hf.get("revision"):
        raise GuardNotReady(f"hf_suite.revision is not pinned ({hf.get('status')}); use --suite-dir")
    from huggingface_hub import hf_hub_download

    for name in list(pin["suite_files_sha256_uncompressed"]) + ["excluded-questions.json"]:
        hf_hub_download(hf["repo"], name, repo_type=hf.get("repo_type", "dataset"), revision=hf["revision"],
                        local_dir=dest)
    return dest


def run(suite_dir: Path, out_dir: Path, kit_dir: Path | None = None, pin_path: Path = PIN) -> tuple[int, dict]:
    from train.guard import build_content_index

    pin = load_pin(pin_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"pin_file": str(pin_path), "pin_sha256": _sha_file(pin_path), "expected": pin["content_index"]}
    rep["kit"] = check_kit(pin, kit_dir)
    rep["suite"] = check_suite(pin, suite_dir)
    rep["manifests"] = check_manifests(pin)
    code = 0
    if not (rep["kit"]["ok"] and rep["suite"]["ok"]):
        code = EXIT["suite_identity"]
    elif not rep["manifests"]["ok"]:
        code = EXIT["manifests"]
    else:
        m = ROOT / "harness" / "manifests"
        idx = out_dir / "index-content-hashes.txt.gz"
        built = build_content_index(idx, [m / "dev-manifest.jsonl.gz", m / "test-manifest.jsonl.gz"], suite_dir)
        rep["observed"] = built
        same = built["sha256"] == pin["content_index"]["sha256"] and built["n"] == pin["content_index"]["n_hashes"]
        if not same:
            rep["new_sha256_recorded"] = built["sha256"]
            code = EXIT["index_mismatch"]
        else:
            rep["index_file"] = str(idx)
    rep["ready"] = code == 0
    rep["exit_code"] = code
    (out_dir / "readiness.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return code, rep


def load_ready_guard(readiness: str | Path, pin_path: Path = PIN):
    """The ``SplitGuard`` a training run must use: harness manifests + the verified content index. Raises
    ``GuardNotReady`` unless the readiness file says ready AND the index file still hashes to the pin."""
    from train.guard import SplitGuard

    p = Path(readiness)
    if not p.exists():
        raise GuardNotReady(f"no readiness file at {p}; run jobs.m2_pilot.guard_ready first")
    rep = json.loads(p.read_text(encoding="utf-8"))
    pin = load_pin(pin_path)
    if not rep.get("ready") or not rep.get("index_file"):
        raise GuardNotReady(f"guard not ready (exit {rep.get('exit_code')}); observed {rep.get('observed')}")
    m = ROOT / "harness" / "manifests"
    g = SplitGuard.from_manifests([m / "dev-manifest.jsonl.gz", m / "test-manifest.jsonl.gz"],
                                  content_index=rep["index_file"])
    ci = g.content_index or {}
    if ci.get("sha256") != pin["content_index"]["sha256"] or ci.get("n") != pin["content_index"]["n_hashes"]:
        raise GuardNotReady(f"content index {ci} != pinned {pin['content_index']}")
    for path, sha in g.manifests.items():
        rel = Path(path).relative_to(ROOT).as_posix()
        if pin["harness_manifests_sha256"].get(rel) != sha:
            raise GuardNotReady(f"harness manifest {rel} sha256 {sha} != pin")
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_pilot.guard_ready")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--suite-dir", type=Path)
    src.add_argument("--hf", action="store_true", help="download the suite from the pinned HF dataset revision")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kit-dir", type=Path, default=None)
    ap.add_argument("--pin", type=Path, default=PIN)
    a = ap.parse_args(argv)
    suite = a.suite_dir
    if a.hf:
        try:
            suite = fetch_hf_suite(load_pin(a.pin), a.out_dir / "suite")
        except GuardNotReady as exc:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            (a.out_dir / "readiness.json").write_text(json.dumps({"ready": False, "exit_code": EXIT["suite_identity"],
                                                                  "error": str(exc)}, indent=1), encoding="utf-8")
            print(str(exc), file=sys.stderr)
            return EXIT["suite_identity"]
    code, rep = run(suite, a.out_dir, a.kit_dir, a.pin)
    print(json.dumps({k: rep.get(k) for k in ("ready", "exit_code", "observed", "new_sha256_recorded")}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
