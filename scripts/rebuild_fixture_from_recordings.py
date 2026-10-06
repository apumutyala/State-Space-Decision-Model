"""Rebuild fixtures/replay/v0 responses after an adapter change, with NO model run.

    python -I scripts/rebuild_fixture_from_recordings.py [--fixture fixtures/replay/v0]

Responses are a deterministic function of the recorded per-option logits and the
adapter. This script keeps ``recordings.jsonl`` byte-identical, so no logit
changes. It rewrites ``requests.jsonl`` and ``responses.jsonl`` from
``scripts/fixture_cases.py`` by running every case through ReplayBackend and the
current adapter.

* Cases whose decision content was never recorded (``replay_miss``) are left
  out and listed under ``pending_live_recording``.
* The timing-dependent deadline case is rebuilt with a replay backend that
  sleeps 5 ms per call, so ``deadline_ms = 1`` is exceeded deterministically.

The manifest keeps the recorded parent ``revision`` (the logits' source) and
adds a ``responses_rebuilt`` block.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fixture_cases import cases as build_cases  # noqa: E402

from decide import Decider, OptionCountTemperature, versions  # noqa: E402
from decide.backends.replay import ReplayBackend  # noqa: E402


class SlowReplay(ReplayBackend):
    def score(self, request, check_deadline):
        time.sleep(0.005)
        check_deadline()
        return super().score(request, check_deadline)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(ROOT / "fixtures" / "replay" / "v0"))
    args = ap.parse_args()
    fx = Path(args.fixture)
    manifest = json.loads((fx / "manifest.json").read_text(encoding="utf-8"))
    rec_hash = sha(fx / "recordings.jsonl")
    cal = manifest["calibration"]
    calibration = OptionCountTemperature(t0=cal["t0"], t1=cal["t1"], t_min=cal["t_min"], fitted_on=cal["fitted_on"])
    replay, slow = ReplayBackend(fx), SlowReplay(fx)
    kept, pending, responses = [], [], {}
    for case in build_cases(manifest["revision"]["revision_id"]):
        backend = slow if case["timing_dependent"] else replay
        d = Decider(backend, calibration=calibration, enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
                    backend_max_input_tokens=manifest["revision"].get("max_context"))
        resp = d.decide(case["request"])
        if (resp["ssdm"]["error"] or {}).get("code") == "replay_miss":
            pending.append(case["case_id"])
            continue
        kept.append(case)
        responses[case["case_id"]] = resp
    with open(fx / "requests.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for case in kept:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    with open(fx / "responses.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for case in kept:
            fh.write(json.dumps({"case_id": case["case_id"], "response": responses[case["case_id"]]},
                                ensure_ascii=False) + "\n")
    assert sha(fx / "recordings.jsonl") == rec_hash
    counts: dict[str, int] = {}
    for case in kept:
        s = responses[case["case_id"]]["ssdm"]
        key = s["status"] + ("" if s["error"] is None else "/" + s["error"]["code"])
        counts[key] = counts.get(key, 0) + 1
    try:
        git_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = None
    manifest["n_cases"] = len(kept)
    manifest["status_counts"] = counts
    manifest["files_sha256"] = {n: sha(fx / n) for n in ("requests.jsonl", "responses.jsonl", "recordings.jsonl")}
    manifest["responses_rebuilt"] = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "by": "python -I scripts/rebuild_fixture_from_recordings.py",
        "git_sha_before_commit": git_sha,
        "adapter_version": versions.ADAPTER_VERSION,
        "envelope_schema": versions.ENVELOPE_SCHEMA,
        "interface": versions.INTERFACE_VERSION,
        "request_fingerprint_algorithm": versions.FINGERPRINT_ALGORITHM,
        "recordings_sha256_unchanged": rec_hash,
        "note": "no model run; logits unchanged; revision above is the recorded parent of the logits",
        "pending_live_recording": pending,
    }
    manifest["replay_contract"] = (
        "ReplayBackend(fixture_dir) + Decider(enable_v0_1_deltas=case.decider.enable_v0_1_deltas) reproduces "
        "responses.jsonl exactly after decide.backends.replay.normalize_response, except timing_dependent cases"
    )
    (fx / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n_cases": len(kept), "status_counts": counts, "pending_live_recording": pending}, indent=1))


if __name__ == "__main__":
    main()
