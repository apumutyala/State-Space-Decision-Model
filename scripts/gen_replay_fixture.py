"""Generate fixtures/replay/v0 from the stock Qwen3.5-0.8B readout.

    python -I scripts/gen_replay_fixture.py --qwen-dir <snapshot> --out fixtures/replay/v0 --runs 3

Runs every case ``--runs`` times on the live model (determinism check), records
per-option logits and plans from run 1, writes the expected responses, then
replays every case through ReplayBackend and checks the normalised responses
match exactly. Also measures live-model permutation equivariance on every ok
case (2 shuffles each). Everything is written into the output dir and a
summary JSON is printed for the receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fixture_cases import cases as build_cases  # noqa: E402

from decide import Decider, OptionCountTemperature  # noqa: E402
from decide.backends.hf_readout import HFReadoutBackend, load_qwen35  # noqa: E402
from decide.backends.replay import ReplayBackend, normalize_response  # noqa: E402
from decide.schema import request_fingerprint  # noqa: E402


class RecordingBackend:
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.recordings: dict[str, dict] = {}

    def revision(self):
        return self.inner.revision()

    def plan(self, request):
        p = self.inner.plan(request)
        rec = self.recordings.setdefault(request_fingerprint(request), {"logits": None})
        rec["plan"] = {
            "inference_calls": p.inference_calls,
            "max_call_input_tokens": p.max_call_input_tokens,
            "total_input_tokens": p.total_input_tokens,
            "per_question": p.per_question,
        }
        return p

    def score(self, request, check_deadline):
        r = self.inner.score(request, check_deadline)
        rec = self.recordings.setdefault(request_fingerprint(request), {})
        rec.update(
            logits=r.logits,
            inference_calls=r.inference_calls,
            total_input_tokens=r.total_input_tokens,
            extras=r.extras,
        )
        return r


def shuffled(request, seed):
    rng = random.Random(seed)
    items = list(request["questions"].items())
    rng.shuffle(items)
    out = dict(request)
    out["questions"] = {}
    for qid, q in items:
        q = dict(q)
        if q["type"] == "choice":
            crit = list(q["criteria"].items())
            rng.shuffle(crit)
            q["criteria"] = dict(crit)
        out["questions"][qid] = q
    return out


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default=str(ROOT / "fixtures" / "replay" / "v0"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch

    t0 = time.time()
    plugin = load_qwen35(args.qwen_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
    load_s = time.time() - t0
    live = HFReadoutBackend(plugin, "jrt-chatml-v1", max_batch_rows=16)
    revision = live.revision()
    calibration = OptionCountTemperature()  # identity: uncalibrated, labelled as such
    the_cases = build_cases(revision["revision_id"])

    rec_backend = RecordingBackend(live)
    runs: list[dict[str, dict]] = []
    for run in range(args.runs):
        backend = rec_backend if run == 0 else live
        run_out = {}
        for case in the_cases:
            d = Decider(backend, calibration=calibration,
                        enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
                        backend_max_input_tokens=plugin.max_context)
            run_out[case["case_id"]] = d.decide(case["request"])
        runs.append(run_out)

    # determinism across runs (served probs, ok cases) + latency stats
    max_prob_diff = 0.0
    latencies: dict[str, list[float]] = {}
    for cid, resp in runs[0].items():
        for r in runs:
            latencies.setdefault(cid, []).append(r[cid]["ssdm"]["diagnostics"]["latency_ms"])
            if resp["ssdm"]["status"] == "ok":
                for qid, probs in resp["ssdm"]["served_probabilities"].items():
                    for o, p in probs.items():
                        max_prob_diff = max(max_prob_diff, abs(r[cid]["ssdm"]["served_probabilities"][qid][o] - p))

    # live equivariance
    eq_max = 0.0
    eq_choice_flips = 0
    eq_checked = 0
    for case in the_cases:
        resp = runs[0][case["case_id"]]
        if resp["ssdm"]["status"] != "ok":
            continue
        d = Decider(live, calibration=calibration, enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
                    backend_max_input_tokens=plugin.max_context)
        for seed in (11, 12):
            perm = shuffled(case["request"], seed)
            other = d.decide(perm)
            eq_checked += 1
            for qid, probs in resp["ssdm"]["served_probabilities"].items():
                for o, p in probs.items():
                    eq_max = max(eq_max, abs(other["ssdm"]["served_probabilities"][qid][o] - p))
                if resp["answers"][qid]["type"] == "choice" and resp["answers"][qid]["choice"] != other["answers"][qid]["choice"]:
                    eq_choice_flips += 1

    # write fixture files
    with open(out / "requests.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for case in the_cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    with open(out / "responses.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for case in the_cases:
            fh.write(json.dumps({"case_id": case["case_id"], "response": runs[0][case["case_id"]]},
                                ensure_ascii=False, sort_keys=True) + "\n")
    with open(out / "recordings.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for fp in sorted(rec_backend.recordings):
            rec = dict(rec_backend.recordings[fp])
            rec["fingerprint"] = fp
            rec.setdefault("inference_calls", None)
            rec.setdefault("total_input_tokens", None)
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    try:
        git_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    except Exception:
        git_sha, dirty = None, None
    status_counts: dict[str, int] = {}
    for case in the_cases:
        s = runs[0][case["case_id"]]["ssdm"]
        key = s["status"] + ("" if s["error"] is None else "/" + s["error"]["code"])
        status_counts[key] = status_counts.get(key, 0) + 1
    manifest = {
        "fixture_id": "replay-v0-qwen3.5-0.8b",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Track D -> Track L handoff (m0): replay fixture for decide(); synthetic requests only",
        "revision": revision,
        "calibration": calibration.identity(),
        "generator": {
            "command": " ".join(["python", "-I", "scripts/gen_replay_fixture.py"] + sys.argv[1:]),
            "git_sha_at_generation": git_sha,
            "worktree_dirty_at_generation": dirty,
            "runs": args.runs,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "billed_usd": 0,
        },
        "n_cases": len(the_cases),
        "status_counts": status_counts,
        "files_sha256": {n: sha256_path(out / n) for n in ("requests.jsonl", "responses.jsonl", "recordings.jsonl")},
        "replay_contract": "ReplayBackend(fixture_dir) + Decider(enable_v0_1_deltas=case.decider.enable_v0_1_deltas) "
                           "reproduces responses.jsonl exactly after decide.backends.replay.normalize_response, "
                           "except cases with timing_dependent=true",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # replay verification
    replay = ReplayBackend(out)
    replay_ok, replay_bad = 0, []
    for case in the_cases:
        if case["timing_dependent"]:
            continue
        d = Decider(replay, calibration=calibration, enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
                    backend_max_input_tokens=plugin.max_context)
        got = normalize_response(d.decide(case["request"]))
        want = normalize_response(runs[0][case["case_id"]])
        if got == want:
            replay_ok += 1
        else:
            replay_bad.append(case["case_id"])

    ok_lat = [statistics.median(latencies[c["case_id"]]) for c in the_cases
              if runs[0][c["case_id"]]["ssdm"]["status"] == "ok"]
    summary = {
        "load_s": round(load_s, 2),
        "revision_id": revision["revision_id"],
        "n_cases": len(the_cases),
        "status_counts": status_counts,
        "runs": args.runs,
        "max_served_prob_diff_across_runs": max_prob_diff,
        "live_equivariance": {"perms_checked": eq_checked, "max_abs_prob_diff": eq_max, "choice_flips": eq_choice_flips},
        "replay_exact_match": f"{replay_ok}/{len([c for c in the_cases if not c['timing_dependent']])}",
        "replay_mismatches": replay_bad,
        "ok_case_latency_ms_median_of_runs": {
            "median": statistics.median(ok_lat),
            "min": min(ok_lat),
            "max": max(ok_lat),
        },
        "per_case": {
            c["case_id"]: {
                "status": runs[0][c["case_id"]]["ssdm"]["status"],
                "code": (runs[0][c["case_id"]]["ssdm"]["error"] or {}).get("code"),
                "answers": runs[0][c["case_id"]]["answers"],
                "latency_ms_runs": latencies[c["case_id"]],
                "calls": runs[0][c["case_id"]]["ssdm"]["diagnostics"]["inference_calls"],
                "input_tokens": runs[0][c["case_id"]]["usage"]["input_tokens"],
            }
            for c in the_cases
        },
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
