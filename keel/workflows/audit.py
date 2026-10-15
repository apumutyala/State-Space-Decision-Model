"""Verify workflow receipts and compare the host-integrated rerun with frozen v0."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from keel.protocol import DecisionRequest
from keel.protocol.receipts import LogCheckpoint, ReceiptIntegrityError, ReceiptLog, replay_check


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(root, *, sample_size=20, seed=20261015, expected=None):
    root = Path(root)
    path = root / "receipts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    log = ReceiptLog(path)
    checkpoint = log.verify(expected=expected)
    receipts = log.receipts()
    episodes = _rows(root / "episodes.jsonl")
    expected_ids = [receipt_id for episode in episodes for receipt_id in episode.get("receipt_ids", [])]
    if expected_ids != [receipt.receipt_id for receipt in receipts]:
        raise ReceiptIntegrityError("episode dispatches and receipt chain disagree")
    count = verified_episodes = 0
    for episode in episodes:
        count += len(episode.get("receipt_ids", []))
        if episode.get("receipt_chain_verified_count") is not None:
            if episode["receipt_chain_verified_count"] != count:
                raise ReceiptIntegrityError("episode checkpoint does not match cumulative receipt count")
            verified_episodes += 1
    indices = sorted(random.Random(seed).sample(range(len(receipts)), min(sample_size, len(receipts))))
    for index in indices:
        receipt = receipts[index]
        request_path = root / "requests" / f"{receipt.request_fingerprint}.json"
        request = DecisionRequest.from_json(json.loads(request_path.read_text(encoding="utf-8")))
        if not replay_check(receipt, request):
            raise ReceiptIntegrityError(f"replay failed at receipt {index}")
    return {"checkpoint": asdict(checkpoint), "receipts_verified": len(receipts),
            "episode_checkpoints_verified": verified_episodes, "receipts_replayed": len(indices),
            "replay_seed": seed, "sample_indices": indices}


def compare_v0(root, reference):
    root, reference = Path(root), Path(reference)
    new = {row["episode_id"]: row for row in _rows(root / "episodes.jsonl")}
    old = {row["episode_id"]: row for row in _rows(reference / "episodes.jsonl") if row["arm"] == "keel"}
    if new.keys() != old.keys():
        raise ValueError("host rerun must contain exactly v0's Keel episodes")
    fields = ("utility", "targeted_asr", "proposed_calls", "executed_calls", "dispatch_errors", "gated_calls",
              "held_calls", "review_episode", "agent_requests", "completion_truncations", "prompt_tokens",
              "completion_tokens", "error", "injections_sha256")
    differences = [{"episode_id": key, "fields": {field: {"v0": old[key][field], "v01": new[key][field]}
                    for field in fields if old[key][field] != new[key][field]}}
                   for key in sorted(new)]
    differences = [row for row in differences if row["fields"]]
    # Historical comparison only: current runs never write the old journal.
    old_calls = {row["request_id"]: row for row in _rows(reference / "decisions.jsonl")}
    receipts = ReceiptLog(root / "receipts.jsonl").receipts()
    current = {receipt.decision_request_id: receipt for receipt in receipts}
    common = old_calls.keys() & current.keys()
    prompt_matches = sum(old_calls[key]["prompt_sha256"] == current[key].prompts[0].sha256 for key in common)
    probability_deltas = [max(abs(old_calls[key]["probabilities"][option] - current[key].answers["q"]["probs"][option])
                              for option in ("allow", "hold")) for key in common]
    (root / "outcome_differences.json").write_text(json.dumps(differences, indent=2) + "\n", encoding="utf-8")
    return {"episodes_compared": len(new), "episodes_identical_on_all_outcome_fields": len(new) - len(differences),
            "differences_by_field": {field: sum(field in row["fields"] for row in differences) for field in fields},
            "call_ids_compared": len(common), "identical_prompt_hashes": prompt_matches,
            "v0_only_calls": len(old_calls.keys() - current.keys()), "v01_only_calls": len(current.keys() - old_calls.keys()),
            "probabilities_exactly_equal": sum(delta == 0 for delta in probability_deltas),
            "maximum_probability_absolute_difference": max(probability_deltas, default=0.0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference", help="Optional frozen v0 run for the one-pass integration comparison.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = json.loads((Path(args.run_dir) / "run.json").read_text())
    expected = LogCheckpoint(**run["receipt_verification"]["checkpoint"])
    report = {"receipts": verify_run(args.run_dir, expected=expected),
              "run_id": run["run_id"], "gpu_minutes_estimate": run["gpu_minutes_estimate"],
              "cost_usd_estimate": run["cost_usd_estimate"]}
    if args.reference:
        report["comparison"] = compare_v0(args.run_dir, args.reference)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
