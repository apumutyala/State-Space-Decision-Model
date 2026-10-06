"""CPU-only integration of the L planner with D's real adapter and test oracle.

This command never loads model weights. Scripted choices exercise transport and
checking; their success is not learned capability or independent task scoring.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .kernel import check_program
from .planner import plan_checked_answer


def run(adapter_root: Path, fixtures: Path) -> dict:
    adapter_root = adapter_root.resolve()
    if not (adapter_root / "decide/backends/oracle.py").is_file() or not (
            adapter_root / "decide/backends/replay.py").is_file():
        raise ValueError("adapter root must contain the reviewed Track D backends")
    sys.path.insert(0, str(adapter_root))
    import decide
    from decide import Decider
    from decide.backends import OracleBackend, ReplayBackend
    from decide.backends.replay import normalize_response
    from decide.calibration import OptionCountTemperature
    from decide.schema import request_fingerprint
    if Path(decide.__file__).resolve().parent != adapter_root / "decide":
        raise ValueError("another decide package was already imported")
    catalog = json.loads((fixtures / "catalog.json").read_text(encoding="utf-8"))
    scenarios = [
        (1, ["Assert one selected fact", "atlas-v1 field repairable"], "accepted"),
        (2, ["Compare two selected", "Fact atlas-v1 field price", "Fact boreal-v1 field price", "Relation greater_than"], "accepted"),
        (2, ["Compare two selected", "Fact atlas-v1 field price", "Fact boreal-v1 field price", "Relation less_than"], "kernel_rejected"),
    ]
    rows = []

    def normalized_response(response):
        # D's helper removes only latency, backend diagnostics and the replay
        # backend's served_by revision note. The request UUID is not hashed.
        normalized = normalize_response(response)
        normalized["ssdm"].pop("request_id", None)
        return normalized

    def checked_meaning(result, host, evidence):
        if result.program is None:
            return None
        checked = check_program(result.program, host, evidence)
        return {"status": checked.status, "reason": checked.reason,
                "outputs": [act.to_dict() for act in checked.outputs]}

    for number, selectors, expected_status in scenarios:
        case_path = next((fixtures / "cases").glob(f"{number:02d}_*.json"))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["host"]["limits"]["max_proposal_calls"] = len(selectors)
        backend = OracleBackend(margin=20.0)
        calibration = OptionCountTemperature()
        adapter = Decider(backend, calibration=calibration)
        requests = []
        oracle_responses = []
        recordings = []

        def provider(request):
            target = selectors[len(requests)]
            qid, question = next(iter(request["questions"].items()))
            candidates = [key for key, description in question["criteria"].items() if target in description]
            if len(candidates) != 1:
                raise ValueError("ambiguous scripted selector")
            backend.labels[qid] = candidates[0]
            requests.append(request)
            response = adapter.decide(request)
            oracle_responses.append(response)
            if response.get("ssdm", {}).get("status") == "ok":
                plan = backend.plan(request)
                scored = backend.score(request, lambda: None)
                recordings.append({
                    "fingerprint": request_fingerprint(request), "logits": scored.logits,
                    "plan": {"inference_calls": plan.inference_calls,
                             "max_call_input_tokens": plan.max_call_input_tokens,
                             "total_input_tokens": plan.total_input_tokens},
                    "inference_calls": scored.inference_calls,
                    "total_input_tokens": scored.total_input_tokens,
                    "extras": scored.extras,
                })
            return response

        result = plan_checked_answer(
            case["question"], case["host"], catalog, provider,
            model="test-oracle", parent_revision=backend.revision()["revision_id"],
            expected_served_config={"calibration": calibration.identity(),
                                    "policy_masking": "none (served covers every supplied option)"},
        )
        with TemporaryDirectory(prefix="ssdm-l0-replay-") as temp:
            fixture_dir = Path(temp)
            manifest = {"fixture_id": "synthetic-l0-planner", "revision": backend.revision(),
                        "calibration": calibration.identity(), "enable_v0_1_deltas": False}
            (fixture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (fixture_dir / "recordings.jsonl").write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in recordings), encoding="utf-8")
            replay_backend = ReplayBackend(fixture_dir)
            replay_adapter = Decider(replay_backend, calibration=calibration)
            replay_requests = []
            replay_responses = []

            def replay_provider(request):
                replay_requests.append(request)
                response = replay_adapter.decide(request)
                replay_responses.append(response)
                return response

            replay_result = plan_checked_answer(
                case["question"], case["host"], catalog, replay_provider,
                model="test-oracle", parent_revision=backend.revision()["revision_id"],
                expected_served_config={"calibration": calibration.identity(),
                                        "policy_masking": "none (served covers every supplied option)"},
            )
            original_meaning = checked_meaning(result, case["host"], catalog)
            replay_meaning = checked_meaning(replay_result, case["host"], catalog)
            replay_fingerprints = [request_fingerprint(request) for request in replay_requests]
            request_ids_differ = all(
                original["ssdm"]["request_id"] != replayed["ssdm"]["request_id"]
                for original, replayed in zip(requests, replay_requests)
            )
            semantic_matches = (result.status == replay_result.status and result.reason == replay_result.reason
                                and result.text == replay_result.text and original_meaning == replay_meaning)
            normalized_oracle = [normalized_response(response) for response in oracle_responses]
            normalized_replay = [normalized_response(response) for response in replay_responses]
            response_matches = (len(oracle_responses) == len(replay_responses)
                                and normalized_oracle == normalized_replay)
            replay_matches = (semantic_matches and response_matches
                              and replay_fingerprints == [record["fingerprint"] for record in recordings]
                              and len(replay_requests) == len(requests) and request_ids_differ)
            misses = None
            if number == 1:
                misses = {}
                for change in ("task", "evidence"):
                    changed_catalog = deepcopy(catalog)
                    changed_task = case["question"]
                    if change == "task":
                        changed_task += " Please repeat the answer."
                    else:
                        changed_catalog["records"][0]["fields"]["price"]["value"] = "201.90"
                    responses = []

                    def miss_provider(request):
                        response = replay_adapter.decide(request)
                        responses.append(response)
                        return response

                    miss_result = plan_checked_answer(
                        changed_task, case["host"], changed_catalog, miss_provider,
                        model="test-oracle", parent_revision=backend.revision()["revision_id"],
                        expected_served_config={"calibration": calibration.identity(),
                                                "policy_masking": "none (served covers every supplied option)"},
                    )
                    misses[change] = {
                        "planner_status": miss_result.status,
                        "adapter_status": responses[0]["ssdm"]["status"] if responses else None,
                        "adapter_error_code": responses[0]["ssdm"]["error"]["code"] if responses else None,
                    }
        rows.append({"case_id": case["case_id"], "scripted_selectors": selectors,
                     "expected_status": expected_status, "status": result.status,
                     "reason": result.reason, "text": result.text,
                     "provider_calls": result.provider_calls, "forward_calls": result.forward_calls,
                     "requests": requests, "trace": result.trace,
                     "recorded_fingerprints": [record["fingerprint"] for record in recordings],
                     "synthetic_recordings": recordings,
                     "replay_status": replay_result.status, "replay_reason": replay_result.reason,
                     "replay_text": replay_result.text, "checked_meaning": original_meaning,
                     "replay_meaning": replay_meaning,
                     "replay_fingerprints": replay_fingerprints,
                     "request_ids_differ": request_ids_differ,
                     "semantic_matches": semantic_matches, "response_matches": response_matches,
                     "oracle_responses": oracle_responses, "replay_responses": replay_responses,
                     "normalized_oracle_responses": normalized_oracle,
                     "normalized_replay_responses": normalized_replay,
                     "replay_matches": replay_matches, "replay_requests": replay_requests,
                     "misses": misses})
    return {"scope": "scripted_oracle_recorded_replay_actual_adapter_CPU_no_model",
            "adapter_sha": subprocess.check_output(["git", "-C", str(adapter_root), "rev-parse", "HEAD"], text=True).strip(),
            "rows": rows,
            "passed": all(row["status"] == row["expected_status"] and row["replay_matches"] for row in rows)
            and all(miss["adapter_error_code"] == "replay_miss"
                    for row in rows if row["misses"] for miss in row["misses"].values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.adapter_root, args.fixtures)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
