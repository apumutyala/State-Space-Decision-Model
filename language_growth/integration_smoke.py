"""CPU-only integration of the L planner with D's real adapter and test oracle.

This command never loads model weights. Scripted choices exercise transport and
checking; their success is not learned capability or independent task scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .planner import plan_checked_answer


def run(adapter_root: Path, fixtures: Path) -> dict:
    adapter_root = adapter_root.resolve()
    if not (adapter_root / "decide/backends/oracle.py").is_file():
        raise ValueError("adapter root must contain the reviewed Track D oracle")
    sys.path.insert(0, str(adapter_root))
    import decide
    from decide import Decider
    from decide.backends import OracleBackend
    from decide.calibration import OptionCountTemperature
    if Path(decide.__file__).resolve().parent != adapter_root / "decide":
        raise ValueError("another decide package was already imported")
    catalog = json.loads((fixtures / "catalog.json").read_text(encoding="utf-8"))
    scenarios = [
        (1, ["Assert one selected fact", "atlas-v1 field repairable"], "accepted"),
        (2, ["Compare two selected", "Fact atlas-v1 field price", "Fact boreal-v1 field price", "Relation greater_than"], "accepted"),
        (2, ["Compare two selected", "Fact atlas-v1 field price", "Fact boreal-v1 field price", "Relation less_than"], "kernel_rejected"),
    ]
    rows = []
    for number, selectors, expected_status in scenarios:
        case_path = next((fixtures / "cases").glob(f"{number:02d}_*.json"))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["host"]["limits"]["max_proposal_calls"] = len(selectors)
        backend = OracleBackend(margin=20.0)
        calibration = OptionCountTemperature()
        adapter = Decider(backend, calibration=calibration)
        requests = []

        def provider(request):
            target = selectors[len(requests)]
            qid, question = next(iter(request["questions"].items()))
            candidates = [key for key, description in question["criteria"].items() if target in description]
            if len(candidates) != 1:
                raise ValueError("ambiguous scripted selector")
            backend.labels[qid] = candidates[0]
            requests.append(request)
            return adapter.decide(request)

        result = plan_checked_answer(
            case["question"], case["host"], catalog, provider,
            model="test-oracle", parent_revision=backend.revision()["revision_id"],
            expected_served_config={"calibration": calibration.identity(),
                                    "policy_masking": "none (served covers every supplied option)"},
        )
        rows.append({"case_id": case["case_id"], "scripted_selectors": selectors,
                     "expected_status": expected_status, "status": result.status,
                     "reason": result.reason, "text": result.text,
                     "provider_calls": result.provider_calls, "forward_calls": result.forward_calls,
                     "requests": requests, "trace": result.trace})
    return {"scope": "scripted_oracle_actual_adapter_CPU_no_model",
            "adapter_sha": subprocess.check_output(["git", "-C", str(adapter_root), "rev-parse", "HEAD"], text=True).strip(),
            "rows": rows,
            "passed": all(row["status"] == row["expected_status"] for row in rows)}


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
