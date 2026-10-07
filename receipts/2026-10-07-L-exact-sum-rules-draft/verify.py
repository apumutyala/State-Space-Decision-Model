"""Reproduce the local #175 boundary checks; no model or external data use."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COMMAND = [sys.executable, "-m", "unittest", "discover", "-s", "language_growth/tests", "-v"]

if sys.get_int_max_str_digits() != 4300:
    raise RuntimeError("receipt requires the pinned 4300-digit integer string limit")
completed = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
log = completed.stdout + completed.stderr
match = re.search(r"Ran (\d+) tests", log)
test_count = int(match.group(1)) if match else None
(HERE / "tests.txt").write_text(log, encoding="utf-8")
files = [
    "language_growth/types.py", "language_growth/kernel.py", "language_growth/realize.py",
    "language_growth/tests/test_exact_sum.py", "interfaces/DECISION_LANGUAGE_V0.md",
    "plans/codex/L1_EXACT_SUM.md", "STATUS.md", "receipts/2026-10-07-L-exact-sum-rules-draft/verify.py",
    "receipts/2026-10-07-L-exact-sum-rules-draft/tests.txt",
]
historical = ROOT / "receipts/2026-10-07-L-exact-sum-integer-carry"
receipt = {
    "scope": "local unpushed issue #175 ExactSum boundary draft, no qualification claim",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
    "python": sys.version,
    "integer_string_digit_limit": sys.get_int_max_str_digits(),
    "command": COMMAND,
    "exit_code": completed.returncode,
    "tests": test_count,
    "rational_oracle_pairs": 121,
    "maximum_case": {"terms": 16, "input_integer_digits": 4300, "scale": 5000,
                     "result_integer_digits": 4302, "complete_output_byte_cap": 262144},
    "versioning": {"document_draft": "0.1.2", "semantic_program": "v0.2", "decide": "v0.1",
                   "provenance": "deferred; future version requires Track D reconciliation"},
    "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
    "preserved_historical_integer_carry_receipt_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(historical.iterdir()) if path.is_file()
    },
    "remaining_gates": ["independent review", "joint Track D/L profile approval",
                        "agent-main port/reconciliation", "parent planner/evaluator integration"],
    "excluded": ["model/weights", "GPU", "Index/heldout", "paid compute", "push/PR/merge"],
}
(HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"exit_code": completed.returncode, "tests": test_count, "rational_oracle_pairs": 121,
                  "receipt": str(HERE / "receipt.json")}))
raise SystemExit(completed.returncode)
