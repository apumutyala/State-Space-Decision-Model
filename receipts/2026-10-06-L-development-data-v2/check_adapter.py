"""Verify the recorded v0.1 development request reaches D's CPU oracle."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide import Decider
from decide.backends import OracleBackend
from language_growth.development_data import generate_development_tasks


class SyntheticRevisionOracle(OracleBackend):
    def revision(self):
        revision = super().revision()
        revision["revision_id"] = "synthetic-no-model:v1"
        return revision


request = generate_development_tasks(17, 1, split="development_train")[0]["model_inputs"][0]
response = Decider(SyntheticRevisionOracle()).decide(request)
summary = {
    "request_schema": request["ssdm"]["schema"],
    "response_status": response["ssdm"]["status"],
    "response_schema": response["ssdm"]["schema"],
    "fingerprint_algorithm": response["ssdm"]["diagnostics"].get("request_fingerprint_algorithm"),
    "backend": "synthetic CPU oracle; no weights",
}
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["response_status"] != "ok":
    raise SystemExit(1)
