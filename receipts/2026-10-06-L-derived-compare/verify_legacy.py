"""Compare all original v0.1 fixture attempts with the unchanged base kernel."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from language_growth.kernel import check_program
from language_growth.realize import _speech, render_checked

BASE = "0f4b8ad2ea717c79858ddfbc4deb55c9b71940af"
source = subprocess.check_output(["git", "show", BASE + ":language_growth/kernel.py"], cwd=ROOT)
baseline = types.ModuleType("language_growth._base_kernel")
exec(compile(source, "<base-kernel>", "exec"), baseline.__dict__)
pack = ROOT / "fixtures/language_growth"
catalog = json.loads((pack / "catalog.json").read_text())
manifest = json.loads((pack / "manifest.json").read_text())
rows = []
for entry in manifest["cases"]:
    item = json.loads((pack / entry["path"]).read_text())
    attempts = [item["proposal"]]
    repaired = item.get("repair", {}).get("repaired_proposal")
    if repaired is not None:
        attempts.append(repaired)
    for ordinal, program in enumerate(attempts):
        before = baseline.check_program(program, item["host"], catalog)
        after = check_program(program, item["host"], catalog)
        assert before.to_dict() == after.to_dict()
        text = None
        if before.status == "accepted":
            baseline.validate_release(before, item["host"], catalog)
            old_text = "\n".join(_speech(act) for act in before.outputs)
            text = render_checked(after, item["host"], catalog)
            assert old_text.encode() == text.encode()
        rows.append({"case_id": item["case_id"], "attempt": ordinal,
                     "status": after.status, "reason": after.reason,
                     "checked_result_sha256": hashlib.sha256(json.dumps(after.to_dict(), sort_keys=True).encode()).hexdigest(),
                     "rendered_sha256": hashlib.sha256(text.encode()).hexdigest() if text is not None else None})
report = {"baseline_commit": BASE, "baseline_kernel_git_bytes_sha256": hashlib.sha256(source).hexdigest(),
          "cases": len(manifest["cases"]), "attempts": len(rows), "mismatches": [], "results": rows}
(Path(__file__).parent / "legacy_compatibility.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Identical checked results and released UTF-8 text for {len(rows)} v0.1 attempts in {len(manifest['cases'])} cases.")
