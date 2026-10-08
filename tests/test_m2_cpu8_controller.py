"""L-42/L-46: no-model, no-network subprocess tests against a fake AUTOR (stdlib only).

Run on WSL: python3 -m unittest discover -s tests -p test_m2_cpu8_controller.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("agent-main-5aac22.bundle", "decision-index-87d4650.bundle",
         "m2_first_cell-4247b340ddde.tar.zst", "package-manifest.json", "run_deadline.py",
         "run_m2_cpu8_tests.sh", "tokenizer.json", "tokenizer_config.json")
FAKE = r'''
import json, os, pathlib, sys
a = sys.argv[1:]
assert a[:2] == ["--workspace", "private-workspace"]
a = a[2:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(a) + "\n")
scenario = os.environ["FAKE_SCENARIO"]
state = pathlib.Path(os.environ["FAKE_NODE"] + ".state")
if a[:2] == ["storage", "cp"]:
    if pathlib.Path(a[2]).is_file():
        if scenario == "upload_failure":
            print("error: request failed Authorization: Bearer secret-token", file=sys.stderr)
            sys.exit(31)
        print("uploaded " + pathlib.Path(a[2]).name)
    else:
        pathlib.Path(a[3]).write_text("fake exported artifact")
elif a[:2] == ["node", "create"]:
    pathlib.Path(os.environ["FAKE_NODE"]).write_text(a[a.index("--name") + 1])
    state.write_text("running")
    print(json.dumps({"name": a[a.index("--name") + 1], "id": "00000000-0000-4000-8000-000000000001"}))
elif a[:2] == ["node", "get"]:
    print(json.dumps({"name": a[2], "id": "00000000-0000-4000-8000-000000000001", "chip": "cpu-8", "status": "running (idle)", "yours": True,
                      "rate": {"node_per_min_usd_micros": 13000 if scenario == "bad_rate" else 8040}}))
elif a[:2] == ["node", "stop"]:
    if scenario == "stop_failure": sys.exit(33)
    state.write_text("stopped")
elif a[0] == "import":
    if scenario == "import_failure": sys.exit(32)
    if scenario == "import_timeout": sys.exit(124)
    print("import complete")
elif a[0] == "logs":
    assert a[2:] == ["cmd-fake1", "-f"], a
    if scenario == "logs_pending_wake":
        print("cmd-fake1 pending_wake")
        sys.exit(0)
    if scenario == "empty_logs": sys.exit(0)
    if scenario == "workload_failure": sys.exit(34)
    report = {"source_sha": "4247b340dddecf620de6ff22751256f4127237cb",
              "launcher_tests": {"skipped": 0, "passed": 71, "failed": 0},
              "junit_counts": {"tests": 71, "failures": 0, "errors": 0, "skipped": 0},
              "context_sha256": "a" * 64, "context_bytes": 1234, "h100_request_submitted": False}
    if scenario == "wrong_source": report["source_sha"] = "b" * 40
    if scenario == "no_junit": del report["junit_counts"]
    if scenario == "junit_failure": report["junit_counts"]["failures"] = 1
    if scenario == "all_skipped": report["junit_counts"]["skipped"] = 71
    if scenario == "skipped_launcher": report["launcher_tests"]["skipped"] = True
    if scenario == "zero_passed": report["launcher_tests"]["passed"] = 0
    if scenario == "launcher_failure": report["launcher_tests"]["failed"] = 1
    if scenario == "junit_wrapper": report["junit_counts"]["tests"] = 0
    if scenario == "bad_context": report["context_sha256"] = None
    if scenario == "submitted": report["h100_request_submitted"] = True
    print(json.dumps(report, indent=2))
    if scenario == "duplicate_report": print(json.dumps(report))
elif a[0] == "export":
    assert state.read_text() == "running", "export must precede stop"
    if scenario == "export_failure": sys.exit(35)
elif a[0] == "run":
    if scenario == "remote_hash_failure" and "sha256sum -c -" in a[-1]: sys.exit(41)
    if "run_deadline.py" in a[-1]:
        assert state.read_text() == "running", "workload must launch on running node"
        assert "-d" in a and "--json" in a and "--on-wake" not in a, a
        registration = {"id": "cmd-fake1", "status": "pending_wake" if scenario == "registration_pending_wake" else "running"}
        if scenario == "registration_missing_id": del registration["id"]
        if scenario == "registration_bad_id": registration["id"] = "cmd-fake1 --other-command"
        if scenario == "registration_unknown_status": registration["status"] = "unknown"
        print(json.dumps(registration))
else:
    raise RuntimeError("unexpected command: " + repr(a))
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "execute fake-AUTOR shell checks on WSL/Linux")
class ControllerTests(unittest.TestCase):
    def attempt(self, scenario):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pkg, out, scripts = base / "package", base / "out", base / "scripts"
            pkg.mkdir()
            scripts.mkdir()
            for name in FILES:
                (pkg / name).write_bytes((name + " frozen fake bytes\n").encode())
            (pkg / "supervise_cpu8.sh").write_text("NEVER UPLOAD THIS EXTRA FILE")
            lines = [hashlib.sha256((pkg / n).read_bytes()).hexdigest() + "  " + n for n in FILES]
            (pkg / "package.sha256").write_text("\n".join(lines) + "\n")
            helper = (ROOT / "scripts/m2_cpu8_json.py").read_text()
            for pin, name in (("MANIFEST_SHA", "package-manifest.json"), ("LIST_SHA", "package.sha256")):
                begin = helper.index(pin + ' = "') + len(pin + ' = "')
                end = helper.index('"', begin)
                helper = helper[:begin] + hashlib.sha256((pkg / name).read_bytes()).hexdigest() + helper[end:]
            (scripts / "m2_cpu8_json.py").write_text(helper)
            shutil.copyfile(ROOT / "scripts/supervise_m2_cpu8.sh", scripts / "supervise_m2_cpu8.sh")
            fake = base / "autor"
            fake.write_text("#!" + sys.executable + "\n" + FAKE)
            fake.chmod(0o700)
            env = dict(os.environ, AUTOR=str(fake), FAKE_SCENARIO=scenario, FAKE_LOG=str(base / "calls"),
                       FAKE_OBJECTS=str(base / "objects"), FAKE_NODE=str(base / "node"),
                       CPU8_STORAGE_FREE_BYTES="0" if scenario == "insufficient_free" else "100000000000")
            result = subprocess.run(["bash", str(scripts / "supervise_m2_cpu8.sh"), str(pkg), str(out),
                                     "private-workspace", "existing-private-bucket", "ubuntu-24.04"],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertTrue((out / "controller.json").is_file(), result.stderr)
            receipt = json.loads((out / "controller.json").read_text())
            calls = [json.loads(line) for line in (base / "calls").read_text().splitlines()] if (base / "calls").exists() else []
            staging = json.loads((out / "staging.json").read_text()) if (out / "staging.json").exists() else {}
            self.assertEqual(receipt["rc"], result.returncode)
            self.assertNotIn("secret-token", json.dumps(receipt))
            if staging:
                self.assertEqual({p["name"] for p in staging["files"]}, set(FILES) | {"package.sha256"})
            return result, receipt, calls, staging

    def test_upload_failure_never_creates_node_and_records_exit(self):
        result, receipt, calls, _ = self.attempt("upload_failure")
        self.assertEqual(result.returncode, 31)
        self.assertEqual(receipt["phase"], "storage_upload")
        self.assertEqual(receipt["current_file"], "agent-main-5aac22.bundle")
        self.assertEqual(receipt["node_state"], "not_created")
        self.assertEqual(receipt["stop_result"], "not_needed")
        self.assertEqual([c[:2] for c in calls], [["storage", "cp"]])
        self.assertEqual(receipt["transfer_events"][0]["rc"], 31)
        self.assertIn("error: request failed", receipt["transfer_events"][0]["stderr"])
        self.assertIn("[REDACTED]", receipt["transfer_events"][0]["stderr"])

    def test_insufficient_free_allowance_refuses_before_any_autor_command(self):
        result, receipt, calls, _ = self.attempt("insufficient_free")
        self.assertEqual(result.returncode, 65)
        self.assertEqual(receipt["phase"], "package_verify")
        self.assertEqual(receipt["stop_result"], "not_needed")
        self.assertEqual(calls, [])

    def test_import_failure_stops_and_never_runs_workload(self):
        result, receipt, calls, _ = self.attempt("import_failure")
        self.assertEqual(result.returncode, 32)
        self.assertEqual(receipt["phase"], "storage_import")
        self.assertEqual(receipt["node_id"], "00000000-0000-4000-8000-000000000001")
        self.assertEqual(receipt["node_state"], "stopped")
        self.assertEqual(calls[-1][:2], ["node", "stop"])
        self.assertEqual(receipt["import_outcome"], "unknown")
        self.assertFalse(any(c[0] in ("run", "logs", "export") for c in calls))

    def test_import_timeout_stops_with_unknown_import_outcome(self):
        result, receipt, calls, _ = self.attempt("import_timeout")
        self.assertEqual(result.returncode, 124)
        self.assertEqual(receipt["import_outcome"], "unknown")
        self.assertEqual(receipt["stop_result"], "stopped")
        self.assertEqual(calls[-1][:2], ["node", "stop"])
        self.assertFalse(any(c[0] in ("run", "logs", "export") for c in calls))

    def test_successful_order_and_frozen_deadlines(self):
        result, receipt, calls, staging = self.attempt("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["phase"], "complete")
        self.assertEqual(receipt["stop_result"], "stopped")
        self.assertTrue(staging["storage_verified"])
        self.assertEqual(staging["storage_cost_usd"], 0)
        self.assertEqual(len(receipt["transfer_events"]), 12)
        self.assertEqual(receipt["transfer_events"][9]["stdout"], "import complete\n")
        self.assertEqual(receipt["transfer_events"][9]["rc"], 0)
        self.assertEqual([c[:2] if c[0] in ("node", "storage") else [c[0]] for c in calls],
                         [["storage", "cp"]] * 9 + [["node", "create"], ["node", "get"],
                          ["import"], ["run"], ["run"], ["logs"], ["export"],
                          ["node", "stop"], ["storage", "cp"]])
        run = next(c for c in calls if "run_deadline.py" in c[-1])
        self.assertIn("-d", run)
        self.assertIn("--json", run)
        self.assertIn("--seconds 8610 --kill-grace-seconds 30 -- bash ./run_m2_cpu8_tests.sh", run[-1])
        self.assertFalse(any("--on-wake" in c for c in calls))
        self.assertEqual(receipt["workload_command_id"], "cmd-fake1")
        self.assertEqual(receipt["workload_rc"], "0")
        self.assertTrue(receipt["workload_completion_verified"])
        source = (ROOT / "scripts/supervise_m2_cpu8.sh").read_text()
        self.assertIn("TOTAL_SECONDS=$((175 * 60))", source)
        self.assertIn('bounded 8700 "${AUTOR_CMD[@]}" logs "$NODE" "$COMMAND_ID" -f', source)

    def test_pending_wake_and_empty_logs_rc0_never_claim_workload_or_export_success(self):
        for scenario in ("registration_pending_wake", "logs_pending_wake", "empty_logs"):
            with self.subTest(scenario=scenario):
                result, receipt, calls, staging = self.attempt(scenario)
                self.assertEqual(result.returncode, 65, result.stderr)
                self.assertIn(receipt["phase"], ("register_run", "workload"))
                self.assertNotEqual(receipt["workload_rc"], "0")
                self.assertFalse(receipt["workload_completion_verified"])
                self.assertEqual(receipt["export_rc"], "not_run")
                self.assertFalse(any(c[0] == "export" for c in calls))
                self.assertEqual(calls[-1][:2], ["node", "stop"])
                self.assertEqual(receipt["node_state"], "stopped")
                self.assertTrue(staging["storage_verified"])
                self.assertEqual(sum("run_deadline.py" in c[-1] for c in calls), 1)
                if scenario != "registration_pending_wake":
                    self.assertEqual(receipt["log_follow_rc"], "0")
                    self.assertEqual(receipt["workload_rc"], "unverified")

    def test_rc0_requires_the_exact_runner_post_junit_report(self):
        for scenario in ("wrong_source", "no_junit", "junit_failure", "all_skipped",
                         "skipped_launcher", "zero_passed", "launcher_failure", "bad_context",
                         "submitted", "duplicate_report"):
            with self.subTest(scenario=scenario):
                result, receipt, calls, _ = self.attempt(scenario)
                self.assertEqual(result.returncode, 65, result.stderr)
                self.assertEqual(receipt["workload_rc"], "unverified")
                self.assertFalse(receipt["workload_completion_verified"])
                self.assertFalse(any(c[0] == "export" for c in calls))
                self.assertEqual(receipt["stop_result"], "stopped")

    def test_unknown_registration_refuses_without_following_newest_command(self):
        for scenario in ("registration_missing_id", "registration_bad_id", "registration_unknown_status"):
            with self.subTest(scenario=scenario):
                result, receipt, calls, _ = self.attempt(scenario)
                self.assertEqual(result.returncode, 65, result.stderr)
                self.assertEqual(receipt["phase"], "register_run")
                self.assertEqual(receipt["workload_rc"], "not_run")
                self.assertFalse(receipt["workload_completion_verified"])
                self.assertFalse(any(c[0] in ("logs", "export") for c in calls))
                self.assertEqual(receipt["stop_result"], "stopped")

    def test_original_runner_zero_root_counts_require_recursive_launcher_passes(self):
        result, receipt, _, _ = self.attempt("junit_wrapper")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["workload_rc"], "0")
        self.assertTrue(receipt["workload_completion_verified"])

    def test_high_rate_refuses_and_stops(self):
        result, receipt, calls, _ = self.attempt("bad_rate")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt["phase"], "node_validate")
        self.assertEqual(calls[-1][:2], ["node", "stop"])
        self.assertFalse(any(c[0] == "import" for c in calls))

    def test_remote_hash_list_failure_stops_before_workload(self):
        result, receipt, calls, staging = self.attempt("remote_hash_failure")
        self.assertEqual(result.returncode, 41)
        self.assertEqual(receipt["phase"], "remote_hash")
        self.assertEqual(receipt["node_state"], "stopped")
        self.assertFalse(staging["storage_verified"])
        self.assertEqual(calls[-1][:2], ["node", "stop"])
        self.assertFalse(any(c[0] == "logs" or "--on-wake" in c for c in calls))

    def test_workload_failure_exports_stops_without_retry(self):
        result, receipt, calls, _ = self.attempt("workload_failure")
        self.assertEqual(result.returncode, 34)
        self.assertEqual(receipt["workload_rc"], "34")
        self.assertFalse(receipt["workload_completion_verified"])
        self.assertEqual(sum(c[0] == "logs" for c in calls), 1)
        self.assertEqual(sum(c[0] == "export" for c in calls), 1)
        self.assertEqual(receipt["stop_result"], "stopped")

    def test_stop_failure_is_nonzero_with_unknown_state(self):
        result, receipt, calls, _ = self.attempt("stop_failure")
        self.assertEqual(result.returncode, 70)
        self.assertEqual(receipt["node_state"], "unknown")
        self.assertEqual(receipt["stop_result"], "failed")
        self.assertFalse(any("--on-wake" in c for c in calls))
        self.assertEqual(sum(c[:2] == ["node", "stop"] for c in calls), 2)


class StaticTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "requires GNU sha256sum on WSL/Linux")
    def test_pinned_checksum_list_is_verified_before_payload_list(self):
        controller = (ROOT / "scripts/supervise_m2_cpu8.sh").read_text()
        helper = (ROOT / "scripts/m2_cpu8_json.py").read_text()
        self.assertIn('LIST_SHA=$(python3 "$JSON_HELPER" list-sha)', controller)
        self.assertIn("printf '%s  package.sha256\\\\n' '$LIST_SHA' | sha256sum -c -", controller)
        self.assertLess(controller.index("sha256sum -c -"), controller.index("sha256sum -c package.sha256"))
        self.assertIn('elif mode == "list-sha":\n        print(LIST_SHA)', helper)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "run_m2_cpu8_tests.sh"
            payload.write_text("changed runner\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / "package.sha256").write_text(digest + "  run_m2_cpu8_tests.sh\n")
            old_check = subprocess.run(["sha256sum", "-c", "package.sha256"], cwd=root,
                                       capture_output=True, text=True)
            self.assertEqual(old_check.returncode, 0)
            pinned = subprocess.check_output([sys.executable, str(ROOT / "scripts/m2_cpu8_json.py"),
                                              "list-sha"], text=True).strip()
            command = f"printf '%s  package.sha256\\n' '{pinned}' | sha256sum -c - && sha256sum -c package.sha256"
            new_check = subprocess.run(["bash", "-lc", command], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(new_check.returncode, 0)

    def test_large_direct_node_copy_cannot_recur(self):
        source = (ROOT / "scripts/supervise_m2_cpu8.sh").read_text()
        self.assertNotIn('"${AUTOR_CMD[@]}" cp', source)
        self.assertNotIn("storage bucket create", source)
        self.assertIn('storage cp "$STAGE/$file"', source)
        self.assertLess(source.index('storage cp "$STAGE/$file"'), source.index('node create --chip cpu-8'))
        self.assertIn('import "$NODE" --connection "storage/$BUCKET"', source)
        self.assertIn('AUTOR_CMD=("$AUTOR" --workspace "$WORKSPACE")', source)
        self.assertIn("trap on_exit EXIT", source)
        self.assertIn("L-42", (ROOT / "LESSONS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
