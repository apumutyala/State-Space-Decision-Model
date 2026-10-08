"""Local JSON/hash gates for the WSL CPU-8 controller; no provider or model imports."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import sys
import uuid

MANIFEST_SHA = "a7b37eaa21d7d44c52c14f08468bd0d1efa114ac32027166e4e59bfd8ce9e17d"
LIST_SHA = "9bd53612a49a9d367bcf19c35459d6cdbc2642aaf39b2d4ea3e210134928719a"
FILES = ("agent-main-5aac22.bundle", "decision-index-87d4650.bundle",
         "m2_first_cell-4247b340ddde.tar.zst", "package-manifest.json", "run_deadline.py",
         "run_m2_cpu8_tests.sh", "tokenizer.json", "tokenizer_config.json", "package.sha256")


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("gate refused")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: str, value) -> None:
    target = Path(path)
    temp = target.with_suffix(".writing")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def main(args: list[str]) -> None:
    mode, *args = args
    if mode == "package":
        directory, output = args
        root = Path(directory)
        require(sha(root / "package-manifest.json") == MANIFEST_SHA)
        require(sha(root / "package.sha256") == LIST_SHA)
        entries = []
        for line in (root / "package.sha256").read_text(encoding="ascii").splitlines():
            digest, name = line.split("  ")
            require(Path(name).name == name and name not in (".", ".."))
            require(name in FILES and not (root / name).is_symlink())
            require(sha(root / name) == digest)
            entries.append({"name": name, "sha256": digest, "bytes": (root / name).stat().st_size})
        entries.append({"name": "package.sha256", "sha256": LIST_SHA,
                        "bytes": (root / "package.sha256").stat().st_size})
        require(len({item["name"] for item in entries}) == len(entries))
        require({item["name"] for item in entries} == set(FILES))
        total = sum(item["bytes"] for item in entries)
        free_bytes = int(os.environ["CPU8_STORAGE_FREE_BYTES"])
        require(total <= free_bytes <= 100_000_000_000)
        save(output, {"files": entries, "bytes": total, "storage_verified": False,
                      "storage_free_tier_bytes_per_month": 100_000_000_000,
                      "storage_cost_usd": 0,
                      "live_preflight_free_bytes": free_bytes,
                      "storage_cost_note": "Fits the live free allowance supplied by the spend preflight."})
    elif mode == "names":
        for item in load(args[0])["files"]:
            print(item["name"])
    elif mode == "list-sha":
        print(LIST_SHA)
    elif mode == "uploaded":
        path, name, key = args
        staged = load(path)
        require(name in FILES)
        staged.setdefault("uploaded_objects", []).append({"key": key, **next(
            item for item in staged["files"] if item["name"] == name)})
        save(path, staged)
    elif mode == "verified":
        staged = load(args[0])
        require(len(staged["uploaded_objects"]) == len(FILES))
        staged["storage_verified"] = True
        staged["verification"] = "import completed and remote sha256sum -c package.sha256 passed"
        save(args[0], staged)
    elif mode == "event":
        path, phase, current, rc, stdout_path, stderr_path = args
        def redact(text: str) -> str:
            text = re.sub(r"(?i)(bearer\s+)[^\s\"'<>]+", r"\1[REDACTED]", text)
            return re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)\b[\"']?\s*[:=]\s*[\"']?)[^\s,\"'}]+",
                          r"\1[REDACTED]", text)
        events = load(path) if Path(path).exists() else []
        events.append({"phase": phase, "current_file": current, "rc": int(rc),
                       "stdout": redact(Path(stdout_path).read_text(encoding="utf-8", errors="replace")),
                       "stderr": redact(Path(stderr_path).read_text(encoding="utf-8", errors="replace"))})
        save(path, events)
    elif mode == "identity":
        path, name = args
        node = load(path)
        require(node["name"] == name)
        identifier = str(uuid.UUID(node["id"]))
        print(identifier, "created")
    elif mode == "node":
        path, name = args
        node = load(path)
        require(node["name"] == name and node["chip"] == "cpu-8" and node["status"].startswith("running") and node["yours"] is True)
        rate = node["rate"]["node_per_min_usd_micros"]
        require(isinstance(rate, int) and not isinstance(rate, bool))
        require(0 < rate <= 12_000)
        identifier = node["id"]
        require(isinstance(identifier, str) and bool(identifier) and not any(c.isspace() for c in identifier))
        print(identifier, "running")
    elif mode == "command":
        command = load(args[0])
        # Unknown provider response shapes fail closed. Never select "newest".
        identifier = command["id"]
        require(isinstance(identifier, str) and re.fullmatch(r"cmd-[a-z0-9]+", identifier) is not None)
        require(command["status"] in ("running", "completed", "complete", "exited"))
        print(identifier)
    elif mode == "completion":
        path, rc = args
        output = Path(path).read_text(encoding="utf-8", errors="strict")
        require("pending_wake" not in output)
        exit_code = int(rc)
        require(0 <= exit_code <= 255)
        if exit_code == 0:
            # The frozen runner prints this only after launch, nonempty artifact
            # checks and JUnit validation. CLI/log-follow success is insufficient.
            reports = []
            decoder = json.JSONDecoder()
            for match in re.finditer(r"(?m)^\{", output):
                try:
                    report, _ = decoder.raw_decode(output[match.start():])
                except ValueError:
                    continue
                if isinstance(report, dict) and "source_sha" in report and "junit_counts" in report:
                    reports.append(report)
            require(len(reports) == 1)
            report = reports[0]
            require(report["source_sha"] == "4247b340dddecf620de6ff22751256f4127237cb")
            require(report["h100_request_submitted"] is False)
            tests = report["launcher_tests"]
            require(isinstance(tests, dict))
            require(all(type(tests[key]) is int and tests[key] >= 0 for key in ("passed", "failed", "skipped")))
            require(tests["passed"] > 0 and tests["failed"] == 0)
            counts = report["junit_counts"]
            require(isinstance(counts, dict))
            require(all(type(counts[key]) is int and counts[key] >= 0 for key in ("tests", "failures", "errors", "skipped")))
            require(counts["failures"] == counts["errors"] == 0)
            # The frozen runner reads the root attributes. A pytest testsuites
            # wrapper has none: all-zero counts then rely on launcher's recursive
            # testcase counts above. Any populated root must agree with them.
            require(all(value == 0 for value in counts.values()) or (
                counts["tests"] == tests["passed"] + tests["skipped"] and counts["skipped"] == tests["skipped"]))
            require(isinstance(report["context_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", report["context_sha256"]) is not None)
            require(type(report["context_bytes"]) is int and report["context_bytes"] > 0)
        print(exit_code)
    elif mode == "receipt":
        path, phase, file, rc, name, node_id, state, stop, run, export, elapsed, imported, command, log_rc, completed = args
        events_path = str(Path(path).parent / "transfer-events.json")
        save(path, {"schema": "ssdm.m2_cpu8_controller.v1", "issue": 171,
                    "phase": phase, "current_file": file, "rc": int(rc),
                    "node_name": name, "node_id": node_id, "node_state": state,
                    "stop_result": stop, "workload_rc": run, "export_rc": export,
                    "workload_command_id": command, "log_follow_rc": log_rc,
                    "workload_completion_verified": completed == "true",
                    "elapsed_seconds": int(elapsed), "max_retries": 0,
                    "import_outcome": imported, "hard_cap_usd": 3.0,
                    "max_exposure_including_original_incident_usd": 2.346700,
                    "transfer_events": load(events_path) if Path(events_path).exists() else []})
    else:
        raise ValueError("unknown local gate")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (AssertionError, KeyError, ValueError, OSError, TypeError):
        # Provider responses and CLI error text can contain secrets; never echo them.
        print("CPU-8 local JSON/hash gate refused", file=sys.stderr)
        raise SystemExit(65)
