"""Local package/source/data-pin checks only; never runs the staged model suite.

Run with zstandard already available in the interpreter. Positional arguments:
new prepared package, original prepared package, refreshed-controller checkout.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import zstandard

package, original, root = map(Path, sys.argv[1:])
source = "11fb140ce64802222d8abe796267e22e14fe4959"
manifest = json.loads((package / "package-manifest.json").read_text())
assert manifest["target"]["source_sha"] == source


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


spec = importlib.util.spec_from_file_location("local_cpu8_gates", root / "scripts/m2_cpu8_json.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
# Synthetic capacity input exercises the local hash gate only. It is not a live
# free-storage observation and must never authorize upload or spend.
os.environ["CPU8_STORAGE_FREE_BYTES"] = "100000000000"
with tempfile.TemporaryDirectory() as scratch:
    receipt = Path(scratch) / "local-hash-gate.json"
    gate.main(["package", str(package), str(receipt)])
    hashes = json.loads(receipt.read_text())
assert len(hashes["files"]) == 9

unchanged = ("decision-index-87d4650.bundle", "m2_first_cell-4247b340ddde.tar.zst",
             "run_deadline.py", "tokenizer.json", "tokenizer_config.json")
for name in unchanged:
    assert digest(package / name) == digest(original / name)
runner = (original / "run_m2_cpu8_tests.sh").read_bytes()
for before, after in (
    (b"SOURCE_SHA=4247b340dddecf620de6ff22751256f4127237cb", b"SOURCE_SHA=" + source.encode()),
    (b"ORIGIN_SHA=5aac22e93968747cd0cf6e45603d55d8a9ac75e4", b"ORIGIN_SHA=" + source.encode()),
    (b"agent-main-5aac22.bundle", b"agent-main-11fb140.bundle"),
    (b'test -s "$OUT/m2_first_cell-4247b340ddde.tar.zst"', b'test -s "$OUT/m2_first_cell-11fb140ce648.tar.zst"'),
    (b'"source_sha": "4247b340dddecf620de6ff22751256f4127237cb"', b'"source_sha": "' + source.encode() + b'"'),
):
    assert runner.count(before) == 1
    runner = runner.replace(before, after)
assert runner == (package / "run_m2_cpu8_tests.sh").read_bytes()

actual_data, context_manifest_sha = {}, None
with (package / "m2_first_cell-4247b340ddde.tar.zst").open("rb") as compressed:
    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if member.isfile() and (member.name.startswith("data/") or member.name == "CONTEXT_MANIFEST.json"):
                    value = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()
                    if member.name == "CONTEXT_MANIFEST.json":
                        context_manifest_sha = value
                    else:
                        assert member.name not in actual_data
                        actual_data[member.name] = value
assert actual_data == manifest["context_data_sha256"]
context = next(item for item in manifest["uploaded_inputs"] if item["remote_name"].endswith(".tar.zst"))
assert context_manifest_sha == context["context_manifest_sha256"]
assert len(actual_data) == 11  # Exact original allowlist excludes sealed TEST/D1B and weights.

git_env = {key: value for key, value in os.environ.items() if key not in ("GIT_DIR", "GIT_WORK_TREE")}
with tempfile.TemporaryDirectory() as directory:
    repo = Path(directory)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=git_env)
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", str(package / "agent-main-11fb140.bundle"),
                    "refs/remotes/origin/agent-main:refs/remotes/origin/agent-main"], check=True, env=git_env)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "refs/remotes/origin/agent-main"],
                                   env=git_env, text=True).strip()
    assert head == source
    original_source = "4247b340dddecf620de6ff22751256f4127237cb"
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", original_source, source, "--",
                    *manifest["reused_context"]["unchanged_pin_paths"]], check=True, env=git_env)
    test = subprocess.check_output(["git", "-C", str(repo), "show", source + ":tests/test_m2_first_cell_shell.py"],
                                   env=git_env, text=True)
    assert 'def _source_files(' in test and 'def _owned_process_state(' in test
    assert '== "live_owned"' in test and 'def _kill_owned_process_if_live(' in test

result = {"source_sha": source, "package_bytes": hashes["bytes"], "files": hashes["files"],
          "verified_context_data_files": actual_data, "context_manifest_sha256": context_manifest_sha,
          "offline_bundle_source_verified": True, "unchanged_pins_verified": True,
          "synthetic_capacity_check_only": True, "live_preflight_performed": False,
          "model_tests_run": False, "provider_calls": False, "cost_usd": 0}
(package / "local-validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ("source_sha", "package_bytes", "offline_bundle_source_verified",
                                             "unchanged_pins_verified", "model_tests_run", "provider_calls")}, indent=2))
