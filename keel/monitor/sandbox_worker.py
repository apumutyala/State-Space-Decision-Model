"""Trusted driver: run ONLY inside a no-network Modal Sandbox, never locally.

Programs run unprivileged in fresh directories and receive input but no expected
outputs. The root supervisor owns the driver and validates no labels itself.
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile

OUTPUT_LIMIT = 1024 * 1024


def limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


def main():
    job = json.load(sys.stdin)
    results = []
    for value in job["inputs"]:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as output:
            os.chmod(directory, 0o755)
            code = os.path.join(directory, "solution.py")
            with open(code, "w", encoding="utf-8") as handle:
                handle.write(job["code"])
            os.chmod(code, 0o444)
            proc = subprocess.Popen([sys.executable, "-I", "-S", code], stdin=subprocess.PIPE,
                stdout=output, stderr=subprocess.DEVNULL, cwd=directory, env={"PATH": "/usr/local/bin:/usr/bin"},
                preexec_fn=limits)
            timed_out = False
            try:
                proc.communicate(value.encode("utf-8"), timeout=job["timeout"])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            output.seek(0)
            raw = output.read(OUTPUT_LIMIT + 1)
            results.append({"stdout": raw.decode("utf-8", errors="replace"), "returncode": proc.returncode,
                            "timed_out": timed_out, "overflow": len(raw) >= OUTPUT_LIMIT})
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    if os.environ.get("KEEL_MODAL_SANDBOX") != "1":
        raise RuntimeError("This execution driver is restricted to Modal Sandboxes")
    main()
