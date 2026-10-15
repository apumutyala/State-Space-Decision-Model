"""CPU-only, no-network execution; the workstation never executes submissions."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from .data import Execution


class ModalExecutor:
    def __init__(self, app, cache: Path, *, test_timeout: float = 2.0):
        import modal
        self.app = app
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.test_timeout = test_timeout
        worker = Path(__file__).with_name("sandbox_worker.py")
        self.worker_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
        self.image = modal.Image.debian_slim(python_version="3.12").add_local_file(str(worker), "/runner.py")
        self.lock = threading.Lock()
        self.runs = []

    def run(self, code: str, inputs: list[str]) -> list[Execution]:
        import modal
        payload = {"code": code, "inputs": inputs, "timeout": self.test_timeout}
        key = hashlib.sha256(json.dumps([self.worker_hash, payload], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            return [Execution(**r) for r in json.loads(path.read_text())["results"]]
        started = time.time()
        sandbox = modal.Sandbox.create("python", "/runner.py", app=self.app, image=self.image,
            block_network=True, cpu=1.0, memory=512, timeout=max(30, int(len(inputs) * (self.test_timeout + 1) + 20)),
            env={"KEEL_MODAL_SANDBOX": "1"})
        try:
            sandbox.stdin.write(json.dumps(payload))
            sandbox.stdin.write_eof()
            sandbox.stdin.drain()
            sandbox.wait()
            if sandbox.returncode != 0:
                raise RuntimeError(f"sandbox supervisor failed: {sandbox.returncode}: {sandbox.stderr.read()}")
            results = [Execution(**r) for r in json.loads(sandbox.stdout.read())]
            if len(results) != len(inputs):
                raise RuntimeError("sandbox omitted execution results")
            record = {"results": [r.__dict__ for r in results], "sandbox_id": sandbox.object_id,
                      "wall_s": time.time() - started, "network": "blocked", "cpu": 1, "memory_mib": 512}
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.lock:
                self.runs.append({k: v for k, v in record.items() if k != "results"})
            return results
        finally:
            sandbox.terminate()
