"""Preparation checks only; these tests never load a checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from language_growth import l0_qwen_cpu_probe as probe


class ProbePreparationTests(unittest.TestCase):
    def test_clock_failure_after_spawn_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "new-run" / "raw.jsonl"
            child = MagicMock()
            child.poll.return_value = None
            with (patch.object(probe, "_check_dependencies", return_value={}),
                  patch("psutil.virtual_memory", return_value=SimpleNamespace(available=9 * 1024**3)),
                  patch.object(probe.subprocess, "Popen", return_value=child),
                  patch.object(probe.time, "monotonic", side_effect=RuntimeError("clock failed"))):
                with self.assertRaisesRegex(RuntimeError, "clock failed"):
                    probe._parent(raw)
            child.kill.assert_called_once_with()
            child.wait.assert_called_once_with()

    def test_one_pinned_validation_comparison(self):
        task = probe._task()
        self.assertEqual(task["task_id"], "synthetic-development_validation-271828-2")
        self.assertEqual(task["training_only"]["kind"], "compare")
        self.assertEqual(task["training_only"]["choice_format"], "production")
        self.assertEqual(task["development_split"], "development_validation")
        self.assertEqual(len(probe._sha_task(task)), 64)

    def test_child_environment_forces_offline_cpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "0"}):
            env = probe._offline_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], "2")

    def test_each_provider_request_and_failure_flushes_raw_jsonl(self):
        class FailingProvider:
            def decide(self, request):
                raise ValueError("scripted failure")

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            provider = probe._LoggedDecider(FailingProvider(), raw)
            with self.assertRaisesRegex(ValueError, "scripted failure"):
                provider.decide({"state": "initial synthetic state"})
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows],
                         ["provider_request", "provider_exception"])
        self.assertEqual(rows[0]["request"]["state"], "initial synthetic state")
        self.assertEqual(rows[1]["exception_type"], "ValueError")

    def test_unmerged_paired_dependency_is_explicit(self):
        with patch.object(probe, "ROOT", Path("/a/nonexistent/ssdm-root")):
            with self.assertRaisesRegex(RuntimeError, "PR #37 dependency missing"):
                probe._check_dependencies()

    def test_launcher_stops_only_its_child_on_timeout_or_low_memory(self):
        for available_after_start, elapsed, expected_code, reason in (
            (9 * 1024**3, 301, 124, "timeout"),
            (2 * 1024**3, 1, 125, "memory_guard"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                raw = Path(directory) / "new-run" / "raw.jsonl"
                child = MagicMock()
                child.poll.return_value = None
                memory = [SimpleNamespace(available=9 * 1024**3),
                          SimpleNamespace(available=available_after_start)]
                with (patch.object(probe, "_check_dependencies", return_value={}),
                      patch("psutil.virtual_memory", side_effect=memory),
                      patch.object(probe.subprocess, "Popen", return_value=child),
                      patch.object(probe.time, "monotonic", side_effect=[0, elapsed]),
                      patch.object(probe.time, "sleep")):
                    code = probe._parent(raw)
                self.assertEqual(code, expected_code)
                child.kill.assert_called_once_with()
                child.wait.assert_called_once_with()
                rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(rows[-1]["event"], "launcher_stopped")
                self.assertEqual(rows[-1]["reason"], reason)

    def test_launcher_kills_owned_child_if_memory_monitor_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "new-run" / "raw.jsonl"
            child = MagicMock()
            child.poll.return_value = None
            with (patch.object(probe, "_check_dependencies", return_value={}),
                  patch("psutil.virtual_memory", side_effect=[
                      SimpleNamespace(available=9 * 1024**3), RuntimeError("monitor failed")]),
                  patch.object(probe.subprocess, "Popen", return_value=child),
                  patch.object(probe.time, "monotonic", side_effect=[0, 1])):
                with self.assertRaisesRegex(RuntimeError, "monitor failed"):
                    probe._parent(raw)
            child.kill.assert_called_once_with()
            child.wait.assert_called_once_with()

    def test_child_rechecks_ram_before_torch_or_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            with (patch.dict(os.environ, probe._offline_env()),
                  patch("psutil.virtual_memory", return_value=SimpleNamespace(available=7 * 1024**3)),
                  patch.object(probe, "_check_dependencies") as dependencies):
                with self.assertRaisesRegex(RuntimeError, "before Torch"):
                    probe._child(raw)
            dependencies.assert_not_called()


if __name__ == "__main__":
    unittest.main()
