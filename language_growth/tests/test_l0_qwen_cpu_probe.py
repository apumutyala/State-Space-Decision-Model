"""Preparation checks only; these tests never load a checkpoint."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from language_growth import l0_qwen_cpu_probe as probe
from language_growth.tests.test_paired_comparison import CONFIG, REVISION, EvidenceProvider


class ProbePreparationTests(unittest.TestCase):
    def test_isolated_root_launcher_help_starts_from_package_directory(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(probe.ROOT / "language_growth")
        for mode in ([], ["--child"]):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [sys.executable, "-I", str(probe.ROOT / "l0_qwen_cpu_probe.py"),
                     *mode, "--help"],
                    cwd=probe.ROOT / "language_growth", env=env,
                    capture_output=True, text=True, timeout=15, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--raw", result.stdout)
                self.assertIn("offline", result.stdout)

    def test_child_invocation_uses_isolated_root_runner_and_offline_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "new-run" / "raw.jsonl"
            child = MagicMock()
            child.poll.return_value = 0
            with (patch.object(probe, "_check_dependencies", return_value={}),
                  patch("psutil.virtual_memory", return_value=SimpleNamespace(available=9 * 1024**3)),
                  patch.object(probe.subprocess, "Popen", return_value=child) as popen):
                self.assertEqual(probe._parent(raw), 0)
            command = [sys.executable, "-I", str(probe.ROOT / "l0_qwen_cpu_probe.py"),
                       "--child", "--raw", str(raw)]
            self.assertEqual(popen.call_args.args, (command,))
            env = popen.call_args.kwargs["env"]
            for name, value in (("CUDA_VISIBLE_DEVICES", "-1"),
                                ("HF_HUB_OFFLINE", "1"),
                                ("TRANSFORMERS_OFFLINE", "1"),
                                ("HF_DATASETS_OFFLINE", "1"),
                                ("HF_HUB_DISABLE_TELEMETRY", "1")):
                self.assertEqual(env[name], value)
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["command"], command)
            self.assertEqual(rows[0]["tasks"], probe._manifest(probe._tasks()))
            self.assertEqual(rows[0]["total"], 2)
            self.assertEqual(rows[0]["timeout_seconds"], 300)
            self.assertEqual(rows[0]["min_start_available_bytes"], 8 * 1024**3)
            self.assertEqual(rows[0]["min_running_available_bytes"], 3 * 1024**3)

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

    def test_two_pinned_validation_comparisons(self):
        tasks = probe._tasks()
        self.assertEqual([task["task_id"] for task in tasks],
                         [f"synthetic-development_validation-271828-{i}" for i in (2, 5)])
        self.assertEqual(tuple(probe._sha_task(task) for task in tasks), probe.TASK_HASHES)
        for task in tasks:
            self.assertEqual(task["training_only"]["kind"], "compare")
            self.assertEqual(task["training_only"]["choice_format"], "production")
            self.assertEqual(task["development_split"], "development_validation")
        self.assertEqual([task["training_only"]["program"]["nodes"][0]["field"]
                          for task in tasks], ["warranty_months", "price"])

    def test_task_pin_rejects_content_drift_before_spawn(self):
        from language_growth.development_data import generate_development_tasks

        for index in (2, 5):
            generated = generate_development_tasks(271828, 6, choice_format="production",
                                                   split="development_validation")
            generated[index]["model_inputs"][0]["state"] += " "
            with (self.subTest(index=index),
                  patch("language_growth.development_data.generate_development_tasks", return_value=generated)):
                with self.assertRaisesRegex(RuntimeError, f"comparison {index} changed"):
                    probe._tasks()

    def test_real_paired_caller_preserves_two_tasks_and_raw_identities(self):
        tasks = probe._tasks()
        provider = EvidenceProvider()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            report = probe._run_cohort(tasks, provider, raw, revision=REVISION, served_config=CONFIG)
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"],
                          report["paired_correct"]), (2, 2, 2, 2))
        self.assertTrue(all(pair["same_initial_state"] is True for pair in report["pairs"]))
        self.assertEqual(len(provider.requests), 10)
        self.assertNotIn("training_only", json.dumps(provider.requests))
        task_reports = [row for row in rows if row["event"] == "task_report"]
        self.assertEqual([row["task_sha256"] for row in task_reports], list(probe.TASK_HASHES))
        first_report = next(i for i, row in enumerate(rows) if row["event"] == "task_report")
        second_start = next(i for i, row in enumerate(rows)
                            if row["event"] == "task_start" and row["task_id"] == tasks[1]["task_id"])
        self.assertLess(first_report, second_start)
        for task, digest in zip(tasks, probe.TASK_HASHES):
            events = [row for row in rows if row["event"].startswith("provider_")
                      and row["task_id"] == task["task_id"]]
            self.assertEqual(len(events), 10)
            self.assertTrue(all(row["task_sha256"] == digest for row in events))
        self.assertEqual(rows[-1]["event"], "paired_report")
        self.assertEqual(rows[-1]["tasks"], probe._manifest(tasks))

    def test_provider_failures_remain_in_both_denominators(self):
        for mode, counts in (("error_direct", (2, 0, 2, 0)),
                             ("error_planner", (2, 2, 0, 0))):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                raw = Path(directory) / "raw.jsonl"
                report = probe._run_cohort(probe._tasks(), EvidenceProvider(mode), raw,
                                           revision=REVISION, served_config=CONFIG)
                self.assertEqual(tuple(report[key] for key in
                                       ("total", "direct_correct", "planner_correct", "paired_correct")),
                                 counts)
                rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
                failures = [row for row in rows if row["event"] == "provider_exception"]
                self.assertEqual([row["task_sha256"] for row in failures], list(probe.TASK_HASHES))

    def test_direct_pre_provider_failures_keep_denominator_with_unknown_initial_identity(self):
        tasks = deepcopy(probe._tasks())
        for task in tasks:
            state = json.loads(task["model_inputs"][0]["state"])
            state["target_option_id"] = "r001"
            task["model_inputs"][0]["state"] = json.dumps(state)
        provider = EvidenceProvider()
        with tempfile.TemporaryDirectory() as directory:
            report = probe._run_cohort(tasks, provider, Path(directory) / "raw.jsonl",
                                      revision=REVISION, served_config=CONFIG)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["direct_correct"], 0)
        self.assertEqual(report["planner_correct"], 2)
        self.assertEqual(report["paired_correct"], 0)
        self.assertEqual(len(provider.requests), 8)
        self.assertTrue(all(pair["direct"]["provider_calls"] == 0 for pair in report["pairs"]))
        self.assertTrue(all(pair["same_initial_state"] is None for pair in report["pairs"]))

    def test_second_task_runner_failure_retains_first_pair(self):
        from language_growth.paired_comparison import run_paired_comparisons

        tasks = probe._tasks()
        def scripted(examples, *args, **kwargs):
            if examples[0]["task_id"] == tasks[1]["task_id"]:
                raise RuntimeError("second task interrupted")
            return run_paired_comparisons(examples, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            with patch("language_growth.paired_comparison.run_paired_comparisons", side_effect=scripted):
                with self.assertRaisesRegex(RuntimeError, "second task interrupted"):
                    probe._run_cohort(tasks, EvidenceProvider(), raw, revision=REVISION, served_config=CONFIG)
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        reports = [row for row in rows if row["event"] == "task_report"]
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["task_sha256"], probe.TASK_HASHES[0])
        self.assertEqual(reports[0]["report"]["total"], 1)
        self.assertEqual(rows[-1]["event"], "task_failure")
        self.assertEqual(rows[-1]["task_sha256"], probe.TASK_HASHES[1])
        self.assertNotIn("paired_report", [row["event"] for row in rows])

    def test_denominator_and_initial_state_checks_reject_corruption(self):
        tasks = probe._tasks()
        with tempfile.TemporaryDirectory() as directory:
            report = probe._run_cohort(tasks, EvidenceProvider(), Path(directory) / "raw.jsonl",
                                      revision=REVISION, served_config=CONFIG)
        corruptions = (
            lambda r: r.update(total=1),
            lambda r: r["pairs"].pop(),
            lambda r: r["pairs"].reverse(),
            lambda r: r.update(direct_correct=1),
            lambda r: r["pairs"][1].update(same_initial_state=False),
            lambda r: r["pairs"][1].update(planner_initial_state_sha256="changed"),
        )
        for mutate in corruptions:
            bad = deepcopy(report)
            mutate(bad)
            with self.assertRaisesRegex(RuntimeError, "paired"):
                probe._check_report(bad, tasks)

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
            provider = probe._LoggedDecider(FailingProvider(), raw, probe._tasks()[0])
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
