"""One synthetic CPU train/save/restore/paired-call composition check."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    out = Path(sys.argv[1])
    exporter_root = Path(sys.argv[2])
    if out.exists():
        raise ValueError("output directory must be fresh")
    out.mkdir(parents=True)
    try:
        import torch

        from decide.adapter import Decider
        from decide.calibration import OptionCountTemperature
        from heads import ByteEncoder, DecisionModel
        from heads.backbone import tiny_backbone
        from heads.decide_backend import DecisionModelBackend
        from train.data import items_from_records
        from train.guard import SplitGuard
        from train.trainer import Trainer, TrainConfig

        from language_growth.development_data import generate_development_tasks
        from language_growth.paired_comparison import run_paired_comparisons

        if torch.cuda.is_available():
            raise AssertionError("CUDA must be hidden for this CPU-only check")
        exporter_path = exporter_root / "language_growth" / "training_records.py"
        exporter_sha = subprocess.check_output(
            ["git", "-C", str(exporter_root), "rev-parse", "HEAD"], text=True).strip()
        if exporter_sha != "83eca37718812800bace6eab781457ee98e999d7":
            raise AssertionError("PR25 exporter checkout changed")
        spec = importlib.util.spec_from_file_location("language_growth.training_records_pr25", exporter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        train_task = generate_development_tasks(17, 3, choice_format="production")[-1]
        validation_task = generate_development_tasks(
            19, 3, choice_format="production", split="development_validation")[-1]
        if train_task["training_only"]["kind"] != "compare" or validation_task["training_only"]["kind"] != "compare":
            raise AssertionError("expected comparison tasks")
        records = module.export_training_records([train_task])
        encoder = ByteEncoder()
        guard = SplitGuard.synthetic_only("paired tiny-parent composition")
        items = items_from_records(records, encoder, guard=guard)
        cfg = TrainConfig(head="snapshot", lora_rank=2, batch_size=1, max_steps=1, seed=17)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = DecisionModel(tiny_backbone(seed=17, hidden=8, layers=1), "snapshot", seed=17)
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise AssertionError("model is not CPU-only")
            trainer = Trainer(model, cfg, items, guard=guard)
            before = {name: value.clone() for name, value in trainer.trainable_state().items()}
            history = trainer.train()
            if len(history) != 1 or not math.isfinite(history[0]["loss"]):
                raise AssertionError("one finite optimizer step required")
            changed = [name for name, value in trainer.trainable_state().items()
                       if not torch.equal(before[name], value)]
            if not any(name.startswith("head.") for name in changed) or not any(
                    name.endswith("lora_B") for name in changed):
                raise AssertionError("expected head and LoRA parameter updates")
            with tempfile.TemporaryDirectory(prefix="ssdm-paired-cpu-checkpoint-") as folder:
                checkpoint = trainer.save(folder)
                checkpoint_sha = _sha(checkpoint)
                restored = DecisionModel(tiny_backbone(seed=17, hidden=8, layers=1), "snapshot", seed=17)
                restored_trainer = Trainer(restored, cfg, items, guard=guard)
                restored_trainer.load(checkpoint)
                if not all(torch.equal(value, restored.state_dict()[name])
                           for name, value in model.state_dict().items()):
                    raise AssertionError("checkpoint restore differs")
                digest = hashlib.sha256()
                for name, tensor in sorted(restored.state_dict().items()):
                    digest.update(name.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                backend = DecisionModelBackend(restored, encoder,
                                               checkpoint={"state_sha256": digest.hexdigest()})
                calibration = OptionCountTemperature()
                parent = Decider(backend, calibration=calibration)
                report = run_paired_comparisons(
                    [validation_task], parent, model="tiny-paired-cpu-test",
                    parent_revision=backend.revision()["revision_id"],
                    expected_served_config={"calibration": calibration.identity(),
                        "policy_masking": "none (served covers every supplied option)"})
            if report["total"] != 1:
                raise AssertionError("paired denominator changed")
            pair = report["pairs"][0]
            summary = {
                "scope": "one random tiny CPU optimizer step and restored-parent paired-call composition; not capability",
                "paired_source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "pr25_exporter_source_sha": exporter_sha,
                "pr25_exporter_file_sha256": _sha(exporter_path),
                "checkpoint_sha256": checkpoint_sha,
                "restored_state_sha256": digest.hexdigest(),
                "train_slots": len(records),
                "train_items": len(items),
                "changed_trainable_parameters": changed,
                "train_history": history,
                "same_initial_state": pair["same_initial_state"],
                "direct": {key: pair["direct"][key] for key in
                           ("status", "reason", "provider_calls", "forward_calls", "correct")},
                "planner": {key: pair["planner"][key] for key in
                            ("status", "reason", "provider_calls", "forward_calls", "task_correct")},
            }
            _write(out / "raw-report.json", report)
            _write(out / "summary.json", summary)
            print(json.dumps(summary, sort_keys=True, default=str))
        finally:
            torch.set_num_threads(old_threads)
    except Exception as exc:
        _write(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
