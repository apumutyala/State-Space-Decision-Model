"""Fit-only L1 cell; data readiness never grants permission to load a model.

The A order is a byte-pinned host attestation, like the Stage 1 qualification.
The independent order pin must come from A's reviewed handoff, not from hashing
an untrusted order in the launching invocation. ``ordered_by`` and ``order_ref``
identify that handoff; these strings do not authenticate its author.
This runner checks its bindings, not the identity of its author or live billing.
Provider job deadlines and billing caps remain external admission requirements.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

PLAN_SHA256 = "9f420d057a7cb3a00703793ac1f1f1a65cf5ae7ea5d9a6d8d701db985101fb8f"
ORDER_SCHEMA = "ssdm.agent_c.l1_fit_order.v1"
SOURCES = ("pinned_multirecord_fit", "balanced_exactdifference_fit")


class FitCellError(ValueError):
    pass


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FitCellError("duplicate JSON member")
        result[key] = value
    return result


def read_json(path: Path, expected: str) -> dict:
    if not _hex(expected, 64):
        raise FitCellError("independent lowercase SHA-256 pin required")
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise FitCellError("artifact byte pin mismatch")
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(FitCellError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise FitCellError("JSON object required")
    return value


def _hex(value, size):
    return isinstance(value, str) and len(value) == size and all(c in "0123456789abcdef" for c in value)


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise FitCellError(f"unsupported {name} schema")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class FitInputs:
    base_fit: Path
    base_census: Path
    base_plan: Path
    supplement: Path
    supplement_census: Path
    combined_plan: Path


@dataclass(frozen=True)
class ValidatedFit:
    plan: dict
    records: dict[str, list]
    census: dict[str, list]
    summaries: dict[str, dict]
    input_directories: tuple[Path, ...] = ()


def validate_fit(inputs: FitInputs) -> ValidatedFit:
    """Validate all ready data without order/guard files, Torch, or weights."""
    from . import l1_combined_difference_preflight as p

    plan = read_json(inputs.combined_plan, PLAN_SHA256)
    base = read_json(inputs.base_plan, p.BASE_PLAN_SHA256)
    if p.preflight_base(inputs.base_fit, inputs.base_census, diagnostic_worlds=6) != base:
        raise FitCellError("base preflight replay mismatch")
    base_bytes = p._read_pinned(inputs.base_fit, p.BASE_FIT_PIN)
    base_census = p._read_pinned(inputs.base_census, p.BASE_CENSUS_PIN)
    base_rows = p._records(base_bytes["training_records.jsonl"])
    base_summary = p._json(base_census["summary.json"])
    supplement, rows = p._load_supplement(inputs.supplement, inputs.supplement_census, base_summary)
    replay = p._combine(base, supplement, base_rows, rows, base_summary)
    replay["pins"] = {"base_plan_sha256": p.BASE_PLAN_SHA256, "base_fit": base["pins"],
                      "difference_supplement": supplement["pins"],
                      "combined_preflight_source_sha256": sha(Path(p.__file__).read_bytes())}
    if replay != plan or plan.get("training_authorized") is not False:
        raise FitCellError("combined preflight replay mismatch")
    supplemental_census = p._read_pinned(inputs.supplement_census, p.CENSUS_PIN)
    return ValidatedFit(plan, dict(zip(SOURCES, (base_rows, rows))),
                        {SOURCES[0]: [p._json(line) for line in base_census["row_counts.jsonl"].splitlines() if line],
                         SOURCES[1]: [p._json(line) for line in supplemental_census["row_counts.jsonl"].splitlines() if line]},
                        {SOURCES[0]: base_summary, SOURCES[1]: p._json(supplemental_census["summary.json"])},
                        (inputs.base_fit, inputs.base_census, inputs.base_plan.parent,
                         inputs.supplement, inputs.supplement_census))


def blockers(*, execute=False, order=None, order_sha256=None, report=None,
             qualification=None, checkpoint=None, snapshot=None, guard=None, tokenizer=None) -> list[str]:
    missing = [name for name, value in (("A order", order), ("A order byte pin", order_sha256),
               ("Stage 1 report", report), ("Stage 1 qualification", qualification),
               ("Stage 1 checkpoint", checkpoint), ("parent snapshot", snapshot),
               ("hash-only guard", guard), ("pinned tokenizer", tokenizer)) if not value]
    if not execute:
        missing.insert(0, "preflight only; execution was not requested")
    return missing


def _proof_is_ancestor(proof: str, current: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", proof, current],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True).returncode == 0


def validate_order(order: dict, *, source_commit: str, host: str, proof_is_ancestor=None) -> None:
    """Exact v1 host-attestation schema; no implicit defaults or synthetic pins."""
    _exact(order, {"schema", "ordered_by", "order_ref", "plan_sha256", "source_commit", "parent", "guard", "recipe", "resources", "spend"}, "A order")
    if (order["schema"] != ORDER_SCHEMA or order["plan_sha256"] != PLAN_SHA256
            or order["ordered_by"] != "agent-a" or not isinstance(order["order_ref"], str)
            or not order["order_ref"].strip()
            or not _hex(order["source_commit"], 40) or order["source_commit"] != source_commit):
        raise FitCellError("A order plan/source binding mismatch")
    _exact(order["parent"], {"report_sha256", "qualification_sha256", "checkpoint_sha256"}, "parent")
    _exact(order["guard"], {"artifact_sha256", "fingerprint_sha256"}, "guard")
    if any(not _hex(v, 64) for group in (order["parent"], order["guard"]) for v in group.values()):
        raise FitCellError("parent/guard pins required")
    recipe = order["recipe"]
    _exact(recipe, {"config_hash", "max_steps", "ckpt_every", "diagnostic_batch_size", "max_seconds"}, "recipe")
    if (not _hex(recipe["config_hash"], 16)
            or any(type(recipe[k]) is not int or recipe[k] < 1 for k in ("max_steps", "diagnostic_batch_size"))
            or type(recipe["ckpt_every"]) is not int or recipe["ckpt_every"] < 0
            or not _positive(recipe["max_seconds"])):
        raise FitCellError("invalid bounded recipe")
    resource = order["resources"]
    _exact(resource, {"host", "device", "dtype", "isolated", "max_peak_gb", "min_free_gb"}, "resources")
    device = resource["device"]
    if (resource["host"] != host or resource["isolated"] is not True
            or not isinstance(device, str) or not (device == "cuda" or
                device.startswith("cuda:") and device[5:].isdigit())
            or resource["dtype"] not in ("float32", "bfloat16")
            or not _positive(resource["max_peak_gb"]) or not _positive(resource["min_free_gb"])):
        raise FitCellError("isolated ordered GPU resource binding required")
    spend = order["spend"]
    _exact(spend, {"authorized", "merged_proof_commit", "acknowledged_issue", "max_cost_usd"}, "spend")
    if (spend["authorized"] is not True or not _hex(spend["merged_proof_commit"], 40)
            or not isinstance(spend["acknowledged_issue"], str) or not spend["acknowledged_issue"].strip()
            or not _positive(spend["max_cost_usd"])):
        raise FitCellError("merged proof and acknowledged spend order required")
    if not (proof_is_ancestor or _proof_is_ancestor)(spend["merged_proof_commit"], source_commit):
        raise FitCellError("merged proof must be the current source commit or an ancestor")


def prepare_items(data: ValidatedFit, encoder, guard, *, items_factory=None):
    """Guard every row before conversion; reject any token or partition drift."""
    if items_factory is None:
        from train.data import items_from_records
        items_factory = items_from_records
    for rows in data.records.values():
        for row in rows:
            guard.check(row)
    train, diagnostic, sidecar = [], [], {}
    all_worlds, all_tasks = set(), set()
    for source in SOURCES:
        rows, census = data.records[source], data.census[source]
        source_plan = data.plan["sources"][source]["source"]
        tw, dw = set(source_plan["training_worlds"]), set(source_plan["diagnostic_worlds"])
        worlds = {r["group_id"] for r in rows}
        tasks = {r["id"] for r in rows}
        if ((len(tw), len(dw)) != (18, 6) or tw & dw or tw | dw != worlds
                or all_worlds & worlds or all_tasks & tasks):
            raise FitCellError("whole-world partition/source identity mismatch")
        all_worlds |= worlds
        all_tasks |= tasks
        items = items_factory(rows, encoder, guard=guard, layout="ordered", template="jrt-rwkv-g1-v1")
        if len(items) != len(rows) or len(census) != len(rows):
            raise FitCellError("Item conversion dropped or added rows")
        expected_encoder = data.summaries[source]["encoder_identity"]
        if encoder.identity() != expected_encoder:
            raise FitCellError("encoder differs from exact census")
        counts = [0, 0]
        for line, (record, token, item) in enumerate(zip(rows, census, items, strict=True), 1):
            qids = list(record["request"]["questions"])
            if len(qids) != 1:
                raise FitCellError("one question per fit record required")
            qid = qids[0]
            identity = f"{record['id']}/{qid}"
            row = item.row
            lengths = [len(c) for c in row.continuations]
            if not lengths:
                raise FitCellError("missing option continuations")
            actual = {"record_line": line, "record_id": record["id"], "slot": qid,
                      "prompt_tokens": len(row.ids), "options": row.n_options,
                      "continuation_tokens": lengths, "max_continuation_tokens": max(lengths),
                      "longest_prompt_plus_continuation_tokens": len(row.ids) + max(lengths),
                      "state_tokens": row.evidence_len}
            if (item.id != identity or identity in sidecar or row.qid != qid
                    or any(token.get(k) != v for k, v in actual.items())
                    or not 1 <= actual["longest_prompt_plus_continuation_tokens"] <= 2048
                    or not 1 <= row.n_options <= 8 or not 1 <= max(lengths) <= 16):
                raise FitCellError(f"exact census/Item mismatch: {source} row {line}")
            if record.get("metadata", {}).get("world_id") != record["group_id"]:
                raise FitCellError("record world sidecar mismatch")
            sidecar[identity] = {"source": source, "world": record["group_id"],
                                 "task": record["id"], "slot": record["metadata"]["slot"]}
            bucket = 0 if record["group_id"] in tw else 1
            (train if bucket == 0 else diagnostic).append(item)
            counts[bucket] += 1
        if counts != [source_plan["training_rows"], source_plan["diagnostic_rows"]]:
            raise FitCellError("source Item partition count mismatch")
    if (len(train), len(diagnostic), len(sidecar), len(all_worlds)) != (3600, 1200, 4800, 48):
        raise FitCellError("combined fit must preserve all 3600/1200 Items and 48 worlds")
    return train, diagnostic, sidecar


class Runtime:
    """Production imports occur only inside the explicitly gated execution path."""
    def inspect(self, **kwargs):
        from .qualified_parent import inspect_qualified_parent
        return inspect_qualified_parent(**kwargs)

    proof_is_ancestor = staticmethod(_proof_is_ancestor)

    def config(self, spec):
        from .rwkv7_m2_loader import validate_reconstruction_metadata
        return validate_reconstruction_metadata(spec.qualification, spec.pin)

    def guard(self, path, **kwargs):
        from .trainable_parent import load_pinned_split_guard
        return load_pinned_split_guard(path, **kwargs)

    def encoder(self, directory, summary):
        for name, expected in summary["tokenizer_sha256"].items():
            if sha((Path(directory) / name).read_bytes()) != expected:
                raise FitCellError("local tokenizer file pin mismatch")
        from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
        from train.run_cell import RwkvSegmentEncoder
        tok = Rwkv7Tokenizer.from_pretrained(str(directory), local_files_only=True)
        return RwkvSegmentEncoder(tok, set(tok.all_special_ids))

    def load(self, checkpoint, **kwargs):
        from .trainable_parent import load_qualified_trainable_parent
        return load_qualified_trainable_parent(checkpoint, **kwargs)

    def trainer(self, parent, **kwargs):
        from .trainable_parent import build_prepared_language_trainer
        return build_prepared_language_trainer(parent, **kwargs)

    def resources(self, resource):
        import torch
        if not torch.cuda.is_available():
            raise FitCellError("ordered CUDA resources unavailable")
        if resource["device"] != "cuda":
            torch.cuda.set_device(resource["device"])
        free, _total = torch.cuda.mem_get_info()
        if free / 1024**3 < resource["min_free_gb"]:
            raise FitCellError("ordered minimum free GPU memory unavailable")
        torch.cuda.reset_peak_memory_stats()

    def peak(self):
        import torch
        return torch.cuda.max_memory_allocated() / 1024**3

    def finite(self, parent):
        import torch
        names = tuple(sorted(n for n, p in parent.model.named_parameters() if p.requires_grad))
        if names != parent.trainable_parameter_names:
            raise FitCellError("trainable inventory changed")
        if any(not bool(torch.isfinite(p).all()) for p in parent.model.parameters() if p.requires_grad):
            raise FitCellError("nonfinite child trainables")

    def diagnostics(self, model, items, batch_size):
        from train.evalcard import predict
        from train.metrics import decision_metrics
        lp, labels, evidence, ks = predict(model, items, batch_size=batch_size)
        return decision_metrics([p[:k] for p, k in zip(lp.exp().tolist(), ks)], labels, evidence)


def _child_config(parent_config, recipe, out):
    if (parent_config.hash() != recipe["config_hash"] or parent_config.loss.kd
            or parent_config.loss.order_consistency_weight or parent_config.axis_loss_weight):
        raise FitCellError("ordered config is incompatible with plain fit Items")
    cfg = replace(parent_config, max_steps=recipe["max_steps"], ckpt_every=recipe["ckpt_every"],
                  ckpt_dir=str(out / "checkpoints"), eval_every=0, require_guard=True)
    if cfg.hash() != parent_config.hash() or any(getattr(cfg, k) != getattr(parent_config, k)
                                               for k in parent_config.EXECUTION):
        raise FitCellError("child config changed parent training/execution identity")
    return cfg


def _write_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_fit(data: ValidatedFit, *, order_path: Path, order_sha256: str,
                source_commit: str, host: str, report: Path, qualification: Path,
                checkpoint: Path, snapshot: Path, guard_path: Path, tokenizer: Path,
                out: Path, runtime=None, items_factory=None) -> dict:
    runtime = runtime or Runtime()
    order = read_json(order_path, order_sha256)
    validate_order(order, source_commit=source_commit, host=host,
                   proof_is_ancestor=runtime.proof_is_ancestor)
    if data.plan.get("training_authorized") is not False:
        raise FitCellError("expected immutable non-authorizing fit preflight")
    output = Path(out).resolve()
    protected_directories = (*data.input_directories, Path(snapshot), Path(tokenizer))
    protected_files = (Path(order_path), Path(report), Path(qualification), Path(checkpoint), Path(guard_path))
    if (any(output == Path(p).resolve() or Path(p).resolve() in output.parents for p in protected_directories)
            or any(output == p.resolve() for p in protected_files)):
        raise FitCellError("output must be separate from all input artifacts")
    parent_pins = order["parent"]
    kwargs = dict(report=report, report_sha256=parent_pins["report_sha256"],
                  qualification=qualification, qualification_sha256=parent_pins["qualification_sha256"])
    pinned_guard = runtime.guard(guard_path, artifact_sha256=order["guard"]["artifact_sha256"],
                                expected_fingerprint_sha256=order["guard"]["fingerprint_sha256"])
    for rows in data.records.values():
        for row in rows:
            pinned_guard.guard.check(row)
    spec = runtime.inspect(**kwargs)
    identity = spec.identity()
    if identity["artifact_sha256"] != parent_pins["checkpoint_sha256"]:
        raise FitCellError("Stage 1 checkpoint/order binding mismatch")
    cfg = _child_config(runtime.config(spec), order["recipe"], Path(out))
    summaries = [data.summaries[s] for s in SOURCES]
    if (summaries[0]["encoder_identity"] != summaries[1]["encoder_identity"]
            or summaries[0]["tokenizer_sha256"] != summaries[1]["tokenizer_sha256"]
            or spec.pin.revision["encoder"] != summaries[0]["encoder_identity"]
            or cfg.layout != "ordered" or spec.pin.revision["prompt_template"]["id"] != "jrt-rwkv-g1-v1"):
        raise FitCellError("Stage 1 tokenizer/layout differs from exact census")
    encoder = runtime.encoder(tokenizer, summaries[0])
    train, diagnostic, sidecar = prepare_items(data, encoder, pinned_guard.guard, items_factory=items_factory)
    runtime.resources(order["resources"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "ssdm.agent_c.l1_fit_receipt.v1", "status": "running",
               "plan_sha256": PLAN_SHA256, "order_sha256": order_sha256, "source_commit": source_commit,
               "ordered_by": order["ordered_by"], "order_ref": order["order_ref"],
               "stage1_parent": identity, "guard": order["guard"], "recipe": order["recipe"],
               "resources": order["resources"], "spend": order["spend"],
               "split": data.plan["split"], "item_sidecar_sha256": sha(canonical(sidecar)),
               "training_rows": len(train), "diagnostic_rows": len(diagnostic),
               "language_qualified": False, "promotion_eligible": False, "checkpoints": []}
    started = time.monotonic()
    try:
        _write_new(out / "item_sidecar.json", sidecar)
        receipt["item_sidecar_sha256"] = _file_sha(out / "item_sidecar.json")
        parent = runtime.load(checkpoint, **kwargs, snapshot=snapshot,
                              device=order["resources"]["device"], dtype=order["resources"]["dtype"])
        if parent.identity != identity or parent.encoder.identity() != encoder.identity():
            raise FitCellError("loaded parent differs from admitted metadata/encoder")
        trainer = runtime.trainer(parent, config=cfg, train_items=train, diagnostic_items=diagnostic,
                                  pinned_guard=pinned_guard)
        receipt.update(config=cfg.to_json(), config_hash=cfg.hash(),
                       lora_modules=list(parent.lora_modules),
                       trainable_parameter_names=list(parent.trainable_parameter_names))
        def budget():
            if (time.monotonic() - started > order["recipe"]["max_seconds"]
                    or runtime.peak() > order["resources"]["max_peak_gb"]):
                raise FitCellError("ordered time/memory budget exceeded")
        def save():
            runtime.finite(parent)
            path = trainer.save()
            receipt["checkpoints"].append({"path": str(path), "sha256": _file_sha(path),
                                            "step": trainer.step})
        budget()
        for _ in range(cfg.max_steps):
            budget()
            loss = trainer.train_step()
            if not math.isfinite(loss):
                raise FitCellError("nonfinite fit loss")
            budget()
            if cfg.ckpt_every and trainer.step % cfg.ckpt_every == 0 and trainer.step != cfg.max_steps:
                save()
        save()
        receipt["diagnostic"] = {"scope": "teacher-forced development_train fit worlds only",
                                 "metrics": runtime.diagnostics(parent.model, diagnostic,
                                                 order["recipe"]["diagnostic_batch_size"])}
        budget()
        receipt.update(status="completed_fit_only", steps=trainer.step,
                       lineage=trainer.lineage, seconds=time.monotonic() - started)
        _write_new(out / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started)
        _write_new(out / "failure.json", receipt)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-fit", "base-census", "base-plan", "supplement", "supplement-census", "combined-plan"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    for name in ("order", "report", "qualification", "checkpoint", "snapshot", "guard", "tokenizer", "out"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--order-sha256")
    a = p.parse_args(argv)
    data = validate_fit(FitInputs(a.base_fit, a.base_census, a.base_plan, a.supplement,
                                  a.supplement_census, a.combined_plan))
    missing = blockers(execute=a.execute, order=a.order, order_sha256=a.order_sha256, report=a.report,
                       qualification=a.qualification, checkpoint=a.checkpoint, snapshot=a.snapshot,
                       guard=a.guard, tokenizer=a.tokenizer)
    if a.execute and not a.out:
        missing.append("fresh output directory")
    if missing:
        print(json.dumps({"status": "data_ready_execution_blocked", "plan_sha256": PLAN_SHA256,
                          "training_rows": 3600, "diagnostic_rows": 1200,
                          "training_authorized": False, "blockers": missing}, sort_keys=True))
        return 2 if a.execute else 0
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True).strip():
        raise FitCellError("execution requires a clean committed source tree")
    execute_fit(data, order_path=a.order, order_sha256=a.order_sha256, source_commit=commit,
                host=platform.node(), report=a.report, qualification=a.qualification,
                checkpoint=a.checkpoint, snapshot=a.snapshot, guard_path=a.guard,
                tokenizer=a.tokenizer, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
