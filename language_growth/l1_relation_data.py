"""Pinned L1 relation-slot data; teacher-forced conditional diagnostic only.

No model, Torch, provider, Index or heldout access. Calibration remains
reserved. The existing free-running validation adapter is unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .l1_world_training import (
    CALIBRATION_RECORDS_SHA256, FIT_RECORDS_SHA256, PARTITION_MANIFEST_SHA256,
    L1WorldTrainingData, _manifest_bytes, _record_bytes, load_l1_world_training_data,
)
from .multirecord_data import MULTIWORLD_VALIDATION_SPLIT
from .multirecord_eval import MULTIWORLD_FIXTURE_DIR, _PINS, _canonical
from .training_records import L1_GENERATOR_SOURCE, L1_TRAINING_ALLOWLIST


BASE_COMMIT = "32d975159c64216193c47478e102d17c8d7e5658"
VALIDATION_TASKS_SHA256 = "da04a584590880c0b525327f6b0203414f9774cf9275c811d39aa7a0bf0df423"
RELATION_QID = "slot-004-relation"
MAX_INPUT_TOKENS = 2048
VOCAB_SHA256 = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
DEFAULT_VOCAB = Path("C:/Users/apumu/research/CHLabs/ssdm-data/hf/models--fla-hub--rwkv7-0.1B-g1/"
                     "snapshots/7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8/rwkv_vocab_v20230424.txt")
CONDITIONAL_SCOPE = "teacher_forced_relation_only_conditional_diagnostic_not_free_running"


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _verify_worlds(data):
    """Recheck ALL complete partitions, including before every Item conversion."""
    checks = ((list(data.fit_records), FIT_RECORDS_SHA256),
              (list(data.calibration_records), CALIBRATION_RECORDS_SHA256),
              (list(data.validation_tasks), VALIDATION_TASKS_SHA256))
    if any(hashlib.sha256(_record_bytes(rows)).hexdigest() != expected for rows, expected in checks):
        raise ValueError("complete pinned 24/8/32 world partitions required before relation filtering")
    if hashlib.sha256(_manifest_bytes(data.partition_manifest)).hexdigest() != PARTITION_MANIFEST_SHA256:
        raise ValueError("pinned whole-world partition manifest changed")
    actual = (frozenset(r["metadata"]["world_id"] for r in data.fit_records),
              frozenset(r["metadata"]["world_id"] for r in data.calibration_records),
              frozenset(t["world_id"] for t in data.validation_tasks))
    if (actual != (data.fit_world_ids, data.calibration_world_ids, data.validation_world_ids)
            or tuple(map(len, actual)) != (24, 8, 32)
            or any(actual[a] & actual[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
        raise ValueError("fit/calibration/validation world partition crossing")


def _relation_records(records):
    return tuple(deepcopy(r) for r in records if set(r["request"]["questions"]) == {RELATION_QID})


def _validation_view(root, world_data):
    """Read exact raw evaluation bytes, never pass them to the training exporter."""
    raw = (root / f"{MULTIWORLD_VALIDATION_SPLIT}.jsonl").read_bytes()
    manifest_bytes = (root / f"{MULTIWORLD_VALIDATION_SPLIT}-manifest.json").read_bytes()
    if (hashlib.sha256(raw).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest()) != _PINS[MULTIWORLD_VALIDATION_SPLIT]:
        raise ValueError("pinned validation source or manifest SHA-256 mismatch")
    tasks, manifest = [json.loads(line) for line in raw.splitlines()], json.loads(manifest_bytes)
    if ({t["task_id"] for t in tasks} != {t["task_id"] for t in world_data.validation_tasks}
            or len(tasks) != 1280 or len(manifest["worlds"]) != 32):
        raise ValueError("complete pinned validation worlds required")
    records, targets = [], {}
    for task in tasks:
        gold = task["evaluator_only"]
        if gold["kind"] != "compare":
            continue
        if [t["slot"] for t in gold["targets"]] != ["operation", "left", "right", "relation"]:
            raise ValueError("comparison slot schedule mismatch")
        prior = []
        for request, target in zip(task["model_inputs"][:3], gold["targets"][:3]):
            qid, question = next(iter(request["questions"].items()))
            if not qid.endswith("-" + target["slot"]):
                raise ValueError("earlier slot identity mismatch")
            prior.append({**target, "description": question["criteria"][target["option_id"]]})
        request = deepcopy(task["model_inputs"][3])
        state = json.loads(request["state"])
        if state["proposed_slot_choices"] != []:
            raise ValueError("authored validation prefixes must be empty")
        state["proposed_slot_choices"] = prior
        request["state"] = _canonical(state)
        records.append({"id": task["task_id"], "request": request, "metadata": {
            "world_id": task["world_id"], "development_split": MULTIWORLD_VALIDATION_SPLIT,
            "scope": CONDITIONAL_SCOPE, "teacher_forced": True, "evaluation_only": True,
            "prior_slots": ["operation", "left", "right"],
            "source_sample_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][0],
            "source_manifest_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][1],
        }})
        targets[f"{task['task_id']}/{RELATION_QID}"] = gold["targets"][3]["option_id"]
    return tuple(records), targets


def assert_content_disjoint(partitions):
    """Use the trainer's normalized state/questions identity, ignoring labels/IDs."""
    from train.cohort import normalized_content_hash

    contents = {name: {normalized_content_hash(r["request"]["state"], r["request"]["questions"])
                       for r in rows} for name, rows in partitions.items()}
    names = list(contents)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if contents[left] & contents[right]:
                raise ValueError(f"normalized-content overlap: {left}/{right}")
    return {name: {"unique_contents": len(values), "sorted_content_sha256": _sha(sorted(values))}
            for name, values in contents.items()}


def check_token_lengths(lengths):
    """Check the entire supplied census; never truncate or drop long rows."""
    if not lengths:
        raise ValueError("complete unique nonempty token-length census required")
    for row_id, row in lengths.items():
        if row["max_call_tokens"] > MAX_INPUT_TOKENS:
            raise ValueError(f"{row_id}: {row['max_call_tokens']} tokens exceeds 2048 (no truncation)")
    return lengths


def _item_lengths(items):
    lengths = {item.id: {"prefix_tokens": len(item.row.ids), "max_call_tokens": max([
        len(item.row.ids), *(len(item.row.ids) + len(c) for c in item.row.continuations),
        *(len(s) for s in item.row.option_seqs)])} for item in items}
    if len(lengths) != len(items):
        raise ValueError("duplicate Item identity in token census")
    return check_token_lengths(lengths)


def encoded_inputs(record, encode, *, template="jrt-rwkv-g1-v1"):
    """Label-free stock prompt inputs, suitable for R0 token/identity proofs.

    This is not a feature cache or LayoutRow replacement. Runtime extraction
    must build real LayoutRows with its active template and recheck lengths.
    """
    from decide.prompt import TEMPLATES, build_prompt_segments, continuation_segments
    from decide.schema import canonical_options

    tpl = TEMPLATES[template] if isinstance(template, str) else template
    request = record["request"]
    if len(request["questions"]) != 1:
        raise ValueError("exactly one slot question required")
    qid, question = next(iter(request["questions"].items()))
    encode_segments = lambda segments: [token for text, trusted in segments for token in encode(text, trusted)]
    return {"qid": qid, "ids": encode_segments(build_prompt_segments(tpl, request, qid)),
            "option_ids": [oid for oid, _ in canonical_options(question)],
            "continuations": [encode_segments(continuation_segments(tpl, oid))
                              for oid, _ in canonical_options(question)],
            "template": tpl.identity(), "encoder": encode.identity()}


@dataclass(frozen=True)
class L1RelationData:
    """Only fit Items may optimize; validation labels are a separate sidecar."""

    world_data: L1WorldTrainingData
    fit_records: tuple
    calibration_records: tuple
    teacher_forced_validation_records: tuple
    teacher_forced_validation_targets: dict
    manifest: dict

    def _verify(self):
        _verify_worlds(self.world_data)
        for name, records in (("fit", self.fit_records), ("calibration", self.calibration_records),
                              ("teacher_forced_validation", self.teacher_forced_validation_records)):
            if hashlib.sha256(_record_bytes(list(records))).hexdigest() != self.manifest["partitions"][name]["rows_sha256"]:
                raise ValueError("relation records changed after pinned selection")
        if _sha(self.teacher_forced_validation_targets) != self.manifest["validation_targets_sha256"]:
            raise ValueError("validation relation target sidecar changed")

    def fit_items(self, encode, *, guard, layout="ordered", template="jrt-rwkv-g1-v1"):
        """Deferred L1r1 runtime API; R0 never imports the Torch-dependent consumer."""
        self._verify()
        # DO NOT move selection above this call: fit_items guards the full pin.
        items, worlds = self.world_data.fit_items(encode, guard=guard, layout=layout, template=template)
        _item_lengths(items)
        selected = [item for item in items if item.row.qid == RELATION_QID]
        if len(selected) != 576:
            raise ValueError("exactly 576 fit relation Items required")
        return selected, {item.id: worlds[item.id] for item in selected}

    def teacher_forced_validation_items(self, encode, *, layout="ordered", template="jrt-rwkv-g1-v1"):
        """Conditional evaluation only; never route these through fit_items."""
        self._verify()
        from train.data import items_from_records
        from train.guard import SplitGuard

        records = [{**r, "labels": {RELATION_QID: self.teacher_forced_validation_targets[f"{r['id']}/{RELATION_QID}"]}}
                   for r in self.teacher_forced_validation_records]
        items = items_from_records(records, encode, layout=layout, template=template,
                                   guard=SplitGuard.synthetic_only(CONDITIONAL_SCOPE))
        _item_lengths(items)
        return items, {f"{r['id']}/{RELATION_QID}": r["metadata"]["world_id"] for r in records}

    def token_length_report(self, encode, *, layout="ordered", template="jrt-rwkv-g1-v1"):
        """Vocabulary-only stock-prompt census, with runtime LayoutRow checks deferred."""
        self._verify()
        if layout != "ordered":
            raise ValueError("R0 stock prompt token census requires ordered layout")
        report = {}
        for name, records in (("full_fit", self.world_data.fit_records),
                              ("full_reserved_calibration", self.world_data.calibration_records),
                              ("teacher_forced_validation", self.teacher_forced_validation_records)):
            lengths = {}
            for record in records:
                inputs = encoded_inputs(record, encode, template=template)
                row_id = f"{record['id']}/{inputs['qid']}"
                if row_id in lengths:
                    raise ValueError("duplicate row identity in token census")
                lengths[row_id] = {"prefix_tokens": len(inputs["ids"]),
                    "max_call_tokens": len(inputs["ids"]) + max(map(len, inputs["continuations"]))}
            report[name] = check_token_lengths(lengths)
        return {"limit": MAX_INPUT_TOKENS, "truncation": False, "encoder": encode.identity(),
                "layout": layout, "template": template, "rows": report,
                "tokenizer_files_only": True, "runtime_layout_acceptance": "deferred_guarded_runner_preflight",
                "summary": {name: {"row_count": len(rows),
                    "min_prefix_tokens": min(r["prefix_tokens"] for r in rows.values()),
                    "max_prefix_tokens": max(r["prefix_tokens"] for r in rows.values()),
                    "max_call_tokens": max(r["max_call_tokens"] for r in rows.values())}
                    for name, rows in report.items()}}


def load_l1_relation_data(*, fixture_dir=None):
    root = MULTIWORLD_FIXTURE_DIR if fixture_dir is None else Path(fixture_dir)
    worlds = load_l1_world_training_data(fixture_dir=root)
    _verify_worlds(worlds)  # Full 24/8/32 verification precedes ANY slot filtering.
    fit, calibration = _relation_records(worlds.fit_records), _relation_records(worlds.calibration_records)
    validation, targets = _validation_view(root, worlds)
    partitions = {"fit": fit, "calibration": calibration, "teacher_forced_validation": validation}
    if tuple(map(len, partitions.values())) != (576, 192, 768):
        raise ValueError("exact 576/192/768 relation rows required")
    contents = assert_content_disjoint({"full_fit": worlds.fit_records,
                                      "full_calibration": worlds.calibration_records,
                                      "teacher_forced_validation": validation})
    summaries = {}
    for name, rows in partitions.items():
        counts = Counter(r["metadata"]["world_id"] for r in rows)
        if set(counts.values()) != {24}:
            raise ValueError("every complete world must contain 24 relation rows")
        summaries[name] = {"row_count": len(rows), "world_count": len(counts),
                           "world_rows": dict(sorted(counts.items())),
                           "rows_sha256": hashlib.sha256(_record_bytes(list(rows))).hexdigest()}
    manifest = {"schema": "ssdm.l1_relation_data.v1", "base_commit": BASE_COMMIT,
                "scope": CONDITIONAL_SCOPE, "fit_only_optimizer": True, "calibration_reserved": True,
                "full_partitions": {"fit_sha256": FIT_RECORDS_SHA256,
                    "calibration_sha256": CALIBRATION_RECORDS_SHA256,
                    "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
                    "validation_free_running_tasks_sha256": VALIDATION_TASKS_SHA256},
                "train_sources": L1_TRAINING_ALLOWLIST,
                "validation_source_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][0],
                "validation_manifest_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][1],
                "generator_source": L1_GENERATOR_SOURCE, "partitions": summaries,
                "validation_targets_sha256": _sha(targets), "normalized_content": contents,
                "token_limit": MAX_INPUT_TOKENS, "tokens_checked": False,
                "cost_usd": 0, "model_execution": False}
    return L1RelationData(worlds, fit, calibration, validation, targets, manifest)


class PinnedRWKVEncoder:
    """Vocabulary-only CPU tokenizer; does not read model config or weights."""

    def __init__(self, vocab_path=DEFAULT_VOCAB):
        path = Path(vocab_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != VOCAB_SHA256:
            raise ValueError("pinned RWKV vocabulary SHA-256 mismatch")
        from decide.rwkv_world_tokenizer import RWKVWorldTokenizer
        self.tokenizer = RWKVWorldTokenizer(path)

    def __call__(self, text, trusted):
        return self.tokenizer.encode(text)

    def identity(self):
        return {"kind": "rwkv-world-greedy", "vocab_sha256": VOCAB_SHA256,
                "control_tokens": False, "model_id": "fla-hub/rwkv7-0.1B-g1",
                "snapshot_revision": "7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh receipt file")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()
    data = load_l1_relation_data()
    tokens = data.token_length_report(PinnedRWKVEncoder(args.vocab))
    result = {"manifest": data.manifest, "token_lengths": tokens}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="ascii") as stream:
        stream.write(_canonical(result) + "\n")


if __name__ == "__main__":
    main()
