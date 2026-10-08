"""Generate a deterministic, synthetic multi-question Index training family for B-1.

The worlds describe fictional telemetry relays. Every record contains four
related relay readings, field-exact questions, cross-field choices and score
labels, and a multi-select represented by one ``noul`` question per relay.
No Index rows, public datasets, model weights, or external text are inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types
from typing import Any, Callable

from data import RECORD_SCHEMA

GENERATOR_NAME = "c_multirecord_synth_v1"
TASK_FAMILY = "synthetic_multirecord_sensor_decision"
FAMILY = "synthetic_multirecord_telemetry"
SOURCE = "synthetic_c_multirecord_v1"
LICENSE = "project (synthetic; generated locally, no third-party text)"
BUILD = "c-multirecord-synth-100k-v1"
DEFAULT_SEED = 20261008
DEFAULT_COUNT = 100_000
MAX_TOKENS = 2048
TEMPLATE = "jrt-rwkv-g1-v1"
LAYOUT = "symmetric"
FIELDS = ("sig", "bat", "drift", "profile")
SCORE_LEVELS = ["very low", "low", "moderate", "high", "very high"]
SCORE_CUTS = (-500, -375, -250, -125)
PROFILE_COUNT = 77
TOKENIZER_PINS = {
    "tokenizer.json": "1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf",
    "tokenizer_config.json": "2b5e075c2c16780061ae272c9f0b52ff620ddd8e87fdf7fa82fc425661667329",
}
TOKENIZER_REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
GENERATOR_SPEC = {
    "name": GENERATOR_NAME,
    "version": 1,
    "task_family": TASK_FAMILY,
    "targets": [TASK_FAMILY],
    "source_split": "synthetic",
    "record_contract": RECORD_SCHEMA,
    "label_method": "deterministic synthetic truth; independent recomputation before write",
    "external_sources": [],
}


class GenerationError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seed(seed: int, index: int, purpose: str) -> int:
    text = f"{GENERATOR_NAME}:{seed}:{index}:{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(text).digest(), "big")


def _priority(node: dict) -> int:
    return 3 * node["sig"] + node["bat"] - node["drift"]


def _priority_order(nodes: list[dict]) -> list[dict]:
    return sorted(nodes, key=lambda node: (-_priority(node), node["node_id"]))


def _id_style(index: int, node_index: int) -> str:
    return ("letters", "words", "opaque")[(index + node_index) % 3]


_ONSETS = ("ba", "be", "di", "du", "fa", "fi", "go", "gu", "la", "lu", "ni")
_ENDINGS = ("ka", "ke", "mi", "mo", "pu", "ri", "zu")
_WORDS = tuple(sorted(a + b for a in _ONSETS for b in _ENDINGS))


def _alpha(index: int) -> str:
    """Stable a..z, aa.. codes; the actual option position is assigned after lexical sorting."""
    chars = []
    n = index
    while True:
        chars.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
        if n < 0:
            return "".join(reversed(chars))


def option_ids(style: str, count: int, *, seed: int, record_index: int, slot: int) -> list[str]:
    if not 2 <= count <= 77:
        raise ValueError("choice option count must be in 2..77")
    if style == "letters":
        ids = [_alpha(i) for i in range(count)]
    elif style == "words":
        ids = list(_WORDS[:count])
    elif style == "opaque":
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        def base36(value: int) -> str:
            digits = ""
            while value:
                value, remainder = divmod(value, 36)
                digits = alphabet[remainder] + digits
            return digits or "0"
        # Compact opaque IDs keep K=77 prompts inside the pinned tokenizer limit.
        ids = ["x" + base36(i).rjust(2, "0") for i in range(count)]
    else:
        raise ValueError(f"unknown option-id style: {style}")
    ids.sort()
    if len(set(ids)) != count:
        raise GenerationError("option-id collision")
    return ids


def _world(seed: int, index: int) -> tuple[list[dict], list[str], dict]:
    rng = random.Random(_seed(seed, index, "world"))
    field_order = list(FIELDS)
    rng.shuffle(field_order)
    node_order = list(range(4))
    rng.shuffle(node_order)
    nodes = []
    for n in range(4):
        nodes.append({
            "node_id": f"N{n}",
            "sig": rng.randint(-110, -35),
            "bat": rng.randint(0, 100),
            "drift": rng.randint(0, 200),
            "profile": rng.randrange(PROFILE_COUNT),
        })
    priority = _priority_order(nodes)
    # The recorded multi-select target is always two distinct relays: this balances noul labels
    # while making the decision depend jointly on all three numerical fields.
    top_two = {node["node_id"] for node in priority[:2]}
    shape = index % 3
    ordered_nodes = [nodes[n] for n in node_order]
    if shape == 0:
        state = {"s": "table", "f": field_order,
                 "r": [[node["node_id"], *(node[field] for field in field_order)] for node in ordered_nodes],
                 "rule": "rank=3*sig+bat-drift; top2"}
    elif shape == 1:
        state = {"s": "map", "f": field_order,
                 "n": [node["node_id"] for node in ordered_nodes],
                 "v": {field: [node[field] for node in ordered_nodes] for field in field_order},
                 "rule": "rank=3*sig+bat-drift; top2"}
    else:
        state = {"s": "frame", "frames": [
                    {"n": node["node_id"], "v": [node[field] for field in field_order]}
                    for node in ordered_nodes],
                 "f": field_order,
                 "rule": "rank=3*sig+bat-drift; top2"}
    truth = {"nodes": nodes, "field_order": field_order, "node_order": node_order,
             "top_two": top_two, "priority": priority, "schema_shape": shape}
    return nodes, field_order, {"state": state, "truth": truth}


def _parse_state(state: dict) -> tuple[list[dict], list[str]]:
    shape = state["s"]
    if shape == "table":
        order = state["f"]
        rows = state["r"]
        nodes = [dict(zip(["node_id", *order], row, strict=True)) for row in rows]
    elif shape == "map":
        order = state["f"]
        node_ids = state["n"]
        nodes = [{"node_id": node_id, **{field: state["v"][field][i] for field in order}}
                 for i, node_id in enumerate(node_ids)]
    elif shape == "frame":
        order = state["f"]
        nodes = [dict(zip(["node_id", *order], [row["n"], *row["v"]], strict=True))
                 for row in state["frames"]]
    else:
        raise GenerationError("unknown state schema shape")
    return nodes, order


def _proposal(node: dict, *, seed: int, record_index: int, node_index: int,
              case_exact: bool) -> dict:
    proposal = dict(node)
    if case_exact:
        return proposal
    rng = random.Random(_seed(seed, record_index, f"corrupt:{node_index}"))
    field = FIELDS[rng.randrange(len(FIELDS))]
    if field == "sig":
        proposal[field] = node[field] + (1 if node[field] < -35 else -1)
    elif field == "bat":
        proposal[field] = node[field] + (1 if node[field] < 100 else -1)
    elif field == "drift":
        proposal[field] = node[field] + (1 if node[field] < 200 else -1)
    else:
        proposal[field] = (node[field] + 1) % PROFILE_COUNT
    return proposal


def make_record(seed: int, index: int) -> dict:
    if type(seed) is not int or seed < 0 or type(index) is not int or index < 0:
        raise ValueError("seed/index must be non-negative integers")
    nodes, field_order, built = _world(seed, index)
    truth = built["truth"]
    question_specs = []
    case_exact = index % 2 == 0
    corrupt_node = None if case_exact else ((index // 2) % 4)
    proposals = [_proposal(node, seed=seed, record_index=index, node_index=n,
                           case_exact=(n != corrupt_node)) for n, node in enumerate(nodes)]
    proposals_by_id = {proposal["node_id"]: proposal for proposal in proposals}
    candidate_tuples = ["|".join(str(proposals_by_id[node["node_id"]][field]) for field in field_order)
                        for node in (nodes[n] for n in truth["node_order"])]
    actual_tuples = ["|".join(str(node[field]) for field in field_order)
                     for node in (nodes[n] for n in truth["node_order"])]
    if (candidate_tuples == actual_tuples) != case_exact:
        raise GenerationError("case-exact proposal label differs from generated tuples")
    if case_exact:
        wording = random.Random(_seed(seed, index, "case-exact-wording")).choice((
            "Are all four proposed rows exact? packet node and field order: ",
            "Do these rows match every packet field? use packet order: ",
            "Case-exact? packet node order and field order: ",
        ))
    else:
        wording = random.Random(_seed(seed, index, "case-exact-wording")).choice((
            "Are these four proposals exact in all fields? packet order: ",
            "Check every proposed row against the packet order: ",
            "Do all four rows match exactly? node and field order: ",
        ))
    case_instructions = wording + " ; ".join(candidate_tuples)
    question_specs.append(("case_exact", {"type": "noul", "instructions": case_instructions},
                           "true" if case_exact else "false",
                           {"kind": "case_exact", "all_fields": list(FIELDS),
                            "candidate_tuples": candidate_tuples}))
    # Four binary candidate questions encode one multi-select set without changing the frozen wire schema.
    for node_index, node in enumerate(nodes):
        wording_rng = random.Random(_seed(seed, index, f"membership-wording:{node_index}"))
        instruction = wording_rng.choice((
            f"Top two: {node['node_id']}?",
            f"Select {node['node_id']} among top two?",
            f"Include {node['node_id']} in top two?",
        ))
        label = "true" if node["node_id"] in truth["top_two"] else "false"
        question_specs.append((f"multi:{node['node_id']}", {"type": "noul", "instructions": instruction}, label,
                               {"kind": "multi_select_member", "node_id": node["node_id"]}))
    # One compact high-cardinality schema choice per record keeps K=77 useful without
    # multiplying hundreds of long tuple options across all questions in the request.
    top = truth["priority"][0]
    profile_count = 2 + (index % 76)
    profile_style_cycle = index // 76
    profile_style = ("letters", "words", "opaque")[profile_style_cycle % 3]
    profile_cycle = profile_style_cycle // 3
    profile_position = profile_cycle % profile_count
    available_profiles = list(range(PROFILE_COUNT))
    distractors = [p for p in available_profiles if p != top["profile"]]
    selected = [top["profile"], *distractors[:profile_count - 1]]
    profile_ids = option_ids(profile_style, profile_count, seed=seed, record_index=index, slot=90)
    profile_desc_by_id = {}
    gold_id = profile_ids[profile_position]
    other_ids = [oid for i, oid in enumerate(profile_ids) if i != profile_position]
    other_desc = [code for code in selected if code != top["profile"]]
    profile_desc_by_id[gold_id] = top["profile"]
    for oid, code in zip(other_ids, other_desc, strict=True):
        profile_desc_by_id[oid] = code
    profile_wording = random.Random(_seed(seed, index, "profile-wording")).choice((
        "Top relay's profile number (0-76)?",
        "Which profile belongs to the top relay?",
        "Profile number for the highest rank?",
    ))
    question_specs.append(("top_profile", {"type": "choice", "instructions": profile_wording,
                                             "criteria": profile_desc_by_id}, gold_id,
                           {"kind": "top_profile", "node_id": top["node_id"],
                            "k": profile_count, "id_style": profile_style,
                            "gold_position": profile_position}))
    score = _priority(top)
    score_label = str(sum(score >= cut for cut in SCORE_CUTS))
    score_wording = random.Random(_seed(seed, index, "score-wording")).choice((
        f"What is the ordinal priority band for {top['node_id']} after combining signal, battery, and drift?",
        f"Score {top['node_id']}'s joint telemetry readiness from the ordered levels.",
        f"Using all three readings, which ordered priority level fits {top['node_id']}?",
    ))
    question_specs.append(("joint_score", {"type": "score", "instructions": score_wording,
                             "criteria": SCORE_LEVELS}, score_label,
                           {"kind": "joint_score", "node_id": top["node_id"]}))
    case_wording = "Are all four proposed rows exact in every field?"

    # Deterministically permute which semantic task receives each sorted question id.
    order = sorted(range(len(question_specs)),
                   key=lambda j: hashlib.sha256(f"c-question-order:{seed}:{index}:{j}".encode()).hexdigest())
    questions, labels, roles = {}, {}, {}
    for position, spec_index in enumerate(order):
        role, question, label, meta = question_specs[spec_index]
        qid = f"q{position:02d}_{role.split(':')[0]}"
        questions[qid] = question
        labels[qid] = label
        roles[qid] = {"role": role, **meta}
    record_id = f"m2:c_multirecord_synth_v1:{seed}:{index:06d}"
    case_qids = sorted(q for q, m in roles.items() if m["kind"] == "case_exact")
    multi_qids = sorted(q for q, m in roles.items() if m["kind"] == "multi_select_member")
    record = {
        "id": record_id,
        "schema": RECORD_SCHEMA,
        "source": SOURCE,
        "source_split": "synthetic",
        "source_id": f"c-multirecord-synth-v1:{seed}:{index:06d}",
        "license": LICENSE,
        "family": FAMILY,
        "state": built["state"],
        "questions": questions,
        "labels": labels,
        "untrusted_fields": [],
        "aug": {"generator": GENERATOR_NAME, "seed": seed, "record_index": index,
                "schema_shape": ("column_table_v1", "register_map_v1", "sample_frame_v1")[built["truth"]["schema_shape"]],
                "field_order": field_order,
                "case_exact": {"question_ids": case_qids, "fields": list(FIELDS), "all_required": True,
                                "exact_case": case_exact},
                "multi_select": {"encoding": "one_noul_per_candidate", "selected_count": 2,
                                 "question_ids": multi_qids},
                "question_roles": roles},
        "_probe": [],
    }
    verify_record(record, built["truth"])
    return record


def verify_record(record: dict, truth: dict | None = None) -> None:
    from data.augment import validate

    validate(record)
    nodes, field_order = _parse_state(record["state"])
    if truth is not None:
        expected = {node["node_id"]: node for node in truth["nodes"]}
        actual = {node["node_id"]: node for node in nodes}
        if actual != expected or field_order != truth["field_order"]:
            raise GenerationError("serialized state does not reproduce its generated world")
    priority = _priority_order(nodes)
    top_two = {node["node_id"] for node in priority[:2]}
    roles = record["aug"]["question_roles"]
    for qid, meta in roles.items():
        question = record["questions"][qid]
        label = record["labels"][qid]
        if meta["kind"] == "multi_select_member":
            if label != ("true" if meta["node_id"] in top_two else "false"):
                raise GenerationError("multi-select noul label differs from joint priority set")
        elif meta["kind"] == "joint_score":
            node = next(n for n in nodes if n["node_id"] == meta["node_id"])
            expected_score = str(sum(_priority(node) >= cut for cut in SCORE_CUTS))
            if label != expected_score:
                raise GenerationError("joint score label differs from signal/battery/drift calculation")
        elif meta["kind"] == "top_profile":
            if meta["node_id"] != priority[0]["node_id"] or question["criteria"][label] != priority[0]["profile"]:
                raise GenerationError("high-cardinality profile choice differs from the top joint score")
        elif meta["kind"] == "case_exact":
            actual_tuples = ["|".join(str(node[field]) for field in field_order) for node in nodes]
            actual_exact = meta["candidate_tuples"] == actual_tuples
            if label != ("true" if actual_exact else "false"):
                raise GenerationError("case-exact label must require every proposed record field")
        else:
            raise GenerationError("unknown question role")
    if (len(record["questions"]) != 7 or sum(m["kind"] == "case_exact" for m in roles.values()) != 1
            or sum(m["kind"] == "multi_select_member" for m in roles.values()) != 4):
        raise GenerationError("all-field multi-question case is incomplete")


class _RWKV7Encoder:
    """Minimal copy of the production tokenizer path; importing train.run_cell would import Torch."""
    def __init__(self, tokenizer):
        from decide.untrusted import ControlTokenError
        self.tokenizer = tokenizer
        self.special = set(tokenizer.all_special_ids) | {0}
        self.ControlTokenError = ControlTokenError

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if not trusted and self.special.intersection(ids):
            raise self.ControlTokenError("untrusted synthetic record produced a special token")
        return ids


def _load_encoder(tokenizer_dir: Path):
    os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    original_import = __import__
    def no_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("Torch import forbidden in the C synthetic census")
        return original_import(name, *args, **kwargs)
    import builtins
    builtins.__import__ = no_torch_import
    try:
        from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
        tokenizer = Rwkv7Tokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        encoder = _RWKV7Encoder(tokenizer)
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("Torch was imported by the tokenizer-only census")
        return tokenizer, encoder
    finally:
        builtins.__import__ = original_import


def _load_layout_without_package_initializers(repo: Path):
    """Load the production row builder without heads/__init__.py, which imports Torch."""
    if "torch" in sys.modules or "heads" in sys.modules or "decide" in sys.modules:
        raise RuntimeError("token census must start before importing decide/heads packages")
    for name in ("decide", "heads"):
        package = types.ModuleType(name)
        package.__path__ = [str(repo / name)]
        sys.modules[name] = package

    def load(name: str, relative: str):
        spec = importlib.util.spec_from_file_location(name, repo / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("decide.schema", "decide/schema.py")
    load("decide.provenance", "decide/provenance.py")
    load("decide.prompt", "decide/prompt.py")
    load("decide.untrusted", "decide/untrusted.py")
    return load("heads.layout", "heads/layout.py").build_row


def _open_gz_writer(path: Path):
    raw = path.open("xb")
    stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
    return raw, stream


def _histogram_summary(hist: list[int]) -> dict:
    total = sum(hist)
    if not total:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None,
                "bins_256": []}
    def q(p):
        target = max(1, int((total * p) + 0.999999))
        acc = 0
        for value, count in enumerate(hist):
            acc += count
            if acc >= target:
                return value
        raise AssertionError("histogram quantile overflow")
    used = [i for i, count in enumerate(hist) if count]
    bins = []
    for start in range(0, MAX_TOKENS + 1, 256):
        count = sum(hist[start:start + 256])
        if count:
            bins.append({"start": start, "end": min(MAX_TOKENS, start + 255), "count": count})
    return {"n": total, "min": used[0], "p50": q(.50), "p95": q(.95), "max": used[-1], "bins_256": bins}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(out: Path, *, tokenizer_dir: Path, seed: int = DEFAULT_SEED,
          count: int = DEFAULT_COUNT, pointer_out: Path | None = None,
          data_root: Path | None = None) -> dict:
    from data.augment import validate
    from harness.heldout import check_generator, check_record

    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("seed/count must be nonnegative/positive integers")
    check_generator(GENERATOR_SPEC)
    print("C multirecord: held-out generator spec passed", file=sys.stderr, flush=True)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing build: {out}")
    pointer_out = Path(pointer_out) if pointer_out is not None else None
    if pointer_out is not None and pointer_out.exists():
        raise FileExistsError(f"refusing to overwrite pointer: {pointer_out}")
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    family_dir = out / "families"
    train_dir.mkdir()
    family_dir.mkdir()
    records_path = train_dir / "records.jsonl.gz"
    manifest_path = train_dir / f"{BUILD}.manifest.jsonl.gz"
    content_path = train_dir / f"{BUILD}.content.jsonl.gz"
    census_path = out / "token_census.jsonl.gz"

    # Verify tokenizer files before loading the tokenizer implementation.
    tokenizer_dir = Path(tokenizer_dir)
    tok_pins = {name: _sha_file(tokenizer_dir / name) for name in TOKENIZER_PINS}
    if tok_pins != TOKENIZER_PINS:
        raise GenerationError("local RWKV7 tokenizer bytes differ from the pinned revision")
    repo = Path(__file__).resolve().parents[1]
    build_row = _load_layout_without_package_initializers(repo)
    from train.cohort import record_content_hash
    tokenizer, encoder = _load_encoder(tokenizer_dir)
    if "torch" in sys.modules:
        raise GenerationError("Torch import forbidden")
    print(f"C multirecord: tokenizer loaded; generating {count} records", file=sys.stderr, flush=True)

    files = []
    streams = []
    for path in (records_path, manifest_path, content_path, census_path):
        raw, stream = _open_gz_writer(path)
        files.append(raw)
        streams.append(stream)
    records_stream, manifest_stream, content_stream, census_stream = streams
    records_hash = hashlib.sha256()
    manifest_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    content_seen: set[str] = set()
    question_types: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    choice_option_counts: Counter[int] = Counter()
    id_styles: Counter[str] = Counter()
    label_positions: Counter[tuple[int, str, int]] = Counter()
    shapes: Counter[str] = Counter()
    field_orders: Counter[str] = Counter()
    prompt_hist = [0] * (MAX_TOKENS + 1)
    total_hist = [0] * (MAX_TOKENS + 1)
    record_hist = [0] * (MAX_TOKENS + 1)
    option_hist = [0] * (MAX_TOKENS + 1)
    serialized_record_hist = [0] * (MAX_TOKENS + 1)
    max_options = max_continuation = records_checked = questions_checked = 0
    heldout_spec_passed = True
    label_method_verified = 0
    type_records: Counter[str] = Counter()

    try:
        for index in range(count):
            record = make_record(seed, index)
            check_record(record)
            validate(record)
            request = {"model": "m2", "state": record["state"], "questions": record["questions"]}
            from decide.schema import validate_legacy
            validate_legacy(request)
            nodes, field_order = _parse_state(record["state"])
            if len(nodes) != 4 or len(field_order) != 4:
                raise GenerationError("world must contain four distinct relays and four fields")
            if len({n["node_id"] for n in nodes}) != 4:
                raise GenerationError("node ids must be unique within each world")
            row_lengths = {}
            record_max = 0
            for qid, question in record["questions"].items():
                row = build_row(request, qid, encoder, template=TEMPLATE, layout=LAYOUT)
                conts = [len(c) for c in row.continuations]
                if not conts:
                    raise GenerationError("question has no option continuations")
                prompt_tokens = len(row.ids)
                option_lengths = [len(seq) for seq in row.option_seqs] if row.option_seqs else [0] * len(conts)
                if len(option_lengths) != len(conts):
                    raise GenerationError("option and continuation rows are misaligned")
                effective_lengths = [prompt_tokens + option_length + cont
                                     for option_length, cont in zip(option_lengths, conts, strict=True)]
                longest = max(effective_lengths)
                max_option_sequence = max(option_lengths)
                if (not 1 <= prompt_tokens <= MAX_TOKENS or longest > MAX_TOKENS
                        or max(conts) > 16 or not 1 <= row.n_options <= 77):
                    raise GenerationError(f"{record['id']}/{qid}: shared={prompt_tokens} option={max_option_sequence} "
                                          f"full={longest} "
                                          f"K={row.n_options} continuation={max(conts)} exceeds exact limits")
                prompt_hist[prompt_tokens] += 1
                total_hist[longest] += 1
                option_hist[max_option_sequence] += 1
                record_max = max(record_max, longest)
                row_lengths[qid] = {"prompt_tokens": prompt_tokens,
                                    "max_option_sequence_tokens": max_option_sequence,
                                    "max_continuation_tokens": max(conts),
                                    "shared_plus_option_plus_continuation_tokens": longest,
                                    "options": row.n_options}
                question_types[question["type"]] += 1
                option_counts[row.n_options] += 1
                if question["type"] == "choice":
                    choice_option_counts[row.n_options] += 1
                max_options = max(max_options, row.n_options)
                max_continuation = max(max_continuation, max(conts))
                role = record["aug"]["question_roles"][qid]
                if role["kind"] == "top_profile":
                    style = role["id_style"]
                    label_positions[(row.n_options, style, role["gold_position"])] += 1
                    id_styles[style] += 1
                questions_checked += 1
            record_hist[record_max] += 1
            shapes[record["aug"]["schema_shape"]] += 1
            field_orders["|".join(field_order)] += 1
            label_method_verified += 1
            # Golds were independently recomputed from the serialized state above.
            verify_record(record)
            record_line = canonical(record) + b"\n"
            serialized_record_tokens = len(encoder(record_line[:-1].decode("utf-8")))
            if not 1 <= serialized_record_tokens <= MAX_TOKENS:
                raise GenerationError(f"{record['id']}: serialized record has {serialized_record_tokens} tokens")
            serialized_record_hist[serialized_record_tokens] += 1
            manifest_line = canonical({"id": record["id"], "sha256": _sha(record_line[:-1]),
                                       "source": record["source"], "family": record["family"]}) + b"\n"
            content_sha = record_content_hash(record)
            if content_sha in content_seen:
                raise GenerationError(f"duplicate synthetic content at record {index}")
            content_seen.add(content_sha)
            content_line = canonical({"id": record["id"], "content_sha256": content_sha}) + b"\n"
            records_hash.update(record_line)
            manifest_hash.update(manifest_line)
            content_hash.update(content_line)
            records_stream.write(record_line)
            manifest_stream.write(manifest_line)
            content_stream.write(content_line)
            census_stream.write(canonical({"id": record["id"], "rows": row_lengths,
                                           "max_effective_prompt_tokens": record_max,
                                           "serialized_record_tokens": serialized_record_tokens}) + b"\n")
            records_checked += 1
            if records_checked % 1000 == 0:
                print(f"C multirecord: checked {records_checked}/{count}", file=sys.stderr, flush=True)
    finally:
        for stream in streams:
            stream.close()
        for raw in files:
            raw.close()
    if "torch" in sys.modules or records_checked != count or label_method_verified != count:
        raise GenerationError("Torch import or record count mismatch")

    # B-1's families-dir reader accepts each immediate *.jsonl.gz as one external family.
    family_file = family_dir / "c_multirecord.jsonl.gz"
    try:
        os.link(records_path, family_file)
    except OSError:
        import shutil
        shutil.copyfile(records_path, family_file)

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if len(commit) != 40:
        raise GenerationError("could not pin generator source commit")
    data_root = Path(data_root or out.parents[1]).resolve()
    try:
        location = out.resolve().relative_to(data_root).as_posix() + "/train_set/"
    except ValueError as exc:
        raise GenerationError("build output must be under the declared data root") from exc
    pointer = {
        "build": f"{BUILD} (python -X utf8 -m data.c_multirecord_synth build)",
        "location": location,
        "records_uncompressed_sha256": records_hash.hexdigest(),
        "manifest": manifest_path.name,
        "manifest_uncompressed_sha256": manifest_hash.hexdigest(),
        "content_map": content_path.name,
        "content_map_uncompressed_sha256": content_hash.hexdigest(),
        "n_records": count,
        "excludes_cohort": "synthetic-only generation; no Index/TEST/retention rows or source data were read; "
                           "the consuming training-set guard remains required",
        "builder": {"module": "data.c_multirecord_synth", "commit": commit},
        "receipt": f"receipts/2026-10-08-C-multirecord-synth/{out.name}/summary.json",
    }
    pointer_bytes = (json.dumps(pointer, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (train_dir / "pointer.json").write_bytes(pointer_bytes)
    if pointer_out is not None:
        pointer_out = Path(pointer_out)
        pointer_out.parent.mkdir(parents=True, exist_ok=True)
        with pointer_out.open("xb") as stream:
            stream.write(pointer_bytes)
    tokenizer_bytes = {name: _sha_file(tokenizer_dir / name) for name in TOKENIZER_PINS}
    code_files = ("data/c_multirecord_synth.py",
                  "decide/schema.py", "decide/prompt.py", "decide/untrusted.py", "heads/layout.py",
                  "third_party/rwkv7_hf/tokenization_rwkv7.py", "train/cohort.py")
    code_pins = {name: _sha_file(repo / name) for name in code_files}
    summary = {
        "schema": "ssdm.c_multirecord_synth_receipt.v1", "generator": GENERATOR_SPEC,
        "seed": seed, "records": records_checked, "questions": questions_checked,
        "case_exact_records": records_checked, "multi_select_records": records_checked,
        "label_method_verified_records": label_method_verified,
        "source": SOURCE, "source_split": "synthetic", "license": LICENSE,
        "worlds": records_checked, "nodes_per_world": 4,
        "task_types_per_record": dict(sorted(question_types.items())),
        "option_counts_by_k": {str(k): option_counts[k] for k in sorted(option_counts)},
        "choice_option_counts_by_k": {str(k): choice_option_counts[k] for k in sorted(choice_option_counts)},
        "option_id_style_counts": dict(sorted(id_styles.items())),
        "field_order_counts": dict(sorted(field_orders.items())),
        "schema_shape_counts": dict(sorted(shapes.items())),
        "max_options": max_options, "max_continuation_tokens": max_continuation,
        "token_limits": {"tokenizer_revision": TOKENIZER_REVISION, "tokenizer_sha256": tokenizer_bytes,
                         "template": TEMPLATE, "layout": LAYOUT, "max_tokens": MAX_TOKENS,
                         "max_options": 77, "prompt_tokens": _histogram_summary(prompt_hist),
                         "max_option_sequence_tokens": _histogram_summary(option_hist),
                         "shared_plus_option_plus_continuation_tokens": _histogram_summary(total_hist),
                         "serialized_record_tokens": _histogram_summary(serialized_record_hist),
                         "record_max_effective_prompt_tokens": _histogram_summary(record_hist)},
        "heldout": {"generator_spec_check_passed": heldout_spec_passed,
                    "record_checks_passed": records_checked, "refused": 0,
                    "frozen_ids": [9, 20, 22, 31, 32, 33, 44, 58, 64]},
        "source_sha256": code_pins,
        "uncompressed_sha256": {"records": records_hash.hexdigest(),
                                 "manifest": manifest_hash.hexdigest(), "content_map": content_hash.hexdigest()},
        "compressed_sha256": {"records": _sha_file(records_path), "manifest": _sha_file(manifest_path),
                              "content_map": _sha_file(content_path), "token_census": _sha_file(census_path)},
        "pointer_sha256": _sha(pointer_bytes), "pointer": pointer,
        "multi_select_representation": "four correlated noul questions per record; no native multi_select wire type",
        "notes": ["All records are fully synthetic; no official source splits were used.",
                  "Every question label is generated from a recomputed telemetry truth function.",
                  "The case-exact question requires all four proposed signatures and every field to match.",
                  "B-1 external admission requires B to register ext:c_multirecord in EXT_GROUP and assign a quota.",
                  "B-1 protected-suite/Index filter was not run by C; no TEST/heldout/cohort rows were read."],
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    (out / "summary.json").write_bytes(summary_bytes)
    checksum = {"summary.json": _sha(summary_bytes), "pointer.json": _sha(pointer_bytes),
                "records.jsonl.gz": _sha_file(records_path), "manifest": _sha_file(manifest_path),
                "content_map": _sha_file(content_path), "token_census.jsonl.gz": _sha_file(census_path),
                "families/c_multirecord.jsonl.gz": _sha_file(family_file)}
    (out / "sha256.json").write_bytes((json.dumps(checksum, indent=2, sort_keys=True) + "\n").encode())
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--tokenizer-dir", type=Path, required=True)
    build_parser.add_argument("--pointer-out", type=Path)
    build_parser.add_argument("--data-root", type=Path)
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build_parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args(argv)
    summary = build(args.out, tokenizer_dir=args.tokenizer_dir, seed=args.seed, count=args.count,
                    pointer_out=args.pointer_out, data_root=args.data_root)
    print(json.dumps({"build": BUILD, "records": summary["records"], "questions": summary["questions"],
                      "max_tokens": summary["token_limits"]["record_max_effective_prompt_tokens"]["max"],
                      "records_sha256": summary["uncompressed_sha256"]["records"],
                      "pointer_sha256": summary["pointer_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
