"""Exact tokenizer-only v1/v2 workload on the explicitly authorized pinned 2,008 DEV rows.

Never scores, constructs a model, accesses expected labels or opens TEST. Outputs
only hashes, run IDs and workload/resource counts, without item text or gold.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from decide.adapter import DEFAULT_LIMITS  # noqa: E402
from decide.backends.hf_readout import ModelPlugin  # noqa: E402
from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend, G1_REVISION  # noqa: E402
from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend  # noqa: E402
from decide.backends.rwkv_g1_filtered_codes import TOKENIZER_SHA256, TOKENIZER_SOURCE_SHA256  # noqa: E402
from decide.backends.state_fork import fork_plan  # noqa: E402
from decide.untrusted import ControlTokenError  # noqa: E402

ROWS_SHA256 = "406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2"
TRAINED_CONTEXT = 25600


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workload(backend, request):
    rows = backend._rows(request)
    prompts = {r["qid"]: r["prompt"] for r in rows}
    if not backend.fork:
        total = sum(map(len, prompts.values()))
        return {"tokens": total, "positions": total, "calls": len(prompts),
                "max_prompt_tokens": max(map(len, prompts.values())), "n_options": len(rows)}
    prefix, calls = fork_plan(rows, backend.fork_max_rows, backend.fork_token_budget,
                              backend.plugin.fork_state_token_budget)
    packs = [c for c in calls if c.kind == "pack"]
    assert prefix > 0 and all(len(c.tails) <= backend.fork_max_rows for c in packs)
    return {"tokens": sum(c.tokens for c in calls),
            "positions": sum(c.tokens if c.kind == "prefix" else c.width * len(c.tails) for c in calls),
            "calls": len(calls), "prefix_tokens": prefix,
            "max_prompt_tokens": max(map(len, prompts.values())), "n_options": len(rows),
            "max_pack_rows": max(len(c.tails) for c in packs),
            "max_pack_positions": max(c.width * len(c.tails) for c in packs),
            "oversized_singleton_packs": sum(len(c.tails) == 1 and c.width > backend.fork_token_budget for c in packs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = REPO / "third_party/rwkv7_hf/tokenization_rwkv7.py"
    if digest(args.rows) != ROWS_SHA256 or digest(args.tokenizer_json) != TOKENIZER_SHA256 or digest(source) != TOKENIZER_SOURCE_SHA256:
        raise ValueError("pinned DEV subset/tokenizer/source hash mismatch")
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    tok = Rwkv7Tokenizer(tokenizer_file=str(args.tokenizer_json))
    special = frozenset(tok.all_special_ids) | {0}
    def encode(text, trusted=True):
        ids = tok.encode(text, add_special_tokens=False)
        if not trusted and special.intersection(ids):
            raise ControlTokenError("untrusted control token")
        return ids
    def no_model(*args, **kwargs):
        raise AssertionError("volume proof must not call a model or state operation")
    no_state = type("NoModelStateOps", (), {"prefill": staticmethod(no_model), "extend": staticmethod(no_model)})()
    plugin = ModelPlugin("rwkv7", "tokenizer-volume-only-no-model", None, encode, no_model,
        files={"tokenizer": TOKENIZER_SHA256, "weights": "not-loaded"}, notes={"vendored_code": G1_REVISION},
        special_ids=special, device="cpu", dtype="not-loaded", max_context=TRAINED_CONTEXT, state_ops=no_state)
    v1 = G1CodeLabelBackend(plugin, tokenizer_decode=tok.decode)
    v2 = G1CodeLabelV2Backend(plugin, tokenizer_decode=tok.decode)
    v2._enc_memo = v1._enc_memo  # Pure tokenization cache only; no recurrent/model state exists.
    cap = min(DEFAULT_LIMITS["max_input_tokens"], TRAINED_CONTEXT)
    per_row = {}
    totals = {scope: {arm: collections.Counter() for arm in ("v1", "v2")}
              for scope in ("all_planned", "admitted", "paired_admitted")}
    admissions = collections.Counter()
    with gzip.open(args.rows, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["_evaluation"]["run_id"]
            if rid in per_row:
                raise ValueError("duplicate DEV run ID")
            request = {"model": "volume-only", "state": row["state"], "questions": row["questions"]}
            # No expected/gold field is used or retained.
            v1._enc_memo.clear()
            record = {"n_questions": len(request["questions"])}
            try:
                a, b = workload(v1, request), workload(v2, request)
                record.update({"v1": a, "v2": b})
                admitted = {"v1": a["max_prompt_tokens"] <= cap, "v2": b["max_prompt_tokens"] <= cap}
                record["admitted"] = admitted
                for arm, work in (("v1", a), ("v2", b)):
                    admissions[arm + ("_admitted" if admitted[arm] else "_input_exhausted")] += 1
                    for scope in ("all_planned", "admitted", "paired_admitted"):
                        if scope == "admitted" and not admitted[arm] or scope == "paired_admitted" and not all(admitted.values()):
                            continue
                        for key in ("tokens", "positions", "calls", "prefix_tokens", "n_options", "oversized_singleton_packs"):
                            totals[scope][arm][key] += work.get(key, 0)
                        for key in ("max_prompt_tokens", "max_pack_rows", "max_pack_positions"):
                            totals[scope][arm][key] = max(totals[scope][arm][key], work.get(key, 0))
            except (ValueError, ControlTokenError) as exc:
                record["planning_refusal"] = type(exc).__name__
                admissions["planning_refusal"] += 1
            per_row[rid] = record
            if len(per_row) % 100 == 0:
                print(json.dumps({"rows": len(per_row), "v1_tokens": totals["all_planned"]["v1"]["tokens"],
                    "v2_positions": totals["all_planned"]["v2"]["positions"]}), flush=True)
    if len(per_row) != 2008:
        raise ValueError("expected exactly the pinned 2,008-row subset")
    result = {"schema": "ssdm.rwkv-g1-code-label-v2-volume.v1", "rows": len(per_row),
        "scope": "explicitly authorized DEV tokenizer-only workload; no labels/scoring/TEST",
        "input_rows_path": str(args.rows.resolve()), "input_rows_sha256": ROWS_SHA256,
        "tokenizer_path": str(args.tokenizer_json.resolve()), "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256,
        "preregistration_commit": "3c9a4b42b3c8358bcfc7457ce5bbee3c1994c664",
        "integration_base": "4b59a73", "calling_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "volume_script_sha256": digest(__file__), "hardware": "local Windows CPU; tokenizer only", "cost_usd": 0,
        "model_loaded": False, "model_forward_calls": 0, "gpu_used": False, "item_text_or_gold_exported": False,
        "effective_input_cap_tokens": cap, "rows_denominator": 2008, "admission_counts": dict(admissions),
        "totals": totals, "per_row": per_row, "v1_revision": v1.revision(), "v2_revision": v2.revision(),
        "notes": ["positions include padding; tokens exclude padding", "admitted counts are workload diagnostics, not a new score subset",
                  "packing token budget is not a hard limit for a singleton long tail", "no measured GPU time/memory or spend quote"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"rows": result["rows"], "admission_counts": result["admission_counts"], "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
