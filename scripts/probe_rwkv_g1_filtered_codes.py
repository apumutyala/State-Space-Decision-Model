"""Tokenizer-only proof of the separately registered filtered C1 component; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from decide.backends.rwkv_g1_filtered_codes import (  # noqa: E402
    CONTEXT_SHA256, G1FilteredTokenReadout, PROOF_CONTEXT, TABLE_SHA256, TOKENIZER_SHA256,
    TOKENIZER_SOURCE_SHA256, VOCAB_SIZE, canonical_bytes, load_frozen_table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = REPO / "third_party/rwkv7_hf/tokenization_rwkv7.py"
    vh = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
    sh = hashlib.sha256(source.read_bytes()).hexdigest()
    if vh != TOKENIZER_SHA256 or sh != TOKENIZER_SOURCE_SHA256:
        raise ValueError("exact vocabulary/reviewed tokenizer source hash mismatch")
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    tok = Rwkv7Tokenizer(tokenizer_file=str(args.tokenizer_json))
    if tok.vocab_size != VOCAB_SIZE:
        raise ValueError("G1k vocabulary size differs from the pinned readout")
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    backend = G1FilteredTokenReadout(encode, tok.decode, tokenizer_sha256=vh, tokenizer_source_sha256=sh,
                                    special_ids=tok.all_special_ids)
    entries = []
    seen = set()
    prefix = encode(PROOF_CONTEXT)
    for ordinal in range(1, 703):
        n, code = ordinal, ""
        while n:
            n, digit = divmod(n - 1, 26)
            code = chr(65 + digit) + code
        ids = encode(" " + code)
        if (len(ids) == 1 and ids[0] not in set(tok.all_special_ids) | {0} and ids[0] not in seen
                and tok.decode(ids) == " " + code and encode(PROOF_CONTEXT + " " + code) == prefix + ids):
            seen.add(ids[0])
            entries.append({"code": code, "token_id": ids[0], "source_ordinal": ordinal})
    if len(entries) != 498 or hashlib.sha256(canonical_bytes(entries[:255])).hexdigest() != TABLE_SHA256:
        raise ValueError("deterministic selection differs from the registration")
    boundary_checks = [{"code": c.code, "token_ids": encode(" " + c.code),
                        "decoded": tok.decode([c.token_id]),
                        "context_token_ids": encode(PROOF_CONTEXT + " " + c.code)[len(prefix):],
                        "prefix_stable": encode(PROOF_CONTEXT + " " + c.code)[:len(prefix)] == prefix}
                       for c in load_frozen_table()]
    result = {"schema": "ssdm.rwkv-g1-filtered-component-proof.v1",
              "status": "tokenizer_component_ready_renderer_blocked", "cost_usd": 0,
              "base_commit": "ee0adb70029fc80c65ee6072c307873034052362",
              "preregistration_commit": "bfbb9acdb28c1e0fa081a86d74c89fa01d86d01c",
              "hardware": "local Windows CPU; tokenizer operations only", "model_loaded": False,
              "model_forward_calls": 0, "checkpoint_loaded": False, "gpu_used": False,
              "tokenizer_path": str(args.tokenizer_json.resolve()), "tokenizer_sha256": vh,
              "tokenizer_source_sha256": sh, "proof_context_sha256": CONTEXT_SHA256,
              "source_candidates": 702, "eligible_candidates": len(entries), "selected_count": 255,
              "last_selected": entries[254], "all_boundary_checks_passed": True,
              "checks": boundary_checks, "revision": backend.revision(),
              "renderer_integration": "blocked: predecessor C1 has no reviewed full request transformation; preserve exact state/provenance semantics",
              "default_changed": False, "serving_enabled": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("status", "source_candidates", "eligible_candidates", "selected_count", "last_selected")}, indent=2))


if __name__ == "__main__":
    main()
