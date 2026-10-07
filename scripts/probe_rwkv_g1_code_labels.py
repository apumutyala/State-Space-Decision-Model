"""CPU/tokenizer-only proof for the segment-preserving code-label arm; no model calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from decide.backends.hf_readout import ModelPlugin  # noqa: E402
from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend, G1_REVISION, render_code_labels  # noqa: E402
from decide.backends.rwkv_g1_filtered_codes import (  # noqa: E402
    TABLE_SHA256, TOKENIZER_SHA256, TOKENIZER_SOURCE_SHA256, load_frozen_table)
from decide.prompt import TEMPLATES, build_prompt_labelled  # noqa: E402
from decide.provenance import criteria_path, instructions_path, validate_provenance  # noqa: E402
from decide.schema import request_fingerprint  # noqa: E402
from decide.untrusted import ControlTokenError  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = REPO / "third_party/rwkv7_hf/tokenization_rwkv7.py"
    vh = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
    sh = hashlib.sha256(source.read_bytes()).hexdigest()
    if vh != TOKENIZER_SHA256 or sh != TOKENIZER_SOURCE_SHA256:
        raise ValueError("exact tokenizer/source pin mismatch")
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    tok = Rwkv7Tokenizer(tokenizer_file=str(args.tokenizer_json))
    special = frozenset(tok.all_special_ids) | {0}
    def encode(text, trusted=True):
        ids = tok.encode(text, add_special_tokens=False)
        if not trusted and special.intersection(ids):
            raise ControlTokenError("untrusted control token")
        return ids
    def no_forward(*args, **kwargs):
        raise AssertionError("tokenizer proof must not call a model")
    plugin = ModelPlugin("rwkv7", "tokenizer-proof-only-no-model", None, encode, no_forward,
        files={"tokenizer": vh, "weights": "not-loaded"}, notes={"vendored_code": G1_REVISION},
        dtype="not-loaded", device="cpu", special_ids=special, max_context=25600)
    backend = G1CodeLabelBackend(plugin, tokenizer_decode=tok.decode)
    request = {"model": "synthetic", "state": "Preserve\n\nparagraphs\r\nand Unicode: λ.",
               "questions": {"q": {"type": "choice", "instructions": "Choose\n\nexactly.",
                   "criteria": {f"original-{i:03}": f"Description {i}: raw\ntext." for i in range(255)}}}}
    def spans(text):
        return [{"start": 0, "end": 2, "trust": "instruction", "origin": "host_template"},
                {"start": 2, "end": len(text), "trust": "data", "origin": "record"}]
    request["ssdm"] = {"provenance": {"schema": "ssdm.provenance.v0", "serializer": "synthetic-proof",
        "default_trust": "data", "segments": {"/state": spans(request["state"]),
            instructions_path("q"): spans(request["questions"]["q"]["instructions"]),
            criteria_path("q", "original-000"): spans(request["questions"]["q"]["criteria"]["original-000"])}}}
    before = copy.deepcopy(request)
    fingerprint = request_fingerprint(request)
    prov = validate_provenance(request, request["ssdm"]["provenance"])
    original = build_prompt_labelled(TEMPLATES["jrt-rwkv-g1-v1"], request, "q", prov)
    labelled = render_code_labels(request, "q", prov)
    assert [s for s in original if s[1] != "template"] == [s for s in labelled if s[1] != "template"]
    full_text = "".join(text for text, _ in labelled)
    whole_prefix = encode(full_text)
    checks = []
    for entry in load_frozen_table():
        combined = encode(full_text + " " + entry.code)
        assert combined == whole_prefix + [entry.token_id]
        assert tok.decode([entry.token_id]) == " " + entry.code
        assert backend.parse_exact_token_ids(request["questions"]["q"], [entry.token_id]) == f"original-{len(checks):03}"
        checks.append({"code": entry.code, "token_id": entry.token_id, "whole_rendered_context_boundary": True})
    plan = backend.plan(request)
    assert not plan.provenance_applied
    assert request == before and request_fingerprint(request) == fingerprint
    tail_unchanged = full_text.endswith(TEMPLATES["jrt-rwkv-g1-v1"].tail)
    assert tail_unchanged and backend._rows(request)[0]["prompt"][0] != 0
    result = {"schema": "ssdm.rwkv-g1-code-label-tokenizer-proof.v1", "status": "cpu_enablement_proved_no_model_measurement",
        "base_commit": "ee0adb70029fc80c65ee6072c307873034052362",
        "preregistration_commit": "1e473f5146fcadd6a34482435ae4be05698593d9",
        "hardware": "local Windows CPU; tokenizer only", "cost_usd": 0,
        "model_loaded": False, "model_forward_calls": 0, "checkpoint_loaded": False, "gpu_used": False,
        "tokenizer_path": str(args.tokenizer_json.resolve()), "tokenizer_sha256": vh,
        "tokenizer_source_sha256": sh, "codebook_table_sha256": TABLE_SHA256,
        "request_sha256": hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "request_fingerprint_unchanged": True, "non_template_segments_exact": True,
        "split_provenance_segments_exact": True, "provenance_mask_applied": False,
        "v1_tail_unchanged": tail_unchanged, "bos_inserted": False, "default_changed": False,
        "selected_count": len(checks), "all_rendered_context_boundaries_exact": True, "checks": checks,
        "synthetic_plan": {"inference_calls": plan.inference_calls, "max_call_input_tokens": plan.max_call_input_tokens,
                           "total_input_tokens": plan.total_input_tokens, "prompt_token_labels": plan.prompt_token_labels},
        "whole_string_token_count_diagnostic_only": len(whole_prefix),
        "revision": backend.revision(), "qualification": "capability/calibration/safety/latency unmeasured"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("status", "selected_count", "non_template_segments_exact",
        "split_provenance_segments_exact", "v1_tail_unchanged", "synthetic_plan")}, indent=2))


if __name__ == "__main__":
    main()
