"""Explicit opt-in, segment-preserving G1 code-label readout; defaults are untouched."""
from __future__ import annotations

import copy
from pathlib import Path
import time

from ..prompt import CANARY_REQUEST, SYSTEM_PROMPT, TEMPLATES, build_prompt_labelled
from ..schema import canonical_json, canonical_options, canonical_question_ids, sha256_text, validate_legacy
from .base import BackendUnsupported, Plan
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_filtered_codes import (G1FilteredTokenReadout, TABLE_SHA256, TOKENIZER_SHA256,
                                    TOKENIZER_SOURCE_SHA256, VOCAB_SIZE, load_frozen_table)

TEMPLATE_ID = "jrt-rwkv-g1-code-label-v1"
TRANSFORM_VERSION = "trusted-target-code-insertion-v1"
BASE_TEMPLATE_ID = "jrt-rwkv-g1-v1"
G1_REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
CODE_SYSTEM_PROMPT = (
    "Read the schema, then the state, then the question. Answer with exactly one of "
    "the question's allowed answer codes."
)


def render_code_labels(request, target_qid, provenance=None):
    """Insert only trusted labels; keep every original non-template segment exact.

    Structural markers are matched only on template-labelled segments, so neither
    untrusted text nor provenance instruction spans can impersonate an OPTION.
    Unexpected changes to the base renderer structure fail closed.
    """
    validate_legacy(request)
    base = TEMPLATES[BASE_TEMPLATE_ID]
    original = build_prompt_labelled(base, request, target_qid, provenance)
    order = canonical_question_ids(request["questions"])
    target_index = order.index(target_qid) + 1
    options = canonical_options(request["questions"][target_qid])
    codes = load_frozen_table()[:len(options)]
    head = base.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n"
    reply = "\nReply with exactly one allowed option id." + base.tail
    in_target = False
    inserted = systems = replies = 0
    output = []
    for text, label in original:
        if label == "template" and text == head:
            output.append((base.head.format(system=CODE_SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", label))
            systems += 1
            continue
        if label == "template" and text == f"FIELD {target_index}\nID: ":
            in_target = True
        if label == "template" and text == "\nEND FIELD":
            in_target = False
        if label == "template" and text == reply:
            output.extend([("\nALLOWED ANSWER CODES: " + ", ".join(c.code for c in codes), "template"),
                           ("\nReply with exactly one allowed answer code." + base.tail, "template")])
            replies += 1
            continue
        output.append((text, label))
        if in_target and label == "template" and text == f"\nOPTION {inserted + 1}: ":
            if inserted >= len(codes):
                raise BackendUnsupported("code_label_renderer_structure_changed", "extra target option marker")
            output.append((f"[ANSWER CODE {codes[inserted].code}] ", "template"))
            inserted += 1
    if systems != 1 or replies != 1 or inserted != len(options):
        raise BackendUnsupported("code_label_renderer_structure_changed", "base renderer markers differ from the registration")
    if ([segment for segment in original if segment[1] != "template"]
            != [segment for segment in output if segment[1] != "template"]):
        raise BackendUnsupported("code_label_segment_mismatch", "request-derived segment preservation failed")
    return output


def renderer_identity():
    canary = {qid: render_code_labels(CANARY_REQUEST, qid) for qid in canonical_question_ids(CANARY_REQUEST["questions"])}
    value = {"id": TEMPLATE_ID, "transform": TRANSFORM_VERSION,
             "base_template": TEMPLATES[BASE_TEMPLATE_ID].identity(), "codebook_table_sha256": TABLE_SHA256,
             "labelled_canary": canary}
    return {"id": TEMPLATE_ID, "sha256": sha256_text(canonical_json(value))}


class G1CodeLabelBackend(HFReadoutBackend):
    """One next-token forward per target question, mapping every code to its original ID."""
    name = "rwkv_g1_code_label_experimental"

    def __init__(self, plugin, *, tokenizer_decode):
        if (plugin.family != "rwkv7" or plugin.files.get("tokenizer") != TOKENIZER_SHA256
                or plugin.notes.get("vendored_code") != G1_REVISION or 0 not in plugin.special_ids):
            raise ValueError("code-label arm requires the pinned G1 tokenizer/plugin and fail-closed control IDs")
        self.codebook = G1FilteredTokenReadout(lambda text: plugin.encode(text, True), tokenizer_decode,
            tokenizer_sha256=plugin.files["tokenizer"], tokenizer_source_sha256=TOKENIZER_SOURCE_SHA256,
            special_ids=plugin.special_ids)
        super().__init__(plugin, BASE_TEMPLATE_ID, max_batch_rows=1, fork=False)

    def _build_revision(self):
        revision = super()._build_revision()
        config = dict(revision["inference_config"])
        sources = [Path(__file__), Path(__file__).with_name("hf_readout.py"),
                   Path(__file__).parents[1] / "prompt.py", Path(__file__).parents[1] / "schema.py",
                   Path(__file__).parents[1] / "calibration.py", Path(__file__).parents[1] / "provenance.py"]
        config.update({"scoring": "log_probability(exact frozen answer token ID | segmented v1 prompt + trusted target code labels)",
                       "scoring_path": "one next-token forward per target question; no continuation token forwarded",
                       "codebook_table_sha256": TABLE_SHA256, "codebook_component": self.codebook.revision(),
                       "request_transform": TRANSFORM_VERSION, "bos_change": False, "tail_change": False,
                       "raw_request_text_change": False,
                       "source_sha256": {p.name: sha256_file(p) for p in sources}})
        config_hash = sha256_text(canonical_json(config))
        template = renderer_identity()
        revision.update({"prompt_template": template, "inference_config": config,
                         "inference_config_sha256": config_hash,
                         "revision_id": f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
                         f";w={self.plugin.files.get('weights', 'unknown')[:12]};tok={TOKENIZER_SHA256[:12]}"
                         f";prompt={template['id']}.{template['sha256'][:8]};cfg={config_hash[:8]}"
                         f";adapter={revision['adapter_version']}"})
        return revision

    def revision(self):
        return copy.deepcopy(self._revision)

    def parse_exact_token_ids(self, question, token_ids):
        return self.codebook.parse_exact_token_ids(question, token_ids)

    def _rows(self, request, label_counts=None):
        provenance = self._provenance(request)
        rows = []
        for qid in canonical_question_ids(request["questions"]):
            labelled = render_code_labels(request, qid, provenance)
            prompt = self._encode([(text, label == "template") for text, label in labelled], label_counts,
                                  [label for _, label in labelled])
            for semantic_id, token_id in self.codebook.option_token_ids(request["questions"][qid]).items():
                rows.append({"qid": qid, "option": semantic_id, "prompt": prompt, "cont": [token_id]})
        return rows

    def plan(self, request):
        labels = {}
        rows = self._rows(request, labels)
        prompts = {r["qid"]: r["prompt"] for r in rows}
        return Plan(inference_calls=len(prompts), max_call_input_tokens=max(map(len, prompts.values())),
                    total_input_tokens=sum(map(len, prompts.values())),
                    per_question={qid: {"prompt_tokens": len(prompt), "options": sum(r["qid"] == qid for r in rows)}
                                  for qid, prompt in prompts.items()}, prompt_token_labels=labels,
                    provenance_applied=False)

    def score(self, request, check_deadline):
        import torch

        rows = self._rows(request)
        by_question = {}
        for row in rows:
            by_question.setdefault(row["qid"], []).append(row)
        logits, latencies, total = {}, [], 0
        for qid, question_rows in by_question.items():
            check_deadline()
            prompt = question_rows[0]["prompt"]
            ids = torch.tensor([prompt], dtype=torch.long, device=self.plugin.device)
            total += len(prompt)
            start = time.perf_counter()
            with torch.inference_mode():
                output = self.plugin.forward(ids, 1)
                if tuple(output.shape) != (1, 1, VOCAB_SIZE):
                    raise BackendUnsupported("backend_output_shape", "expected one full-vocabulary next-token logit vector")
                if not bool(torch.isfinite(output).all()):
                    raise BackendUnsupported("non_finite_logits", "nonfinite full-vocabulary logits")
                log_probs = torch.log_softmax(output[0, -1].float(), dim=-1)
                token_ids = torch.tensor([r["cont"][0] for r in question_rows], device=log_probs.device)
                values = log_probs.index_select(0, token_ids).tolist()
            if self.plugin.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(round((time.perf_counter() - start) * 1000, 3))
            logits[qid] = {r["option"]: value for r, value in zip(question_rows, values)}
        result = self._result(rows, logits, len(by_question), total, latencies)
        result.extras["scoring_path"] = "one-next-token-forward-per-question"
        return result
