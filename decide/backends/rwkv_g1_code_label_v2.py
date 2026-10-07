"""All-field code-label v2: identical schema/state prefix and bounded RWKV state forks."""
from __future__ import annotations

from pathlib import Path

from ..prompt import CANARY_REQUEST, SYSTEM_PROMPT, TEMPLATES, build_prompt_labelled
from ..schema import canonical_json, canonical_options, canonical_question_ids, sha256_text, validate_legacy
from .base import BackendUnsupported, Plan
from .hf_readout import HFReadoutBackend, sha256_file
from .rwkv_g1_code_label import BASE_TEMPLATE_ID, CODE_SYSTEM_PROMPT, G1CodeLabelBackend, G1_REVISION
from .rwkv_g1_filtered_codes import (G1FilteredTokenReadout, TABLE_SHA256, TOKENIZER_SHA256,
                                    TOKENIZER_SOURCE_SHA256, VOCAB_SIZE, load_frozen_table)
from .state_fork import fork_plan, fork_score

TEMPLATE_ID = "jrt-rwkv-g1-code-label-v2"
TRANSFORM_VERSION = "trusted-all-field-code-insertion-v2"
MAX_FORK_ROWS = 4
FORK_TOKEN_BUDGET = 4096


def render_all_field_code_labels(request, target_qid, provenance=None):
    validate_legacy(request)
    base = TEMPLATES[BASE_TEMPLATE_ID]
    original = build_prompt_labelled(base, request, target_qid, provenance)
    order = canonical_question_ids(request["questions"])
    table = load_frozen_table()
    codes = {qid: table[:len(canonical_options(request["questions"][qid]))] for qid in order}
    markers = {f"FIELD {i + 1}\nID: ": qid for i, qid in enumerate(order)}
    head = base.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n"
    reply = "\nReply with exactly one allowed option id." + base.tail
    inserted = dict.fromkeys(order, 0)
    active = None
    systems = replies = 0
    output = []
    for text, label in original:
        if label == "template" and text == head:
            output.append((base.head.format(system=CODE_SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", label))
            systems += 1
            continue
        if label == "template" and text in markers:
            active = markers[text]
        if label == "template" and text == "\nEND FIELD":
            active = None
        if label == "template" and text == reply:
            output.extend([("\nALLOWED ANSWER CODES: " + ", ".join(c.code for c in codes[target_qid]), "template"),
                           ("\nReply with exactly one allowed answer code." + base.tail, "template")])
            replies += 1
            continue
        output.append((text, label))
        if active is not None and label == "template" and text == f"\nOPTION {inserted[active] + 1}: ":
            if inserted[active] >= len(codes[active]):
                raise BackendUnsupported("code_label_renderer_structure_changed", "extra schema option marker")
            output.append((f"[ANSWER CODE {codes[active][inserted[active]].code}] ", "template"))
            inserted[active] += 1
    if systems != 1 or replies != 1 or any(inserted[qid] != len(codes[qid]) for qid in order):
        raise BackendUnsupported("code_label_renderer_structure_changed", "base renderer markers differ from v2 registration")
    if [s for s in original if s[1] != "template"] != [s for s in output if s[1] != "template"]:
        raise BackendUnsupported("code_label_segment_mismatch", "v2 request-derived segment preservation failed")
    return output


def renderer_identity():
    value = {"id": TEMPLATE_ID, "transform": TRANSFORM_VERSION,
             "base_template": TEMPLATES[BASE_TEMPLATE_ID].identity(), "codebook_table_sha256": TABLE_SHA256,
             "labelled_canary": {qid: render_all_field_code_labels(CANARY_REQUEST, qid)
                                  for qid in canonical_question_ids(CANARY_REQUEST["questions"])}}
    return {"id": TEMPLATE_ID, "sha256": sha256_text(canonical_json(value))}


class _CheckedStateOps:
    """Keep the existing state operations/scorer; validate the returned vocabulary tensors."""
    def __init__(self, ops):
        self.ops = ops

    @staticmethod
    def _checked(result, shape):
        import torch

        logits, state = result
        if tuple(logits.shape) != shape:
            raise BackendUnsupported("backend_output_shape", "unexpected state-fork vocabulary output shape")
        if not bool(torch.isfinite(logits).all()):
            raise BackendUnsupported("non_finite_logits", "nonfinite state-fork vocabulary output")
        return logits, state

    def prefill(self, ids):
        return self._checked(self.ops.prefill(ids), (1, 1, VOCAB_SIZE))

    def extend(self, state, ids, n_last):
        return self._checked(self.ops.extend(state, ids, n_last), (ids.shape[0], n_last, VOCAB_SIZE))


class G1CodeLabelV2Backend(G1CodeLabelBackend):
    name = "rwkv_g1_code_label_v2_experimental"

    def __init__(self, plugin, *, tokenizer_decode, fork=True, fork_max_rows=MAX_FORK_ROWS,
                 fork_token_budget=FORK_TOKEN_BUDGET):
        if (plugin.family != "rwkv7" or plugin.files.get("tokenizer") != TOKENIZER_SHA256
                or plugin.notes.get("vendored_code") != G1_REVISION or 0 not in plugin.special_ids):
            raise ValueError("v2 requires the pinned G1 tokenizer/plugin and fail-closed control IDs")
        if (type(fork_max_rows) is not int or not 1 <= fork_max_rows <= MAX_FORK_ROWS
                or type(fork_token_budget) is not int or not 1 <= fork_token_budget <= FORK_TOKEN_BUDGET):
            raise ValueError("v2 pack configuration exceeds the registered conservative cap")
        if fork and (plugin.state_ops is None or not all(callable(getattr(plugin.state_ops, k, None)) for k in ("prefill", "extend"))):
            raise ValueError("v2 fork scoring requires state operations; no recompute fallback")
        self.codebook = G1FilteredTokenReadout(lambda text: plugin.encode(text, True), tokenizer_decode,
            tokenizer_sha256=plugin.files["tokenizer"], tokenizer_source_sha256=TOKENIZER_SOURCE_SHA256,
            special_ids=plugin.special_ids)
        self._fork_ops = _CheckedStateOps(plugin.state_ops) if fork else None
        HFReadoutBackend.__init__(self, plugin, BASE_TEMPLATE_ID, max_batch_rows=1, fork=fork,
                                 fork_max_rows=fork_max_rows, fork_token_budget=fork_token_budget)

    def _build_revision(self):
        revision = G1CodeLabelBackend._build_revision(self)
        config = dict(revision["inference_config"])
        sources = dict(config["source_sha256"])
        sources.update({p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("state_fork.py"))})
        config.update({"request_transform": TRANSFORM_VERSION,
                       "scoring": "exact one-token code log probability; identical v2 prompts under fork/recompute",
                       "scoring_path": "existing RWKV prefix-fork scorer" if self.fork else "diagnostic next-token recompute",
                       "rows": "one single-token continuation per supplied option; bounded packed question tails",
                       "fork_max_rows": self.fork_max_rows, "fork_token_budget": self.fork_token_budget,
                       "fork_state_token_budget": self.plugin.fork_state_token_budget,
                       "full_output_finite_check": True, "source_sha256": sources})
        config_hash = sha256_text(canonical_json(config))
        template = renderer_identity()
        revision.update({"prompt_template": template, "inference_config": config, "inference_config_sha256": config_hash,
            "revision_id": f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
            f";w={self.plugin.files.get('weights', 'unknown')[:12]};tok={TOKENIZER_SHA256[:12]}"
            f";prompt={template['id']}.{template['sha256'][:8]};cfg={config_hash[:8]};adapter={revision['adapter_version']}"})
        return revision

    def _rows(self, request, label_counts=None):
        provenance = self._provenance(request)
        rows = []
        for qid in canonical_question_ids(request["questions"]):
            labelled = render_all_field_code_labels(request, qid, provenance)
            prompt = self._encode([(text, label == "template") for text, label in labelled], label_counts,
                                  [label for _, label in labelled])
            for semantic_id, token_id in self.codebook.option_token_ids(request["questions"][qid]).items():
                rows.append({"qid": qid, "option": semantic_id, "prompt": prompt, "cont": [token_id]})
        return rows

    def plan(self, request):
        if not self.fork:
            return G1CodeLabelBackend.plan(self, request)
        labels = {}
        rows = self._rows(request, labels)
        prefix, calls = fork_plan(rows, self.fork_max_rows, self.fork_token_budget, self.plugin.fork_state_token_budget)
        if prefix < 1:
            raise BackendUnsupported("fork_shared_prefix_missing", "v2 requires a shared prefix; no recompute fallback")
        prompts = {row["qid"]: row["prompt"] for row in rows}
        return Plan(inference_calls=len(calls), max_call_input_tokens=max(map(len, prompts.values())),
                    total_input_tokens=sum(c.tokens for c in calls),
                    per_question={qid: {"prompt_tokens": len(prompt), "options": sum(r["qid"] == qid for r in rows)}
                                  for qid, prompt in prompts.items()}, prompt_token_labels=labels,
                    provenance_applied=False)

    def score(self, request, check_deadline):
        if not self.fork:
            return G1CodeLabelBackend.score(self, request, check_deadline)
        rows = self._rows(request)
        if not self._use_fork(rows):
            raise BackendUnsupported("fork_shared_prefix_missing", "v2 requires a shared prefix; no recompute fallback")
        logits, calls, total, latencies = fork_score(rows, self._fork_ops, self.fork_max_rows, self.plugin.device,
            check_deadline, self.fork_token_budget, self.plugin.fork_state_token_budget)
        result = self._result(rows, logits, calls, total, latencies)
        result.extras["scoring_path"] = "all-field-code-label-v2-prefix-fork"
        return result
