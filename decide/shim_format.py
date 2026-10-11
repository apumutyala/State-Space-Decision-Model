"""The H2O-Lightning-4B v1.1 shim prompt as a TRAINABLE readout (templates ``jrt-h2o-shim:<family>``).

The host scan (job-zbtbj, claude/jev-host-scan @91a506a) measured stock Qwen3.5-4B on the 231 public JevBench items at
open-I 46.1 under v3w and 56.3 under H2O-Lightning's own shim prompt + label readout (the vendored, unmodified
``third_party/h2o_lightning_shim``, which reproduced H2O's card at 205/231). This module trains and serves exactly that
format, so train == serve:

  prompt   ``Contract.question_prompt`` of the vendored shim (serve_config.json): the Qwen chat turn with thinking off,
           the shim's system text, "record: <state, compact JSON> / question: <instructions> / options: / A) name:
           desc ...", the "Answer:" prefill; tokenized as ONE string with the chat format's special tokens honoured
           (= vLLM /tokenize, add_special_tokens false). Request text that encodes to an extra control token is
           refused (ControlTokenError), never served as markup.
  answer   option k (in the shim's own order, the request's criteria order) is the shim label ``labels[k]``; its
           continuation is the ONE token of " <label>" after "Answer:" (the shim's check_backend probe, re-run here on
           this tokenizer). The option AXIS of the row is canonical (``decide.schema.canonical_options``), each
           canonical option carrying its shim label's token.
  spans    state (evidence; an empty state falls back to the instruction span), the question block (field), one span
           per option line, all from the tokenizer's character offsets.

``LEAN`` (``jrt-h2o-shim:qwen3.5-4b-lean``, rescore only): the shim prompt without the system turn and with the shim's
own head / tail state truncation ("\\n...\\n", ``Shim.truncate_state``) at ``LEAN_STATE_TOKENS`` state tokens.

The shim's serving post-processing (temperature 0.8 on every type, the yes/no floor 0.801) is NOT part of the trained
readout (our cell trains its own calibration); jobs.m2_first_cell.screen reports it as a score-only variant.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import canonical_json, canonical_options, sha256_text

SHIM_ID = "jrt-h2o-shim"
#: tokenization canary (in the identity): scripts on which tokenizer implementations have been seen to differ
CANARY = "Jara (\u091c\u0930\u093e) na\u00efve caf\u00e9 \u2014 \u6771\u4eac \U0001f642 <tag>\u00a0x  \t y\r\nz"
VERSION = "h2o-shim-trainable-1"
#: lean state cap (head + tail halves): the largest cap whose mean prompt over the 231 public JevBench items is <= ~460
#: tokens (456.8 vs 653.6 for the full shim prompt, -30 %; 37 of 231 states truncated; CPU sweep 2026-10-09)
LEAN_STATE_TOKENS = 1472
#: family -> (decide.v3w_generic backbone id: the tokenizer files, lean?)
FAMILIES = {"qwen3.5-0.8b": ("qwen3.5-0.8b", False), "qwen3.5-2b": ("qwen3.5-2b", False),
            "qwen3.5-4b": ("qwen3.5-4b", False), "qwen3.5-4b-lean": ("qwen3.5-4b", True)}
#: template id -> sha256 of ``ShimSpec.identity()`` from the pinned tokenizer files (scripts/check_sizecliff_cpu.py;
#: identical under transformers 5.14.1 and the image's 5.17.0, canary included)
PINNED_SPEC_SHA256: dict[str, str] = {
    "jrt-h2o-shim:qwen3.5-0.8b": "54ffc797cddbc15afc9b46712b3b0bb891d6fbe77e7da4c794dc5ca70d25f832",
    "jrt-h2o-shim:qwen3.5-2b": "97fc6954c0d05de5149fb4f74eed974c268f64efb6b97b42cebfb318cc592896",
    "jrt-h2o-shim:qwen3.5-4b": "f0be274c21511b96cc6c8c8d0411098ab46d0970077a59929b17d7b5293fd7ec",
    "jrt-h2o-shim:qwen3.5-4b-lean": "53556b83410cd067727c0f158cfe9d16ce2ee98b816e0037b5649e2c45826cf9",
}


def template_id(family: str) -> str:
    return f"{SHIM_ID}:{family}"


def is_shim_template(template: Any) -> bool:
    tid = getattr(template, "id", template)
    return isinstance(tid, str) and tid.startswith(SHIM_ID + ":") and tid.split(":", 1)[1] in FAMILIES


@dataclass
class ShimSpec:
    family: str
    tok_dir: Path
    lean: bool = False
    _mod: Any = None
    _c: Any = None
    label_ids: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import third_party.h2o_lightning_shim as H

        from .v3w_generic import BACKBONES, HFEncoder

        self._mod, self._c = H.load_contract()
        if self.lean:
            self._c.system = None
        bb = BACKBONES[FAMILIES[self.family][0]]
        from transformers import AutoTokenizer

        self.enc = HFEncoder(self.tok_dir, name=bb.tokenizer_dir)  # control ids, identity (tokenizer.json sha256)
        # the TOKENIZATION is transformers' own (= vLLM /tokenize, the shim's path): it differs from the raw
        # tokenizers-library encode on some scripts (Devanagari), so the identity carries a canary of it
        self.tok = AutoTokenizer.from_pretrained(str(self.tok_dir))
        self.template_id = template_id(self.family)
        self.labels = list(self._c.labels)
        # Shim.check_backend's probe on this tokenizer: every label is ONE distinct token after " " at the answer slot
        n = len(self.labels)
        probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(n)}}
        text = self._c.question_prompt(self._mod.render_state({"x": 1}, self._c.state_compact), probe)[0]
        base = self.ids(text)
        bad = []
        for lab in self.labels:
            full = self.ids(text + " " + lab)
            if len(full) != len(base) + 1 or full[: len(base)] != base:
                bad.append(lab)
            else:
                self.label_ids[lab] = full[-1]
        if bad or len(set(self.label_ids.values())) != len(self.label_ids):
            raise ValueError(f"{self.template_id}: shim labels not one distinct token at the answer slot: {bad[:8]}")
        self._skeleton_ctrl = self._ctrl(self._c.question_prompt("x", {"type": "noul", "instructions": "x"})[0])

    # -------------------------------------------------------------- tokens
    def _encoding(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """(ids, char offsets): transformers' whole-string tokenization, special tokens honoured, none added."""
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(e["input_ids"]), [tuple(o) for o in e["offset_mapping"]]

    def ids(self, text: str) -> list[int]:
        return self._encoding(text)[0]

    def _ctrl(self, text: str) -> int:
        return sum(t in self.enc.control_ids for t in self.ids(text))

    # -------------------------------------------------------------- prompt
    def state_text(self, state: Any) -> str:
        s = self._mod.render_state(state, self._c.state_compact)
        if self.lean:
            t = self.ids(s)
            if len(t) > LEAN_STATE_TOKENS:
                h = LEAN_STATE_TOKENS // 2
                s = self.tok.decode(t[:h]) + "\n...\n" + self.tok.decode(t[-h:])
        return s

    def render(self, request: dict[str, Any], qid: str) -> tuple[str, list[str], list[str], list[str], str]:
        """(prompt text, shim option names, descriptions, labels, state text) of one question."""
        st = self.state_text(request["state"])
        text, names, descs, labels = self._c.question_prompt(st, request["questions"][qid], qid)
        return text, list(names), list(descs), list(labels), st

    def questions(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Per question (canonical qid order): prompt ids, the CANONICAL option ids and each one's label token."""
        from .schema import canonical_question_ids

        out = []
        for qid in canonical_question_ids(request["questions"]):
            text, names, _, labels, _ = self.render(request, qid)
            lab_of = dict(zip(names, labels))
            opts = [o for o, _ in canonical_options(request["questions"][qid])]
            out.append({"qid": qid, "text": text, "prompt": self.ids(text), "options": opts, "coded": True,
                        "tokens": [self.label_ids[lab_of[o]] for o in opts], "labels": [lab_of[o] for o in opts]})
        return out

    # -------------------------------------------------------------- training / serving row
    def row(self, request: dict[str, Any], qid: str, option_order=None, empty_state: str = "reread_instructions"):
        import bisect

        from decide.provenance import instructions_path, state_path
        from decide.untrusted import ControlTokenError
        from heads.layout import G1V2_LAYOUT, ID_PATH, LayoutRow, declared_instruction_paths, is_empty_state

        q = request["questions"][qid]
        text, names, descs, labels, st = self.render(request, qid)
        ids, offs = self._encoding(text)
        if sum(t in self.enc.control_ids for t in ids) != self._skeleton_ctrl:
            raise ControlTokenError("request text encodes to a control token; refused")
        starts = [a for a, _ in offs]

        def span(a: int, b: int) -> tuple[int, int]:
            lo = bisect.bisect_right(starts, a) - 1
            lo = max(lo, 0)
            while lo < len(offs) and offs[lo][1] <= a:
                lo += 1
            hi = bisect.bisect_left(starts, b)
            return (lo, max(lo, hi))

        u0 = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        rec = "record: "
        if not text.startswith(rec, u0):
            raise ValueError("shim prompt: the user turn does not open with 'record: '")
        s0 = u0 + len(rec)
        s1 = s0 + len(st)
        if text[s0:s1] != st:
            raise ValueError("shim prompt: state text not found verbatim")
        qhead = "\nquestion: "
        if not text.startswith(qhead, s1):
            raise ValueError("shim prompt: no question line after the state")
        i0 = s1 + len(qhead)
        o_hdr = text.index("\noptions:\n", i0)
        i1 = o_hdr
        pos = o_hdr + len("\noptions:\n")
        line_span: dict[str, tuple[int, int]] = {}
        for lab, nm, ds in zip(labels, names, descs):
            line = f"{lab}) {nm}: {ds}"
            if text.startswith(line, pos):
                end = pos + len(line)
            elif text.startswith(line.rstrip(), pos):  # the chat template strips the user content's tail
                end = pos + len(line.rstrip())
            else:
                raise ValueError("shim prompt: option line not found in order")
            line_span[nm] = span(pos, end)
            pos = end + 1
        field_span = span(s1, pos - 1)
        state_span, evidence = span(s0, s1), "state"
        if is_empty_state(request["state"]) or state_span[1] <= state_span[0]:
            if empty_state != "reread_instructions":
                raise ValueError("state is empty (no evidence content); heads need at least one evidence token")
            state_span, evidence = span(i0, i1), "reread_instructions"
        opts = [o for o, _ in canonical_options(q)]
        perm = list(range(len(opts))) if option_order is None else list(option_order)
        if sorted(perm) != list(range(len(opts))):
            raise ValueError("option_order must be a permutation of range(n_options)")
        if q["type"] == "score" and perm != sorted(perm):
            raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
        lab_of = dict(zip(names, labels))
        if set(lab_of) != set(opts):
            raise ValueError("shim option names != the canonical options")
        st_sp, in_sp = span(s0, s1), span(i0, i1)
        caller = [st_sp, in_sp] + [line_span[o] for o in opts]
        fields = [(st_sp[0], st_sp[1], state_path()), (in_sp[0], in_sp[1], instructions_path(qid))] + \
                 [(line_span[o][0], line_span[o][1], ID_PATH) for o in opts]
        return LayoutRow(
            qid=qid, qtype=q["type"], option_ids=[opts[k] for k in perm], ids=ids, state_span=state_span,
            field_span=field_span, option_spans=[line_span[opts[k]] for k in perm], option_seqs=[],
            continuations=[[self.label_ids[lab_of[opts[k]]]] for k in perm], layout=G1V2_LAYOUT,
            meta={"template": self.template_id, "perm": perm, "evidence": evidence,
                  "untrusted_spans": [st_sp], "caller_spans": [c for c in caller if c[1] > c[0]],
                  "field_spans": [f for f in fields if f[1] > f[0]], "cont_spans": [[] for _ in perm],
                  "option_seq_spans": [], "trusted_paths": declared_instruction_paths(request), "bos": None,
                  "coded": True, "words": False, "catalog_chars": None,
                  "shim_labels": [lab_of[opts[k]] for k in perm]})

    # -------------------------------------------------------------- identity
    def identity(self) -> dict[str, str]:
        import third_party.h2o_lightning_shim as H

        c = self._c
        value = {"id": self.template_id, "version": VERSION, "shim": f"{H.REPO}@{H.TAG}",
                 "shim_files_sha256": dict(H.FILES_SHA256), "system": c.system, "layout": c.layout,
                 "prefill": c.prefill, "state_compact": c.state_compact, "chat": c.chat,
                 "labels_sha256": sha256_text(canonical_json(self.labels)), "label_ids": self.label_ids,
                 "lean": {"system": None, "state_tokens": LEAN_STATE_TOKENS} if self.lean else None,
                 "tokenizer": self.enc.identity(), "tokenization": "transformers AutoTokenizer, whole string",
                 "canary_ids": self.ids(CANARY)}
        return {"id": self.template_id, "sha256": sha256_text(canonical_json(value))}


# ------------------------------------------------------------------------------------------- binding (as v3w_generic)
_SPECS: dict[str, ShimSpec] = {}


def _env_key(tid: str) -> str:
    return "SSDM_SHIM_TOKDIR_" + "".join(c if c.isalnum() else "_" for c in tid.split(":", 1)[1]).upper()


def make_spec(tid: str, tokenizer_dir: str | Path | None = None) -> ShimSpec:
    from .v3w_generic import BACKBONES, DEFAULT_TOKENIZERS

    fam = tid.split(":", 1)[1]
    bb_id, lean = FAMILIES[fam]
    d = Path(tokenizer_dir) if tokenizer_dir else DEFAULT_TOKENIZERS / BACKBONES[bb_id].tokenizer_dir
    return ShimSpec(fam, d, lean)


def bind_template(template: Any, tokenizer_dir: str | Path | None = None) -> ShimSpec:
    """Build the spec from ``tokenizer_dir`` (default: the local tokenizers root) and check its identity against
    ``PINNED_SPEC_SHA256`` (a different tokenizer / shim fails closed)."""
    tid = getattr(template, "id", template)
    if not is_shim_template(tid):
        raise KeyError(f"{tid!r} is not a shim template")
    spec = make_spec(tid, tokenizer_dir)
    got, want = spec.identity()["sha256"], PINNED_SPEC_SHA256.get(tid)
    if want is None:
        raise ValueError(f"{tid}: no pinned spec identity; refusing an unpinned readout")
    if got != want:
        raise ValueError(f"{tid}: spec identity {got[:16]} != pinned {want[:16]}")
    _SPECS[tid] = spec
    if tokenizer_dir:
        os.environ[_env_key(tid)] = str(Path(tokenizer_dir).resolve())
    return spec


def spec_for_template(template: Any) -> ShimSpec:
    tid = getattr(template, "id", template)
    if tid in _SPECS:
        return _SPECS[tid]
    return bind_template(tid, os.environ.get(_env_key(tid)))


def template_obj(tid: str):
    from .prompt import Template

    return Template(id=tid, head="", tail="", terminator="\n")
