"""Tokenizer- and chat-format-generic v3w readout: ``make_v3w(backbone_id)`` gives any backbone the v3w readout
(decide.prompt_g1v3 v3w: v3 code key, word ids by their unique first token) and its matching TRAINING row builder.

The v3w BODY (schema -> state -> question re-read, code key, "Reply with exactly one option code.") is the one
``render_g1v3`` renders, unchanged and backbone-invariant (blank lines collapsed, as v3w always does). What is
backbone-specific is factored out:

  (a) code table: ``build_code_table`` derives the contiguous single-token code set for a tokenizer (A..Z, then AA..ZZ
      lexicographic minus WORDLIKE English words / abbreviations), keeping a code only if " <code>" is exactly ONE new
      token after the format's own answer prefix (``ChatFormat.answer_ctx``, whole-string tokenization) and its token
      is not already taken; the no-space form ("ANSWER:<code>") is recorded per code and, with ``require_nospace``
      also required. RWKV7-G1k keeps today's frozen table (decide/data/rwkv_g1_v3_codes.json), which this builder
      re-derives exactly (tests/test_v3w_generic.py, scripts/check_v3w_generic.py).
  (b) word ids: ``word_ids_ok`` (non-opaque ids, first token of " <id>" after the answer prefix unique among the
      question's options) with the format's own tokenizer; otherwise codes, as v3w.
  (c) chat formats: ``RWKV_G1`` is the G1 format exactly as today (token 0, "<think></think>\\n", no "\\n\\n" in a
      turn). HF backbones get ``format_from_chat_template``: the model's OWN chat template (tokenizer_config.json /
      chat_template.jinja, read-only, rendered in jinja2's immutable sandbox as transformers does) with sentinel
      system / user messages, ``add_generation_prompt=True`` and ``enable_thinking=False`` where the template
      supports it; head = text before the user sentinel, tail = text after it + "ANSWER:". A generation prompt that
      opens "<think>" with no no-think switch (QwQ) is closed with "\\n</think>\\n\\n" (Qwen3's own empty-think
      form) and recorded as ``think_closed``. Templateless base models get ``plain_format`` (the G1 turn layout
      without the think block; "\\n\\n" is its turn separator, so the body's blank-line collapse applies). The whole
      prompt string is tokenized once; control ids are allowed only where the format itself puts them.
  (d) ``make_v3w(backbone_id)`` -> ``V3W``: ``questions(request)`` (the readout's per-question prompt ids, option
      order, scored token ids, word-id flag, code-key label width: G1V3Backend(mode="words")._questions for any
      backbone) and ``train_row(request, qid)`` (heads.layout layout-g1v2 row built from the same segments: ids ==
      the readout's prompt ids, continuations == the scored tokens; scripts/check_v3w_generic.py proves it).

For ``rwkv7-g1k`` the factory reproduces today's v3w byte for byte: same text, same [0] + whole-string ids, same
code table, same template id (jrt-rwkv-g1-v3w) and its train rows are heads.layout's existing v3w rows.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .prompt import G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE
from .prompt_g1v3 import (CATALOG_CHARS_LADDER, G1V3W_ID, V3_MAX_K, WORDLIKE, code_candidates, load_v3_codes,
                          question_codes, render_g1v3, wants_catalog, word_ids_ok, identity_order)
from .schema import canonical_json, canonical_options, canonical_question_ids, sha256_text

GENERIC_ID = "jrt-v3w-generic"
GENERIC_VERSION = "v3w-generic-1"
ANSWER = "ANSWER:"
RWKV_ANSWER_CTX = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"  # decide.backends.rwkv_g1_v3.ANSWER_CTX
TERMINATOR = G1V2_TEMPLATE.terminator  # fallback (k > code table) multi-token continuations: " <id>\n"
_G1_HEAD = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
_G1_TAIL = G1V2_TEMPLATE.tail
SYS_SENTINEL = "@@SSDM_SYSTEM_8f3a@@"
USR_SENTINEL = "@@SSDM_USER_8f3a@@"
DEFAULT_TOKENIZERS = Path(os.environ.get("SSDM_TOKENIZERS", r"C:\Users\apumu\research\CHLabs\ssdm-data\tokenizers"))


# ----------------------------------------------------------------------------------------------------- chat formats
@dataclass(frozen=True)
class ChatFormat:
    """head (``{system}`` placeholder, before the v3w body), tail (after it; ends with "ANSWER:"), ids prepended to
    the whole-string tokens (``bos``) and the answer prefix used for the single-token proofs (``answer_ctx``)."""

    family: str
    head: str
    tail: str
    bos: tuple[int, ...]
    answer_ctx: str
    system: str = G1V2_SYSTEM_PROMPT
    source: str = "builtin"  # builtin | chat_template | plain
    template_sha256: str | None = None
    no_think: str | None = None  # enable_thinking=False | think_closed | builtin | None (no think block)
    system_role: bool = True  # False: the template refused a system message; the system text leads the user turn

    def head_text(self) -> str:
        return self.head.replace("{system}", self.system)

    def full(self, body: str) -> str:
        return self.head_text() + body + self.tail

    def identity(self) -> dict[str, Any]:
        return {"family": self.family, "head": self.head, "tail": self.tail, "bos": list(self.bos),
                "answer_ctx": self.answer_ctx, "system": self.system, "source": self.source,
                "template_sha256": self.template_sha256, "no_think": self.no_think, "system_role": self.system_role}


#: the RWKV7 G1 format exactly as decide.prompt.G1V2_TEMPLATE / decide.backends.rwkv_g1_v3 use it today
RWKV_G1 = ChatFormat(family="rwkv7-g1", head=G1V2_TEMPLATE.head, tail=G1V2_TEMPLATE.tail, bos=(0,),
                     answer_ctx=RWKV_ANSWER_CTX, source="builtin", no_think="builtin")


def _generation_tag():
    """transformers' ``{% generation %}...{% endgeneration %}`` block (its assistant-token mask marker, used by the
    LFM2 templates): rendered as its body, as transformers does when no mask is requested."""
    from jinja2 import nodes
    from jinja2.ext import Extension

    class GenerationBlock(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_body", []), [], [], body).set_lineno(lineno)

        def _body(self, caller):
            return caller()

    return GenerationBlock


def _sandbox_render(template: str, **kw) -> str:
    """Render a chat template the way transformers does (jinja2 ImmutableSandboxedEnvironment, trim/lstrip blocks):
    the template is data, never imported or executed as Python."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(f"chat template: {msg}")

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols, _generation_tag()])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "1970-01-01"
    return env.from_string(template).render(**kw)


def _token_str(v: Any) -> str | None:
    return v.get("content") if isinstance(v, dict) else v


def read_chat_template(tok_dir: Path) -> str | None:
    """The model's own chat template: chat_template.jinja if present, else tokenizer_config.json's (string, or the
    entry named "default" of a list)."""
    p = Path(tok_dir) / "chat_template.jinja"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    cfg = json.loads((Path(tok_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    ct = cfg.get("chat_template")
    if isinstance(ct, list):
        ct = next((c.get("template") for c in ct if isinstance(c, dict) and c.get("name") == "default"), None)
    return ct if isinstance(ct, str) and ct.strip() else None


def format_from_chat_template(family: str, tok_dir: Path, system: str = G1V2_SYSTEM_PROMPT) -> ChatFormat:
    cfg = json.loads((Path(tok_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    ct = read_chat_template(tok_dir)
    if ct is None:
        raise ValueError(f"{tok_dir}: no chat template")
    kw: dict[str, Any] = {"add_generation_prompt": True, "bos_token": _token_str(cfg.get("bos_token")) or "",
                          "eos_token": _token_str(cfg.get("eos_token")) or "", "tools": None, "documents": None}
    no_think = None
    if "enable_thinking" in ct:
        kw["enable_thinking"] = False
        no_think = "enable_thinking=False"
    system_role = True
    try:
        r = _sandbox_render(ct, messages=[{"role": "system", "content": SYS_SENTINEL},
                                          {"role": "user", "content": USR_SENTINEL}], **kw)
        if r.count(SYS_SENTINEL) != 1:
            raise ValueError("system sentinel not rendered once")
    except ValueError:
        system_role = False
        r = _sandbox_render(ct, messages=[{"role": "user", "content": SYS_SENTINEL + "\n" + USR_SENTINEL}], **kw)
    if r.count(USR_SENTINEL) != 1 or r.count(SYS_SENTINEL) != 1 or r.index(SYS_SENTINEL) > r.index(USR_SENTINEL):
        raise ValueError(f"{family}: chat template does not render the system and user messages once, in order")
    i = r.index(USR_SENTINEL)
    head, tail = r[:i].replace(SYS_SENTINEL, "{system}"), r[i + len(USR_SENTINEL):]
    stripped = tail.rstrip()
    if no_think is None and stripped.endswith("<think>"):  # forced-think generation prompt (QwQ): close it empty
        tail = stripped + "\n\n</think>\n\n"
        no_think = "think_closed"
    tail += ANSWER
    fmt = ChatFormat(family=family, head=head, tail=tail, bos=(), answer_ctx="", system=system,
                     source="chat_template", template_sha256=hashlib.sha256(ct.encode("utf-8")).hexdigest(),
                     no_think=no_think, system_role=system_role)
    return ChatFormat(**{**fmt.__dict__, "answer_ctx": fmt.full("Reply with exactly one option code.")})


def plain_format(family: str, bos: tuple[int, ...] = (), system: str = G1V2_SYSTEM_PROMPT) -> ChatFormat:
    """Templateless base model: the G1 turn layout without a think block ("\\n\\n" separates turns)."""
    fmt = ChatFormat(family=family, head="System: {system}\n\nUser: ", tail="\n\nAssistant: " + ANSWER, bos=bos,
                     answer_ctx="", system=system, source="plain", no_think=None)
    return ChatFormat(**{**fmt.__dict__, "answer_ctx": fmt.full("Reply with exactly one option code.")})


def check_template_passthrough(fmt: ChatFormat, tok_dir: Path, body: str) -> bool:
    """True when the chat template renders ``body`` as user content verbatim (head + body + tail - "ANSWER:")."""
    if fmt.source != "chat_template":
        return True
    ct = read_chat_template(tok_dir)
    cfg = json.loads((Path(tok_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    kw: dict[str, Any] = {"add_generation_prompt": True, "bos_token": _token_str(cfg.get("bos_token")) or "",
                          "eos_token": _token_str(cfg.get("eos_token")) or "", "tools": None, "documents": None}
    if fmt.no_think == "enable_thinking=False":
        kw["enable_thinking"] = False
    msgs = ([{"role": "system", "content": fmt.system}, {"role": "user", "content": body}] if fmt.system_role
            else [{"role": "user", "content": fmt.system + "\n" + body}])
    r = _sandbox_render(ct, messages=msgs, **kw)
    want = fmt.full(body)[: -len(ANSWER)]
    if fmt.no_think == "think_closed":
        want = want[: -len("\n\n</think>\n\n")]
        r = r.rstrip()
        want = want.rstrip()
    return r == want


# --------------------------------------------------------------------------------------------------------- encoders
class HFEncoder:
    """Whole-string encoder over a tokenizer.json (``tokenizers`` library, data only; no repo code):
    ``enc(text, trusted)`` -> ids without auto-added specials; untrusted text that yields a control (added-token) id
    is refused. ``encode_pieces`` gives ids + byte offsets (heads.layout maps tokens back to char spans)."""

    def __init__(self, tok_dir: Path) -> None:
        from tokenizers import Tokenizer

        self.dir = Path(tok_dir)
        path = self.dir / "tokenizer.json"
        self.tok = Tokenizer.from_file(str(path))
        self.tokenizer_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding="utf-8"))
        # control ids: special added tokens, and non-special added tokens that are markup (Qwen "<think>"), not the
        # whitespace-run tokens GPT-NeoX vocabularies add (xLSTM) -- those are ordinary text
        self.control_ids = frozenset(int(t["id"]) for t in data.get("added_tokens", [])
                                     if t.get("special") or t["content"].strip())
        self.added = {int(t["id"]): t["content"] for t in data.get("added_tokens", [])}
        self.special_strings = tuple(t["content"] for t in data.get("added_tokens", []) if t.get("special"))
        cfg = json.loads((self.dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.bos_token = _token_str(cfg.get("bos_token"))
        self.add_bos = bool(cfg.get("add_bos_token"))
        self.bos_id = self.tok.token_to_id(self.bos_token) if self.bos_token else None

    def identity(self) -> dict[str, Any]:
        return {"kind": "hf-tokenizers-whole-string", "dir": self.dir.name, "tokenizer_sha256": self.tokenizer_sha256,
                "untrusted": "fail closed on added-token ids"}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        from decide.untrusted import ControlTokenError

        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not trusted and self.control_ids.intersection(ids):
            raise ControlTokenError("untrusted text encodes to a control (added) token")
        return ids

    def encode_pieces(self, pieces: list[tuple[str, bool]]) -> tuple[list[int], list[int]]:
        """(ids, byte starts) of consecutive text pieces; a piece with honor=False is encoded with special added
        tokens matched as PLAIN TEXT (``encode_special_tokens``), so request text can never become a control token.
        ``tokenizers`` splits a string at added-token matches before the model runs, so encoding the pieces between
        the format's own special tokens separately IS the whole-string encoding whenever the request text holds no
        special-token string (scripts/check_v3w_generic.py counts it)."""
        ids: list[int] = []
        starts: list[int] = []
        base = 0
        for text, honor in pieces:
            if not text:
                continue
            self.tok.encode_special_tokens = not honor
            try:
                enc = self.tok.encode(text, add_special_tokens=False)
            finally:
                self.tok.encode_special_tokens = False
            byte_at = [0]
            for ch in text:
                byte_at.append(byte_at[-1] + len(ch.encode("utf-8")))
            last = base
            for a, _ in enc.offsets:
                last = max(last, base + byte_at[a])
                starts.append(last)
            ids += enc.ids
            base += byte_at[-1]
        return ids, starts

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)


# ----------------------------------------------------------------------------------------------------- code tables
def build_code_table(encode: Callable[[str, bool], list[int]], answer_ctx: str, max_k: int = V3_MAX_K,
                     require_nospace: bool = False, banned: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
    """[{code, token_id, nospace_token_id}] for the first ``max_k`` code candidates (A..Z, AA..ZZ minus WORDLIKE) whose
    " <code>" is exactly one new token after ``answer_ctx`` (whole-string tokenization), distinct from every earlier
    code's token and not in ``banned``. ``nospace_token_id``: the single token of "<code>" right after the context,
    else None (required when ``require_nospace``). For RWKV7-G1k (RWKV_ANSWER_CTX) the first 255 are today's
    decide/data/rwkv_g1_v3_codes.json."""
    base = encode(answer_ctx, True)
    n = len(base)
    out: list[dict[str, Any]] = []
    taken: set[int] = set()
    for c in code_candidates():
        ids = encode(answer_ctx + " " + c, True)
        if not (ids[:n] == base and len(ids) == n + 1) or ids[-1] in taken or ids[-1] in banned:
            continue
        ns = encode(answer_ctx + c, True)
        ns_tok = ns[-1] if ns[:n] == base and len(ns) == n + 1 else None
        if require_nospace and ns_tok is None:
            continue
        out.append({"code": c, "token_id": ids[-1], "nospace_token_id": ns_tok})
        taken.add(ids[-1])
        if len(out) == max_k:
            break
    return out


# ------------------------------------------------------------------------------------------------- backbone registry
@dataclass(frozen=True)
class Backbone:
    id: str
    family: str
    tokenizer_dir: str  # directory name under the tokenizers root
    fmt: str  # "rwkv_g1" | "chat_template" | "plain"
    max_context: int
    hf_repo: str | None = None
    attention_free: bool = True
    note: str = ""


BACKBONES: dict[str, Backbone] = {b.id: b for b in [
    Backbone("rwkv7-g1k-13.3b", "rwkv7-g1", "RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
             "rwkv_g1", 25600, "RWKV/RWKV7-G1k-13.3B-20260930"),
    Backbone("qwen3.5-9b", "qwen3.5", "Qwen--Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a", "chat_template",
             262144, "Qwen/Qwen3.5-9B", False, "Q9-AF v2 (all attention layers -> GDN) uses this tokenizer + template"),
    Backbone("qwerky-72b", "qwerky", "featherless-ai--Qwerky-72B@6dfe3c27be6dddf9e1ff1bae94c6c4b915696bb6",
             "chat_template", 32768, "featherless-ai/Qwerky-72B"),
    Backbone("qwerky-qwq-32b", "qwerky-qwq", "featherless-ai--Qwerky-QwQ-32B@6801e98e72a3e03b4611cdd816a0532411f6e1e0",
             "chat_template", 32768, "featherless-ai/Qwerky-QwQ-32B"),
    Backbone("qrwkv6-32b", "qrwkv6", "recursal--QRWKV6-32B-Instruct-Preview-v0.1@2da855ca372cefab920716eca30f11ed65d6c708",
             "chat_template", 16384, "recursal/QRWKV6-32B-Instruct-Preview-v0.1"),
    Backbone("falcon-mamba-7b-instruct", "falcon-mamba", "tiiuae--falcon-mamba-7b-instruct@b250fc9399d14f56aca18e9ea70bbfb1f73479eb",
             "chat_template", 8192, "tiiuae/falcon-mamba-7b-instruct"),
    Backbone("mamba-codestral-7b", "mamba-codestral", "mistralai--Mamba-Codestral-7B-v0.1@4f086c08c1e0f07bdc50ca25125dbbf7475d21da",
             "plain", 32768, "mistralai/Mamba-Codestral-7B-v0.1", note="no chat template in the HF repo"),
    Backbone("xlstm-7b", "xlstm", "NX-AI--xLSTM-7b@9dc507bd0939cf372a4a4f667335651d8e49dddb", "plain", 8192,
             "NX-AI/xLSTM-7b", note="no chat template in the HF repo"),
    # jobs/jev_host_scan (JevBench host scan, 2026-10-09): small hosts read stock, without the prefix fork
    Backbone("rwkv7-g1i-2.9b", "rwkv7-g1", "RWKV--RWKV7-2.9B-20260805@aae64511841dd32f5235f6bcc7a931c6ef806e6c",
             "rwkv_g1", 16384, "RWKV/RWKV7-2.9B-20260805", note="G1i; tokenizer.json identical to G1k's"),
    Backbone("qwen3.5-4b", "qwen3.5", "Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "chat_template",
             262144, "Qwen/Qwen3.5-4B", False),
    Backbone("qwen3.5-2b", "qwen3.5", "Qwen--Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc", "chat_template",
             262144, "Qwen/Qwen3.5-2B", False),
    Backbone("lfm2.5-2.6b", "lfm2.5", "LiquidAI--LFM2.5-2.6B@654f9463ce32b05d0429d76fe1f580b27d4c1ac0", "chat_template",
             131072, "LiquidAI/LFM2.5-2.6B", False, "hybrid short-conv + attention"),
    Backbone("gemma-4-e4b-it", "gemma4", "google--gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2",
             "chat_template", 131072, "google/gemma-4-E4B-it", False),
    Backbone("nemotron-3-nano-4b", "nemotron-h", "nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
             "chat_template", 262144, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", False, "hybrid Mamba2 + attention"),
]}


# ------------------------------------------------------------------------------------------------------- the readout
@dataclass
class V3W:
    """v3w readout + matching training rows for one backbone (see the module docstring)."""

    backbone: Backbone
    fmt: ChatFormat
    encode: Any  # encode(text, trusted) -> ids; heads.layout encoder (token_bytes or byte_starts)
    codes: list[dict[str, Any]]
    budget: int
    template_id: str
    control_ids: frozenset[int]
    _first: dict[str, int | None] = field(default_factory=dict)
    _ctx_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.table = [e["code"] for e in self.codes]
        self.code_token = {e["code"]: e["token_id"] for e in self.codes}
        self._ctx_ids = self.encode_prompt(self.fmt.answer_ctx)
        self._head_tail_ctrl = (sum(t in self.control_ids for t in self.encode(self.fmt.head_text(), True))
                                + sum(t in self.control_ids for t in self.encode(self.fmt.tail, True)))
        for e in self.codes:  # F4 proof on this tokenizer (as G1V3Backend.__init__)
            if self.encode_prompt(self.fmt.answer_ctx, " " + e["code"]) != self._ctx_ids + [e["token_id"]]:
                raise ValueError(f"{self.backbone.id}: code {e['code']!r} is not one token after the answer prefix")

    @property
    def is_legacy_rwkv(self) -> bool:
        return self.fmt is RWKV_G1

    # -------------------------------------------------------------- prompts
    def body(self, request: dict[str, Any], qid: str, catalog_chars: int | None, words: frozenset[str]) -> str:
        text = render_g1v3(request, qid, self.table, catalog_chars, False, True, words)
        return text[len(_G1_HEAD):len(text) - len(_G1_TAIL)]

    def render(self, request: dict[str, Any], qid: str, catalog_chars: int | None, words: frozenset[str]) -> str:
        if self.is_legacy_rwkv:
            return render_g1v3(request, qid, self.table, catalog_chars, False, True, words)
        return self.fmt.full(self.body(request, qid, catalog_chars, words))

    def body_ids(self, text: str) -> list[int]:
        """Whole-string tokens of a full prompt text (no ``bos``), control ids refused outside the format's own."""
        from .backends.base import BackendUnsupported

        if self.is_legacy_rwkv:
            ids = list(self.encode(text, True))
            bad = 0 in ids
        else:
            ids = self.encode_prompt(text)
            bad = sum(t in self.control_ids for t in ids) != self._head_tail_ctrl
        if bad:
            raise BackendUnsupported("control_tokens_in_input", "request text encodes to a control token; refused")
        return ids

    def _pieces(self, text: str, suffix: str = "") -> list[tuple[str, bool]]:
        """``text`` = head + body + tail: the format's own special tokens (honored) and the text between them; the
        span from the head's last special to the tail's first one (request body included) is one piece with special
        tokens matched as plain text. ``suffix`` (an answer continuation) extends the last piece."""
        import re

        head, tail = self.fmt.head_text(), self.fmt.tail
        if not (text.startswith(head) and text.endswith(tail)) or len(text) < len(head) + len(tail):
            raise ValueError("prompt text does not carry the format's head and tail")
        sp = [s for s in getattr(self.encode, "special_strings", ()) if s and (s in head or s in tail)]
        rx = re.compile("(" + "|".join(re.escape(s) for s in sorted(sp, key=len, reverse=True)) + ")") if sp else None

        def split(t: str) -> list[tuple[str, bool]]:
            if rx is None:
                return [(t, False)]
            return [(x, i % 2 == 1) for i, x in enumerate(rx.split(t)) if x]

        hp, tp = split(head), split(tail)
        h_rest = hp.pop()[0] if hp and not hp[-1][1] else ""
        t_lead = tp.pop(0)[0] if tp and not tp[0][1] else ""
        mid = h_rest + text[len(head):len(text) - len(tail)] + t_lead
        pieces = hp + [(mid, False)] + tp
        if suffix:
            if pieces[-1][1]:
                pieces.append((suffix, False))
            else:
                pieces[-1] = (pieces[-1][0] + suffix, False)
        return pieces

    def encode_prompt(self, text: str, suffix: str = "") -> list[int]:
        """Tokens of a head + body + tail prompt (+ ``suffix``): the whole string at once, special tokens only where
        the format puts them (HF encoders); the plain whole-string encoding otherwise."""
        if self.is_legacy_rwkv or not hasattr(self.encode, "encode_pieces"):
            return list(self.encode(text + suffix, True))
        return self.encode.encode_pieces(self._pieces(text, suffix))[0]

    def byte_starts(self, text: str, ids: list[int]) -> list[int]:
        """Byte offset of each token of ``encode_prompt(text)`` (heads.layout span mapping)."""
        if self.is_legacy_rwkv or not hasattr(self.encode, "encode_pieces"):
            out, pos = [], 0
            for t in ids:
                out.append(pos)
                pos += len(self.encode.token_bytes(t))
            if pos != len(text.encode("utf-8")):
                raise ValueError("token bytes do not tile the prompt")
            return out
        got, starts = self.encode.encode_pieces(self._pieces(text))
        if got != list(ids):
            raise ValueError("byte_starts: ids are not encode_prompt(text)")
        return starts

    def prompt_ids(self, text: str) -> list[int]:
        return list(self.fmt.bos) + self.body_ids(text)

    def first_token(self, oid: str) -> int | None:
        """The answer token of " <oid>" right after the answer prefix (None if the tokenization does not extend the
        prefix's, or the token is a control id)."""
        if oid not in self._first:
            ids = self.encode_prompt(self.fmt.answer_ctx, " " + oid)
            n = len(self._ctx_ids)
            bad = {0} if self.is_legacy_rwkv else self.control_ids
            self._first[oid] = ids[n] if len(ids) > n and ids[:n] == self._ctx_ids and ids[n] not in bad else None
        return self._first[oid]

    def word_qids(self, request: dict[str, Any]) -> frozenset[str]:
        qs = request["questions"]
        return frozenset(k for k in canonical_question_ids(qs) if word_ids_ok(qs[k], self.first_token))

    def choose(self, request: dict[str, Any], qid: str, words: frozenset[str]):
        """(text, prompt ids, label chars) of the longest-label prompt within the budget (G1V3Backend._render)."""
        rq = request["questions"][qid]
        ladder = CATALOG_CHARS_LADDER if (qid not in words and wants_catalog(rq, self.table)) else (None,)
        text, ids = "", []
        for n in ladder:
            text = self.render(request, qid, n, words)
            ids = self.prompt_ids(text)
            if len(ids) <= self.budget:
                return text, ids, n
        return text, ids, None

    def questions(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Per question: the readout's prompt ids, option order and scored tokens (G1V3Backend(mode="words"))."""
        out = []
        words = self.word_qids(request)
        for qid in canonical_question_ids(request["questions"]):
            text, prompt, n = self.choose(request, qid, words)
            rq = request["questions"][qid]
            opts = [o for o, _ in canonical_options(rq)]
            coded = len(opts) <= len(self.table)
            q = {"qid": qid, "text": text, "prompt": prompt, "options": opts, "coded": coded, "catalog_chars": n,
                 "catalog_wanted": qid not in words and wants_catalog(rq, self.table), "words": qid in words}
            if qid in words:
                q["coded"], q["codes"] = True, None
                q["tokens"] = [self.first_token(o) for o in opts]
            elif coded:
                q["codes"] = question_codes(rq, identity_order(rq), self.table)
                q["tokens"] = [self.code_token[c] for c in q["codes"]]
            else:
                q["conts"] = [list(self.encode(" " + o + TERMINATOR, True)) for o in opts]
            out.append(q)
        return out

    # -------------------------------------------------------------- training
    def train_row(self, request: dict[str, Any], qid: str, option_order=None, empty_state: str = "reread_instructions"):
        """heads.layout row (layout g1v2) whose ids are the readout's prompt ids and whose continuations are the
        readout's scored tokens. RWKV7-G1k: exactly heads.layout.build_row(template=jrt-rwkv-g1-v3w)."""
        from heads.layout import build_row, build_row_v3w_generic

        if self.is_legacy_rwkv:
            return build_row(request, qid, self.encode, template=G1V3W_ID, layout="g1v2", option_order=option_order,
                             empty_state=empty_state)
        return build_row_v3w_generic(request, qid, self, option_order, empty_state)

    # -------------------------------------------------------------- identity
    def identity(self) -> dict[str, str]:
        if self.is_legacy_rwkv:
            from .backends.rwkv_g1_v3 import renderer_identity

            return renderer_identity("words")
        value = {"id": self.template_id, "version": GENERIC_VERSION, "format": self.fmt.identity(),
                 "codes": self.codes, "wordlike_excluded": sorted(WORDLIKE), "budget": self.budget,
                 "catalog": list(CATALOG_CHARS_LADDER), "word_ids": "non-opaque ids, unique first token after ANSWER:",
                 "tokenizer": getattr(self.encode, "identity", lambda: None)()}
        return {"id": self.template_id, "sha256": sha256_text(canonical_json(value))}


def _rwkv_encoder(tok_dir: Path):
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
    from train.run_cell import RwkvSegmentEncoder

    tok = Rwkv7Tokenizer.from_pretrained(str(tok_dir))
    return RwkvSegmentEncoder(tok, set(tok.all_special_ids))


def make_v3w(backbone_id: str, tokenizers_root: str | Path | None = None, budget: int | None = None,
             require_nospace: bool | None = None) -> V3W:
    """The v3w readout + training-row builder for ``backbone_id`` (a key of BACKBONES). ``budget``: the readout's
    prompt-token budget for the code-key ladder (default min(max_context, adapter max_input_tokens) - 1, as
    G1V3Backend). ``require_nospace`` (default False; the frozen RWKV table never uses it) also requires
    each code to be one token WITHOUT the leading space. Off by default: in Qwen-family BPE ":A" is one pretoken, so
    "ANSWER:A" re-tokenizes the colon and A..F would be dropped; the scored form is " <code>" (always proved), the
    no-space form is recorded per code (``nospace_token_id``) and reported by scripts/check_v3w_generic.py."""
    from .adapter import DEFAULT_LIMITS

    bb = BACKBONES[backbone_id]
    tok_dir = Path(tokenizers_root or DEFAULT_TOKENIZERS) / bb.tokenizer_dir
    if budget is None:
        budget = min(bb.max_context, DEFAULT_LIMITS["max_input_tokens"]) - 1
    if bb.fmt == "rwkv_g1":
        enc = _rwkv_encoder(tok_dir)
        entries, _ = load_v3_codes()
        codes = [{"code": e["code"], "token_id": e["token_id"]} for e in entries]
        return V3W(bb, RWKV_G1, enc, codes, budget, G1V3W_ID, frozenset({0}))
    enc = HFEncoder(tok_dir)
    if bb.fmt == "chat_template":
        fmt = format_from_chat_template(bb.family, tok_dir)
    else:
        fmt = plain_format(bb.family, (enc.bos_id,) if enc.add_bos and enc.bos_id is not None else ())
    codes = build_code_table(enc, fmt.answer_ctx, V3_MAX_K, bool(require_nospace), enc.control_ids)
    return V3W(bb, fmt, enc, codes, budget, f"{GENERIC_ID}:{bb.family}", enc.control_ids)
