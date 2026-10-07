"""Experimental C1 token-ID readout component; complete request rendering is blocked.

This explicit opt-in component never imports or calls a model. It maps already
computed vocabulary logits or exactly one token ID to original semantic IDs.
It is not a usable decide(request) backend until the separate renderer contract
is implemented and reviewed; plan/score fail closed rather than inventing it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from ..calibration import softmax
from ..schema import canonical_options, validate_legacy, wire_answer
from .base import BackendUnsupported

CODEBOOK_ID = "rwkv-g1-filtered-codebook-v1"
MAX_K, VOCAB_SIZE = 255, 65536
TOKENIZER_SHA256 = "1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf"
TOKENIZER_SOURCE_SHA256 = "58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65"
CODE_STRINGS_SHA256 = "2878be9488928b108ff83d0587e41cc25f433a023407a3040e459a274db5d0df"
TABLE_SHA256 = "7226adf44b9529726b2f44e66b60257c92a50126a3f30131caad6ab0be231466"
PROOF_CONTEXT = "User: Choose the fixed code.\n\nAssistant: <think></think>\nANSWER:"
CONTEXT_SHA256 = "597cc204a1db6d8bbe9c4c2e3e2948084cd95695b6bec5598a0565c1e6d9c165"
TABLE_PATH = Path(__file__).resolve().parents[1] / "data/rwkv_g1_filtered_v1.json"


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class TokenCode:
    code: str
    token_id: int
    source_ordinal: int


def load_frozen_table(path=TABLE_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema": "ssdm.rwkv-g1-filtered-codebook.v1", "codebook_id": CODEBOOK_ID,
                "max_k": MAX_K, "source_candidates": 702, "eligible_candidates": 498,
                "tokenizer_sha256": TOKENIZER_SHA256, "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256,
                "proof_context": PROOF_CONTEXT, "proof_context_sha256": CONTEXT_SHA256,
                "codes_sha256": CODE_STRINGS_SHA256, "code_token_table_sha256": TABLE_SHA256}
    if any(data.get(k) != v for k, v in required.items()):
        raise ValueError("frozen codebook identity differs from the registration")
    entries = data.get("codes")
    if not isinstance(entries, list) or len(entries) != MAX_K or sha256(canonical_bytes(entries)) != TABLE_SHA256:
        raise ValueError("frozen code/token/ordinal table differs from the registration")
    codes = tuple(TokenCode(**entry) for entry in entries)
    if (len({c.code for c in codes}) != MAX_K or len({c.token_id for c in codes}) != MAX_K
            or any(type(c.token_id) is not int or not 0 < c.token_id < VOCAB_SIZE
                   or not c.code.isascii() or not c.code.isalpha() or not c.code.isupper() for c in codes)
            or sha256(canonical_bytes([c.code for c in codes])) != CODE_STRINGS_SHA256):
        raise ValueError("invalid frozen token-ID bijection")
    return codes


@dataclass(frozen=True, init=False)
class G1FilteredTokenReadout:
    """Standalone codebook backend; caller tokenizer identities are recorded assertions.

    The constructor proves actual supplied encode/decode behavior for every frozen
    code and the exact recorded context. This is not authentication of a callback,
    model provenance or a proof for an unimplemented complete request renderer.
    """
    name = "rwkv_g1_filtered_codes_experimental"
    _codes: tuple[TokenCode, ...]
    _source_hash: str
    _file_hash: str

    def __init__(self, encode, decode, *, tokenizer_sha256, tokenizer_source_sha256,
                 special_ids=frozenset({0}), table_path=TABLE_PATH):
        if tokenizer_sha256 != TOKENIZER_SHA256 or tokenizer_source_sha256 != TOKENIZER_SOURCE_SHA256:
            raise ValueError("requires the pinned G1k tokenizer and reviewed tokenizer source")
        object.__setattr__(self, "_codes", load_frozen_table(table_path))
        prefix = list(encode(PROOF_CONTEXT))
        if not prefix or any(type(i) is not int or not 0 <= i < VOCAB_SIZE for i in prefix):
            raise ValueError("invalid tokenizer prefix")
        controls = set(special_ids) | {0}
        for entry in self._codes:
            text = " " + entry.code
            ids = list(encode(text))
            if (ids != [entry.token_id] or entry.token_id in controls or decode(ids) != text
                    or list(encode(PROOF_CONTEXT + text)) != prefix + [entry.token_id]):
                raise ValueError(f"tokenizer/context proof failed for frozen code {entry.code}")
        object.__setattr__(self, "_source_hash", sha256(Path(__file__).read_bytes()))
        object.__setattr__(self, "_file_hash", sha256(Path(table_path).read_bytes()))

    def revision(self):
        return {"revision_id": f"{CODEBOOK_ID};tok={TOKENIZER_SHA256};table={TABLE_SHA256};impl={self._source_hash};renderer=blocked",
                "backend": self.name, "family": "rwkv7", "model_id": None, "snapshot": None,
                "codebook_id": CODEBOOK_ID, "codebook_table_sha256": TABLE_SHA256,
                "tokenizer_sha256": TOKENIZER_SHA256, "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256,
                "implementation_sha256": self._source_hash, "codebook_file_sha256": self._file_hash,
                "proof_context_sha256": CONTEXT_SHA256, "max_context": None,
                "renderer_integration": "blocked", "model_provenance": "not established by this component"}

    def option_token_ids(self, question):
        validate_legacy({"model": "component-check", "state": "", "questions": {"q": question}})
        options = canonical_options(question)
        semantic_ids = [oid for oid, _ in options]
        if len(set(semantic_ids)) != len(options):
            raise ValueError("duplicate canonical semantic IDs")
        return {oid: self._codes[i].token_id for i, (oid, _) in enumerate(options)}

    def parse_exact_token_ids(self, question, token_ids):
        if (not isinstance(token_ids, (list, tuple)) or len(token_ids) != 1
                or type(token_ids[0]) is not int):
            raise ValueError("answer must be exactly one integer token ID")
        inverse = {token: oid for oid, token in self.option_token_ids(question).items()}
        if token_ids[0] not in inverse:
            raise ValueError("token is not one of this question's allowed answer codes")
        return inverse[token_ids[0]]

    def logits_for_options(self, question, vocabulary_logits):
        mapping = self.option_token_ids(question)
        if isinstance(vocabulary_logits, (str, bytes, dict)) or len(vocabulary_logits) != VOCAB_SIZE:
            raise ValueError("requires the complete 65536-element G1k vocabulary logit vector")
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in vocabulary_logits):
            raise ValueError("vocabulary logits must be finite real numbers")
        return {oid: float(vocabulary_logits[token]) for oid, token in mapping.items()}

    def raw_probabilities(self, question, vocabulary_logits):
        logits = self.logits_for_options(question, vocabulary_logits)
        values = softmax(list(logits.values()))
        return dict(zip(logits, values))

    def wire_answer_from_logits(self, question, vocabulary_logits):
        """Ordinary legacy conversion only; not a checked D response or calibrated stage."""
        return wire_answer(question, self.raw_probabilities(question, vocabulary_logits))

    def plan(self, request):
        raise BackendUnsupported("renderer_integration_blocked", "no reviewed full C1 request renderer exists")

    def score(self, request, check_deadline):
        raise BackendUnsupported("renderer_integration_blocked", "no reviewed full C1 request renderer exists")
