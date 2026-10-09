"""`AutoModel.from_pretrained(repo, trust_remote_code=True)`: LFM2-VL with the System One API.

    model.system_one(state, {name: question}, images=None)    # one forward pass
    model.system_one_batch([(state, {name: question}), ...])  # many states, packed with no padding
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

from .lfm2_vl import Lfm2VlForConditionalGeneration
from .runner import SystemOne


class D1Model(Lfm2VlForConditionalGeneration):
    @cached_property
    def engine(self) -> SystemOne:
        from transformers import AutoTokenizer

        return SystemOne(model=self.eval(), tokenizer=AutoTokenizer.from_pretrained(self.name_or_path))

    def system_one(self, state: Any, questions: Mapping[str, Any], images: Sequence | None = None) -> dict:
        """Named questions over a state (text, JSON, or None with images alone):
        `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`."""
        return self.engine.system_one(state, questions, images)

    def system_one_batch(self, requests: Sequence[tuple]) -> list[dict]:
        """`(state, questions)` or `(state, questions, images)` requests, one response each."""
        return self.engine.system_one_batch(requests)
