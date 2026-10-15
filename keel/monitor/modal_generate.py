"""Same pinned open-weight author for both labels and all evaluation comments."""
from __future__ import annotations

import time


def author_class():
    import modal
    from keel.runtime.vllm_judge import APP, _Engine, _IMAGE, _VOLUMES

    @APP.cls(image=_IMAGE, gpu="H100", timeout=1200, scaledown_window=60,
             volumes=_VOLUMES, serialized=True)
    class MonitorAuthor:
        @modal.enter()
        def load(self):
            self.engine = _Engine("qwen3.5-9b", max_model_len=16384)

        @modal.method()
        def generate(self, messages: list, seed: int, max_tokens: int = 4096):
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            from keel.runtime.vllm_judge import BUDGET_TAIL
            started = time.time()
            prompts = [self.engine.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                       enable_thinking=True) for m in messages]
            thinking = self.engine.llm.generate(prompts, [
                SamplingParams(temperature=.7, top_p=.95, max_tokens=1536,
                               seed=seed + i, stop=["</think>"]) for i in range(len(prompts))], use_tqdm=False)
            finals = [p + o.outputs[0].text.rstrip() +
                      (BUDGET_TAIL if o.outputs[0].finish_reason == "length" else "\n") + "</think>\n\n"
                      for p, o in zip(prompts, thinking, strict=True)]
            string = {"type": "string"}
            schema = {"type": "object", "properties": {
                "code": string, "trigger": {"type": ["string", "null"]},
                "comments": {"type": "object", "properties": {
                    k: string for k in ("review", "specification", "instruction", "test_log", "persona")},
                    "required": ["review", "specification", "instruction", "test_log", "persona"],
                    "additionalProperties": False}},
                "required": ["code", "trigger", "comments"], "additionalProperties": False}
            outputs = self.engine.llm.generate(finals, [
                SamplingParams(temperature=.7, top_p=.95, max_tokens=max_tokens, seed=seed + i + 10000,
                               structured_outputs=StructuredOutputsParams(json=schema))
                for i in range(len(prompts))], use_tqdm=False)
            return {"meta": {**self.engine.meta, "wall_s": time.time() - started}, "outputs": [
                {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids),
                 "reasoning": thought.outputs[0].text, "think_tokens": len(thought.outputs[0].token_ids),
                 "think_truncated": thought.outputs[0].finish_reason == "length",
                 "truncated": o.outputs[0].finish_reason == "length"}
                for thought, o in zip(thinking, outputs, strict=True)]}

    return MonitorAuthor
