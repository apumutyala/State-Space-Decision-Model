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
            started = time.time()
            prompts = [self.engine.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                       enable_thinking=False) for m in messages]
            outputs = self.engine.llm.generate(prompts, SamplingParams(temperature=.7, top_p=.95,
                       max_tokens=max_tokens, seed=seed), use_tqdm=False)
            return {"meta": {**self.engine.meta, "wall_s": time.time() - started}, "outputs": [
                {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids),
                 "truncated": o.outputs[0].finish_reason == "length"} for o in outputs]}

    return MonitorAuthor
