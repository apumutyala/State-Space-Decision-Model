"""One-pass typed decisions on a causal LFM backbone.

The model sees the state and the question once, and the answer is read off the
logits at the answer slot. Nothing is decoded, so a schema violation is
impossible and `tokens generated per decision` is zero.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .api import SystemOneApi
from .lfm2_vl import Lfm2ForCausalLM, Lfm2VlForConditionalGeneration
from .prompt import (
    DEFAULT_MODEL,
    DEFAULT_STATE_STYLE,
    DEFAULT_SYSTEM,
    IM_START,
    Question,
    default_lead,
    prefix_text,
    readout,
    render,
    suffix_text,
)

# Every picture is bounded at this many pixels before the processor.
VISION_MAX_PIXELS = 1024 * 1024


# LFM2 models run on the hybrid stack (`hybrid.py`); any other model type through transformers as it is.
MODELS = {
    "lfm2_vl": Lfm2VlForConditionalGeneration,
    "lfm2": Lfm2ForCausalLM,
}


def load_backbone(model_id: str = DEFAULT_MODEL, dtype=torch.bfloat16):
    """The checkpoint and its tokenizer, with SDPA attention."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kind = PretrainedConfig.get_config_dict(model_id)[0].get("model_type")
    cls = MODELS.get(kind, AutoModelForImageTextToText)
    return cls.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa"), tok


def cap_pixels(image, max_pixels: int = VISION_MAX_PIXELS):
    """A picture downscaled to at most `max_pixels`, bicubic."""
    image = image.convert("RGB") if hasattr(image, "convert") else image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    try:
        from PIL import Image
    except ImportError as e:  # optional for text
        raise ImportError("resizing a picture needs Pillow") from e

    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)


class SystemOne(SystemOneApi):
    """State in, calibrated distribution out, one forward pass."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        calibration=None,
        lead: str | None = None,
        state_style: str = DEFAULT_STATE_STYLE,
        system: str = DEFAULT_SYSTEM,
        option_style: str = "desc",
        compile: bool = False,
        token_budget: int = 65536,
        model=None,
        tokenizer=None,
    ):
        """`model` and `tokenizer`, when given, are a backbone already loaded (`D1Model` passes itself);
        it stays on its device unless `device` says otherwise."""
        if model is None:
            model, tokenizer = load_backbone(model_id)
        else:
            model_id, device = model.config._name_or_path, device or next(model.parameters()).device
        self.model, self.tokenizer = model, tokenizer
        self.model_id = model_id
        if lead is None:
            lead = default_lead(self.model.config.model_type)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        bos = getattr(self.tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.calibration = calibration
        self.lead = lead
        self.state_style = state_style
        self.system = system
        self.option_style = option_style
        self.token_budget = token_budget
        self.processor = None
        # CUDA graphs for single questions on NVIDIA, where eager time is mostly kernel launches.
        if compile and torch.version.hip:
            raise ValueError("compile=True needs CUDA: on ROCm the CUDA graphs fault after a few dozen calls")
        self._one_pass = (torch.compile(self.model.forward, mode="reduce-overhead")
                          if compile else self.model)

    # ---------------------------------------------------------------- prompt

    def render(self, state: Any, q: Question) -> str:
        return render(
            self.tokenizer, state, q, self.bos, self.lead, self.state_style,
            self.system, self.option_style,
        )

    # --------------------------------------------------------------- forward

    def _logz_ids(self, rows: list[list[int]]) -> list[torch.Tensor]:
        """Log-softmax at the answer slot, one row per token list, in one pass.

        The rows are one tree (`hybrid.py`): their common start is its trunk and
        is read once; the rest of each row is a branch, packed with no padding.
        Mathematically each row alone; in bf16 the kernels differ by batch shape.
        """
        if len(rows) == 1:  # nothing to share: a plain chain is faster than a tree of one
            row = self._one_pass(input_ids=torch.tensor(rows, device=self.device), logits_to_keep=1).logits[0, -1]
            return [row.float() - torch.logsumexp(row.float(), dim=-1)]
        shared = 0  # every row keeps at least its last token
        while shared < min(map(len, rows)) - 1 and len({r[shared] for r in rows}) == 1:
            shared += 1
        return self._tree_logz(rows[0][:shared], [r[shared:] for r in rows])

    def _tree_logz(self, trunk: list[int], rows: list[list[int]], **vision) -> list[torch.Tensor]:
        """`trunk` then each of `rows`, read at each row's end."""
        packed = torch.tensor([t for r in rows for t in r], device=self.device)
        lengths = torch.tensor([len(r) for r in rows], device=self.device)
        trunk = torch.tensor([trunk], dtype=torch.long, device=self.device)
        logits = self.model.answer(trunk, packed, lengths, **vision).float()
        return list(logits - torch.logsumexp(logits, dim=-1, keepdim=True))

    def plan_batches(self, texts: Sequence[str], token_budget: int | None = None) -> list[list[int]]:
        """Consecutive batches of at most `token_budget` tokens: rows are packed,
        so a batch costs its real tokens."""
        return self._plan([len(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts], token_budget)

    def _plan(self, lengths: Sequence[int], token_budget: int | None = None) -> list[list[int]]:
        if not lengths:
            return []
        budget = token_budget or self.token_budget
        out: list[list[int]] = [[]]
        used = 0
        for i, n in enumerate(lengths):
            if out[-1] and used + n > budget:
                out.append([])
                used = 0
            out[-1].append(i)
            used += n
        return out

    # --------------------------------------------------------------- readout

    def _readout(self, q: Question, logz: torch.Tensor) -> list[float]:
        return readout(self.tokenizer, q, logz, self.calibration)

    # ------------------------------------------------------------------- api

    @torch.inference_mode()
    def run(self, requests: Sequence[tuple[Any, list[Question], Sequence]]) -> list[tuple[list[list[float]], int]]:
        """Each `(state, questions, images)` request's probabilities and the tokens it read. Requests of one
        question and no pictures are packed together, one tree per token budget; any other request is its
        own pass, its state (and pictures) the trunk and its questions the branches."""
        out: list = [None] * len(requests)
        single = [i for i, (_, qs, images) in enumerate(requests) if len(qs) == 1 and not images]
        rows = [self.tokenizer.encode(self.render(requests[i][0], requests[i][1][0]), add_special_tokens=False)
                for i in single]
        for chunk in self._plan([len(r) for r in rows]):
            for j, z in zip(chunk, self._logz_ids([rows[j] for j in chunk])):
                out[single[j]] = ([self._readout(requests[single[j]][1][0], z)], len(rows[j]))
        for i, (state, qs, images) in enumerate(requests):
            if out[i] is None:
                out[i] = self._request(state, qs, images)
        return out

    def _request(self, state: Any, qs: list[Question], images: Sequence) -> tuple[list[list[float]], int]:
        pics = [cap_pixels(im) for im in images]
        prefix = prefix_text(self.tokenizer, state, self.bos, self.state_style, self.system,
                             self._image_markup(len(pics)) if pics else "")
        suffixes = [suffix_text(self.tokenizer, q, self.lead, self.option_style) for q in qs]
        vision: dict = {}
        if not pics:
            trunk = self.tokenizer.encode(prefix, add_special_tokens=False)
        elif len(qs) == 1:  # the whole prompt in one plain pass
            inputs = self._image_inputs(prefix + suffixes[0], pics)
            row = self._one_pass(**inputs, logits_to_keep=1).logits[0, -1].float()
            return [self._readout(qs[0], row - torch.logsumexp(row, dim=-1))], int(inputs["input_ids"].shape[1])
        else:
            vision = self._image_inputs(prefix, pics)
            trunk = vision.pop("input_ids")[0].tolist()
            vision.pop("attention_mask", None)
        branches = [self.tokenizer.encode(s, add_special_tokens=False) for s in suffixes]
        probs: list[list[float]] = []
        for chunk in self._plan([len(b) for b in branches]):
            zs = self._tree_logz(trunk, [branches[j] for j in chunk], **vision)
            probs += [self._readout(qs[j], z) for j, z in zip(chunk, zs)]
        return probs, len(trunk) + sum(map(len, branches))

    def tokens(self, state: Any, questions: Sequence[Question]) -> int:
        """The longest prompt one of `questions` makes over a text state: its state's tokens and its own."""
        prefix = prefix_text(self.tokenizer, state, self.bos, self.state_style, self.system)
        return len(self.tokenizer.encode(prefix, add_special_tokens=False)) + max(
            len(self.tokenizer.encode(suffix_text(self.tokenizer, q, self.lead, self.option_style),
                                      add_special_tokens=False)) for q in questions)

    # ---------------------------------------------------------------- vision

    def _image_markup(self, n: int) -> str:
        """What the chat template writes for `n` images at the head of a user turn (`<image>` each on LFM2-VL)."""
        if self.processor is None:
            self.processor = self._load_processor()
        msgs = [{"role": "user", "content": [*([{"type": "image"}] * n), {"type": "text", "text": "\x00"}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        head = f"{IM_START}user\n"
        return text[text.index(head) + len(head):text.index("\x00")]

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)

    def _image_inputs(self, text: str, images: Sequence) -> dict:
        """Token ids and pixel inputs for one prompt holding `images`; the prompt carries its own BOS."""
        inputs = self.processor(text=[text], images=[list(images)], return_tensors="pt", add_special_tokens=False)
        if "pixel_attention_mask" in inputs:
            # LFM2-VL's processor pads every image to 1024 patches and the tower
            # masks the padding out; cutting it gives the same answer for up to
            # half the work.
            n = int(inputs["pixel_attention_mask"].sum(1).max())
            inputs["pixel_values"] = inputs["pixel_values"][:, :n]
            inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :n]
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
