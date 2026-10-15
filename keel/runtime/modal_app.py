"""Shared Modal definitions for Keel: images, the model-cache volume and GPU defaults.

Every Keel GPU job imports from here so model weights download once (volume ``ssdm-hf-cache``) and the inference
image is identical across Card evaluations and teacher labelling.
"""
from __future__ import annotations

import modal

HF_CACHE = "/root/.cache/huggingface"
hf_cache = modal.Volume.from_name("ssdm-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("keel-results", create_if_missing=True)

#: vLLM inference image (the FlashInfer sampler JIT needs nvcc, which debian_slim lacks; the default sampler is used)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.30.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

#: pinned model revisions used across Keel experiments
MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "h2o-lightning-4b": ("h2oai/h2o-lightning-4b", "193ad740925b176a3b70a5a13a7cff2f2fadd01e"),
}
DEFAULT_GPU = "H100"
