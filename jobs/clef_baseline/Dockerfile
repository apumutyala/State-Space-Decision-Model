# Clef baseline job image. Built by SF Compute from the uploaded context (build_context.sh).
#
# Base: the same pinned torch 2.13.0 image as qual/probe9b/rwkv7 (torch 2.13.0 pins triton==3.7.1, the first Triton
# fla 0.5.2 trusts for the gated GDN backward on Hopper, fla #640). Clef needs this too: the S1 --grad-probe and the
# whole of job B (GCG-lite) call backward() through Qwen3.5's Gated-DeltaNet layers, and transformers 5.10.2 routes
# those layers to fla's chunk_gated_delta_rule whenever fla imports (modeling_qwen3_5.py), so on the old torch 2.11 /
# triton 3.6 image fla would raise (#640) at S1. FLA_TILELANG=0 pins fla's Triton backward (the path fla_guard.py
# checks and smoke-tests); tilelang is NOT installed.
#
# Declared deviation from Cloudflare's model card ("Tested with torch 2.11 and transformers 5.10.2"): torch 2.11 ->
# 2.13.0 (+ triton 3.7.1, fla/fla-core 0.5.2 pinned). transformers stays at the card's 5.10.2, so the Qwen3.5 model
# code and AutoProcessor are the card's. joint_schema_model.py uses only stable torch.nn modules. Equivalence is
# checked in-image by baselines/clef_probes.py (S1): decide vs systemone parity, and fast fla kernels vs
# transformers' torch reference GDN (kernel_path). The full stack is in revision()["inference_config"]["kernel_stack"].
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel@sha256:09c0260351b12b617408ff5711e08d88ccdbb9a21f7f6d7de67e39039b022ae1
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 MAX_JOBS=8 PIP_BREAK_SYSTEM_PACKAGES=1 FLA_TILELANG=0
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "torch==2.13.0" "triton==3.7.1" "transformers==5.10.2" "flash-linear-attention==0.5.2" "fla-core==0.5.2" "einops==0.8.2" \
      "huggingface_hub==1.32.0" "safetensors==0.8.0" "accelerate==1.15.0" "numpy==2.2.6" pytest
# causal-conv1d (pinned; built from source against torch 2.13 with the base image's nvcc 12.6). Best effort: without
# it transformers uses its torch conv fallback (fla's GDN kernels are used either way); kernel_stack records it.
RUN pip install --no-cache-dir --no-build-isolation "causal-conv1d==1.7.0" || echo "causal-conv1d unavailable"
# Build-time fail-fast (no GPU needed): nothing above may have swapped torch/triton for other builds.
RUN python -c "import torch, triton; v = (torch.__version__, torch.version.cuda, triton.__version__); assert v[0].split('+')[0] == '2.13.0' and v[1] == '12.6' and v[2] == '3.7.1', v; print('torch/cuda/triton', v)"
# Official Decision Index kit, pinned (MIT); harness from claude/m0-harness (in the context, pinned).
RUN git clone https://github.com/apolinario/decision-index /opt/decision-index && \
    git -C /opt/decision-index checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf && \
    pip install --no-cache-dir --no-deps /opt/decision-index
COPY app/ /app/
COPY data/ /data/
RUN pip install --no-cache-dir --no-deps /app/harness && chmod +x /app/jobs/clef_baseline/run_job.sh
WORKDIR /app
CMD ["/app/jobs/clef_baseline/run_job.sh"]
