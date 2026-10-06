# Clef baseline job image. Built by SF Compute from the uploaded context (build_context.sh).
# Pins: torch 2.11.0 / transformers 5.10.2 = the versions Cloudflare's model card says it tested.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUTF8=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3.12 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --upgrade pip && \
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install transformers==5.10.2 safetensors huggingface_hub numpy pytest
# Fast Gated-DeltaNet kernels (the "fastest path the model's code supports"); exact versions are
# recorded by `pip freeze` in env/environment.txt. If either fails to import, transformers falls back
# to the torch implementation and latency is reported as such (receipt field kernel_stack).
RUN pip install flash-linear-attention causal-conv1d || true
# Official Decision Index kit, pinned (MIT); harness from claude/m0-harness (in the context, pinned).
RUN git clone https://github.com/apolinario/decision-index /opt/decision-index && \
    git -C /opt/decision-index checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf && \
    pip install --no-deps /opt/decision-index
COPY app/ /app/
COPY data/ /data/
RUN pip install --no-deps /app/harness && chmod +x /app/jobs/clef_baseline/run_job.sh
WORKDIR /app
CMD ["/app/jobs/clef_baseline/run_job.sh"]
