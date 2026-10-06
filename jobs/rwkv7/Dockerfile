# rwkv7 image (stages A readout and B LoRA pilot share it, so stage B's depends_on sees the same context).
# Same base digest and pins as jobs/qual (transformers 5.17.0 >= the 5.15 the RWKV7-G1k card asks for). The RWKV-7
# code is the repo's own bundled code, vendored + sha-pinned in third_party/rwkv7_hf (no trust_remote_code); its
# "chunked" WKV is pure PyTorch, so no extra kernel package is needed. PRIVATE data: DEV rows + m2 pilot records.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "transformers==5.17.0" "einops==0.8.2" "pyarrow==25.0.1" "huggingface_hub==1.32.0" \
      "safetensors==0.8.0" "accelerate==1.15.0" "numpy==2.2.6" pytest
RUN git clone https://github.com/apolinario/decision-index /opt/decision-index && \
    git -C /opt/decision-index checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf && \
    pip install --no-cache-dir --no-deps /opt/decision-index
COPY code/m1 /opt/ssdm/m1
COPY data/ /data/
COPY CONTEXT_MANIFEST.json /opt/ssdm/CONTEXT_MANIFEST.json
WORKDIR /opt/ssdm/m1
CMD ["bash", "/opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh"]
