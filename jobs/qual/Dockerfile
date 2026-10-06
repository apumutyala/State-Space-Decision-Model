# Rung-2 hardware qualification image (pure qualification: no m0 phase, so only the m1 code is copied).
# A failed pip pin fails the build (fail fast); build time is counted in the cost cap (jobs/qual/job_params.json).
# Base: torch 2.13.0, whose wheel pins triton==3.7.1 -- the first Triton fla 0.5.2 trusts for the gated GDN backward on
# Hopper (fla #640; SF job-jbw22 failed on the old torch 2.8.0 / triton 3.4.0 image). Ubuntu 24.04 system Python 3.12
# (PEP 668 marker present, hence PIP_BREAK_SYSTEM_PACKAGES). FLA_TILELANG=0 pins fla's Triton backward (the one
# tests/test_gdn.py measures); jobs/common/fla_guard.py re-checks all of this at setup and exits 5 otherwise.
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel@sha256:09c0260351b12b617408ff5711e08d88ccdbb9a21f7f6d7de67e39039b022ae1
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MAX_JOBS=8 PIP_BREAK_SYSTEM_PACKAGES=1 FLA_TILELANG=0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "torch==2.13.0" "triton==3.7.1" "transformers==5.17.0" "flash-linear-attention==0.5.2" "fla-core==0.5.2" "einops==0.8.2" \
      "pyarrow==25.0.1" "huggingface_hub==1.32.0" "safetensors==0.8.0" "accelerate==1.15.0" "numpy==2.2.6" pytest
# Build-time fail-fast (no GPU needed): the pip line above must not have swapped torch/triton for other builds.
RUN python -c "import torch, triton; v = (torch.__version__, torch.version.cuda, triton.__version__); assert v[0].split('+')[0] == '2.13.0' and v[1] == '12.6' and v[2] == '3.7.1', v; print('torch/cuda/triton', v)"
COPY code/m1 /opt/ssdm/m1
COPY CONTEXT_MANIFEST.json /opt/ssdm/CONTEXT_MANIFEST.json
WORKDIR /opt/ssdm/m1
CMD ["bash", "/opt/ssdm/m1/jobs/qual/qual_job.sh"]
