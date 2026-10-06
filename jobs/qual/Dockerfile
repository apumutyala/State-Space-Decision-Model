# Rung-2 hardware qualification image (pure qualification: no m0 phase, so only the m1 code is copied).
# A failed pip pin fails the build (fail fast); build time is counted in the cost cap (jobs/qual/job_params.json).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "transformers==5.17.0" "flash-linear-attention==0.5.2" "fla-core==0.5.2" "einops==0.8.2" \
      "pyarrow==25.0.1" "huggingface_hub==1.32.0" "safetensors==0.8.0" "accelerate==1.15.0" "numpy==2.2.6" pytest
COPY code/m1 /opt/ssdm/m1
COPY CONTEXT_MANIFEST.json /opt/ssdm/CONTEXT_MANIFEST.json
WORKDIR /opt/ssdm/m1
CMD ["bash", "/opt/ssdm/m1/jobs/qual/qual_job.sh"]
