# Evidence lane: CPU-only image (no torch, no GPU code), identical to jobs/ev_cal_land/Dockerfile: the frozen harness +
# the pinned official Decision Index kit + numpy. The 0.2.1 suite files and DEV rows are data under /data.
FROM python:3.12.7-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy==2.2.6"
RUN git clone https://github.com/apolinario/decision-index /opt/decision-index && \
    git -C /opt/decision-index checkout 87d4650b42b377c0291a89c1f1a879f9b31082bf && \
    pip install --no-cache-dir --no-deps /opt/decision-index
COPY code/m1/ /opt/ssdm/m1/
COPY data/ /data/
COPY CONTEXT_MANIFEST.json /opt/ssdm/CONTEXT_MANIFEST.json
RUN python -c "import numpy, decision_index; print('numpy', numpy.__version__)"
WORKDIR /opt/ssdm/m1
CMD ["bash", "/opt/ssdm/m1/jobs/ev_pool_sample/ev_pool_sample_job.sh"]
