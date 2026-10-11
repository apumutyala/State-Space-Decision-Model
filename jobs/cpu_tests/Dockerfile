# CPU-only test image (chip cpu-8): torch CPU wheel + the repo's job-test pins (transformers 5.17.0 as jobs/qual).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" && \
    pip install "transformers==5.17.0" "numpy==2.2.6" "pyarrow==25.0.1" "safetensors==0.8.0" "huggingface_hub==1.32.0" \
      zstandard pytest
RUN python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
COPY trees /opt/trees
COPY tokenizers /opt/tokenizers
ENV SSDM_TOKENIZERS=/opt/tokenizers
COPY job /opt/job
COPY CONTEXT_MANIFEST.json /opt/job/CONTEXT_MANIFEST.json
WORKDIR /opt/job
CMD ["bash", "/opt/job/run.sh"]
