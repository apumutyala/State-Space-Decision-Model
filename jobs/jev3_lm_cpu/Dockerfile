# JEV3-LM CPU job (chip cpu-8): the CPU test suites on one branch tree. Same image recipe as jobs/cpu_tests (torch CPU wheel + the job-test pins).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" && \
    pip install "transformers==5.17.0" "numpy==2.2.6" "pyarrow==25.0.1" "safetensors==0.8.0" "huggingface_hub==1.32.0" \
      zstandard pytest
RUN python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
COPY tree /opt/tree
COPY data /data
COPY job /opt/job
COPY CONTEXT_MANIFEST.json /opt/job/CONTEXT_MANIFEST.json
ENV SSDM_DATA=/data SSDM_TOKENIZERS=/data/tokenizers
WORKDIR /opt/tree
CMD ["bash", "/opt/job/run.sh"]
