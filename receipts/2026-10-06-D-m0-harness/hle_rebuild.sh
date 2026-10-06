#!/bin/bash
# WSL: place HLE where the kit expects it, normalise + freeze (no other downloads), cut 0.2, stage.
set -e
cd ~/ssdm-data
mkdir -p work/artifacts/benchmark-suite/raw/hle/data
cp /mnt/c/Users/apumu/research/CHLabs/ssdm-data/hle-raw/data/test-00000-of-00001.parquet work/artifacts/benchmark-suite/raw/hle/data/
# the HLE normaliser records the HF download-cache signature/etag in every row's provenance
mkdir -p work/artifacts/benchmark-suite/raw/hle/.cache/huggingface/download/data
sed 's/$//' /mnt/c/Users/apumu/research/CHLabs/ssdm-data/hle-raw/.cache/huggingface/download/data/test-00000-of-00001.parquet.metadata > work/artifacts/benchmark-suite/raw/hle/.cache/huggingface/download/data/test-00000-of-00001.parquet.metadata
.venv/bin/python -I -m decision_index suite rebuild --work work --only 45 --skip-download > logs/rebuild-hle.log 2>&1
grep -E '"byte_identical"|"rows_byte_identical"|"requests"' logs/rebuild-hle.log | head
mkdir -p /mnt/c/Users/apumu/research/CHLabs/ssdm-data/suite-0.2
cp work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz /mnt/c/Users/apumu/research/CHLabs/ssdm-data/suite-0.2/
