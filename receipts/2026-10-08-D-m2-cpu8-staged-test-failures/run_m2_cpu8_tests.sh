#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${HOME:?}/m2-cpu8-validation"
INPUT="$BASE/input"
WORK="$BASE/work"
OUT="$WORK/launch-output"
REPO="$WORK/repo"
CTX="$WORK/context"
DATA="$WORK/SSDM_DATA"
VENV="$WORK/outer-venv"
SOURCE_SHA=4247b340dddecf620de6ff22751256f4127237cb
ORIGIN_SHA=5aac22e93968747cd0cf6e45603d55d8a9ac75e4
KIT_SHA=87d4650b42b377c0291a89c1f1a879f9b31082bf

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUTF8=1
export UV_CACHE_DIR="$WORK/uv-cache"

mkdir -p "$BASE" "$WORK" "$DATA"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse output directory: $OUT" >&2
  exit 70
fi
mkdir -p "$INPUT"
cd "$INPUT"
sha256sum -c package.sha256

# The catalog image has Python, Git and uv; keep launcher-only zstandard isolated.
uv venv "$VENV"
uv pip install --python "$VENV/bin/python" zstandard

# Extract the already-pinned context archive on the CPU node, not the laptop.
mkdir -p "$CTX"
"$VENV/bin/python" - "$INPUT/m2_first_cell-4247b340ddde.tar.zst" "$CTX" <<'PY'
import sys, tarfile, zstandard
src, dest = sys.argv[1:]
with open(src, "rb") as compressed:
    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            archive.extractall(dest, filter="data")
PY

# Reconstruct exactly the SSDM_DATA layout required by the staged launcher tests.
mkdir -p \
  "$DATA/m2/build/sample-20000-seed20261006" \
  "$DATA/m2/build/sample-150000-seed20261006" \
  "$DATA/m2/parent-arm/job-uhxb5" \
  "$DATA/suite-0.2" "$DATA/slices" \
  "$DATA/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba"
cp "$CTX/data/m2/records.jsonl.gz" "$DATA/m2/build/sample-20000-seed20261006/records.jsonl.gz"
cp "$CTX/data/m2-150k/records.jsonl.gz" \
   "$CTX/data/m2-150k/sample-150000-seed20261006.manifest.jsonl.gz" \
   "$CTX/data/m2-150k/sample-150000-seed20261006.content.jsonl.gz" \
   "$DATA/m2/build/sample-150000-seed20261006/"
cp "$CTX/data/suite-0.2/selected-rows.jsonl.gz" \
   "$CTX/data/suite-0.2/added-rows.jsonl.gz" \
   "$CTX/data/suite-0.2/excluded-questions.json" "$DATA/suite-0.2/"
cp "$CTX/data/slices/dev-rows.jsonl.gz" "$DATA/slices/"
cp "$CTX/data/parent_arm_seed/arm.json" \
   "$CTX/data/parent_arm_seed/units-g12.json" \
   "$CTX/data/parent_arm_seed/units-index.json" "$DATA/m2/parent-arm/job-uhxb5/"
cp "$INPUT/tokenizer.json" "$INPUT/tokenizer_config.json" \
   "$DATA/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba/"

# Restore the pinned Decision Index source tree with its .git identity for guard_ready.
mkdir -p "$DATA/decision-index"
git -C "$DATA/decision-index" init -q
git -C "$DATA/decision-index" fetch -q "$INPUT/decision-index-87d4650.bundle" HEAD:refs/heads/main
git -C "$DATA/decision-index" checkout -q --detach "$KIT_SHA"
test "$(git -C "$DATA/decision-index" rev-parse HEAD)" = "$KIT_SHA"
test -d "$DATA/decision-index/.git"

# Verify exact source and local main ancestry; do not fetch GitHub credentials into the node.
git init -q "$REPO"
git -C "$REPO" fetch -q "$INPUT/agent-main-5aac22.bundle" \
  refs/remotes/origin/agent-main:refs/remotes/origin/agent-main
git -C "$REPO" checkout -q --detach "$SOURCE_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$SOURCE_SHA"
git -C "$REPO" merge-base --is-ancestor "$SOURCE_SHA" refs/remotes/origin/agent-main

export SSDM_DATA="$DATA"
export SSDM_DI_KIT="$DATA/decision-index"
export HF_HUB_CACHE="$WORK/hf-cache"
mkdir -p "$HF_HUB_CACHE" "$(dirname "$OUT")"

# This runs the standard builder's staged tests and creates submit.json, but never calls the provider.
source "$VENV/bin/activate"
cd "$REPO"
python -m jobs.common.launch m2_first_cell \
  --sha "$SOURCE_SHA" \
  --out "$OUT" \
  --data "$DATA" \
  --repo "$REPO" \
  --no-fetch

# The launcher only returns zero after the selected suite passes with permitted skips.
test -s "$OUT/submit.json"
test -s "$OUT/CONTEXT_MANIFEST.json"
test -s "$OUT/m2_first_cell-4247b340ddde.tar.zst"
test -s "$OUT/stage/junit-m2_first_cell.xml"
"$VENV/bin/python" - "$OUT/submit.json" "$OUT/stage/junit-m2_first_cell.xml" <<'PY'
import json, sys, xml.etree.ElementTree as ET
submit_path, junit_path = sys.argv[1:]
submit = json.load(open(submit_path, encoding="utf-8"))
tests = submit.get("tests")
if not isinstance(tests, dict) or tests.get("skipped") is True:
    raise SystemExit("launcher did not record a completed staged test suite")
root = ET.parse(junit_path).getroot()
counts = {key: int(root.attrib.get(key, "0")) for key in ("tests", "failures", "errors", "skipped")}
if counts["failures"] or counts["errors"]:
    raise SystemExit(f"JUnit failures/errors: {counts}")
print(json.dumps({"source_sha": "4247b340dddecf620de6ff22751256f4127237cb",
                  "launcher_tests": tests, "junit_counts": counts,
                  "context_sha256": submit.get("context", {}).get("sha256"),
                  "context_bytes": submit.get("context", {}).get("size_bytes"),
                  "h100_request_submitted": False}, indent=2))
PY
