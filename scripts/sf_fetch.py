"""Model/data fetch for the paid qualification run (runs ON the node; I/O only).

python -I scripts/sf_fetch.py --stage first   # Qwen3.5-0.8B + wikitext + MMLU (needed by phase A/0/1)
python -I scripts/sf_fetch.py --stage rest    # RWKV7-0.1B, Mamba-3-187M + tokenizer, Qwen3.5-9B (background)
python -I scripts/sf_fetch.py --stage probe9b # wikitext + MMLU + Qwen3.5-9B (jobs/probe9b)
python -I scripts/sf_fetch.py --stage q9      # Qwen3.5-9B only (jobs/q9_feature_cache)
The paid job runs this inside its container (jobs/qual/qual_job.sh).

Writes $SSDM_DATA_ROOT/paths.env (shell `export NAME=path` lines) and DONE_<stage> markers.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

from huggingface_hub import snapshot_download

ROOT = os.environ["SSDM_DATA_ROOT"]
CACHE = os.path.join(ROOT, "hf")
MAMBA3_TOK = os.environ.get("MAMBA3_TOKENIZER_REPO", "unsloth/Llama-3.1-8B-Instruct")  # ungated Llama-3.1 tokenizer
MAMBA3_TOK_REV = os.environ.get("MAMBA3_TOKENIZER_REV", "4699cc75b550f9c6f3173fb80f4703b62d946aa5")
# every item is pinned to an immutable revision (commit sha); the code also hashes data files (convert/data.py)

ITEMS = {
    "first": [
        ("Q08_DIR", "Qwen/Qwen3.5-0.8B", "model", ["*.json", "*.safetensors", "*.txt", "*.jinja"],
         "2fc06364715b967f1860aea9cf38778875588b17"),
        ("WIKITEXT_DIR", "Salesforce/wikitext", "dataset", ["wikitext-103-raw-v1/*", "wikitext-2-raw-v1/*"],
         "b08601e04326c79dfdd32d625aee71d232d685c3"),
        ("MMLU_DIR", "cais/mmlu", "dataset", ["all/test-*", "all/dev-*"], "c30699e8356da336a370243923dbaf21066bb9fe"),
    ],
    "rest": [
        ("RWKV7_DIR", "fla-hub/rwkv7-0.1B-g1", "model", None, "7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8"),
        ("MAMBA3_DIR", "state-spaces/mamba3-siso-187m", "model", None, "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"),
        ("MAMBA3_TOK_DIR", MAMBA3_TOK, "model", ["tokenizer*", "special_tokens_map.json"], MAMBA3_TOK_REV),
        ("Q9B_DIR", "Qwen/Qwen3.5-9B", "model", ["*.json", "*.safetensors", "*.txt", "*.jinja"],
         "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ],
    "probe9b": [
        ("WIKITEXT_DIR", "Salesforce/wikitext", "dataset", ["wikitext-103-raw-v1/*", "wikitext-2-raw-v1/*"],
         "b08601e04326c79dfdd32d625aee71d232d685c3"),
        ("MMLU_DIR", "cais/mmlu", "dataset", ["all/test-*", "all/dev-*"], "c30699e8356da336a370243923dbaf21066bb9fe"),
        ("Q9B_DIR", "Qwen/Qwen3.5-9B", "model", ["*.json", "*.safetensors", "*.txt", "*.jinja"],
         "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ],
    "q9": [
        ("Q9B_DIR", "Qwen/Qwen3.5-9B", "model", ["*.json", "*.safetensors", "*.txt", "*.jinja"],
         "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(ITEMS), required=True)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    env = os.path.join(ROOT, "paths.env")
    for name, repo, kind, pats, rev in ITEMS[a.stage]:
        t = time.time()
        try:
            p = snapshot_download(repo, repo_type=kind, cache_dir=CACHE, allow_patterns=pats, revision=rev)
            with open(env, "a") as f:
                f.write(f"export {name}={p}\n")
            open(os.path.join(ROOT, f"DONE_{name}"), "w").write(repo)
            print(f"fetched {repo} -> {p} in {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001  one failed item must not block the others
            open(os.path.join(ROOT, f"FAILED_{name}"), "w").write(f"{repo}: {e!r}")
            print(f"FAILED {repo}: {e!r}", flush=True)
    open(os.path.join(ROOT, f"DONE_{a.stage}"), "w").write(str(sorted(glob.glob(os.path.join(ROOT, 'DONE_*')))))


if __name__ == "__main__":
    main()
