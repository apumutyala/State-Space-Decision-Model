"""Read-only, local tokenizer-only shape sample of the pinned ExactSum fit export.

Emit one JSON object to stdout. No model is constructed; no weights are opened.
The caller may redirect stdout to a receipt file. The watchdog bounds the entire
run, including imports, hashing and both gzip passes, to at most 180 seconds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
TOKENIZER_PINS = {
    "tokenizer.json": "1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf",
    "tokenizer_config.json": "2b5e075c2c16780061ae272c9f0b52ff620ddd8e87fdf7fa82fc425661667329",
}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    ordered = sorted(values)
    return {str(q): ordered[max(0, math.ceil(q * len(ordered)) - 1)]
            for q in (0, 0.5, 0.9, 0.95, 0.99, 1)}


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--expected-gzip-sha256", required=True)
    parser.add_argument("--expected-jsonl-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 180:
        parser.error("--max-seconds must be in (0, 180]")
    started = time.monotonic()
    finished = threading.Event()
    phase = "initialization"

    def watchdog():
        if not finished.wait(args.max_seconds):
            print(json.dumps({"status": "hard_cap_stopped", "phase": phase,
                              "wall_s": round(time.monotonic() - started, 3)}), flush=True)
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for key, value in {"PYTHONDONTWRITEBYTECODE": "1", "USE_TORCH": "0", "USE_TF": "0",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}.items():
            os.environ[key] = value
        sys.dont_write_bytecode = True
        import psutil
        process = psutil.Process()
        process.cpu_affinity(process.cpu_affinity()[:1])
        if sys.platform == "win32":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        sys.path.insert(0, str(ROOT))
        phase = "verify_compressed_input_and_tokenizer"
        gzip_sha = sha_file(args.records)
        if gzip_sha != args.expected_gzip_sha256:
            raise ValueError("gzip_sha256_mismatch")
        tokenizer_sha = {name: sha_file(args.tokenizer_dir / name) for name in TOKENIZER_PINS}
        if tokenizer_sha != TOKENIZER_PINS:
            raise ValueError("tokenizer_sha256_mismatch")

        phase = "count_and_verify_decompressed_input"
        stage = time.monotonic()
        count, raw_bytes, raw_sha = 0, 0, hashlib.sha256()
        with gzip.open(args.records, "rb") as stream:
            for line in stream:
                raw_sha.update(line)
                raw_bytes += len(line)
                count += bool(line.strip())
        if raw_sha.hexdigest() != args.expected_jsonl_sha256:
            raise ValueError("jsonl_sha256_mismatch")
        if count != 52224:
            raise ValueError("expected_52224_fit_records")
        count_s = time.monotonic() - stage
        if time.monotonic() - started > args.max_seconds * 0.78:
            raise TimeoutError("input_pass_threatens_cap")

        # Exact sample rule used by the preceding ephemeral probe.
        indices = [i * (count - 1) // 255 for i in range(256)]
        selected, samples, tasks = set(indices), [], set()
        phase = "collect_sample_and_count_tasks"
        stage = time.monotonic()
        index = -1
        with gzip.open(args.records, "rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                index += 1
                record = json.loads(line)
                tasks.add(record["id"])
                if index in selected:
                    samples.append(record)
        if index + 1 != count or len(samples) != 256 or len(tasks) != 5184:
            raise ValueError("fit_count_or_sample_mismatch")
        collect_s = time.monotonic() - stage
        if time.monotonic() - started > args.max_seconds * 0.89:
            raise TimeoutError("collection_pass_threatens_cap")

        phase = "load_local_tokenizer"
        from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
        from heads.layout import build_row
        tokenizer = Rwkv7Tokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
        special = set(tokenizer.all_special_ids)

        @lru_cache(maxsize=1024)
        def encode(text, trusted=True):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not trusted and special.intersection(ids):
                raise ValueError("untrusted_special_token")
            return ids

        phase = "tokenize_sample"
        stage = time.monotonic()
        ks, ts, continuations, sample_tasks = [], [], [], set()
        for record in samples:
            request = record["request"]
            if len(record["labels"]) != 1 or len(request["questions"]) != 1:
                raise ValueError("expected_one_question_per_fit_record")
            qid, gold = next(iter(record["labels"].items()))
            row = build_row(request, qid, encode, template="jrt-rwkv-g1-v1", layout="ordered")
            if gold not in row.option_ids:
                raise ValueError("gold_not_offered")
            ks.append(row.n_options)
            ts.append(len(row.ids))
            continuations.append(max(map(len, row.continuations)))
            sample_tasks.add(record["id"])
        tokenize_s = time.monotonic() - stage
        output = {
            "schema": "ssdm.exact_sum_fit_shape_sample.v1", "status": "complete",
            "command_argv": [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]],
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: sha_file(ROOT / name) for name in
                              ("scripts/profile_exact_sum_fit_shapes.py", "heads/layout.py", "decide/prompt.py",
                               "decide/schema.py", "third_party/rwkv7_hf/tokenization_rwkv7.py")},
            "input": {"path": str(args.records.resolve()), "gzip_sha256": gzip_sha,
                      "jsonl_sha256": raw_sha.hexdigest(), "jsonl_bytes": raw_bytes,
                      "rows": count, "tasks": len(tasks)},
            "tokenizer": {"path": str(args.tokenizer_dir.resolve()),
                          "repo": "RWKV/RWKV7-G1k-13.3B-20260930", "revision": TOKENIZER_REVISION,
                          "files_sha256": tokenizer_sha, "local_files_only": True},
            "sample": {"rule": "0-based nonblank record positions floor(i*(N-1)/255), i=0..255",
                       "indices": indices,
                       "indices_sha256": hashlib.sha256(("\n".join(map(str, indices)) + "\n").encode("ascii")).hexdigest(),
                       "rows": len(ks), "tasks": len(sample_tasks),
                       "K_nearest_rank_quantiles": quantiles(ks), "T_nearest_rank_quantiles": quantiles(ts),
                       "K_counts": dict(sorted(Counter(ks).items())), "continuation_max": max(continuations),
                       "caps": {"max_k": 16, "max_t": 2048, "max_continuation": 16},
                       "pass_K_count": sum(k <= 16 for k in ks), "pass_T_count": sum(t <= 2048 for t in ts),
                       "pass_all_count": sum(k <= 16 and t <= 2048 and c <= 16
                                             for k, t, c in zip(ks, ts, continuations))},
            "environment": {"python": sys.version, "python_executable": sys.executable,
                            "platform": platform.platform(), "cpu": cpu_name(),
                            "physical_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
                            "cpu_affinity": process.cpu_affinity(), "workers": 1,
                            "packages": {name: importlib.metadata.version(name)
                                         for name in ("transformers", "psutil", "tokenizers")},
                            "torch_imported_indirectly": "torch" in sys.modules},
            "timing": {"wall_s": round(time.monotonic() - started, 3), "count_s": round(count_s, 3),
                       "collect_s": round(collect_s, 3), "tokenize_s": round(tokenize_s, 3),
                       "hard_cap_s": args.max_seconds},
            "rss_mb": round(process.memory_info().rss / 1048576, 1),
            "limits": ["Sample extrema and quantiles are not full-corpus extrema or quantiles.",
                       "Uniform positions are deterministic, not independent random sampling.",
                       "Fit-only provenance is inherited from the hash-pinned export receipt/manifest.",
                       "No model construction, weights, forward/backward, GPU, network or dataset writes.",
                       "No capability, safety, calibration or held-out qualification result."],
        }
        print(json.dumps(output, indent=2), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "phase": phase, "error": str(error),
                          "wall_s": round(time.monotonic() - started, 3)}), flush=True)
        return 1
    finally:
        finished.set()


if __name__ == "__main__":
    raise SystemExit(main())
