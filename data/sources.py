"""Pinned public TRAIN-split sources used by the m2 sample build, and their fetcher.

Downloads are untrusted data: each source gets its own directory under ``config.SOURCES_DIR``; nothing there is
executed or imported; parquet/json/csv are parsed as data only. Every file's sha256 is written to
``data/manifests/sources.lock.json`` on first fetch and verified on every later fetch/load.

Only splits named ``train`` (or the source's equivalent training file) are listed. Test/dev/validation files of
any Index source are never fetched here; CLINC150's single file carries all splits, and only its ``train`` and
``oos_train`` keys are read (enforced in convert.py and tested).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from . import config

MAX_BYTES = 250 * 1024 * 1024

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"

# id -> (kind, url-or-repo, revision, [files], license, index_overlap_note)
REGISTRY: dict[str, dict] = {
    "banking77": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "files": ["train.csv"], "license": "CC-BY-4.0", "index": "BANKING77 (4): Index uses test.csv",
    },
    "clinc150": {
        "url": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
        "files": ["data_full.json"], "license": "CC-BY-3.0",
        "index": "CLINC150+OOS (5): Index uses test + oos_test; this file contains them, only train/oos_train are read",
    },
    "anli": {"repo": "facebook/anli", "rev": "8e4813d81f46d313dac7892e1c28076917cfcdf9",
             "files": [f"plain_text/train_r{i}-00000-of-00001.parquet" for i in (1, 2, 3)],
             "license": "CC-BY-NC-4.0", "index": "ANLI (12): Index uses test_r1-r3"},
    "hellaswag": {"repo": "Rowan/hellaswag", "rev": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  "files": ["data/train-00000-of-00001.parquet"], "license": "MIT",
                  "index": "HellaSwag (29): Index uses validation"},
    "winogrande": {"repo": "allenai/winogrande", "rev": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   "files": ["winogrande_xl/train-00000-of-00001.parquet"], "license": "CC-BY-4.0 (per AI2 release)",
                   "index": "WinoGrande (28): Index uses winogrande_xl validation"},
    "gsm8k": {"repo": "openai/gsm8k", "rev": "740312add88f781978c0658806c59bc2815b9866",
              "files": ["main/train-00000-of-00001.parquet"], "license": "MIT", "index": "GSM8K (30): Index uses test"},
    "when2call": {"repo": "nvidia/When2Call", "rev": "0582f7749df63a96fdc3070932e83e72396ace53",
                  "files": ["train/when2call_train_sft.jsonl"], "license": "CC-BY-4.0",
                  "index": "When2Call (62): Index uses test/when2call_test_mcq.jsonl"},
    "toolace": {"repo": "Team-ACE/ToolACE", "rev": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
                "files": ["data.json"], "license": "Apache-2.0", "index": "none (general tool-selection source)"},
    "sst5": {"repo": "SetFit/sst5", "rev": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
             "files": ["train.jsonl"], "license": "unspecified on HF; SST released for research use",
             "index": "none (ordinal 'score' source)"},
    "boolq": {"repo": "google/boolq", "rev": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "files": ["data/train-00000-of-00001.parquet"], "license": "CC-BY-SA-3.0",
              "index": "none (noul source)"},
    "nq": {"repo": "sentence-transformers/natural-questions", "rev": "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17",
           "files": ["pair/train-00000-of-00001.parquet"], "license": "CC-BY-SA-3.0 (Natural Questions)",
           "index": "none (retrieval-style source, up to 255 candidates)"},
    "mmlu_aux": {"repo": "cais/mmlu", "rev": "c30699e8356da336a370243923dbaf21066bb9fe",
                 "files": ["auxiliary_train/train-00000-of-00001.parquet"], "license": "MIT (constituents: ARC, "
                 "OBQA, RACE, MCTest; see SOURCES.md)", "index": "MMLU (24, display-only)/ARC (26/27, display-only): "
                 "auxiliary_train draws from ARC/OBQA/RACE train splits"},
}

LOCK = config.MANIFEST_DIR / "sources.lock.json"


def _url(src: dict, f: str) -> str:
    if "repo" in src:
        return HF.format(repo=src["repo"], rev=src["rev"], path=f)
    return src["url"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _read_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}


def local_path(sid: str, f: str) -> Path:
    return config.SOURCES_DIR / sid / Path(f).name


def fetch(sid: str) -> dict:
    src = REGISTRY[sid]
    lock = _read_lock()
    out = {}
    for f in src["files"]:
        dest = local_path(sid, f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = f"{sid}/{f}"
        if not dest.exists():
            url = _url(src, f)
            req = urllib.request.Request(url, headers={"User-Agent": "ssdm-data/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                n = int(r.headers.get("Content-Length") or 0)
                if n > MAX_BYTES:
                    raise RuntimeError(f"{key}: {n} bytes exceeds the light-build cap {MAX_BYTES}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    got = 0
                    for blk in iter(lambda: r.read(1 << 20), b""):
                        got += len(blk)
                        if got > MAX_BYTES:
                            raise RuntimeError(f"{key}: exceeded {MAX_BYTES} bytes")
                        fh.write(blk)
                tmp.replace(dest)
        sha = _sha256(dest)
        if key in lock and lock[key]["sha256"] != sha:
            raise RuntimeError(f"{key}: sha256 {sha} != locked {lock[key]['sha256']}")
        lock[key] = {"sha256": sha, "bytes": dest.stat().st_size, "url": _url(src, f),
                     "revision": src.get("rev", "branch head at fetch (sha256 pinned)"), "license": src["license"]}
        out[key] = lock[key]
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=1) + "\n", encoding="utf-8")
    return out


def fetch_all(ids=None) -> dict:
    res = {}
    for sid in ids or REGISTRY:
        res.update(fetch(sid))
    return res


def verified_path(sid: str, f: str) -> Path:
    """Path of a fetched file after checking its sha256 against the lock."""
    p = local_path(sid, f)
    lock = _read_lock()
    key = f"{sid}/{f}"
    if key not in lock:
        raise FileNotFoundError(f"{key} not fetched (python -X utf8 -I -m ssdm_data fetch {sid})")
    if _sha256(p) != lock[key]["sha256"]:
        raise RuntimeError(f"{key}: local file does not match sources.lock.json")
    return p
