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
GH = "https://raw.githubusercontent.com"

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
                   "files": ["winogrande_xl/train-00000-of-00001.parquet"], "license": "Apache-2.0 (allenai/winogrande GitHub LICENSE); HF card: none stated",
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
           "files": ["pair/train-00000-of-00001.parquet"], "license": "HF card: none stated; upstream Natural Questions CC-BY-SA-3.0",
           "index": "none (retrieval-style source, up to 255 candidates)"},
    "mmlu_aux": {"repo": "cais/mmlu", "rev": "c30699e8356da336a370243923dbaf21066bb9fe",
                 "files": ["auxiliary_train/train-00000-of-00001.parquet"], "license": "MIT (constituents: ARC, "
                 "OBQA, RACE, MCTest; see SOURCES.md)", "index": "MMLU (24, display-only)/ARC (26/27, display-only): "
                 "auxiliary_train draws from ARC/OBQA/RACE train splits"},
    # ---- added for the 150k build (Track D m2-data-150k); same rules: train splits only, sha256-locked
    # Glaive FC v2 ships one 271 MB JSON (over MAX_BYTES); the Hub's parquet conversion of the same main-branch
    # commit (e7f4b64) is pinned instead: refs/convert/parquet @ 8862406 (columns: system, chat).
    "glaive_fc": {"repo": "glaiveai/glaive-function-calling-v2", "rev": "8862406ff857565709af3dca7fed589b84edf235",
                  "files": ["default/train/0000.parquet"], "license": "Apache-2.0",
                  "index": "BFCL (1): no train split; Glaive FC v2 is a proxy tool-selection source (single train file)"},
    # SemEval-2020 Task 7 (Humicroedit) train splits via the tasksource parquet mirror; Index uses subtask-2 TEST
    "humicroedit_s1": {"repo": "tasksource/humicroedit", "rev": "f5a16e65b0854032ab9b4c82ca182a6381b83bd5",
                       "files": ["subtask-1/train-00000-of-00001.parquet"],
                       "license": "unspecified on HF; SemEval-2020 T7 research release (verify)",
                       "index": "Humicroedit (21): Index uses subtask-2 test; headlines recur across subtasks"},
    "humicroedit_s2": {"repo": "tasksource/humicroedit", "rev": "f5a16e65b0854032ab9b4c82ca182a6381b83bd5",
                       "files": ["subtask-2/train.parquet"],
                       "license": "unspecified on HF; SemEval-2020 T7 research release (verify)",
                       "index": "Humicroedit (21): Index uses subtask-2 test; headlines recur across subtasks"},
    # ---- added for b1-draw100k-v3 (2026-10-08; operator-approved official TRAIN splits). Each source lives in its
    # own directory under SOURCES_DIR/dl-2026-10-08/<name>/ ("dir"); "urls" maps each file to its pinned URL. Data
    # only: nothing there is executed; zip / sqlite / parquet are read as data by data/b1_v3.py (python -I).
    "acos": {"dir": "dl-2026-10-08/acos", "rev": "45d179a3dcc6a3dedd848d81b16f2552454805fe",
             "urls": {f: f"{GH}/NUSTM/ACOS/45d179a3dcc6a3dedd848d81b16f2552454805fe/data/{f}"
                      for f in ("Laptop-ACOS/laptop_quad_train.tsv", "Restaurant-ACOS/rest16_quad_train.tsv")},
             "license": "none stated in NUSTM/ACOS (no LICENSE file; research release, ACL 2021); no clause forbids training",
             "index": "ACOS (38): Index uses laptop_quad_test + rest16_quad_test (1,399 reviews) at the same commit"},
    "nli4ct": {"dir": "dl-2026-10-08/nli4ct", "rev": "7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e",
               "urls": {"training_data.zip": f"{GH}/ai-systems/Task-2-SemEval-2024/7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e/training_data.zip"},
               "license": "none stated in the repo (SemEval-2024 Task 2 research release); trials are public ClinicalTrials.gov records",
               "index": "NLI4CT (42): Index uses test.json (5,500) at the same commit; this zip = train.json + dev.json + the "
                        "shared CT corpus (Index trial_archive_sha256); only train.json statements are read"},
    "vast": {"dir": "dl-2026-10-08/vast", "rev": "e7c4775182b184730f350995f8260579c9e066fe",
             "urls": {"vast_train.csv": f"{GH}/emilyallaway/zero-shot-stance/e7c4775182b184730f350995f8260579c9e066fe/data/VAST/vast_train.csv"},
             "license": "none stated in the repo (research release, ACL 2020; NYT comments)",
             "index": "VAST (41): Index uses vast_test.csv (3,006)"},
    "ragtruth": {"dir": "dl-2026-10-08/ragtruth", "rev": "c103204b9ce28d6bbad859304bf30de72b8ed8fe",
                 "urls": {f: f"{GH}/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/{f}"
                          for f in ("response.jsonl", "source_info.jsonl")},
                 "license": "MIT",
                 "index": "RAGTruth (59): Index uses the 2,700 split=test responses; response.jsonl holds both splits and "
                          "only split=train rows are read (source_ids of the two splits are disjoint)"},
    "hover": {"dir": "dl-2026-10-08/hover", "rev": "39b84697f196308f398a251a7aea9b82ae0f0562", "max_bytes": 2_300_000_000,
              "urls": {"hover_train_release_v1.1.json": f"{GH}/hover-nlp/hover/39b84697f196308f398a251a7aea9b82ae0f0562/data/hover/hover_train_release_v1.1.json",
                       "wiki_wo_links.db": "https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db"},
              "license": "CC-BY-SA-4.0 (data, hover-nlp.github.io); code MIT; wiki_wo_links.db = the official HoVer "
                         "Wikipedia dump (download_data.sh), read only for the supporting-fact documents",
              "index": "HoVer (61): Index uses 4,000 labelled dev claims (test labels are not public)"},
    "habermas": {"dir": "dl-2026-10-08/habermas", "rev": "7923b71966c14136077c78d7841bc9e1a182dfe0", "max_bytes": 450_000_000,
                 "urls": {"hm_all_candidate_comparisons.parquet": "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"},
                 "license": "CC-BY-4.0 (data; google-deepmind/habermas_machine README @7923b71); code Apache-2.0",
                 "index": "Habermas Machine (50): Index uses EVAL_* cohorts (question.split IID_TEST / OOD_TEST) of this same "
                          "file (Index source_sha256 7cf8d5ce...); only TRAINING_DATA_V1-V4 rows with question.split TRAIN are read"},
    "mnli": {"repo": "nyu-mll/multi_nli", "rev": "da70db2af9d09693783c3320c4249840212ee221", "dir": "dl-2026-10-08/mnli",
             "files": ["data/train-00000-of-00001.parquet"],
             "license": "HF card: cc-by-3.0 / cc-by-sa-3.0 / mit / other (OANC + fiction genres; see MultiNLI paper)",
             "index": "none (MNLI is not an Index catalog; generic NLI source; fingerprint-checked against every Index row)"},
    # ---- added for bschema-draw-v4 (2026-10-09; data/b_schema_levers_v4.py). Own directories under dl-2026-10-09/.
    "esci": {"dir": "dl-2026-10-09/esci", "rev": "8113b17a5d4099e20243282c926f1bc1a08a4d13",
             "urls": {"train-00000-of-00011.parquet": "https://huggingface.co/datasets/tasksource/esci/resolve/"
                      "8113b17a5d4099e20243282c926f1bc1a08a4d13/data/train-00000-of-00011-2d36455632bef8a2.parquet"},
             "license": "Apache-2.0 (tasksource/esci HF card; upstream amazon-science/esci-data Apache-2.0)",
             "index": "Amazon-ESCI (37): Index uses the 5,000 released TEST query-product pairs; this shard is the "
                      "official train split (ESCI splits by query); pairs whose product_id or query is in any Index "
                      "ESCI row are dropped (scripts/b_schema_v4_export.py)"},
    "phish_curated": {"dir": "dl-2026-10-09/phish_curated", "rev": "539567cee21d03c83bd2fb1b142c18a80904eb72",
                      "urls": {f: "https://huggingface.co/datasets/kudzaiprichard/aura-phishing-email-corpus/resolve/"
                               f"539567cee21d03c83bd2fb1b142c18a80904eb72/{f}"
                               for f in ("CEAS_08.csv", "Nazario.csv", "SpamAssasin.csv")},
                      "license": "CC-BY-4.0 (Zenodo 10.5281/zenodo.8339691; the mirror is byte-identical, MD5s match "
                                 "the Zenodo API)",
                      "index": "none: PhishNChips (56) has no train split and its Index rows are generated emails "
                               "(links from a live phishing-URL database); this public corpus has the same task shape"},
}

LOCK = config.MANIFEST_DIR / "sources.lock.json"


def files(src: dict) -> list[str]:
    return list(src.get("urls") or src["files"])


def _url(src: dict, f: str) -> str:
    if "urls" in src:
        return src["urls"][f]
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
    return config.SOURCES_DIR / REGISTRY.get(sid, {}).get("dir", sid) / Path(f).name


def fetch(sid: str) -> dict:
    src = REGISTRY[sid]
    lock = _read_lock()
    out = {}
    cap = src.get("max_bytes", MAX_BYTES)
    for f in files(src):
        dest = local_path(sid, f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = f"{sid}/{f}"
        if not dest.exists():
            url = _url(src, f)
            req = urllib.request.Request(url, headers={"User-Agent": "ssdm-data/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                n = int(r.headers.get("Content-Length") or 0)
                if n > cap:
                    raise RuntimeError(f"{key}: {n} bytes exceeds the cap {cap}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    got = 0
                    for blk in iter(lambda: r.read(1 << 20), b""):
                        got += len(blk)
                        if got > cap:
                            raise RuntimeError(f"{key}: exceeded {cap} bytes")
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
