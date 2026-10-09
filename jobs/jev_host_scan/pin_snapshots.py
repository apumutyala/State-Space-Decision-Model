"""Pin every candidate's snapshot (network; CPU; run once when the candidate list changes).

    python -m jobs.jev_host_scan.pin_snapshots [--tokenizers <root>] [--only id,id]

For each candidate in jobs/jev_host_scan/candidates.json: the HF file list AT THE PINNED REVISION, filtered by its
allow_patterns (huggingface_hub's fnmatch rule), one ``<sha256>  <file>`` line per file into its snapshot_pin. Weight
files (*.safetensors) take the sha256 the Hub records for the LFS object (= the file's sha256); all other files are downloaded
at the revision into ``<tokenizers root>/<Org>--<Name>@<revision>/`` (tokenizer, config and template files, the readout
code's local identity source, never weights) and hashed. The job re-hashes every downloaded file against these lines.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "jobs/jev_host_scan/candidates.json"
API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def tok_dir_name(repo: str, rev: str) -> str:
    return f"{repo.replace('/', '--')}@{rev}"


def allowed(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ssdm-jev-host-scan"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def pin_one(c: dict, tok_root: Path) -> dict:
    info = json.loads(_get(API.format(repo=c["repo"], rev=c["revision"])))
    if info.get("sha") != c["revision"]:
        raise SystemExit(f"{c['id']}: the Hub resolved {c['revision']} to {info.get('sha')}")
    files = [s for s in info["siblings"] if allowed(s["rfilename"], c["allow_patterns"])]
    d = tok_root / tok_dir_name(c["repo"], c["revision"])
    lines, nbytes = [], 0
    for s in sorted(files, key=lambda s: s["rfilename"]):
        name = s["rfilename"]
        lfs = s.get("lfs") or {}
        if lfs.get("sha256") and name.endswith(".safetensors"):
            sha = lfs["sha256"]
        else:  # every non-weight file, LFS or not (tokenizer.json is LFS in several repos)
            b = _get(RESOLVE.format(repo=c["repo"], rev=c["revision"], path=name))
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(b)
            sha = hashlib.sha256(b).hexdigest()
        nbytes += int(s.get("size") or 0)
        lines.append(f"{sha}  {name}")
    out = ROOT / c["snapshot_pin"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(lines) + "\n").encode())
    return {"id": c["id"], "files": len(lines), "bytes": nbytes, "tokenizer_dir": str(d)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizers", default=os.environ.get("SSDM_TOKENIZERS", str(Path(os.environ.get(
        "SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data")) / "tokenizers")))
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = set(filter(None, a.only.split(",")))
    for c in json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]:
        if not only or c["id"] in only:
            print(json.dumps(pin_one(c, Path(a.tokenizers))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
