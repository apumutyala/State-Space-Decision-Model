"""Install a SAVED m2 cell checkpoint from a job artifact into a checkpoint dir, every byte sha256-pinned (CPU, I/O only).

Shared by the eval-only first-cell modes (``first_cell_job.sh hijack_eval``, BOARD H1; ``fulldev_eval``): import
``install`` (or run the CLI) instead of re-implementing the fetch.

    python -X utf8 -m jobs.m2_first_cell.checkpoint_loader install --params <job_params_<mode>.json> \
        --cell-src <data>/checkpoint/cell.json --cell-config <mode cell config> --ck-dir <ck>/cell \
        --work <scratch> --cell-out <out>/cell.json [--url <artifact URL>] [--out <checkpoint.json>]

The job_params ``checkpoint`` block (``checkpoint_block`` validates it)::

    {"artifact": "art-xxxxx", "job": "job-xxxxx", "artifact_sha256": <sha of the artifact tar as downloaded>,
     "member": "./ckpt/ckpt-stepNNNNNNN.pt", "ckpt_sha256": ..., "latest_json_sha256": ...,
     "cell_json_sha256": <sha of the cell.json the training run wrote, shipped as data/checkpoint/cell.json>,
     "url_env": "CKPT_URL"}

An INFERENCE export (``train.ckpt_export``; what a first-cell run writes to ``/output/ckpt`` since job-igcu4's
over-cap upload) is accepted too: ``"member": "./ckpt/trainable-stepNNNNNNN.safetensors"`` plus ``"sidecar_sha256"``
(its ``trainable-stepNNNNNNN.json``, installed beside it); ``Trainer.load`` restores it for eval.

What ``install`` guarantees (else ``CheckpointError`` with ``code`` 5 = usage / 10 = identity):
  1. the shipped cell.json has the pinned sha256, its ``config`` EQUALS the mode's committed cell config, and its ladder
     has a T_max (``train.m2_gates`` restores the checkpoint through ``Trainer.load``, which then checks config hash +
     lineage: backbone digest, tokenizer, layout and training items);
  2. the artifact (``https://`` or ``file://``; URL from ``--url`` or ``$<url_env>``, never from the artifact) is
     streamed to disk with its sha256 checked against the pin; only the two named members are read from the tar (no
     extractall, no paths from the archive are used), each written under a fixed name and sha256-checked;
  3. ``latest.json`` names exactly the pinned checkpoint file, as its ``path`` (role ``latest``) or as its ``best``
     pointer (role ``best``: a ``best_ckpt`` run's scored checkpoint, pinned directly). When the pinned member is the
     last checkpoint of an inference export whose ``latest.json`` carries a ``best`` pointer, the best export is
     installed too (its safetensors + sidecar sha256s come from the pinned latest.json) and is what ``path`` names:
     eval-only modes score the SAME checkpoint the training run's own evals did (``Trainer.scored``). The record's
     ``role`` says which was installed.
A checkpoint dir that already holds the pinned file (a restart) is re-hashed and reused without a download.
The artifact is untrusted data: nothing from it is executed here (the .pt is only hashed; ``Trainer.load`` in the GPU
job unpickles it after this pin check, exactly as a training run's own checkpoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

KEYS = ("artifact", "artifact_sha256", "member", "ckpt_sha256", "latest_json_sha256", "cell_json_sha256", "url_env")
MEMBER_RE = re.compile(r"\./ckpt/(ckpt-step[0-9]+\.pt|trainable-step[0-9]+\.safetensors)")
CHUNK = 8 << 20


class CheckpointError(RuntimeError):
    def __init__(self, msg: str, code: int = 10):
        super().__init__(msg)
        self.code = code


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_block(params: dict) -> dict:
    """The validated ``checkpoint`` block of a job_params dict (``CheckpointError`` code 5 if absent / malformed)."""
    ck = params.get("checkpoint")
    if not isinstance(ck, dict):
        raise CheckpointError("job_params has no checkpoint block", 5)
    miss = [k for k in KEYS if not isinstance(ck.get(k), str) or not ck[k]]
    if miss:
        raise CheckpointError(f"checkpoint block missing {miss}", 5)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", ck["url_env"]) or not MEMBER_RE.fullmatch(ck["member"]):
        raise CheckpointError(f"checkpoint block url_env / member malformed: {ck['url_env']!r} {ck['member']!r}", 5)
    shas = ["artifact_sha256", "ckpt_sha256", "latest_json_sha256", "cell_json_sha256"]
    if ck["member"].endswith(".safetensors"):
        shas.append("sidecar_sha256")
        if not isinstance(ck.get("sidecar_sha256"), str):
            raise CheckpointError("checkpoint block: an inference-export member needs sidecar_sha256", 5)
    for k in shas:
        if not re.fullmatch(r"[0-9a-f]{64}", ck[k]):
            raise CheckpointError(f"checkpoint.{k} is not a sha256", 5)
    return ck


def check_cell(cell_src: str | Path, cell_config: str | Path, pin: str) -> dict:
    if not Path(cell_src).is_file() or sha256_file(cell_src) != pin:
        raise CheckpointError(f"shipped cell.json {cell_src} missing or sha256 != checkpoint.cell_json_sha256")
    cell = json.loads(Path(cell_src).read_text(encoding="utf-8"))
    cfg = json.loads(Path(cell_config).read_text(encoding="utf-8"))
    if cell.get("config") != cfg:
        diff = sorted(k for k in set(cell.get("config") or {}) | set(cfg) if (cell.get("config") or {}).get(k) != cfg.get(k))
        raise CheckpointError(f"checkpoint cell.json config != {cell_config} on {diff[:8]}")
    if not (cell.get("ladder") or {}).get("T_max"):
        raise CheckpointError("checkpoint cell.json has no ladder.T_max")
    return cell


def download(url: str, dst: Path) -> str:
    """Stream ``url`` (https:// or file://) to ``dst``; returns its sha256."""
    if not url.startswith(("https://", "file://")):
        raise CheckpointError("the artifact URL must be https:// (get_artifact) or file:// (tests)", 5)
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:  # noqa: S310 - scheme checked above
            for b in iter(lambda: r.read(CHUNK), b""):
                h.update(b)
                f.write(b)
    except OSError as e:
        raise CheckpointError(f"artifact download failed: {type(e).__name__}: {e}") from e
    return h.hexdigest()


def _extract(tar_path: Path, member: str, dst: Path, want: str, what: str) -> None:
    """Copy ONE named regular-file member to ``dst`` (a fixed path), installed only if its sha256 == ``want``."""
    with tarfile.open(tar_path) as t:
        names = {member, member.removeprefix("./")}
        m = next((x for x in t.getmembers() if x.name in names), None)
        if m is None or not m.isfile():
            raise CheckpointError(f"artifact has no regular file {member}")
        src = t.extractfile(m)
        h = hashlib.sha256()
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as f:
            for b in iter(lambda: src.read(CHUNK), b""):
                h.update(b)
                f.write(b)
    if h.hexdigest() != want:
        raise CheckpointError(f"{what} sha256 {h.hexdigest()} != the pin (left at {tmp.name}, not installed)")
    os.replace(tmp, dst)


def install(block: dict, *, cell_src: str | Path, cell_config: str | Path, ck_dir: str | Path, work: str | Path,
            cell_out: str | Path | None = None, url: str | None = None) -> dict[str, Any]:
    """Verify + install the pinned checkpoint into ``ck_dir`` (``<file>.pt`` + ``latest.json``) and copy the verified
    cell.json to ``cell_out``. Returns the identity record (also what the job writes as out/checkpoint.json)."""
    ck = checkpoint_block({"checkpoint": block})
    check_cell(cell_src, cell_config, ck["cell_json_sha256"])
    ck_dir, work = Path(ck_dir), Path(work)
    ck_dir.mkdir(parents=True, exist_ok=True)
    fname = ck["member"].rsplit("/", 1)[1]
    pt, latest = ck_dir / fname, ck_dir / "latest.json"
    side = None  # an inference export's JSON sidecar (train.ckpt_export), installed beside the .safetensors
    if fname.endswith(".safetensors"):
        side = ck_dir / (fname[: -len(".safetensors")] + ".json")
    reused = pt.is_file() and sha256_file(pt) == ck["ckpt_sha256"] and latest.is_file() \
        and sha256_file(latest) == ck["latest_json_sha256"] \
        and (side is None or (side.is_file() and sha256_file(side) == ck["sidecar_sha256"]))
    if not reused:
        url = url or os.environ.get(ck["url_env"], "")
        if not url:
            raise CheckpointError(f"${ck['url_env']} is not set (the get_artifact {ck['artifact']} URL)", 5)
        tar = work / "artifact.tar"
        got = download(url, tar)
        if got != ck["artifact_sha256"]:
            raise CheckpointError(f"artifact {ck['artifact']} sha256 {got} != checkpoint.artifact_sha256")
        _extract(tar, "./ckpt/latest.json", latest, ck["latest_json_sha256"], "latest.json")
        _extract(tar, ck["member"], pt, ck["ckpt_sha256"], fname)
        if side is not None:
            _extract(tar, f"./ckpt/{side.name}", side, ck["sidecar_sha256"], side.name)
    rec = json.loads(latest.read_text(encoding="utf-8"))
    best = rec.get("best") if isinstance(rec.get("best"), dict) else None
    if rec.get("path") == fname:
        role = "latest"
    elif best is not None and best.get("path") == fname:
        role = "best"  # the scored checkpoint itself is pinned
    else:
        raise CheckpointError(f"latest.json names neither {fname} as its path nor as its best pointer")
    scored, scored_sha = pt, ck["ckpt_sha256"]
    if role == "latest" and best is not None and best.get("path") != fname and fname.endswith(".safetensors"):
        # a best_ckpt run: install its scored (best) export too, pinned through the pinned latest.json
        bname = str(best.get("path") or "")
        if not re.fullmatch(r"trainable-step[0-9]+\.safetensors", bname) or not all(
                re.fullmatch(r"[0-9a-f]{64}", str(best.get(k) or "")) for k in ("sha256", "sidecar_sha256")):
            raise CheckpointError(f"latest.json best pointer malformed: {best!r}")
        bpt, bside = ck_dir / bname, ck_dir / (bname[: -len(".safetensors")] + ".json")
        have = (bpt.is_file() and sha256_file(bpt) == best["sha256"] and bside.is_file()
                and sha256_file(bside) == best["sidecar_sha256"])
        if not have:
            tar = work / "artifact.tar"
            if not tar.is_file():
                url = url or os.environ.get(ck["url_env"], "")
                if not url:
                    raise CheckpointError(f"${ck['url_env']} is not set (the get_artifact {ck['artifact']} URL)", 5)
                got = download(url, tar)
                if got != ck["artifact_sha256"]:
                    raise CheckpointError(f"artifact {ck['artifact']} sha256 {got} != checkpoint.artifact_sha256")
            _extract(tar, f"./ckpt/{bname}", bpt, best["sha256"], bname)
            _extract(tar, f"./ckpt/{bside.name}", bside, best["sidecar_sha256"], bside.name)
        scored, scored_sha, role = bpt, best["sha256"], "best"
    if cell_out is not None:
        Path(cell_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cell_src, cell_out)
    return {"artifact": ck["artifact"], "job": ck.get("job"), "artifact_sha256": ck["artifact_sha256"],
            "file": scored.name, "path": str(scored), "sha256": scored_sha, "role": role, "member": fname,
            "member_sha256": ck["ckpt_sha256"], "cell_json_sha256": ck["cell_json_sha256"], "reused": reused}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.checkpoint_loader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--params", required=True)
    i.add_argument("--cell-src", required=True)
    i.add_argument("--cell-config", required=True)
    i.add_argument("--ck-dir", required=True)
    i.add_argument("--work", required=True)
    i.add_argument("--cell-out", default=None)
    i.add_argument("--url", default=None, help="default: $<checkpoint.url_env>")
    i.add_argument("--out", default=None, help="write the identity record here")
    a = ap.parse_args(argv)
    try:
        block = checkpoint_block(json.loads(Path(a.params).read_text(encoding="utf-8")))
        rec = install(block, cell_src=a.cell_src, cell_config=a.cell_config, ck_dir=a.ck_dir, work=a.work,
                      cell_out=a.cell_out, url=a.url)
    except CheckpointError as e:
        print(f"checkpoint: {e}", file=sys.stderr)
        return e.code
    text = json.dumps(rec, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
