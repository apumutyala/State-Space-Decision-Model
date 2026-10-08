"""Compute (and optionally write) a saved-checkpoint job_params ``checkpoint`` block from a DOWNLOADED artifact (CPU, I/O).

    python -I -X utf8 scripts/fill_ckpt_pins.py <artifact tar | dir holding it | job-xxxxx> \
        --params jobs/m2_first_cell/job_params_readout_ab_d3.json [--artifact art-xxxxx] [--cell <cell.json>] [--write]

Input: the artifact tar exactly as downloaded (get_artifact), a directory holding one ``*.tar`` / ``*.tar.gz`` /
``*.tgz`` (and optionally the run's ``cell.json``), or a job id, resolved to ``$SSDM_DATA/m2/checkpoints/<job>/``
(default ``SSDM_DATA`` = ``~/research/CHLabs/ssdm-data``). Computes, the way ``jobs.m2_first_cell.checkpoint_loader``
checks them in the job:
  artifact_sha256      sha256 of the tar file as downloaded
  member               ./ckpt/<file> named by the tar's ./ckpt/latest.json "path" (must match ./ckpt/ckpt-stepN.pt)
  ckpt_sha256          sha256 of that member;  latest_json_sha256: sha256 of ./ckpt/latest.json
  cell_json_sha256     sha256 of the run's cell.json: ``--cell``, else ``<dir>/cell.json``, else the tar's own
                       cell.json member (top-level ./cell.json or ./out/cell.json), copied to
                       ``$SSDM_DATA/m2/checkpoints/<job>/cell.json`` (the context data source; never overwritten when
                       it differs)
and checks cell.json["config"] == the params' cell_config (the job refuses otherwise) and T_max present.
``--write`` rewrites the params' checkpoint block (+ the context's cell.json source) in place; without it the block is
printed. The artifact is untrusted data: members are only hashed (and cell.json parsed as JSON), never extracted by
their own paths, never executed. No URL is read or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMBER_RE = re.compile(r"\./ckpt/ckpt-step[0-9]+\.pt")
CHUNK = 8 << 20


def data_root() -> Path:
    return Path(os.environ.get("SSDM_DATA") or Path.home() / "research/CHLabs/ssdm-data")


def sha_stream(f) -> str:
    h = hashlib.sha256()
    for b in iter(lambda: f.read(CHUNK), b""):
        h.update(b)
    return h.hexdigest()


def resolve(src: str) -> tuple[Path, Path | None, str | None]:
    """(tar path, a cell.json next to it or None, job id or None)."""
    job = src if re.fullmatch(r"job-[a-z0-9]+", src) else None
    p = data_root() / "m2" / "checkpoints" / src if job else Path(src)
    if p.is_file():
        return p, (p.parent / "cell.json" if (p.parent / "cell.json").is_file() else None), job
    if not p.is_dir():
        raise SystemExit(f"{p}: no such file or directory")
    tars = sorted(x for x in p.iterdir() if x.is_file() and x.name.endswith((".tar", ".tar.gz", ".tgz")))
    if len(tars) != 1:
        raise SystemExit(f"{p}: want exactly one artifact tar, found {[t.name for t in tars]}")
    return tars[0], (p / "cell.json" if (p / "cell.json").is_file() else None), job


def _member(t: tarfile.TarFile, *names: str):
    want = set(names) | {n.removeprefix("./") for n in names}
    return next((m for m in t.getmembers() if m.name in want and m.isfile()), None)


def pins(tar: Path, cell: Path | None) -> tuple[dict, bytes]:
    with open(tar, "rb") as f:
        art = sha_stream(f)
    with tarfile.open(tar) as t:
        lm = _member(t, "./ckpt/latest.json")
        if lm is None:
            raise SystemExit("artifact has no regular file ./ckpt/latest.json")
        latest = t.extractfile(lm).read()
        name = json.loads(latest).get("path")
        member = f"./ckpt/{name}"
        if not isinstance(name, str) or not MEMBER_RE.fullmatch(member):
            raise SystemExit(f"latest.json path {name!r} is not ckpt-stepN.pt")
        cm = _member(t, member)
        if cm is None:
            raise SystemExit(f"artifact has no regular file {member}")
        ck = sha_stream(t.extractfile(cm))
        if cell is not None:
            cell_bytes = cell.read_bytes()
        else:
            m = _member(t, "./cell.json", "./out/cell.json")
            if m is None:
                raise SystemExit("no cell.json: pass --cell (the run's out/cell.json)")
            cell_bytes = t.extractfile(m).read()
    return {"artifact_sha256": art, "member": member, "ckpt_sha256": ck,
            "latest_json_sha256": hashlib.sha256(latest).hexdigest(),
            "cell_json_sha256": hashlib.sha256(cell_bytes).hexdigest()}, cell_bytes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fill_ckpt_pins.py")
    ap.add_argument("src", help="artifact tar, a dir holding it, or a job id ($SSDM_DATA/m2/checkpoints/<job>/)")
    ap.add_argument("--params", required=True)
    ap.add_argument("--artifact", default=None, help="the artifact id (art-xxxxx); default: keep the params' value")
    ap.add_argument("--job", default=None, help="the training job id; default: the src job id or the params' value")
    ap.add_argument("--cell", default=None, help="the run's cell.json (default: next to the tar, else in the tar)")
    ap.add_argument("--write", action="store_true", help="rewrite the params file's checkpoint block in place")
    a = ap.parse_args(argv)
    tar, cell, job = resolve(a.src)
    got, cell_bytes = pins(tar, Path(a.cell) if a.cell else cell)
    params_path = Path(a.params)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / params["cell_config"]).read_text(encoding="utf-8"))
    run_cell = json.loads(cell_bytes)
    if run_cell.get("config") != cfg:
        diff = sorted(k for k in set(run_cell.get("config") or {}) | set(cfg) if (run_cell.get("config") or {}).get(k) != cfg.get(k))
        raise SystemExit(f"cell.json config != {params['cell_config']} on {diff[:8]} (the job would refuse it)")
    if not (run_cell.get("ladder") or {}).get("T_max"):
        raise SystemExit("cell.json has no ladder.T_max (the job would refuse it)")
    block = dict(params.get("checkpoint") or {})
    job = a.job or job or block.get("job")
    art = a.artifact or block.get("artifact")
    if not (isinstance(art, str) and re.fullmatch(r"art-[a-z0-9]+", art)):
        raise SystemExit(f"artifact id {art!r}: pass --artifact art-xxxxx")
    block.update({"artifact": art, "job": job, **got})
    block.setdefault("url_env", "CKPT_URL")
    rel = f"m2/checkpoints/{job}/cell.json"
    dst = data_root() / rel
    if dst.is_file() and dst.read_bytes() != cell_bytes:
        raise SystemExit(f"{dst} exists and differs from this run's cell.json; not overwritten")
    out = {"checkpoint": block, "context_cell_json": rel, "tar": str(tar)}
    if a.write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            dst.write_bytes(cell_bytes)
        block["_note"] = (block.get("_note", "").split(" PINS FILLED")[0]
                          + f" PINS FILLED by scripts/fill_ckpt_pins.py from {tar.name}.")
        params["checkpoint"] = block
        for d in (params.get("context") or {}).get("data", []):
            if d.get("dest") == "checkpoint/cell.json":
                d["src"] = rel
        env = ((params.get("submit") or {}).get("env") or {})
        if "CKPT_URL" in env:
            env["CKPT_URL"] = f"<get_artifact {art} url: fill at submit>"
        params_path.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
