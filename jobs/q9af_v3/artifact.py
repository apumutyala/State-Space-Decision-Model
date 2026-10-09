"""Chain job A -> job B (and ship v2's export as the reference arm): sha256-pinned job artifacts.

    python -m jobs.q9af_v3.artifact fill  --params jobs/q9af_v3/job_params_B.json --which stageA|v2 \
        --tar <the downloaded artifact tar> --artifact art-xxxxx --job job-xxxxx
    python -X utf8 -m jobs.q9af_v3.artifact fetch --params <job_params_B.json> --which stageA|v2 --dest <dir> [--url U]
    python -m jobs.q9af_v3.artifact status --params <job_params_B.json>

``fill`` (laptop, after job A ends / for v2 after job-7ji99 ends): hashes the whole tar and each pinned member and
writes them into the params block (``stageA`` / ``v2``); commit that, push, re-package job B. ``fetch`` (in job B): the
signed URL comes ONLY from the environment (``url_env``, filled into submit.json at submit time, never committed);
the tar is streamed to disk and must match ``artifact_sha256``; only the named members are read (never extractall,
never a path from the archive), each written under its fixed basename and installed only if its sha256 matches.
Nothing from an artifact is executed here (the .pt files are only hashed; job B loads them after this check).
Exit codes: 0 ok, 5 usage / unpinned, 10 identity (sha256 mismatch, missing member).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

from jobs.m2_first_cell.checkpoint_loader import CheckpointError, _extract, download, sha256_file

#: members each artifact must contain (archive paths as the platform tars /output)
MEMBERS = {
    "stageA": ["./stageA/gdn_converted.pt", "./stageA/export.json", "./stageA/lc.u32", "./stageA/onpolicy.npz",
               "./stageA/gen_heldout.npz", "./stageA/heldout_web.u32", "./stageA/heldout_chat.u32",
               "./stageA/heldout_math.u32", "./stageA/mix.json", "./stageA/split.json", "./stageA/results_A.json",
               "./stageA/gen.json"],
    "v2": ["./gdn_converted.pt", "./export.json"],
}
URL_ENV = {"stageA": "STAGEA_URL", "v2": "V2_URL"}


def block(params: dict, which: str) -> dict:
    b = params.get(which)
    if not isinstance(b, dict):
        raise CheckpointError(f"job_params has no {which} block", 5)
    return b


def pinned(b: dict) -> bool:
    return isinstance(b.get("artifact_sha256"), str) and len(b["artifact_sha256"]) == 64 \
        and isinstance(b.get("members"), dict) and all(isinstance(v, str) and len(v) == 64 for v in b["members"].values())


def _member_sha(t: tarfile.TarFile, name: str) -> str:
    names = {name, name.removeprefix("./")}
    m = next((x for x in t.getmembers() if x.name in names), None)
    if m is None or not m.isfile():
        raise CheckpointError(f"artifact has no regular file {name}")
    h = hashlib.sha256()
    f = t.extractfile(m)
    for b in iter(lambda: f.read(8 << 20), b""):
        h.update(b)
    return h.hexdigest()


def fill(params_path: str, which: str, tar_path: str, artifact: str, job: str) -> dict:
    p = json.loads(Path(params_path).read_text(encoding="utf-8"))
    b = block(p, which)
    with tarfile.open(tar_path) as t:
        members = {m: _member_sha(t, m) for m in MEMBERS[which]}
    b.update({"artifact": artifact, "job": job, "artifact_sha256": sha256_file(tar_path), "members": members,
              "url_env": URL_ENV[which]})
    # the signed URL never enters git: a placeholder that sf_submit fills from the submitting shell (refused if unset)
    p.setdefault("submit", {}).setdefault("env", {})[URL_ENV[which]] = f"<get_artifact {artifact} url: fill at submit>"
    Path(params_path).write_text(json.dumps(p, indent=1) + "\n", encoding="utf-8", newline="\n")
    return b


def fetch(params: dict, which: str, dest: str | Path, url: str | None = None, work: str | Path | None = None) -> dict:
    b = block(params, which)
    if not pinned(b):
        raise CheckpointError(f"{which}: artifact not pinned in job_params (run `artifact fill` and re-package)", 5)
    if sorted(b["members"]) != sorted(MEMBERS[which]):
        raise CheckpointError(f"{which}: pinned members {sorted(b['members'])} != {MEMBERS[which]}", 5)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    have = all((dest / Path(m).name).is_file() and sha256_file(dest / Path(m).name) == s for m, s in b["members"].items())
    if not have:
        url = url or os.environ.get(b.get("url_env") or URL_ENV[which], "")
        if not url or url.startswith("<"):
            raise CheckpointError(f"${b.get('url_env') or URL_ENV[which]} is not set (get_artifact {b.get('artifact')})", 5)
        tar = Path(work or dest) / f"{which}.artifact.tar"
        got = download(url, tar)
        if got != b["artifact_sha256"]:
            raise CheckpointError(f"{which} artifact sha256 {got} != pin {b['artifact_sha256']}")
        for m, s in b["members"].items():
            _extract(tar, m, dest / Path(m).name, s, m)
        tar.unlink()
    return {"which": which, "artifact": b.get("artifact"), "job": b.get("job"), "artifact_sha256": b["artifact_sha256"],
            "members": b["members"], "dest": str(dest), "reused": have}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.q9af_v3.artifact")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill")
    g = sub.add_parser("fetch")
    s = sub.add_parser("status")
    for x in (f, g, s):
        x.add_argument("--params", required=True)
    for x in (f, g):
        x.add_argument("--which", choices=sorted(MEMBERS), required=True)
    f.add_argument("--tar", required=True)
    f.add_argument("--artifact", required=True)
    f.add_argument("--job", required=True)
    g.add_argument("--dest", required=True)
    g.add_argument("--url", default=None)
    g.add_argument("--work", default=None)
    g.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "fill":
            rec = fill(a.params, a.which, a.tar, a.artifact, a.job)
        elif a.cmd == "status":
            p = json.loads(Path(a.params).read_text(encoding="utf-8"))
            rec = {w: {"pinned": pinned(p.get(w) or {}), "artifact": (p.get(w) or {}).get("artifact")} for w in MEMBERS}
        else:
            rec = fetch(json.loads(Path(a.params).read_text(encoding="utf-8")), a.which, a.dest, a.url, a.work)
    except CheckpointError as e:
        print(f"artifact: {e}", file=sys.stderr)
        return e.code
    text = json.dumps(rec, indent=1)
    if getattr(a, "out", None):
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
