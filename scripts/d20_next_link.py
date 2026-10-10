"""D20 chain hand-off: package and submit link k+1 the minute link k lands (JOB_SPEC.md 'D20', train/chain.py).

    python -I -X utf8 scripts/d20_next_link.py <finished link-k job id> [--dry] [--full-hash] [--out DIR] [--data DIR]

1. ``autor job get <id> --json``: status succeeded, result.decision is a valid chain-link decision
   (train.chain.valid_result, verdict link_done) for link k of 3; artifact ready -> artifact id, artifact.sha256, the
   presigned artifact.download.url, decision.manifest_sha256.
2. Verify the artifact holds a RESUMABLE checkpoint before anything is submitted. The tar is read with HTTP Range
   requests (no 4.7 GB download): ./chain/CHAIN.json must hash to decision.manifest_sha256 with chain id = the shared
   cell config's, link k, of 3; latest.json / cell.json hash to CHAIN.json's members and agree on step / config_hash;
   cell.json config == the shared cell config; the latest ckpt-step*.pt (a torch zip) has its data.pkl unpickled with
   every global stubbed (no torch needed, no code runs): step == CHAIN.json step, config_hash, non-empty AdamW
   optimizer state (exp_avg + exp_avg_sq for every param), scheduler state, rng, epoch, cursor. ``--full-hash`` also
   streams the whole artifact through sha256 (== artifact.sha256; the job re-checks it and every member on install).
3. Fill the __FILL_FROM_LINK<k>_*__ placeholders of job_params_d20_v4_chain_l<k+1>.json (text substitution, format
   kept), re-validate (train.chain.chain_block + the sha's own jobs/common/lint.py checks).
4. Real run: commit (Apuroop Mutyala) + push to stack/rwkv7/d20-v4-chain, package with jobs.common.launch from the
   pushed sha, then ``python -I jobs/common/sf_submit.py <out> --dry`` and the real submit, with D20_CHAIN_URL (the
   presigned URL) only in the child process environment (never on disk, never printed).
   ``--dry``: steps 1-3 only (the params file is restored afterwards); prints the commands the real run would execute
   and, if ``--out`` already holds a package (submit.json), runs sf_submit --dry on it.
"""
from __future__ import annotations

import argparse
import builtins
import collections
import datetime as dt
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the script dir; the repo's own modules are trusted code

from train.chain import CKPT_RE, ChainError, chain_block, chain_id, valid_result  # noqa: E402

BRANCH = "stack/rwkv7/d20-v4-chain"
OF = 3
CELL = "jobs/m2_first_cell/m2_first_cell_rwkv7_d20_v4_chain.json"
PARAMS = "jobs/m2_first_cell/job_params_d20_v4_chain_l{k}.json"
URL_ENV = "D20_CHAIN_URL"
AUTOR = ["wsl.exe", "-d", "bi-ubuntu", "--", "/root/.local/bin/autor", "--workspace", "default"]
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
AUTHOR = {"GIT_AUTHOR_NAME": "Apuroop Mutyala", "GIT_AUTHOR_EMAIL": "apumutyala@gmail.com",
          "GIT_COMMITTER_NAME": "Apuroop Mutyala", "GIT_COMMITTER_EMAIL": "apumutyala@gmail.com"}
HEX64 = re.compile(r"[0-9a-f]{64}")
BLOCK = 1 << 20


class HandoffError(RuntimeError):
    pass


def say(msg: str) -> None:
    print(f"[d20_next_link] {msg}", flush=True)


# ----------------------------------------------------------------------------- platform
def job_get(job_id: str) -> dict:
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    r = subprocess.run(AUTOR + ["job", "get", job_id, "--json", "-q"], capture_output=True, env=env, timeout=180)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode or not out.strip():
        raise HandoffError(f"autor job get {job_id} rc={r.returncode}: {out[:400]} {r.stderr.decode('utf-8', 'replace')[:400]}")
    return json.loads(out)


def link_source(job: dict) -> dict:
    """The finished link's identity from ``job get``: its decision, artifact id / sha256 / URL."""
    jid = job.get("job_id")
    if job.get("status") != "succeeded":
        raise HandoffError(f"{jid} status {job.get('status')!r} (need succeeded); attempts {job.get('attempt_history')}")
    res = job.get("result") or {}
    dec = res.get("decision") or {}
    if not valid_result(dec):
        raise HandoffError(f"{jid} result.decision is not a chain-link decision (train.chain.valid_result): "
                           f"schema {dec.get('schema')!r} verdict {dec.get('verdict')!r} link {dec.get('link')!r}")
    if dec["of"] != OF:
        raise HandoffError(f"{jid} is link {dec['link']} of {dec['of']}, not of {OF}")
    if res.get("ok") is not True or res.get("exit_code") not in (0, None):
        raise HandoffError(f"{jid} result ok={res.get('ok')} exit_code={res.get('exit_code')}")
    art = job.get("artifact") or {}
    url = (art.get("download") or {}).get("url") or ""
    if art.get("state") != "ready" or not HEX64.fullmatch(art.get("sha256") or "") or not art.get("artifact_id"):
        raise HandoffError(f"{jid} artifact not ready / no sha256: { {k: art.get(k) for k in ('artifact_id', 'state', 'sha256')} }")
    if not url.startswith("https://"):
        raise HandoffError(f"{jid} artifact {art['artifact_id']} has no https download URL")
    exp = (art.get("download") or {}).get("expires_at")
    return {"job": jid, "artifact": art["artifact_id"], "artifact_sha256": art["sha256"],
            "size_bytes": art.get("size_bytes"), "manifest_sha256": dec["manifest_sha256"], "link": dec["link"],
            "step": dec["step"], "decision": dec, "url": url, "url_expires_at": exp}


# ----------------------------------------------------------------------------- ranged artifact reader
class RangeReader(io.RawIOBase):
    """A seekable read-only view of an https:// object via HTTP Range (1 MiB blocks, small cache) or a file://."""

    def __init__(self, url: str):
        super().__init__()
        self.url, self.pos, self.cache, self.requests = url, 0, collections.OrderedDict(), 0
        if url.startswith("file://"):
            self.local = open(urllib.request.url2pathname(url[len("file://"):]), "rb")
            self.size = os.fstat(self.local.fileno()).st_size
        elif url.startswith("https://"):
            self.local = None
            first = self._get(0, 511, probe=True)
            self.cache[0] = first
        else:
            raise HandoffError("artifact URL must be https:// or file://")

    def _get(self, a: int, b: int, probe: bool = False) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - scheme checked in __init__
            self.requests += 1
            if r.status != 206:
                raise HandoffError(f"the artifact store ignored the Range request (HTTP {r.status}); rerun with --full-hash")
            if probe:
                cr = r.headers.get("Content-Range", "")
                m = re.fullmatch(r"bytes \d+-\d+/(\d+)", cr.strip())
                if not m:
                    raise HandoffError(f"no usable Content-Range header: {cr!r}")
                self.size = int(m.group(1))
            return r.read()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if self.local is not None:
            self.local.seek(i * BLOCK)
            return self.local.read(BLOCK)
        if i in self.cache and (len(self.cache[i]) == BLOCK or (i + 1) * BLOCK >= self.size):
            self.cache.move_to_end(i)
            return self.cache[i]
        b = self._get(i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1)
        self.cache[i] = b
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return b

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, BLOCK)
            chunk = self._block(i)[o:o + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


# ----------------------------------------------------------------------------- torch checkpoint without torch
class _Stub:
    """Placeholder for any global a checkpoint pickle names (tensors, numpy arrays, torch types): never executed."""
    _qual = "?"

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, state):
        self.state = state

    def __repr__(self):
        return f"<stub {self._qual}>"


_STUBS: dict[str, type] = {}
_SAFE = {("collections", "OrderedDict"): collections.OrderedDict, ("_codecs", "encode"): __import__("_codecs").encode}
_SAFE.update({("builtins", n): getattr(builtins, n) for n in ("set", "frozenset", "slice", "complex", "bytearray",
                                                              "tuple", "list", "dict", "int", "float", "bool", "str")})


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return _SAFE[(module, name)]
        q = f"{module}.{name}"
        if q not in _STUBS:
            _STUBS[q] = type(f"Stub_{name}", (_Stub,), {"_qual": q})
        return _STUBS[q]

    def persistent_load(self, pid):
        return ("storage", pid)


def ckpt_state(fileobj) -> dict:
    """The top-level dict of a torch.save zip checkpoint, from its data.pkl only (tensors stay stubs)."""
    with zipfile.ZipFile(fileobj) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(names) != 1:
            raise HandoffError(f"checkpoint zip has {len(names)} data.pkl records")
        st = StubUnpickler(io.BytesIO(z.read(names[0]))).load()
    if not isinstance(st, dict):
        raise HandoffError(f"checkpoint pickle is a {type(st).__name__}, not a dict")
    return st


def resumable_problems(st: dict, man: dict) -> tuple[list[str], dict]:
    """Why ``st`` (a full ckpt-step*.pt state) cannot resume link k exactly ([] = resumable), plus a summary."""
    bad = []
    opt, sch, rng = st.get("optimizer"), st.get("scheduler"), st.get("rng")
    if type(st.get("step")) is not int or st["step"] != man["step"]:
        bad.append(f"step {st.get('step')!r} != CHAIN.json step {man['step']}")
    if st.get("config_hash") != man["config_hash"]:
        bad.append(f"config_hash {st.get('config_hash')!r} != CHAIN.json {man['config_hash']!r}")
    for k in ("epoch", "cursor"):
        if type(st.get(k)) is not int:
            bad.append(f"data position {k} missing ({st.get(k)!r})")
    n_state = n_params = 0
    if not isinstance(opt, dict) or not isinstance(opt.get("state"), dict) or not opt["state"]:
        bad.append("no optimizer state (trainable-only checkpoint: AdamW moments missing)")
    else:
        n_state = len(opt["state"])
        n_params = sum(len(g.get("params") or []) for g in opt.get("param_groups") or [] if isinstance(g, dict))
        miss = [i for i, s in opt["state"].items() if not (isinstance(s, dict) and "exp_avg" in s and "exp_avg_sq" in s)]
        if miss:
            bad.append(f"{len(miss)} optimizer entries without exp_avg / exp_avg_sq")
        if n_params != n_state:
            bad.append(f"optimizer state for {n_state} params, param_groups list {n_params}")
    if not isinstance(sch, dict) or not sch:
        bad.append("no scheduler state")
    if not isinstance(rng, dict) or any(rng.get(k) is None for k in ("torch", "python", "numpy")):
        bad.append("rng state missing (torch / python / numpy)")
    if not isinstance(st.get("trainable"), dict) or not st["trainable"]:
        bad.append("no trainable weights")
    ac = st.get("anti_collapse")
    summary = {"step": st.get("step"), "epoch": st.get("epoch"), "cursor": st.get("cursor"),
               "optimizer_params": n_state, "trainable_tensors": len(st.get("trainable") or {}),
               "scheduler_last_epoch": (sch or {}).get("last_epoch") if isinstance(sch, dict) else None,
               "horizon": (ac or {}).get("horizon") if isinstance(ac, dict) else None,
               "horizon_fixed": (ac or {}).get("horizon_fixed") if isinstance(ac, dict) else None}
    if isinstance(ac, dict) and not ac.get("horizon_fixed"):
        bad.append("anti_collapse.horizon_fixed is false: the cosine horizon was not fixed in link 1")
    return bad, summary


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_artifact(url: str, src: dict, cfg: dict, full_hash: bool = False) -> dict:
    """Check link k's artifact (see the module doc); raises HandoffError on the first problem."""
    k = src["link"]
    rr = RangeReader(url)
    head = rr.read(2)
    rr.seek(0)
    if head == b"\x1f\x8b":
        raise HandoffError("artifact is gzip-compressed: Range walk impossible; rerun with --full-hash")
    if src.get("size_bytes") and rr.size != src["size_bytes"]:
        raise HandoffError(f"artifact size {rr.size} != job get size_bytes {src['size_bytes']}")
    with tarfile.open(fileobj=io.BufferedReader(rr, buffer_size=1 << 16), mode="r:") as t:
        mem = {m.name.removeprefix("./"): m for m in t.getmembers() if m.isfile()}

        def read(name: str) -> bytes:
            if name not in mem:
                raise HandoffError(f"artifact has no {name}")
            return t.extractfile(mem[name]).read()

        raw = read("chain/CHAIN.json")
        if _sha(raw) != src["manifest_sha256"]:
            raise HandoffError(f"chain/CHAIN.json sha256 {_sha(raw)} != decision.manifest_sha256 {src['manifest_sha256']}")
        man = json.loads(raw)
        want = {"format": "ssdm-chain-link/v1", "chain_id": chain_id(cfg), "link": k, "of": OF}
        bad = {x: man.get(x) for x, v in want.items() if man.get(x) != v}
        if bad:
            raise HandoffError(f"CHAIN.json {bad} != {want}")
        if man.get("step") != src["step"]:
            raise HandoffError(f"CHAIN.json step {man.get('step')} != decision step {src['step']}")
        members = man.get("members") or {}
        need = {"latest.json", "cell.json", man.get("latest")} | ({man["best"]["file"]} if man.get("best") else set())
        if set(members) != need:
            raise HandoffError(f"CHAIN.json members {sorted(members)} != {sorted(map(str, need))}")
        for name, m in members.items():
            tm = mem.get(f"chain/{name}")
            if tm is None or tm.size != m["bytes"]:
                raise HandoffError(f"chain/{name}: {'missing' if tm is None else f'{tm.size} bytes'} != CHAIN.json {m['bytes']}")
        js = {}
        for name in ("latest.json", "cell.json"):
            b = read(f"chain/{name}")
            if _sha(b) != members[name]["sha256"]:
                raise HandoffError(f"chain/{name} sha256 != CHAIN.json")
            js[name] = json.loads(b)
        rec, cell = js["latest.json"], js["cell.json"]
        if (rec.get("path"), rec.get("step"), rec.get("config_hash")) != (man["latest"], man["step"], man["config_hash"]):
            raise HandoffError("latest.json disagrees with CHAIN.json (path / step / config_hash)")
        if not CKPT_RE.fullmatch(man["latest"]) or members[man["latest"]].get("state") != "full":
            raise HandoffError(f"latest member {man['latest']} is not a full checkpoint ({members[man['latest']]})")
        if cell.get("config") != cfg:
            raise HandoffError("cell.json config != the shared cell config (another chain)")
        if (cell.get("chain") or {}).get("link") != k:
            raise HandoffError(f"cell.json chain link {(cell.get('chain') or {}).get('link')} != {k}")
        if not (cell.get("ladder") or {}).get("T_max"):
            raise HandoffError("cell.json has no ladder.T_max")
        clock = cell.get("train_clock") or {}
        if clock.get("step") != man["step"]:
            raise HandoffError(f"cell.json train_clock step {clock.get('step')} != {man['step']}")
        st = ckpt_state(t.extractfile(mem[f"chain/{man['latest']}"]))
        probs, summ = resumable_problems(st, man)
        if probs:
            raise HandoffError(f"{man['latest']} is not resumable: {'; '.join(probs)}")
    used_min = float(clock.get("used_s") or 0) / 60
    out = {"chain_id": man["chain_id"], "link": k, "step": man["step"], "latest": man["latest"], "best": man.get("best"),
           "stopped": man.get("stopped"), "train_clock_used_min": round(used_min, 1),
           "global_clock_min": cfg.get("max_minutes"), "global_clock_left_min": round(cfg["max_minutes"] - used_min, 1),
           "max_steps": cfg.get("max_steps"), "checkpoint": summ, "range_requests": rr.requests,
           "artifact_bytes": rr.size}
    if full_hash:
        say(f"--full-hash: streaming {rr.size / 1e9:.2f} GB through sha256 (nothing written to disk)")
        h = hashlib.sha256()
        f = rr.local if rr.local is not None else urllib.request.urlopen(url, timeout=120)  # noqa: S310
        if rr.local is not None:
            f.seek(0)
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
        if h.hexdigest() != src["artifact_sha256"]:
            raise HandoffError(f"artifact sha256 {h.hexdigest()} != job get artifact.sha256 {src['artifact_sha256']}")
        out["artifact_sha256_verified"] = True
    return out


# ----------------------------------------------------------------------------- params
def placeholders(k_prev: int) -> dict[str, str]:
    F = f"__FILL_FROM_LINK{k_prev}"
    return {"job": f"{F}_JOB__", "artifact": f"{F}_ARTIFACT__", "artifact_sha256": f"{F}_ARTIFACT_SHA256__",
            "manifest_sha256": f"{F}_MANIFEST_SHA256__"}


def fill(text: str, k_prev: int, src: dict) -> str:
    """Substitute link k_prev's identity into the next link's params text (refuses a file filled differently)."""
    p = json.loads(text)
    frm = (p.get("chain") or {}).get("from") or {}
    ph = placeholders(k_prev)
    for f, tok in ph.items():
        cur = frm.get(f)
        if cur == tok:
            if text.count(json.dumps(tok)) != 1:
                raise HandoffError(f"placeholder {tok} must appear exactly once")
            text = text.replace(json.dumps(tok), json.dumps(src[f]))
        elif cur != src[f]:
            raise HandoffError(f"chain.from.{f} already set to {cur!r} (this hand-off brings {src[f]!r}): refusing")
    return text


def validate_params(text: str, k: int) -> dict:
    p = json.loads(text)
    try:
        ch = chain_block(p)
    except ChainError as e:
        raise HandoffError(f"chain block invalid: {e}") from e
    if ch is None or ch["link"] != k or ch["of"] != OF:
        raise HandoffError(f"params chain {ch} is not link {k} of {OF}")
    if ch["from"]["url_env"] != URL_ENV:
        raise HandoffError(f"chain.from.url_env {ch['from']['url_env']} != {URL_ENV}")
    env = (p.get("submit") or {}).get("env") or {}
    if not str(env.get(URL_ENV, "")).startswith("<"):
        raise HandoffError(f"submit.env.{URL_ENV} must be a '<...>' placeholder (sf_submit fills it from the shell)")
    if p.get("mode") != f"d20_v4_chain_l{k}" or p.get("cell_config") != CELL:
        raise HandoffError(f"mode / cell_config wrong: {p.get('mode')} {p.get('cell_config')}")
    return p


def run_lint(rel: str) -> None:
    from jobs.common import lint
    res = lint.lint_params(rel, ROOT)
    bad, waived = lint.failures(rel, res)
    if bad or waived:
        raise HandoffError(f"lint failed for {rel}: {json.dumps({**bad, **waived}, indent=1)}")
    say(f"lint OK for {rel} ({len(res)} checks)")


def git(*args: str, env: dict | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env={**os.environ, **(env or {})})
    if r.returncode:
        raise HandoffError(f"git {' '.join(args)} rc={r.returncode}: {r.stderr.strip()[:600]}")
    return r.stdout.strip()


def sf_submit(out: Path, url: str, dry: bool) -> int:
    cmd = [sys.executable, "-I", str(ROOT / "jobs/common/sf_submit.py"), str(out)] + (["--dry"] if dry else [])
    say(("DRY " if dry else "") + "submit: " + " ".join(cmd) + f"  (with {URL_ENV} in the child env only)")
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, URL_ENV: url}).returncode


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/d20_next_link.py", description=__doc__.split("\n")[0])
    ap.add_argument("job", help="the FINISHED link-k job id (e.g. job-7d3u9 for link 1)")
    ap.add_argument("--dry", action="store_true", help="verify + fill + validate only; no commit / push / package / submit")
    ap.add_argument("--full-hash", action="store_true", help="also stream the whole artifact through sha256")
    ap.add_argument("--out", type=Path, default=None, help="package dir (default: ../ssdm-wt/pkg-d20-l<k+1>-<sha>)")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tests", action="store_true", help="pass --tests to jobs.common.launch (CPU test gate)")
    ap.add_argument("--job-json", type=Path, default=None, help="tests: read the job get JSON from this file")
    ap.add_argument("--url", default=None, help="tests: artifact URL override (file://)")
    a = ap.parse_args(argv)

    job = json.loads(a.job_json.read_text(encoding="utf-8")) if a.job_json else job_get(a.job)
    src = link_source(job)
    k_prev, k = src["link"], src["link"] + 1
    say(f"{src['job']}: link {k_prev}/{OF} link_done at step {src['step']}; artifact {src['artifact']} "
        f"sha256 {src['artifact_sha256'][:12]}..; manifest {src['manifest_sha256'][:12]}..; URL expires {src['url_expires_at']}")
    exp = src["url_expires_at"]
    if exp and dt.datetime.fromisoformat(re.sub(r"\.\d+", "", exp).replace("Z", "+00:00")) \
            < dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=24):
        raise HandoffError(f"the presigned URL expires at {exp} (< 24 h, the next link may queue): re-run to mint a fresh one")
    cfg = json.loads((ROOT / CELL).read_text(encoding="utf-8"))
    rep = verify_artifact(a.url or src["url"], src, cfg, a.full_hash)
    say("artifact verified (resumable): " + json.dumps(rep))
    rel = PARAMS.format(k=k)
    if k == OF:
        say(f"link {k} is the FINAL link: it installs link {k_prev}, trains the {rep['global_clock_left_min']} min left "
            "of the global clock and runs the screen")
    if rep["global_clock_left_min"] <= 0:
        say(f"WARNING: the global clock is spent ({rep['train_clock_used_min']} of {rep['global_clock_min']} min): link {k} "
            "trains 0 steps" + (" and only screens" if k == OF else "; consider going straight to the final link"))
    path = ROOT / rel
    orig_bytes = path.read_bytes()  # bytes in, bytes out: no CRLF translation on Windows (lint / launch refuse CRs)
    orig = orig_bytes.decode("utf-8")
    text = fill(orig, k_prev, src)
    validate_params(text, k)
    path.write_bytes(text.encode("utf-8"))
    try:
        run_lint(rel)
    finally:
        if a.dry:
            path.write_bytes(orig_bytes)
    say(f"{rel}: chain.from = {{job {src['job']}, artifact {src['artifact']}, artifact_sha256 {src['artifact_sha256']}, "
        f"manifest_sha256 {src['manifest_sha256']}, url_env {URL_ENV}}}")
    launch_cmd = [sys.executable, "-X", "utf8", "-m", "jobs.common.launch", "m2_first_cell", "--mode",
                  f"d20_v4_chain_l{k}", "--sha", "<pushed sha>", "--out", "<out>", "--data", str(a.data)] + \
                 (["--tests"] if a.tests else [])
    if a.dry:
        say("DRY: params restored; a real run would: git commit + push " + BRANCH + "; " + " ".join(launch_cmd)
            + f"; python -I jobs/common/sf_submit.py <out> (with {URL_ENV} set)")
        if a.out and (a.out / "submit.json").is_file():
            return sf_submit(a.out, src["url"], dry=True)
        return 0
    if git("rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise HandoffError(f"worktree {ROOT} is not on {BRANCH}")
    if orig != text:
        git("add", rel)
        git("commit", "-m", f"D20 link {k} params: chain.from = link {k_prev} {src['job']} {src['artifact']} "
                            f"(artifact {src['artifact_sha256'][:12]}, manifest {src['manifest_sha256'][:12]})", env=AUTHOR)
    git("push", "origin", f"HEAD:{BRANCH}")
    sha = git("rev-parse", "HEAD")
    out = a.out or ROOT.parent / f"pkg-d20-l{k}-{sha[:8]}"
    launch_cmd[launch_cmd.index("<pushed sha>")] = sha
    launch_cmd[launch_cmd.index("<out>")] = str(out)
    say("package: " + " ".join(launch_cmd))
    r = subprocess.run(launch_cmd, cwd=ROOT)
    if r.returncode:
        raise HandoffError(f"jobs.common.launch rc={r.returncode}")
    sub = json.loads((out / "submit.json").read_text(encoding="utf-8"))
    env = next(c["args"] for c in sub["calls"] if c["tool"] == "submit_job").get("env") or {}
    if not str(env.get(URL_ENV, "")).startswith("<"):
        raise HandoffError(f"packaged submit.json env has no {URL_ENV} placeholder: the job could not install link {k_prev}")
    if sf_submit(out, src["url"], dry=True):
        raise HandoffError("sf_submit --dry failed")
    return sf_submit(out, src["url"], dry=False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HandoffError as e:
        say(f"STOP: {e}")
        sys.exit(2)
