"""Chained cells: ONE training run (one cell config, one global schedule) carried over several SF jobs ("links").

Why: an SF job is capped at 899 min (``job_max_duration_ceiling_minutes`` 900), and a cell job spends ~515 of them on
setup, guards, the parent arm, tokenization and the screen, leaving a ~380-min training clock (under one epoch of the
D16-D19 draws). The platform keeps ONE checkpoint slot per job (a new job starts with an empty slot; ``resume_source``
only says whether the job's resume MODE was declared or inferred), so a run that spans jobs carries its state in the
job's /output artifact: link k exports a full resumable checkpoint, link k+1 downloads it (sha256-pinned), verifies
lineage + config and continues as if uninterrupted.

The contract
  * cell config: the SAME file for every link. ``max_minutes`` is the TOTAL training clock of the chain and
    ``max_steps`` the global cap. Everything that defines the trajectory is decided once: the cosine horizon is
    projected in link 1 from the GLOBAL clock and fixed in the checkpoint; data order (seed, epoch, cursor), optimizer,
    scheduler, RNG, dev cards, best checkpoint and collapse-guard state all ride in the checkpoint; the cumulative
    ``train_clock`` rides in cell.json. ``Trainer.train(N) == train(k); save; resume; train(N - k)`` is exact on CPU
    (tests/test_chain.py runs the chain through this module's export / install).
  * job params, one file per link: ``"chain": {"link": k, "of": N, "from": {...}}`` + ``"cell_config"`` (shared).
    ``from`` (links >= 2) pins link k-1's artifact: {"job", "artifact", "artifact_sha256" (get_job artifact.sha256),
    "manifest_sha256" (link k-1's declared result: decision.manifest_sha256), "url_env"}. The presigned URL
    (get_artifact) goes into the submit env at submit time, never into git.
  * link k < N (first_cell_job.sh): setup + guard + heldout (and the draw's attack / doc-overlap gates inside the cell)
    run as in every cell job; [link >= 2: install link k-1]; cell trains until the JOB's deadline (every minute the
    earlier phases did not use); export /output/chain; exit. No parent arm, no screen / evals, no inference export.
  * link N: install link N-1, train the rest of the global clock, then the cell's own stage (screen or gates).

Export (``export``): <out>/chain/{CHAIN.json, latest.json, cell.json, <latest>.pt, [<best>.pt]}. The latest checkpoint
is the full .pt (trainable fp32 + AdamW moments + scheduler + RNG + counters + anti-collapse state: 12 bytes per
trainable parameter). A ``best_ckpt`` run's best checkpoint, when it is not the latest, goes FULL if both fit
``budget_bytes``, else TRAINABLE-ONLY (``optimizer`` None, 4 bytes / param): the same fp32 weights, so scoring is
unchanged; only a collapse-guard rollback to that best keeps the current AdamW moments instead of the best's (recorded
in the event, ``Trainer.collapse``). The job artifact is ONE object capped at 5 GiB (5,368,709,120 bytes, the store's
single-PUT bound): D18 at LoRA r64 = 293.3M trainable = 3.52 GB latest + 1.17 GB slim best = 4.69 GB fits; r256
(1.157B trainable = 13.9 GB latest alone) does not, and ``check`` refuses it in setup.

Install (``install``): the artifact (https:// or file://) is streamed with its sha256 checked; CHAIN.json must have the
pinned sha256, the chain id (sha256 of the canonical cell config), link == k-1 and of == N; every other member is
checked against CHAIN.json; cell.json's config must EQUAL this link's cell config. ``Trainer.load`` then checks config
hash + lineage (backbone, tokenizer, layout, training items) at resume: run_cell exit 29 on a mismatch. A restart of
the same link (provider fault: its own newer checkpoint in $GMN_CHECKPOINT_DIR) keeps that checkpoint.

    python -X utf8 -m train.chain check   --params P --cell-config C [--budget-bytes B] [--out F]
    python -X utf8 -m train.chain install --params P --cell-config C --ck-dir D --work W [--cell-out F] [--out F]
    python -X utf8 -m train.chain export  --params P --cell-config C --ck-dir D --cell F --out-dir O --decision F
    python -X utf8 -m train.chain cell-minutes --params P --elapsed-s S
    python -X utf8 -m train.chain plan    --base-params P --of N
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

FORMAT = "ssdm-chain-link/v1"
SCHEMA = "ssdm.chain.link.v1"
#: the job artifact store's single-PUT bound (``autor limits`` max_artifact_bytes; jobs/common/size_guard.py)
ARTIFACT_CAP = 5_368_709_120
#: /output size-guard cap of a chain link (headroom under ARTIFACT_CAP for the tar and the final phase export)
SIZE_CAP = 5_200_000_000
#: bytes for /output/chain (the rest of SIZE_CAP holds the phase tarballs, logs and manifests)
BUDGET = 4_900_000_000
#: AdamW: fp32 weight + exp_avg + exp_avg_sq per trainable parameter (the scheduler / RNG / history are small)
BYTES_FULL, BYTES_TRAINABLE = 12, 4
#: job ceiling (``job_max_duration_ceiling_minutes`` 900; D16-D18 schedule to 899)
CEILING_MIN = 899
#: phases that run AFTER the cell phase (their caps + reserve are kept free when a link trains to its deadline)
POST_CELL = ("chain_export", "screen", "evals", "report")
#: phases an intermediate link never runs
EVAL_PHASES = ("parent_g12", "parent_index", "screen", "evals", "report", "checkpoint", "fulldev", "parent_full")
CKPT_RE = re.compile(r"ckpt-step[0-9]{7}\.pt")
CHUNK = 8 << 20


class ChainError(RuntimeError):
    def __init__(self, msg: str, code: int = 10):
        super().__init__(msg)
        self.code = code


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def chain_id(cfg: dict) -> str:
    """Identity of a chain = its cell config (canonical JSON); every link trains under exactly this config."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- params block
def chain_block(params: dict) -> dict | None:
    """The validated ``chain`` block of a job_params dict (None when absent; ``ChainError`` code 5 if malformed)."""
    ch = params.get("chain")
    if ch is None:
        return None
    if not isinstance(ch, dict):
        raise ChainError("chain block must be an object", 5)
    k, n = ch.get("link"), ch.get("of")
    if type(k) is not int or type(n) is not int or n < 2 or not 1 <= k <= n:
        raise ChainError(f"chain link/of must be ints with 1 <= link <= of, of >= 2 (got {k!r}/{n!r})", 5)
    if params.get("eval_only", "none") != "none":
        raise ChainError("a chain link cannot be eval_only", 5)
    if not isinstance(params.get("cell_config"), str) or not params["cell_config"]:
        raise ChainError("a chain link names its shared cell_config", 5)
    src = ch.get("from")
    if k == 1:
        if src is not None:
            raise ChainError("link 1 starts the chain: no 'from' block", 5)
        return ch
    if not isinstance(src, dict):
        raise ChainError(f"link {k} needs a 'from' block pinning link {k - 1}'s artifact", 5)
    for key in ("artifact", "artifact_sha256", "manifest_sha256", "url_env"):
        if not isinstance(src.get(key), str) or not src[key]:
            raise ChainError(f"chain.from.{key} missing", 5)
    for key in ("artifact_sha256", "manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", src[key]):
            raise ChainError(f"chain.from.{key} is not a sha256", 5)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", src["url_env"]):
        raise ChainError(f"chain.from.url_env malformed: {src['url_env']!r}", 5)
    return ch


def intermediate(ch: dict | None) -> bool:
    return bool(ch) and ch["link"] < ch["of"]


# ----------------------------------------------------------------------------- size
def trainable_params(cfg: dict) -> dict[str, int]:
    """Trainable parameter count of a cell: LoRA (rank x sum over the backbone spec's lora_shapes of in + out) + the
    head at the spec's d_model. None for a backbone without lora_shapes. Imports torch + heads (setup only)."""
    from heads.backbone import BACKBONES
    from heads.heads import make_head

    spec = BACKBONES.get({"rwkv7": "rwkv7-13.3b"}.get(cfg.get("parent", "rwkv7"), cfg.get("parent", "")))
    if spec is None or not spec.lora_shapes:
        raise ChainError(f"no backbone spec with lora_shapes for parent {cfg.get('parent')!r}", 5)
    r = int(cfg["lora_rank"])
    lora = r * sum(cnt * (i + o) for i, o, cnt in spec.lora_shapes)
    head = make_head(cfg["head"], spec.d_model, **(cfg.get("head_kwargs") or {}))
    nh = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {"lora": lora, "head": nh, "total": lora + nh, "backbone": spec.name, "lora_rank": r}


def state_bytes(n_trainable: int) -> dict[str, int]:
    return {"latest_full": BYTES_FULL * n_trainable, "best_full": BYTES_FULL * n_trainable,
            "best_trainable": BYTES_TRAINABLE * n_trainable, "inference_bf16": 2 * n_trainable}


def check(params: dict, cfg: dict, budget: int = BUDGET) -> dict[str, Any]:
    """Setup preflight of a chain link: the block is valid and an intermediate link's export (latest full + best
    trainable-only, the worst case) fits ``budget``. ``ChainError`` code 5 otherwise."""
    ch = chain_block(params)
    if ch is None:
        raise ChainError("job params have no chain block", 5)
    n = trainable_params(cfg)
    sb = state_bytes(n["total"])
    need = sb["latest_full"] + sb["best_trainable"] if cfg.get("best_ckpt") else sb["latest_full"]
    rec = {"chain": {k: ch[k] for k in ("link", "of")}, "chain_id": chain_id(cfg), "trainable": n, "bytes": sb,
           "export_bytes_worst": need, "budget_bytes": budget, "artifact_cap_bytes": ARTIFACT_CAP,
           "fits": need <= budget}
    if intermediate(ch) and not rec["fits"]:
        raise ChainError(f"chain export ~{need / 1e9:.2f} GB (latest {sb['latest_full'] / 1e9:.2f} GB full"
                         f"{' + best trainable-only' if cfg.get('best_ckpt') else ''}; {n['total']:,} trainable "
                         f"params at LoRA r{n['lora_rank']}) > the {budget / 1e9:.2f} GB budget under the 5 GiB job "
                         "artifact cap: this cell cannot chain through job artifacts (needs a bucket transport)", 5)
    return rec


# ----------------------------------------------------------------------------- export
def _copy_hash(src: Path, dst: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as f, open(tmp, "wb") as g:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
            g.write(b)
            n += len(b)
    os.replace(tmp, dst)
    return h.hexdigest(), n


def _has_optimizer(path: Path) -> bool:
    import torch

    st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return st.get("optimizer") is not None


def _write_slim(src: Path, dst: Path) -> tuple[str, int]:
    """A trainable-only copy of a full checkpoint: the same dict with optimizer / scheduler / RNG dropped."""
    import torch

    st = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    st = {**st, "optimizer": None, "scheduler": None, "rng": None, "chain_state": "trainable"}
    tmp = dst.with_name(dst.name + ".part")
    torch.save(st, tmp)
    del st
    os.replace(tmp, dst)
    return sha256_file(dst), dst.stat().st_size


def _write_json(p: Path, obj: Any) -> tuple[str, int]:
    tmp = p.with_name(p.name + ".part")
    b = json.dumps(obj, indent=1, sort_keys=True).encode()
    tmp.write_bytes(b)
    os.replace(tmp, p)
    return hashlib.sha256(b).hexdigest(), len(b)


def export(*, ck_dir: str | Path, cell_path: str | Path, cfg: dict, ch: dict, out_dir: str | Path,
           budget: int = BUDGET) -> dict[str, Any]:
    """Write link ``ch['link']``'s resumable state to ``out_dir`` (see the module doc). CHAIN.json is written LAST, so
    a torn export has no manifest. Returns the decision record (``valid_result``)."""
    ck_dir, out = Path(ck_dir), Path(out_dir)
    cell = json.loads(Path(cell_path).read_text(encoding="utf-8"))
    if cell.get("config") != cfg:
        raise ChainError("cell.json config != the chain's cell config")
    rec = json.loads((ck_dir / "latest.json").read_text(encoding="utf-8"))
    latest = ck_dir / rec["path"]
    if not CKPT_RE.fullmatch(rec["path"]) or not latest.is_file():
        raise ChainError(f"latest.json names {rec['path']!r}: not a full checkpoint in {ck_dir}")
    clock = cell.get("train_clock") or {}
    if clock.get("step") != rec["step"]:
        raise ChainError(f"cell.json train_clock step {clock.get('step')} != latest checkpoint step {rec['step']}")
    tr = cell.get("train") or {}
    if tr.get("stopped") != "link_end" and "verdict" not in tr:
        raise ChainError(f"the cell did not end its link (train record {sorted(tr)})")
    if (cell.get("chain") or {}).get("link") != ch["link"]:
        raise ChainError(f"cell.json chain link {(cell.get('chain') or {}).get('link')} != this link {ch['link']}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "CHAIN.json").exists():
        os.replace(out / "CHAIN.json", out / "CHAIN.json.stale")  # never left naming files this export rewrites
    lsize = latest.stat().st_size
    if lsize > budget:
        raise ChainError(f"latest checkpoint {lsize:,} bytes > the chain budget {budget:,}", 5)
    members: dict[str, dict] = {}
    sha, n = _copy_hash(latest, out / latest.name)
    members[latest.name] = {"sha256": sha, "bytes": n, "role": "latest", "state": "full"}
    best = None
    bp = (rec.get("best") or {}).get("path")
    if bp and bp != rec["path"]:
        src = ck_dir / bp
        if not CKPT_RE.fullmatch(bp) or not src.is_file():
            raise ChainError(f"latest.json best pointer {bp!r} has no checkpoint in {ck_dir}")
        if _has_optimizer(src) and lsize + src.stat().st_size <= budget:
            sha, n = _copy_hash(src, out / bp)
            state = "full"
        elif _has_optimizer(src):
            sha, n = _write_slim(src, out / bp)
            state = "trainable"
        else:  # already trainable-only (carried from an earlier link)
            sha, n = _copy_hash(src, out / bp)
            state = "trainable"
        if lsize + n > budget:
            raise ChainError(f"latest + best ({lsize + n:,} bytes) > the chain budget {budget:,}", 5)
        members[bp] = {"sha256": sha, "bytes": n, "role": "best", "state": state}
        best = {"file": bp, "state": state, "step": rec["best"].get("step")}
    for name, obj in (("latest.json", rec), ("cell.json", cell)):
        sha, n = _write_json(out / name, obj)
        members[name] = {"sha256": sha, "bytes": n}
    man = {"format": FORMAT, "chain_id": chain_id(cfg), "link": ch["link"], "of": ch["of"],
           "step": rec["step"], "config_hash": rec["config_hash"], "latest": rec["path"], "best": best,
           "train_clock": clock, "stopped": tr.get("stopped") or "complete", "members": members,
           "from": ch.get("from")}
    msha, mb = _write_json(out / "CHAIN.json", man)
    total = sum(m["bytes"] for m in members.values()) + mb
    return {"schema": SCHEMA, "verdict": "link_done", "link": ch["link"], "of": ch["of"], "chain_id": man["chain_id"],
            "step": rec["step"], "train_clock": clock, "stopped": man["stopped"], "best": best,
            "manifest_sha256": msha, "export_bytes": total, "budget_bytes": budget,
            "next": f"link {ch['link'] + 1}: chain.from = {{artifact: <this job's artifact>, artifact_sha256: <get_job "
                    f"artifact.sha256>, manifest_sha256: {msha}}}"}


def valid_result(d: dict) -> bool:
    """finalize.py validator of an intermediate link's decision (out/chain_link.json)."""
    try:
        return (d.get("schema") == SCHEMA and d.get("verdict") == "link_done"
                and bool(re.fullmatch(r"[0-9a-f]{64}", d.get("manifest_sha256") or ""))
                and type(d.get("link")) is int and type(d.get("of")) is int and 1 <= d["link"] < d["of"]
                and type(d.get("step")) is int and d["step"] > 0 and 0 < d.get("export_bytes", 0) <= ARTIFACT_CAP)
    except (AttributeError, TypeError):
        return False


# ----------------------------------------------------------------------------- install
def install(ch: dict, *, cfg: dict, ck_dir: str | Path, work: str | Path, cell_out: str | Path | None = None,
            url: str | None = None) -> dict[str, Any]:
    """Install link ``ch['link'] - 1``'s exported state into ``ck_dir`` (see the module doc)."""
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError, _extract, download

    if ch["link"] < 2:
        raise ChainError("link 1 has nothing to install", 5)
    src = ch["from"]
    ck_dir, work = Path(ck_dir), Path(work)
    ck_dir.mkdir(parents=True, exist_ok=True)
    marker = ck_dir / "CHAIN_INSTALLED.json"
    if marker.is_file():
        m = json.loads(marker.read_text(encoding="utf-8"))
        if m.get("manifest_sha256") == src["manifest_sha256"] and m.get("link") == ch["link"] \
                and (ck_dir / "latest.json").is_file():
            # a restart of this link: its own checkpoints (at or past the installed step) stay; run_cell resumes them
            if cell_out is not None and (ck_dir / "cell.json").is_file():
                Path(cell_out).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ck_dir / "cell.json", cell_out)
            return {**m, "reused": True}
        raise ChainError(f"{marker} pins another link / manifest: {m.get('link')} {m.get('manifest_sha256')}")
    if (ck_dir / "latest.json").exists():
        raise ChainError(f"{ck_dir} already holds a checkpoint that was not installed from this chain link")
    url = url or os.environ.get(src["url_env"], "")
    if not url:
        raise ChainError(f"${src['url_env']} is not set (the get_artifact {src['artifact']} URL)", 5)
    tar = work / "chain-artifact.tar"
    try:
        got = download(url, tar)
        if got != src["artifact_sha256"]:
            raise ChainError(f"artifact {src['artifact']} sha256 {got} != chain.from.artifact_sha256")
        stage = work / "chain-stage"
        stage.mkdir(parents=True, exist_ok=True)
        _extract(tar, "./chain/CHAIN.json", stage / "CHAIN.json", src["manifest_sha256"], "CHAIN.json")
        man = json.loads((stage / "CHAIN.json").read_text(encoding="utf-8"))
        want = {"format": FORMAT, "chain_id": chain_id(cfg), "link": ch["link"] - 1, "of": ch["of"]}
        bad = {k: man.get(k) for k, v in want.items() if man.get(k) != v}
        if bad:
            raise ChainError(f"CHAIN.json {bad} != this link's {({k: want[k] for k in bad})} "
                             "(another cell config, another chain, or the wrong link)")
        mem = man.get("members") or {}
        names = set(mem)
        need = {"latest.json", "cell.json", man.get("latest")} | ({man["best"]["file"]} if man.get("best") else set())
        if names != need or not all(CKPT_RE.fullmatch(n) for n in names - {"latest.json", "cell.json"}):
            raise ChainError(f"CHAIN.json members {sorted(names)} != {sorted(str(x) for x in need)}")
        for name in sorted(names):
            _extract(tar, f"./chain/{name}", stage / name, mem[name]["sha256"], name)
    except CheckpointError as e:
        raise ChainError(str(e), e.code) from e
    cell = json.loads((stage / "cell.json").read_text(encoding="utf-8"))
    if cell.get("config") != cfg:
        raise ChainError("the carried cell.json config != this link's cell config")
    if not (cell.get("ladder") or {}).get("T_max"):
        raise ChainError("the carried cell.json has no ladder.T_max")
    if (cell.get("chain") or {}).get("link") != ch["link"] - 1:
        raise ChainError(f"the carried cell.json is link {(cell.get('chain') or {}).get('link')}, not {ch['link'] - 1}")
    rec = json.loads((stage / "latest.json").read_text(encoding="utf-8"))
    if rec.get("path") != man["latest"] or rec.get("step") != man["step"] or rec.get("config_hash") != man["config_hash"]:
        raise ChainError("latest.json disagrees with CHAIN.json (path / step / config_hash)")
    if ((rec.get("best") or {}).get("path") or None) not in (None, man["latest"], (man.get("best") or {}).get("file")):
        raise ChainError("latest.json's best pointer names a checkpoint the link did not carry")
    for name in sorted(names - {"latest.json"}):  # checkpoints + cell.json first, latest.json (the resume point) last
        os.replace(stage / name, ck_dir / name)
    os.replace(stage / "latest.json", ck_dir / "latest.json")
    tar.unlink()  # job scratch (4-5 GB): the bytes now live in ck_dir
    out = {"chain_id": man["chain_id"], "link": ch["link"], "from_link": man["link"], "of": ch["of"],
           "step": man["step"], "config_hash": man["config_hash"], "best": man.get("best"),
           "train_clock": man.get("train_clock"), "artifact": src["artifact"], "job": src.get("job"),
           "artifact_sha256": src["artifact_sha256"], "manifest_sha256": src["manifest_sha256"], "reused": False}
    _write_json(marker, out)
    if cell_out is not None:
        Path(cell_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ck_dir / "cell.json", cell_out)
    return out


# ----------------------------------------------------------------------------- run_cell hooks
def resume_check(ch: dict, cell: dict, cfg: dict, resumed: bool) -> str | None:
    """None when a link may train from this state, else why not (run_cell exit 29): link >= 2 must resume the
    installed checkpoint of link k-1 (or its own, after an in-job restart); any link resumes only its own chain."""
    prev = cell.get("chain") or {}
    if ch["link"] >= 2 and not resumed:
        return f"chain link {ch['link']} found no checkpoint to resume (install link {ch['link'] - 1} first)"
    if resumed and prev:
        if prev.get("chain_id") != chain_id(cfg) or prev.get("of") != ch["of"]:
            return f"cell.json is from chain {prev.get('chain_id')} (of {prev.get('of')}), not this one"
        if prev.get("link") not in (ch["link"] - 1, ch["link"]):
            return f"cell.json is link {prev.get('link')}: link {ch['link']} resumes only link {ch['link'] - 1}"
    elif resumed and ch["link"] >= 2:
        return "the resumed cell.json carries no chain record"
    return None


def stop_reason(now: float, t_end: float, deadline: float | None, step: int, max_steps: int,
                stopped: str | None, ch: dict | None) -> str:
    """Why the training loop ended: "link_end" (an intermediate link reached its job deadline with global clock and
    steps left: export and continue in the next link) or "complete" (max_steps, the global clock, the collapse guard,
    or the FINAL link's deadline: the run ends here, exactly as an uninterrupted run would at that point)."""
    if (intermediate(ch) and deadline is not None and now >= deadline and now < t_end and step < max_steps
            and not stopped):
        return "link_end"
    return "complete"


def cell_minutes(params: dict, elapsed_s: float) -> int:
    """A chain link's cell-phase timeout: the job's max_duration minus the time used so far minus the caps of the
    phases after the cell (and the reserve), so the link trains with every minute the earlier phases left."""
    caps = params["phase_caps_minutes"]
    post = sum(int(caps[k]) for k in POST_CELL if k in caps) + int(params["reserve_minutes"])
    return max(1, int(params["max_duration_minutes"]) - math.ceil(elapsed_s / 60) - post)


# ----------------------------------------------------------------------------- planning
def link_caps(base: dict, link: int, of: int, *, chain_min: int = 15, export_min: int = 10,
              ceiling: int = CEILING_MIN) -> dict[str, int]:
    """Phase caps of link ``link`` of ``of`` derived from a single-job mode's params: intermediate links drop the
    parent-arm / eval phases and add chain_export; links >= 2 add chain (install); the cell takes the rest of the
    ceiling, so caps + reserve == max_duration (the launch lint's schedule rule)."""
    caps = {k: int(v) for k, v in base["phase_caps_minutes"].items()}
    if link < of:
        for k in EVAL_PHASES:
            caps.pop(k, None)
        caps["chain_export"] = export_min
    if link >= 2:
        caps["chain"] = chain_min
    caps["cell"] = ceiling - sum(v for k, v in caps.items() if k != "cell") - int(base["reserve_minutes"])
    return caps


def plan(base: dict, cfg: dict, of: int, *, pre_train_min: float | None = None, tail_min: int = 8,
         ceiling: int = CEILING_MIN) -> dict[str, Any]:
    """Per-link caps and training minutes for a chain of ``of`` links built from a single-job mode (``base`` params,
    ``cfg`` its cell config). ``train_min_floor``: every earlier phase runs to its cap and the cell's pre-training
    work (data draw + guard scans, tokenize, model load, smoke / ladder) takes the base mode's whole allowance
    (cell cap - max_minutes); ``train_min_expected``: the base params' expected_minutes for the earlier phases and
    ``pre_train_min`` (default: the same allowance). A plan with ``max_minutes`` = the sum of the floors always
    finishes the global schedule (the cosine reaches its floor) inside the final link."""
    allowance = int(base["phase_caps_minutes"]["cell"]) - int(cfg["max_minutes"])
    pre = allowance if pre_train_min is None else pre_train_min
    exp = base.get("expected_minutes") or {}
    links = []
    for k in range(1, of + 1):
        caps = link_caps(base, k, of, ceiling=ceiling)
        post = sum(caps[p] for p in POST_CELL if p in caps) + int(base["reserve_minutes"])
        before = [p for p in caps if p != "cell" and p not in POST_CELL]
        floor = ceiling - sum(caps[p] for p in before) - post - allowance - tail_min
        expected = ceiling - sum(float(exp.get(p, caps[p])) if p != "chain" else 5.0 for p in before) - post \
            - pre - tail_min
        links.append({"link": k, "phase_caps_minutes": caps, "max_duration_minutes": ceiling,
                      "train_min_floor": floor, "train_min_expected": round(expected, 1),
                      "runs": "train + export" if k < of else "train the rest + " + base.get("stage", "gates")})
    return {"of": of, "ceiling_min": ceiling, "pre_train_allowance_min": allowance, "pre_train_expected_min": pre,
            "tail_min": tail_min, "links": links,
            "max_minutes_guaranteed": sum(x["train_min_floor"] for x in links),
            "max_minutes_expected": round(sum(x["train_min_expected"] for x in links), 1),
            "single_job_train_min": int(cfg["max_minutes"])}


# ----------------------------------------------------------------------------- CLI
def _load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.chain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--params", required=True)
    c.add_argument("--cell-config", required=True)
    c.add_argument("--budget-bytes", type=int, default=BUDGET)
    c.add_argument("--out", default=None)
    i = sub.add_parser("install")
    i.add_argument("--params", required=True)
    i.add_argument("--cell-config", required=True)
    i.add_argument("--ck-dir", required=True)
    i.add_argument("--work", required=True)
    i.add_argument("--cell-out", default=None)
    i.add_argument("--url", default=None, help="default: $<chain.from.url_env>")
    i.add_argument("--out", default=None)
    e = sub.add_parser("export")
    e.add_argument("--params", required=True)
    e.add_argument("--cell-config", required=True)
    e.add_argument("--ck-dir", required=True)
    e.add_argument("--cell", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--budget-bytes", type=int, default=BUDGET)
    e.add_argument("--decision", required=True)
    m = sub.add_parser("cell-minutes")
    m.add_argument("--params", required=True)
    m.add_argument("--elapsed-s", type=float, required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base-params", required=True)
    p.add_argument("--cell-config", default=None, help="default: the base params' cell_config")
    p.add_argument("--of", type=int, required=True)
    p.add_argument("--pre-train-min", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "cell-minutes":
            print(cell_minutes(_load(a.params), a.elapsed_s))
            return 0
        if a.cmd == "plan":
            base = _load(a.base_params)
            rec = plan(base, _load(a.cell_config or base["cell_config"]), a.of, pre_train_min=a.pre_train_min)
        else:
            params, cfg = _load(a.params), _load(a.cell_config)
            ch = chain_block(params)
            if ch is None:
                raise ChainError("job params have no chain block", 5)
            if a.cmd == "check":
                rec = check(params, cfg, a.budget_bytes)
            elif a.cmd == "install":
                rec = install(ch, cfg=cfg, ck_dir=a.ck_dir, work=a.work, cell_out=a.cell_out, url=a.url)
            else:
                if not intermediate(ch):
                    raise ChainError(f"link {ch['link']} of {ch['of']} is the final link: nothing to export", 5)
                rec = export(ck_dir=a.ck_dir, cell_path=a.cell, cfg=cfg, ch=ch, out_dir=a.out_dir,
                             budget=a.budget_bytes)
                _write_json(Path(a.decision), rec)
    except ChainError as err:
        print(f"chain: {err}", file=sys.stderr)
        return err.code
    text = json.dumps(rec, indent=1)
    if getattr(a, "out", None):
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
