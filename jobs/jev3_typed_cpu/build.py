"""Package the JEV3-TYPED CPU job (draw build + CPU suites) with LOW RAM: git archive -> tar on disk, data files and job
files appended member by member, then a streaming zstd copy. Writes <out>/jev3_typed_cpu.tar.zst + submit.json (the
jobs/common/sf_submit.py receipt format, as jobs/cpu_tests/build.py).

  uv run --no-project --with zstandard python -I jobs/jev3_typed_cpu/build.py <repo> <sha> <ssdm-data> <out>
env: DRAW (1/0), TESTS, KEXPR (passed to run.sh), MAX_MIN (job clock; cpu-8 $0.27 per 30 min)
"""
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile

import zstandard

HERE = pathlib.Path(__file__).resolve().parent
MTIME = 1767225600
KEEP_RECEIPTS = ("receipts/2026-10-11-jev3-sampler/jev3_train_exclude_subs.txt",
                 "receipts/2026-10-11-jev3-regate/jev3_exclude_ids.txt",
                 "receipts/2026-10-11-jev2-final-regate/mix_exclude_ids.txt",
                 "receipts/2026-10-11-jev2a-v2-regate/jev2a-v2_musique_provenance_unverified_ids.txt")
DATA = ("m2/build/bschema-draw-jev1/records.jsonl.gz", "m2/build/bschema-draw-jev1/bschema-draw-jev1.manifest.jsonl.gz",
        "m2/build/bschema-draw-jev1/bschema-draw-jev1.content.jsonl.gz",
        "m2/build/bschema-draw-jev3/records.jsonl.gz", "m2/build/bschema-draw-jev3/bschema-draw-jev3.manifest.jsonl.gz",
        "m2/build/bschema-draw-jev3/bschema-draw-jev3.content.jsonl.gz",
        "m2/build/bschema-draw-jev3-slice/records.jsonl.gz",
        "m2/build/bschema-draw-jev3-slice/bschema-draw-jev3-slice.manifest.jsonl.gz",
        "m2/build/bschema-draw-jev3-slice/bschema-draw-jev3-slice.content.jsonl.gz")
DATA_TYPED = ("m2/build/bschema-draw-jev3-typed/records.jsonl.gz",
              "m2/build/bschema-draw-jev3-typed/bschema-draw-jev3-typed.manifest.jsonl.gz",
              "m2/build/bschema-draw-jev3-typed/bschema-draw-jev3-typed.content.jsonl.gz")
TOKENIZER = "tokenizers/Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "chat_template.jinja",
             "special_tokens_map.json", "generation_config.json")


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def add_bytes(t, name, b, mode=0o644):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.mtime = len(b), mode, MTIME
    t.addfile(ti, io.BytesIO(b))


def main():
    repo, sha, data, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    full = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{sha}^{{commit}}"], check=True, capture_output=True,
                          text=True).stdout.strip()
    tar_path = out / "ctx.tar"
    with open(tar_path, "wb") as f:  # the tree without receipts/ (streamed by git to disk)
        subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar",
                        "--prefix=tree/", full, "--", ".", ":!receipts"], check=True, stdout=f)
    data_sha, man = {}, {}
    with tarfile.open(tar_path, "a", format=tarfile.GNU_FORMAT) as t:
        keep = subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar",
                               "--prefix=tree/", full, "--", *KEEP_RECEIPTS], check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(keep)) as k:
            for m in k.getmembers():
                if m.isfile():
                    t.addfile(m, k.extractfile(m))
        jev = sorted((data / "jevbench").iterdir())
        pub = json.loads((repo / "jevkit/data/jevbench_public_pin.json").read_text(encoding="utf-8"))
        jdir = data / "jevbench" / pub["commit"]
        rels = list(DATA) + (list(DATA_TYPED) if os.environ.get("DRAW", "1") == "0" else []) + [f"jevbench/{pub['commit']}/{n}" for n in pub["files"]] + \
            [f"{TOKENIZER}/{n}" for n in TOK_FILES if (data / TOKENIZER / n).is_file()]
        assert jdir.is_dir() and jev, "no public JevBench files under <data>/jevbench"
        for rel in rels:
            p = data / rel
            assert p.is_file(), f"missing {p}"
            data_sha[rel] = sha_file(p)
            t.add(str(p), arcname=f"data/{rel}", recursive=False,
                  filter=lambda ti: (setattr(ti, "mtime", MTIME), setattr(ti, "mode", 0o644), ti)[-1])
        for fn in ("run.sh",):
            add_bytes(t, f"job/{fn}", (HERE / fn).read_bytes().replace(b"\r\n", b"\n"), 0o755)
        add_bytes(t, "Dockerfile", (HERE / "Dockerfile").read_bytes().replace(b"\r\n", b"\n"))
        man = {"job": "jev3_typed_cpu", "tree": full, "data_sha256": data_sha,
               "env": {k: os.environ[k] for k in ("DRAW", "TESTS", "KEXPR") if k in os.environ}}
        add_bytes(t, "CONTEXT_MANIFEST.json", json.dumps(man, indent=1).encode() + b"\n")
    f_out = out / "jev3_typed_cpu.tar.zst"
    with open(tar_path, "rb") as src, open(f_out, "wb") as dst:
        zstandard.ZstdCompressor(level=9).copy_stream(src, dst)
    tar_path.unlink()
    sha_blob, size = sha_file(f_out), f_out.stat().st_size
    max_min = int(os.environ.get("MAX_MIN", "30"))
    env = {k: os.environ[k] for k in ("DRAW", "TESTS", "KEXPR") if k in os.environ}
    checks = {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/JEV3_TYPED_CPU_DONE"},
                                               {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
              "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"}, "interval_s": 300,
                         "grace_period_s": 900, "fail_threshold": 2, "on_fail": "kill"}}
    sub = {"chip": "cpu-8", "mission": "ssdm-cpu-tests", "idempotency_key": f"ssdm-jev3-typed-cpu-{sha_blob[:12]}",
           "label": f"jev3_typed_cpu: draw={env.get('DRAW', '1')} tests on {full[:8]}", "dockerfile_path": "Dockerfile",
           "build_timeout_minutes": 20, "command": "bash /opt/job/run.sh", "env": env,
           "max_duration_minutes": max_min, "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
           "hf_cache": False, "checks": checks}
    receipt = {"job": "jev3_typed_cpu", "trees": {"tree": full},
               "context": {"file": str(f_out), "sha256": sha_blob, "size_bytes": size},
               "cost": {"worst_case_usd": round(0.27 * max_min / 30, 3), "basis": f"cpu-8 $0.27 per 30 min -> {max_min} min"},
               "calls": [{"tool": "open_mission", "args": {"name": "ssdm-cpu-tests"}}, {"tool": "submit_job", "args": sub}]}
    (out / "submit.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "CONTEXT_MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(sha_blob, size)


main()
