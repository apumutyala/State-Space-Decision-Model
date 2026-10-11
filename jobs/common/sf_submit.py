"""Upload a launcher package and submit it via `autor api` (WSL). Usage: python -I jobs/common/sf_submit.py <launcher --out dir containing submit.json> [--dry]"""
import hashlib, json, os, subprocess, sys, pathlib, time

BLOCK = 1 << 20  # low-RAM: the package is only ever read in 1 MiB blocks (peak RSS ~ tens of MB)

AUTOR = ["wsl.exe", "-d", "bi-ubuntu", "--", "/root/.local/bin/autor", "--workspace", "default", "api"]


def api(method, path, body=None):
    r = subprocess.run(AUTOR + [method, path, "--json", "-q"] + (["-d", json.dumps(body)] if body is not None else []),
                       capture_output=True)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode:
        raise SystemExit(f"{method} {path} rc={r.returncode}: {out[:800]} {r.stderr.decode('utf-8','replace')[:800]}")
    return json.loads(out) if out.strip() else {}


def main():
    d = pathlib.Path(sys.argv[1]); dry = "--dry" in sys.argv
    s = json.loads((d / "submit.json").read_text(encoding="utf-8"))
    ctx = s["context"]; calls = {c["tool"]: c["args"] for c in s["calls"]}
    sub = dict(calls["submit_job"]); mission = calls["open_mission"]["name"]
    # env values packaged as "<...>" placeholders (e.g. CKPT_URL, a signed URL) come from THIS shell's environment at
    # submit time, so the secret is never written to disk; a placeholder left unfilled refuses a real submit
    env = dict(sub.get("env") or {})
    holes = [k for k, v in env.items() if isinstance(v, str) and v.startswith("<")]
    for k in holes:
        if os.environ.get(k):
            env[k] = os.environ[k]
    sub["env"] = env
    left = [k for k in holes if env[k].startswith("<")]
    print("env placeholders:", {k: ("from shell env" if k not in left else "UNFILLED") for k in holes})
    if left and not dry:
        raise SystemExit(f"export {', '.join(left)} in this shell first (placeholders in submit.json are never sent)")
    print("mission", mission, "worst", s["cost"]["worst_case_usd"])
    # the bytes uploaded must be the bytes create_context declares: sha256 + size re-checked, streamed (1 MiB blocks)
    f = pathlib.Path(ctx["file"]); h = hashlib.sha256(); n = 0
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(BLOCK), b""):
            h.update(b); n += len(b)
    if (h.hexdigest(), n) != (ctx["sha256"], ctx["size_bytes"]):
        raise SystemExit(f"{f}: sha256/size {h.hexdigest()} {n} != submit.json {ctx['sha256']} {ctx['size_bytes']}")
    print("context file verified", ctx["sha256"][:12], n, "bytes")
    if dry:
        shown = {**sub, "env": {k: ("<set from shell env, not printed>" if k in holes and k not in left else v)
                                for k, v in env.items()}}
        print(json.dumps(shown, indent=1)); return
    api("post", "/missions", {"name": mission})
    c = api("post", "/contexts", {"sha256": ctx["sha256"], "size_bytes": ctx["size_bytes"]})
    cid = c["context_id"]; up = c["upload"]
    urls = up["multipart"]["urls"] if "multipart" in up else [up.get("url") or up.get("put_url")]
    part = up.get("multipart", {}).get("part_size", ctx["size_bytes"])
    for i, u in enumerate(urls):
        # low-RAM (2026-10-11): curl -T reads from disk; a single part uploads the package file itself, a multipart
        # part is copied to a temp file in 1 MiB blocks (never the whole file in memory)
        tmp = f if len(urls) == 1 else d / f"part{i}"
        if tmp != f:
            with open(f, "rb") as src, open(tmp, "wb") as w:
                src.seek(i * part); left = part
                while left > 0 and (b := src.read(min(BLOCK, left))):
                    w.write(b); left -= len(b)
        r = subprocess.run(["curl", "-s", "-o", "NUL", "-w", "%{http_code}", "-T", str(tmp), u], capture_output=True)
        if tmp != f:
            tmp.unlink()
        code = r.stdout.decode()
        print("part", i, code)
        if code != "200":
            raise SystemExit("upload failed")
    fin = api("post", f"/contexts/{cid}/finalize")
    print("finalize", fin.get("status"))
    for _ in range(120):  # finalize is async: wait for the context to be ready before posting the job
        if api("get", f"/contexts/{cid}").get("status") == "ready":
            break
        time.sleep(5)
    else:
        raise SystemExit(f"context {cid} not ready after 10 min")
    sub.pop("context_id", None)
    sub.update({"context_id": cid, "mission": mission})
    j = api("post", "/jobs", sub)
    print("JOB", j.get("job_id"), j.get("status"), "max_cost", j.get("max_cost_usd"))


main()
