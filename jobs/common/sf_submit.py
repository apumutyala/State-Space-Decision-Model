"""Upload a launcher package and submit it via `autor api` (WSL). Usage: python -I jobs/common/sf_submit.py <launcher --out dir containing submit.json> [--dry]"""
import json, subprocess, sys, pathlib, time

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
    print("mission", mission, "worst", s["cost"]["worst_case_usd"])
    if dry:
        print(json.dumps(sub, indent=1)); return
    api("post", "/missions", {"name": mission})
    c = api("post", "/contexts", {"sha256": ctx["sha256"], "size_bytes": ctx["size_bytes"]})
    cid = c["context_id"]; up = c["upload"]; f = pathlib.Path(ctx["file"])
    urls = up["multipart"]["urls"] if "multipart" in up else [up.get("url") or up.get("put_url")]
    part = up.get("multipart", {}).get("part_size", ctx["size_bytes"])
    data = f.read_bytes()
    for i, u in enumerate(urls):
        chunk = data[i * part:(i + 1) * part]
        tmp = d / f"part{i}"; tmp.write_bytes(chunk)
        r = subprocess.run(["curl", "-s", "-o", "NUL", "-w", "%{http_code}", "-T", str(tmp), u], capture_output=True)
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
