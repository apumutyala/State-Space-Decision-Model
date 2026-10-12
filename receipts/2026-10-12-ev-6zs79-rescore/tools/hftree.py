"""Print every file of a HF repo at a revision with size and LFS sha256 (data only). Usage: hftree.py repo rev [out.sha256]"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
repo, rev = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
with urllib.request.urlopen(url, timeout=60) as r:
    tree = json.load(r)
lines = []
for f in tree:
    if f.get("type") != "file":
        continue
    sha = (f.get("lfs") or {}).get("oid")
    print(f"{f['path']}\t{f.get('size')}\t{sha or 'git:' + f.get('oid', '')}")
    lines.append((f["path"], sha))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as fh:
        for p, s in lines:
            fh.write(f"{s}  {p}\n" if s else f"#git  {p}\n")
