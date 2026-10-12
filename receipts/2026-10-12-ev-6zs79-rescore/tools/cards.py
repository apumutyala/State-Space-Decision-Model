"""Fetch HF model cards (README.md) for board open-weights rows and grep for public-item counts.

Data only: writes each card into outdir, prints matching lines. Run with python -I.
"""
import json
import os
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
board, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
a = json.load(open(board, encoding="utf-8"))
pat = re.compile(r"(/\s*231\b|\b231\b|public (items|set)|easy\s*\d+\s*/\s*48|/\s*111\b|/\s*72\b)", re.I)
for s in a["systems"]:
    repo = s.get("repo") or ""
    m = re.match(r"https://huggingface.co/([^/]+/[^/?#]+)", repo)
    if not m:
        continue
    rid = m.group(1)
    url = f"https://huggingface.co/{rid}/raw/main/README.md"
    fn = os.path.join(outdir, rid.replace("/", "__") + ".md")
    try:
        if not os.path.exists(fn):
            with urllib.request.urlopen(url, timeout=30) as r:
                open(fn, "wb").write(r.read())
        txt = open(fn, encoding="utf-8", errors="replace").read()
    except Exception as e:  # noqa: BLE001
        print(f"## {s['key']} {rid} FETCH FAIL {e}")
        continue
    hits = [ln.strip()[:260] for ln in txt.splitlines() if pat.search(ln)]
    ax = s["axes"]
    print(f"## {s['key']} {rid} open={s.get('open')} I={ax.get('intelligence')!s:.5} hits={len(hits)}")
    for h in hits[:8]:
        print("   ", h)
