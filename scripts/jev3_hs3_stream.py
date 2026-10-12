"""Stream HelpSteer3 feedback/train.jsonl.gz (CC-BY-4.0, ~102 MB, never written to disk) and cache the short,
human-consensus-graded responses for the jev3 adequacy family (spec amendment A1, FAMILY_SPEC_JEV3.md).

    python -X utf8 -I scripts/jev3_hs3_stream.py --out <hs3_cache.jsonl.gz>

Gold: each annotator's free-text feedback opens with "The response is {not|slightly|partially|mostly|perfectly}
helpful". good = mostly / perfectly, bad = not / slightly, partially = neutral. A response is kept only with >= 2 parsed
verdicts, a strict majority on one side and NO verdict on the other side (human consensus; never relabelled).
Kept: English, domain general / stem, context (all turns) + response <= 2,400 chars. Both responses of a context may be
kept (each is separately graded); they share the slice group (the context).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.request

URL = ("https://huggingface.co/datasets/nvidia/HelpSteer3/resolve/f6d145777bcbde96137596340fab89793acd1031/"
       "feedback/train.jsonl.gz")
V = re.compile(r"the response is (not|slightly|partially|mostly|perfectly) helpful", re.I)
SIDE = {"not": -1, "slightly": -1, "partially": 0, "mostly": 1, "perfectly": 1}


def verdict(fb) -> tuple[int | None, list]:
    if isinstance(fb, str):
        try:
            fb = json.loads(fb)
        except ValueError:
            fb = [fb]
    vs = []
    for t in fb or []:
        m = V.search(str(t)[:200])
        if m:
            vs.append(m.group(1).lower())
    s = [SIDE[v] for v in vs]
    pos, neg = s.count(1), s.count(-1)
    if len(vs) >= 2 and pos > len(vs) / 2 and neg == 0:
        return 1, vs
    if len(vs) >= 2 and neg > len(vs) / 2 and pos == 0:
        return 0, vs
    return None, vs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    h = hashlib.sha256()
    st = {"lines": 0, "domain_lang": 0, "too_long": 0, "kept_responses": 0, "no_consensus": 0}
    req = urllib.request.Request(URL, headers={"User-Agent": "ssdm-jev3"})
    with urllib.request.urlopen(req, timeout=300) as resp, gzip.open(a.out, "wt", encoding="utf-8") as fo:
        class Tee(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                d = resp.read(len(b))
                h.update(d)
                b[:len(d)] = d
                return len(d)
        g = gzip.GzipFile(fileobj=io.BufferedReader(Tee(), 1 << 20))
        for ln, raw in enumerate(io.TextIOWrapper(g, encoding="utf-8", newline="\n")):
            st["lines"] += 1
            d = json.loads(raw)
            if d.get("domain") not in ("general", "stem") or (d.get("language") or "english") != "english":
                st["domain_lang"] += 1
                continue
            ctx = d["context"] if isinstance(d["context"], list) else json.loads(d["context"])
            ctx_len = sum(len(t.get("content") or "") for t in ctx)
            for k in (1, 2):
                r = d[f"response{k}"]
                if ctx_len + len(r) > 2400:
                    st["too_long"] += 1
                    continue
                v, vs = verdict(d[f"feedback{k}"])
                if v is None:
                    st["no_consensus"] += 1
                    continue
                st["kept_responses"] += 1
                fo.write(json.dumps({"line": ln, "k": k, "line_sha256": hashlib.sha256(raw.rstrip("\n").encode()).hexdigest(),
                                     "domain": d["domain"], "context": ctx, "response": r, "good": v,
                                     "verdicts": vs}, ensure_ascii=False) + "\n")
        fo.write(json.dumps({"_meta": {"url": URL, "gz_sha256": h.hexdigest(), "stats": st}}) + "\n")
    print(json.dumps({"gz_sha256": h.hexdigest(), **st}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
