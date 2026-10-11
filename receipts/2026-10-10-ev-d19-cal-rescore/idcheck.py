"""Fit-row exclusion check: the 804 pool ids (and their source-item tokens) vs the frozen DEV/TEST manifests,
proxy600 and the JevBench public pin/fingerprints. Usage: idcheck.py <pool_ids.json> <agent-main worktree>"""
import gzip, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ids = json.load(open(sys.argv[1]))["ids"]
W = Path(sys.argv[2])
files = {"dev-manifest": W / "harness/manifests/dev-manifest.jsonl.gz", "test-manifest": W / "harness/manifests/test-manifest.jsonl.gz",
         "proxy600": W / "jobs/common/dev_proxy600.json", "jev-pin": W / "jevkit/data/jevbench_public_pin.json",
         "jev-fingerprints": W / "jevkit/data/jevbench_public_fingerprints.json"}
txt = {}
for k, p in files.items():
    b = p.read_bytes()
    txt[k] = (gzip.decompress(b) if p.suffix == ".gz" else b).decode("utf-8")
    print(k, p.relative_to(W), "sha256", hashlib.sha256(b).hexdigest()[:16], "chars", len(txt[k]))


def token(i):  # m2:<source>:<item>[~perturb]/<field>
    m = re.match(r"m2:([^:]+):([^/~]+)", i)
    return (m.group(1), m.group(2)) if m else (None, i)


src = Counter(token(i)[0] for i in ids)
print("pool n", len(ids), "unique", len(set(ids)), "sources", dict(src))
toks = {token(i)[1] for i in ids}
print("unique source-item tokens", len(toks))
res = {}
for k, t in txt.items():
    full = sum(1 for i in ids if i in t)
    tk = sorted(x for x in toks if len(x) >= 8 and x in t)
    res[k] = {"full_id_hits": full, "source_item_token_hits": len(tk), "examples": tk[:5]}
print(json.dumps(res, indent=1))
print("short tokens (<8 chars, not searched):", sum(1 for x in toks if len(x) < 8))
