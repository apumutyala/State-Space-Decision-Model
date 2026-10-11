"""In-job landing record (A-PREREG-JEV3-SYM; the evidence lane's rescore kit reads it): phase timestamps (the per-phase
export records of MANIFEST.jsonl, UTC) + the gate return codes + every out/ file's size, UTC mtime and sha256, so the
selection-before-public ordering (slice_first.json vs the screen preds) is checkable from the artifact alone.

    python landing.py <work/out> <OUT/MANIFEST.jsonl> '<GATES json>' <work/out/landing.json>
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main(argv) -> int:
    out_dir, manifest, gates, dest = Path(argv[1]), Path(argv[2]), argv[3], Path(argv[4])
    phases = []
    if manifest.is_file():
        for ln in manifest.read_text(encoding="utf-8").splitlines():
            try:
                phases.append(json.loads(ln))
            except ValueError:
                pass
    files = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p == dest:
            continue
        st = p.stat()
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        files.append({"path": str(p.relative_to(out_dir)).replace(os.sep, "/"), "bytes": st.st_size,
                      "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                      "sha256": h.hexdigest()})
    try:
        g = json.loads(gates)
    except ValueError:
        g = {"_unparsed": gates}
    rec = {"schema": "ssdm-landing-v1", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gates": g, "phases": phases, "files": files}
    sf = next((f for f in files if f["path"].endswith("slice_first.json")), None)
    preds = [f for f in files if f["path"].endswith(".preds.jsonl")]
    if sf and preds:  # ordering evidence: the slice record predates every public-231 prediction file
        rec["slice_first_before_public_preds"] = all(sf["mtime_utc"] <= f["mtime_utc"] for f in preds)
    dest.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
