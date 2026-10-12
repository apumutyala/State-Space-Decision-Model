"""Convert an ssdm-landing-v1 landing.json (files[].path relative to out/, mtime_utc) to the kit's --mtimes {name: epoch} map."""
import json, sys
from datetime import datetime, timezone
d = json.load(open(sys.argv[1], encoding="utf-8"))
m = {}
for f in d["files"]:
    p = f["path"]
    t = datetime.strptime(f["mtime_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    if p.startswith("screen/jevkit/"):
        k = p[len("screen/jevkit/"):]
    elif p == "cell.json":
        k = "cell.json"
    else:
        k = "out/" + p
    m[k] = t
json.dump({"source": sys.argv[1], "mtimes": m}, open(sys.argv[2], "w", encoding="utf-8"), indent=1, sort_keys=True)
print(len(m))
