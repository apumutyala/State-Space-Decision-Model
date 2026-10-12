import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
keys = sys.argv[2].split(",")
for m in sorted(set(re.findall(r"https?://[^\s\"\\)<>]+", t))):
    if any(k in m.lower() for k in keys):
        print(m)
