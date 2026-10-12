import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
a = json.load(open(sys.argv[1], encoding="utf-8"))
for s in a["systems"]:
    if s["key"] in sys.argv[2].split(","):
        print("##", s["key"])
        print(json.dumps(s.get("cost"), indent=1)[:1500])
        print(json.dumps(s.get("speed")))
        print("calib", json.dumps(s.get("calibration"))[:600])
