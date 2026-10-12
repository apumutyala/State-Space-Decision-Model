"""Print board rows (pins, endpoint condition, CIs, measured_in) for given keys (data only)."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
a = json.load(open(sys.argv[1], encoding="utf-8"))
keys = sys.argv[2].split(",")
for s in a["systems"]:
    if s["key"] not in keys:
        continue
    v = s.get("v16") or {}
    print("##", s["key"], "|", s.get("model_pin"), "| measured_in", s.get("measured_in"), "| last", s.get("last_measured_on"))
    print("   axes", {k: round(x, 2) for k, x in s["axes"].items() if x is not None}, "A", round(s["scores"]["A"], 2), "cap", round(s["capability"], 2))
    print("   ci95", json.dumps(v.get("ci95"))[:400])
    it = s["intelligence"]
    print("   I_open", round(it.get("I_open"), 2), "I_sealed", round(it.get("I_sealed"), 2))
    print("   endpoint:", (s.get("endpoint_condition") or "")[:500])
    print("   prov:", json.dumps(s.get("provenance"))[:300])
