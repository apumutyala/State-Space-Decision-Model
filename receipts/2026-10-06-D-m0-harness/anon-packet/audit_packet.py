"""Author-side leak audit of packet.json.gz against every DEV row (needs DEV rows).
Counts DEV identifiers (run/group/case ids, payload hashes, >=8 chars) and option/instruction
strings (>=12 chars) that appear anywhere in the packet text. Expected: all zero."""
import gzip, json
from pathlib import Path
from harness.cli import load_slice
from harness.cases import case_of

text = gzip.open(Path(__file__).with_name("packet.json.gz"), "rt", encoding="utf-8").read()
rows = load_slice("dev")
cid = case_of(rows)
ids = set()
strings = set()
for r in rows:
    e = r["_evaluation"]
    ids |= {e["run_id"], str(e["group_id"]), e["payload_sha256"], cid(r)}
    for q in r["questions"].values():
        for v in list(q.get("criteria", {}).values()) + [q.get("instructions")]:
            if isinstance(v, str):
                strings.add(v)
ids = {i for i in ids if len(i) >= 8}
strings = {s for s in strings if len(s) >= 12}
print(json.dumps({"packet_bytes_uncompressed": len(text), "dev_ids_checked": len(ids), "dev_ids_found": sum(i in text for i in ids), "dev_strings_checked": len(strings), "dev_strings_found": sum(s in text for s in strings)}, indent=1))
