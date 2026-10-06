"""Tokenizer-only (CPU, no weights): are fixture v0 inputs token-identical under the
29b858c untrusted encoding (split_special_tokens only) and decide.untrusted (ce7ae36)?"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from transformers import AutoTokenizer

from decide.prompt import TEMPLATES, build_prompt_segments, continuation_segments
from decide.schema import canonical_options
from decide.untrusted import UntrustedEncoder

tok = AutoTokenizer.from_pretrained(sys.argv[2])
new = UntrustedEncoder(tok)
tmpl = TEMPLATES["jrt-chatml-v1"]


def enc(segs, mode):
    ids = []
    for t, trusted in segs:
        if trusted:
            ids += tok(t, add_special_tokens=False).input_ids
        elif mode == "old":
            ids += tok(t, add_special_tokens=False, split_special_tokens=True).input_ids
        else:
            ids += new.encode(t)
    return ids


resp = {json.loads(l)["case_id"]: json.loads(l)["response"] for l in open(root / "fixtures/replay/v0/responses.jsonl", encoding="utf-8")}
checked = diff = 0
for line in open(root / "fixtures/replay/v0/requests.jsonl", encoding="utf-8"):
    case = json.loads(line)
    s = resp[case["case_id"]]["ssdm"]
    if s["status"] != "ok" and (s["error"] or {}).get("code") not in ("input_tokens_exceeded", "inference_calls_exceeded", "deadline_exceeded"):
        continue
    req = case["request"]
    for qid, q in req["questions"].items():
        segs = [build_prompt_segments(tmpl, req, qid)] + [continuation_segments(tmpl, o) for o, _ in canonical_options(q)]
        for sg in segs:
            checked += 1
            if enc(sg, "old") != enc(sg, "new"):
                diff += 1
                print("DIFF", case["case_id"], qid)
print(json.dumps({"sequences_checked": checked, "sequences_differing": diff, "n_cases": len(resp)}))
