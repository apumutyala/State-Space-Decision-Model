"""CPU, tokenizer only: render every public item through the vendored torchcast shim + TorchcastEngine.render and compare
the prompt token count with the author's release run (usage.prompt_tokens; vLLM counts the rendered chat prompt).

    uv run --no-project --with transformers==5.17.0 --with jinja2 python tc_tokcheck.py <repo root> <tok dir> <ref results.jsonl>
"""
import json
import sys
from pathlib import Path

root, tokdir, ref = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else "plain"
sys.path.insert(0, str(root))
import third_party.torchcast_shim as T  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

mod = T.load_shim()
tok = AutoTokenizer.from_pretrained(tokdir)
refs = {}
for line in open(ref, encoding="utf-8"):
    r = json.loads(line)
    refs[r["task_id"]] = r
items = load_public()
same = diff = 0
diffs = []
for it in items:
    q = it.question
    opts = mod.options_from(q.get("criteria"), q.get("type"))
    prompt = mod.build_prompt(it.state or "", q.get("instructions") or "", opts)
    msgs = [{"role": "system", "content": mod.SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **mod.CHAT_TEMPLATE_KWARGS)
    n = len(tok(text, add_special_tokens=MODE == "special").input_ids)
    rn = (refs.get(it.id) or {}).get("usage", {}).get("prompt_tokens")
    if rn == n:
        same += 1
    else:
        diff += 1
        diffs.append((it.id, n, rn))
import collections  # noqa: E402

print(json.dumps({"mode": MODE, "items": len(items), "ref_items": len(refs), "same": same, "diff": diff,
                  "delta_hist": collections.Counter((rn or 0) - n for _, n, rn in diffs), "first_diffs": diffs[:5],
                  "head": text[:80], "first_ids": tok(text, add_special_tokens=MODE == "special").input_ids[:5],
                  "bos": [tok.bos_token, tok.bos_token_id]}, indent=1, default=str))
