"""Real Qwen3.5-0.8B (same hybrid GDN/attention family as the 9B) on CPU: is the job's self-check gap bf16 noise?

For a few real rows (the first rows of the job's own self-check sources: training records and SELECT DEV requests):
  cache_X   = jobs.q9_feature_cache.extract.extract_request in dtype X (the production cached-fork path)
  plain_X   = a plain full-sequence forward over row.ids + continuation_k in dtype X (the self-check reference)
Reports rel diffs of the fork feature and abs diffs of the LM continuation score:
  cache_bf16 vs plain_bf16   (what the job's self-check measures)
  cache_fp32 vs plain_fp32   (the logic: must be ~1e-6 if the cached path is exact)
  cache_bf16 vs plain_fp32 and plain_bf16 vs plain_fp32   (each bf16 path's distance to the fp32 truth)
"""
import gzip
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, sys.argv[1])
from transformers import AutoTokenizer  # noqa: E402

from heads.layout import hf_segment_encoder  # noqa: E402
from jobs.q9_feature_cache import extract as X  # noqa: E402

MODEL = sys.argv[2]
DEV = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4

enc = hf_segment_encoder(AutoTokenizer.from_pretrained(MODEL))
reqs = []
with gzip.open(DEV, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        req, _ = X.dev_request(r, "stock")
        rows, bad = X.build_rows(req, enc)
        if rows and not bad and max(len(x.ids) for x in rows) < 1500 and len(rows) <= 3:
            reqs.append((r["_evaluation"]["run_id"], rows))
        if len(reqs) >= N:
            break

text32, w32, _ = X.load(MODEL, "cpu", "float32")
text16, w16, _ = X.load(MODEL, "cpu", "bfloat16")


@torch.no_grad()
def plain(text, w, row, k):
    seq = row.ids + row.continuations[k]
    h = text(input_ids=torch.tensor([seq]), use_cache=False).last_hidden_state[0]
    lp = torch.log_softmax((h[len(row.ids) - 1: len(seq) - 1] @ w.T).float(), -1)
    return h[-1].float(), float(lp.gather(-1, torch.tensor(row.continuations[k]).unsqueeze(-1)).sum())


def rel(a, b):
    return float((a - b).norm() / b.norm())


out = []
for rid, rows in reqs:
    c16, _ = X.extract_request(text16, w16, rows, opt_chunk=32)
    c32, _ = X.extract_request(text32, w32, rows, opt_chunk=32)
    for r, f16, f32 in zip(rows, c16, c32):
        for k in range(min(r.n_options, 3)):
            p16, l16 = plain(text16, w16, r, k)
            p32, l32 = plain(text32, w32, r, k)
            cf16, cf32 = f16["fork"][k].float(), f32["fork"][k].float()  # stored bf16 in both
            out.append({"rid": rid, "qid": r.qid, "k": k, "T": len(r.ids), "n_cont": len(r.continuations[k]),
                        "fork_rel_cache16_vs_plain16": rel(cf16, p16),
                        "fork_rel_cache32_vs_plain32": rel(cf32, p32),
                        "fork_rel_cache16_vs_truth": rel(cf16, p32), "fork_rel_plain16_vs_truth": rel(p16, p32),
                        "lm_cache16_vs_plain16": abs(float(f16["lm_lp"][k]) - l16),
                        "lm_cache32_vs_plain32": abs(float(f32["lm_lp"][k]) - l32),
                        "lm_cache16_vs_truth": abs(float(f16["lm_lp"][k]) - l32), "lm_plain16_vs_truth": abs(l16 - l32)})
            print(json.dumps({k2: (round(v, 5) if isinstance(v, float) else v) for k2, v in out[-1].items()}), flush=True)
Path(sys.argv[5] if len(sys.argv) > 5 else "q9fcl_probe08.json").write_text(json.dumps(out, indent=1))
