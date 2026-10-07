"""Self-check (v3 since the PR #111 review) (jobs.q9_feature_cache.extract.selfcheck) on real Qwen3.5-0.8B, CPU, bf16 production:
the correct cached fork, and the same path with deliberately injected fork bugs (to show the gate's power).

Bugs (each corrupts the state copy the fork continues from):
  conv_zero   GDN conv states zeroed (the causal-conv left context lost at the fork)
  rec_zero    GDN recurrent states zeroed (the delta-rule memory lost at the fork)
  rec_half    GDN recurrent states scaled by 0.5 (a subtle state error)
  kv_drop1    the last prefix position dropped from every attention layer's KV cache (an off-by-one)
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

MODEL, DEV, N, OUT = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg = json.loads((Path(sys.argv[1]) / "jobs/q9_feature_cache/cache_config.json").read_text(encoding="utf-8"))
sc = cfg["selfcheck"]
enc = hf_segment_encoder(AutoTokenizer.from_pretrained(MODEL))
reqs = []
with gzip.open(DEV, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        req, _ = X.dev_request(r, "stock")
        rows, bad = X.build_rows(req, enc)
        if rows and not bad and max(len(x.ids) for x in rows) < 1200 and len(rows) <= 2:
            reqs.append(rows)
        if len(reqs) >= N:
            break
text, w, _ = X.load(MODEL, "cpu", "bfloat16")
orig_clone = X._clone


def corrupt(kind):
    def clone(cache):
        c = orig_clone(cache)
        for layer in c.layers:
            if kind in ("conv_zero", "rec_zero", "rec_half"):
                name = "conv_states" if kind == "conv_zero" else "recurrent_states"
                slots = getattr(layer, name, None)
                items = slots.items() if isinstance(slots, dict) else enumerate(slots) if isinstance(slots, list) else ()
                for k, v in list(items):
                    if isinstance(v, torch.Tensor) and v.ndim > 0:
                        slots[k] = v * 0.5 if kind == "rec_half" else torch.zeros_like(v)
            elif kind == "kv_drop1":
                for name in ("keys", "values"):
                    v = getattr(layer, name, None)
                    if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[-2] > 1:
                        setattr(layer, name, v[..., :-1, :].contiguous())
        return c
    return clone


res = {}
for kind in ("correct", "conv_zero", "rec_zero", "rec_half", "kv_drop1"):
    X._clone = orig_clone if kind == "correct" else corrupt(kind)
    ok, recs, info = X.selfcheck(text, w, reqs, 32, sc)
    res[kind] = {"pass": ok, "checks": recs, "info": info}
    print(kind, "PASS" if ok else "FAIL", json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                                      for r in recs]), flush=True)
X._clone = orig_clone
Path(OUT).write_text(json.dumps({"model": "Qwen/Qwen3.5-0.8B (local snapshot)", "device": "cpu", "dtype": "bfloat16",
                                 "selfcheck_constants": sc, "results": res}, indent=1), encoding="utf-8")
