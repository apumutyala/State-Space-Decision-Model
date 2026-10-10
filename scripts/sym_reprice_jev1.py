"""Training-clock re-price: computed tokens per row, sym-fork (lean) vs the JEV1-4B g1v2 shim rows, on a systematic
sample of the bschema-draw-jev1 records (real Qwen3.5-4B tokenizer, the cell's keep_item caps)."""
import gzip
import json
import math
import statistics as st
import sys

sys.path.insert(0, r"C:\Users\apumu\research\CHLabs\ssdm-wt\sym-invariant")
from decide.shim_format import bind_template  # noqa: E402
from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS  # noqa: E402
from heads.backbone import MIN_CONT_WIDTH  # noqa: E402
from heads.layout import build_row, resolve_template  # noqa: E402
from train.guard import record_request  # noqa: E402
from train.run_cell import keep_item  # noqa: E402

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
REC = r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev1\records.jsonl.gz"
bind_template("jrt-h2o-shim:qwen3.5-4b", DEFAULT_TOKENIZERS / BACKBONES["qwen3.5-4b"].tokenizer_dir)
tpl = resolve_template("jrt-h2o-shim:qwen3.5-4b")
CHUNK = 64  # cont_chunk the r1 probe settled on (job-v6qm3)


def computed(row, sym):
    """tokens the backbone runs: prefix + every continuation call, right-padded per chunk of CHUNK continuations."""
    conts = [len(c) for c in row.continuations]
    tot = len(row.ids)
    for s in range(0, len(conts), CHUNK):
        ch = conts[s:s + CHUNK]
        tot += len(ch) * max(max(ch), MIN_CONT_WIDTH)
    return tot


g, s, kept_g, kept_s, ks, n = [], [], 0, 0, [], 0
with gzip.open(REC, "rt", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i % STEP:
            continue
        rec = json.loads(line)
        req = record_request(rec)
        for qid in rec["labels"]:
            n += 1
            rg = build_row(req, qid, None, template=tpl, layout="g1v2")
            rs = build_row(req, qid, None, template=tpl, layout="sym-fork")
            okg, oks = keep_item(rg, 255, 2048), keep_item(rs, 255, 2048)
            kept_g += okg
            kept_s += oks
            if okg and oks:
                g.append((len(rg.ids) + rg.n_options, computed(rg, False)))
                s.append((len(rs.ids) + sum(len(c) for c in rs.continuations), computed(rs, True)))
                ks.append(rg.n_options)
fed_g, comp_g = st.mean(x for x, _ in g), st.mean(y for _, y in g)
fed_s, comp_s = st.mean(x for x, _ in s), st.mean(y for _, y in s)
f = comp_s / comp_g
print(json.dumps({"sample_every": STEP, "items": n, "kept_g1v2": kept_g, "kept_sym": kept_s, "paired": len(g),
                  "K_mean": st.mean(ks), "K_max": max(ks), "K_gt16": sum(k > 16 for k in ks),
                  "fed_g1v2": round(fed_g, 1), "fed_sym": round(fed_s, 1), "computed_g1v2": round(comp_g, 1),
                  "computed_sym": round(comp_s, 1), "factor_computed": round(f, 4), "factor_fed": round(fed_s / fed_g, 4),
                  "s_per_step_sym": round(10.92 * f, 2),
                  "clock_min": math.ceil(1121 * 10.92 * f * 1.3 / 60)}, indent=1))
