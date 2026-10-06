"""Smoke-test the HF readout on small local checkpoints, one backbone per process.

    python -I scripts/smoke_backbones.py --family qwen3.5 --model-dir <snap> [--extras <dir> ...]
    python -I scripts/smoke_backbones.py --family rwkv7   --model-dir <snap> --extras .venv-extras
    python -I scripts/smoke_backbones.py --family mamba3  --model-dir <snap> --tokenizer-dir <llama3.1 tok> \
        --extras .venv-extras-mamba .venv-extras-t35

Runs 4 synthetic requests (3 repeats each; median latency), a permutation
equivariance check, and an option-order probe: the same two-way question with
option ids renamed so the canonical (sorted) order flips. A readout without
position bias keeps the same *meaning* on top; flipping with the order signals
a first-listed-option bias. Prints one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQS = {
    "routing": {"model": "smoke", "state": "Our checkout started returning errors and orders are blocked.",
                "questions": {"department": {"type": "choice", "instructions": "Which team should handle the message?",
                                             "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
                              "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                              "outage": {"type": "noul", "instructions": "Is a service down?"}}},
    "invoice": {"model": "smoke", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
                "questions": {"status": {"type": "choice", "instructions": "What is the invoice status?",
                                         "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
                              "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}},
    "capital": {"model": "smoke", "state": "Geography quiz.",
                "questions": {"capital": {"type": "choice", "instructions": "Which city is the capital of France?",
                                          "criteria": {"berlin": "Berlin", "madrid": "Madrid", "paris": "Paris", "rome": "Rome"}}}},
    "sentiment": {"model": "smoke", "state": "Terrible service, I waited two hours and the food was cold.",
                  "questions": {"sentiment": {"type": "score", "instructions": "Rate the review sentiment.",
                                              "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}}},
}
# Option-order probe: same meanings, ids chosen so sorted order is A=(billing, technical) vs B=(a_technical, b_billing)
PROBE_A = {"model": "smoke", "state": "Our checkout started returning errors and orders are blocked.",
           "questions": {"q": {"type": "choice", "instructions": "Which team should handle the message?",
                               "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}}}}
PROBE_B = {"model": "smoke", "state": "Our checkout started returning errors and orders are blocked.",
           "questions": {"q": {"type": "choice", "instructions": "Which team should handle the message?",
                               "criteria": {"b_billing": "Payments or invoices", "a_technical": "Bugs or outages"}}}}
TEMPLATE = {"qwen3.5": "jrt-chatml-v1", "rwkv7": "jrt-rwkv-g1-v1", "mamba3": "jrt-plain-v1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(TEMPLATE))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--extras", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    for i, e in enumerate(args.extras):
        p = str((ROOT / e).resolve())
        sys.path.insert(1 + i, p)
        tcc = Path(p) / "triton" / "runtime" / "tcc" / "tcc.exe"
        if tcc.exists():  # triton-windows looks for its bundled TinyCC only in site-packages
            os.environ["CC"] = str(tcc)

    from decide import Decider
    from decide.backends import hf_readout as H

    summary = {"family": args.family, "model_id": args.model_id, "device": args.device, "dtype": args.dtype,
               "extras": args.extras}
    try:
        t0 = time.time()
        if args.family == "qwen3.5":
            plug = H.load_qwen35(args.model_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
        elif args.family == "rwkv7":
            plug = H.load_rwkv7(args.model_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
        else:
            plug = H.load_mamba3(args.model_dir, args.tokenizer_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
        summary["load_s"] = round(time.time() - t0, 2)
        summary["loaded"] = True
    except Exception as exc:
        summary["loaded"] = False
        summary["load_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        summary["traceback"] = traceback.format_exc(limit=4)[-2000:]
        print(json.dumps(summary, indent=1))
        return
    backend = H.HFReadoutBackend(plug, TEMPLATE[args.family])
    d = Decider(backend, backend_max_input_tokens=plug.max_context)
    summary["revision_id"] = backend.revision()["revision_id"]
    summary["kernel_stack"] = plug.kernel_stack
    summary["notes"] = plug.notes
    import torch

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    results = {}
    for name, req in REQS.items():
        lat, resp = [], None
        for _ in range(3):
            resp = d.decide(req)
            lat.append(resp["ssdm"]["diagnostics"]["latency_ms"])
        s = resp["ssdm"]
        results[name] = {
            "status": s["status"], "error": s["error"], "answers": resp["answers"],
            "latency_ms": {"median": statistics.median(lat), "min": min(lat), "max": max(lat), "n": len(lat)},
            "calls": s["diagnostics"]["inference_calls"], "input_tokens": resp["usage"]["input_tokens"],
        }
        # equivariance on a reversed key order
        rev = dict(req)
        rev["questions"] = {}
        for qid, q in reversed(list(req["questions"].items())):
            q = dict(q)
            if q["type"] == "choice":
                q["criteria"] = dict(reversed(list(q["criteria"].items())))
            rev["questions"][qid] = q
        r2 = d.decide(rev)
        diff = 0.0
        if s["status"] == "ok" and r2["ssdm"]["status"] == "ok":
            for qid, probs in s["served_probabilities"].items():
                for o, p in probs.items():
                    diff = max(diff, abs(r2["ssdm"]["served_probabilities"][qid][o] - p))
        results[name]["equivariance_max_abs_diff"] = diff
    summary["requests"] = results
    pa, pb = d.decide(PROBE_A), d.decide(PROBE_B)
    if pa["ssdm"]["status"] == "ok" and pb["ssdm"]["status"] == "ok":
        a, b = pa["ssdm"]["served_probabilities"]["q"], pb["ssdm"]["served_probabilities"]["q"]
        summary["option_order_probe"] = {
            "A_sorted_first=billing": a, "B_sorted_first=a_technical": b,
            "p_technical_A": a["technical"], "p_technical_B": b["a_technical"],
            "first_listed_bias_signal": (a["billing"] > 0.5) and (b["a_technical"] > 0.5),
        }
    else:
        summary["option_order_probe"] = {"error": [pa["ssdm"]["error"], pb["ssdm"]["error"]]}
    if args.device.startswith("cuda"):
        summary["peak_cuda_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        summary["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
