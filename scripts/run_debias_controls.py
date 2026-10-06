"""Run the debias controls (decide/debias.py) on one backbone.

    python -I scripts/run_debias_controls.py --family qwen3.5 --model-dir <snap> --model-id Qwen/Qwen3.5-0.8B --device cpu --dtype float32
    (rwkv7 / mamba3 as in scripts/smoke_backbones.py; those need --device cuda)

Prints one JSON report per probe question. Required before any stock-readout
m0 number is used to decide readout design (PR #6 review, B3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBES = [
    ({"model": "debias", "state": "Our checkout started returning errors and orders are blocked.",
      "questions": {"q": {"type": "choice", "instructions": "Which team should handle the message?",
                          "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}}}}, "q"),
    ({"model": "debias", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "status": "overdue"}},
      "questions": {"q": {"type": "choice", "instructions": "What is the invoice status?",
                          "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}}}}, "q"),
    ({"model": "debias", "state": "Geography quiz.",
      "questions": {"q": {"type": "choice", "instructions": "Which city is the capital of France?",
                          "criteria": {"berlin": "Berlin", "madrid": "Madrid", "paris": "Paris", "rome": "Rome"}}}}, "q"),
]
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
        if tcc.exists():
            os.environ["CC"] = str(tcc)
    from decide import Decider
    from decide.backends import hf_readout as H
    from decide.debias import run_debias_controls

    if args.family == "qwen3.5":
        plug = H.load_qwen35(args.model_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
    elif args.family == "rwkv7":
        plug = H.load_rwkv7(args.model_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
    else:
        plug = H.load_mamba3(args.model_dir, args.tokenizer_dir, device=args.device, dtype=args.dtype, model_id=args.model_id)
    backend = H.HFReadoutBackend(plug, TEMPLATE[args.family])
    d = Decider(backend, backend_max_input_tokens=plug.max_context)
    out = {"revision_id": backend.revision()["revision_id"], "device": args.device, "dtype": args.dtype,
           "reports": [run_debias_controls(d, req, qid) for req, qid in PROBES]}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
