"""CPU stand-ins for the fetch / GPU / kit steps of jobs/q9af_v2/q9af_v2_job.sh (Q9AF2_DRYRUN=1, tests/test_q9af_v2.py).

Everything else in the runner (pins of the params, the measured-rate preflight, the DEV-row sha256 check, the proxy600
identity + rows, the rate-guard preflights on the stub's throughput, the rule and finalize) runs for real.
    python -m tests.q9af_v2_stub setup|fetch|mix|pipeline|export|kit|subset ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ATTN = [3, 7, 11, 15, 19, 23, 27, 31]
PROXY_SHA = "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6"
#: the stub's proxy diff (env Q9AF2_STUB_POINT / _LO override it so tests can walk the verdict ladder)
POINT = float(os.environ.get("Q9AF2_STUB_POINT", "-6.0"))
LO = float(os.environ.get("Q9AF2_STUB_LO", "-9.5"))
KL = float(os.environ.get("Q9AF2_STUB_KL", "0.08"))


def _w(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "setup":
        _w(Path(args[0]) / "fla_guard.json", {"dry_run": True})
    elif cmd == "fetch":
        root = Path(args[0])
        q9b = root / "hf" / "fake-q9b"
        q9b.mkdir(parents=True, exist_ok=True)
        (q9b / "config.json").write_text("{}", encoding="utf-8")
        env = []
        for name in ("WIKITEXT_DIR", "MMLU_DIR", "Q9B_DIR", "FINEWEB_EDU_DIR", "ULTRACHAT_DIR", "FINEMATH_DIR"):
            d = q9b if name == "Q9B_DIR" else root / "hf" / name.lower()
            d.mkdir(parents=True, exist_ok=True)
            env.append(f"export {name}={d.as_posix()}")
            (root / f"DONE_{name}").write_text(name, encoding="utf-8")
        (root / "paths.env").write_text("\n".join(env) + "\n", encoding="utf-8", newline="\n")
    elif cmd == "mix":
        _w(Path(args[0]) / "mix.json", {"dry_run": True, "train_tokens": 1000})
    elif cmd == "pipeline":
        ck = Path(args[0])
        thr = {"4096": {"tok_s": 22000.0}, "call_latency": {"small_call_s": 0.07, "pack_call_s": 0.24}}
        probe = {"ppl": {"ctx1024": {"ppl": 8.4}, "ctx4096": {"ppl": 7.2}}, "mmlu": {"acc": 0.8},
                 "mqar": {"mean_acc": 0.92}, "niah": {"mean_acc": 1.0}}
        _w(ck / "pipeline" / "results.json", {
            "final_heldout": {"mix": {"kl": KL, "delta_nll": 0.05, "hidden_cos": 0.97}},
            "probe_stock": probe, "probe_conv8": probe, "init_cmp": {"chosen": "reuse_attn"},
            "stage1": {"tokens": 1}, "stage2": {"tokens": 2, "curve": []},
            "mmlu_full": {"stock": {"acc": 0.78}, "final": {"acc": 0.75}},
            "throughput": {"stock": thr, "converted_8": thr}})
        (ck / "pipeline" / "train_log.jsonl").write_text("{}\n", encoding="utf-8")
        _w(ck / "pipeline" / "config.json", {"dry_run": True})
    elif cmd == "export":
        out, meta = Path(args[0]), Path(args[1])
        (out / "gdn_converted.pt").write_bytes(b"dry")
        _w(meta, {"ok": True, "mode": "final", "converted": ATTN, "attention_free": True, "file": "gdn_converted.pt",
                  "attention_layers_kept": []})
    elif cmd == "kit":
        d = Path(args[1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("{}\n", encoding="utf-8")
    elif cmd == "subset":
        if args[0] == "compare":
            print(json.dumps({"B": 2000, "seed": 0, "subset_ids_sha256": PROXY_SHA, "rows": 644,
                              "diff": {"index": {"point": POINT, "lo": LO, "hi": POINT + 3.5}}}))
        else:
            print(json.dumps({"subset_ids_sha256": PROXY_SHA, "rows": 644}))
    else:
        print(f"unknown stub command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
