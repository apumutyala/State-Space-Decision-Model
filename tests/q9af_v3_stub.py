"""CPU stand-ins for the fetch / GPU / kit steps of jobs/q9af_v3/q9af_v3_job.sh (Q9AF3_DRYRUN=1, tests/test_q9af_v3.py).

Everything else in the runner (params caps, the measured-rate preflight, the shipped-data sha256 check, the artifact
pin status, the proxy600 identity + rows, the rate-guard preflights on the stub's throughput, the stage-A record, the
gate and finalize) runs for real.
    python -m tests.q9af_v3_stub setup|fetch|mix|gen|stagea|stageb|export|kit|subset|v3wid|fetch_artifact ...
Env Q9AF3_STUB_REF_POINT / _REF_LO steer the v3 - reference compare so tests can walk the verdicts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ATTN = [3, 7, 11, 15, 19, 23, 27, 31]
PROXY_SHA = "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6"
POINT = float(os.environ.get("Q9AF3_STUB_REF_POINT", "1.5"))
LO = float(os.environ.get("Q9AF3_STUB_REF_LO", "0.4"))


def _w(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _b(p, data=b"dry"):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_bytes(data)


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "setup":
        _w(Path(args[0]) / "fla_guard.json", {"dry_run": True})
    elif cmd == "v3wid":
        _w(args[0], {"dry_run": True})
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
        d = Path(args[0])
        _w(d / "mix.json", {"dry_run": True, "train_tokens": 1000, "decontam": "full-eval-v1"})
        for s in ("web", "chat", "math"):
            _b(d / f"heldout_{s}.u32")
    elif cmd == "gen":
        d = Path(args[0])
        for f in ("lc.u32", "onpolicy.npz", "gen_heldout.npz", "gen_rows.npy", "gen_mask.npy"):
            _b(d / f)
        _w(d / "split.json", {"dry_run": True})
        _w(d / "results.json", {"env": {"dry_run": True}, "gen": {"completion_tokens": 12000000, "tok_s": 2700.0}})
    elif cmd == "stagea":
        ck = Path(args[0])
        _b(ck / "gdn" / "stageA.pt")
        _w(ck / "pipeline" / "results.json", {
            "plan": {"order": [27, 3, 19, 7, 31, 11, 23, 15], "sensitivity": {str(i): 0.01 for i in ATTN},
                     "stages": [{"adds": [27, 3]}, {"adds": [19, 7]}, {"adds": [31, 11]}, {"adds": [23, 15]}]},
            "stages": {f"stage{k}": {"adds": [], "best": {"crit": 0.1}, "tokens": 1, "minutes_budget": 90,
                                     "kept_last": True} for k in range(1, 5)},
            "final": {"heldout": {"mix": {"kl": 0.09}}, "gen_heldout": {"kl": 0.12}}})
    elif cmd == "stageb":
        ck = Path(args[0])
        _b(ck / "gdn" / "final.pt")
        thr = {"4096": {"tok_s": 22000.0}, "call_latency": {"small_call_s": 0.07, "pack_call_s": 0.24}}
        _w(ck / "pipeline" / "results.json", {
            "entry": {"heldout": {"mix": {"kl": 0.09}}, "long": {"kl": 0.2}},
            "long_context": {"tokens": 1}, "onpolicy": {"tokens": 1},
            "final": {"heldout": {"mix": {"kl": 0.085}}, "long": {"kl": 0.15}, "gen_heldout": {"kl": 0.11}},
            "throughput": {"stock": thr, "converted_8": thr}})
    elif cmd == "export":
        out, meta = Path(args[0]), Path(args[1])
        _b(out / "gdn_converted.pt")
        _w(meta, {"ok": True, "mode": "final", "converted": ATTN, "attention_free": True, "file": "gdn_converted.pt"})
    elif cmd == "fetch_artifact":
        which, dest = args[0], Path(args[1])
        _b(dest / "gdn_converted.pt")
        _w(dest / "export.json", {"ok": True, "mode": "final", "converted": ATTN, "attention_free": True})
        if which == "stageA":
            for f in ("lc.u32", "onpolicy.npz", "gen_heldout.npz", "heldout_web.u32", "heldout_chat.u32",
                      "heldout_math.u32"):
                _b(dest / f)
    elif cmd == "kit":
        d = Path(args[1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("{}\n", encoding="utf-8")
    elif cmd == "subset":
        if args[0] == "compare":
            cand = args[args.index("--candidate") + 1]
            base = args[args.index("--baseline") + 1]
            p, lo = (POINT, LO) if "v3w_v3" in cand else (-6.0, -9.5)  # v3 vs anything / v2 vs stock
            print(json.dumps({"B": 2000, "seed": 0, "subset_ids_sha256": PROXY_SHA, "rows": 644,
                              "candidate": cand, "baseline": base,
                              "diff": {"index": {"point": p, "lo": lo, "hi": p + 3.5}}}))
        else:
            print(json.dumps({"subset_ids_sha256": PROXY_SHA, "rows": 644}))
    else:
        print(f"unknown stub command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
