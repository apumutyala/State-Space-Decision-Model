"""The REAL convert.pipeline on the tiny stand-in (dry run of jobs/full8/full8_job.sh phase P).

`python -m tests.full8_dryrun.pipeline_tiny --config configs/full8_h100.json --out <CK>/pipeline` takes the REAL job
config, shrinks it only in size (sequence length, batch, steps, windows, data volume: written next to the output as
dryrun_config.json and recorded by the pipeline as config.json), replaces the 9B load / wikitext tokenisation / the
probe suite / prefill timing with CPU stand-ins, and runs convert.pipeline.main unchanged: stage 1, KL importance,
the 8 progressive steps with the enforced gate and rollback, checkpoints into $FULL8_CK/gdn, PHASE_HOOK, resume.

Env (dry run only):
  FULL8_DRYRUN_GATE   "real" = the tiny model's own gate numbers;  "pass" (default) = the real evaluate runs, then its
                      numbers are replaced by small deterministic, rule-consistent ones (every layer passes);
                      "reject:<l1,l2>" = as pass, but any trial containing those layers fails the cosine bound
  FULL8_DRYRUN_DEADLINE_AFTER_TRAIN=<n>  the deadline passes right after the n-th stage-2 run (A1 B2 test); the
                      shrunk config then sets probe_after_deadline false, so that layer's probes are deadline-skipped
                      and must be repaired on resume
  FULL8_DRYRUN_CRASH_AFTER_STEP=<k>   hard-kill the process (os._exit 137, no cleanup: a preemption) right after the
                      k-th progressive step has been persisted (results.json + prog_k.pt + the checkpoint POST)
"""
from __future__ import annotations

import copy
import json
import os
import sys


def shrink(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c.update(seq_len=24, batch=4, micro_batch=2, model_dir=None, dry_run=True)
    c.pop("vram_fraction", None)
    c["data"]["train_max_tokens"] = 4000
    c["stage1"]["steps"] = 3
    c["stage2"]["steps"] = 3
    c["gate"]["n_windows"] = 4
    c["throughput"] = {"lengths": [24], "reps": 1}
    if os.environ.get("FULL8_DRYRUN_DEADLINE_AFTER_TRAIN"):
        c["progressive"]["probe_after_deadline"] = False
    return c


def _layer_w(i):
    return 0.001 * (1 + (int(i) * 5) % 8)  # a deterministic, non-trivial importance order


def patch(pipeline):
    import numpy as np
    import torch  # noqa: F401

    from probes import throughput as T
    from tests.full8_dryrun.tiny import VOCAB, mmlu_rows, stub_probes, tiny_convertible, tokenizer

    pipeline.load = lambda *a, **k: (tiny_convertible(0), None, "tiny")
    toks = {s: np.random.default_rng(i).integers(0, VOCAB, 6000).astype(np.uint32)
            for i, s in enumerate(("train", "gate", "eval"))}
    pipeline.data.build = lambda tok, s, max_tokens=None: (s, {"split": s, "dry_run": True})
    pipeline.data.load_tokens = lambda p: toks[p]
    pipeline.probes.run_all = stub_probes
    real_mmlu_full, ttok, rows = pipeline.probes.mmlu_full, tokenizer(), mmlu_rows()
    # the REAL full-MMLU scorer (batching, padding, letter logits) on the tiny model, with a tiny tokenizer and a
    # synthetic item set of the real size
    # (padding multiple 32 instead of 128 on CPU only: the tiny prompts are ~30 tokens; scoring is unchanged)
    pipeline.probes.mmlu_full = lambda model, tok, **k: real_mmlu_full(model, ttok, rows=rows, mult=32, **k)
    # prefill timing is a GPU number: the stand-in replays job-5cmat's measured minimum (so the runner's real Index
    # volume preflight sees a 9B-like rate); the per-call latency is MEASURED for real on the tiny model (both arms:
    # the 5cmat dtype failure must not recur), then its values are replaced by job-5cmat's 9B receipts, since a loaded
    # CPU's 0.1-0.6 s per call would make the runner's real Index preflight flaky
    pipeline.measure_throughput = lambda model, modes, lengths, reps=3: {
        str(L): {"tok_s": 22144.5, "dry_run": "job-5cmat prefill rate replayed"} for L in lengths}
    def call_latency(model, modes):
        cpu = T.call_latency(model, modes, prefix_len=32, tail=(2, 4), small_len=4, reps=2)
        assert cpu["pack_call_s"] > 0 and cpu["small_call_s"] > 0
        return {**cpu, "pack_call_s": 0.2369, "small_call_s": 0.0493, "measured_on_cpu": cpu,
                "dry_run": "job-5cmat call latency replayed"}

    pipeline.call_latency = call_latency
    mode = os.environ.get("FULL8_DRYRUN_GATE", "pass")
    if mode != "real":
        reject = {int(x) for x in mode.split(":", 1)[1].split(",")} if mode.startswith("reject:") else set()
        real_eval = pipeline.align.evaluate

        def evaluate(model, windows, modes, teacher=None):
            r = real_eval(model, windows, modes, teacher=teacher)  # the real evaluate runs; its numbers are replaced
            stud = [int(i) for i, m in modes.items() if m == "student"]
            d = 0.002 * len(stud)
            r.update(nll_student=r["nll_teacher"] + d, delta_nll=d, kl=sum(_layer_w(i) for i in stud),
                     hidden_cos=0.9 if reject & set(stud) else 0.999, dry_run_gate=mode)
            return r

        pipeline.align.evaluate = evaluate
    crash = os.environ.get("FULL8_DRYRUN_CRASH_AFTER_STEP")
    dl_after = os.environ.get("FULL8_DRYRUN_DEADLINE_AFTER_TRAIN")
    if crash or dl_after:
        real_prog = pipeline.run_progressive

        def run_progressive(order, prog, gate, ops, max_layers, past_deadline, persist, log, **kw):
            n_train = {"n": 0}
            if dl_after:
                real_train = ops.train

                def train(*a):
                    out = real_train(*a)
                    n_train["n"] += 1
                    return out

                ops.train = train
                past_deadline = lambda: n_train["n"] >= int(dl_after)  # noqa: E731

            def persist_then_maybe_die(p):
                persist(p)
                if crash and len(p["steps"]) == int(crash):
                    print(f"DRY RUN: simulated preemption after step {crash}", flush=True)
                    os._exit(137)
            return real_prog(order, prog, gate, ops, max_layers, past_deadline, persist_then_maybe_die, log, **kw)

        pipeline.run_progressive = run_progressive


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.path.isdir("/opt/ssdm/m1"):
        raise SystemExit("dry-run pipeline refused inside the job image")
    cfg_path, out = argv[argv.index("--config") + 1], argv[argv.index("--out") + 1]
    small = shrink(json.load(open(cfg_path, encoding="utf-8")))
    sp = os.path.join(os.path.dirname(os.path.abspath(out)), "dryrun_config.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(small, f, indent=1)
    from convert import pipeline

    patch(pipeline)
    sys.argv = ["convert.pipeline", "--config", sp, "--out", out]
    pipeline.main()


if __name__ == "__main__":
    main()
