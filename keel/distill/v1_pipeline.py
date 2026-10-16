"""Resumable author / cross-family verify / soft-label execution with a shared spend ceiling."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from keel.distill.authoring import author_schema, materialize, plan, prompt, summarize
from keel.distill.sources import digest, guard_draw, read_rows, write_rows
from keel.distill.verification import VERIFY_PROMPT, blind_payload, capability_labels, guard_authored

OSS_ID = "openai/gpt-oss-20b"
OSS_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
RATE = 3.95 / 3600


def make_app(model: str):
    import modal
    from keel.runtime.modal_app import HF_CACHE, hf_cache, results_vol, vllm_image
    from keel.runtime.vllm_judge import _Engine

    app = modal.App("keel-distill-v1-" + model)

    @app.cls(image=vllm_image.add_local_python_source("keel"), gpu="H100", timeout=7200,
             scaledown_window=30, serialized=True, memory=32768, cpu=8,
             volumes={HF_CACHE: hf_cache, "/results": results_vol})
    class Generator:
        @modal.enter()
        def load(self):
            if model == "qwen":
                self.engine = _Engine("qwen3.5-9b", max_model_len=8192)
                self.llm, self.tok, self.meta = self.engine.llm, self.engine.tok, self.engine.meta
            elif model == "oss":
                from vllm import LLM, __version__
                t0 = time.time()
                self.llm = LLM(model=OSS_ID, revision=OSS_REVISION, max_model_len=8192,
                               gpu_memory_utilization=0.88, enable_prefix_caching=True,
                               max_num_seqs=128, seed=20261015)
                self.tok = self.llm.get_tokenizer()
                self.meta = {"model": OSS_ID, "revision": OSS_REVISION, "vllm": __version__,
                             "load_s": time.time() - t0}
            else:
                raise ValueError("unknown generator model")

        @modal.method()
        def generate(self, messages: list[dict], *, max_think=512, max_output=2048, seed=20261015):
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            start = time.time()
            prompts = []
            for m in messages:
                chat = [{"role": "system", "content": m["system"]}, {"role": "user", "content": m["user"]}]
                if model == "qwen":
                    p = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                else:
                    p = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
                    p += "<|channel|>analysis<|message|>"
                prompts.append(p)
            stop = ["</think>"] if model == "qwen" else ["<|end|>"]
            traces = self.llm.generate(prompts, SamplingParams(max_tokens=max_think, temperature=0.7,
                                       top_p=0.95, stop=stop, skip_special_tokens=False, seed=seed), use_tqdm=False)
            finals = []
            for p, t in zip(prompts, traces):
                text = t.outputs[0].text
                if model == "qwen":
                    finals.append(p + text + "\n</think>\n")
                else:
                    # Harmony analysis ends before a separate final-channel message.
                    text = text.split("<|start|>assistant<|channel|>final")[0]
                    finals.append(p + text + "<|end|><|start|>assistant<|channel|>final<|message|>")
            sampling = [SamplingParams(max_tokens=max_output, temperature=0.7, top_p=0.95, seed=seed + 1,
                        structured_outputs=(StructuredOutputsParams(json=m["schema"]) if "schema" in m
                                            else StructuredOutputsParams(json_object=True))) for m in messages]
            outputs = self.llm.generate(finals, sampling, use_tqdm=False)
            results = []
            for trace, output in zip(traces, outputs):
                o, t = output.outputs[0], trace.outputs[0]
                try:
                    parsed = json.loads(o.text)
                except json.JSONDecodeError:
                    parsed = None
                results.append({"value": parsed, "text": o.text, "reasoning": t.text,
                                "think_tokens": len(t.token_ids), "output_tokens": len(o.token_ids),
                                "think_truncated": t.finish_reason == "length", "output_truncated": o.finish_reason == "length"})
            return {"results": results, "wall_s": time.time() - start, "meta": self.meta}

        @modal.method()
        def score(self, jobs, **kwargs):
            if model != "qwen":
                raise ValueError("soft labels require Qwen")
            return self.engine.score(jobs, **kwargs)

    return app, Generator


def spending(out: Path) -> float:
    path = out / "spend.jsonl"
    return sum(r["usd_upper_estimate"] for r in read_rows(path)) if path.exists() else 0.0


def run_phase(args):
    import modal
    from keel.paths import private_root
    from keel.protocol import DecisionRequest
    from keel.distill.triplets import request_of
    from keel.runtime.vllm_judge import judge

    out = Path(args.out) if args.out else private_root() / "distill_v1"
    out.mkdir(parents=True, exist_ok=True)
    if spending(out) + args.reserve >= args.cap:
        raise RuntimeError("reserved spend leaves no GPU budget")
    phase = args.phase
    model = "oss" if phase in {"smoke-oss", "verify"} else "qwen"
    app, cls = make_app(model)
    existing = []
    if phase in {"author", "repair"}:
        jobs = plan(args.items)
        if phase == "repair":
            original = out / "original-authored.jsonl.gz"
            if not original.exists():
                write_rows(original, read_rows(out / "authored.jsonl.gz"))
            existing = read_rows(original)
            done = {r["distill"]["group"] for r in existing}
            jobs = [j for j in jobs if j["id"] not in done]
    elif phase in {"verify", "label"}:
        jobs = read_rows(out / "authored.jsonl.gz")
    else:
        jobs = [{"id": "smoke", "system": "Reason, then return JSON with answer_text and ambiguous.",
                 "user": "A cabinet has 17 blue binders. Eight are removed and three returned. How many remain?"}]
    costs = {"date": datetime.now(timezone.utc).date().isoformat(), "function": phase,
             "model": model, "status": "running"}
    start = time.time()
    results = {}
    authored, failures = list(existing), collections.Counter()
    try:
        with modal.enable_output(), app.run():
            server = cls()
            for offset in range(0, len(jobs), args.batch):
                batch = jobs[offset:offset + args.batch]
                saved_path = out / f"{phase}-{offset:05d}.json"
                identity = digest(batch)
                if saved_path.exists():
                    saved = json.loads(saved_path.read_text())
                    if saved["input_sha256"] != identity:
                        raise ValueError("cached batch input mismatch")
                else:
                    if spending(out) + (time.time() - start) * RATE + args.reserve >= args.cap:
                        raise RuntimeError("spend ceiling reached before next batch")
                    if phase in {"author", "repair"}:
                        messages = [{"system": prompt(j), "user": "Solve BOTH cases fully before the final JSON. "
                                     "The final object is the finished item, not a plan or revision. "
                                     "The state must contain the literal {{FACT}} slot for a noul pair.",
                                     "schema": author_schema(j["type"])} for j in batch]
                        got = server.generate.remote(messages, max_think=1536, max_output=2800, seed=20000 + offset)
                    elif phase == "verify":
                        messages = [{"system": VERIFY_PROMPT, "user": json.dumps(blind_payload(r), ensure_ascii=False)} for r in batch]
                        got = server.generate.remote(messages, max_think=768, max_output=640, seed=30000 + offset)
                    elif phase == "label":
                        got = judge([request_of(r) for r in batch], "qwen3.5-9b", server=server,
                                    mode="trust", think=True, n_samples=2, max_think=1024, seed=40000 + offset)
                    else:
                        got = server.generate.remote(batch, max_think=256, max_output=128)
                    saved = {"input_sha256": identity, "result": got}
                    saved_path.write_text(json.dumps(saved))
                got = saved["result"]
                if phase in {"author", "repair"}:
                    for job, result in zip(batch, got["results"]):
                        try:
                            if result["output_truncated"]:
                                raise ValueError("truncated JSON")
                            authored.extend(materialize(job, result["value"], got["meta"]))
                        except (ValueError, TypeError, KeyError) as e:
                            failures[str(e)] += 1
                elif phase == "verify":
                    results.update({r["id"]: (v["value"] or {}) if not v["output_truncated"] else {}
                                    for r, v in zip(batch, got["results"])})
                elif phase == "label":
                    results.update({r["id"]: v for r, v in zip(batch, got)})
                else:
                    if got["results"][0]["value"].get("answer_text") not in {"12", 12}:
                        raise RuntimeError("cross-family smoke arithmetic failed")
                    (out / "oss-smoke.json").write_text(json.dumps(got, indent=2))
                print(json.dumps({"phase": phase, "completed": offset + len(batch), "total": len(jobs),
                                  "wall_s": round(time.time() - start)}), flush=True)
        if phase in {"author", "repair"}:
            kept, guard_counts = guard_authored(authored)
            write_rows(out / "authored-unguarded.jsonl.gz", authored)
            write_rows(out / "authored.jsonl.gz", kept)
            guards = guard_draw(out / "authored.jsonl.gz")
            report = {"planned_items": args.items, "parsed": summarize(authored), "guarded": summarize(kept),
                      "schema_rejections": dict(failures), "guard_rejections": guard_counts, "guards": guards}
            (out / f"{phase}-yield.json").write_text(json.dumps(report, indent=2))
        elif phase in {"verify", "label"}:
            (out / f"{phase}-results.json").write_text(json.dumps(results))
            if phase == "label":
                verdicts = json.loads((out / "verify-results.json").read_text())
                labels, report = capability_labels(jobs, verdicts, results)
                write_rows(out / "capability-labels.jsonl.gz", labels)
                (out / "label-yield.json").write_text(json.dumps(report, indent=2))
        costs["status"] = "complete"
    finally:
        costs["status"] = "failed" if costs["status"] == "running" else costs["status"]
        costs["wall_s"] = time.time() - start
        costs["gpu_minutes_upper_estimate"] = costs["wall_s"] / 60
        costs["usd_upper_estimate"] = costs["wall_s"] * RATE
        with (out / "spend.jsonl").open("a") as f:
            f.write(json.dumps(costs) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["smoke-oss", "author", "repair", "verify", "label"])
    p.add_argument("--out")
    p.add_argument("--items", type=int, default=7200)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--cap", type=float, default=35)
    p.add_argument("--reserve", type=float, default=6)
    run_phase(p.parse_args())


if __name__ == "__main__":
    main()
