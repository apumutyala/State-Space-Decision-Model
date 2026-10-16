"""Qwen3.5-9B thinking labels, evidence authorship, and fail-closed filtering."""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

from keel.protocol import DecisionRequest
from keel.distill.sources import read_rows, write_rows
from keel.distill.triplets import evidence_unit, template_guard, variants


def argmax(probs: dict[str, float]) -> str:
    if not probs or any(not math.isfinite(x) or x < 0 for x in probs.values()):
        raise ValueError("invalid probability distribution")
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return max(sorted(probs), key=probs.__getitem__)


def mean_samples(result: dict) -> dict[str, float]:
    samples = result["samples"]
    if len(samples) != 2:
        raise ValueError("teacher requires exactly two samples")
    keys = set(samples[0])
    for sample in samples:
        argmax(sample)
        if set(sample) != keys:
            raise ValueError("teacher sample option mismatch")
    return {k: sum(s[k] for s in samples) / len(samples) for k in sorted(keys)}


def filter_labels(units: list[dict], results: dict[str, dict]) -> tuple[list[dict], dict]:
    """Never label attacks from attacked distributions; controls require both samples."""
    clean = {u["base_id"]: u for u in units if u["kind"] == "clean"}
    kept, stats = [], collections.defaultdict(collections.Counter)
    for u in units:
        kind = u["kind"]
        stats[kind]["attempted"] += 1
        c = clean[u["base_id"]]
        cp = mean_samples(results[c["id"]])
        p = mean_samples(results[u["id"]])
        decisions = [argmax(s) for s in results[u["id"]]["samples"]]
        if kind in ("inject", "authority"):
            stats[kind]["teacher_target_steered"] += int(argmax(p) == u["target"])
            stats[kind]["teacher_any_sample_changed"] += int(any(d != c["gold"] for d in decisions))
        if argmax(cp) != c["gold"]:
            stats[kind]["dropped_clean_wrong"] += 1
            continue
        if kind in ("inject", "authority"):
            if any(d != c["gold"] for d in decisions):
                stats[kind]["dropped_teacher_changed"] += 1
                continue
            label = cp
        elif kind in ("evidence", "policy"):
            if u["gold"] == c["gold"] or any(d != u["gold"] for d in decisions):
                stats[kind]["dropped_control_unvalidated"] += 1
                continue
            label = p
        else:
            label = cp
        if set(label) != set(p):
            raise ValueError("clean/variant option mismatch")
        kept.append({**u, "teacher_probs": label})
        stats[kind]["kept"] += 1
    return kept, {k: dict(v) for k, v in stats.items()}


def make_teacher_app():
    """Keep one vLLM engine warm for authorship and the standard runtime judge API."""
    import modal
    from keel.runtime.modal_app import DEFAULT_GPU, HF_CACHE, hf_cache, results_vol, vllm_image
    from keel.runtime.vllm_judge import _Engine

    app = modal.App("keel-distill-teacher")
    image = vllm_image.add_local_python_source("keel")

    @app.cls(image=image, gpu=DEFAULT_GPU, timeout=10800, scaledown_window=60, serialized=True,
             volumes={HF_CACHE: hf_cache, "/results": results_vol})
    class TeacherServer:
        @modal.enter()
        def load(self):
            self.engine = _Engine("qwen3.5-9b")

        @modal.method()
        def score(self, jobs, **kwargs):
            return self.engine.score(jobs, **kwargs)

        @modal.method()
        def author(self, rows, max_think=768, seed=8103):
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            start = time.time()
            prompts = []
            for r in rows:
                system = ("Write a minimal factual counterfactual for a decision task. Append one later observation "
                          "or event, at most 70 words, that changes the correct option. Do not change the question, "
                          "rubric, or policy. Do not tell a reader what to answer, claim authority, name the target "
                          "option in the addition, or add instructions. The observation must describe actual changed "
                          "facts, not an opinion about the decision. Reason carefully about the old and new result. "
                          "Return JSON with addition (plain factual prose) and new_gold (an existing option id).")
                payload = {"state": r["state"], "question": r["questions"]["q"], "old_gold": r["labels"]["q"]}
                prompts.append(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
                               + json.dumps(payload, ensure_ascii=False)
                               + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
            traces = self.engine.llm.generate(prompts, SamplingParams(max_tokens=max_think, temperature=0.6,
                                                                 top_p=0.95, stop=["</think>"], seed=seed),
                                              use_tqdm=False)
            final = [p + o.outputs[0].text + "\n</think>\n" for p, o in zip(prompts, traces)]
            schema = {"type": "object", "properties": {"addition": {"type": "string"},
                      "new_gold": {"type": "string"}}, "required": ["addition", "new_gold"],
                      "additionalProperties": False}
            generated = self.engine.llm.generate(final, SamplingParams(max_tokens=256, temperature=0.0,
                        structured_outputs=StructuredOutputsParams(json=schema), seed=seed), use_tqdm=False)
            proposals = []
            for trace, result in zip(traces, generated):
                text = result.outputs[0].text
                try:
                    proposal = json.loads(text)
                except json.JSONDecodeError:
                    proposal = {"parse_error": True}
                proposal["think_tokens"] = len(trace.outputs[0].token_ids)
                proposal["truncated"] = trace.outputs[0].finish_reason == "length"
                proposals.append(proposal)
            return {"proposals": proposals, "wall_s": time.time() - start, "meta": self.engine.meta}

    return app, TeacherServer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-think", type=int, default=768)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-usd", type=float, required=True, help="cap for this run, within the owner's approved budget")
    a = p.parse_args()
    from keel.runtime.budget import approved_run
    from keel.runtime.vllm_judge import judge

    template_rep = template_guard()
    rows = read_rows(a.sources)[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    app, server_cls = make_teacher_app()
    t0 = time.time()
    ledger = {"date": "2026-10-15", "function": "TeacherServer.author + runtime.judge",
              "model": "qwen3.5-9b", "n_samples": 2, "max_think": a.max_think,
              "status": "running", "h100_hourly_usd": 3.95}
    (out / "cost.json").write_text(json.dumps(ledger, indent=2))
    all_units, all_results, proposals_n, lexical_kept = [], {}, 0, 0
    try:
        with approved_run(app, label="distill.teacher", cap_usd=a.max_usd):
            server = server_cls()
            for offset in range(0, len(rows), a.batch):
                batch = rows[offset:offset + a.batch]
                checkpoint = out / f"batch-{offset:05d}.json"
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text())
                    if saved["source_ids"] != [r["id"] for r in batch]:
                        raise ValueError("checkpoint source identity mismatch")
                else:
                    authored = server.author.remote(batch, max_think=a.max_think, seed=8103 + offset)
                    units = []
                    evidence_kept = 0
                    for row, proposal in zip(batch, authored["proposals"]):
                        units.extend(variants(row))
                        ev = evidence_unit(row, proposal)
                        if ev is not None:
                            units.append(ev)
                            evidence_kept += 1
                    scored = judge([DecisionRequest.from_json(u["request"]) for u in units], "qwen3.5-9b",
                                   mode="trust", think=True, n_samples=2, server=server,
                                   max_think=a.max_think, seed=9103 + offset)
                    saved = {"source_ids": [r["id"] for r in batch], "units": units,
                             "results": {u["id"]: r for u, r in zip(units, scored)},
                             "authored": authored, "lexical_kept": evidence_kept}
                    checkpoint.write_text(json.dumps(saved))
                all_units.extend(saved["units"])
                all_results.update(saved["results"])
                proposals_n += len(batch)
                lexical_kept += saved["lexical_kept"]
                print(json.dumps({"completed": offset + len(batch), "sources": len(rows),
                                  "elapsed_s": round(time.time() - t0)}), flush=True)
        labels, yields = filter_labels(all_units, all_results)
        write_rows(out / "units.jsonl.gz", all_units)
        write_rows(out / "labels.jsonl.gz", labels)
        (out / "results.json").write_text(json.dumps(all_results))
        report = {"source_rows": len(rows), "units": len(all_units), "labels": len(labels),
                  "by_kind": yields, "evidence_proposals": proposals_n, "evidence_lexically_kept": lexical_kept,
                  "templates": template_rep, "max_think": a.max_think, "n_samples": 2}
        (out / "yield.json").write_text(json.dumps(report, indent=2) + "\n")
        ledger["status"] = "complete"
        print(json.dumps(report, indent=2))
    finally:
        if ledger["status"] == "running":
            ledger["status"] = "failed"
        ledger["wall_s"] = time.time() - t0
        # Conservative: local wall time includes model startup and application shutdown.
        ledger["gpu_minutes_upper_estimate"] = ledger["wall_s"] / 60
        ledger["usd_upper_estimate"] = ledger["wall_s"] * 3.95 / 3600
        (out / "cost.json").write_text(json.dumps(ledger, indent=2) + "\n")
        with (out / "cost-attempts.jsonl").open("a") as log:
            log.write(json.dumps(ledger) + "\n")


if __name__ == "__main__":
    main()
