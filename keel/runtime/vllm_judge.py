"""The Keel judge runtime: label-token scoring with vLLM on Modal, one-pass or think-first.

Prompts are rendered locally (``keel.protocol.render``) and shipped as plain strings, so the GPU container needs no
Keel code beyond this module. Usage from Python::

    from keel.runtime.vllm_judge import APP, score, judge
    with APP.run():
        out = judge(requests, "qwen3.5-4b", mode="trust", think=False)

``score`` reads every label's log-probability at the answer slot (top-``max_logprobs`` raw log-probs; a missing
label is reported, never silently imputed). In think mode it samples ``n_samples`` reasoning traces per prompt, closes
each one (budget-forced when cut) and reads the same slot.
"""
from __future__ import annotations

import time
from typing import Any

import modal

from .modal_app import DEFAULT_GPU, HF_CACHE, MODELS, hf_cache, vllm_image

APP = modal.App("keel-judge")
BUDGET_TAIL = "\n\nConsidering the limited time, I have to give the answer now.\n"


@APP.function(image=vllm_image.add_local_python_source("keel"), gpu=DEFAULT_GPU, timeout=3 * 3600,
              volumes={HF_CACHE: hf_cache})
def score(model_id: str, revision: str, jobs: list[dict[str, Any]], think: bool = False, n_samples: int = 1,
          max_think: int = 3072, temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, seed: int = 0,
          keep_reasoning: bool = False, max_model_len: int = 16384) -> dict[str, Any]:
    """jobs: [{"id", "prompt", "labels"}]. Returns per job the label log-probs of each sample."""
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model_id, revision=revision, max_model_len=max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=0.88, seed=seed, max_logprobs=256)
    tok = llm.get_tokenizer()
    load_s = time.time() - t0

    def label_ids(prompt: str, labels: list[str]) -> list[int]:
        base = tok.encode(prompt, add_special_tokens=False)
        out = []
        for lab in labels:
            full = tok.encode(prompt + " " + lab, add_special_tokens=False)
            if len(full) != len(base) + 1 or full[: len(base)] != base:
                raise ValueError(f"label {lab!r} is not one token at the answer slot")
            out.append(full[-1])
        return out

    # every label set shares one probe: labels are single tokens after "Answer:" in this chat format
    probe = jobs[0]["prompt"] if not think else jobs[0]["prompt"] + "x\n</think>\n\nAnswer:"
    all_labels = sorted({lab for j in jobs for lab in j["labels"]})
    lab_id = dict(zip(all_labels, label_ids(probe, all_labels)))
    read_sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=256)

    def read(prompts: list[str], labelss: list[list[str]]) -> list[tuple[list[float | None], int]]:
        res = []
        for o, labels in zip(llm.generate(prompts, read_sp, use_tqdm=False), labelss):
            got = {tid: lp.logprob for tid, lp in o.outputs[0].logprobs[0].items()}
            vals = [got.get(lab_id[lab]) for lab in labels]
            res.append((vals, sum(v is None for v in vals)))
        return res

    results: list[dict[str, Any]] = [{"id": j["id"], "samples": []} for j in jobs]
    t1 = time.time()
    if not think:
        for r, (vals, miss) in zip(results, read([j["prompt"] for j in jobs], [j["labels"] for j in jobs])):
            r["samples"].append({"logprobs": vals, "missing": miss, "think_tokens": 0, "truncated": False})
    else:
        gen_sp = SamplingParams(n=n_samples, max_tokens=max_think, temperature=temperature, top_p=top_p, top_k=top_k,
                                stop=["</think>"], seed=seed)
        gens = llm.generate([j["prompt"] for j in jobs], gen_sp, use_tqdm=False)
        finals, owners, metas = [], [], []
        for k, (j, g) in enumerate(zip(jobs, gens)):
            for o in g.outputs:
                tr = o.finish_reason == "length"
                finals.append(j["prompt"] + o.text.rstrip() + (BUDGET_TAIL if tr else "\n") + "</think>\n\nAnswer:")
                owners.append(k)
                metas.append((len(o.token_ids), tr, o.text if keep_reasoning else o.text[-400:]))
        for k, (vals, miss), (ntok, tr, text) in zip(owners, read(finals, [jobs[i]["labels"] for i in owners]), metas):
            results[k]["samples"].append({"logprobs": vals, "missing": miss, "think_tokens": ntok, "truncated": tr,
                                          "reasoning": text})
    return {"model": model_id, "revision": revision, "vllm": vllm.__version__, "load_s": load_s,
            "wall_s": time.time() - t1, "think": think, "n_samples": n_samples, "results": results}


def judge(requests: list, model_key: str, *, mode: str = "trust", think: bool = False, canonical: bool = True,
          qid: str = "q", n_samples: int = 1, temperature_out: float = 1.0, **kw: Any) -> list[dict[str, Any]]:
    """Render ``requests`` locally, score them remotely, return per request the option probabilities (mean over
    samples), the per-sample probabilities and the run metadata. Call inside ``with APP.run():``."""
    from ..protocol.render import probs_from_logprobs, render

    model_id, revision = MODELS[model_key]
    rendered = [render(r, qid, mode=mode, think=think, canonical=canonical) for r in requests]
    jobs = [{"id": r.request_id, "prompt": x.prompt, "labels": list(x.labels)} for r, x in zip(requests, rendered)]
    out = score.remote(model_id, revision, jobs, think=think, n_samples=n_samples, **kw)
    res = []
    for r, x, o in zip(requests, rendered, out["results"]):
        per = []
        for s in o["samples"]:
            if s["missing"]:
                raise RuntimeError(f"{r.request_id}: {s['missing']} label(s) outside the returned log-probs")
            per.append(probs_from_logprobs(x, s["logprobs"], temperature_out))
        mean = {k: sum(p[k] for p in per) / len(per) for k in per[0]}
        res.append({"request_id": r.request_id, "probs": mean, "samples": per, "option_ids": list(x.option_ids),
                    "think_tokens": [s["think_tokens"] for s in o["samples"]],
                    "reasoning": [s.get("reasoning") for s in o["samples"]]})
    meta = {k: v for k, v in out.items() if k != "results"}
    for x in res:
        x["meta"] = meta
    return res
