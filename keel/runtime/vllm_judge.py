"""The Keel judge runtime: label-token scoring with vLLM on Modal, one-pass or think-first.

Prompts are rendered locally (``keel.protocol.render``) and shipped as plain strings. Two entry points share one
engine:

  ``score``        a one-shot Modal function (loads the model, scores one batch, exits): batch evaluations.
  ``JudgeServer``  a Modal class that keeps the model loaded across calls: adaptive attack search, interactive use.

Usage::

    from keel.runtime.vllm_judge import APP, judge, JudgeServer
    with APP.run():
        out = judge(requests, "qwen3.5-4b", mode="trust", think=False)            # one-shot
        srv = JudgeServer(model="qwen3.5-4b")
        out = judge(requests, "qwen3.5-4b", server=srv)                            # warm server

Models are keys of ``MODELS`` (pinned revisions) or paths on the ``keel-results`` volume (``/results/...``, e.g. a
merged student checkpoint). ``score`` reads every label's log-probability at the answer slot from the top
``max_logprobs`` raw log-probs; a missing label is reported, never silently imputed. In think mode it samples
``n_samples`` reasoning traces per prompt, closes each one (budget-forced when cut) and reads the same slot.
"""
# no `from __future__ import annotations`: modal.parameter() needs real (not string) annotations

import time
from typing import Any

import modal

from .modal_app import DEFAULT_GPU, HF_CACHE, MODELS, hf_cache, results_vol, vllm_image

APP = modal.App("keel-judge")
RESULTS = "/results"
BUDGET_TAIL = "\n\nConsidering the limited time, I have to give the answer now.\n"
_IMAGE = vllm_image.add_local_python_source("keel")
_VOLUMES = {HF_CACHE: hf_cache, RESULTS: results_vol}


def resolve(model: str) -> tuple[str, str | None]:
    """(model id or path, pinned revision or None) for a ``MODELS`` key or a ``/results/...`` checkpoint path."""
    if model in MODELS:
        return MODELS[model]
    if model.startswith(RESULTS + "/"):
        return model, None
    raise KeyError(f"unknown model {model!r}: use a MODELS key or a {RESULTS}/... checkpoint path")


class _Engine:
    def __init__(self, model: str, max_model_len: int = 16384, seed: int = 0) -> None:
        import vllm
        from vllm import LLM

        t0 = time.time()
        model_id, revision = resolve(model)
        if model_id.startswith(RESULTS + "/"):
            results_vol.reload()
        self.llm = LLM(model=model_id, revision=revision, max_model_len=max_model_len, enable_prefix_caching=True,
                       gpu_memory_utilization=0.88, seed=seed, max_logprobs=256)
        self.tok = self.llm.get_tokenizer()
        self.meta = {"model": model_id, "revision": revision, "vllm": vllm.__version__, "load_s": time.time() - t0}
        self._lab_id: dict[str, int] = {}

    def _label_ids(self, probe: str, labels: list[str]) -> None:
        todo = [lab for lab in labels if lab not in self._lab_id]
        if not todo:
            return
        base = self.tok.encode(probe, add_special_tokens=False)
        for lab in todo:
            full = self.tok.encode(probe + " " + lab, add_special_tokens=False)
            if len(full) != len(base) + 1 or full[: len(base)] != base:
                raise ValueError(f"label {lab!r} is not one token at the answer slot")
            self._lab_id[lab] = full[-1]

    def _read(self, prompts: list[str], labelss: list[list[str]]) -> list[tuple[list[float | None], int]]:
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=256)
        res = []
        for o, labels in zip(self.llm.generate(prompts, sp, use_tqdm=False), labelss):
            got = {tid: lp.logprob for tid, lp in o.outputs[0].logprobs[0].items()}
            vals = [got.get(self._lab_id[lab]) for lab in labels]
            res.append((vals, sum(v is None for v in vals)))
        return res

    def score(self, jobs: list[dict[str, Any]], think: bool = False, n_samples: int = 1, max_think: int = 3072,
              temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, seed: int = 0,
              keep_reasoning: bool = False) -> dict[str, Any]:
        from vllm import SamplingParams

        if not jobs:
            return {**self.meta, "wall_s": 0.0, "think": think, "n_samples": n_samples, "results": []}
        probe = jobs[0]["prompt"] + ("x\n</think>\n\nAnswer:" if think else "")
        self._label_ids(probe, sorted({lab for j in jobs for lab in j["labels"]}))
        results: list[dict[str, Any]] = [{"id": j["id"], "samples": []} for j in jobs]
        t1 = time.time()
        if not think:
            for r, (vals, miss) in zip(results, self._read([j["prompt"] for j in jobs], [j["labels"] for j in jobs])):
                r["samples"].append({"logprobs": vals, "missing": miss, "think_tokens": 0, "truncated": False})
        else:
            gen_sp = SamplingParams(n=n_samples, max_tokens=max_think, temperature=temperature, top_p=top_p,
                                    top_k=top_k, stop=["</think>"], seed=seed)
            finals, owners, metas = [], [], []
            for k, (j, g) in enumerate(zip(jobs, self.llm.generate([j["prompt"] for j in jobs], gen_sp,
                                                                  use_tqdm=False))):
                for o in g.outputs:
                    tr = o.finish_reason == "length"
                    finals.append(j["prompt"] + o.text.rstrip() + (BUDGET_TAIL if tr else "\n") + "</think>\n\nAnswer:")
                    owners.append(k)
                    metas.append((len(o.token_ids), tr, o.text if keep_reasoning else o.text[-400:]))
            read = self._read(finals, [jobs[i]["labels"] for i in owners])
            for k, (vals, miss), (ntok, tr, text) in zip(owners, read, metas):
                results[k]["samples"].append({"logprobs": vals, "missing": miss, "think_tokens": ntok,
                                              "truncated": tr, "reasoning": text})
        return {**self.meta, "wall_s": time.time() - t1, "think": think, "n_samples": n_samples, "results": results}


@APP.function(image=_IMAGE, gpu=DEFAULT_GPU, timeout=3 * 3600, volumes=_VOLUMES)
def score(model: str, jobs: list[dict[str, Any]], max_model_len: int = 16384, **kw: Any) -> dict[str, Any]:
    """One-shot: load ``model``, score ``jobs`` ([{"id", "prompt", "labels"}]), return per job the samples."""
    return _Engine(model, max_model_len, kw.get("seed", 0)).score(jobs, **kw)


@APP.cls(image=_IMAGE, gpu=DEFAULT_GPU, timeout=3 * 3600, volumes=_VOLUMES, scaledown_window=600)
class JudgeServer:
    model: str = modal.parameter()
    max_model_len: int = modal.parameter(default=16384)

    @modal.enter()
    def load(self) -> None:
        self.engine = _Engine(self.model, self.max_model_len)

    @modal.method()
    def score(self, jobs: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        return self.engine.score(jobs, **kw)


def judge(requests: list, model: str, *, mode: str = "trust", think: bool = False, canonical: bool = True,
          qid: str = "q", n_samples: int = 1, temperature_out: float = 1.0, server: Any = None,
          **kw: Any) -> list[dict[str, Any]]:
    """Render ``requests`` locally, score them remotely (one-shot, or on ``server``), return per request the option
    probabilities (mean over samples), the per-sample probabilities and the run metadata. Call inside
    ``with APP.run():``."""
    from ..protocol.render import probs_from_logprobs, render

    rendered = [render(r, qid, mode=mode, think=think, canonical=canonical) for r in requests]
    jobs = [{"id": r.request_id, "prompt": x.prompt, "labels": list(x.labels)} for r, x in zip(requests, rendered)]
    kw = {"think": think, "n_samples": n_samples, **kw}
    out = server.score.remote(jobs, **kw) if server is not None else score.remote(model, jobs, **kw)
    res = []
    meta = {k: v for k, v in out.items() if k != "results"}
    for r, x, o in zip(requests, rendered, out["results"]):
        per = []
        for s in o["samples"]:
            if s["missing"]:
                raise RuntimeError(f"{r.request_id}: {s['missing']} label(s) outside the returned log-probs")
            per.append(probs_from_logprobs(x, s["logprobs"], temperature_out))
        mean = {k: sum(p[k] for p in per) / len(per) for k in per[0]}
        res.append({"request_id": r.request_id, "probs": mean, "samples": per, "option_ids": list(x.option_ids),
                    "think_tokens": [s["think_tokens"] for s in o["samples"]],
                    "reasoning": [s.get("reasoning") for s in o["samples"]], "meta": meta})
    return res
