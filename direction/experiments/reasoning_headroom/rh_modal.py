"""Reasoning-headroom probe (chief-of-staff diagnostic, 2026-10-14).

Question: how much JevBench public-231 Intelligence does a stock Qwen3.5 hybrid-SSM backbone gain when it may
think before the same H2O-shim answer readout, and at what token/latency cost?

Arms per model (same prompt content, same label readout, same shim post-processing applied offline):
  nothink  H2O-Lightning shim prompt, thinking off, label-token log-probs at the "Answer:" slot (one forward pass)
  think    same system + user turn, thinking on; generate until </think> (budget-forced at MAX_THINK), then
           "</think>\n\nAnswer:" and the same label-token read

EVAL-ONLY: items are the pinned public JevBench items; nothing is trained, tuned or selected on them.
"""
from __future__ import annotations

import json
import time

import modal

APP = modal.App("ssdm-reasoning-headroom")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("ssdm-hf-cache", create_if_missing=True)

SYSTEM = ("You are a decision engine. You read a record and answer one question about it by choosing exactly one "
          "option. Reply with a single letter.")
LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
MAX_THINK = 3072
BUDGET_TAIL = "\n\nConsidering the limited time, I have to give the answer now.\n"


def render_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def options_of(q):
    typ = q.get("type", "choice")
    crit = q.get("criteria")
    if typ == "noul":
        crit = crit or {}
        t = crit.get("true", crit.get("yes"))
        f = crit.get("false", crit.get("no"))
        return ["true", "false"], [t if t not in (None, "") else "the statement holds",
                                   f if f not in (None, "") else "it does not"]
    if typ == "score":
        return [str(i) for i in range(len(crit))], [c if isinstance(c, str) else json.dumps(c) for c in crit]
    names = list(crit)
    return names, [crit[n] if isinstance(crit[n], str) else json.dumps(crit[n]) for n in names]


def user_turn(item):
    q = item["question"]
    names, descs = options_of(q)
    labels = LABELS[: len(names)]
    instr = q.get("instructions", "")
    instr = instr if isinstance(instr, str) else json.dumps(instr, indent=1)
    body = "\n".join(f"{l}) {n}: {d}" for l, n, d in zip(labels, names, descs))
    return f"record: {render_state(item['state'])}\nquestion: {instr.lstrip()}\noptions:\n{body}", names, labels


def chat(user, think):
    head = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user.strip()}<|im_end|>\n<|im_start|>assistant\n"
    return head + ("<think>\n" if think else "<think>\n\n</think>\n\nAnswer:")


@APP.function(image=image, gpu="H100", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def run(model_id: str, revision: str, items: list[dict], timed_ids: list[str]) -> dict:
    import vllm
    from vllm import LLM, SamplingParams

    t_load = time.time()
    llm = LLM(model=model_id, revision=revision, max_model_len=16384, enable_prefix_caching=True,
              gpu_memory_utilization=0.85, seed=0, max_logprobs=256)
    tok = llm.get_tokenizer()
    load_s = time.time() - t_load

    # label token ids at the answer slot (the shim's check_backend probe)
    probe = chat("record: {\"x\":1}\nquestion: Which?\noptions:\n" + "\n".join(f"{l}) o: d" for l in LABELS), False)
    base = tok.encode(probe, add_special_tokens=False)
    lab_id = {}
    for l in LABELS:
        full = tok.encode(probe + " " + l, add_special_tokens=False)
        assert len(full) == len(base) + 1 and full[: len(base)] == base, l
        lab_id[l] = full[-1]

    read_sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=256)

    def read(prompts, labelss):
        outs = llm.generate(prompts, read_sp, use_tqdm=False)
        res = []
        for o, labels in zip(outs, labelss):
            top = o.outputs[0].logprobs[0]
            got = {tid: lp.logprob for tid, lp in top.items()}
            floor = min(got.values()) - 1.0
            miss = sum(lab_id[l] not in got for l in labels)
            res.append(([got.get(lab_id[l], floor) for l in labels], miss, len(o.prompt_token_ids)))
        return res

    rows = [(it, *user_turn(it)) for it in items]
    out = {"model": model_id, "revision": revision, "vllm": vllm.__version__, "load_s": load_s, "arms": {}}

    # ---- nothink: one batched read
    t0 = time.time()
    r0 = read([chat(u, False) for _, u, _, _ in rows], [labels for _, _, _, labels in rows])
    out["arms"]["nothink"] = {"wall_s": time.time() - t0, "items": [
        {"id": it["id"], "type": it["question"]["type"], "names": names, "logprobs": lp, "missing": miss,
         "input_tokens": nin, "output_tokens": 0} for (it, _, names, _), (lp, miss, nin) in zip(rows, r0)]}

    # ---- think: batched generation, budget-forced close, then the same read
    gen_sp = SamplingParams(max_tokens=MAX_THINK, temperature=0.6, top_p=0.95, top_k=20, stop=["</think>"], seed=0)
    t0 = time.time()
    p_think = [chat(u, True) for _, u, _, _ in rows]
    gens = llm.generate(p_think, gen_sp, use_tqdm=False)
    finals, meta = [], []
    for p, g in zip(p_think, gens):
        txt = g.outputs[0].text
        truncated = g.outputs[0].finish_reason == "length"
        body = txt.rstrip() + (BUDGET_TAIL if truncated else "\n")
        finals.append(p + body + "</think>\n\nAnswer:")
        meta.append((len(g.outputs[0].token_ids), truncated, txt[-400:]))
    r1 = read(finals, [labels for _, _, _, labels in rows])
    out["arms"]["think"] = {"wall_s": time.time() - t0, "items": [
        {"id": it["id"], "type": it["question"]["type"], "names": names, "logprobs": lp, "missing": miss,
         "input_tokens": nin - ntk, "output_tokens": ntk, "truncated": tr, "think_tail": tail}
        for (it, _, names, _), (lp, miss, nin), (ntk, tr, tail) in zip(rows, r1, meta)]}

    # ---- single-stream latency on the timed ids (the board times standard-tier items)
    lat = {"nothink": {}, "think": {}}
    by_id = {it["id"]: (it, u, labels) for it, u, _, labels in rows}
    llm.generate([chat(rows[0][1], False)], read_sp, use_tqdm=False)  # warm
    for iid in timed_ids:
        it, u, labels = by_id[iid]
        t0 = time.time()
        read([chat(u, False)], [labels])
        lat["nothink"][iid] = time.time() - t0
        t0 = time.time()
        g = llm.generate([chat(u, True)], gen_sp, use_tqdm=False)[0]
        tr = g.outputs[0].finish_reason == "length"
        read([chat(u, True) + g.outputs[0].text.rstrip() + (BUDGET_TAIL if tr else "\n") + "</think>\n\nAnswer:"],
             [labels])
        lat["think"][iid] = time.time() - t0
    out["latency"] = lat
    return out


@APP.local_entrypoint()
def main(models: str, items_path: str, out_dir: str, limit: int = 0):
    import pathlib

    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    if limit:
        items = items[:limit]
    timed = [it["id"] for it in items if it["id"].startswith("std-") or it.get("_tier") == "standard"]
    specs = [m.split("@") for m in models.split(",")]
    calls = [run.spawn(mid, rev, items, timed) for mid, rev in specs]
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for (mid, _), c in zip(specs, calls):
        res = c.get()
        p = pathlib.Path(out_dir) / (mid.split("/")[-1] + ".json")
        p.write_text(json.dumps(res), encoding="utf-8")
        print("wrote", p, "load_s", round(res["load_s"], 1), "vllm", res["vllm"],
              {a: round(v["wall_s"], 1) for a, v in res["arms"].items()})
