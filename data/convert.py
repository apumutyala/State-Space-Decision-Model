"""Source loaders: pinned public TRAIN splits -> base items (one decision each), before schema augmentation.

A base item is format-neutral:

    {"id", "source", "source_split", "source_id", "license", "family",
     "fields": {name: text, ...},            # the item content, in natural order
     "qtype": "choice" | "noul" | "score",
     "options": {semantic_id: description}   # choice; list of level descriptions for score; None for noul
     "gold": option id | "true"/"false" | level index (int),
     "item_fields": fields that carry the item (dedup 'item' tier; the rest is context: tool catalogs, ...),
     "item_options": "all" | "gold" | "none" -- which option descriptions are item content (MCQ endings) rather
                     than a shared label set (intents, NLI relations),
     "opaque_ids": True when the loader's option ids would reveal the gold (gsm8k "correct", when2call "gold")
                   or the source row; render never shows such ids}

``augment.render`` turns it into a SystemOne record (state + typed questions + labels). Loaders only read the
train files listed in ``sources.REGISTRY``; every file is sha256-checked against the lock before parsing.
"""
from __future__ import annotations

import csv
import json
import random
from typing import Iterator

from . import sources as S

SPLIT = "train"


def _lic(sid: str) -> str:
    return S.REGISTRY[sid]["license"]


def _parquet_rows(sid: str, fname: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(S.verified_path(sid, fname)).to_pylist()


def _order(n: int, sid: str, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(f"{seed}|order|{sid}").shuffle(idx)
    return idx


def _base(sid, i, family, fields, qtype, options, gold, split=SPLIT, **extra):
    return {"id": f"m2:{sid}:{i}", "source": sid, "source_split": split, "source_id": str(i), "license": _lic(sid),
            "family": family, "fields": fields, "qtype": qtype, "options": options, "gold": gold, **extra}


def _slug_desc(slug: str) -> str:
    return slug.replace("_", " ")


# ----------------------------------------------------------------------------------------------- loaders
def banking77(seed: int) -> Iterator[dict]:
    with open(S.verified_path("banking77", "train.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    intents = sorted({r["category"] for r in rows})
    opts = {c: _slug_desc(c) for c in intents}
    for i in _order(len(rows), "banking77", seed):
        r = rows[i]
        yield _base("banking77", i, "intent_banking", {"customer_message": r["text"]}, "choice", opts, r["category"],
                    item_fields=["customer_message"], item_options="none")


def clinc150(seed: int) -> Iterator[dict]:
    d = json.load(open(S.verified_path("clinc150", "data_full.json"), encoding="utf-8"))
    rows = [(t, l, "train") for t, l in d["train"]] + [(t, "oos", "oos_train") for t, _ in d["oos_train"]]
    intents = sorted({l for _, l, _ in rows})
    opts = {c: ("out of scope: none of the supported intents" if c == "oos" else _slug_desc(c)) for c in intents}
    for i in _order(len(rows), "clinc150", seed):
        t, l, split = rows[i]
        yield _base("clinc150", i, "intent_general", {"user_request": t}, "choice", opts, l, split=split,
                    item_fields=["user_request"], item_options="none")


NLI = {"entailment": "The premise entails the hypothesis.", "neutral": "The premise neither entails nor "
       "contradicts the hypothesis.", "contradiction": "The premise contradicts the hypothesis."}


def anli(seed: int) -> Iterator[dict]:
    rows = []
    for rnd in (1, 2, 3):
        for r in _parquet_rows("anli", f"plain_text/train_r{rnd}-00000-of-00001.parquet"):
            rows.append(r)
    labels = ["entailment", "neutral", "contradiction"]
    for i in _order(len(rows), "anli", seed):
        r = rows[i]
        yield _base("anli", r["uid"], "nli", {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "choice",
                    dict(NLI), labels[int(r["label"])], item_fields=["premise", "hypothesis"], item_options="none")


def hellaswag(seed: int) -> Iterator[dict]:
    rows = _parquet_rows("hellaswag", "data/train-00000-of-00001.parquet")
    for i in _order(len(rows), "hellaswag", seed):
        r = rows[i]
        if r["label"] in (None, ""):
            continue
        opts = {f"ending_{k}": e.strip() for k, e in enumerate(r["endings"])}
        # ``ind`` is not unique within the train file; the row index is the record id, ``ind`` the source id
        yield _base("hellaswag", i, "continuation", {"activity": r["activity_label"], "context": r["ctx"]},
                    "choice", opts, f"ending_{int(r['label'])}", item_fields=["context"], item_options="all",
                    source_ind=str(r["ind"]))


def winogrande(seed: int) -> Iterator[dict]:
    rows = _parquet_rows("winogrande", "winogrande_xl/train-00000-of-00001.parquet")
    for i in _order(len(rows), "winogrande", seed):
        r = rows[i]
        opts = {"first": r["option1"], "second": r["option2"]}
        yield _base("winogrande", i, "fill_blank", {"sentence": r["sentence"]}, "choice", opts,
                    "first" if r["answer"] == "1" else "second", item_fields=["sentence"], item_options="all")


def _num_fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def gsm8k_distractors(gold: float, k: int, rng: random.Random) -> list[str]:
    cands = set()
    g = gold
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50, 100):
        cands.update({g + d, g - d})
    cands.update({g * 2, g / 2, g * 10, g / 10, g * 3, g + g / 4, g - g / 4})
    s = _num_fmt(g)
    if len(s) > 1 and s.isdigit():
        cands.add(float(s[::-1]))
    cands = [c for c in cands if c != g and (g < 0 or c >= 0)]
    cands = sorted({_num_fmt(round(c, 2)) for c in cands} - {_num_fmt(g)})
    rng.shuffle(cands)
    return cands[:k]


def gsm8k(seed: int) -> Iterator[dict]:
    rows = _parquet_rows("gsm8k", "main/train-00000-of-00001.parquet")
    for i in _order(len(rows), "gsm8k", seed):
        r = rows[i]
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(ans)
        except ValueError:
            continue
        rng = random.Random(f"{seed}|gsm8k|{i}")
        k = rng.choice([3, 9])  # 4- and 10-choice, like the Index adaptation
        opts = {"correct": _num_fmt(g)}
        for j, d in enumerate(gsm8k_distractors(g, k, rng)):
            opts[f"distractor_{j}"] = d
        yield _base("gsm8k", i, "math_mcq", {"problem": r["question"]}, "choice", opts, "correct", opaque_ids=True,
                    item_fields=["problem"], item_options="none")


def mmlu_aux(seed: int) -> Iterator[dict]:
    rows = _parquet_rows("mmlu_aux", "auxiliary_train/train-00000-of-00001.parquet")
    for i in _order(len(rows), "mmlu_aux", seed):
        r = rows[i]["train"]
        ch = r["choices"]
        if not ch or len(ch) < 2:
            continue
        opts = {f"choice_{k}": c for k, c in enumerate(ch)}
        yield _base("mmlu_aux", i, "mcq", {"question": r["question"]}, "choice", opts, f"choice_{int(r['answer'])}",
                    item_fields=["question"], item_options="all")


SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]


def sst5(seed: int) -> Iterator[dict]:
    rows = [json.loads(l) for l in open(S.verified_path("sst5", "train.jsonl"), encoding="utf-8") if l.strip()]
    for i in _order(len(rows), "sst5", seed):
        r = rows[i]
        yield _base("sst5", i, "sentiment_score", {"review": r["text"]}, "score", list(SST5), int(r["label"]),
                    item_fields=["review"], item_options="none")


def boolq(seed: int) -> Iterator[dict]:
    rows = _parquet_rows("boolq", "data/train-00000-of-00001.parquet")
    for i in _order(len(rows), "boolq", seed):
        r = rows[i]
        yield _base("boolq", i, "yesno_passage", {"passage": r["passage"], "question": r["question"].rstrip("?") + "?"},
                    "noul", None, "true" if r["answer"] else "false", item_fields=["passage", "question"],
                    item_options="none")


W2C_TOOLCALL = '{{"name": "{name}", "arguments": {{}}}}'


def when2call(seed: int) -> Iterator[dict]:
    rows = [json.loads(l) for l in open(S.verified_path("when2call", "train/when2call_train_sft.jsonl"), encoding="utf-8")]
    responses = [r["messages"][-1]["content"] for r in rows]
    asks = [k for k, a in enumerate(responses) if a.strip().endswith("?")]
    refusals = [k for k, a in enumerate(responses) if not a.strip().endswith("?")]
    for i in _order(len(rows), "when2call", seed):
        r = rows[i]
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        gold_txt = responses[i]
        gold_kind = "ask" if gold_txt.strip().endswith("?") else "decline"
        rng = random.Random(f"{seed}|w2c|{i}")
        opts = {"gold": gold_txt}
        other = rng.choice(refusals if gold_kind == "ask" else asks)
        if other != i:
            opts["other_kind"] = responses[other]
        tools = [json.loads(t) if isinstance(t, str) else t for t in r["tools"]]
        if tools:
            opts["tool_call"] = W2C_TOOLCALL.format(name=rng.choice(tools)["name"])
        opts["direct_answer"] = rng.choice(["Sure, I went ahead and completed that request for you just now.",
                                            "Done. Everything you asked for has been taken care of on my side.",
                                            "Here is what you asked for, finished without needing any tools."])
        yield _base("when2call", i, "response_select", {"available_tools": tools, "user_question": user}, "choice",
                    opts, "gold", opaque_ids=True, item_fields=["user_question"], item_options="all", label_note=f"gold = source SFT response ({gold_kind}); distractors drawn by rule")


def toolace(seed: int) -> Iterator[dict]:
    rows = json.load(open(S.verified_path("toolace", "data.json"), encoding="utf-8"))
    for i in _order(len(rows), "toolace", seed):
        r = rows[i]
        sysmsg = r["system"]
        at = sysmsg.find("[", sysmsg.find("invoke:"))
        if sysmsg.find("invoke:") < 0 or at < 0:
            continue
        try:
            tools, _ = json.JSONDecoder().raw_decode(sysmsg[at:])
        except json.JSONDecodeError:
            continue
        if not isinstance(tools, list):
            continue
        names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
        if len(names) < 2 or len(set(names)) != len(names):
            continue
        conv = r["conversations"]
        k = next((j for j, c in enumerate(conv) if c["from"] == "assistant" and c["value"].strip().startswith("[")),
                 None)
        if k is None or k == 0:
            continue
        call = conv[k]["value"]
        called = [n for n in names if f"{n}(" in call]
        if not called:
            continue
        dialogue = [{"role": c["from"], "content": c["value"]} for c in conv[:k]]
        fields = {"tools": tools, "dialogue": dialogue}
        if len(called) == 1:
            opts = {n: n for n in names}
            opts["no_tool"] = "none of the listed tools"
            yield _base("toolace", i, "tool_select", fields, "choice", opts, called[0], item_fields=["dialogue"],
                        item_options="none")
        else:
            # multi-select: one yes/no question per tool (BFCL-style), carried as a multi-question base
            yield _base("toolace", i, "tool_multi", fields, "multi_yesno", {n: n for n in names},
                        sorted(called), item_fields=["dialogue"], item_options="none")


def nq_pairs(seed: int) -> list[dict]:
    return _parquet_rows("nq", "pair/train-00000-of-00001.parquet")


def nq(seed: int) -> Iterator[dict]:
    rows = nq_pairs(seed)
    for i in _order(len(rows), "nq", seed):
        rng = random.Random(f"{seed}|nq|{i}")
        k = rng.choice([8, 16, 32, 64, 128, 255])
        neg = set()
        while len(neg) < k - 1:
            j = rng.randrange(len(rows))
            if j != i and rows[j]["answer"] != rows[i]["answer"]:
                neg.add(j)
        opts = {f"passage_{i}": rows[i]["answer"]}
        opts.update({f"passage_{j}": rows[j]["answer"] for j in sorted(neg)})
        yield _base("nq", i, "retrieval", {"query": rows[i]["query"]}, "choice", opts, f"passage_{i}",
                    opaque_ids=True, item_fields=["query"], item_options="gold")


LOADERS = {"banking77": banking77, "clinc150": clinc150, "anli": anli, "hellaswag": hellaswag,
           "winogrande": winogrande, "gsm8k": gsm8k, "mmlu_aux": mmlu_aux, "sst5": sst5, "boolq": boolq,
           "when2call": when2call, "toolace": toolace, "nq": nq}
