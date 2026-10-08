"""API-Bank-tool-selection (catalog 3) from the official API-Bank TRAINING split.

Source: HF ``liminghao1630/API-Bank`` @ 12e8158b7628c168f07e8f31fbbe3445e99f44cf, ``training-data/lv1-api-train.json``
(MIT). Each sample = 2-4 API descriptions + a dialogue prefix + the gold ``API-Request: [Name(...)]``. The test split
(``test-data/``) and the Index's ``lv1-lv2-samples`` are never read.

Eval presentation copied from the Index rows: ``state = {dialogue: [{role: User|AI, text} | {role: API, api_name,
param_dict, result: {api_name, input, output, exception}}], available_tools: [{name, description, input_parameters,
output_parameters}], task: "API-Bank current API tool selection"}``; one choice question ``tool`` with ids
``option_0..`` whose descriptions are tool names; the eval instruction verbatim. The eval shows the full 53-tool
catalog; here the catalog is the sample's given APIs + distractors from the training API pool (half nearest by
description token overlap, half random) capped at 16 options (consumer limit). Catalog order and option order are
shuffled independently; the gold option position is uniform.
"""
from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .. import config
from .common import make_record, shuffled_options

FAMILY = "API-Bank-tool-selection"
SOURCE = "b1fam_apibank"
LICENSE = "MIT (API-Bank, liminghao1630/API-Bank training-data)"
REVISION = "12e8158b7628c168f07e8f31fbbe3445e99f44cf"
FILES = {"lv1-api-train.json": "df55440a51e1ed86c261b3c1be1e636ac41a0f337eb3d3c23c7f1c8d5ba8154e"}
SRC_DIR = config.SOURCES_DIR / SOURCE
TASK = "API-Bank current API tool selection"
INSTR = ("Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in state. "
         "Do not infer arguments or use the current/future API result.")
PER_DIALOGUE = 3

_REQ = re.compile(r"^API-Request:\s*\[(.*)\]\s*$", re.S)
_TOK = re.compile(r"[a-z]+")


def _sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def parse_call(s: str) -> tuple[str, dict] | None:
    """``Name(k='v', ...)`` -> (name, params). Parsed with ``ast`` only (values via literal_eval; never executed)."""
    try:
        node = ast.parse(s.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
        return None
    return node.func.id, {k.arg: _lit(k.value) for k in node.keywords if k.arg}


def _result(s: str):
    s = s.strip()
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:
            pass
    return s


def parse_sample(x: dict) -> dict | None:
    lines = x["input"].split("\n")
    apis = [json.loads(l) for l in lines if l.startswith('{"apiCode"')]
    rest = [l for l in lines if not l.startswith('{"apiCode"')]
    dialogue: list[dict] = []
    for l in rest:
        if l.startswith("Generate API Request:"):
            break
        if l.startswith("User: "):
            dialogue.append({"role": "User", "text": l[6:].strip()})
        elif l.startswith("AI: "):
            dialogue.append({"role": "AI", "text": l[4:].strip()})
        elif l.startswith("API-Request: "):
            body = l[len("API-Request: "):]
            call, _, res = body.partition("]->")
            pc = parse_call(call.lstrip("["))
            if pc is None:
                return None
            name, params = pc
            dialogue.append({"role": "API", "api_name": name, "param_dict": params,
                             "result": {"api_name": name, "input": params, "output": _result(res), "exception": None}})
        elif l.strip() and dialogue and "text" in dialogue[-1]:
            dialogue[-1]["text"] += "\n" + l.strip()
        elif l.strip():
            return None
    m = _REQ.match(x["output"].strip())
    pc = parse_call(m.group(1)) if m else None
    if not pc or not dialogue or dialogue[-1]["role"] != "User":
        return None
    gold = pc[0]
    if gold == "ToolSearcher" or gold not in {a["apiCode"] for a in apis}:
        return None
    return {"apis": apis, "dialogue": dialogue, "gold": gold}


def tool_entry(a: dict) -> dict:
    return {"name": a["apiCode"], "description": a.get("description", ""),
            "input_parameters": a.get("parameters") or {}, "output_parameters": a.get("response") or {}}


def _toks(a: dict) -> set[str]:
    return set(_TOK.findall((a["apiCode"] + " " + a.get("description", "")).lower()))


def load(seed: int, target: int = 3000, log=print) -> list[dict]:
    for fn, want in FILES.items():
        got = _sha(SRC_DIR / fn)
        if got != want:
            raise RuntimeError(f"{fn}: sha256 {got} != pinned {want}")
    rows = json.load(open(SRC_DIR / "lv1-api-train.json", encoding="utf-8"))
    pool: dict[str, dict] = {}
    for x in rows:
        for l in x["input"].split("\n"):
            if l.startswith('{"apiCode"'):
                a = json.loads(l)
                pool.setdefault(a["apiCode"], a)
    pool.pop("ToolSearcher", None)
    names = sorted(pool)
    toks = {n: _toks(pool[n]) for n in names}
    parsed, bad = [], 0
    for i, x in enumerate(rows):
        p = parse_sample(x)
        if p is None:
            bad += 1
            continue
        p["idx"] = i
        parsed.append(p)
    order = list(range(len(parsed)))
    random.Random(f"{seed}|apibank|order").shuffle(order)
    per_dlg: Counter = Counter()
    out = []
    stats = Counter(parse_failed_or_toolsearcher=bad, parsed=len(parsed))
    for j in order:
        if len(out) >= target:
            break
        p = parsed[j]
        dkey = p["dialogue"][0]["text"][:200] + "|" + ",".join(sorted(a["apiCode"] for a in p["apis"]))
        if per_dlg[dkey] >= PER_DIALOGUE:
            stats["skipped_per_dialogue_cap"] += 1
            continue
        per_dlg[dkey] += 1
        rng = random.Random(f"{seed}|apibank|{p['idx']}")
        given = []
        for a in p["apis"]:
            if a["apiCode"] not in {g["apiCode"] for g in given}:
                given.append(a)
        n_opts = rng.choice([max(len(given), 4)] + list(range(max(len(given), 6), 17)))
        have = {a["apiCode"] for a in given}
        g_t = toks[p["gold"]] if p["gold"] in toks else _toks(next(a for a in given if a["apiCode"] == p["gold"]))
        near = sorted((n for n in names if n not in have),
                      key=lambda n: (-len(g_t & toks[n]) / (len(g_t | toks[n]) or 1), n))[:40]
        rng.shuffle(near)
        extra: list[str] = []
        k = n_opts - len(given)
        for n in near[: (k + 1) // 2]:
            extra.append(n)
        while len(extra) < k:
            n = names[rng.randrange(len(names))]
            if n not in have and n not in extra:
                extra.append(n)
        catalog = given + [pool[n] for n in extra]
        rng.shuffle(catalog)
        others = [a["apiCode"] for a in catalog if a["apiCode"] != p["gold"]]
        opts, pos = shuffled_options(p["gold"], others, rng, len(catalog))
        criteria = {f"option_{i}": n for i, n in enumerate(opts)}
        state = {"dialogue": p["dialogue"], "available_tools": [tool_entry(a) for a in catalog], "task": TASK}
        q = {"tool": {"type": "choice", "instructions": INSTR, "criteria": criteria}}
        item = {"dialogue": [t.get("text") or {"api_name": t["api_name"], "param_dict": t["param_dict"]}
                             for t in p["dialogue"]]}
        rec = make_record(family=FAMILY, source=SOURCE, source_id=f"lv1-{p['idx']}", source_split="train",
                          provenance="train-split", license=LICENSE, state=state, questions=q,
                          labels={"tool": f"option_{pos}"},
                          aug={"n_options": len(opts), "gold_pos": pos, "n_given_apis": len(given),
                               "distractors": "given+near+random", "source_file": "training-data/lv1-api-train.json",
                               "source_revision": REVISION},
                          probe={"state": item, "questions": {}})
        out.append(rec)
    stats["records"] = len(out)
    log(f"  api-bank: {dict(stats)}")
    return out
