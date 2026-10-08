"""BFCL-tool-selection (catalog 1): SYNTHETIC generator (BFCL publishes no train split).

Eval presentation copied from the Index rows: ``state = {conversation: [[{role, content}, ...]], functions: [{name,
description, parameters: {type: "dict", properties, required}}], task: "BFCL tool-name selection"}``; one yes/no
question per candidate, id ``tool_<name>``, the eval instruction prefix + ``"\\n\\n\\n\\nCandidate tool: <name>"``,
criteria ``{yes: "Invoke this tool.", no: "Do not invoke this tool."}``. Exactly one candidate is gold (as in every
Index row). Candidate count follows the eval mix (1: ~25%, 2-10 most, cap 16 = MAX_QUESTIONS); candidates are the
gold function, same-domain siblings first (hard negatives) and then other domains; candidate order is shuffled, so
the gold position is uniform.

Correctness: the request is rendered from a template OF the gold function with every slot bound to a value, so the
gold is the only candidate whose purpose matches (siblings in the same domain have disjoint purposes; see
``bfcl_catalog``). No model is involved.
"""
from __future__ import annotations

import random
import re

from . import bfcl_catalog as K
from .common import make_record

FAMILY = "BFCL-tool-selection"
SOURCE = "b1fam_bfcl_synth"
LICENSE = "generated (repo code; no third-party text)"
TASK = "BFCL tool-name selection"
INSTR = ("Given the complete user conversation and the published function schemas in state, mark each candidate tool "
         "yes if it should be invoked to answer the request, or no otherwise. This evaluates tool-name selection only; "
         "do not produce arguments or call ordering.\n\n\n\nCandidate tool: ")
CRIT = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
# candidate-count weights (eval: 1:657 2:219 3:299 4:176 5:191 6:78 7:29 8:18 9:11 10:14 26:1 37:1); single
# candidates are trivially "yes", so they are down-weighted relative to the eval mix
N_WEIGHTS = {1: 25, 2: 14, 3: 17, 4: 12, 5: 12, 6: 7, 7: 4, 8: 3, 9: 2, 10: 2, 12: 1, 16: 1}
_SLOT = re.compile(r"\{(\w+)\}")


def style_name(snake: str, domain: str, style: str) -> str:
    if style == "camel":
        a, *b = snake.split("_")
        return a + "".join(x.title() for x in b)
    if style == "dotted":
        return f"{domain}.{snake}"
    return snake


def schema(fn: tuple, name: str) -> dict:
    _, desc, params, _ = fn
    props = {}
    for pname, typ, pdesc, _req, enum in params:
        p = {"type": typ, "description": pdesc}
        if enum:
            p["enum"] = list(enum)
        props[pname] = p
    return {"name": name, "description": desc,
            "parameters": {"type": "dict", "properties": props, "required": [p[0] for p in params if p[3]]}}


def fill(template: str, rng: random.Random) -> str:
    return _SLOT.sub(lambda m: rng.choice(K.SLOTS[m.group(1)]), template)


def generate(seed: int, target: int = 2400, log=print) -> list[dict]:
    allf = [(d, fn) for d, fns in K.CATALOG.items() for fn in fns]
    names = [fn[0] for _, fn in allf]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate function names in the BFCL catalog")
    for _, fn in allf:
        for t in fn[3]:
            for s in _SLOT.findall(t):
                if s not in K.SLOTS:
                    raise AssertionError(f"{fn[0]}: unknown slot {s}")
    ns, ws = zip(*N_WEIGHTS.items())
    out, seen, per_req = [], set(), {}
    k = 0
    while len(out) < target:
        rng = random.Random(f"{seed}|bfcl|{k}")
        k += 1
        if k > 50 * target:
            raise RuntimeError(f"bfcl: only {len(out)} distinct records after {k} draws")
        dom, gold = allf[rng.randrange(len(allf))]
        n = min(rng.choices(ns, ws)[0], len(allf))
        sib = [fn for fn in K.CATALOG[dom] if fn[0] != gold[0]]
        rng.shuffle(sib)
        oth = [(d, fn) for d, fn in allf if d != dom]
        rng.shuffle(oth)
        cands = [(dom, gold)] + [(dom, fn) for fn in sib[: max(0, min(len(sib), (n - 1 + 1) // 2 + 1))]]
        cands = cands[:n]
        cands += oth[: n - len(cands)]
        rng.shuffle(cands)
        style = rng.choice(["snake", "snake", "camel", "dotted"])
        request = fill(rng.choice(gold[3]), rng)
        conv = []
        if rng.random() < 0.3:
            conv.append({"role": "system", "content": rng.choice(K.SYSTEM_MESSAGES)})
        if rng.random() < 0.15:
            u, a = rng.choice(K.PRIOR_TURNS)
            conv += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        conv.append({"role": "user", "content": request})
        key = (request, tuple(sorted(fn[0] for _, fn in cands)), style)
        if key in seen or per_req.get(request, 0) >= 2:
            continue
        seen.add(key)
        per_req[request] = per_req.get(request, 0) + 1
        fnames = [style_name(fn[0], d, style) for d, fn in cands]
        gname = style_name(gold[0], dom, style)
        state = {"conversation": [conv], "functions": [schema(fn, nm) for (d, fn), nm in zip(cands, fnames)],
                 "task": TASK}
        qs = {f"tool_{nm}": {"type": "choice", "instructions": INSTR + nm, "criteria": dict(CRIT)} for nm in fnames}
        labels = {f"tool_{nm}": ("yes" if nm == gname else "no") for nm in fnames}
        rec = make_record(family=FAMILY, source=SOURCE, source_id=f"{seed}-{k - 1}", source_split="synthetic",
                          provenance="synthetic", license=LICENSE, state=state, questions=qs, labels=labels,
                          aug={"n_candidates": len(fnames), "gold_pos": fnames.index(gname), "gold_fn": gold[0],
                               "domain": dom, "name_style": style, "generator": "data.b1_families.bfcl"},
                          probe={"state": {"request": request}, "questions": {}})
        out.append(rec)
    log(f"  bfcl: {len(out)} records from {k} draws ({len(allf)} functions, {len(K.CATALOG)} domains)")
    return out
