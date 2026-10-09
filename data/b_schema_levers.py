"""bschema-draw-v3 lever rows (coordinator 2026-10-09, data-lever analysis claude/b-gap-breakdown @47222a0).

Builds NEW training rows (format coverage and natural-rate families) from TRAIN splits already on disk, to be merged
into bschema-draw230k-v2e by ``scripts/b_schema_swap_pool.py`` (P3 checks) after a token check. No target rate or shape
comes from DEV / eval rows (agent-a's hard condition): every target is recorded with its source in
``target_provenance`` (file, split, computed value).

Levers (each row carries ``aug.lever``):
  apibank_long  API-Bank lv1 TRAIN dialogues with LONG tool catalogs: the sample's given APIs + distractors from the
                lv1 train API pool (1,767 APIs; half nearest by name/description token overlap, half random), compact
                one-line tool entries (name + description). K per row = uniform over [max(17, K_fit // 2), K_fit],
                K_fit = the largest catalog whose estimated length fits the 2,048-token budget (format coverage; the
                train catalog is the 1,767-API pool, and the token budget, not the eval, sets K).
  mmlu_k10      MMLU auxiliary_train items widened to 10 options. The gold is unchanged. Distractors are NON-gold
                options of the most similar other aux questions (rare-word overlap of question text; RACE siblings of
                the same passage come first: the "same subject" proxy, since aux_train has no subject field). An
                option is never borrowed when its normalised text equals any of the item's own options (the gold
                above all), and aggregate options ("all of the above", "A and B") are never borrowed or widened.
  boolq_yesno   BoolQ TRAIN as a {yes, no} CHOICE view of every train item at its natural rate (62.3 % yes).
  clinc_k151    CLINC150 full-catalog views (150 intents + out-of-scope = 151 options) over train + oos_train sampled
                uniformly, so out-of-scope stays at its TRAIN share (100 / 15,100).
  banking_k77   BANKING77 full 77-option catalogs over train sampled uniformly (natural label rates). The count is
                set so the draw's K77 share equals D16's (b1-draw100k-v2 minus d1a rows; a training draw).
Every row gets a schema view (option-id style, instruction frame, question name) from ``data.b_schema``. Rows per source
item stay <= MAX_REUSE (8) counting the parent draw's rows.

    python -X utf8 -m data.b_schema_levers --parent <v2e records.jsonl.gz> --src-dir <parquet exports> --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path

from . import config
from .b_schema import AGGREGATE, ID_STYLES, MAX_REUSE, canonical, paraphrase, relabel, rename_questions

SEED = 20261009
SRC = config.SOURCES_DIR
TOK_BUDGET = 1750  # estimated tokens (chars / CHARS_PER_TOK); the trainer token check is the real gate
CHARS_PER_TOK = 3.0
_WORD = re.compile(r"[a-z]{4,}")
STOP = set("that this with from what which when where would could should there their they them have will were "
           "been about into than then also only some more most such very your other these those many much each "
           "does following passage according author text question best true false".split())


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).casefold()).strip()


def _jsonl_gz(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _view(rec: dict, rng: random.Random, styles=ID_STYLES, keep_ids: bool = False) -> dict:
    """Schema view: option-id style + position, instruction frame, question name (no K change, no type switch)."""
    (qid, q), = rec["questions"].items()
    gold = rec["labels"][qid]
    ops = []
    style = "keep" if keep_ids else rng.choice(styles)
    if style == "words" and len(q["criteria"]) > 44:
        style = "letters"
    q, gold = relabel(q, gold, rng, style)
    ops.append(f"ids:{style}")
    if rng.random() < 0.75:
        ins, op = paraphrase(q.get("instructions"), q["type"], rng)
        if op:
            q = {**q, "instructions": ins}
            ops.append(op)
    nq = qid
    if rng.random() < 0.6:
        nq = rename_questions([qid], rng)[qid]
        ops.append("qid_rename")
    rec["questions"] = {nq: q}
    rec["labels"] = {nq: gold}
    rec["aug"]["schema_synth"] = {"ops": ops, "seed": SEED, "lever_view": True}
    return rec


def _record(src: str, src_id: str, tag: str, family: str, license_: str, state, question: dict, gold: str,
            lever: dict, split: str = "train") -> dict:
    base = f"m2:{src}:{src_id}"
    return {"schema": "ssdm-train-v0", "id": f"{base}~view:{tag}", "source": src, "source_id": src_id,
            "source_split": split, "family": family, "license": license_, "provenance": "train-split",
            "state": state, "questions": {"q": question}, "labels": {"q": gold}, "untrusted_fields": [],
            "aug": {"b1_view": {"kind": tag, "parent_id": base, "base_id": base}, "lever": lever}}


#: source items held out by cohort v2 / v3 (data.b1.exclusion_sets): never generated (the P3 checker would refuse them)
BLOCKED: set[str] = set()


def _take(reuse: collections.Counter, src: str, src_id: str) -> bool:
    k = f"m2:{src}:{src_id}"
    if k in BLOCKED or reuse[k] >= MAX_REUSE:
        return False
    reuse[k] += 1
    return True


# ----------------------------------------------------------------------------------------------- API-Bank long
APIBANK_REV = "12e8158b7628c168f07e8f31fbbe3445e99f44cf"
APIBANK_LIC = "MIT (API-Bank, liminghao1630/API-Bank training-data)"
APIBANK_SHA = "df55440a51e1ed86c261b3c1be1e636ac41a0f337eb3d3c23c7f1c8d5ba8154e"
APIBANK_TASK = "API-Bank current API tool selection"
APIBANK_INSTR = ("Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in "
                 "state. Do not infer arguments or use the current/future API result.")
_REQ = re.compile(r"^API-Request:\s*\[(.*)\]\s*$", re.S)
_TOK = re.compile(r"[a-z]+")


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def _parse_call(s: str):
    """``Name(k='v', ...)`` parsed with ``ast`` only (never executed); as data/b1_families/apibank.py @a4bc718."""
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


def _parse_sample(x: dict):
    lines = x["input"].split("\n")
    apis = [json.loads(l) for l in lines if l.startswith('{"apiCode"')]
    dialogue = []
    for l in (l for l in lines if not l.startswith('{"apiCode"')):
        if l.startswith("Generate API Request:"):
            break
        if l.startswith("User: "):
            dialogue.append({"role": "User", "text": l[6:].strip()})
        elif l.startswith("AI: "):
            dialogue.append({"role": "AI", "text": l[4:].strip()})
        elif l.startswith("API-Request: "):
            call, _, res = l[len("API-Request: "):].partition("]->")
            pc = _parse_call(call.lstrip("["))
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
    pc = _parse_call(m.group(1)) if m else None
    if not pc or not dialogue or dialogue[-1]["role"] != "User":
        return None
    gold = pc[0]
    if gold == "ToolSearcher" or gold not in {a["apiCode"] for a in apis}:
        return None
    return {"apis": apis, "dialogue": dialogue, "gold": gold}


def _canon_name(n: str) -> str:
    return re.sub(r"[^a-z0-9]", "", n.casefold())


def apibank_long(reuse, prov, log=print) -> list[dict]:
    p = SRC / "b1fam_apibank" / "lv1-api-train.json"
    if _sha_file(p) != APIBANK_SHA:
        raise SystemExit("lv1-api-train.json sha drift")
    rows = json.load(open(p, encoding="utf-8"))
    pool = {}
    for x in rows:
        for l in x["input"].split("\n"):
            if l.startswith('{"apiCode"'):
                a = json.loads(l)
                pool.setdefault(a["apiCode"], a)
    pool.pop("ToolSearcher", None)
    names = sorted(pool)
    toks = {n: set(_TOK.findall((n + " " + pool[n].get("description", "")).lower())) for n in names}
    entry = {n: {"name": n, "description": " ".join(str(pool[n].get("description", "")).split())} for n in names}
    elen = {n: len(json.dumps(entry[n], ensure_ascii=False)) + len(n) + 12 for n in names}
    prov["apibank_long"] = {"file": "b1fam_apibank/lv1-api-train.json", "sha256": APIBANK_SHA, "split": "train",
                            "revision": APIBANK_REV, "n_samples": len(rows), "train_api_pool": len(names),
                            "given_apis_per_sample": "2-4",
                            "K_rule": f"uniform over [max(17, K_fit // 2), K_fit]; K_fit = largest catalog with estimated "
                                      f"tokens (chars / {CHARS_PER_TOK}) <= {TOK_BUDGET}; the train API pool "
                                      f"({len(names)}) bounds K, the token budget sets it (no eval shape)"}
    out, stats, kh = [], collections.Counter(), collections.Counter()
    per_dlg = collections.Counter()
    for i, x in enumerate(rows):
        s = _parse_sample(x)
        if s is None:
            stats["parse_failed_or_toolsearcher"] += 1
            continue
        dkey = s["dialogue"][0]["text"][:200] + "|" + ",".join(sorted(a["apiCode"] for a in s["apis"]))
        if per_dlg[dkey] >= 3:
            stats["per_dialogue_cap"] += 1
            continue
        rng = random.Random(f"{SEED}|apil|{i}")
        given = []
        for a in s["apis"]:
            if a["apiCode"] in pool and a["apiCode"] not in {g for g in given}:
                given.append(a["apiCode"])
        if s["gold"] not in given:
            stats["gold_not_in_pool"] += 1
            continue
        gc = _canon_name(s["gold"])
        fixed = len(json.dumps(s["dialogue"], ensure_ascii=False)) + len(APIBANK_INSTR) + len(APIBANK_TASK) + 200
        g_t = toks[s["gold"]]
        cand = [n for n in names if n not in given and _canon_name(n) != gc]
        near = sorted(cand, key=lambda n: (-len(g_t & toks[n]) / (len(g_t | toks[n]) or 1), n))[:120]
        rng.shuffle(near)
        far = [n for n in cand if n not in set(near)]
        rng.shuffle(far)
        order = []
        ni = fi = 0
        while len(order) < len(cand) and len(order) < 400:  # alternate near / random
            if (len(order) % 2 == 0 and ni < len(near)) or fi >= len(far):
                order.append(near[ni]); ni += 1
            else:
                order.append(far[fi]); fi += 1
        chars = fixed + sum(elen[n] for n in given)
        kfit = len(given)
        for n in order:
            if (chars + elen[n]) / CHARS_PER_TOK > TOK_BUDGET:
                break
            chars += elen[n]
            kfit += 1
        if kfit < 17:
            stats["k_fit_below_17"] += 1
            continue
        if not _take(reuse, "b1fam_apibank", f"lv1-{i}"):
            stats["held_out_or_reuse_cap"] += 1
            continue
        per_dlg[dkey] += 1
        k = rng.randint(max(17, kfit // 2), kfit)
        catalog = given + order[: k - len(given)]
        rng.shuffle(catalog)
        crit = {f"option_{j}": n for j, n in enumerate(catalog)}
        gold = f"option_{catalog.index(s['gold'])}"
        state = {"dialogue": s["dialogue"], "available_tools": [entry[n] for n in catalog], "task": APIBANK_TASK}
        q = {"type": "choice", "instructions": APIBANK_INSTR, "criteria": crit}
        rec = _record("b1fam_apibank", f"lv1-{i}", "apil", "API-Bank-tool-selection", APIBANK_LIC, state, q, gold,
                      {"lever": "apibank_long", "k": k, "k_fit_est": kfit, "n_given_apis": len(given),
                       "source_file": "training-data/lv1-api-train.json", "source_revision": APIBANK_REV})
        out.append(_view(rec, rng))
        kh[k // 10 * 10] += 1
    stats["records"] = len(out)
    prov["apibank_long"]["built"] = dict(stats)
    log("apibank_long", dict(stats), dict(sorted(kh.items())))
    return out


# ------------------------------------------------------------------------------------------------- MMLU K10
MMLU_SHA = "ae6662576ef989fed82a9289da1b89e950499c320f6849b052ec344dfcb709eb"
MMLU_LIC = "MIT (constituents: ARC, OBQA, RACE, MCTest; see SOURCES.md)"
MMLU_INSTR = ("Answer the multiple-choice question.", "Choose the best answer.", "Which option is correct?",
              "Select the correct answer to the question.", "Pick the right choice.",
              "Read the question and choose the most accurate option.")


def mmlu_k10(rows_path: Path, reuse, prov, n_target: int, log=print) -> list[dict]:
    rows = [r["train"] for r in _jsonl_gz(rows_path)]
    words = []
    df = collections.Counter()
    for r in rows:
        w = set(_WORD.findall(r["question"].casefold())) - STOP
        words.append(w)
        df.update(w)
    post = collections.defaultdict(list)
    for i, w in enumerate(words):
        for t in w:
            if 2 <= df[t] <= 300:
                post[t].append(i)
    order = list(range(len(rows)))
    random.Random(f"{SEED}|k10|order").shuffle(order)
    out, stats = [], collections.Counter()
    for i in order:
        if len(out) >= n_target:
            break
        r = rows[i]
        ch = [str(c) for c in (r.get("choices") or [])]
        if len(ch) < 2 or len(ch) > 9 or not (0 <= int(r["answer"]) < len(ch)):
            stats["bad_shape"] += 1
            continue
        if any(AGGREGATE.search(c) for c in ch):
            stats["aggregate_option"] += 1
            continue
        own = {_norm(c) for c in ch}
        if "" in own or len(own) < len(ch):
            stats["dup_or_empty_option"] += 1
            continue
        rare = sorted((t for t in words[i] if t in post), key=lambda t: df[t])[:8]
        sc = collections.Counter()
        for t in rare:
            idf = 1.0 / math.log(1 + df[t])
            for j in post[t]:
                if j != i:
                    sc[j] += idf
        pool = []
        seen = set(own)
        for j, _ in sc.most_common(40):
            rj = rows[j]
            for a, c in enumerate(rj.get("choices") or []):
                c = str(c)
                n = _norm(c)
                if a == int(rj["answer"]) or not n or n in seen or AGGREGATE.search(c) or len(c) > 300:
                    continue
                seen.add(n)
                pool.append(c)
        need = 10 - len(ch)
        if len(pool) < need:
            stats["too_few_distractors"] += 1
            continue
        if not _take(reuse, "mmlu_aux", str(i)):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(f"{SEED}|k10|{i}")
        opts = ch + pool[:need]
        crit = {f"choice_{k}": c for k, c in enumerate(opts)}
        q = {"type": "choice", "instructions": rng.choice(MMLU_INSTR), "criteria": crit}
        rec = _record("mmlu_aux", str(i), "k10", "mcq", MMLU_LIC, {"question": r["question"]}, q,
                      f"choice_{int(r['answer'])}",
                      {"lever": "mmlu_k10", "k": 10, "k_source": len(ch), "n_borrowed": need,
                       "distractor_rule": "non-gold options of rare-word-nearest aux questions; text != any own option"})
        out.append(_view(rec, rng))
    stats["records"] = len(out)
    prov["mmlu_k10"] = {"file": "mmlu_aux/train-00000-of-00001.parquet (cais/mmlu auxiliary_train)", "sha256": MMLU_SHA,
                        "split": "auxiliary_train", "n_rows": len(rows), "K": "10 (format view; MMLU-Pro shape)",
                        "n_target": n_target, "n_target_rule": "= mmlu_aux/mcq rows in v2e (keep the family volume)",
                        "built": dict(stats)}
    log("mmlu_k10", dict(stats))
    return out


# ------------------------------------------------------------------------------------------------- BoolQ yes/no
BOOLQ_SHA = "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623"
BOOLQ_INSTR = ("Using only the passage, answer the question: yes or no?", "Based on the passage, answer the yes/no question.",
               "Does the passage support answering yes? Choose yes or no.", "Answer from the passage: yes or no?",
               "According to the text, is the answer yes or no?", "Read the passage and decide: yes or no.")


def boolq_yesno(rows_path: Path, reuse, prov, log=print) -> list[dict]:
    rows = list(_jsonl_gz(rows_path))
    out, stats = [], collections.Counter()
    yes = sum(bool(r["answer"]) for r in rows)
    for i, r in enumerate(rows):
        if not _take(reuse, "boolq", str(i)):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(f"{SEED}|yn|{i}")
        crit = {"yes": "Yes.", "no": "No."}
        if rng.random() < 0.5:
            crit = {"no": "No.", "yes": "Yes."}
        q = {"type": "choice", "instructions": rng.choice(BOOLQ_INSTR), "criteria": crit}
        state = {"passage": r["passage"], "question": r["question"].rstrip("?") + "?"}
        if rng.random() < 0.5:
            state = {"question": state["question"], "passage": state["passage"]}
        rec = _record("boolq", str(i), "yn", "yesno_passage", "CC-BY-SA-3.0", state, q, "yes" if r["answer"] else "no",
                      {"lever": "boolq_yesno"})
        out.append(_view(rec, rng, keep_ids=True))
    stats["records"] = len(out)
    stats["yes"] = sum(1 for x in out if next(iter(x["labels"].values())) == "yes")
    prov["boolq_yesno"] = {"file": "boolq/train-00000-of-00001.parquet (google/boolq)", "sha256": BOOLQ_SHA,
                           "split": "train", "n_rows": len(rows), "train_yes_rate": round(yes / len(rows), 4),
                           "rule": "every train item not held out by cohort v2/v3 once as a {yes, no} choice; label = the train answer (natural rate)",
                           "yes_rate_of_built_rows": round(stats["yes"] / max(1, stats["records"]), 4),
                           "built": dict(stats)}
    log("boolq_yesno", dict(stats))
    return out


# --------------------------------------------------------------------------------------------- CLINC / BANKING
def _slug(s: str) -> str:
    return s.replace("_", " ")


INTENT_INSTR = {
    "intent_general": ("Classify the intent of this user request; use the out-of-scope option if no intent applies.",
                       "Which intent does the user's utterance express (or is it out of scope)?",
                       "Map the request to one supported intent, or mark it out of scope.",
                       "Identify the user's intent from the list; pick out of scope when nothing fits."),
    "intent_banking": ("Classify the banking intent of this customer message.",
                       "Which support intent does the customer's message express?",
                       "Route this banking request to the matching intent.",
                       "Pick the intent label that best fits the customer's request."),
}
CLINC_STYLES = ("letters", "digits", "digits1", "option_k", "opaque", "keep")
#: measured on the first build (v3w): K77 fits 2,048 only with digit ids (median 1,981; letters 2,149, option_k 2,290,
#: keep 3,305, opaque 3,620); CLINC K151 never fits (min 3,174 with digit ids), so it is reported, not built
K77_STYLES = ("digits", "digits1")


def clinc_k151(reuse, prov, n_target: int, log=print) -> list[dict]:
    p = SRC / "clinc150" / "data_full.json"
    d = json.load(open(p, encoding="utf-8"))
    rows = [(t, l, "train") for t, l in d["train"]] + [(t, "oos", "oos_train") for t, _ in d["oos_train"]]
    intents = sorted({l for _, l, _ in rows})
    opts = {c: ("out of scope: none of the supported intents" if c == "oos" else _slug(c)) for c in intents}
    order = list(range(len(rows)))
    random.Random(f"{SEED}|k151|order").shuffle(order)
    out, stats = [], collections.Counter()
    for i in order:
        if len(out) >= n_target:
            break
        t, l, split = rows[i]
        if not _take(reuse, "clinc150", str(i)):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(f"{SEED}|k151|{i}")
        q = {"type": "choice", "instructions": rng.choice(INTENT_INSTR["intent_general"]), "criteria": dict(opts)}
        rec = _record("clinc150", str(i), "k151", "intent_general", "CC-BY-3.0", {"user_request": t}, q, l,
                      {"lever": "clinc_k151", "k": len(opts)}, split=split)
        out.append(_view(rec, rng, styles=CLINC_STYLES))
        stats["oos" if l == "oos" else "in_scope"] += 1
    n_tr, n_oos = len(d["train"]), len(d["oos_train"])
    prov["clinc_k151"] = {"file": "clinc150/data_full.json (keys train + oos_train only)", "sha256": _sha_file(p),
                          "split": "train + oos_train", "n_train": n_tr, "n_oos_train": n_oos,
                          "oos_train_share": round(n_oos / (n_tr + n_oos), 5), "K": len(opts),
                          "rule": "uniform over train + oos_train rows: out-of-scope at its TRAIN share",
                          "n_target": n_target, "built": dict(stats)}
    log("clinc_k151", dict(stats))
    return out


def banking_k77(reuse, prov, n_target: int, d16: dict, log=print) -> list[dict]:
    p = SRC / "banking77" / "train.csv"
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    intents = sorted({r["category"] for r in rows})
    opts = {c: _slug(c) for c in intents}
    order = list(range(len(rows)))
    random.Random(f"{SEED}|k77|order").shuffle(order)
    out, stats = [], collections.Counter()
    for i in order:
        if len(out) >= n_target:
            break
        if not _take(reuse, "banking77", str(i)):
            stats["held_out_or_reuse_cap"] += 1
            continue
        rng = random.Random(f"{SEED}|k77|{i}")
        q = {"type": "choice", "instructions": rng.choice(INTENT_INSTR["intent_banking"]), "criteria": dict(opts)}
        rec = _record("banking77", str(i), "k77", "intent_banking", "CC-BY-4.0", {"customer_message": rows[i]["text"]},
                      q, rows[i]["category"], {"lever": "banking_k77", "k": len(opts)})
        out.append(_view(rec, rng, styles=K77_STYLES))
    stats["records"] = len(out)
    prov["banking_k77"] = {"file": "banking77/train.csv", "sha256": _sha_file(p), "split": "train",
                           "n_rows": len(rows), "K": len(opts), "rule": "uniform over train rows (natural label rates)",
                           "count_rule": d16, "n_target": n_target, "built": dict(stats)}
    log("banking_k77", dict(stats))
    return out


def parent_stats(parent: Path) -> dict:
    """Rows per source item (MAX_REUSE accounting) and family counts of the parent draw."""
    from .b1 import item_key
    reuse = collections.Counter()
    fam = collections.Counter()
    k77 = 0
    n = 0
    for r in _jsonl_gz(parent):
        n += 1
        pid = r.get("aug", {}).get("b1_view", {}).get("parent_id", r["id"])
        reuse[item_key(pid)] += 1
        fam[f"{r['source']}/{r['family']}"] += 1
        if r["source"] == "banking77" and r["family"] == "intent_banking":
            if any(q.get("type") == "choice" and len(q.get("criteria") or {}) == 77 for q in r["questions"].values()):
                k77 += 1
    return {"n": n, "reuse": reuse, "fam": fam, "banking_k77": k77}


def d16_k77() -> dict:
    """D16 = b1-draw100k-v2 minus aug.injection.family == d1a rows: its BANKING77 77-option row share."""
    p = config.BUILD_DIR / "b1-draw100k-v2" / "records.jsonl.gz"
    n = k77 = 0
    for r in _jsonl_gz(p):
        if ((r.get("aug") or {}).get("injection") or {}).get("family") == "d1a":
            continue
        n += 1
        if r["source"] == "banking77" and any(q.get("type") == "choice" and len(q.get("criteria") or {}) == 77
                                              for q in r["questions"].values()):
            k77 += 1
    return {"file": "ssdm-data/m2/build/b1-draw100k-v2/records.jsonl.gz minus d1a rows (D16 training draw)",
            "rows": n, "banking77_k77_rows": k77, "k77_row_share": round(k77 / n, 5)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="data.b_schema_levers")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--src-dir", required=True, help="dir with mmlu_aux.jsonl.gz / boolq.jsonl.gz parquet exports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clinc-k151", type=int, default=0, help="0: not built (K151 never fits 2,048 under v3w)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcd = Path(a.src_dir)
    from .b1 import exclusion_sets
    BLOCKED.update(exclusion_sets()["items"])
    ps = parent_stats(Path(a.parent))
    reuse = ps["reuse"]
    prov: dict = {}
    rows = []
    rows += apibank_long(reuse, prov)
    rows += mmlu_k10(srcd / "mmlu_aux.jsonl.gz", reuse, prov, n_target=ps["fam"]["mmlu_aux/mcq"])
    rows += boolq_yesno(srcd / "boolq.jsonl.gz", reuse, prov)
    if a.clinc_k151:
        rows += clinc_k151(reuse, prov, n_target=a.clinc_k151)
    else:
        prov["clinc_k151"] = {"built": 0, "reason": "first build: 0 of 3,000 K151 views fit 2,048 tokens under v3w (min 3,174 with digit ids, median 3,206; letters 3,445, option_k 3,810, keep 4,375, opaque 6,321)"}
    d16 = d16_k77()
    # K77 rows so the final draw's K77 share equals D16's: (have + x) / (N + x) = s  ->  x = (s N - have) / (1 - s)
    n_final = ps["n"] + len(rows)
    s = d16["k77_row_share"]
    x = max(0, math.ceil((s * n_final - ps["banking_k77"]) / (1 - s) * 1.10))  # +10 %; trimmed after the token check
    d16 = {**d16, "parent_k77_rows": ps["banking_k77"], "rows_before_k77": n_final,
           "formula": "generate ceil(1.10 * (share * N - parent_k77) / (1 - share)); keep (share * N' - parent_k77) / (1 - share) after the token check (N' = final non-K77 rows)", "x_generated": x}
    rows += banking_k77(reuse, prov, n_target=x, d16=d16)
    with gzip.open(out / "levers.jsonl.gz", "wt", encoding="utf-8") as fo:
        for r in rows:
            fo.write(canonical(r) + "\n")
    (out / "target_provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(rows), "by_lever": collections.Counter(r["aug"]["lever"]["lever"] for r in rows)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
