"""bschema-draw-jev2: TRAIN-ONLY rows for the 10 JevBench families that bschema-draw-jev1 does not cover.

jev1 (``data/jev1.py``) covers routing / adequacy / policy / intent / ordinal (severity) / extraction (enum) /
long_policy / state tracking. The pinned 231 public JevBench items (EVAL-ONLY) also carry the families multi_hop,
judge_hard, temporal_numeric, fact, tool_selection, probability, trap, ambiguous, tradeoff and adversarial. This module
builds rows for those ten, in exactly the jev1 row schema (``ssdm-train-v0``: ``state``, ``questions{q: {type,
instructions, criteria}}``, ``labels``; one question per row; option keys in a seeded per-row shuffle that the
serialiser must keep). Each family's task shape comes from jevkit's docs, the family NAME and the public items'
FORMAT only (question type, option count, state shape and length); no item text is read, copied or paraphrased.
FAMILY_SPEC.md (receipt dir) maps every family to its sources and generators.

Stage 1 (bschema-draw-jev2a): multi_hop, judge_hard, temporal_numeric, fact, tool_selection.
Stage 2 (bschema-draw-jev2b): probability, trap, ambiguous, tradeoff, adversarial.

Public TRAIN splits used: MuSiQue-Ans v1.0 train (multi_hop), GSM8K train (judge_hard / temporal_numeric, disjoint
hash partitions), BoolQ train (fact), Glaive function-calling v2 train (tool_selection). Everything else is a
deterministic generator written here with exact, rule-computed gold. Nothing here reads a JevBench item, an Index
DEV/TEST row, a held-out row or MD1 data.

Row compatibility with jev1: the per-row metadata lives under ``aug.jev1`` (``family8`` = the family, ``sub``,
``item_keys``) exactly as in jev1, plus ``draw: "jev2"``; ids end in ``~view:jev2-<view>`` (disjoint from jev1 ids).
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

from data.jev1 import choice_q, item_keys, noul_q, norm, pick, score_q, slug

SCHEMA = "ssdm-train-v0"
SEED = 20261011
STAGE_A = ("multi_hop", "judge_hard", "temporal_numeric", "fact", "tool_selection")
STAGE_B = ("probability", "trap", "ambiguous", "tradeoff", "adversarial")
FAMILIES = STAGE_A + STAGE_B
RAW = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\raw\jev2")
LIC = {"musique": "CC-BY-4.0 (MuSiQue v1.0, StonyBrookNLP; HF mirror bdsaglam/musique)", "gsm8k": "MIT",
       "boolq": "CC-BY-SA-3.0", "glaive_fc": "Apache-2.0", "jev2synth": "own generator (repo)"}
#: generator specs for harness.heldout.check_generator (fail-closed: each declares its task family)
GEN_SPECS = {
    "jev2_ref_chase": {"name": "jev2_ref_chase", "task_family": "register_archive_reference_chasing"},
    "jev2_judge_check": {"name": "jev2_judge_check", "task_family": "response_correctness_judging"},
    "jev2_temporal": {"name": "jev2_temporal", "task_family": "date_deadline_amount_rules"},
    "jev2_fact_record": {"name": "jev2_fact_record", "task_family": "explicit_fact_yes_no"},
    "jev2_tool_pick": {"name": "jev2_tool_pick", "task_family": "single_tool_pick"},
}


# ------------------------------------------------------------------------------------------------ helpers
def rng(*key: Any) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, (SEED,) + key)).encode()).hexdigest())


def part(key: str, n: int) -> int:
    """Deterministic label-blind partition of a source item into one of n pools."""
    return int(hashlib.sha256(f"jev2-part|{key}".encode()).hexdigest()[:8], 16) % n


def record(fam: str, sub: str, source: str, parent: str, view: str, state: Any, q: dict, label: Any,
           item: dict, split: str = "train", **meta) -> dict:
    """One ssdm-train-v0 row, jev1 layout. ``parent`` = the source item id (cohort / dedup identity), ``item`` = its
    item-tier texts (the Index fingerprint probe)."""
    if q["type"] == "noul":
        label = "true" if label in (True, "true") else "false"
    elif q["type"] == "score":
        label = str(int(label))
    return {"id": f"{parent}~view:jev2-{view}", "family": f"jev2_{sub}", "source": source,
            "source_id": parent.split(":", 2)[-1], "source_split": split, "license": LIC[source], "schema": SCHEMA,
            "state": state, "questions": {"q": q}, "labels": {"q": label}, "untrusted_fields": [],
            "aug": {"b1_view": {"base_id": parent, "kind": "schema", "parent_id": parent},
                    "jev1": {"family8": fam, "sub": sub, "draw": "jev2", "item_keys": item_keys(state, item), **meta}}}


def opts_from_texts(texts: list[str], prefix: str = "") -> tuple[dict[str, str], dict[str, str]]:
    """Option ids from the option texts (slug, made unique) -> ({id: text}, {text: id})."""
    out, back = {}, {}
    for t in texts:
        base = (prefix + slug(t))[:58] or "x"
        k, j = base, 2
        while k in out:
            k = f"{base[:55]}_{j}"
            j += 1
        out[k] = t
        back[t] = k
    return out, back


def uniq_norm(xs: list[str], banned: set[str] = frozenset()) -> list[str]:
    seen, out = set(banned), []
    for x in xs:
        n = norm(x)
        if n and n not in seen:
            seen.add(n)
            out.append(x)
    return out


TF = {"true": "yes", "false": "no"}


def yn(instr: str) -> dict:
    return noul_q(instr, "yes", "no")


def money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_date(d: dt.date, style: int = 0) -> str:
    if style == 1:
        return d.strftime("%d %B %Y").lstrip("0")
    if style == 2:
        return d.strftime("%B %d, %Y").replace(" 0", " ")
    return d.isoformat()


# ================================================================================================ 1. multi_hop
SYL = ["ka", "lo", "mi", "ra", "ven", "tor", "sel", "dun", "pra", "quel", "bri", "zan", "hol", "mer", "fen", "gast",
       "ul", "or", "tis", "nar", "vel", "cor", "ash", "lin", "bex", "dor", "sim", "pel", "rok", "tha", "wen", "jas"]
ASSET_KINDS = ["relay", "gateway", "ledger service", "build runner", "sensor array", "billing worker", "tile cache",
               "payment router", "search shard", "backup vault", "label printer", "cold-storage unit", "kiosk fleet",
               "scheduler", "message broker", "telemetry hub", "badge reader", "forklift pool", "edge cluster"]
TEAM_WORDS = ["Ops", "Platform", "Desk", "Field Unit", "Reliability", "Infra", "Support Cell", "Core", "Logistics",
              "Systems", "Workshop", "Services"]
CITIES = ["Lisbon", "Porto", "Lyon", "Turin", "Gdansk", "Leeds", "Cork", "Malmo", "Tartu", "Brno", "Graz", "Bilbao",
          "Ghent", "Utrecht", "Aarhus", "Bergen", "Tampere", "Riga", "Ljubljana", "Split", "Plovdiv", "Cluj", "Basel",
          "Bremen", "Leipzig", "Nantes", "Seville", "Bologna", "Kaunas", "Oulu", "Dundee", "Galway", "Austin",
          "Halifax", "Denver", "Raleigh", "Tucson", "Spokane", "Omaha", "Calgary"]
FIRST = ["Ana", "Bram", "Chen", "Dalia", "Eitan", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kemi", "Lars", "Mira", "Nuno",
         "Oona", "Pavel", "Rhea", "Sami", "Tomas", "Uma", "Vera", "Wim", "Yara", "Zeno", "Ines", "Kofi", "Leila", "Marek"]
LAST = ["Abara", "Brandt", "Costa", "Duval", "Eriksen", "Ferro", "Gallo", "Horvat", "Ibsen", "Jansen", "Kowal", "Lindqvist",
        "Moreau", "Novak", "Okafor", "Petrov", "Quinn", "Rossi", "Sato", "Toivonen", "Ueda", "Varga", "Weber", "Zubiri"]
NOISE = ["Quarterly access review for {t} completed with no findings.", "Spare parts budget for {a} approved.",
         "{t} published its on-call rota for the next quarter.", "Firmware audit of {a} scheduled.",
         "Site walk-through at {s} rescheduled.", "Naming convention for asset codes reaffirmed.",
         "{t} requested extra monitoring on {a}; request noted, no ownership change.",
         "Insurance certificate for site {s} renewed.", "Training session on incident handover held for {t}."]


def _pname(r: random.Random, used: set) -> str:
    while True:
        n = "".join(r.choice(SYL) for _ in range(r.randint(2, 3))).capitalize()
        if n not in used:
            used.add(n)
            return n


def _world(r: random.Random) -> dict:
    used: set = set()
    nt, ns, na = r.randint(5, 8), r.randint(4, 6), r.randint(7, 11)
    codes = r.sample(range(10, 99), nt)
    people = r.sample([f"{f} {l}" for f in FIRST for l in LAST], nt * 2)
    teams = {f"T-{c}": {"name": f"{_pname(r, used)} {r.choice(TEAM_WORDS)}", "contact": people[j]}
             for j, c in enumerate(codes)}
    spare_people = people[nt:]
    cities = r.sample(CITIES, ns)
    sites = {}
    for c in cities:
        k = f"S{r.randint(1, 9)}{r.choice('ABCDEFGHJK')}"
        while k in sites:
            k = f"S{r.randint(1, 9)}{r.choice('ABCDEFGHJK')}"
        sites[k] = c
    acodes: set = set()
    assets = {}
    tk, sk = sorted(teams), sorted(sites)
    for _ in range(na):
        c = f"{r.choice(['AX', 'BK', 'CQ', 'DM', 'EV', 'FR', 'GL'])}-{r.randint(1000, 9999)}"
        while c in acodes:
            c = f"{r.choice(['AX', 'BK', 'CQ', 'DM', 'EV', 'FR', 'GL'])}-{r.randint(1000, 9999)}"
        acodes.add(c)
        assets[c] = {"name": f"{_pname(r, used)} {r.choice(ASSET_KINDS)}", "owner": r.choice(tk), "site": r.choice(sk)}
    return {"teams": teams, "sites": sites, "assets": assets, "spare_people": spare_people}


def _ref_chase(i: int) -> dict:
    r = rng("ref-chase", i)
    w = _world(r)
    teams, sites, assets = w["teams"], w["sites"], w["assets"]
    ac, tk, sk = sorted(assets), sorted(teams), sorted(sites)
    t0 = dt.date(2024, 1, 1) + dt.timedelta(days=r.randint(0, 300))
    day = lambda k: t0 + dt.timedelta(days=k)  # noqa: E731
    events: list[tuple[dt.date, str, dict]] = []  # (date, kind, payload)
    # replacements: old -> new (old retired)
    r.shuffle(ac)
    n_rep = r.randint(2, 3)
    repl = {ac[2 * j]: ac[2 * j + 1] for j in range(n_rep)}
    for old, new in repl.items():
        events.append((day(r.randint(5, 120)), "replace", {"old": old, "new": new}))
    # ownership moves (some cancelled by a footnote), relocations, contact amendment
    mv_assets = list(repl.values()) + r.sample(ac, 2)
    for a in mv_assets:
        for _ in range(r.randint(0, 2)):
            events.append((day(r.randint(130, 400)), "move", {"asset": a, "to": r.choice(tk), "cancel": r.random() < 0.35}))
    for a in r.sample(list(repl.values()), r.randint(1, len(repl))):
        if r.random() < 0.6:
            events.append((day(r.randint(130, 400)), "relocate", {"asset": a, "to": r.choice(sk)}))
    for _ in range(r.randint(3, 6)):
        events.append((day(r.randint(1, 420)), "noise", {"t": r.choice(tk), "a": r.choice(ac), "s": r.choice(sk)}))
    events.sort(key=lambda e: (e[0], e[1], json.dumps(e[2], sort_keys=True)))
    # apply
    owner = {a: assets[a]["owner"] for a in assets}
    site = {a: assets[a]["site"] for a in assets}
    history: dict[str, list[str]] = {a: [owner[a]] for a in assets}
    n_moves = {a: 0 for a in assets}
    archive, foot = [], {}
    for d, kind, p in events:
        ds = d.isoformat()
        if kind == "replace":
            o, n = p["old"], p["new"]
            ref = assets[o]["name"] if r.random() < 0.5 else o
            archive.append(f"{ds}: {ref} was retired; its workload moved to {n}.")
        elif kind == "move":
            a, to = p["asset"], p["to"]
            if to == owner[a]:
                continue
            fn = None
            if p["cancel"]:
                fn = str(len(foot) + 1)
                cd = d + dt.timedelta(days=r.randint(2, 20))
                foot[fn] = (f"The transfer in this entry was cancelled on {cd.isoformat()} before it took effect; "
                            f"{a} stayed with {owner[a]}.")
            archive.append(f"{ds}: Ownership of {a} transferred from {owner[a]} to {to}." + (f" [{fn}]" if fn else ""))
            if not p["cancel"]:
                owner[a] = to
                history[a].append(to)
                n_moves[a] += 1
        elif kind == "relocate":
            a, to = p["asset"], p["to"]
            if to == site[a]:
                continue
            archive.append(f"{ds}: {a} was physically moved from site {site[a]} to site {to}.")
            site[a] = to
        else:
            archive.append(f"{ds}: " + pick(r, NOISE).format(**p))
    contact = {t: teams[t]["contact"] for t in teams}
    amend_team = r.choice(tk)
    new_contact = w["spare_people"][0]
    as_of = day(430)
    amendment = None
    if r.random() < 0.6:
        amendment = (f"Amendment effective {day(425).isoformat()}: the escalation contact for {amend_team} is now "
                     f"{new_contact}; earlier contact listings for that team are superseded.")
        contact[amend_team] = new_contact
    state = {"as_of": as_of.isoformat(),
             "asset_register": [{"code": a, "name": assets[a]["name"], "owner_team_at_registration": assets[a]["owner"],
                                 "site_at_registration": assets[a]["site"]} for a in sorted(assets)],
             "teams": {t: {"name": teams[t]["name"], "escalation_contact": teams[t]["contact"]} for t in tk},
             "sites": dict(sorted(sites.items())), "archive": archive}
    if foot:
        state["footnotes"] = foot
    if amendment:
        state["amendment"] = amendment
    parent = f"m2:jev2synth:ref_chase:{i}"
    old = r.choice(sorted(repl))
    new = repl[old]
    oname = assets[old]["name"]
    kind = r.random()
    K = r.choice([4, 4, 5, 6])
    if kind < 0.1:  # score: count effective ownership changes
        a = r.choice(mv_assets)
        levels = ["none", "exactly one", "exactly two", "three or more"]
        q = score_q(f"Up to the as-of date, how many ownership transfers of {assets[a]['name']} actually took effect? "
                    "Cancelled transfers do not count.", levels)
        return record("multi_hop", "ref_chase_count", "jev2synth", parent, "count", state, q, min(3, n_moves[a]),
                      {"archive": archive}, split="synthetic", gen="jev2_ref_chase")
    if kind < 0.5:  # owner team of the replacement
        gold_t = owner[new]
        wrong = [owner[old]] + history[new][:-1] + [owner[a] for a in assets if a not in (old, new)]
        names = uniq_norm([teams[gold_t]["name"]] + [teams[t]["name"] for t in wrong] +
                          [teams[t]["name"] for t in r.sample(tk, len(tk))])
        gold_txt, instr = teams[gold_t]["name"], pick(r, [
            f"Which team owns the asset that took over the workload of {oname}, as of the as-of date?",
            f"{oname} was retired. Which team is responsible for its replacement now?",
            f"Follow the records: who currently owns the replacement for {oname}?"])
        sub = "ref_chase_owner"
    elif kind < 0.78:  # city of the replacement
        gold_s = site[new]
        wrong = [site[old], assets[new]["site"]] + [s for s in sk]
        names = uniq_norm([sites[gold_s]] + [sites[s] for s in wrong])
        gold_txt, instr = sites[gold_s], pick(r, [
            f"In which city is the replacement for {oname} located as of the as-of date?",
            f"Where (city) does the asset that replaced {oname} sit now?"])
        sub = "ref_chase_city"
    else:  # escalation contact of the current owner of an asset
        a = r.choice(mv_assets)
        t = owner[a]
        gold_txt = contact[t]
        wrong = [teams[t]["contact"]] + [contact[x] for x in history[a]] + [contact[assets[a]["owner"]]] + \
            [contact[x] for x in r.sample(tk, len(tk))]
        names = uniq_norm([gold_txt] + wrong)
        instr = pick(r, [f"Who is the escalation contact for the team that owns {assets[a]['name']} as of the as-of "
                         "date?", f"An issue hits {a}. Whom should it be escalated to, per the records?"])
        sub = "ref_chase_contact"
    names = names[:K]
    if r.random() < 0.15:
        cand = gold_txt if r.random() < 0.5 else (names[1] if len(names) > 1 else gold_txt)
        q = yn(f"{instr} Is the answer {cand}?")
        return record("multi_hop", sub + "_yn", "jev2synth", parent, "yn", state, q, cand == gold_txt,
                      {"archive": archive}, split="synthetic", gen="jev2_ref_chase")
    opts, back = opts_from_texts(names)
    q, g = choice_q(instr, opts, back[gold_txt], r)
    return record("multi_hop", sub, "jev2synth", parent, "chase", state, q, g, {"archive": archive},
                  split="synthetic", gen="jev2_ref_chase")


def _is_num(s: str) -> bool:
    return bool(re.fullmatch(r"-?[\d,]+(\.\d+)?", s.strip()))


def musique_rows(n: int) -> Iterator[dict]:
    """MuSiQue-Ans v1.0 TRAIN (streamed line by line; hash-selected 1/5, first n eligible)."""
    p = RAW / "musique" / "musique_ans_v1.0_train.jsonl"
    pool: dict[str, list[str]] = {"num": [], "txt": []}
    k = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if k >= n:
                return
            d = json.loads(line)
            typ = "num" if _is_num(d["answer"]) else "txt"
            pool[typ] = (pool[typ] + [d["answer"]])[-200:]
            if not d.get("answerable", True) or part("musique:" + d["id"], 5):
                continue
            sup = [x for x in d["paragraphs"] if x["is_supporting"]]
            if sum(len(x["paragraph_text"]) for x in sup) > 3200:
                continue
            row = _musique_row(d, sup, pool[typ])
            if row:
                k += 1
                yield row


def _musique_row(d: dict, sup: list[dict], pool: list[str]) -> dict | None:
    r = rng("musique", d["id"])
    gold = d["answer"].strip()
    banned = {norm(a) for a in d.get("answer_aliases") or []}
    inter = [s["answer"] for s in d["question_decomposition"][:-1]]
    others = [x for x in d["paragraphs"] if not x["is_supporting"]]
    dist = r.sample(others, min(2, len(others)))
    docs = [{"title": x["title"], "text": x["paragraph_text"]} for x in sup + dist]
    r.shuffle(docs)
    cands = list(inter)
    if _is_num(gold):
        g = int(float(gold.replace(",", ""))) if re.fullmatch(r"-?[\d,]+", gold) else None
        if g is not None:
            cands += [str(g + dlt) for dlt in r.sample([-12, -7, -3, -2, -1, 1, 2, 3, 5, 10], 6)]
    pl = [x for x in pool if norm(x) != norm(gold)]
    cands += r.sample(pl, min(6, len(pl))) + [x["title"] for x in dist]
    K = r.choice([4, 4, 5])
    names = uniq_norm([gold] + cands, banned - {norm(gold)})
    if len(names) < 4 or names[0] != gold:
        return None
    names = names[:K]
    opts, back = opts_from_texts(names)
    q, g = choice_q(pick(r, ["Using only the documents, answer: ", "Follow the documents step by step. ",
                             "Combine the documents to answer: "]) + d["question"], opts, back[gold], r)
    return record("multi_hop", "musique", "musique", f"m2:musique:{d['id']}", "mh", {"documents": docs}, q, g,
                  {"documents": docs}, hops=len(d["question_decomposition"]))


def gen_multi_hop(n_synth: int, n_musique: int) -> Iterator[dict]:
    for i in range(n_synth):
        yield _ref_chase(i)
    yield from musique_rows(n_musique)


# ================================================================================================ 2. judge_hard
def gsm8k_rows() -> list[dict]:
    import pyarrow.parquet as pq
    from data import sources as S
    return pq.read_table(S.verified_path("gsm8k", "main/train-00000-of-00001.parquet")).to_pylist()


_ANN = re.compile(r"<<[^<>]*>>")


def _num_fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _gsm_wrong(g: float, r: random.Random) -> str:
    from data.convert import gsm8k_distractors
    return gsm8k_distractors(g, 1, r)[0]


def gen_gsm_judge(rows: list[dict], n: int) -> Iterator[dict]:
    k = 0
    for i, x in enumerate(rows):
        if k >= n:
            return
        if part(f"gsm8k:{i}", 2) != 0:  # partition 0 -> judge_hard, 1 -> temporal_numeric
            continue
        ans = x["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(ans)
        except ValueError:
            continue
        body = x["answer"].split("####")[0].strip()
        lines = [_ANN.sub("", l).strip() for l in body.split("\n") if l.strip()]
        r = rng("gsm-judge", i)
        mode = r.choices(["clean", "final", "step"], [0.5, 0.25, 0.25])[0]
        final = _num_fmt(g)
        if mode == "final":
            w = _gsm_wrong(g, r)
            last = lines[-1]
            pat = re.compile(r"(?<![\d.])" + re.escape(final) + r"(?![\d.])")
            if pat.search(last):  # replace only the LAST occurrence of the true result in the last step
                ms = list(pat.finditer(last))
                m = ms[-1]
                lines[-1] = last[:m.start()] + w + last[m.end():]
            final = w
        elif mode == "step":
            cand = [j for j, l in enumerate(lines[:-1]) if re.search(r"=\s*\$?-?\d", l)]
            if not cand:
                mode = "clean"
            else:
                j = r.choice(cand)
                m = re.search(r"(=\s*\$?)(-?\d[\d,]*(?:\.\d+)?)", lines[j])
                v = float(m.group(2).replace(",", ""))
                w = _gsm_wrong(v, r)
                lines[j] = lines[j][:m.start(2)] + w + lines[j][m.end(2):]
        resp = "\n".join(lines) + f"\nFinal answer: {final}"
        if mode == "step" and r.random() < 0.6 or mode == "clean" and r.random() < 0.3:
            q = noul_q(pick(r, ["Check the working: is every arithmetic step in the response correct?",
                                "Judge the solution's arithmetic. Is each calculation step right?"]),
                       "every calculation in the response is correct", "at least one calculation is wrong")
            gold = mode == "clean"
            view = "judge-steps"
        else:
            q = noul_q(pick(r, ["Is the response's final answer correct for the problem?",
                                "Judge the saved answer: does the response reach the correct final answer?",
                                "Grade the response. Is its final answer right?"]),
                       "the final answer is correct", "the final answer is wrong")
            gold = mode != "final"
            view = "judge-final"
        k += 1
        yield record("judge_hard", f"gsm_{view.split('-')[1]}", "gsm8k", f"m2:gsm8k:{i}", view,
                     {"request": x["question"], "response": resp}, q, gold, {"request": x["question"]},
                     corruption=mode)


UNITS = [("km", "miles", lambda x: x * 0.621371, 1.0), ("miles", "km", lambda x: x * 1.609344, 1.0),
         ("kg", "pounds", lambda x: x * 2.204623, 1.0), ("pounds", "kg", lambda x: x * 0.453592, 1.0),
         ("degrees Celsius", "degrees Fahrenheit", lambda x: x * 9 / 5 + 32, 0.0),
         ("degrees Fahrenheit", "degrees Celsius", lambda x: (x - 32) * 5 / 9, 0.0),
         ("litres", "US gallons", lambda x: x * 0.264172, 1.0), ("inches", "centimetres", lambda x: x * 2.54, 1.0),
         ("metres", "feet", lambda x: x * 3.28084, 1.0)]
TOPICS = ["the new office opening", "a pricing change", "the spring sale", "a maintenance window", "the quarterly "
          "results call", "a product recall", "the team offsite", "the holiday schedule", "a security update",
          "the loyalty programme", "a delivery delay", "the beta launch"]
SUBJ = {"head": ["Update on", "Heads-up about", "Details of", "Notice:", "Reminder about", "News on"],
        "tail": ["for all customers", "this week", "and what changes", "starting Monday", "for your account", ""]}
NOTE_NAMES = ["Priya Raman", "Tom Becker", "Lucia Ortega", "Ken Mori", "Ada Nwosu", "Sven Holm", "Rosa Lind"]


def _judge_synth(i: int) -> dict:
    r = rng("judge-synth", i)
    kind = r.choice(["units", "date", "weekday", "sort", "subject", "json", "price", "sum", "letters"])
    bad = r.random() < 0.5
    flaw = None
    if kind == "units":
        u1, u2, f, _ = r.choice(UNITS)
        x = r.choice([r.randint(2, 400), round(r.uniform(1, 100), 1)])
        d = r.choice([1, 2])
        y = round(f(x), d)
        req = f"Convert {x} {u1} to {u2}, rounded to {d} decimal place{'s' if d > 1 else ''}."
        if bad:
            flaw = r.choice(["factor", "digit", "round"])
            if flaw == "factor":
                yw = round(f(x) * r.choice([0.9, 1.1, 1.25]), d)
            elif flaw == "digit":
                yw = round(y + r.choice([-1, 1]) * r.choice([1, 10]) * (10 ** -d) * r.randint(2, 9), d)
            else:
                yw = round(y + r.choice([-1, 1]) * 10 ** -d, d)
            if yw == y:
                yw = round(y + 10 ** -d * 3, d)
            y = yw
        resp = pick(r, [f"{x} {u1} is {y:.{d}f} {u2}.", f"That comes to {y:.{d}f} {u2}.",
                        f"Answer: {y:.{d}f} {u2}"])
    elif kind in ("date", "weekday"):
        base = dt.date(2025, 1, 1) + dt.timedelta(days=r.randint(0, 900))
        if kind == "date":
            n = r.randint(3, 200)
            ans = base + dt.timedelta(days=n)
            req = f"What is the date {n} days after {fmt_date(base, 2)}?"
            if bad:
                flaw = r.choice(["off_by_one", "month"])
                ans = ans + dt.timedelta(days=r.choice([-1, 1])) if flaw == "off_by_one" else \
                    ans + dt.timedelta(days=r.choice([-30, 31, 28]))
            resp = pick(r, [f"It is {fmt_date(ans, 2)}.", f"{n} days later is {fmt_date(ans, 1)}.",
                            f"The date will be {fmt_date(ans, 0)}."])
        else:
            req = f"Which day of the week is {fmt_date(base, 2)}?"
            wd = base.strftime("%A")
            if bad:
                flaw = "weekday"
                wd = (base + dt.timedelta(days=r.choice([1, 2, -1, 3]))).strftime("%A")
            resp = pick(r, [f"{fmt_date(base, 2)} is a {wd}.", f"It falls on a {wd}."])
    elif kind == "sort":
        items = r.sample(["lamp", "kettle", "chair", "rug", "mirror", "shelf", "clock", "vase", "desk", "fan", "stool",
                          "blender", "toaster", "pillow"], r.randint(4, 7))
        prices = r.sample(range(8, 400), len(items))
        pl = list(zip(items, prices))
        asc = r.random() < 0.6
        srt = sorted(pl, key=lambda t: t[1], reverse=not asc)
        req = (f"Sort these items by price, {'lowest' if asc else 'highest'} first: " +
               ", ".join(f"{a} ${p}" for a, p in pl) + ".")
        if bad:
            flaw = "swap"
            j = r.randrange(len(srt) - 1)
            srt[j], srt[j + 1] = srt[j + 1], srt[j]
        resp = ", ".join(f"{a} (${p})" for a, p in srt)
    elif kind == "subject":
        topic, kw, mx = r.choice(TOPICS), r.choice(["update", "today", "important", "schedule", "details"]), r.randint(6, 10)
        req = (f"Write an email subject line about {topic}. Requirements: at most {mx} words, include the word "
               f"'{kw}', and no exclamation marks.")
        words = f"{pick(r, SUBJ['head'])} {topic} {pick(r, SUBJ['tail'])}".split()
        if kw not in [norm(x) for x in words]:
            words.insert(r.randrange(len(words) + 1), kw)
        while len(words) > mx:
            words.pop(r.randrange(len(words)))
            if kw not in [norm(x) for x in words]:
                words.append(kw)
        if bad:
            flaw = r.choice(["too_long", "missing_word", "exclamation"])
            if flaw == "too_long":
                words += ["for", "everyone", "on", "our", "list", "please", "read"][: mx - len(words) + r.randint(1, 3)]
            elif flaw == "missing_word":
                words = [x for x in words if norm(x) != kw]
            else:
                words[-1] += "!"
        resp = " ".join(words)
        resp = resp[0].upper() + resp[1:]
        # recompute the gold from the rules (exact): word count, keyword, no '!'
        ok = len(resp.split()) <= mx and kw in [norm(x) for x in resp.split()] and "!" not in resp
        bad = not ok
    elif kind == "json":
        nm, city, tot = r.choice(NOTE_NAMES), r.choice(CITIES), round(r.uniform(10, 900), 2)
        note = pick(r, [f"Order from {nm} in {city}; total due {money(tot)}.",
                        f"{nm} ({city}) placed an order totalling {money(tot)}.",
                        f"Ship to {city}. Customer: {nm}. Amount: {money(tot)}."])
        req = f"Extract the customer name, city and order total from this note as JSON with keys name, city, total (a number): \"{note}\""
        obj = {"name": nm, "city": city, "total": tot}
        if bad:
            flaw = r.choice(["value", "missing_key", "invalid"])
            if flaw == "value":
                obj["total"] = round(tot + r.choice([-10, 10, 1, -1]), 2)
            elif flaw == "missing_key":
                obj.pop(r.choice(["city", "total"]))
        resp = json.dumps(obj)
        if bad and flaw == "invalid":
            resp = resp[:-1] + ",}"
    elif kind == "price":
        p, d, t = r.choice([19.99, 45, 80, 120, 249.5, 64]), r.choice([10, 15, 20, 25, 30]), r.choice([5, 7, 8, 10])
        ok_v = round(p * (1 - d / 100) * (1 + t / 100), 2)
        req = f"An item costs {money(p)}. It is discounted {d}% and then {t}% sales tax is added. What is the final price?"
        v = ok_v
        if bad:
            flaw = r.choice(["net_percent", "tax_on_list"])
            v = round(p * (1 - (d - t) / 100), 2) if flaw == "net_percent" else round(p * (1 - d / 100) + p * t / 100, 2)
            if v == ok_v:
                v = round(ok_v + 1, 2)
        resp = pick(r, [f"The final price is {money(v)}.", f"After discount and tax: {money(v)}.", f"{money(v)}"])
    elif kind == "sum":
        lines = [(pick(r, ["design", "hosting", "support", "licences", "travel", "printing", "training", "audit"]),
                  r.randint(20, 2000)) for _ in range(r.randint(3, 6))]
        tot = sum(v for _, v in lines)
        req = "Add up these invoice lines and give the total: " + "; ".join(f"{a} ${v}" for a, v in lines) + "."
        if bad:
            flaw = r.choice(["omit", "carry"])
            tot = tot - lines[r.randrange(len(lines))][1] if flaw == "omit" else tot + r.choice([-100, 100, 10, -10])
        resp = pick(r, [f"The total is ${tot:,}.", f"Total: ${tot}", f"These lines sum to ${tot:,}."])
    else:
        word = r.choice(["strawberry", "bookkeeper", "committee", "mississippi", "parallel", "assessment", "balloon",
                         "occurrence", "reservoir", "tomorrow", "necessary", "embarrass", "possession", "letterbox"])
        ch = r.choice(sorted(set(word)))
        c = word.count(ch)
        req = f"How many times does the letter '{ch}' appear in the word '{word}'?"
        if bad:
            flaw = "count"
            c = c + r.choice([1, -1]) if c > 1 else c + 1
        resp = pick(r, [f"The letter '{ch}' appears {c} time{'s' if c != 1 else ''}.", f"{c}"])
    q = noul_q(pick(r, ["Does the response correctly and completely satisfy the request?",
                        "Judge the response: is it correct and does it meet every requirement in the request?",
                        "Is this response acceptable, i.e. fully correct for what was asked?"]),
               "the response is correct and meets every requirement", "the response has an error or misses a requirement")
    if not bad:
        flaw = None
    return record("judge_hard", f"judge_{kind}", "jev2synth", f"m2:jev2synth:judge_check:{i}", "judge",
                  {"request": req, "response": resp}, q, not bad, {"request": req, "response": resp},
                  split="synthetic", gen="jev2_judge_check", flaw=flaw)


def gen_judge(n_gsm: int, n_synth: int, gsm: list[dict]) -> Iterator[dict]:
    yield from gen_gsm_judge(gsm, n_gsm)
    for i in range(n_synth):
        yield _judge_synth(i)


# ================================================================================================ 3. temporal_numeric
HOLI_NAMES = ["founders day", "spring holiday", "summer bank holiday", "harvest day", "winter closure"]


def _bizdays(start: dt.date, n: int, hol: set) -> dt.date:
    d = start
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            k += 1
    return d


def _date_opts(gold: dt.date, wrong: list[dt.date], r: random.Random, K: int, style: int):
    names = uniq_norm([fmt_date(gold, style)] + [fmt_date(x, style) for x in wrong] +
                      [fmt_date(gold + dt.timedelta(days=x), style) for x in r.sample([-3, -2, -1, 1, 2, 3, 4, 7], 8)])
    names = names[:K]
    opts, back = opts_from_texts(names, "d_")
    return opts, back[fmt_date(gold, style)]


def _temporal(i: int) -> dict:
    r = rng("temporal", i)
    kind = r.choice(["sla", "return", "warranty", "late_fee", "prorate", "tz", "tenure", "days_late"])
    parent = f"m2:jev2synth:temporal:{i}"
    style = r.choice([0, 1, 2])
    K = r.choice([3, 4, 5, 5, 6])
    base = dt.date(2025, 1, 1) + dt.timedelta(days=r.randint(0, 700))

    def out(sub, state, q, g, **m):
        return record("temporal_numeric", f"temporal_{sub}", "jev2synth", parent, sub, state, q, g,
                      {"event": state["event"]} if isinstance(state, dict) else {}, split="synthetic",
                      gen="jev2_temporal", **m)

    if kind == "sla":
        n = r.randint(2, 7)
        cutoff = r.choice([16, 17, 18])
        hol = sorted({base + dt.timedelta(days=r.randint(1, 12)) for _ in range(r.randint(0, 2))})
        hol = [h for h in hol if h.weekday() < 5]
        hh = r.randint(7, 21)
        rules = (f"First response is due within {n} business days. Business days are Monday to Friday, excluding the "
                 f"company holidays listed. A ticket opened at or after {cutoff}:00 counts as opened on the next "
                 f"business day." + (" Holidays: " + "; ".join(f"{fmt_date(h, style)} ({r.choice(HOLI_NAMES)})"
                                                                 for h in hol) + "." if hol else ""))
        event = f"Ticket opened on {base.strftime('%A')} {fmt_date(base, style)} at {hh:02d}:{r.choice([0, 15, 30, 45]):02d}."
        hs = set(hol)
        # opened after the cutoff, on a weekend or on a holiday -> counts as opened on the next business day
        start_eff = _bizdays(base, 1, hs) if (hh >= cutoff or base.weekday() >= 5 or base in hs) else base
        gold = _bizdays(start_eff, n, hs)
        wrong = [base + dt.timedelta(days=n), _bizdays(base, n, set()), _bizdays(base, n, hs), _bizdays(start_eff, n, set())]
        state = {"rules": rules, "event": event}
        if r.random() < 0.2:
            cand = r.choice([gold] + [x for x in wrong if x != gold])
            return out("sla_yn", state, yn(f"Is the first response due on {fmt_date(cand, style)}?"), cand == gold)
        opts, g = _date_opts(gold, wrong, r, K, style)
        q, g = choice_q("By which date is the first response due?", opts, g, r)
        return out("sla", state, q, g)
    if kind == "return":
        n_std, n_sale = r.choice([14, 30, 30, 60]), r.choice([7, 10, 14])
        sale = r.random() < 0.4
        n = n_sale if sale else n_std
        deliv = base
        last = deliv + dt.timedelta(days=n)
        req = last + dt.timedelta(days=r.choice([-5, -1, 0, 0, 1, 2, 6]))
        rules = (f"Items may be returned within {n_std} days of delivery; sale items within {n_sale} days. The "
                 f"delivery day is day 0, so the last valid day is delivery date plus the window.")
        event = (f"{'Sale item' if sale else 'Item'} delivered {fmt_date(deliv, style)}; return requested "
                 f"{fmt_date(req, style)}.")
        state = {"rules": rules, "event": event}
        if r.random() < 0.6:
            return out("return_yn", state, yn("Is this return request within the allowed window?"), req <= last)
        wrong = [deliv + dt.timedelta(days=(n_std if sale else n_sale)), last - dt.timedelta(days=1)]
        opts, g = _date_opts(last, wrong, r, K, style)
        q, g = choice_q("What is the last day on which this item can be returned?", opts, g, r)
        return out("return", state, q, g)
    if kind == "warranty":
        months = r.choice([6, 12, 18, 24, 36])
        d0 = dt.date(r.choice([2023, 2024, 2025]), r.randint(1, 12), r.randint(1, 28))
        if r.random() < 0.35:  # month-end start to exercise the rule
            m = r.randint(1, 12)
            y = r.choice([2023, 2024, 2025])
            d0 = (dt.date(y + (m // 12), m % 12 + 1, 1) - dt.timedelta(days=1))
        y, m = divmod(d0.month - 1 + months, 12)
        y += d0.year
        m += 1
        import calendar
        last_day = calendar.monthrange(y, m)[1]
        gold = dt.date(y, m, min(d0.day, last_day))
        rules = (f"The warranty runs {months} months from the purchase date and ends on the same day of the month; "
                 f"if that month has no such day, it ends on the last day of that month.")
        event = f"Purchased on {fmt_date(d0, style)}."
        state = {"rules": rules, "event": event}
        wrong = [d0 + dt.timedelta(days=round(months * 30.4)), d0 + dt.timedelta(days=months * 30),
                 gold - dt.timedelta(days=1), gold + dt.timedelta(days=1)]
        opts, g = _date_opts(gold, wrong, r, K, style)
        q, g = choice_q("On which date does the warranty end?", opts, g, r)
        return out("warranty", state, q, g)
    if kind == "late_fee":
        amt = r.choice([400, 750, 1200, 1800, 2500, 3200])
        pct = r.choice([1, 1.5, 2])
        per = r.choice([15, 30])
        cap = r.choice([3, 4, 6])
        late = r.choice([0, r.randint(1, 200)])
        periods = 0 if late <= 0 else min(cap, -(-late // per))
        fee = round(amt * pct / 100 * periods, 2)
        due = base
        paid = due + dt.timedelta(days=late)
        rules = (f"Invoices are due on the due date. A late fee of {pct}% of the invoice amount applies for each "
                 f"started {per}-day period after the due date, up to {cap} periods. Payment on the due date is on time.")
        event = f"Invoice of {money(amt)} due {fmt_date(due, style)}; paid {fmt_date(paid, style)}."
        state = {"rules": rules, "event": event}
        alt = {round(amt * pct / 100 * p, 2) for p in range(0, cap + 3)} | {round(amt * pct / 100 * (late // per), 2),
                                                                             round(amt * pct / 100 * late / per, 2)}
        names = uniq_norm([money(fee)] + [money(x) for x in sorted(alt, key=lambda x: abs(x - fee))[1:]])
        names = names[:K]
        opts, back = opts_from_texts(names, "fee_")
        q, g = choice_q("What late fee is owed?", opts, back[money(fee)], r)
        return out("late_fee", state, q, g)
    if kind == "prorate":
        old, new = sorted(r.sample([9, 15, 20, 29, 45, 60, 99], 2))
        cyc = 30
        used = r.randint(1, 29)
        rem = cyc - used
        charge = round((new - old) * rem / cyc, 2)
        rules = (f"Plans bill in {cyc}-day cycles. An upgrade takes effect immediately and is charged the price "
                 f"difference times the share of the cycle remaining (remaining days / {cyc}), rounded to the cent.")
        event = (f"Customer on the {money(old)} plan upgrades to the {money(new)} plan after {used} days of the current "
                 f"cycle have been used.")
        state = {"rules": rules, "event": event}
        alt = [round((new - old) * used / cyc, 2), round(new * rem / cyc, 2), round(new - old, 2),
               round((new - old) * (rem + 1) / cyc, 2), round((new - old) * (rem - 1) / cyc, 2)]
        names = uniq_norm([money(charge)] + [money(x) for x in alt])[:K]
        opts, back = opts_from_texts(names, "amt_")
        q, g = choice_q("How much is charged for the upgrade?", opts, back[money(charge)], r)
        return out("prorate", state, q, g)
    if kind == "tz":
        zones = {"UTC-8": -8, "UTC-5": -5, "UTC-3": -3, "UTC+0": 0, "UTC+1": 1, "UTC+2": 2, "UTC+3": 3,
                 "UTC+5:30": 5.5, "UTC+8": 8, "UTC+9": 9, "UTC+10": 10}
        a, b = r.sample(sorted(zones), 2)
        used: set = set()  # invented office names: real cities would clash with their real time zones
        ca, cb = f"the {_pname(r, used)} office", f"the {_pname(r, used)} office"
        h, mi = r.randint(0, 23), r.choice([0, 30])
        t0 = dt.datetime(2026, 3, 10, h, mi)
        tb = t0 + dt.timedelta(hours=zones[b] - zones[a])
        rules = f"{ca} uses {a} and {cb} uses {b} all year (no daylight-saving changes)."
        event = f"A call is booked for {t0.strftime('%H:%M')} on 10 March 2026, {ca} time."
        state = {"rules": rules, "event": event}

        def tf(x):
            return x.strftime("%H:%M on %d March").replace(" 0", " ")
        wrong = [t0 + dt.timedelta(hours=zones[a] - zones[b]), tb + dt.timedelta(hours=1), tb - dt.timedelta(hours=1),
                 tb + dt.timedelta(minutes=30), t0, tb + dt.timedelta(hours=12)]
        names = uniq_norm([tf(tb)] + [tf(x) for x in wrong])[:K]
        opts, back = opts_from_texts(names, "t_")
        q, g = choice_q(f"What local time and date is the call in {cb}?", opts, back[tf(tb)], r)
        return out("tz", state, q, g)
    if kind == "tenure":
        yrs = r.choice([3, 5, 7, 10])
        ref = dt.date(r.choice([2025, 2026]), r.choice([1, 4, 7]), 1)
        start = ref - dt.timedelta(days=round(yrs * 365.25) + r.choice([-40, -2, -1, 0, 1, 3, 60]))
        # full years of service on ref
        full = ref.year - start.year - ((ref.month, ref.day) < (start.month, start.day))
        rules = (f"Employees qualify for the long-service award if they have completed at least {yrs} full years of "
                 f"service on the assessment date ({fmt_date(ref, style)}).")
        event = f"Employee start date: {fmt_date(start, style)}."
        state = {"rules": rules, "event": event}
        return out("tenure_yn", state, yn("Does the employee qualify for the award?"), full >= yrs)
    # days_late -> score bucket
    due = base
    late = r.choice([0, 0, r.randint(1, 3), r.randint(4, 7), r.randint(8, 30), -r.randint(1, 5)])
    done = due + dt.timedelta(days=late)
    levels = ["on time or early", "1 to 3 days late", "4 to 7 days late", "more than 7 days late"]
    lv = 0 if late <= 0 else 1 if late <= 3 else 2 if late <= 7 else 3
    rules = "Lateness is counted in calendar days after the due date; delivery on the due date is on time."
    event = f"Deliverable due {fmt_date(due, style)}; delivered {fmt_date(done, style)}."
    state = {"rules": rules, "event": event}
    return out("days_late", state, score_q("How late was the delivery?", levels), lv)


def gen_gsm_choice(rows: list[dict], n: int) -> Iterator[dict]:
    from data.convert import gsm8k_distractors
    k = 0
    for i, x in enumerate(rows):
        if k >= n:
            return
        if part(f"gsm8k:{i}", 2) != 1:
            continue
        ans = x["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(ans)
        except ValueError:
            continue
        r = rng("gsm-choice", i)
        K = r.choice([4, 5, 6])
        names = uniq_norm([_num_fmt(g)] + gsm8k_distractors(g, K + 3, r))[:K]
        opts, back = opts_from_texts(names, "v_")
        q, gg = choice_q(pick(r, ["Work out the answer. Which value is correct?", "Solve the problem and pick the "
                                  "correct number.", "Which option is the exact answer?"]), opts, back[_num_fmt(g)], r)
        k += 1
        yield record("temporal_numeric", "gsm_choice", "gsm8k", f"m2:gsm8k:{i}", "num", x["question"], q, gg,
                     {"problem": x["question"]})


def gen_temporal(n_synth: int, n_gsm: int, gsm: list[dict]) -> Iterator[dict]:
    for i in range(n_synth):
        yield _temporal(i)
    yield from gen_gsm_choice(gsm, n_gsm)


# ================================================================================================ 4. fact
FACT_DOMAINS = {
    "order": (["Order {id} for {n} {item} ships from {city} on {date}.", "Order {id}: {n} {item}, dispatched from "
               "{city}, paid by {pay}.", "{n} {item} under order {id} were paid by {pay} and leave {city} on {date}."],
              {"id": lambda r: f"#{r.randint(10000, 99999)}", "n": lambda r: str(r.randint(2, 40)),
               "item": lambda r: r.choice(["desk lamps", "phone cases", "yoga mats", "kettles", "notebooks", "jackets",
                                           "water bottles", "headsets"]),
               "city": lambda r: r.choice(CITIES), "date": lambda r: fmt_date(dt.date(2026, r.randint(1, 12),
                                                                                       r.randint(1, 28)), 1),
               "pay": lambda r: r.choice(["card", "bank transfer", "invoice", "gift card", "PayPal"])}),
    "flight": (["Flight {code} to {city} departs at {time} from gate {gate}.", "Flight {code} leaves gate {gate} for "
                "{city} at {time}.", "Boarding for {city} on {code} starts at gate {gate}; departure {time}."],
               {"code": lambda r: f"{r.choice(['KQ', 'LX', 'TP', 'SN', 'AY', 'OS'])}{r.randint(100, 999)}",
                "city": lambda r: r.choice(CITIES), "time": lambda r: f"{r.randint(5, 22):02d}:{r.choice([0, 15, 25, 40, 55]):02d}",
                "gate": lambda r: f"{r.choice('ABCDEF')}{r.randint(1, 40)}"}),
    "staff": (["{name} joined the {dept} team in {year} and works from {city}.", "{name}, based in {city}, has led {dept} "
               "since {year}.", "{name} moved to {dept} in {year}; office: {city}."],
              {"name": lambda r: f"{r.choice(FIRST)} {r.choice(LAST)}", "dept": lambda r: r.choice(
                  ["finance", "legal", "design", "sales", "support", "research", "procurement", "security"]),
               "year": lambda r: str(r.randint(2008, 2026)), "city": lambda r: r.choice(CITIES)}),
    "invoice": (["Invoice {id} for {amt} is due on {date}.", "Invoice {id} ({amt}) was issued to {client}.",
                 "{client} owes {amt} on invoice {id}, due {date}."],
                {"id": lambda r: f"INV-{r.randint(1000, 9999)}", "amt": lambda r: money(r.randint(50, 9000)),
                 "date": lambda r: fmt_date(dt.date(2026, r.randint(1, 12), r.randint(1, 28)), 2),
                 "client": lambda r: f"{r.choice(['Kestrel', 'Lumen', 'Arbor', 'Cobalt', 'Harbor', 'Quill'])} "
                                     f"{r.choice(['Ltd', 'GmbH', 'Inc', 'Studio', 'Partners'])}"}),
    "room": (["Meeting room {room} seats {n} and has a {kit}.", "Room {room} on floor {floor} holds {n} people.",
              "The {kit} in room {room} (floor {floor}) is out of order."],
             {"room": lambda r: f"{r.choice('ABCDEFGH')}{r.randint(10, 60)}", "n": lambda r: str(r.randint(4, 30)),
              "kit": lambda r: r.choice(["projector", "whiteboard", "video bar", "speakerphone", "wall screen"]),
              "floor": lambda r: str(r.randint(1, 12))}),
    "product": (["The {prod} weighs {kg} kg and comes in {colour}.", "{prod}: {colour} finish, {years}-year warranty.",
                 "Our {prod} has a {years}-year warranty and weighs {kg} kg."],
                {"prod": lambda r: f"{r.choice(['Aero', 'Nova', 'Terra', 'Pulse', 'Echo'])} "
                                   f"{r.choice(['blender', 'drill', 'backpack', 'monitor', 'heater', 'scooter'])}",
                 "kg": lambda r: str(round(r.uniform(0.4, 18), 1)), "colour": lambda r: r.choice(
                     ["graphite", "sand", "olive", "navy", "white", "red"]), "years": lambda r: str(r.randint(1, 5))}),
    "loan": (["{name} borrowed '{book}' on {date}; it is due back in {n} days.", "'{book}' is on loan to {name} until "
              "{date}.", "The library copy of '{book}' was returned by {name} on {date}."],
             {"name": lambda r: f"{r.choice(FIRST)} {r.choice(LAST)}", "book": lambda r: r.choice(
                 ["Salt and Stone", "The Glass Orchard", "Northern Lines", "A Field Guide to Rain", "Quiet Engines",
                  "The Ninth Harbour", "Maps of Ember"]), "date": lambda r: fmt_date(
                 dt.date(2026, r.randint(1, 12), r.randint(1, 28)), 1), "n": lambda r: str(r.choice([7, 14, 21, 28]))}),
    "account": (["Account {acct} is on the {plan} plan with {seats} seats.", "The {plan} plan for account {acct} renews on "
                 "{date}.", "Account {acct} was downgraded to {plan} and now has {seats} seats."],
                {"acct": lambda r: f"AC-{r.randint(100, 999)}{r.choice('XYZQ')}", "plan": lambda r: r.choice(
                    ["Basic", "Team", "Business", "Enterprise", "Starter"]), "seats": lambda r: str(r.randint(1, 250)),
                 "date": lambda r: fmt_date(dt.date(2026, r.randint(1, 12), r.randint(1, 28)), 2)}),
}
FACT_Q = {"id": "the reference", "n": "the quantity", "item": "the item", "city": "the city", "date": "the date",
          "pay": "the payment method", "code": "the flight number", "time": "the time", "gate": "the gate",
          "name": "the person", "dept": "the department", "year": "the year", "amt": "the amount", "client": "the client",
          "room": "the room", "kit": "the equipment", "floor": "the floor", "prod": "the product", "kg": "the weight",
          "colour": "the colour", "years": "the warranty length", "book": "the book", "acct": "the account",
          "plan": "the plan", "seats": "the number of seats"}


def _fact(i: int) -> dict:
    r = rng("fact", i)
    dom = r.choice(sorted(FACT_DOMAINS))
    tpls, gens = FACT_DOMAINS[dom]
    tpl = r.choice(tpls)
    slots = re.findall(r"\{(\w+)\}", tpl)
    val = {s: gens[s](r) for s in dict.fromkeys(slots)}
    text = tpl.format(**val)
    text = text[0].upper() + text[1:]
    s = r.choice(slots)
    mode = r.random()
    if mode < 0.45:
        cand, gold = val[s], True
    elif mode < 0.88:
        cand = gens[s](r)
        tries = 0
        while norm(cand) == norm(val[s]) and tries < 20:
            cand, tries = gens[s](r), tries + 1
        if norm(cand) == norm(val[s]):
            cand = val[s] + "0"
        gold = False
    else:  # a slot the text does not mention: the text does not state it -> no
        missing = [x for x in gens if x not in slots]
        if not missing:
            cand, gold = val[s], True
        else:
            s = r.choice(missing)
            cand, gold = gens[s](r), False
    instr = pick(r, [f"Does the text state that {FACT_Q[s]} is {cand}?", f"According to the text, is {FACT_Q[s]} "
                     f"{cand}?", f"Is it stated explicitly that {FACT_Q[s]} is {cand}?"])
    q = noul_q(instr, "yes: the text states this explicitly", "no: the text says otherwise or does not say it")
    return record("fact", f"fact_{dom}", "jev2synth", f"m2:jev2synth:fact:{i}", "fact", text, q, gold, {},
                  split="synthetic", gen="jev2_fact_record", slot=s)


def boolq_rows() -> list[dict]:
    import pyarrow.parquet as pq
    from data import sources as S
    return pq.read_table(S.verified_path("boolq", "data/train-00000-of-00001.parquet")).to_pylist()


def gen_fact(n_synth: int, n_boolq: int) -> Iterator[dict]:
    for i in range(n_synth):
        yield _fact(i)
    rows = boolq_rows()
    k = 0
    for i, x in enumerate(rows):
        if k >= n_boolq:
            break
        if part(f"boolq:{i}", 3) != 0 or len(x["passage"]) > 1400:
            continue
        r = rng("boolq", i)
        qq = x["question"].strip().rstrip("?")
        qq = qq[0].upper() + qq[1:] + "?"
        q = noul_q(pick(r, ["Based only on the text: ", "Answer from the passage: ", ""]) + qq,
                   "yes, according to the text", "no, according to the text")
        k += 1
        yield record("fact", "boolq", "boolq", f"m2:boolq:{i}", "fact", x["passage"], q, bool(x["answer"]), {})


# ================================================================================================ 5. tool_selection
GLAIVE_FN_HEAD = "Use them if required -"
GENERIC = {"get", "create", "calculate", "search", "find", "check", "generate", "convert", "add", "send", "list",
           "info", "details", "data", "the", "a", "an", "of", "for", "to", "by", "and", "or", "with", "in", "on", "new",
           "random", "user", "update", "set", "make", "fetch", "retrieve", "information", "specified", "given", "based"}


def _toks(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.casefold()) if t and t not in GENERIC and len(t) > 2}


def _glaive_tools(sysmsg: str) -> list[dict] | None:
    at = sysmsg.find(GLAIVE_FN_HEAD)
    if at < 0:
        return None
    s, pos, tools, dec = sysmsg[at + len(GLAIVE_FN_HEAD):], 0, [], json.JSONDecoder()
    try:
        while True:
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break
            obj, pos = dec.raw_decode(s, pos)
            tools.append(obj)
    except json.JSONDecodeError:
        return None
    ok = [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]
          and isinstance(t.get("description"), str) and t["description"].strip()]
    return ok if len(ok) == len(tools) else None


def _glaive_batches():
    import pyarrow.parquet as pq
    from data import sources as S
    pf = pq.ParquetFile(S.verified_path("glaive_fc", "default/train/0000.parquet"))
    i = 0
    for b in pf.iter_batches(batch_size=1000, columns=["system", "chat"]):
        for sysmsg, chat in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            yield i, sysmsg or "", chat or ""
            i += 1


def gen_glaive(n: int) -> Iterator[dict]:
    pool: dict[str, str] = {}  # function name -> description (first seen), streamed pass 1
    for _, sysmsg, _c in _glaive_batches():
        tools = _glaive_tools(sysmsg)
        for t in tools or []:
            if len(pool) < 30000 and t["name"] not in pool and len(t["name"]) <= 60:
                pool[t["name"]] = t["description"].strip()[:160]
    names = sorted(pool)
    k = 0
    for i, sysmsg, chat in _glaive_batches():
        if k >= n:
            return
        if part(f"glaive:{i}", 6):
            continue
        tools = _glaive_tools(sysmsg)
        if not tools or not chat.startswith("USER:") or "\nASSISTANT:" not in chat:
            continue
        user, rest = chat[len("USER:"):].split("\nASSISTANT:", 1)
        user, first = user.strip(), rest.split("<|endoftext|>", 1)[0].strip()
        if not user or len(user) > 240 or not first.startswith("<functioncall>"):
            continue
        m = re.search(r'"name"\s*:\s*"([^"]+)"', first)
        own = {t["name"]: t["description"].strip()[:160] for t in tools}
        if not m or m.group(1) not in own:
            continue
        gold = m.group(1)
        r = rng("glaive", i)
        gt, gd = _toks(gold), _toks(own[gold])
        dis = [x for x in own if x != gold]
        tries = 0
        while len(dis) < 4 and tries < 400:
            tries += 1
            c = names[r.randrange(len(names))]
            ct = _toks(c) | _toks(pool[c])
            if c in dis or c == gold or ct & gt or len(_toks(pool[c]) & gd) >= 2:
                continue
            dis.append(c)
        dis = dis[:4]
        opts = {x: own.get(x) or pool[x] for x in [gold] + dis}
        if len(dis) < 4 or any(len(x) > 64 or "<|" in x or "|>" in x for x in opts) or \
                len({norm(v) for v in opts.values()}) < 5 or len({norm(x) for x in opts}) < 5:
            continue
        q, g = choice_q(pick(r, ["Which tool should be called for this request?", "Pick the one tool that handles the "
                                 "user's request.", "Select the function the assistant should call."]), opts, gold, r)
        k += 1
        yield record("tool_selection", "glaive", "glaive_fc", f"m2:glaive_fc:{i}", "tool", user, q, g, {})


TOOLS = {  # domain -> {tool: (description, [request templates])}
    "calendar": {"create_event": ("Add a new event to the user's calendar", ["Put a dentist appointment on my calendar for {day} at {hour}.", "Schedule a call with {person} on {day}."]),
                 "cancel_event": ("Remove an existing calendar event", ["Cancel my {day} meeting with {person}.", "Delete the {hour} appointment on {day}."]),
                 "list_events": ("Show the events on a given day", ["What's on my calendar for {day}?", "Do I have anything booked on {day}?"]),
                 "find_free_slot": ("Find an open time slot shared by attendees", ["When are {person} and I both free {day}?", "Find a free hour for me and {person} on {day}."])},
    "email": {"send_email": ("Send an email message to a recipient", ["Email {person} that I'll be late.", "Send {person} a note saying the report is ready."]),
              "search_inbox": ("Search received emails by keyword or sender", ["Find the email {person} sent about the invoice.", "Look for messages mentioning {thing} in my inbox."]),
              "archive_email": ("Move an email out of the inbox into the archive", ["Archive the newsletter from {store}.", "File away {person}'s last email."])},
    "weather": {"get_forecast": ("Weather forecast for a location and date", ["Will it rain in {city} {day}?", "What's the forecast for {city} this weekend?"]),
                "get_air_quality": ("Current air-quality index for a location", ["How's the air quality in {city} right now?", "Is the air in {city} safe for a run today?"])},
    "travel": {"search_flights": ("Search flights between two cities on a date", ["Find flights from {city} to {city2} on {day}.", "Any cheap flights to {city2} {day}?"]),
               "book_hotel": ("Reserve a hotel room", ["Book a hotel in {city} for two nights.", "Reserve a room near the station in {city2}."]),
               "check_flight_status": ("Live status of a specific flight", ["Is flight {flight} on time?", "Has {flight} landed yet?"]),
               "convert_currency": ("Convert an amount between currencies", ["How much is {amount} dollars in euros?", "Convert {amount} pounds to yen."])},
    "shopping": {"track_package": ("Tracking status of a shipped order", ["Where is my package from {store}?", "Track order {order}."]),
                 "add_to_cart": ("Add a product to the shopping cart", ["Add a {product} to my cart.", "Put two {product}s in the basket."]),
                 "start_return": ("Open a return for a delivered order", ["I want to return the {product} I bought from {store}.", "Start a return for order {order}."])},
    "finance": {"get_balance": ("Current balance of a bank account", ["How much is in my checking account?", "What's my savings balance?"]),
                "transfer_funds": ("Move money between the user's own accounts", ["Move {amount} dollars from savings to checking.", "Transfer {amount} to my savings."]),
                "get_stock_price": ("Latest price of a stock ticker", ["What's {ticker} trading at?", "Price of {ticker} right now?"]),
                "split_bill": ("Split a bill amount among people", ["Split a {amount} dollar dinner between {n} of us.", "We owe {amount}; divide it by {n}."])},
    "home": {"set_thermostat": ("Set the target temperature of the thermostat", ["Set the heating to {temp} degrees.", "Make it {temp} degrees in the living room."]),
             "toggle_lights": ("Turn lights on or off in a room", ["Turn off the kitchen lights.", "Switch on the lights in the {room}."]),
             "lock_door": ("Lock a smart door lock", ["Lock the front door.", "Make sure the {room} door is locked."])},
    "food": {"find_recipe": ("Find recipes for a dish or ingredient", ["Give me a recipe that uses {food}.", "How do I make {food} soup?"]),
             "order_takeout": ("Place a food delivery order", ["Order {food} from the place around the corner.", "Get me {food} delivered for dinner."]),
             "nutrition_facts": ("Calories and nutrients of a food", ["How many calories are in {food}?", "Nutrition info for a serving of {food}?"])},
    "work": {"create_ticket": ("Open a ticket in the issue tracker", ["Log a bug: the {thing} page won't load.", "File a ticket about the broken {thing}."]),
             "translate_text": ("Translate text into another language", ["Translate 'see you {day}' into Spanish.", "How do you say 'thank you' in Japanese?"]),
             "set_reminder": ("Create a reminder at a time", ["Remind me to call {person} at {hour}.", "Set a reminder for {day} to pay rent."]),
             "start_timer": ("Start a countdown timer", ["Set a timer for {n} minutes.", "Start a {n}-minute timer for the pasta."])},
    "media": {"play_music": ("Play a song, artist or playlist", ["Play some {genre}.", "Put on my {genre} playlist."]),
              "get_news": ("Latest news headlines on a topic", ["What's the latest news about {thing}?", "Any headlines on {city} today?"]),
              "find_movie_times": ("Showtimes at nearby cinemas", ["When is the new {genre} movie showing tonight?", "Showtimes in {city} this evening?"])},
}
TOOL_SLOTS = {"day": ["tomorrow", "Friday", "next Monday", "the 14th", "Thursday", "Saturday morning"],
              "hour": ["9am", "noon", "3:30pm", "6pm", "10:15"], "person": FIRST, "city": CITIES, "city2": CITIES,
              "thing": ["printer", "login", "dashboard", "checkout", "search", "export", "settings", "the merger",
                        "the budget"],
              "product": ["phone charger", "desk chair", "kettle", "yoga mat", "backpack", "monitor stand", "lamp",
                          "water bottle", "keyboard", "rain jacket"],
              "store": ["Nordbay", "Pixelmart", "Greenleaf", "Homewell", "Brightcart", "Tallow & Co"],
              "flight": ["LX318", "KQ102", "TP907", "AY66", "SN2281", "OS455"],
              "amount": ["50", "120", "300", "1,000", "75", "18.40", "2,500", "640"],
              "order": [f"#{x}" for x in (48213, 90127, 33450, 71802, 26019, 55174, 80431)],
              "ticker": ["AAPL", "MSFT", "NVDA", "TSLA", "SAP", "ASML", "SHOP"], "n": ["3", "4", "5", "6", "10", "25", "40"],
              "temp": ["19", "20", "21", "22", "68", "70", "72"], "room": ["bedroom", "garage", "office", "hallway",
                                                                         "kitchen", "nursery", "basement"],
              "food": ["lentils", "pumpkin", "salmon", "tofu", "pad thai", "chickpeas", "spinach", "risotto", "tacos"],
              "genre": ["jazz", "lo-fi", "classical", "horror", "comedy", "indie rock", "folk", "sci-fi", "hip hop"]}
TOOL_PRE = ["", "", "", "Hey, ", "Please ", "Quick one: ", "Could you help? ", "Hi! "]
TOOL_POST = ["", "", "", " Thanks.", " Please.", " Asap."]


def _tool_pick(i: int) -> dict:
    r = rng("tool-pick", i)
    dom = r.choice(sorted(TOOLS))
    gold = r.choice(sorted(TOOLS[dom]))
    desc, tpls = TOOLS[dom][gold]
    req = re.sub(r"\{(\w+)\}", lambda m: r.choice(TOOL_SLOTS[m.group(1)]), r.choice(tpls))
    pre = r.choice(TOOL_PRE)
    if pre == "Please ":
        req = req[0].lower() + req[1:]
    req = pre + req + r.choice(TOOL_POST)
    sib = [t for t in TOOLS[dom] if t != gold]
    other = [(d, t) for d in sorted(TOOLS) if d != dom for t in TOOLS[d]]
    dis = r.sample(sib, min(len(sib), r.randint(1, 3)))
    while len(dis) < 4:
        d, t = r.choice(other)
        if t not in dis:
            dis.append(t)
    allt = {t: v[0] for d in TOOLS for t, v in TOOLS[d].items()}
    opts = {t: allt[t] for t in [gold] + dis}
    q, g = choice_q(pick(r, ["Which tool should be called for this request?", "Pick the one tool that handles the "
                             "user's request.", "Select the function the assistant should call."]), opts, gold, r)
    return record("tool_selection", f"tool_{dom}", "jev2synth", f"m2:jev2synth:tool_pick:{i}", "tool", req, q, g, {},
                  split="synthetic", gen="jev2_tool_pick")


def gen_tools(n_glaive: int, n_synth: int) -> Iterator[dict]:
    yield from gen_glaive(n_glaive)
    for i in range(n_synth):
        yield _tool_pick(i)


# ================================================================================================ the pool
#: per-family sub-targets (rows generated; the finalize step trims each family to its target after the gates)
PLAN = {
    "multi_hop": {"ref_chase": 5600, "musique": 2600},
    "judge_hard": {"gsm8k": 3000, "synth": 5000},
    "temporal_numeric": {"synth": 6400, "gsm8k": 1800},
    "fact": {"synth": 5400, "boolq": 2800},
    "tool_selection": {"glaive": 4600, "synth": 4200},
}


def generate(families=STAGE_A) -> Iterator[dict]:
    p = PLAN
    gsm = gsm8k_rows() if any(f in families for f in ("judge_hard", "temporal_numeric")) else []
    if "multi_hop" in families:
        yield from gen_multi_hop(p["multi_hop"]["ref_chase"], p["multi_hop"]["musique"])
    if "judge_hard" in families:
        yield from gen_judge(p["judge_hard"]["gsm8k"], p["judge_hard"]["synth"], gsm)
    if "temporal_numeric" in families:
        yield from gen_temporal(p["temporal_numeric"]["synth"], p["temporal_numeric"]["gsm8k"], gsm)
    if "fact" in families:
        yield from gen_fact(p["fact"]["synth"], p["fact"]["boolq"])
    if "tool_selection" in families:
        yield from gen_tools(p["tool_selection"]["glaive"], p["tool_selection"]["synth"])
    fb = [f for f in STAGE_B if f in families]
    if fb:
        from data.jev2b import generate_b
        yield from generate_b(fb)


def generate_stage(stage: str) -> Iterator[dict]:
    if stage == "c":
        from data.jev2c import generate_c
        yield from generate_c(STAGE_C)
    else:
        yield from generate(STAGE_A if stage == "a" else STAGE_B)


#: stage c = long_policy at eval length (JEV1_AUDIT.md); its family name matches jev1's long_policy
STAGE_C = ("long_policy",)

# stage b / c generators register their specs and plan here; imported last (they import this module)
from data.jev2b import GEN_SPECS_B, PLAN_B  # noqa: E402
from data.jev2c import GEN_SPECS_C, PLAN_C  # noqa: E402

GEN_SPECS.update(GEN_SPECS_B)
GEN_SPECS.update(GEN_SPECS_C)
PLAN.update(PLAN_B)
PLAN.update(PLAN_C)
