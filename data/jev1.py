"""bschema-draw-jev1: a JevBench-FAMILY training draw from public TRAIN splits + deterministic synthetic generators.

Why: the size-cliff side finding (PREREG A-PREREG-SIZECLIFF OUTCOME) -- training Qwen3.5-4B on the Decision-Index-aimed
draw lowered its JevBench score (open-I 56.3 -> 51.0). JevBench's task shape is "application state + bounded rubric in,
typed answer out" (choice / noul / score). This module builds rows in exactly that request shape (``ssdm-train-v0``:
``state``, ``questions{q: {type, instructions, criteria}}``, ``labels``), one question per row, as the jevkit ``Item``
carries one question, so the 4B host (H2O shim readout ``jrt-h2o-shim:qwen3.5-4b``) trains on the shape it is scored on.

Eight families (``FAMILIES``), each from public train splits and / or generators written here:

  routing         MASSIVE en-US train (scenario -> assistant team), synthetic ticket routing (ordered rule list, first
                  match wins), synthetic model routing (cheapest catalog model meeting every constraint)
  adequacy        HelpSteer2 train (human 0-4 ratings -> score / noul), HelpSteer2 preference split=train (pairwise
                  choice, ties dropped), synthetic agent-action audits against a requirement rubric
  policy          Aegis 2.0 (Nemotron Content Safety v2) train, human labels only, as policy yes/no; synthetic access /
                  conduct policies with rule-derived gold
  intent          MASSIVE en-US train intents, BANKING77 train, CLINC150 train + oos_train
  severity        synthetic incident rubrics (score, ordinal, rule-derived level), SST-5 train (ordinal sentiment)
  enum            MASSIVE en-US train slot annotations (slot type / slot value), synthetic notices with distractor values
  long_policy     synthetic long policy documents (expense / returns / leave) with an executable rule engine for gold
  state_tracking  MultiWOZ 2.2 train dialogue state, synthetic event logs (orders / tickets / reservations / accounts)

Nothing here reads a JevBench item, an Index DEV/TEST row, a held-out row or MD1 data. Option presentation order is a
seeded shuffle per row (the shim letters options in the request's criteria order), so the gold position is uniform by
construction; gold appears exactly once (normalised text) in every choice question.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "ssdm-train-v0"
SEED = 20261010
FAMILIES = ("routing", "adequacy", "policy", "intent", "severity", "enum", "long_policy", "state_tracking")
SRC = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\sources")
DL = SRC / "dl-2026-10-10"
LIC = {"massive": "CC-BY-4.0", "helpsteer2": "CC-BY-4.0", "hs2pref": "CC-BY-4.0", "aegis2": "CC-BY-4.0",
       "banking77": "CC-BY-4.0", "clinc150": "CC-BY-3.0", "sst5": "unspecified on HF; SST research release",
       "multiwoz22": "MIT (budzianowski/multiwoz LICENSE; HF card apache-2.0)", "jev1synth": "own generator (repo)"}
#: generator specs for harness.heldout.check_generator (fail-closed: each declares its task family)
GEN_SPECS = {
    "jev1_ticket_routing": {"name": "jev1_ticket_routing", "task_family": "rule_list_ticket_routing"},
    "jev1_model_routing": {"name": "jev1_model_routing", "task_family": "constraint_model_routing"},
    "jev1_agent_audit": {"name": "jev1_agent_audit", "task_family": "agent_action_rubric_audit"},
    "jev1_access_policy": {"name": "jev1_access_policy", "task_family": "access_policy_yes_no"},
    "jev1_severity": {"name": "jev1_severity", "task_family": "incident_severity_rubric"},
    "jev1_notice_enum": {"name": "jev1_notice_enum", "task_family": "notice_enum_extraction"},
    "jev1_long_policy": {"name": "jev1_long_policy", "task_family": "long_policy_rule_engine"},
    "jev1_event_state": {"name": "jev1_event_state", "task_family": "business_event_log_state"},
}


# ------------------------------------------------------------------------------------------------ helpers
def rng(*key: Any) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, (SEED,) + key)).encode()).hexdigest())


def norm(s: Any) -> str:
    return re.sub(r"\W+", " ", str(s).casefold()).strip()


def shuffled(d: dict, r: random.Random) -> dict:
    ks = list(d)
    r.shuffle(ks)
    return {k: d[k] for k in ks}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).casefold()).strip("_")[:60] or "x"


def record(fam: str, sub: str, source: str, parent: str, view: str, state: Any, q: dict, label: Any,
           item: dict, split: str = "train", **meta) -> dict:
    """One ssdm-train-v0 row. ``parent`` = the source item id (cohort / dedup identity), ``item`` = its item-tier
    texts (the Index fingerprint probe)."""
    if q["type"] == "noul":
        label = "true" if label in (True, "true") else "false"
    elif q["type"] == "score":
        label = str(int(label))
    return {"id": f"{parent}~view:jev1-{view}", "family": f"jev1_{sub}", "source": source,
            "source_id": parent.split(":", 2)[-1], "source_split": split, "license": LIC[source], "schema": SCHEMA,
            "state": state, "questions": {"q": q}, "labels": {"q": label}, "untrusted_fields": [],
            "aug": {"b1_view": {"base_id": parent, "kind": "schema", "parent_id": parent},
                    "jev1": {"family8": fam, "sub": sub, "item_keys": item_keys(state, item), **meta}}}


def item_keys(state: Any, item: dict) -> list[str] | None:
    """State keys holding the item-tier text (the Index fingerprint probe); None = the whole state. Only key NAMES are
    stored in ``aug`` (texts in a non-content key would be read as tags by harness.heldout)."""
    if not isinstance(state, dict):
        return None
    ks = sorted(k for k in item if k in state)
    return ks or None


def choice_q(instr: str, opts: dict[str, str], gold: str, r: random.Random) -> tuple[dict, str]:
    assert gold in opts and len(opts) >= 2
    texts = [norm(v) for v in opts.values()]
    assert len(set(texts)) == len(texts), opts
    return {"type": "choice", "instructions": instr, "criteria": shuffled(opts, r)}, gold


def noul_q(instr: str, true_desc: str | None = None, false_desc: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instr}
    if true_desc:
        q["criteria"] = {"true": true_desc, "false": false_desc}
    return q


def score_q(instr: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instr, "criteria": list(levels)}


def pick(r: random.Random, xs):
    return xs[r.randrange(len(xs))]


def part(key: str, n: int) -> int:
    """Deterministic label-blind partition of a source item into one of n pools."""
    return int(hashlib.sha256(f"jev1-part|{key}".encode()).hexdigest()[:8], 16) % n


# ------------------------------------------------------------------------------------------------ source loaders
def massive_rows() -> list[dict]:
    t = tarfile.open(DL / "massive" / "amazon-massive-dataset-1.0.tar.gz", "r:gz")
    for m in t:
        if m.name.endswith("/en-US.jsonl"):
            rows = [json.loads(l) for l in t.extractfile(m)]
            return [x for x in rows if x["partition"] == "train"]  # TRAIN partition only (never dev / test)
    raise FileNotFoundError("en-US.jsonl not in the MASSIVE tarball")


SCENARIO_DESC = {
    "alarm": "alarm team: setting, changing, listing or removing alarms", "audio": "audio team: device volume and "
    "muting", "calendar": "calendar team: events, reminders and schedule lookups", "cooking": "cooking team: recipes "
    "and cooking help", "datetime": "date-and-time team: current time, dates, time zones", "email": "email team: "
    "reading, sending and managing email and contacts' addresses", "general": "general team: jokes, chit-chat, "
    "repeats and questions about the assistant", "iot": "connected-home team: lights, plugs, cleaning robots, coffee "
    "machines and similar connected gadgets", "lists": "lists team: shopping and to-do lists", "music": "music team: "
    "playing music and music preferences", "news": "news team: headlines and news queries", "play": "playback team: "
    "radio, podcasts, audiobooks and games", "qa": "question-answering team: facts, definitions, maths, stock prices",
    "recommendation": "recommendation team: places, events and movies to suggest", "social": "social team: posting "
    "to or complaining on social media", "takeaway": "takeaway team: ordering food and order status", "transport": ""
    "transport team: tickets, taxis, traffic and train info", "weather": "weather team: forecasts and conditions"}


def helpsteer2_rows() -> list[dict]:
    with gzip.open(DL / "helpsteer2" / "train.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def hs2pref_rows() -> list[dict]:
    out = []
    with gzip.open(DL / "helpsteer2" / "preference.jsonl.gz", "rt", encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            d = json.loads(l)
            if d.get("split") == "train":  # train split only
                d["_i"] = i
                out.append(d)
    return out


def aegis_rows() -> list[dict]:
    return json.load(open(DL / "aegis2" / "train.json", encoding="utf-8"))


def banking_rows() -> list[dict]:
    with open(SRC / "banking77" / "train.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clinc_rows() -> list[tuple[str, str, str]]:
    d = json.load(open(SRC / "clinc150" / "data_full.json", encoding="utf-8"))
    return [(t, l, "train") for t, l in d["train"]] + [(t, "oos", "oos_train") for t, _ in d["oos_train"]]


def sst5_rows() -> list[dict]:
    return [json.loads(l) for l in open(SRC / "sst5" / "train.jsonl", encoding="utf-8") if l.strip()]


def multiwoz_dialogues() -> Iterator[dict]:
    for f in sorted((DL / "multiwoz22").glob("dialogues_*.json")):
        yield from json.load(open(f, encoding="utf-8"))  # all files are the TRAIN folder of MultiWOZ 2.2


# ================================================================================================ 1. routing
def gen_routing(n_massive: int, n_ticket: int, n_model: int, massive: list[dict]) -> Iterator[dict]:
    # --- MASSIVE scenario routing (pool 0 of 3)
    pool = [x for x in massive if part("massive:" + x["id"], 3) == 0]
    scen = sorted(SCENARIO_DESC)
    for x in pool[:n_massive]:
        r = rng("route-massive", x["id"])
        k = r.randint(4, 10)
        others = [s for s in scen if s != x["scenario"]]
        r.shuffle(others)
        opts = {s: SCENARIO_DESC[s] for s in [x["scenario"]] + others[: k - 1]}
        parent = f"m2:massive:{x['id']}"
        if r.random() < 0.75:
            q, g = choice_q(pick(r, ["Which team should handle this request to the voice assistant?",
                                     "Route the user's request to the team that owns it.",
                                     "Pick the assistant team responsible for this utterance."]), opts, x["scenario"], r)
            yield record("routing", "massive_scenario", "massive", parent, "route", {"utterance": x["utt"],
                         "channel": pick(r, ["voice", "smart speaker", "phone app"])}, q, g, {"utterance": x["utt"]})
        else:
            cand = x["scenario"] if r.random() < 0.5 else others[0]
            q = noul_q(f"Should this request be routed to the {cand} team ({SCENARIO_DESC[cand].split(': ', 1)[1]})?")
            yield record("routing", "massive_scenario_yn", "massive", parent, "route-yn", {"utterance": x["utt"]},
                         q, cand == x["scenario"], {"utterance": x["utt"]})
    # --- synthetic ticket routing (ordered rule list)
    for i in range(n_ticket):
        yield _ticket_route(i)
    for i in range(n_model):
        yield _model_route(i)


TICKET_AREAS = {
    "billing": ["I was charged twice for my {plan} subscription this month.", "My invoice shows a {amt} fee I don't "
                "recognise.", "Please refund the {amt} charge from last week.", "Why did my {plan} renewal price go up?"],
    "account": ["I can't log in after resetting my password.", "Please change the email on my account.",
                "My two-factor codes stopped arriving.", "I want to close my account and remove my data."],
    "shipping": ["My order {oid} hasn't arrived and tracking hasn't moved in {d} days.", "The parcel for {oid} was "
                 "delivered to the wrong address.", "Can I change the delivery date for {oid}?"],
    "technical": ["The app crashes whenever I open the reports page.", "Exports to CSV are empty since the update.",
                  "Sync between my laptop and phone stopped working.", "The API returns 500 errors on every call."],
    "sales": ["We'd like a quote for {seats} seats on the {plan} plan.", "Do you offer discounts for nonprofits?",
              "Can we upgrade mid-cycle to the {plan} plan?"],
}
QUEUES = {
    "legal_review": "handled by counsel: lawsuits, lawyers, demand letters",
    "security_response": "handled by the security on-call: compromised accounts, unknown logins",
    "vip_desk": "handled by named account managers for enterprise contracts",
    "refunds_high_value": "handled by finance controllers: billing disputes above the threshold",
    "billing_general": "handled by billing specialists: invoices, renewals, small refunds",
    "identity_access": "handled by the identity team: sign-in, 2FA, email changes",
    "logistics": "handled by the carrier liaison team: stalled or misdelivered parcels",
    "engineering_l2": "handled by API engineers: errors and outages in the public API",
    "support_l1": "handled by first-line agents: anything no earlier rule claims",
    "sales_inbound": "handled by account executives: quotes, upgrades, discounts",
    "language_desk": "handled by multilingual agents: messages not written in English",
    "privacy_office": "handled by the data-protection officer: deletion and closure requests",
}


def _ticket_rules(r: random.Random) -> list[dict]:
    """An ordered routing policy: each rule = (condition text, predicate spec, queue). Order is the precedence."""
    thr = r.choice([200, 300, 500, 750, 1000])
    days = r.choice([5, 7, 10])
    rules = [
        {"if": "the message mentions a lawyer, lawsuit, or legal action", "p": ("legal",), "queue": "legal_review"},
        {"if": "the message reports a possible account compromise or unauthorised access", "p": ("security",),
         "queue": "security_response"},
        {"if": "the customer asks to delete their data or close their account", "p": ("privacy",),
         "queue": "privacy_office"},
        {"if": "the customer tier is enterprise", "p": ("tier", "enterprise"), "queue": "vip_desk"},
        {"if": f"the area is billing and the disputed amount is at least ${thr}", "p": ("billing_amt", thr),
         "queue": "refunds_high_value"},
        {"if": "the area is billing", "p": ("area", "billing"), "queue": "billing_general"},
        {"if": "the area is account", "p": ("area", "account"), "queue": "identity_access"},
        {"if": f"the area is shipping and the order has been stuck for more than {days} days", "p": ("ship_days", days),
         "queue": "logistics"},
        {"if": "the area is technical and the product is the API", "p": ("api",), "queue": "engineering_l2"},
        {"if": "the area is sales", "p": ("area", "sales"), "queue": "sales_inbound"},
        {"if": "the message is not in English", "p": ("lang",), "queue": "language_desk"},
    ]
    # perturb precedence: swap a few adjacent rules, drop one optional rule; the final catch-all stays last
    for _ in range(r.randint(0, 3)):
        j = r.randrange(len(rules) - 1)
        rules[j], rules[j + 1] = rules[j + 1], rules[j]
    if r.random() < 0.5:
        rules.pop(r.randrange(len(rules)))
    rules.append({"if": "no earlier rule matches", "p": ("any",), "queue": "support_l1"})
    return rules


def _ticket_route(i: int) -> dict:
    r = rng("ticket", i)
    area = r.choice(sorted(TICKET_AREAS))
    amt = r.choice([45, 89, 120, 240, 310, 480, 520, 760, 990, 1250])
    t = {"area": area, "tier": r.choice(["free", "standard", "standard", "pro", "enterprise"]),
         "language": "English" if r.random() < 0.85 else r.choice(["Spanish", "German", "Portuguese", "French"]),
         "product": r.choice(["web app", "mobile app", "API", "desktop client"]),
         "legal": r.random() < 0.08, "security": r.random() < 0.08, "privacy": r.random() < 0.07,
         "amount": amt if area == "billing" else None, "stuck_days": r.randint(1, 15) if area == "shipping" else None}
    if t["privacy"]:
        t["area"] = "account"
    msg = pick(r, TICKET_AREAS[t["area"]]).format(plan=r.choice(["Basic", "Team", "Pro"]), amt=f"${amt}",
                                                  oid=f"#{r.randint(10000, 99999)}", d=t["stuck_days"] or 3,
                                                  seats=r.choice([5, 20, 50, 200]))
    if t["privacy"]:
        msg = "Please delete all of my personal data and close the account. " + msg
    if t["legal"]:
        msg += " If this isn't fixed I will have my lawyer contact you."
    if t["security"]:
        msg += " I also see logins from a country I've never visited."
    rules = _ticket_rules(r)

    def match(p):
        k = p[0]
        return {"legal": t["legal"], "security": t["security"], "privacy": t["privacy"],
                "tier": k == "tier" and t["tier"] == (p[1] if len(p) > 1 else None),
                "billing_amt": k == "billing_amt" and t["area"] == "billing" and (t["amount"] or 0) >= p[1],
                "area": k == "area" and t["area"] == (p[1] if len(p) > 1 else None),
                "ship_days": k == "ship_days" and t["area"] == "shipping" and (t["stuck_days"] or 0) > p[1],
                "api": t["area"] == "technical" and t["product"] == "API", "lang": t["language"] != "English",
                "any": True}[k]

    gold = next(x["queue"] for x in rules if match(x["p"]))
    ticket = {"ticket_id": f"T-{r.randint(100000, 999999)}", "customer_tier": t["tier"], "area": t["area"],
              "product": t["product"], "language": t["language"], "message": msg}
    if t["amount"] is not None:
        ticket["disputed_amount_usd"] = t["amount"]
    if t["stuck_days"] is not None:
        ticket["days_without_tracking_update"] = t["stuck_days"]
    state = {"routing_policy": [f"{j + 1}. If {x['if']}, route to {x['queue']}." for j, x in enumerate(rules)],
             "ticket": ticket}
    used = sorted({x["queue"] for x in rules})
    parent = f"m2:jev1synth:ticket_routing:{i}"
    if r.random() < 0.8:
        opts = {qn: QUEUES[qn] for qn in used}
        q, g = choice_q(pick(r, ["Apply the routing policy in order; the first matching rule decides. Which queue "
                                 "receives this ticket?", "Using the numbered routing rules (first match wins), "
                                 "route the ticket.", "Which queue does the routing policy send this ticket to? Rules "
                                 "are checked top to bottom and the first match applies."]), opts, gold, r)
        return record("routing", "ticket_routing", "jev1synth", parent, "route", state, q, g, {"ticket": ticket},
                      split="synthetic", gen="jev1_ticket_routing")
    cand = gold if r.random() < 0.5 else r.choice([u for u in used if u != gold])
    q = noul_q(f"Under the routing policy (rules checked in order, first match wins), does this ticket go to "
               f"{cand}?")
    return record("routing", "ticket_routing_yn", "jev1synth", parent, "route-yn", state, q, cand == gold,
                  {"ticket": ticket}, split="synthetic", gen="jev1_ticket_routing")


MODEL_NAMES = ["atlas-s", "atlas-m", "atlas-l", "brisk-mini", "brisk-pro", "cobalt-7", "cobalt-34", "delta-edge",
               "ember-chat", "ember-code", "fjord-vision", "gale-long", "harbor-xs", "iris-multi", "juniper-max"]
CAPS = ["code", "vision", "tool_calls", "multilingual", "json_mode"]


def _model_route(i: int) -> dict:
    r = rng("model-route", i)
    k = r.randint(4, 8)
    names = r.sample(MODEL_NAMES, k)
    cat = []
    for nm in names:
        cat.append({"name": nm, "context_tokens": r.choice([8000, 16000, 32000, 128000, 200000]),
                    "capabilities": sorted(r.sample(CAPS, r.randint(1, 4))),
                    "usd_per_million_tokens": r.choice([0.1, 0.25, 0.4, 0.6, 1.0, 1.5, 3.0, 5.0, 8.0, 15.0]),
                    "p50_latency_ms": r.choice([120, 200, 350, 600, 900, 1500, 2500]),
                    "quality_tier": r.randint(1, 5)})
    # make costs unique so "cheapest" is well defined
    seen = set()
    for m in cat:
        while m["usd_per_million_tokens"] in seen:
            m["usd_per_million_tokens"] = round(m["usd_per_million_tokens"] + 0.05, 2)
        seen.add(m["usd_per_million_tokens"])
    req = {"task": r.choice(["summarise a contract", "classify support emails", "answer a chart question",
                             "write a SQL migration", "translate product pages", "extract invoice fields",
                             "plan a multi-step API workflow"]),
           "input_tokens": r.choice([1500, 6000, 12000, 30000, 90000, 150000]),
           "required_capabilities": sorted(r.sample(CAPS, r.randint(0, 2))),
           "max_p50_latency_ms": r.choice([300, 700, 1000, 2000, 5000]), "min_quality_tier": r.randint(1, 4)}
    ok = [m for m in cat if m["context_tokens"] >= req["input_tokens"]
          and set(req["required_capabilities"]) <= set(m["capabilities"])
          and m["p50_latency_ms"] <= req["max_p50_latency_ms"] and m["quality_tier"] >= req["min_quality_tier"]]
    gold = min(ok, key=lambda m: (m["usd_per_million_tokens"], m["p50_latency_ms"]))["name"] if ok else "no_eligible_model"
    rule = ("A model is eligible when its context window holds the input, it has every required capability, its p50 "
            "latency is within the budget and its quality tier is at least the minimum. Route to the cheapest "
            "eligible model (ties: lower latency). If no model is eligible, reject the request.")
    state = {"routing_rule": rule, "model_catalog": cat, "request": req}
    parent = f"m2:jev1synth:model_routing:{i}"
    if r.random() < 0.75:
        opts = {m["name"]: f"route to {m['name']}" for m in cat}
        opts["no_eligible_model"] = "reject: no catalog model meets every constraint"
        q, g = choice_q(pick(r, ["Which model should serve this request under the routing rule?",
                                 "Apply the routing rule to the catalog. Where does the request go?"]), opts, gold, r)
        return record("routing", "model_routing", "jev1synth", parent, "route", state, q, g, {"request": req,
                      "catalog": cat}, split="synthetic", gen="jev1_model_routing")
    q = noul_q("Is at least one catalog model eligible for this request under the routing rule?")
    return record("routing", "model_routing_yn", "jev1synth", parent, "route-yn", state, q, bool(ok),
                  {"request": req, "catalog": cat}, split="synthetic", gen="jev1_model_routing")


# ================================================================================================ 2. adequacy
HS_ATTR = {
    "helpfulness": ["not helpful at all: ignores or fails the request", "slightly helpful: addresses little of the "
                    "request", "partially helpful: useful but misses key parts", "mostly helpful: covers the request "
                    "with minor gaps", "fully helpful: completely satisfies the request"],
    "correctness": ["contains no correct or relevant information", "mostly incorrect or with major errors",
                    "a mix of correct and incorrect or missing information", "mostly correct with minor errors or "
                    "omissions", "fully correct and complete"],
    "coherence": ["incoherent or unreadable", "mostly unclear, hard to follow", "partly clear with noticeable "
                  "contradictions or confusion", "mostly clear and consistent", "perfectly clear and self-consistent"],
}


def _hs_turns(prompt: str) -> Any:
    """HelpSteer2 multi-turn prompts join turns with <extra_id_1>User / Assistant markers; render as a list."""
    if "<extra_id_1>" not in prompt:
        return prompt
    parts = re.split(r"<extra_id_1>(User|Assistant)\n?", prompt)
    turns = [{"role": "user", "text": parts[0].strip()}] if parts[0].strip() else []
    for j in range(1, len(parts) - 1, 2):
        turns.append({"role": parts[j].lower(), "text": parts[j + 1].strip()})
    return turns


def gen_adequacy(n_score: int, n_noul: int, n_pref: int, n_audit: int, hs: list[dict], pref: list[dict]) -> Iterator[dict]:
    order = list(range(len(hs)))
    rng("hs2-order").shuffle(order)
    seen_p: set[str] = set()
    sel = []
    for i in order:  # one HelpSteer2 row per prompt (rows share prompts in pairs; the pref split reuses them)
        if len(hs[i]["prompt"]) + len(hs[i]["response"]) <= 5200 and norm(hs[i]["prompt"]) not in seen_p:
            seen_p.add(norm(hs[i]["prompt"]))
            sel.append(i)
    # pools: score uses the first slice, noul the next (one source row -> one family row)
    sc, nl = sel[: n_score], sel[n_score: n_score + int(n_noul * 1.6)]
    for i in sc:
        x, r = hs[i], rng("hs2-score", i)
        attr = r.choice(sorted(HS_ATTR))
        lv = HS_ATTR[attr]
        rev = r.random() < 0.3
        levels = list(reversed(lv)) if rev else lv
        g = int(x[attr])
        g = 4 - g if rev else g
        q = score_q(pick(r, [f"Grade the AI response's {attr} on the rubric levels.", f"Rate the response for "
                             f"{attr} using the scale.", f"How would a careful reviewer score the response's {attr}?"]),
                    levels)
        st = {"user_request": _hs_turns(x["prompt"]), "ai_response": x["response"]}
        yield record("adequacy", "hs2_score", "helpsteer2", f"m2:helpsteer2:{i}", f"score-{attr}", st, q, g,
                     {"prompt": x["prompt"], "response": x["response"]}, attribute=attr, reversed_scale=rev)
    k = 0
    k_used = 0
    for i in nl:
        if k >= n_noul:
            break
        k_used += 1
        x, r = hs[i], rng("hs2-noul", i)
        attr = r.choice(["correctness", "helpfulness"])
        v = int(x[attr])
        if v == 2:  # the scale midpoint is not decisive for a yes/no bar; drop it (label-derived, recorded)
            continue
        bar = {"correctness": "factually correct with at most minor errors or omissions (3 or 4 on a 0-4 scale)",
               "helpfulness": "mostly or fully helpful for what the user asked (3 or 4 on a 0-4 scale)"}[attr]
        q = noul_q(f"Does the AI response meet this bar: {bar}?", "the response meets the bar",
                   "the response falls short of the bar")
        st = {"user_request": _hs_turns(x["prompt"]), "ai_response": x["response"]}
        k += 1
        yield record("adequacy", "hs2_bar", "helpsteer2", f"m2:helpsteer2:{i}", f"bar-{attr}", st, q, v >= 3,
                     {"prompt": x["prompt"], "response": x["response"]}, attribute=attr)
    used_p = {norm(hs[i]["prompt"]) for i in sc} | {norm(hs[i]["prompt"]) for i in nl[:k_used]}
    k = 0
    po = list(range(len(pref)))
    rng("hs2pref-order").shuffle(po)
    for j in po:
        if k >= n_pref:
            break
        x = pref[j]
        s = int(x["preference_strength"])
        if s == 0 or len(x["prompt"]) + len(x["response_1"]) + len(x["response_2"]) > 6400:
            continue
        if norm(x["prompt"]) in used_p:
            continue
        used_p.add(norm(x["prompt"]))
        r = rng("hs2pref", x["_i"])
        gold = "response_1" if s < 0 else "response_2"
        flip = r.random() < 0.5  # which response is shown as A / B is part of the presentation
        a, b = ("response_2", "response_1") if flip else ("response_1", "response_2")
        st = {"user_request": _hs_turns(x["prompt"]), "response_a": x[a], "response_b": x[b]}
        lab = {a: "response_a", b: "response_b"}
        opts = {"response_a": "Response A is better overall (helpfulness, correctness, instruction following)",
                "response_b": "Response B is better overall (helpfulness, correctness, instruction following)"}
        q, g = choice_q(pick(r, ["Which response better serves the user?", "Judge the two candidate replies; which "
                                 "one is better?", "Compare the responses against the request. Pick the better one."]),
                        opts, lab[gold], r)
        k += 1
        yield record("adequacy", "hs2_pref", "hs2pref", f"m2:hs2pref:{x['_i']}", "pref", st, q, g,
                     {"prompt": x["prompt"], "response_1": x["response_1"], "response_2": x["response_2"]},
                     strength=s)
    for i in range(n_audit):
        yield _agent_audit(i)


def _agent_audit(i: int) -> dict:
    r = rng("audit", i)
    limit = r.choice([50, 100, 150, 200, 250])
    reqs = [("verify", "verify the customer's identity (order number + email) before changing anything"),
            ("limit", f"never issue a refund above ${limit} without a supervisor_approval step"),
            ("pii", "never paste a full card number or password into a message"),
            ("confirm", "summarise the change back to the customer before closing the ticket"),
            ("tool", "use the refund tool for refunds, never a manual balance edit")]
    use = r.sample(reqs, r.randint(3, 5))
    amt = r.choice([20, 45, 80, 120, 180, 230, 300])
    steps, viol = [], []
    if "verify" in [u[0] for u in use] or r.random() < 0.5:
        if r.random() < 0.8:
            steps.append({"action": "lookup_order", "args": {"order": f"A{r.randint(1000, 9999)}", "email_matched": True}})
        else:
            steps.append({"action": "lookup_order", "args": {"order": f"A{r.randint(1000, 9999)}", "email_matched": False}})
            viol.append("verify")
    else:
        viol.append("verify")
    approve = amt > limit and r.random() < 0.6
    if approve:
        steps.append({"action": "supervisor_approval", "args": {"amount": amt, "approved": True}})
    manual = r.random() < 0.15
    steps.append({"action": "edit_balance" if manual else "refund", "args": {"amount": amt}})
    if manual:
        viol.append("tool")
    if amt > limit and not approve:
        viol.append("limit")
    leak = r.random() < 0.12
    msg = ("Done! Your refund is on card 4111 1111 1111 1111." if leak else
           pick(r, ["Your refund has been issued.", f"I've refunded ${amt} to your original payment method.",
                    "All set."]))
    if leak:
        viol.append("pii")
    steps.append({"action": "send_message", "args": {"text": msg}})
    summ = r.random() < 0.75
    if summ:
        steps.insert(len(steps) - 1, {"action": "send_message", "args": {"text": f"To confirm: refund of ${amt} for "
                                                                         "your order, then I'll close the ticket."}})
    else:
        viol.append("confirm")
    steps.append({"action": "close_ticket", "args": {}})
    active = [u for u in use]
    broken = [u[0] for u in active if u[0] in viol]
    state = {"requirements": [f"R{j + 1}: the agent must {u[1]}" for j, u in enumerate(active)],
             "task": f"customer asks for a ${amt} refund", "agent_trace": steps}
    parent = f"m2:jev1synth:agent_audit:{i}"
    t = r.random()
    if t >= 0.72:
        levels = ["no requirement is violated", "exactly one requirement is violated", "exactly two requirements are "
                  "violated", "three or more requirements are violated"]
        q = score_q("How many of the listed requirements does the agent's trace violate?", levels)
        return record("adequacy", "agent_audit_count", "jev1synth", parent, "audit-count", state, q,
                      min(3, len(broken)), {"trace": steps}, split="synthetic", gen="jev1_agent_audit")
    if t < 0.45:
        q = noul_q("Does the agent's trace satisfy every listed requirement?", "every requirement is satisfied",
                   "at least one requirement is violated")
        return record("adequacy", "agent_audit_yn", "jev1synth", parent, "audit-yn", state, q, not broken,
                      {"trace": steps}, split="synthetic", gen="jev1_agent_audit")
    opts = {f"R{j + 1}": f"R{j + 1} is the first listed requirement the agent violated" for j in range(len(active))}
    opts["none"] = "the agent violated none of the requirements"
    first = next((f"R{j + 1}" for j, u in enumerate(active) if u[0] in broken), "none")
    q, g = choice_q("Which is the first listed requirement (lowest number) that the agent's trace violates?", opts,
                    first, r)
    return record("adequacy", "agent_audit_first", "jev1synth", parent, "audit-first", state, q, g, {"trace": steps},
                  split="synthetic", gen="jev1_agent_audit")


# ================================================================================================ 3. policy yes/no
AEGIS_CATS = {
    "Violence": "threats, incitement or glorification of physical harm", "Sexual": "sexually explicit content",
    "Criminal Planning/Confessions": "planning, facilitating or confessing to crimes",
    "Guns and Illegal Weapons": "acquiring, making or using illegal weapons",
    "Controlled/Regulated Substances": "obtaining, making or misusing drugs and regulated substances",
    "Suicide and Self Harm": "encouraging or instructing self-harm", "Sexual (minor)": "any sexual content involving minors",
    "Hate/Identity Hate": "attacks on people for protected characteristics", "PII/Privacy": "exposing or seeking "
    "private personal information", "Harassment": "targeted abuse, bullying or intimidation", "Threat": "statements "
    "of intent to harm", "Profanity": "gratuitous obscene language", "Needs Caution": "borderline content that needs "
    "care", "Manipulation": "deceiving or psychologically manipulating people", "Fraud/Deception": "scams, fraud or "
    "impersonation", "Malware": "creating or deploying malicious code", "High Risk Gov Decision Making": "automating "
    "high-stakes government decisions", "Political/Misinformation/Conspiracy": "political misinformation or "
    "conspiracy promotion", "Copyright/Trademark/Plagiarism": "infringing intellectual property",
    "Unauthorized Advice": "unqualified medical, legal or financial advice", "Illegal Activity": "other illegal "
    "activity", "Immoral/Unethical": "clearly unethical conduct"}


def gen_policy(n_aegis: int, n_access: int, aegis: list[dict]) -> Iterator[dict]:
    order = list(range(len(aegis)))
    rng("aegis-order").shuffle(order)
    k = 0
    cats = sorted(AEGIS_CATS)
    for j in order:
        if k >= n_aegis:
            break
        x = aegis[j]
        if x["prompt"] == "REDACTED" or x.get("reconstruction_id_if_redacted") or len(x["prompt"]) > 4000:
            continue
        r = rng("aegis", x["id"])
        resp = x.get("response")
        has_resp = bool(resp) and x.get("response_label") in ("safe", "unsafe") and len(resp) <= 4000
        target = "response" if has_resp and r.random() < 0.5 else "prompt"
        lab = x["response_label"] if target == "response" else x["prompt_label"]
        lab_src = x.get("response_label_source") if target == "response" else "human"
        vc = [c.strip() for c in (x.get("violated_categories") or "").split(",") if c.strip()]
        st = {"user_message": x["prompt"]}
        if target == "response":
            st["assistant_reply"] = resp
        if resp is None and r.random() < 0.4:
            # category question: only on prompt-only rows, where violated_categories annotate the (human-labelled)
            # user message itself
            if lab == "unsafe" and any(v in AEGIS_CATS for v in vc) and r.random() < 0.5:
                c = r.choice([v for v in vc if v in AEGIS_CATS])
            else:
                c = r.choice([v for v in cats if v not in vc])
            q = noul_q(f"Does the user message violate the policy category '{c}' ({AEGIS_CATS[c]})?",
                       f"it violates '{c}'", f"it does not violate '{c}'")
            gold = lab == "unsafe" and c in vc
            sub = "aegis_category"
        else:
            pol = r.sample(cats, r.randint(6, 12))
            st = {"content_policy": {c: AEGIS_CATS[c] for c in sorted(pol)}, **st}
            what = "assistant reply" if target == "response" else "user message"
            q = noul_q(pick(r, [f"Is the {what} unsafe under the content policy (any category, or otherwise clearly "
                                f"harmful)?", f"Should the {what} be flagged as unsafe?"]),
                       f"the {what} is unsafe", f"the {what} is safe")
            gold = lab == "unsafe"
            sub = "aegis_unsafe"
        k += 1
        yield record("policy", sub, "aegis2", f"m2:aegis2:{x['id']}", f"pol-{target}", st, q, gold,
                     {"prompt": x["prompt"], "response": resp or ""}, target=target, label_source=lab_src)
    for i in range(n_access):
        yield _access_policy(i)


ROLES = ["support_agent", "billing_analyst", "data_scientist", "contractor", "hr_partner", "engineer", "auditor"]
DATA = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
PURPOSES = ["customer_support", "fraud_investigation", "model_training", "payroll", "marketing", "incident_response",
            "annual_audit"]


def _access_policy(i: int) -> dict:
    r = rng("access", i)
    clear = {ro: r.randint(0, 3) for ro in ROLES}
    clear["contractor"] = min(clear["contractor"], 1)
    banned = r.sample(PURPOSES, 2)
    need_mfa = r.choice([2, 3])
    rules = [f"P1. A role may read data up to its clearance level: " + ", ".join(
                 f"{ro}={[k for k, v in DATA.items() if v == c][0]}" for ro, c in clear.items()) + ".",
             f"P2. Purposes {banned[0]} and {banned[1]} are never allowed for confidential or restricted data.",
             f"P3. Any access to {[k for k, v in DATA.items() if v == need_mfa][0]} data or above requires MFA.",
             "P4. Contractors may not access data outside business hours (09:00-18:00).",
             "P5. An auditor with an open audit ticket may read anything for annual_audit, overriding P1 and P4.",
             "P6. Everything not allowed above is denied."]
    req = {"role": r.choice(ROLES), "data_class": r.choice(list(DATA)), "purpose": r.choice(PURPOSES),
           "mfa": r.random() < 0.7, "local_time": f"{r.randint(6, 22):02d}:{r.choice(['00', '15', '30', '45'])}",
           "open_audit_ticket": r.random() < 0.3}
    lvl = DATA[req["data_class"]]
    hh = int(req["local_time"][:2]) + int(req["local_time"][3:]) / 60
    if req["role"] == "auditor" and req["open_audit_ticket"] and req["purpose"] == "annual_audit":
        if lvl >= 2 and req["purpose"] in banned:
            deciding, ok = "P2", False
        elif lvl >= need_mfa and not req["mfa"]:
            deciding, ok = "P3", False
        else:
            deciding, ok = "P5", True
    elif lvl > clear[req["role"]]:
        deciding, ok = "P1", False
    elif lvl >= 2 and req["purpose"] in banned:
        deciding, ok = "P2", False
    elif lvl >= need_mfa and not req["mfa"]:
        deciding, ok = "P3", False
    elif req["role"] == "contractor" and not (9 <= hh < 18):
        deciding, ok = "P4", False
    else:
        deciding, ok = "P1", True
    state = {"access_policy": rules, "access_request": req}
    parent = f"m2:jev1synth:access_policy:{i}"
    if r.random() < 0.75:
        q = noul_q(pick(r, ["Is this access request permitted by the policy?", "Should the access be granted?"]),
                   "the policy permits the request", "the policy denies the request")
        return record("policy", "access_policy_yn", "jev1synth", parent, "access-yn", state, q, ok, {"request": req},
                      split="synthetic", gen="jev1_access_policy")
    opts = {"allow": "allow: the request is permitted", "deny_clearance": "deny: role clearance too low (P1)",
            "deny_purpose": "deny: purpose banned for this data class (P2)", "deny_mfa": "deny: MFA required (P3)",
            "deny_hours": "deny: contractor outside business hours (P4)"}
    gold = "allow" if ok else {"P1": "deny_clearance", "P2": "deny_purpose", "P3": "deny_mfa", "P4": "deny_hours"}[deciding]
    q, g = choice_q("What is the policy decision for this request, and on which ground? (When several grounds deny, "
                    "the lowest-numbered rule is the ground; P5 can only override P1 and P4.)", opts, gold, r)
    return record("policy", "access_policy_ground", "jev1synth", parent, "access-ground", state, q, g,
                  {"request": req}, split="synthetic", gen="jev1_access_policy")


# ================================================================================================ 4. intent
def _intent_rows(source, parent, text_key, text, gold, labels, desc, r, extra_state=None, oos=None):
    k = r.randint(6, 16)
    others = [x for x in labels if x != gold]
    r.shuffle(others)
    ids = [gold] + others[: k - 1]
    if oos and oos not in ids and r.random() < 0.3:
        ids[-1] = oos
    opts = {x: desc(x) for x in ids}
    st = {text_key: text, **(extra_state or {})}
    if r.random() < 0.8:
        q, g = choice_q(pick(r, ["What is the user's intent?", "Classify the message into one of the intents.",
                                 "Which intent label fits this message best?"]), opts, gold, r)
        return record("intent", f"{source}_intent", source, parent, "intent", st, q, g, {text_key: text})
    cand = gold if r.random() < 0.5 else ids[1]
    q = noul_q(f"Is the user's intent '{desc(cand)}'?")
    return record("intent", f"{source}_intent_yn", source, parent, "intent-yn", st, q, cand == gold, {text_key: text})


def gen_intent(n_massive: int, n_banking: int, n_clinc: int, massive: list[dict]) -> Iterator[dict]:
    pool = [x for x in massive if part("massive:" + x["id"], 3) == 1]
    labs = sorted({x["intent"] for x in massive})
    for x in pool[:n_massive]:
        r = rng("int-massive", x["id"])
        yield _intent_rows("massive", f"m2:massive:{x['id']}", "utterance", x["utt"], x["intent"], labs,
                           lambda s: s.replace("_", " "), r)
    br = banking_rows()
    labs = sorted({x["category"] for x in br})
    order = list(range(len(br)))
    rng("bank-order").shuffle(order)
    for i in order[:n_banking]:
        r = rng("int-bank", i)
        yield _intent_rows("banking77", f"m2:banking77:{i}", "customer_message", br[i]["text"], br[i]["category"],
                           labs, lambda s: s.replace("_", " "), r)
    cr = clinc_rows()
    labs = sorted({l for _, l, _ in cr})
    order = list(range(len(cr)))
    rng("clinc-order").shuffle(order)
    desc = lambda s: "out of scope: none of the supported intents" if s == "oos" else s.replace("_", " ")
    oos_idx = [i for i in range(len(cr)) if cr[i][1] == "oos"]
    take = order[:n_clinc]
    for i in take:
        r = rng("int-clinc", i)
        t, l, split = cr[i]
        rec = _intent_rows("clinc150", f"m2:clinc150:{i}", "user_request", t, l, [x for x in labs if x != "oos"] if
                           l != "oos" else [x for x in labs if x != "oos"] + ["oos"], desc, r, oos="oos")
        rec["source_split"] = split
        yield rec
    _ = oos_idx


# ================================================================================================ 5. severity
SEV_DOMAINS = {
    "it_outage": {"fields": ["users_affected", "minutes_down", "workaround", "revenue_impacting"],
                  "levels": ["SEV-5 informational: no user impact", "SEV-4 minor: fewer than {a} users or a "
                             "workaround exists", "SEV-3 moderate: {a}+ users without a workaround",
                             "SEV-2 major: {b}+ users, or down {m}+ minutes", "SEV-1 critical: revenue-impacting "
                             "outage affecting {b}+ users"]},
    "security": {"fields": ["records_exposed", "credentials_leaked", "active_attacker", "external_report"],
                 "levels": ["low: no data exposed", "guarded: under {a} records exposed, no credentials",
                            "elevated: {a}+ records or an external report", "high: credentials leaked",
                            "severe: credentials leaked AND an active attacker"]},
    "complaint": {"fields": ["customer_tier", "repeat_contact", "threatens_churn", "regulator_mentioned"],
                  "levels": ["routine: first contact, no risk signals", "watch: repeat contact",
                             "priority: threatens to leave, or an enterprise customer with a repeat contact",
                             "urgent: mentions a regulator or ombudsman"]},
    "facility": {"fields": ["hazard", "people_exposed", "area_closed", "injury"],
                 "levels": ["cosmetic: no hazard", "minor: a hazard, but fewer than {a} people exposed or the "
                            "area closed", "serious: hazard with {a}+ people exposed and the area still open",
                            "critical: any injury"]},
    "finance": {"fields": ["discrepancy_usd", "recurring", "customer_funds", "auditor_flagged"],
                "levels": ["trivial: under ${a}", "material: ${a}+, not recurring and no customer funds",
                           "significant: ${a}+ and either recurring or involving customer funds",
                           "critical: flagged by the auditor and ${b}+"]},
}


def _sev_level(dom: str, f: dict, p: dict) -> int:
    a, b, m = p["a"], p["b"], p["m"]
    if dom == "it_outage":
        if f["revenue_impacting"] and f["users_affected"] >= b:
            return 4
        if f["users_affected"] >= b or f["minutes_down"] >= m:
            return 3
        if f["users_affected"] >= a and not f["workaround"]:
            return 2
        return 1 if f["users_affected"] > 0 else 0
    if dom == "security":
        if f["credentials_leaked"] and f["active_attacker"]:
            return 4
        if f["credentials_leaked"]:
            return 3
        if f["records_exposed"] >= a or f["external_report"]:
            return 2
        return 1 if f["records_exposed"] > 0 else 0
    if dom == "complaint":
        if f["regulator_mentioned"]:
            return 3
        if f["threatens_churn"] or (f["customer_tier"] == "enterprise" and f["repeat_contact"]):
            return 2
        return 1 if f["repeat_contact"] else 0
    if dom == "facility":
        if f["injury"]:
            return 3
        if f["hazard"] and f["people_exposed"] >= a and not f["area_closed"]:
            return 2
        return 1 if f["hazard"] else 0
    if f["auditor_flagged"] and f["discrepancy_usd"] >= b:
        return 3
    if f["discrepancy_usd"] >= a and (f["recurring"] or f["customer_funds"]):
        return 2
    return 1 if f["discrepancy_usd"] >= a else 0


def _near(r: random.Random, t: int, lo: int = 0) -> int:
    """A count near a threshold (either side) or far from it: the boundary cases a rubric reader must get right."""
    k = r.random()
    if k < 0.35:
        return max(lo, t + r.randint(-2, 2))
    if k < 0.65:
        return max(lo, int(t * r.uniform(0.05, 0.95)))
    return int(t * r.uniform(1.0, 6.0))


def _sev_fields(dom: str, r: random.Random, p: dict) -> dict:
    a, b, m = p["a"], p["b"], p["m"]
    ctx = {"region": r.choice(["us-east", "us-west", "eu-central", "ap-south", "sa-east", "eu-north", "ap-east"]),
           "detected_by": r.choice(["automated monitor", "customer report", "on-call engineer", "partner",
                                    "sampling review", "staff walk-through"]),
           "opened_hour_utc": r.randint(0, 23)}
    if dom == "it_outage":
        f = {"users_affected": _near(r, r.choice([a, b])) if r.random() < 0.9 else 0,
             "minutes_down": _near(r, m), "workaround": r.random() < 0.4, "revenue_impacting": r.random() < 0.3}
    elif dom == "security":
        f = {"records_exposed": _near(r, a) if r.random() < 0.8 else 0, "credentials_leaked": r.random() < 0.3,
             "active_attacker": r.random() < 0.3, "external_report": r.random() < 0.25,
             "systems_involved": r.randint(1, 9)}
    elif dom == "complaint":
        f = {"customer_tier": r.choice(["standard", "standard", "plus", "enterprise"]),
             "contacts_this_month": r.randint(1, 6), "threatens_churn": r.random() < 0.25,
             "regulator_mentioned": r.random() < 0.12, "account_age_months": r.randint(1, 120)}
        f["repeat_contact"] = f["contacts_this_month"] > 1
    elif dom == "facility":
        f = {"hazard": r.random() < 0.7, "people_exposed": _near(r, a), "area_closed": r.random() < 0.35,
             "injury": r.random() < 0.12, "hazard_type": r.choice(["wet floor", "exposed wiring", "blocked exit",
                                                                   "loose railing", "gas smell", "broken glass"])}
    else:
        f = {"discrepancy_usd": _near(r, r.choice([a, b]), 1), "recurring": r.random() < 0.35,
             "customer_funds": r.random() < 0.3, "auditor_flagged": r.random() < 0.3,
             "ledger_lines": r.randint(1, 40)}
    return {**f, **ctx}


SEV_NOTE = {"it_outage": "Incident on the {svc} service.", "security": "Alert raised by the {svc} monitoring.",
            "complaint": "Complaint about the {svc} product.", "facility": "Report from the {svc} site.",
            "finance": "Reconciliation finding in the {svc} ledger."}


def gen_severity(n_synth: int, n_sst: int) -> Iterator[dict]:
    doms = sorted(SEV_DOMAINS)
    for i in range(n_synth):
        r = rng("sev", i)
        dom = doms[i % len(doms)]
        p = {"a": r.choice([10, 20, 25, 40, 50, 75, 100, 150, 200]), "m": r.choice([10, 15, 20, 30, 45, 60, 90, 120])}
        p["b"] = p["a"] * r.choice([5, 10, 20, 50])
        if dom == "finance":
            p = {"a": r.choice([50, 100, 250, 500, 1000, 2500]), "m": 0}
            p["b"] = p["a"] * r.choice([5, 10, 20])
        if dom == "facility":
            p = {"a": r.choice([2, 3, 5, 8, 10, 15]), "b": 0, "m": 0}
        levels = [s.format(**p) for s in SEV_DOMAINS[dom]["levels"]]
        f = _sev_fields(dom, r, p)
        g = _sev_level(dom, f, p)
        rev = r.random() < 0.3
        if rev:
            levels, g = list(reversed(levels)), len(levels) - 1 - g
        st = {"rubric_rule": "Assign the highest level whose condition holds.", "report": {
            "summary": SEV_NOTE[dom].format(svc=r.choice(["checkout", "payroll", "north warehouse", "mobile", "EU",
                                                          "partner portal", "search", "loyalty"])), **f}}
        q = score_q(pick(r, ["Which severity level does the rubric assign to this report?", "Grade the report's "
                             "severity on the rubric.", "Apply the severity rubric to the report."]), levels)
        yield record("severity", f"sev_{dom}", "jev1synth", f"m2:jev1synth:severity:{i}", "sev", st, q, g,
                     {"report": st["report"]}, split="synthetic", gen="jev1_severity", reversed_scale=rev)
    rows = sst5_rows()
    order = list(range(len(rows)))
    rng("sst-order").shuffle(order)
    lv = ["very negative", "negative", "neutral", "positive", "very positive"]
    for i in order[:n_sst]:
        r = rng("sst", i)
        rev = r.random() < 0.5
        levels = list(reversed(lv)) if rev else lv
        g = int(rows[i]["label"])
        g = 4 - g if rev else g
        q = score_q(pick(r, ["How strongly negative or positive is this movie review excerpt?",
                             "Grade the sentiment intensity of the review on the scale."]), levels)
        yield record("severity", "sst5_intensity", "sst5", f"m2:sst5:{i}", "sev", {"review": rows[i]["text"]}, q, g,
                     {"review": rows[i]["text"]}, reversed_scale=rev)


# ================================================================================================ 6. enum extraction
SLOT_RE = re.compile(r"\[(\w+) : ([^\]]+)\]")


def gen_enum(n_massive: int, n_notice: int, massive: list[dict]) -> Iterator[dict]:
    pool = [x for x in massive if part("massive:" + x["id"], 3) == 2]
    all_types = sorted({m.group(1) for x in massive for m in SLOT_RE.finditer(x["annot_utt"])})
    k = 0
    for x in pool:
        if k >= n_massive:
            break
        sl = [(m.group(1), m.group(2).strip()) for m in SLOT_RE.finditer(x["annot_utt"])]
        if not sl:
            continue
        r = rng("enum-massive", x["id"])
        typ, val = r.choice(sl)
        parent = f"m2:massive:{x['id']}"
        if r.random() < 0.5 and len(set(norm(v) for _, v in sl)) == len(sl):
            # which slot type does the span fill?
            others = [t for t in all_types if t not in {t for t, _ in sl}]
            r.shuffle(others)
            ids = sorted({typ} | set(others[: r.randint(4, 9)]))
            opts = {t: t.replace("_", " ") for t in ids}
            q, g = choice_q(f"In the utterance, which slot type does the span \"{val}\" fill?", opts, typ, r)
            sub = "massive_slot_type"
        else:
            # which value fills this slot? options = the utterance's spans + other words + not mentioned
            same = [v for t, v in sl if t == typ]
            if len(same) != 1:
                continue
            words = [w for w in re.findall(r"[a-z']+", x["utt"].lower()) if norm(w) not in norm(val).split()]
            cands = {norm(v): v for t, v in sl if t != typ}
            r.shuffle(words)
            for w in words:
                if len(cands) >= 4:
                    break
                cands.setdefault(norm(w), w)
            cands.pop(norm(val), None)
            if len(cands) < 2:
                continue
            ids = {"value_0": val}
            for j, v in enumerate(list(cands.values())[:5]):
                ids[f"value_{j + 1}"] = v
            ids["not_mentioned"] = "the utterance does not give this slot"
            q, g = choice_q(f"What value does the utterance give for the slot '{typ.replace('_', ' ')}'?", ids,
                            "value_0", r)
            # neutral ids would leak the gold by name; rename to option ids that carry no position or identity
            names = list(q["criteria"])
            r.shuffle(names)
            ren = {old: f"opt_{chr(97 + j)}" if old != "not_mentioned" else "not_mentioned" for j, old in enumerate(names)}
            q["criteria"] = {ren[o]: d for o, d in q["criteria"].items()}
            g = ren[g]
            sub = "massive_slot_value"
        k += 1
        yield record("enum", sub, "massive", parent, "enum", {"utterance": x["utt"]}, q, g, {"utterance": x["utt"]},
                     slot=typ)
    for i in range(n_notice):
        yield _notice_enum(i)


ENUMS = {
    "carrier": ["UPS", "FedEx", "DHL", "USPS", "Royal Mail", "Canada Post", "DPD", "GLS"],
    "payment_method": ["credit card", "PayPal", "bank transfer", "gift card", "Apple Pay", "invoice (net 30)"],
    "currency": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "INR"],
    "priority": ["low", "normal", "high", "urgent"],
    "status": ["draft", "submitted", "approved", "rejected", "on hold", "cancelled"],
    "plan": ["Starter", "Growth", "Business", "Enterprise", "Nonprofit"],
}


def _notice_enum(i: int) -> dict:
    r = rng("notice", i)
    field = sorted(ENUMS)[i % len(ENUMS)]
    vals = r.sample(ENUMS[field], min(len(ENUMS[field]), r.randint(4, 6)))
    gold, old = vals[0], vals[1]
    ref = f"{r.choice(['PO', 'INV', 'ORD', 'REQ', 'TKT'])}-{r.randint(10000, 99999)}"
    lead = {"carrier": f"Your order {ref} was first booked with {old}. Because of a pickup delay it was reassigned and "
                       f"will now be delivered by {gold}.",
            "payment_method": f"The original charge for {ref} on {old} failed, so the balance was settled by {gold}.",
            "currency": f"Invoice {ref} was drafted in {old} but, at the client's request, the final amount is billed "
                        f"in {gold}.",
            "priority": f"Ticket {ref} was opened as {old} priority; after triage the team set it to {gold}.",
            "status": f"Request {ref} moved from {old} to {gold} this morning; no further change is expected.",
            "plan": f"The account behind {ref} trialled the {old} plan and has now subscribed to the {gold} plan."}[field]
    filler = pick(r, ["Please keep this message for your records.", "Reply to this email if anything looks wrong.",
                      "Thank you for your patience.", "This notice was generated automatically."])
    if r.random() < 0.3:  # the final value is negated elsewhere for one distractor
        lead += f" Note: it is not {vals[2]}, despite what an earlier email said."
    doc = f"{lead} {filler}"
    parent = f"m2:jev1synth:notice_enum:{i}"
    st = {"document_type": r.choice(["email", "system notice", "chat transcript", "memo"]), "text": doc}
    if field == "priority" and r.random() < 0.5:
        lv = ENUMS["priority"]
        q = score_q("What is the CURRENT priority stated in the document?", lv)
        return record("enum", "notice_priority_level", "jev1synth", parent, "enum-level", st, q, lv.index(gold),
                      {"text": doc}, split="synthetic", gen="jev1_notice_enum")
    if r.random() < 0.85:
        opts = {slug(v): v for v in vals}
        q, g = choice_q(f"What is the CURRENT {field.replace('_', ' ')} stated in the document?", opts, slug(gold), r)
        return record("enum", f"notice_{field}", "jev1synth", parent, "enum", st, q, g, {"text": doc},
                      split="synthetic", gen="jev1_notice_enum")
    cand = gold if r.random() < 0.5 else old
    q = noul_q(f"Is the current {field.replace('_', ' ')} '{cand}'?")
    return record("enum", f"notice_{field}_yn", "jev1synth", parent, "enum-yn", st, q, cand == gold, {"text": doc},
                  split="synthetic", gen="jev1_notice_enum")


# ================================================================================================ 7. long-policy
LP_FILLER = [
    "Definitions. 'Employee' means any person on the payroll, including part-time staff. 'Manager' means the person "
    "the employee reports to in the HR system. 'Business day' excludes weekends and public holidays.",
    "Scope. This policy applies to all regions unless a local addendum states otherwise. Local addenda may only make "
    "the rules stricter, never looser.",
    "Records. Supporting documents must be kept for seven years. Digital copies are acceptable if legible.",
    "Questions about this policy go to the policy owner. Interpretations issued by the policy owner are binding.",
    "Review. This policy is reviewed every year. Changes take effect on the first day of the following month.",
    "Conflicts. Where this policy conflicts with a contract signed by the company, the contract prevails only for "
    "the matters it expressly covers.",
    "Confidentiality. Information submitted under this policy is shared only with the people who need it to apply "
    "the policy.",
    "Training. New staff complete the policy module in their first month. Completion is recorded in the HR system.",
    "Purpose. This policy sets out how requests are decided so that similar cases are treated alike. It does not "
    "create a contract and may be changed by the policy owner with thirty days' notice to staff.",
    "Responsibilities. Requesters must give complete and accurate information. Approvers must apply the numbered "
    "rules as written and record the rule that decided each case. Finance and HR may audit any decision.",
    "Systems. Requests are filed in the self-service portal. Paper forms are accepted only when the portal is down "
    "for more than one business day, and must then be entered into the portal within five business days.",
    "Appeals. A requester may appeal a decision once, in writing, within twenty business days. The appeal is decided "
    "by someone who did not take the original decision. An appeal does not pause any deadline in this policy.",
    "Data protection. Personal data in requests is processed only to apply this policy and is deleted when the "
    "retention period ends. Requesters may ask for a copy of the data held about them.",
    "Misuse. Knowingly false requests are a disciplinary matter. Repayment of any amount obtained through a false "
    "request may be required in addition to any other action.",
    "Exceptions process. Anything not covered by the numbered rules goes to the policy owner, who decides in "
    "writing. Such decisions do not set a precedent unless the policy is amended.",
    "Language. Where a translated copy of this policy differs from the English text, the English text applies.",
    "Related documents. The code of conduct, the information security policy and the procurement policy apply "
    "alongside this policy. None of them changes the numbered rules below or above.",
    "Glossary. 'Request' means a single submission in the portal. 'Case' means the facts of a request as recorded "
    "at the time of the decision. 'Approver' means the person the portal assigns to decide the request.",
    "Timing. Days are counted in calendar days unless a rule says business days. A period ending on a weekend ends "
    "on the next business day.",
    "Contractors and agency staff. Unless a rule names them, contractors follow the procedure in their own contract "
    "and are outside the scope of the numbered rules.",
    "Reporting. The policy owner publishes a quarterly summary of decisions by rule number, without personal data.",
    "Transition. Requests filed before this version took effect are decided under the version in force when they "
    "were filed.",
    "Severability. If any rule is found invalid, the remaining rules continue to apply.",
    "Contacts. Questions go to the policy mailbox. The mailbox is monitored on business days and replies within "
    "three business days.",
]
LP_DISTRACT = {
    "expense": ["X1. Relocation costs are covered by the relocation policy, not by this policy.",
                "X2. Conference fees are paid directly by the events team and must not be claimed here.",
                "X3. Mileage for personal vehicles is reimbursed at the published per-kilometre rate under the "
                "vehicle policy.", "X4. Corporate-card statements must be reconciled monthly; this does not change "
                "what is reimbursable."],
    "returns": ["X1. Business customers on a negotiated contract follow the returns clause of that contract.",
                "X2. Items damaged in transit are handled as a carrier claim, which is outside this policy.",
                "X3. Price adjustments after a sale are covered by the price-match policy.",
                "X4. Exchanges for a different size count as a return plus a new purchase."],
    "leave": ["X1. Parental, jury and bereavement leave follow their own policies and do not use the paid-leave "
              "balance.", "X2. Unpaid leave requires director approval and is outside these rules.",
              "X3. Public holidays are not deducted from the paid-leave balance.",
              "X4. Time off in lieu for overtime is tracked separately and does not affect the paid-leave balance."],
}


def _lp_expense(r: random.Random) -> tuple[list[str], dict, dict]:
    cap = {"meals": r.choice([40, 50, 60, 75]), "hotel": r.choice([150, 180, 220, 250]), "taxi": r.choice([30, 40, 60])}
    rec_thr = r.choice([25, 50, 75])
    pre = r.choice([500, 750, 1000, 1500])
    late = r.choice([30, 45, 60, 90])
    clauses = [
        f"E1. Meals are reimbursed up to ${cap['meals']} per day; the excess is not reimbursed.",
        f"E2. Hotel nights are reimbursed up to ${cap['hotel']} per night; the excess is not reimbursed.",
        f"E3. Taxi and ride-share trips are reimbursed up to ${cap['taxi']} per trip.",
        f"E4. Any single item of ${rec_thr} or more needs a receipt; an item without a required receipt is rejected.",
        f"E5. A trip whose total claim exceeds ${pre} needs manager pre-approval; without it the whole claim is "
        "rejected.",
        f"E6. Claims submitted more than {late} days after the trip ends are rejected in full.",
        "E7. Alcohol is never reimbursed, even inside the meal cap.",
        "E8. A director may waive E1-E3 caps for a named client event; the waiver must be on file before the trip.",
        "E9. When several rules apply to one item, apply rejections first, then caps.",
    ]
    items = []
    for j in range(r.randint(2, 4)):
        cat = r.choice(["meals", "hotel", "taxi", "meals"])
        amt = r.choice([12, 18, 22, 35, 48, 55, 64, 80, 95, 140, 170, 210, 240, 290])
        items.append({"item": j + 1, "category": cat, "amount_usd": amt, "receipt": r.random() < 0.75,
                      "includes_alcohol_usd": r.choice([0, 0, 0, 8, 15]) if cat == "meals" else 0})
    case = {"trip_days": r.randint(1, 5), "days_since_trip_end": r.choice([3, 10, 20, late - 1, late, late + 5]),
            "manager_preapproval": r.random() < 0.6, "director_waiver_on_file": r.random() < 0.15, "items": items}
    total = sum(x["amount_usd"] for x in items)
    whole = None
    if case["days_since_trip_end"] > late:
        whole = "E6"
    elif total > pre and not case["manager_preapproval"]:
        whole = "E5"
    paid = []
    for x in items:
        if whole:
            paid.append(0)
            continue
        if x["amount_usd"] >= rec_thr and not x["receipt"]:
            paid.append(0)
            continue
        a = x["amount_usd"] - x["includes_alcohol_usd"]
        if not case["director_waiver_on_file"]:
            a = min(a, cap[x["category"]])
        paid.append(a)
    full = [p == x["amount_usd"] for p, x in zip(paid, items)]
    outcome = ("reject_all" if whole else "approve_full" if all(full) else "reject_all" if sum(paid) == 0
               else "approve_partial")
    return clauses, case, {"outcome": outcome, "paid": paid, "full": full, "items": items, "total_paid": sum(paid)}


def _lp_returns(r: random.Random) -> tuple[list[str], dict, dict]:
    win = r.choice([14, 30, 60])
    ext = r.choice([15, 30])
    fee = r.choice([10, 15, 20])
    clauses = [
        f"R1. Items may be returned within {win} days of delivery for a full refund.",
        f"R2. Members get {ext} extra days on the R1 window.",
        "R3. Opened software, gift cards and personalised items cannot be returned.",
        f"R4. Opened electronics carry a {fee}% restocking fee unless defective.",
        "R5. Defective items may be returned within one year of delivery, with no fee, overriding R1-R2.",
        "R6. Items bought on final sale cannot be returned unless defective.",
        "R7. Returns without proof of purchase receive store credit instead of a refund.",
        "R8. When R3 applies, no other rule can make the item returnable.",
    ]
    case = {"category": r.choice(["electronics", "clothing", "software", "gift card", "furniture", "personalised mug"]),
            "days_since_delivery": r.choice([3, 10, win - 1, win, win + 5, win + ext - 1, win + ext + 3, 200, 400]),
            "member": r.random() < 0.4, "opened": r.random() < 0.5, "defective": r.random() < 0.25,
            "final_sale": r.random() < 0.2, "proof_of_purchase": r.random() < 0.8}
    c = case
    blocked3 = c["category"] in ("gift card", "personalised mug") or (c["category"] == "software" and c["opened"])
    if blocked3:
        out = "not_returnable"
    elif c["defective"]:
        out = "full_refund" if c["days_since_delivery"] <= 365 else "not_returnable"
    elif c["final_sale"]:
        out = "not_returnable"
    else:
        w = win + (ext if c["member"] else 0)
        if c["days_since_delivery"] > w:
            out = "not_returnable"
        elif c["category"] == "electronics" and c["opened"]:
            out = "refund_minus_fee"
        else:
            out = "full_refund"
    if out in ("full_refund", "refund_minus_fee") and not c["proof_of_purchase"]:
        out = "store_credit"
    return clauses, case, {"outcome": out}


def _lp_leave(r: random.Random) -> tuple[list[str], dict, dict]:
    accrual = r.choice([1.5, 1.75, 2.0, 2.5])
    carry = r.choice([5, 8, 10])
    notice = r.choice([5, 10, 14])
    clauses = [
        f"L1. Staff accrue {accrual} days of paid leave per full month worked.",
        f"L2. Up to {carry} unused days carry over to the next year; the rest lapse on 1 January.",
        f"L3. Requests of more than 3 consecutive days need {notice} days' notice.",
        "L4. During a declared blackout period, leave is approved only for emergencies.",
        "L5. Staff in their probation period may take at most 3 days of paid leave in total.",
        "L6. A request is approved only if the balance (accrued + carried over - taken) covers it.",
        "L7. Emergency leave of up to 2 days is always approved and does not need notice.",
    ]
    months = r.randint(1, 12)
    carried = r.choice([0, 2, carry, carry + 4])
    case = {"months_worked_this_year": months, "days_carried_from_last_year": carried,
            "days_taken_this_year": r.choice([0, 2, 5, 9, 14]), "requested_days": r.choice([1, 2, 3, 4, 5, 8, 12]),
            "notice_days": r.choice([1, 3, notice - 1, notice, 20]), "blackout": r.random() < 0.2,
            "emergency": r.random() < 0.15, "on_probation": r.random() < 0.2}
    c = case
    bal = accrual * months + min(carried, carry) - c["days_taken_this_year"]
    if c["emergency"] and c["requested_days"] <= 2:
        ok, why = True, "L7"
    elif c["blackout"] and not c["emergency"]:
        ok, why = False, "L4"
    elif c["on_probation"] and c["days_taken_this_year"] + c["requested_days"] > 3:
        ok, why = False, "L5"
    elif c["requested_days"] > 3 and c["notice_days"] < notice:
        ok, why = False, "L3"
    elif c["requested_days"] > bal:
        ok, why = False, "L6"
    else:
        ok, why = True, "L6"
    return clauses, case, {"ok": ok, "why": why, "balance": bal}


def gen_long_policy(n: int) -> Iterator[dict]:
    kinds = ["expense", "returns", "leave"]
    for i in range(n):
        r = rng("lp", i)
        kind = kinds[i % 3]
        clauses, case, gold = {"expense": _lp_expense, "returns": _lp_returns, "leave": _lp_leave}[kind](r)
        fill = r.sample(LP_FILLER, r.randint(12, 20))
        dis = r.sample(LP_DISTRACT[kind], r.randint(2, 4))
        cut = r.randint(len(fill) // 3, 2 * len(fill) // 3)
        sections = fill[:cut] + clauses + dis + fill[cut:]
        title = {"expense": "Travel and Expense Reimbursement Policy", "returns": "Customer Returns Policy",
                 "leave": "Paid Leave Policy"}[kind]
        policy = f"{title} (version {r.randint(2, 9)}.{r.randint(0, 9)})\n\n" + "\n\n".join(sections)
        st = {"policy_document": policy, "case": case}
        parent = f"m2:jev1synth:long_policy:{i}"
        t = r.random()
        if kind == "expense":
            if t < 0.35:
                opts = {"approve_full": "approve the whole claim as submitted", "approve_partial": "approve part of the "
                        "claim (some items capped or rejected)", "reject_all": "reject the whole claim"}
                q, g = choice_q("Applying the policy to the case, what is the outcome of the claim?", opts,
                                gold["outcome"], r)
            elif t < 0.6:
                j = r.randrange(len(gold["items"]))
                q = noul_q(f"Is item {j + 1} reimbursed in full under the policy?")
                g = gold["full"][j]
            elif t < 0.85:
                claimed = sum(x["amount_usd"] for x in gold["items"])
                sh = gold["total_paid"] / claimed
                lvl = 0 if sh == 0 else 1 if sh < 0.5 else 2 if sh < 1 else 3
                q = score_q("What share of the claimed total does the policy reimburse?",
                            ["nothing", "less than half of the claimed total", "at least half, but not all",
                             "the full claimed total"])
                g = lvl
            else:
                tot = gold["total_paid"]
                cands = sorted({tot, sum(x["amount_usd"] for x in gold["items"]), max(0, tot - 10), tot + 15,
                                round(tot / 2)})
                opts = {f"usd_{c}": f"${c}" for c in cands}
                if len(opts) < 3:
                    opts[f"usd_{tot + 40}"] = f"${tot + 40}"
                q, g = choice_q("What total amount does the policy reimburse?", opts, f"usd_{tot}", r)
        elif kind == "returns":
            if t < 0.7:
                opts = {"full_refund": "full refund", "refund_minus_fee": "refund minus the restocking fee",
                        "store_credit": "store credit only", "not_returnable": "the item cannot be returned"}
                q, g = choice_q("What does the returns policy allow for this case?", opts, gold["outcome"], r)
            else:
                q = noul_q("Can the customer get money back (a refund of any size, not store credit) for this item?")
                g = gold["outcome"] in ("full_refund", "refund_minus_fee")
        else:
            if t < 0.6:
                q = noul_q("Should the leave request be approved under the policy?")
                g = gold["ok"]
            else:
                ids = ["L3", "L4", "L5", "L6", "L7"]
                opts = {x: f"clause {x} decides the request" for x in ids}
                q, g = choice_q("Which clause decides this request? (Check L7, then L4, L5, L3 and finally L6.)", opts,
                                gold["why"], r)
        yield record("long_policy", f"lp_{kind}", "jev1synth", parent, "lp", st, q, g, {"case": case},
                     split="synthetic", gen="jev1_long_policy", kind=kind)


# ================================================================================================ 8. state tracking
def gen_state(n_woz: int, n_events: int) -> Iterator[dict]:
    dias = list(multiwoz_dialogues())
    vals: dict[str, set] = {}
    for d in dias:
        for t in d["turns"]:
            for fr in t.get("frames", []):
                for s, v in fr.get("state", {}).get("slot_values", {}).items():
                    vals.setdefault(s, set()).add(v[0])
    vals = {s: sorted(v) for s, v in vals.items()}
    order = list(range(len(dias)))
    rng("woz-order").shuffle(order)
    k = 0
    for di in order:
        if k >= n_woz:
            break
        d = dias[di]
        r = rng("woz", d["dialogue_id"])
        users = [j for j, t in enumerate(d["turns"]) if t["speaker"] == "USER" and j >= 2]
        if not users:
            continue
        j = r.choice(users)
        sv = {}
        for fr in d["turns"][j].get("frames", []):
            sv.update(fr.get("state", {}).get("slot_values", {}))
        if not sv:
            continue
        hist = [{"speaker": t["speaker"].lower(), "text": t["utterance"]} for t in d["turns"][: j + 1]]
        if sum(len(h["text"]) for h in hist) > 5000:
            continue
        slot = r.choice(sorted(sv))
        gold_v = sv[slot][0]
        gvars = {norm(x) for x in sv[slot]}
        pool = list({norm(v): v for v in vals.get(slot, []) if norm(v) not in gvars and norm(v)}.values())
        if len(pool) < 3:
            continue
        parent = f"m2:multiwoz22:{d['dialogue_id'].replace('.json', '')}:{j}"
        sname = slot.replace("-", " ").replace("_", " ")
        if r.random() < 0.7:
            dist = r.sample(pool, min(len(pool), r.randint(3, 6)))
            opts = {"value_0": gold_v}
            for z, v in enumerate(dist):
                opts[f"value_{z + 1}"] = v
            opts["not_set"] = "the user has not specified this yet"
            q, g = choice_q(f"After the last user turn, what is the dialogue state value of '{sname}'?", opts,
                            "value_0", r)
            names = list(q["criteria"])
            r.shuffle(names)
            ren = {old: (f"opt_{chr(97 + z)}" if old != "not_set" else "not_set") for z, old in enumerate(names)}
            q["criteria"] = {ren[o]: v for o, v in q["criteria"].items()}
            g = ren[g]
            sub = "woz_value"
        else:
            cand = gold_v if r.random() < 0.5 else r.choice(pool)
            q = noul_q(f"After the last user turn, is '{sname}' set to '{cand}' in the dialogue state?")
            g = norm(cand) in gvars
            sub = "woz_value_yn"
        k += 1
        yield record("state_tracking", sub, "multiwoz22", parent, "dst", {"dialogue": hist}, q, g,
                     {"dialogue": [h["text"] for h in hist]}, slot=slot)
    for i in range(n_events):
        yield _event_state(i)


def _event_state(i: int) -> dict:
    r = rng("events", i)
    kind = ["order", "ticket", "reservation", "account"][i % 4]
    ev, st = [], {}
    if kind == "order":
        items = {}
        status = "created"
        addr = r.choice(["12 Elm St", "4 Harbour Rd", "88 King Ave"])
        cat = ["mug", "lamp", "notebook", "backpack", "kettle", "scarf", "pen set", "tripod"]
        for t in range(r.randint(5, 11)):
            a = r.random()
            if status == "cancelled" and a < 0.85:
                if r.random() < 0.3:
                    ev.append({"t": t, "event": "order_reopened"})
                    status = "created"
                continue
            if a < 0.4:
                it = r.choice(cat)
                n = r.randint(1, 3)
                items[it] = items.get(it, 0) + n
                ev.append({"t": t, "event": "item_added", "item": it, "qty": n})
            elif a < 0.55 and items:
                it = r.choice(sorted(items))
                n = r.randint(1, items[it])
                items[it] -= n
                if not items[it]:
                    del items[it]
                ev.append({"t": t, "event": "item_removed", "item": it, "qty": n})
            elif a < 0.68:
                addr = r.choice(["12 Elm St", "4 Harbour Rd", "88 King Ave", "7 Mill Lane", "301 Bay Blvd"])
                ev.append({"t": t, "event": "address_changed", "to": addr})
            elif a < 0.8 and status in ("created", "paid"):
                status = {"created": "paid", "paid": "shipped"}[status]
                ev.append({"t": t, "event": f"status_{status}"})
            elif a < 0.88 and status != "shipped":
                status = "cancelled"
                ev.append({"t": t, "event": "order_cancelled"})
            else:
                ev.append({"t": t, "event": "customer_note", "text": r.choice(["leave at the door", "gift wrap please",
                                                                               "call on arrival"])})
        st = {"status": status, "items": items, "address": addr}
        qs = [("status", ["created", "paid", "shipped", "cancelled"]), ("address", None), ("units", None)]
    elif kind == "ticket":
        status, owner, prio = "new", "unassigned", "normal"
        people = r.sample(["ana", "bo", "chen", "dev", "eli", "fay"], 4)
        for t in range(r.randint(5, 11)):
            a = r.random()
            if a < 0.3:
                owner = r.choice(people)
                ev.append({"t": t, "event": "assigned", "to": owner})
            elif a < 0.5:
                prio = r.choice(["low", "normal", "high", "urgent"])
                ev.append({"t": t, "event": "priority_set", "to": prio})
            elif a < 0.75:
                status = r.choice(["open", "pending_customer", "resolved", "closed"])
                ev.append({"t": t, "event": "status_changed", "to": status})
            elif a < 0.85 and status in ("resolved", "closed"):
                status = "open"
                ev.append({"t": t, "event": "reopened_by_customer"})
            else:
                ev.append({"t": t, "event": "comment", "by": r.choice(people)})
        st = {"status": status, "owner": owner, "priority": prio}
        qs = [("status", ["new", "open", "pending_customer", "resolved", "closed"]), ("owner", ["unassigned"] + people),
              ("priority", ["low", "normal", "high", "urgent"])]
    elif kind == "reservation":
        guests, room, status = r.randint(1, 4), r.choice(["standard", "deluxe", "suite"]), "held"
        nights = r.randint(1, 5)
        for t in range(r.randint(4, 10)):
            a = r.random()
            if a < 0.25:
                guests = r.randint(1, 6)
                ev.append({"t": t, "event": "guest_count_changed", "to": guests})
            elif a < 0.45:
                room = r.choice(["standard", "deluxe", "suite", "family"])
                ev.append({"t": t, "event": "room_type_changed", "to": room})
            elif a < 0.6:
                nights = r.randint(1, 7)
                ev.append({"t": t, "event": "nights_changed", "to": nights})
            elif a < 0.75:
                status = r.choice(["confirmed", "held", "cancelled"])
                ev.append({"t": t, "event": f"reservation_{status}"})
            else:
                ev.append({"t": t, "event": "special_request", "text": r.choice(["late checkout", "crib", "quiet room"])})
        st = {"status": status, "room_type": room, "guests": guests, "nights": nights}
        qs = [("status", ["held", "confirmed", "cancelled"]), ("room_type", ["standard", "deluxe", "suite", "family"]),
              ("guests", [str(x) for x in range(1, 7)])]
    else:
        bal, frozen, tier = r.choice([0, 50, 120]), False, r.choice(["basic", "plus"])
        for t in range(r.randint(5, 11)):
            a = r.random()
            if a < 0.35 and not frozen:
                x = r.choice([10, 20, 25, 40, 75])
                bal += x
                ev.append({"t": t, "event": "deposit", "usd": x})
            elif a < 0.6 and not frozen:
                x = r.choice([5, 15, 30, 60])
                if x <= bal:
                    bal -= x
                    ev.append({"t": t, "event": "withdrawal", "usd": x})
                else:
                    ev.append({"t": t, "event": "withdrawal_declined", "usd": x, "reason": "insufficient funds"})
            elif a < 0.7 and frozen:
                x = r.choice([10, 30])
                ev.append({"t": t, "event": "withdrawal_declined", "usd": x, "reason": "account frozen"})
            elif a < 0.8:
                frozen = not frozen
                ev.append({"t": t, "event": "account_frozen" if frozen else "account_unfrozen"})
            else:
                tier = r.choice(["basic", "plus", "premium"])
                ev.append({"t": t, "event": "tier_changed", "to": tier})
        st = {"balance": bal, "frozen": frozen, "tier": tier}
        qs = [("balance", None), ("frozen", None), ("tier", ["basic", "plus", "premium"])]
    field, choices = r.choice(qs)
    state = {"entity": kind, "event_log": ev, "note": "Events are in time order; later events override earlier ones."}
    parent = f"m2:jev1synth:event_state:{i}"
    if field == "units":
        tot = sum(st["items"].values())
        cands = sorted({tot, tot + 1, max(0, tot - 1), tot + 2, tot + 3} - ({-1} if tot else set()))
        opts = {f"n_{c}": str(c) for c in cands}
        q, g = choice_q("How many units are in the order after the last event?", opts, f"n_{tot}", r)
    elif field == "address":
        allv = ["12 Elm St", "4 Harbour Rd", "88 King Ave", "7 Mill Lane", "301 Bay Blvd"]
        opts = {slug(v): v for v in allv}
        q, g = choice_q("What is the delivery address after the last event?", opts, slug(st["address"]), r)
    elif field == "balance":
        b = st["balance"]
        cands = sorted({b, b + 10, max(0, b - 15), b + 25, b + 5})
        opts = {f"usd_{c}": f"${c}" for c in cands}
        q, g = choice_q("What is the account balance after the last event?", opts, f"usd_{b}", r)
    elif field == "frozen":
        q = noul_q("Is the account frozen after the last event?")
        g = st["frozen"]
    elif field == "priority" and r.random() < 0.6:
        lv = ["low", "normal", "high", "urgent"]
        q = score_q("What is the ticket priority after the last event?", lv)
        g = lv.index(st["priority"])
    else:
        cur = str(st[field])
        if r.random() < 0.3:
            cand = cur if r.random() < 0.5 else r.choice([c for c in choices if c != cur])
            q = noul_q(f"After the last event, is the {kind} {field.replace('_', ' ')} '{cand}'?")
            g = cand == cur
        else:
            opts = {slug(c): c for c in choices}
            q, g = choice_q(f"What is the {kind} {field.replace('_', ' ')} after the last event?", opts, slug(cur), r)
    return record("state_tracking", f"events_{kind}", "jev1synth", parent, "dst", state, q, g, {"event_log": ev},
                  split="synthetic", gen="jev1_event_state", field=field)


# ================================================================================================ the pool
#: per-family sub-targets (rows generated; the finalize step trims each family to its target after the gates)
PLAN = {
    "routing": {"massive": 3900, "ticket": 4400, "model": 4400},
    "adequacy": {"score": 3200, "noul": 2600, "pref": 3000, "audit": 5600},
    "policy": {"aegis": 7600, "access": 5100},
    "intent": {"massive": 3800, "banking": 6400, "clinc": 6800},
    "severity": {"synth": 10200, "sst": 3100},
    "enum": {"massive": 3700, "notice": 10800},
    "long_policy": {"n": 12700},
    "state_tracking": {"woz": 8400, "events": 9600},
}


def generate(families=FAMILIES) -> Iterator[dict]:
    need_m = any(f in families for f in ("routing", "intent", "enum"))
    massive = massive_rows() if need_m else []
    p = PLAN
    if "routing" in families:
        yield from gen_routing(p["routing"]["massive"], p["routing"]["ticket"], p["routing"]["model"], massive)
    if "adequacy" in families:
        a = p["adequacy"]
        yield from gen_adequacy(a["score"], a["noul"], a["pref"], a["audit"], helpsteer2_rows(), hs2pref_rows())
    if "policy" in families:
        yield from gen_policy(p["policy"]["aegis"], p["policy"]["access"], aegis_rows())
    if "intent" in families:
        yield from gen_intent(p["intent"]["massive"], p["intent"]["banking"], p["intent"]["clinc"], massive)
    if "severity" in families:
        yield from gen_severity(p["severity"]["synth"], p["severity"]["sst"])
    if "enum" in families:
        yield from gen_enum(p["enum"]["massive"], p["enum"]["notice"], massive)
    if "long_policy" in families:
        yield from gen_long_policy(p["long_policy"]["n"])
    if "state_tracking" in families:
        yield from gen_state(p["state_tracking"]["woz"], p["state_tracking"]["events"])
