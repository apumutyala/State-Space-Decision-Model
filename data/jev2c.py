"""bschema-draw-jev2 stage c: long_policy rebuilt for the eval regime (JEV1_AUDIT.md fixes, claude/ev-jev1-audit).

The audit found that jev1 long_policy rows were too short and the wrong shape compared with the public long_policy
items. jev1 rows: p50 934 / max 1,262 shim tokens; a dict state with typed case fields; three templates; 66% "no"
labels; empty noul criteria; one instruction that lists the rule order. Public long_policy items (shape only): plain
strings of 2,350-3,952 tokens with explicit noul criteria.

This slice:

- **State.** A single plain-text record: a policy handbook in prose, with the case narrated in prose at the end. The
  sampled document length is 8,500-15,500 characters (about 2,000-4,000 shim tokens); the build keeps
  2,000 <= tokens <= 4,096.
- **Domains.** 12 policy domains, each an executable rule engine with overrides, exceptions and a dated amendment
  (applies only when it is in force on the case date). Gold comes from the engine.
- **Distractors.** One or two other domains' chapters (rules only, no case) and generated administrative sections pad
  the document. The target chapter's position is random.
- **Questions.** noul (explicit true/false descriptions, paraphrased) or a choice over the domain's outcomes (K 3-5).
- **Labels.** Gold is drawn first, then the case is resampled until it matches: noul 50/50 per domain, choice
  uniform over outcomes.
- **Instructions.** Prefix x core x suffix paraphrases (>= 200 distinct strings), and none names a rule order.
- **ContractNLI train** (CC-BY-4.0; HF mirror kiddothe2b/contract-nli, contract_nli_long; ``subset == train``
  only). The full NDA (8,000-15,500 characters) is the plain-string state, and the task is a 3-way
  entailed / contradicted / not-mentioned choice. Labels are balanced by per-label quotas.
"""
from __future__ import annotations

import json
import zipfile
from typing import Callable, Iterator

from data.jev1 import LP_FILLER, choice_q, noul_q, score_q
from data.jev2 import CITIES, FIRST, LAST, LIC, RAW, _pname, money, record, rng

GEN_SPECS_C = {"jev2_long_handbook": {"name": "jev2_long_handbook", "task_family": "long_handbook_rule_engine"}}
LIC.update({"contractnli": "CC-BY-4.0 (ContractNLI, Koreeda & Manning 2021; HF mirror kiddothe2b/contract-nli)"})

PREFIX = ["", "Based on the handbook, ", "Apply the policy text to the case. ", "Read the whole document first. ",
          "Using only the rules written above, ", "As the reviewer on duty, ", "Under the version of the policy in "
          "force for this case, ", "Taking every applicable clause into account, ", "For the case described at the "
          "end, ", "Going by the written policy, "]
SUFFIX = ["", " Answer from the policy alone.", " Unwritten practice does not count.", " Base the answer on the "
          "document.", " Consider exceptions and amendments where they apply.", " Decide as the policy requires.",
          " Treat the case facts as accurate."]
ADMIN_T = ["Workplace safety", "Information security", "Visitors", "Parking", "Equipment loans", "Office hours",
           "Document retention", "Gifts and hospitality", "Conflicts of interest", "Social media", "Accessibility",
           "Emergency procedures", "Printing and post", "Meeting rooms", "Learning budget", "Performance reviews",
           "Recruitment", "Onboarding", "Offboarding", "Whistleblowing", "Environmental practices", "Clean desk",
           "Lost property", "Team events", "Mentoring", "Volunteering days"]
S1 = ["Staff must follow the instructions posted at each site.", "Line managers explain this section to new "
      "joiners.", "The facilities team keeps the current forms.", "Breaches are handled under the disciplinary "
      "procedure.", "Requests are logged in the self-service portal.", "Local law prevails where it sets a higher "
      "standard.", "Records created under this section are kept for three years.", "Questions go to the section "
      "owner named on the intranet.", "This section is reviewed every two years.", "Guidance notes may be issued "
      "but are not binding."]
S2 = ["Nothing in this section changes the rules of any other chapter.", "Contractors follow the equivalent clause in "
      "their contract.", "Teams may agree stricter local practice, recorded in their charter.", "A quarterly summary "
      "of requests is reported to leadership.", "Staff may ask for reasonable adjustments to anything here.",
      "Costs arising from this section are charged to the requesting team.", "Where two managers disagree, the "
      "section owner decides.", "Deadlines here are counted in business days."]


def _person(r) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


# ------------------------------------------------------------------------------------------------ domains
# each domain: f(r) -> dict(title, rules[list of prose paragraphs], case (prose), outcome, outcomes{id: desc},
#                          positive{ids counted as "yes"}, cores_choice[list], cores_yn[list], crit[(true, false)...])


def d_expense(r):
    meal, hotel, rec, pre, late = (r.choice([40, 50, 60, 75]), r.choice([150, 180, 220]), r.choice([25, 50]),
                                   r.choice([600, 800, 1000]), r.choice([30, 45, 60]))
    am_meal = meal + r.choice([10, 15, 20])
    am_date, case_day = r.randint(3, 25), r.randint(1, 28)
    amend_on = case_day >= am_date
    m = am_meal if amend_on else meal
    items = [("meals", r.choice([18, 35, 48, 55, 64, 72, 90])), ("hotel", r.choice([120, 160, 190, 240])),
             ("taxi", r.choice([15, 25, 40]))][: r.randint(2, 3)]
    receipts = [r.random() < 0.8 for _ in items]
    days_after = r.choice([5, 12, late - 2, late + 3])
    preok = r.random() < 0.6
    total = sum(a for _, a in items)
    if days_after > late or (total > pre and not preok):
        out = "reject_all"
    else:
        paid = []
        for (c, a), rc in zip(items, receipts):
            if a >= rec and not rc:
                paid.append(0)
            else:
                paid.append(min(a, {"meals": m, "hotel": hotel, "taxi": 10 ** 6}[c]))
        out = ("approve_full" if paid == [a for _, a in items] else "reject_all" if sum(paid) == 0
               else "approve_partial")
    rules = [f"Meals are reimbursed up to {money(meal)} per day; any excess is the traveller's own cost.",
             f"Hotel nights are reimbursed up to {money(hotel)} per night. Taxis are reimbursed at cost.",
             f"Any single item of {money(rec)} or more must have a receipt; an item missing a required receipt is "
             "not reimbursed, although the rest of the claim is still assessed.",
             f"If the claim total is above {money(pre)}, manager pre-approval is required; without it the entire "
             "claim is refused, whatever else applies.",
             f"A claim filed more than {late} days after the trip ended is refused in full; this overrides every "
             "other provision of the chapter.",
             f"Amendment (effective day {am_date} of the current month): the daily meal limit becomes "
             f"{money(am_meal)}. Claims for trips that ended before that day keep the old limit."]
    desc = ", ".join(f"{c} {money(a)} ({'receipt attached' if rc else 'no receipt'})" for (c, a), rc in zip(items, receipts))
    case = (f"{_person(r)} returned from a work trip that ended on day {case_day} of the current month and filed an "
            f"expense claim {days_after} days later. Items: {desc}. Manager pre-approval was "
            f"{'obtained' if preok else 'not obtained'} before the trip.")
    return dict(title="Travel and Expense Reimbursement", rules=rules, case=case, outcome=out,
                outcomes={"approve_full": "pay the whole claim as filed", "approve_partial": "pay part of the claim",
                          "reject_all": "pay nothing"}, positive={"approve_full"},
                cores_choice=["what happens to this expense claim?", "how is the claim settled?",
                              "what is the outcome for the claim?"],
                cores_yn=["is the claim paid exactly as filed?", "will the traveller be reimbursed every item in full?"],
                crit=[("yes: every item is paid in full", "no: something is reduced or refused"),
                      ("the claim is paid in full", "the claim is cut or refused")])


def d_returns(r):
    win, ext, fee = r.choice([14, 30, 60]), r.choice([15, 30]), r.choice([10, 15, 20])
    cat = r.choice(["headphones", "a winter coat", "a software licence", "a gift card", "a dining chair",
                    "an engraved pen", "a laptop"])
    elec = cat in ("headphones", "a laptop")
    days = r.choice([3, 10, win - 1, win + 5, win + ext - 1, win + ext + 4, 200, 400])
    member, opened, defective, final, proof = (r.random() < 0.4, r.random() < 0.5, r.random() < 0.25,
                                               r.random() < 0.2, r.random() < 0.8)
    blocked = cat in ("a gift card", "an engraved pen") or (cat == "a software licence" and opened)
    if blocked:
        out = "not_returnable"
    elif defective:
        out = "full_refund" if days <= 365 else "not_returnable"
    elif final:
        out = "not_returnable"
    elif days > win + (ext if member else 0):
        out = "not_returnable"
    else:
        out = "refund_minus_fee" if (elec and opened) else "full_refund"
    if out in ("full_refund", "refund_minus_fee") and not proof:
        out = "store_credit"
    rules = [f"Goods may be returned within {win} days of delivery for a full refund. Loyalty members receive "
             f"{ext} additional days.", f"Opened electronics are refunded minus a {fee}% restocking fee unless they are "
             "defective.", "Gift cards, personalised or engraved goods, and opened software can never be returned; no "
             "other clause can change that.", "Defective goods may be returned within one year of delivery with no "
             "fee, regardless of the normal window.", "Goods sold as final sale cannot be returned unless defective.",
             "Without proof of purchase, any refund that would otherwise be due is given as store credit instead."]
    case = (f"A customer{' who is a loyalty member' if member else ''} wants to return {cat} delivered {days} days "
            f"ago. The item is {'opened' if opened else 'unopened'}{' and defective' if defective else ''}"
            f"{', and was bought on final sale' if final else ''}. Proof of purchase is "
            f"{'available' if proof else 'missing'}.")
    return dict(title="Customer Returns", rules=rules, case=case, outcome=out,
                outcomes={"full_refund": "a full refund", "refund_minus_fee": "a refund minus the restocking fee",
                          "store_credit": "store credit only", "not_returnable": "no return is accepted"},
                positive={"full_refund", "refund_minus_fee"},
                cores_choice=["what does the customer receive?", "how should the return be handled?",
                              "what remedy applies?"],
                cores_yn=["does the customer get money back?", "is a cash refund (of any size) due?"],
                crit=[("yes: a refund in money is due", "no: only store credit or nothing"),
                      ("money is refunded", "no money is refunded")])


def d_leave(r):
    accr, carry, notice = r.choice([1.5, 2.0, 2.5]), r.choice([5, 8]), r.choice([7, 10, 14])
    months, carried, taken = r.randint(1, 12), r.choice([0, 3, carry + 3]), r.choice([0, 2, 5, 9])
    req, given = r.choice([1, 2, 4, 6, 10]), r.choice([2, notice - 1, notice, 20])
    blackout, emergency, probation = r.random() < 0.2, r.random() < 0.15, r.random() < 0.2
    if emergency:  # emergency leave is defined as at most two days: no ambiguous long "emergency" requests
        req = r.choice([1, 2])
    bal =accr * months + min(carried, carry) - taken
    if emergency and req <= 2:
        out = "approve"
    elif blackout:
        out = "deny"
    elif probation and taken + req > 3:
        out = "deny"
    elif req > 3 and given < notice:
        out = "deny"
    else:
        out = "approve" if req <= bal else "deny"
    rules = [f"Staff earn {accr} days of paid leave for each full month worked this year, and may bring up to {carry} "
             "unused days from last year; anything above that lapsed on 1 January.", f"Requests longer than three "
             f"consecutive days need at least {notice} days' notice.", "During a declared blackout period, leave is "
             "refused unless it is emergency leave.", "Staff still on probation may take no more than three days of "
             "paid leave in total this year.", "A request is granted only if the available balance covers it.",
             "Emergency leave of up to two days is always granted, without notice, and is not limited by any other "
             "clause of this chapter."]
    case = (f"{_person(r)} has worked {months} full months this year, carried {carried} unused days from last year and "
            f"has already taken {taken} days. They ask for {req} day{'s' if req > 1 else ''} with {given} days' notice"
            f"{' as emergency leave' if emergency else ''}. {'A blackout period is declared for those dates. ' if blackout else ''}"
            f"{'They are still on probation.' if probation else 'They have completed probation.'}")
    return dict(title="Paid Leave", rules=rules, case=case, outcome=out,
                outcomes={"approve": "grant the leave", "deny": "refuse the leave"}, positive={"approve"},
                cores_choice=[], cores_yn=["should the leave request be granted?", "is the leave approved?"],
                crit=[("yes: the leave is granted", "no: the leave is refused"),
                      ("the request is approved", "the request is turned down")])


def d_procurement(r):
    a, b = r.choice([500, 1000, 2500]), r.choice([10000, 25000])
    am_a, am_day, case_day = a * 2, r.randint(5, 25), r.randint(1, 28)
    lim = am_a if case_day >= am_day else a
    amt = r.choice([300, 800, 1800, 3500, 9000, 18000, 40000])
    quotes, listed, safety, director = r.choice([0, 1, 2, 3]), r.random() < 0.75, r.random() < 0.15, r.random() < 0.4
    if not listed and not (safety and amt <= b):
        out = "reject"
    elif safety and amt <= b:
        out = "approve"
    elif amt <= lim:
        out = "approve"
    elif amt <= b:
        out = "approve" if quotes >= 2 else "more_quotes"
    else:
        out = "approve" if (quotes >= 3 and director) else "escalate"
    rules = [f"Purchases up to {money(a)} may be made on a company card from an approved supplier.",
             f"Purchases above that and up to {money(b)} need at least two written quotes.",
             f"Purchases above {money(b)} need three quotes and director sign-off; without both they go to the "
             "procurement board.", "Suppliers must be on the approved list; an unlisted supplier is refused.",
             f"Safety-critical purchases up to {money(b)} may proceed immediately, from any supplier and without "
             "quotes. This exception overrides the supplier-list and quote rules.",
             f"Amendment (effective day {am_day} of this month): the card limit in the first rule becomes "
             f"{money(am_a)}. Earlier requests keep the old limit."]
    case = (f"On day {case_day} of this month, {_person(r)} requested a {money(amt)} purchase from a supplier that is "
            f"{'on' if listed else 'not on'} the approved list, with {quotes} written quote{'s' if quotes != 1 else ''}. "
            f"The item is {'safety-critical' if safety else 'not safety-critical'}; director sign-off is "
            f"{'on file' if director else 'not on file'}.")
    return dict(title="Procurement", rules=rules, case=case, outcome=out,
                outcomes={"approve": "the purchase may go ahead", "more_quotes": "more quotes are needed first",
                          "escalate": "it goes to the procurement board", "reject": "the purchase is refused"},
                positive={"approve"}, cores_choice=["what happens to this purchase request?",
                                                    "how should the purchase request be handled?"],
                cores_yn=["may the purchase go ahead now?", "can the buyer proceed with this purchase?"],
                crit=[("yes: it can proceed as requested", "no: it is blocked, escalated or needs more quotes"),
                      ("the purchase is cleared", "the purchase is not cleared")])


def d_equipment(r):
    allow = r.choice([300, 400, 500, 750])
    used, cost = r.choice([0, 100, 250, 400]), r.choice([80, 150, 260, 420, 600])
    item = r.choice(["an office chair", "a monitor", "a standing desk", "a laptop", "a headset", "a printer"])
    part, prob = r.random() < 0.3, r.random() < 0.2
    eligible_items = {"an office chair", "a monitor", "a standing desk", "a headset"}
    cap = allow / 2 if part else allow
    left = max(0, cap - used)
    if prob or item not in eligible_items:
        out = "none"
    elif cost <= left:
        out = "full"
    elif left > 0:
        out = "partial"
    else:
        out = "none"
    rules = [f"Remote workers have a home-equipment allowance of {money(allow)} per calendar year.",
             "Part-time staff receive half of the allowance.", "The allowance covers chairs, monitors, desks and "
             "headsets. Laptops and printers are supplied by IT and are never paid from the allowance.",
             "Staff on probation cannot use the allowance.", "A purchase above the remaining allowance is paid up to "
             "what remains; the rest is the employee's cost."]
    case = (f"{_person(r)}, a {'part-time' if part else 'full-time'} remote worker{' on probation' if prob else ''}, "
            f"has used {money(used)} of this year's allowance and now claims {money(cost)} for {item}.")
    return dict(title="Home Equipment Allowance", rules=rules, case=case, outcome=out,
                outcomes={"full": "the whole cost is paid", "partial": "only part of the cost is paid",
                          "none": "nothing is paid"}, positive={"full"},
                cores_choice=["how much of the claim does the allowance pay?", "what is paid on this claim?"],
                cores_yn=["is the full cost covered?", "does the allowance pay the whole claim?"],
                crit=[("yes: the whole cost is reimbursed", "no: part or none of it is reimbursed"),
                      ("fully covered", "not fully covered")])


def d_access(r):
    cls = r.choice(["internal", "confidential", "restricted"])
    role = r.choice(["employee", "contractor"])
    trained, mgr, dpo, incident = r.random() < 0.75, r.random() < 0.7, r.random() < 0.4, r.random() < 0.12
    if incident and role == "employee":
        out = "grant"
    elif cls == "restricted" and role == "contractor":
        out = "deny"
    elif not trained:
        out = "deny"
    elif cls == "internal":
        out = "grant"
    elif cls == "confidential":
        out = "grant" if mgr else "masked"
    else:
        out = "grant" if (mgr and dpo) else "masked" if mgr else "deny"
    rules = ["Internal data is available to any trained member of staff.", "Confidential data needs manager approval; "
             "without it, only a masked extract may be given.", "Restricted data needs both manager and data-protection "
             "officer approval; with manager approval alone, a masked extract may be given.", "Anyone who has not "
             "completed data-handling training is refused.", "Contractors are never given restricted data.",
             "During a declared security incident, an employee on the response team gets the access they request, "
             "whatever the other rules say. This does not extend to contractors."]
    case = (f"{'An' if role == 'employee' else 'A'} {role} ({_person(r)}) requests {cls} data. Training is {'complete' if trained else 'not complete'}; "
            f"manager approval {'given' if mgr else 'not given'}; data-protection approval {'given' if dpo else 'not given'}."
            f"{' A security incident is declared and the requester is on the response team.' if incident else ''}")
    return dict(title="Data Access", rules=rules, case=case, outcome=out,
                outcomes={"grant": "full access", "masked": "a masked extract only", "deny": "no access"},
                positive={"grant"}, cores_choice=["what access should be given?", "how should the access request be "
                                                  "answered?"], cores_yn=["should full access be granted?",
                                                                          "does the requester get full access?"],
                crit=[("yes: full access is granted", "no: masked or no access"), ("full access", "less than full access")])


def d_subscription(r):
    plan = r.choice(["monthly", "annual"])
    days = r.choice([3, 10, 13, 15, 40, 200])
    promo, outage = r.random() < 0.25, r.random() < 0.15
    cool = r.choice([14, 30])
    if days <= cool:
        out = "full_refund"
    elif outage:
        out = "service_credit"
    elif plan == "annual" and not promo:
        out = "prorated_refund"
    else:
        out = "no_refund"
    rules = [f"Any plan cancelled within {cool} days of its latest renewal is refunded in full.", "After that, annual "
             "plans are refunded pro rata for the unused months, unless they were bought at a promotional price.",
             "Monthly plans are not refunded after the cooling-off period.", "A customer who cancels after the "
             "cooling-off period because of a documented outage of more than 24 hours receives a service credit "
             "instead of any refund."]
    case = (f"A customer cancels a {plan} plan {days} days after its latest renewal"
            f"{', bought at a promotional price' if promo else ''}."
            f"{' They cite a documented 30-hour outage last week.' if outage else ''}")
    return dict(title="Subscriptions and Cancellations", rules=rules, case=case, outcome=out,
                outcomes={"full_refund": "a full refund", "prorated_refund": "a pro-rata refund",
                          "service_credit": "a service credit", "no_refund": "nothing"},
                positive={"full_refund", "prorated_refund"},
                cores_choice=["what does the customer receive on cancellation?", "what is the cancellation outcome?"],
                cores_yn=["is any money refunded?", "does the customer get a cash refund?"],
                crit=[("yes: some money is refunded", "no: a credit or nothing"), ("refund due", "no refund due")])


def d_travel(r):
    hours = r.choice([2, 4, 7, 9, 11, 14])
    director, overnight, client = r.random() < 0.3, r.random() < 0.3, r.random() < 0.2
    client_cls = r.choice(["economy", "premium_economy"])
    if client:
        out = client_cls
    elif hours > 10 and (director or overnight):
        out = "business"
    elif hours > 6:
        out = "premium_economy"
    else:
        out = "economy"
    rules = ["Flights are booked in economy by default.", "Flights over six hours may be booked in premium economy.",
             "Flights over ten hours may be booked in business class by directors, or by anyone on an overnight "
             "flight.", "Trips billed to a client follow the client's travel terms, which replace this chapter's class "
             "rules entirely."]
    case = (f"{_person(r)}{', a director,' if director else ''} is flying {hours} hours"
            f"{' overnight' if overnight else ''}." + (f" The trip is billed to a client whose terms allow "
                                                        f"{client_cls.replace('_', ' ')} at most." if client else
                                                        " The trip is not billed to a client."))
    return dict(title="Travel Booking", rules=rules, case=case, outcome=out,
                outcomes={"economy": "economy", "premium_economy": "premium economy", "business": "business class"},
                positive={"business", "premium_economy"},
                cores_choice=["which cabin class may be booked?", "what is the highest class allowed for this flight?"],
                cores_yn=["may anything above economy be booked?", "is an upgrade over economy allowed?"],
                crit=[("yes: a class above economy is allowed", "no: economy only"),
                      ("above economy allowed", "economy only")])


def d_overtime(r):
    band = r.choice(["hourly", "salaried below band 5", "salaried band 5 or above"])
    hours = r.choice([38, 42, 46, 52])
    pre, holiday = r.random() < 0.7, r.random() < 0.2
    if band == "salaried band 5 or above":
        out = "toil"
    elif hours <= 40 and not holiday:
        out = "none"
    elif not pre:
        out = "none"
    elif holiday:
        out = "double"
    else:
        out = "time_half"
    rules = ["Hourly and junior salaried staff are paid overtime for hours above 40 in a week at time and a half.",
             "Approved work on a public holiday is paid at double time, whatever the weekly total.", "Overtime and "
             "holiday pay are paid only if the extra work was approved in advance.", "Staff at band 5 or above receive time off in lieu instead of overtime pay; "
             "no other clause applies to them."]
    case = (f"{_person(r)} ({band}) worked {hours} hours this week"
            f"{', including a public holiday' if holiday else ''}. The extra hours were "
            f"{'approved in advance' if pre else 'not approved in advance'}.")
    return dict(title="Overtime", rules=rules, case=case, outcome=out,
                outcomes={"none": "no overtime pay", "time_half": "time and a half", "double": "double time",
                          "toil": "time off in lieu"}, positive={"time_half", "double"},
                cores_choice=["how is the extra work compensated?", "what compensation applies?"],
                cores_yn=["is overtime pay due?", "will the employee receive overtime pay?"],
                crit=[("yes: overtime pay is due", "no: no overtime pay (time off or nothing)"),
                      ("paid overtime", "no paid overtime")])


def d_warranty(r):
    term = r.choice([12, 24])
    months, plan = r.choice([3, 11, 13, 20, 25, 30]), r.random() < 0.35
    dmg = r.choice(["fault", "fault", "accident"])
    tamper, fails = r.random() < 0.15, r.choice([0, 0, 1, 2])
    cover = term + (12 if plan else 0)
    if tamper:
        out = "paid_repair"
    elif months > cover:
        out = "paid_repair"
    elif dmg == "accident" and not plan:
        out = "paid_repair"
    elif fails >= 2:
        out = "replacement"
    else:
        out = "free_repair"
    rules = [f"Products are covered for {term} months against manufacturing faults.", "The care plan adds 12 months and "
             "also covers accidental damage.", "A product that has already failed repair twice is replaced instead of "
             "repaired.", "Signs of tampering void all cover; such repairs are charged."]
    case = (f"A customer brings in a product bought {months} months ago with "
            f"{'accidental damage' if dmg == 'accident' else 'a manufacturing fault'}. "
            f"{'They hold the care plan. ' if plan else 'They have no care plan. '}"
            f"{'The technician found signs of tampering. ' if tamper else ''}It has failed repair {fails} time"
            f"{'s' if fails != 1 else ''} before.")
    return dict(title="Warranty Service", rules=rules, case=case, outcome=out,
                outcomes={"free_repair": "a free repair", "replacement": "a free replacement",
                          "paid_repair": "a charged repair"}, positive={"free_repair", "replacement"},
                cores_choice=["what service does the customer get?", "how is the claim resolved?"],
                cores_yn=["is the service free of charge?", "will the customer pay nothing?"],
                crit=[("yes: no charge", "no: the customer pays"), ("free", "charged")])


def d_latefee(r):
    first, hard, amt = r.random() < 0.4, r.random() < 0.25, r.choice([15, 40, 90, 150])
    sup, thr = r.random() < 0.5, r.choice([50, 100])
    if first and amt <= thr:
        out = "waive"
    elif hard:
        out = "waive" if (amt <= thr or sup) else "half"
    elif amt > thr:
        out = "charge"
    else:
        out = "half" if sup else "charge"
    rules = [f"A customer's first late fee in twelve months is waived automatically if it is {money(thr)} or less.",
             f"Documented hardship waives the fee; above {money(thr)} that needs supervisor approval, otherwise half is "
             "waived.", f"Other fees up to {money(thr)} may be halved with supervisor approval.", "All other late fees "
             "are charged in full."]
    case = (f"A {money(amt)} late fee is due. It is {'the first' if first else 'not the first'} in twelve months. "
            f"Hardship is {'documented' if hard else 'not documented'}; supervisor approval is "
            f"{'given' if sup else 'not given'}.")
    return dict(title="Late Fees", rules=rules, case=case, outcome=out,
                outcomes={"waive": "the fee is waived", "half": "half the fee is charged", "charge": "the full fee is "
                          "charged"}, positive={"waive"}, cores_choice=["what happens to the late fee?"],
                cores_yn=["is the late fee waived entirely?", "does the customer escape the fee completely?"],
                crit=[("yes: fully waived", "no: some or all of it is charged"), ("waived", "not waived")])


def d_building(r):
    who = r.choice(["employee", "visitor", "contractor"])
    area = r.choice(["office floors", "the laboratory", "the server room"])
    after, sponsor, safety, escort = r.random() < 0.3, r.random() < 0.6, r.random() < 0.6, r.random() < 0.5
    if area == "the server room" and who != "employee":
        out = "deny"
    elif area == "the laboratory" and not safety:
        out = "deny"
    elif after and not sponsor:
        out = "deny"
    elif who == "visitor":
        out = "escorted" if escort else "deny"
    else:
        out = "grant"
    rules = ["Employees may enter office floors at any time during opening hours.", "Visitors must be escorted at all "
             "times; an unescorted visitor is refused entry.", "The laboratory requires completed safety training, for "
             "everyone.", "Only employees may enter the server room.", "After-hours entry requires a named sponsor's "
             "approval, for everyone."]
    case = (f"A {who} wants to enter {area}{' after hours' if after else ' during opening hours'}. Safety training: "
            f"{'done' if safety else 'not done'}. Sponsor approval: {'yes' if sponsor else 'no'}. Escort available: "
            f"{'yes' if escort else 'no'}.")
    return dict(title="Building Access", rules=rules, case=case, outcome=out,
                outcomes={"grant": "entry on their own", "escorted": "entry with an escort", "deny": "no entry"},
                positive={"grant", "escorted"}, cores_choice=["what entry is allowed?"],
                cores_yn=["is the person allowed in?", "may they enter?"],
                crit=[("yes: entry is allowed", "no: entry is refused"), ("allowed in", "refused")])


DOMAINS: dict[str, Callable] = {"expense": d_expense, "returns": d_returns, "leave": d_leave, "procurement": d_procurement,
                                "equipment": d_equipment, "access": d_access, "subscription": d_subscription,
                                "travel": d_travel, "overtime": d_overtime, "warranty": d_warranty, "latefee": d_latefee,
                                "building": d_building}


#: ordinal readings of three domains (score questions; levels low -> high, the gold is the outcome's level)
ORDINAL = {"expense": {"reject_all": "none of the claimed amount is paid", "approve_partial": "part of the claimed "
                       "amount is paid", "approve_full": "the whole claimed amount is paid"},
           "equipment": {"none": "none of the cost is paid", "partial": "part of the cost is paid",
                         "full": "the whole cost is paid"},
           "latefee": {"charge": "none of the fee is waived", "half": "half of the fee is waived",
                       "waive": "all of the fee is waived"}}
SCORE_CORE = {"expense": ["how much of the claim is reimbursed?", "rate how much of the claimed amount is paid."],
              "equipment": ["how much of the purchase does the allowance cover?", "rate the share of the cost paid."],
              "latefee": ["how much of the late fee is waived?", "rate how much of the fee the customer is spared."]}
#: options no case can earn (every generated case falls squarely under its chapter); used only to vary K
EXTRA = {"refer_owner": "refer the case to the policy owner as not covered by any rule",
         "not_applicable": "the chapter does not apply to this case",
         "defer": "postpone the decision until next quarter's policy review"}


def instruction_pool() -> set[str]:
    out = set()
    for f in DOMAINS.values():
        d = f(rng("pool"))
        for c in d["cores_choice"] + d["cores_yn"]:
            for p in PREFIX:
                for s in SUFFIX:
                    out.add(p + (c[0].upper() + c[1:] if not p or p.endswith(". ") else c) + s)
    return out


def _chapter(d: dict, num: int) -> str:
    paras = [f"{num}.{j + 1} {t}" for j, t in enumerate(d["rules"])]
    return f"Chapter {num}: {d['title']}\n" + "\n".join(paras)


def _admin(r, used: set) -> str:
    n = f"{r.choice('ABDGKMPS')}{r.randint(1, 99)}"
    while n in used:
        n = f"{r.choice('ABDGKMPS')}{r.randint(1, 99)}"
    used.add(n)
    body = " ".join(r.sample(S1, r.randint(2, 4)) + r.sample(S2, r.randint(1, 3)))
    if r.random() < 0.5:
        body += " " + r.choice(LP_FILLER)
    return f"Section {n}. {r.choice(ADMIN_T)}. {body}"


def _handbook(i: int) -> dict | None:
    names = sorted(DOMAINS)
    dom = names[i % len(names)]
    r = rng("lpl", i)
    probe = DOMAINS[dom](rng("lpl-probe", i))
    can_choice = len(probe["outcomes"]) >= 3 and probe["cores_choice"]
    u = r.random()
    qtype = ("score" if dom in ORDINAL and u < 0.2 else "choice" if can_choice and u < 0.6 else "noul")
    want = r.choice(sorted(probe["outcomes"])) if qtype in ("choice", "score") else (r.random() < 0.5)
    d = None
    for t in range(400):  # draw the gold first, then resample the case until it matches (label balance)
        cand = DOMAINS[dom](rng("lpl-case", i, t))
        ok = (cand["outcome"] == want if qtype in ("choice", "score")
              else ((cand["outcome"] in cand["positive"]) == want))
        if ok:
            d = cand
            break
    if d is None:
        return None
    others = [x for x in names if x != dom]
    extra = [DOMAINS[x](rng("lpl-other", i, x)) for x in r.sample(others, r.randint(1, 2))]
    chs = [d] + extra
    r.shuffle(chs)
    org = f"{_pname(r, set())} {r.choice(['Group', 'Ltd', 'Cooperative', 'Holdings', 'Foundation', 'Partners'])}"
    blocks = [_chapter(c, j + 1) for j, c in enumerate(chs)]
    target_chars = r.randint(8500, 15500)
    used: set = set()
    slots: list[list[str]] = [[] for _ in range(len(blocks) + 1)]
    total = sum(len(b) for b in blocks) + len(d["case"])
    while total < target_chars:
        s = _admin(r, used)
        slots[r.randrange(len(slots))].append(s)
        total += len(s) + 2
    parts = []
    for j, b in enumerate(blocks):
        parts += slots[j] + [b]
    parts += slots[-1]
    doc = (f"{org} staff and customer policy handbook, edition {r.randint(3, 12)}.{r.randint(0, 9)}. Chapters apply "
           f"only to the matters they name.\n\n" + "\n\n".join(parts) + f"\n\nCase record ({r.choice(CITIES)} office): "
           + d["case"])
    pre, suf = r.choice(PREFIX), r.choice(SUFFIX)
    if qtype == "score":
        core = r.choice(SCORE_CORE[dom])
        instr = pre + (core[0].upper() + core[1:] if not pre or pre.endswith(". ") else core) + suf
        q, g = score_q(instr, list(ORDINAL[dom].values())), list(ORDINAL[dom]).index(d["outcome"])
    elif qtype == "choice":
        core = r.choice(d["cores_choice"])
        instr = pre + (core[0].upper() + core[1:] if not pre or pre.endswith(". ") else core) + suf
        opts = dict(d["outcomes"])
        k = r.choice([3, 4, 5, 6])
        for x in r.sample(sorted(EXTRA), len(EXTRA)):  # always-wrong options (every case is covered) to reach K
            if len(opts) >= k:
                break
            opts[x] = EXTRA[x]
        q, g = choice_q(instr, opts, d["outcome"], r)
    else:
        core = r.choice(d["cores_yn"])
        instr = pre + (core[0].upper() + core[1:] if not pre or pre.endswith(". ") else core) + suf
        tdesc, fdesc = r.choice(d["crit"])
        q, g = noul_q(instr, tdesc, fdesc), d["outcome"] in d["positive"]
    return record("long_policy", f"lpl_{dom}", "jev2synth", f"m2:jev2synth:long_handbook:{i}", "lpl", doc, q, g, {},
                  split="synthetic", gen="jev2_long_handbook", domain=dom, target_chapter=chs.index(d) + 1,
                  qtype=qtype)


CNLI = {"entailment": "the agreement entails the statement", "contradiction": "the agreement contradicts the "
        "statement", "neutral": "the agreement does not address it (neither entailed nor contradicted)"}
CNLI_PRE = ["Read the agreement. Statement: ", "Check the contract against this statement: ", "Does the agreement "
            "support, contradict or not address this? ", "Compare the statement with the full contract text: ",
            "Using only the agreement above, assess: "]


def contractnli_rows(n: int) -> Iterator[dict]:
    z = zipfile.ZipFile(RAW / "contractnli" / "contract_nli_long.zip")
    quota = {k: n // 3 + 1 for k in CNLI}
    with z.open("contract_nli_long.jsonl") as fh:
        for di, line in enumerate(fh):
            if sum(quota.values()) <= 0:
                return
            d = json.loads(line)
            if d.get("subset") != "train":  # TRAIN only (the Index uses ContractNLI test)
                continue
            doc = d["premise"]
            if not 8000 <= len(doc) <= 15500:
                continue
            r = rng("cnli", di)
            hl = list(enumerate(d["hypothesises/labels"]))
            r.shuffle(hl)
            took = 0
            for j, x in hl:
                lab = x["label"]
                lab = {0: "contradiction", 1: "entailment", 2: "neutral"}.get(lab, lab) if isinstance(lab, int) else lab
                if lab not in CNLI or quota[lab] <= 0 or took >= 3:
                    continue
                quota[lab] -= 1
                took += 1
                q, g = choice_q(r.choice(CNLI_PRE) + x["hypothesis"], dict(CNLI), lab, r)
                yield record("long_policy", "lpl_contractnli", "contractnli", f"m2:contractnli:{di}", f"cnli{j}", doc,
                             q, g, {}, hypothesis_index=j)


PLAN_C = {"long_policy": {"handbook": 1800, "contractnli": 300}}


def generate_c(families) -> Iterator[dict]:
    if "long_policy" in families:
        for i in range(PLAN_C["long_policy"]["handbook"]):
            row = _handbook(i)
            if row is not None:
                yield row
        yield from contractnli_rows(PLAN_C["long_policy"]["contractnli"])
