"""Latency / cost meter: serial wall-clock per request, token counts per decision, $ per 1,000 decisions.

Latency (JevBench protocol): one request at a time, one decision per request, timed on the open standard + judge
items; p50 / p95 by linear interpolation; Speed from the adjusted values (self-hosted s' = 2 s + 0.15 s, APIs raw).
Our in-process wall time excludes the evaluator's network hop to a hosted endpoint; the x2 + 0.15 s adjustment is
the board's own stand-in for that and for load, and it is an assumption, not a measurement (as the board says).

Cost: USD per 1,000 decisions = 1000 x mean over decisions of (input_tokens x in_price + output_tokens x out_price),
on the common item set (``exclude_ids``: the board drops the 23 items outside the Jev reference input range). The
price is an explicit, recorded input - never a default:

* ``list_tariff``           a public, bookable list price in effect >= 30 days at ``as_of`` (``listed_since``)
* ``base_model_reference``  the base model's public market price (OpenRouter or equivalent) - the board's floor
* ``size_class_estimate``   no tariff and no base-model listing: a stated size-class estimate (labelled estimate)

``scoring_price`` applies the board's floor: max(cost at the list tariff, cost at the base-model reference), compared
per decision (so shifting price between input and output tokens cannot dodge it).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import scoring as SC
from . import spec
from .items import Item

PRICE_KINDS = ("list_tariff", "base_model_reference", "size_class_estimate")


class PriceError(ValueError):
    pass


@dataclass(frozen=True)
class PriceSource:
    kind: str
    usd_per_m_input: float
    usd_per_m_output: float
    source: str  # URL / page / table the numbers come from - required
    as_of: str  # ISO date the price was read
    listed_since: str | None = None  # list_tariff: date the price took effect (30-day rule)
    model: str | None = None  # the model (or base model) the price is for

    def validate(self) -> PriceSource:
        if self.kind not in PRICE_KINDS:
            raise PriceError(f"price kind must be one of {PRICE_KINDS}")
        if not self.source or not str(self.source).strip():
            raise PriceError("price source must be stated")
        if self.usd_per_m_input < 0 or self.usd_per_m_output < 0:
            raise PriceError("prices must be >= 0")
        as_of = dt.date.fromisoformat(self.as_of)
        if self.kind == "list_tariff":
            if not self.listed_since:
                raise PriceError("list_tariff needs listed_since (the 30-day rule)")
            age = (as_of - dt.date.fromisoformat(self.listed_since)).days
            if age < 30:
                raise PriceError(f"list tariff in effect {age} days at {self.as_of} (< 30): use the previous price "
                                 "or the base-model reference")
        return self

    def usd_per_decision(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.usd_per_m_input + output_tokens * self.usd_per_m_output) / 1e6

    def to_dict(self) -> dict:
        return asdict(self)


def parse_price(spec_str: str) -> PriceSource:
    """``kind=list_tariff,in=0.05,out=10,source=https://...,as_of=2026-10-05,listed_since=2026-08-01`` or a JSON
    file path (one object, or {"price": {...}, "reference": {...}})."""
    p = Path(spec_str)
    if p.suffix == ".json" and p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        d = d.get("price", d)
        return PriceSource(kind=d["kind"], usd_per_m_input=float(d["usd_per_m_input"]),
                           usd_per_m_output=float(d["usd_per_m_output"]), source=d["source"], as_of=d["as_of"],
                           listed_since=d.get("listed_since"), model=d.get("model")).validate()
    kv = {}
    for part in spec_str.split(","):
        if "=" not in part:
            raise PriceError(f"bad price field {part!r}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    try:
        return PriceSource(kind=kv["kind"], usd_per_m_input=float(kv["in"]), usd_per_m_output=float(kv["out"]),
                           source=kv["source"], as_of=kv["as_of"], listed_since=kv.get("listed_since"),
                           model=kv.get("model")).validate()
    except KeyError as e:
        raise PriceError(f"price spec missing {e}") from e


# ------------------------------------------------------------------------------------------- measurement
def measure(answer_one: Callable[[Item], dict], items: Iterable[Item], clock: Callable[[], float] = time.perf_counter,
            warmup: int = 2) -> list[dict]:
    """Serial, one request per decision. ``answer_one(item)`` -> {"usage": {...}, "status": ...}. Warm-up requests
    (the first ``warmup`` items, re-sent) are not recorded, as the kit's Engine.warmup."""
    items = list(items)
    for it in items[:warmup]:
        answer_one(it)
    out = []
    for it in items:
        t0 = clock()
        r = answer_one(it)
        el = clock() - t0
        u = r.get("usage") or {}
        out.append({"id": it.id, "latency_s": el, "status": r.get("status", "ok"),
                    "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")})
    return out


def timed_subset(items: Sequence[Item], tiers: Sequence[str] = spec.TIMED_TIERS) -> list[Item]:
    """The board times the open standard + judge items (~124 in v1.6.1)."""
    return [it for it in items if it.side == "open" and it.tier in tiers]


def latency_summary(latencies: Sequence[float], self_hosted: bool = True) -> dict:
    ok = [float(x) for x in latencies if x is not None]
    p50, p95 = SC.percentile(ok, 0.5), SC.percentile(ok, 0.95)
    out = SC.speed(p50, p95, self_hosted=self_hosted)
    out["n"] = len(ok)
    return out


def scoring_price(records: Sequence[dict], price: PriceSource, reference: PriceSource | None = None) -> tuple[PriceSource, str]:
    """Board floor: the higher per-decision cost of the list tariff and the base-model reference."""
    if reference is None or price.kind != "list_tariff":
        return price, "as declared"
    a, b = (cost_per_1000(records, p) for p in (price, reference))
    if b is not None and a is not None and b > a:
        return reference, "price floor: base-model reference price applied"
    return price, "list tariff (>= base-model reference)"


def cost_per_1000(records: Sequence[dict], price: PriceSource, exclude_ids: Iterable[str] = ()) -> float | None:
    ex = set(exclude_ids)
    rows = [r for r in records if r.get("id") not in ex and r.get("input_tokens") is not None]
    if not rows:
        return None
    per = [price.usd_per_decision(r["input_tokens"], r.get("output_tokens") or 0) for r in rows]
    return 1000.0 * sum(per) / len(per)


def cost_summary(records: Sequence[dict], price: PriceSource, reference: PriceSource | None = None,
                 exclude_ids: Iterable[str] = ()) -> dict:
    ex = set(exclude_ids)
    chosen, rule = scoring_price(records, price, reference)
    usd = cost_per_1000(records, chosen, ex)
    rows = [r for r in records if r.get("id") not in ex and r.get("input_tokens") is not None]
    missing = sum(1 for r in records if r.get("id") not in ex and r.get("input_tokens") is None)
    return {"usd_per_1000": usd, "score": SC.cost_score(usd), "price": chosen.to_dict(), "price_rule": rule,
            "declared_price": price.to_dict(), "reference_price": reference.to_dict() if reference else None,
            "kind": "estimate" if chosen.kind != "list_tariff" else "list",
            "n_decisions": len(rows), "n_missing_usage": missing, "n_excluded": len(ex),
            "mean_input_tokens": sum(r["input_tokens"] for r in rows) / len(rows) if rows else None,
            "mean_output_tokens": sum((r.get("output_tokens") or 0) for r in rows) / len(rows) if rows else None}


def records_from_preds(preds: Iterable[dict]) -> list[dict]:
    """jevkit prediction records -> meter records (usage + latency)."""
    out = []
    for p in preds:
        u = p.get("usage") or {}
        out.append({"id": p.get("id"), "latency_s": p.get("latency_s"), "status": p.get("status", "ok"),
                    "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")})
    return out
