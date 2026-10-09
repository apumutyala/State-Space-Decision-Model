"""/v1/systemone adapter: one path for JevBench items, Decision Index rows and our decision backends.

Wire format (TypeSafe ``POST /v1/systemone``)::

    request  {"state": str|object|array, "model": str,
              "questions": {qid: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}}
    response {"model": str, "answers": {qid: answer}, "usage": {"input_tokens": n, "output_tokens": n}}
    noul   -> {"type": "noul", "noul": P(yes)}
    choice -> {"type": "choice", "choice": key, "probabilities": {key: p} (sum 1), "confidence": c}
    score  -> {"type": "score", "score": E[level], "probabilities": {"0".."K-1": p}, "legend": {...}, "confidence": c}

Targets ``SystemOneAdapter`` accepts (``kind="auto"`` detects them):

* ``decider``  anything with ``.decide(request) -> response`` carrying the ``ssdm`` envelope: ``decide.Decider`` over
               any backend (the RWKV7 / Qwen readouts of ``jobs.common.readout_engine``, ``baselines.clef_backend``).
               ``ssdm.status != ok`` becomes a non-ok record (wrong for Intelligence), never an answer.
* ``engine``   a Decision Index kit engine: ``engine(state, questions) -> (response, raw)`` - ``ReadoutIndexEngine``,
               ``ClefIndexEngine``, the kit's ``RandomEngine``; the kit's ``Unsupported`` becomes ``unsupported``.
* ``request``  a plain ``fn(request) -> response`` (an HTTP client for a hosted /v1/systemone endpoint).

The question id sent to a backend is a neutral constant (``"q"``) by default: JevBench item ids carry the family name
("hard-opus-a-long_policy-01") and our Clef / readout prompts include question ids, so sending them would leak.

Decision Index <-> JevBench: ``di_row_to_items`` splits a DI row (``questions`` map + ``expected`` map, noul gold as a
bool) into JevBench items; ``item_to_di_row`` goes back (``_evaluation.run_id`` = item id), so a JevBench item set can
be run by ``python -m decision_index run`` and its ``results.jsonl`` read back by ``di_result_to_pred``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import Any

from .answers import argmax_label
from .items import Item

NEUTRAL_QID = "q"


class AdapterError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- request / answer
def item_request(item: Item, model: str = "jevkit", qid: str = NEUTRAL_QID) -> dict:
    return {"model": model, "state": item.state, "questions": {qid: item.question}}


def normalize_answer(question: dict, answer: Any) -> Any:
    """Fill the derivable SystemOne fields a backend may omit (``type``, choice ``choice``/``confidence``, score
    ``score``/``legend``/``confidence``); never touches the probabilities (validity is judged on them as sent)."""
    if not isinstance(answer, dict):
        return answer
    a = dict(answer)
    typ = question["type"]
    a.setdefault("type", typ)
    probs = a.get("probabilities")
    numeric = isinstance(probs, dict) and probs and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in probs.values())
    if typ == "choice" and numeric:
        a.setdefault("choice", argmax_label({str(k): float(v) for k, v in probs.items()}))
        k = len(probs)
        a.setdefault("confidence", (k * max(probs.values()) - 1) / (k - 1) if k > 1 else 1.0)
    elif typ == "score" and numeric:
        try:
            a.setdefault("score", sum(int(k) * float(v) for k, v in probs.items()))
        except ValueError:
            pass
        crit = question.get("criteria") or []
        a.setdefault("legend", {str(i): (c if isinstance(c, str) else str(c)) for i, c in enumerate(crit)})
        a.setdefault("confidence", max(probs.values()))
    return a


def _detect(target: Any) -> str:
    if hasattr(target, "decide"):
        return "decider"
    if callable(target):
        return "engine" if getattr(target, "name", None) is not None and hasattr(target, "warmup") else "request"
    raise TypeError("target must be a Decider-like object or a callable")


class SystemOneAdapter:
    """Uniform ``request -> SystemOne response`` over a decider, a kit engine or a request callable."""

    def __init__(self, target: Any, kind: str = "auto", model_name: str | None = None,
                 unsupported_exc: tuple[type[BaseException], ...] = ()):
        self.target = target
        self.kind = _detect(target) if kind == "auto" else kind
        if self.kind not in ("decider", "engine", "request"):
            raise ValueError(f"unknown kind {kind!r}")
        self.model_name = model_name or getattr(target, "name", None) or type(target).__name__
        try:  # the kit's Unsupported when the kit is installed (job images); CPU tests pass their own
            from decision_index.engines.base import Unsupported as KitUnsupported  # type: ignore
            self.unsupported_exc = tuple(unsupported_exc) + (KitUnsupported,)
        except Exception:  # noqa: BLE001
            self.unsupported_exc = tuple(unsupported_exc)

    def __call__(self, request: dict) -> dict:
        """-> {"model", "answers", "usage", "status", "error"}; status ok / unsupported / exhausted / error."""
        try:
            if self.kind == "decider":
                resp = self.target.decide(request)
                ss = resp.get("ssdm") or {}
                status = ss.get("status", "ok")
                if status != "ok":
                    err = ss.get("error") or resp.get("error") or {}
                    return {"model": resp.get("model"), "answers": {}, "usage": resp.get("usage") or {},
                            "status": status, "error": f"{err.get('code')}: {err.get('message')}"}
            elif self.kind == "engine":
                resp, _raw = self.target(request["state"], request["questions"])
            else:
                resp = self.target(request)
        except self.unsupported_exc as e:  # type: ignore[misc]
            return {"model": self.model_name, "answers": {}, "usage": {}, "status": "unsupported", "error": str(e)}
        except Exception as e:  # noqa: BLE001 - a crashed request is a wrong answer, recorded, never retried silently
            return {"model": self.model_name, "answers": {}, "usage": {}, "status": "error",
                    "error": f"{type(e).__name__}: {e}"}
        answers = {q: normalize_answer(request["questions"][q], a)
                   for q, a in (resp.get("answers") or {}).items() if q in request["questions"]}
        usage = dict(resp.get("usage") or {})
        usage.setdefault("input_tokens", None)
        usage.setdefault("output_tokens", 0 if self.kind == "decider" else None)
        return {"model": resp.get("model") or self.model_name, "answers": answers, "usage": usage, "status": "ok",
                "error": None}

    def answer_item(self, item: Item, clock: Callable[[], float] = time.perf_counter, qid: str = NEUTRAL_QID) -> dict:
        """One item, one request, timed: the prediction record ``jevkit score`` reads."""
        req = item_request(item, self.model_name, qid)
        t0 = clock()
        resp = self(req)
        dt = clock() - t0
        return {"id": item.id, "answer": resp["answers"].get(qid), "status": resp["status"], "error": resp["error"],
                "usage": resp["usage"], "latency_s": dt, "model": resp["model"]}


def run_items(adapter: SystemOneAdapter, items: Iterable[Item], clock: Callable[[], float] = time.perf_counter) -> list[dict]:
    """Serial, one request at a time (the JevBench timing protocol)."""
    return [adapter.answer_item(it, clock) for it in items]


# ------------------------------------------------------------------------------------------- prediction files
def preds_to_answers(records: Iterable[dict]) -> dict[str, Any]:
    """Prediction records -> {item id: answer}. Accepts jevkit records ({id, answer, status}) and raw SystemOne
    responses keyed by item id ({answers: {item_id: answer}})."""
    out: dict[str, Any] = {}
    for r in records:
        if "id" in r:
            out[str(r["id"])] = r.get("answer") if r.get("status", "ok") == "ok" else None
        elif isinstance(r.get("answers"), dict):
            for k, a in r["answers"].items():
                out[str(k)] = a
        else:
            raise ValueError("prediction record has neither 'id' nor 'answers'")
    return out


# ------------------------------------------------------------------------------------------- Decision Index rows
def _noul_gold_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in ("yes", "true", "1"):
        return "yes"
    if s in ("no", "false", "0"):
        return "no"
    raise ValueError(f"cannot read noul gold {v!r}")


def di_row_to_items(row: dict, tier: str = "standard", split: str | None = None) -> list[Item]:
    """One DI suite row (``questions`` + ``expected`` maps) -> one JevBench item per scored question."""
    rid = str((row.get("_evaluation") or {}).get("run_id") or row["id"])
    exp = row.get("expected") or {}
    out = []
    for qk, q in row["questions"].items():
        gold = exp.get(qk) if isinstance(exp, dict) else exp
        if gold is None:
            continue  # unjudged / unscored question
        if q["type"] == "noul":
            labels, gold = ["no", "yes"], _noul_gold_to_label(gold)
        elif q["type"] == "choice":
            labels, gold = [str(k) for k in q["criteria"]], str(gold)
        else:
            labels, gold = [str(i) for i in range(len(q["criteria"]))], int(gold)
        out.append(Item(id=f"{rid}::{qk}", family=str(row.get("family", "unknown")), state=row.get("state"),
                        question=q, labels=labels, expected=gold, split=split or str(row.get("split", "public")),
                        tier=tier, group=str((row.get("_evaluation") or {}).get("group_id") or "") or None,
                        provenance={"decision_index_run_id": rid, "question_key": qk}).validate())
    return out


def item_to_di_row(item: Item, qid: str = NEUTRAL_QID) -> dict:
    """JevBench item -> a DI-kit-runnable row (noul gold as a bool, as the kit's scorers expect)."""
    gold: Any = item.expected
    if item.type == "noul" and gold is not None:
        gold = gold == "yes"
    return {"id": item.id, "family": item.family, "split": item.split, "state": item.state,
            "questions": {qid: item.question}, "expected": {qid: gold},
            "_evaluation": {"run_id": item.id, "track": f"jevbench_{item.tier}", "group_id": item.group}}


def di_result_to_pred(result: dict, qid: str = NEUTRAL_QID) -> dict:
    """A kit ``results.jsonl`` row (of an ``item_to_di_row`` suite) -> a jevkit prediction record."""
    rid = str(result.get("run_id") or (result.get("_evaluation") or {}).get("run_id"))
    ok = result.get("status") == "ok"
    resp = result.get("response") or {}
    ms = result.get("model_request_wall_ms", result.get("total_wall_ms"))
    return {"id": rid, "answer": (resp.get("answers") or {}).get(qid) if ok else None,
            "status": result.get("status"), "error": result.get("error"), "usage": resp.get("usage") or {},
            "latency_s": None if ms is None else ms / 1000.0, "model": resp.get("model")}
