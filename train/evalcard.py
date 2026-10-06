"""Per-checkpoint eval card hook: decision metrics + harness score + steer rate.

* Decision metrics (accuracy, NLL, Brier, binned and debiased ECE, per length bin) are always
  computed here on the eval items.
* Index-format score uses ``harness.score`` (PR #7) and steer rate uses ``safety_eval.metrics``
  (PR #8), both on agent-main. Each is reported as ``{"status": "stub"}`` with the reason unless
  importable AND the caller passes the inputs it needs (harness-format rows + predictor; steer
  variants + decider). A stub is never reported as a number.
* The steer summary is reconciled against the expected unit set (the variants), so its
  ``complete`` flag is meaningful; without it every family reports "completeness unverifiable".
* ``cohort`` (e.g. the development-cohort manifest hashes) is stamped into the card (SAFETY_SPEC
  G12). ``keep_units=True`` returns the per-unit steer rows and harness results under
  ``card["units"]`` for paired comparisons (``train.promotion``); they are written to a separate
  ``units-step*.json`` file, not into the card.
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .calibration import FeatureTemperature, features
from .metrics import decision_metrics


def _try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # noqa: BLE001 - report any import failure as a stub reason
        return None, f"{type(exc).__name__}: {exc}"


@torch.no_grad()
def predict(model, items, batch_size: int = 8) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    model.eval()
    lps, labels, ev, ks = [], [], [], []
    kmax = max(i.row.n_options for i in items)
    for s in range(0, len(items), batch_size):
        chunk = items[s : s + batch_size]
        lp = model([i.row for i in chunk]).log_probs.detach().float().cpu()
        pad = torch.full((lp.shape[0], kmax), float("-inf"))
        pad[:, : lp.shape[1]] = lp
        lps.append(pad)
        labels += [i.label for i in chunk]
        ev += [i.row.evidence_len for i in chunk]
        ks += [i.row.n_options for i in chunk]
    return torch.cat(lps), labels, ev, ks


def run_card(model, items, step: int, out_dir: str | Path | None = None,
             temperature: FeatureTemperature | None = None,
             index_rows: list | None = None, index_predict: Callable | None = None,
             steer_variants: list | None = None, steer_decide: Callable | None = None,
             cohort: dict | None = None, keep_units: bool = False, n_boot: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    if temperature is not None and temperature.fitted_on is not None:
        eval_hash = hashlib.sha256("\n".join(sorted(i.id for i in items)).encode()).hexdigest()
        if temperature.fitted_on == eval_hash:
            raise ValueError("temperature was fitted on the eval items themselves; use a separate calibration split")
        content = [i.row.meta.get("content_sha256") for i in items]
        temperature.assert_disjoint([i.id for i in items], "eval items",
                                    content=[c for c in content if c])  # shared record OR shared content raises
    lp, labels, ev, ks = predict(model, items)
    card: dict[str, Any] = {"step": step, "n_items": len(items), "cohort": cohort}
    units: dict[str, Any] = {}
    probs = lp.exp().tolist()
    card["raw"] = decision_metrics([p[:k] for p, k in zip(probs, ks)], labels, ev)
    units["decision_probs"] = [p[:k] for p, k in zip(probs, ks)]
    if temperature is not None:
        lpt = temperature.apply(lp, features(ev, ks))
        pt = lpt.exp().tolist()
        card["tempered"] = decision_metrics([p[:k] for p, k in zip(pt, ks)], labels, ev)
        card["temperature"] = temperature.identity()
        units["decision_probs_tempered"] = [p[:k] for p, k in zip(pt, ks)]
    harness, why_h = _try_import("harness.score")
    if harness is None or index_rows is None or index_predict is None:
        card["index_score"] = {"status": "stub", "reason": why_h or "no Index dev-slice rows/predictor passed"}
    else:
        results = index_predict(index_rows)
        card["index_score"] = {"status": "ok", "n_rows": len(index_rows),
                               "statuses": dict(collections.Counter(r["status"] for r in results.values()))}
        present = {r["_evaluation"]["catalog_id"] for r in index_rows}
        from harness.suite import index_ids

        if set(index_ids()) <= present:  # the kit's official aggregate needs every index benchmark
            off = harness.official(index_rows, results)
            card["index_score"].update(method="official", scores=off["scores"], values=off["values"])
        else:  # partial slice: the harness's renormalised dev-slice index over the benchmarks present
            pt = harness.point(harness.decompose(index_rows, results), renormalize=True)
            card["index_score"].update(method="decompose+point (renormalised over present benchmarks)",
                                       scores={"index": pt["index"]}, values=pt["per_benchmark"],
                                       benchmarks_present=sorted(present))
        units["index_results"] = results
    metrics, why_s = _try_import("safety_eval.metrics")
    if metrics is None or steer_variants is None or steer_decide is None:
        card["steer"] = {"status": "stub", "reason": why_s or "no steer variants/decider passed"}
    else:
        rows = metrics.run(steer_decide, steer_variants)
        kw = {} if n_boot is None else {"n_boot": n_boot}
        card["steer"] = {"status": "ok", "summary": metrics.summarise(rows, expected=steer_variants, **kw)}
        units["steer_rows"] = rows
    card["seconds"] = round(time.time() - t0, 3)
    if out_dir is not None:
        p = Path(out_dir) / f"card-step{step:07d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(card, indent=1, default=str), encoding="utf-8")
        if keep_units:
            (Path(out_dir) / f"units-step{step:07d}.json").write_text(json.dumps(units, default=str), encoding="utf-8")
    if keep_units:
        card["units"] = units
    return card
