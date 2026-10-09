"""Per-readout temperature refit for a saved cell scored under a NON-training readout (``readout_ab``, ``fulldev_eval
--readout``).

Why. The served temperature of a cell is ONE ``train.calibration.FeatureTemperature`` fitted on the training
template's prompts (``train.m2_gates.fit_child_temperature`` on the cohort ``calibration`` role). Applied to another
readout's logits it is "the deployed calibrator on another prompt": job-39gnn (D1 v1 vs v3w) saw ECE10 0.024 -> 0.061
under v3w with it. The refit here is the SAME procedure on the SAME calibration rows with the scores produced under the
readout being served:

  * rows: the cohort v2 ``calibration`` role (1,000 records; ``train.cohort`` role_use "temperature fit only"), the
    items the training readout's fit used (ids = ``ctx.raw_items``), re-rendered under the readout. An item the
    readout's own ``keep_item`` filter (K cap, ``eval_max_t``) would drop is left out and counted, never replaced;
  * objective / bounds: ``FeatureTemperature.fit`` defaults (NLL of the tempered distribution with frozen logits,
    L-BFGS 200 iterations, strong Wolfe, l2 1e-4 on the feature weights, t_min 0.05, the identity start), through
    ``train.m2_gates.fit_child_temperature`` itself;
  * disjointness: the in-job proof (content hashes vs cohort dev = "eval", retention cohort v3 = "retention", the
    training items = "train") PLUS the Index DEV content: the 2,008-row subset and the full 12,104-row DEV when the
    rows are given (``index_dev_subset`` / ``index_dev_full``); any shared content raises ``CalibrationLeak`` before
    any forward. DEV / TEST / held-out Index rows and cohort v3 are never fitted on.

A temperature is a per-question monotone rescaling of the logits, so the argmax never changes; the scorer's Index is
computed on the untempered ``response`` either way. ``refit_answers`` asserts the argmax per question and re-derives
the raw-template served answers from the same logits as a self-check (they must equal the adapter's exactly).

The training readout is never refitted: its fit, decider, files and identities are byte-identical to before.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REFIT_SCHEMA = "ssdm.m2.readout_temperature.v1"


@dataclass
class CalibrationContext:
    """What ``fulldev_eval.load_child`` already built for the in-job fit, kept for a refit under another readout."""

    model: Any
    encode: Any
    guard: Any
    spec: dict  # train.m2_gates.unit_spec(cfg): max_k, max_t, template, layout (the training readout)
    calib_records: list  # the cohort calibration role's records (train-record format)
    raw_items: list  # the calibration items of the in-job fit (training readout)
    disjoint_items: dict  # {"eval": items, "retention": items, "train": items}: the in-job fit's disjoint_from
    raw: Any = None  # the in-job train.calibration.FeatureTemperature
    cache: dict = field(default_factory=dict)


def index_content(rows) -> list[str]:
    """Normalized content hash of each Index row (``train.cohort.normalized_content_hash``: the hash calibration items
    carry in ``row.meta['content_sha256']``)."""
    from train.cohort import normalized_content_hash

    return [normalized_content_hash(r["state"], r["questions"]) for r in rows]


def read_index_rows(path: str | Path) -> list[dict]:
    from harness.suite import read_rows

    return read_rows(str(path))


def readout_items(ctx: CalibrationContext, template: str, layout: str) -> tuple[list, dict[str, Any]]:
    """The in-job fit's calibration items re-rendered under (``template``, ``layout``): same records, same guard, the
    readout's own ``keep_item`` filter; restricted to the ids the training readout's fit used (never new rows)."""
    from train import m2_gates as G

    spec = {**ctx.spec, "template": template, "layout": layout}
    its = G.decision_items(ctx.calib_records, ctx.encode, spec, ctx.guard)
    raw_ids = [i.id for i in ctx.raw_items]
    keep = set(raw_ids)
    got = [i for i in its if i.id in keep]
    have = {i.id for i in got}
    extra = sorted({i.id for i in its} - keep)
    info = {"n_raw_items": len(raw_ids), "n_items": len(got), "dropped_by_readout_filter": sorted(keep - have),
            "n_not_in_raw_fit": len(extra), "spec": spec}
    if not got:
        raise ValueError(f"no calibration item survives under {template}/{layout}")
    return got, info


def refit(ctx: CalibrationContext, template: str, layout: str, index_rows: dict[str, list] | None = None,
          items_info: tuple[list, dict] | None = None):
    """(train.calibration.FeatureTemperature, info): the in-job procedure on the same calibration rows, scored under
    the readout. ``index_rows`` = {name: Index rows} added to the disjointness proof (content)."""
    from train import m2_gates as G
    from train.calibration import check_disjoint, item_content

    items, info = items_info or readout_items(ctx, template, layout)
    info = dict(info)
    idx = {name: index_content(rows) for name, rows in (index_rows or {}).items()}
    cal = item_content(items)
    # Index disjointness BEFORE any forward (content); then the fit's own proof (eval / retention / train + index)
    info["index_disjointness"] = check_disjoint(cal, idx) if idx else {}
    info["index_rows"] = {k: len(v) for k, v in idx.items()}
    ft = G.fit_child_temperature(ctx.model, items, {**ctx.disjoint_items, **idx})
    info["fit_report"] = ft.fit_report
    return ft, info


def serving_json(st) -> dict[str, Any]:
    """A decide.calibration.FeatureTemperature as JSON (persisted so a restart serves the SAME refit)."""
    return {"schema": REFIT_SCHEMA, "w": list(st.w), "t_min": st.t_min, "fitted_on": st.fitted_on,
            "disjointness": [list(x) for x in st.disjointness], "calib_content": list(st.calib_content),
            "proof_sha256": st.proof_sha256}


def serving_from_json(d: dict):
    from decide.calibration import FeatureTemperature

    if d.get("schema") != REFIT_SCHEMA:
        raise ValueError(f"refit file schema {d.get('schema')!r} != {REFIT_SCHEMA}")
    st = FeatureTemperature(w=tuple(d["w"]), t_min=float(d["t_min"]), fitted_on=d["fitted_on"],
                            disjointness=tuple(tuple(x) for x in d["disjointness"]),
                            calib_content=tuple(d["calib_content"]), proof_sha256=d["proof_sha256"])
    st.assert_servable()
    return st


def refit_served(ctx: CalibrationContext, template: str, layout: str, cache_file: Path | None,
                 index_rows: dict[str, list] | None = None) -> tuple[Any, dict[str, Any]]:
    """The serving form of the readout's refit + its identity block. A persisted ``cache_file`` (the resume dir) is
    reused when its calibration content set equals this readout's items (a restart serves the SAME temperature, no
    second forward over the calibration rows); otherwise the refit runs and is persisted."""
    from decide.calibration import content_set_sha256
    from train.calibration import item_content

    items, info0 = readout_items(ctx, template, layout)
    want = content_set_sha256(item_content(items))
    if cache_file is not None and cache_file.is_file():
        d = json.loads(cache_file.read_text(encoding="utf-8"))
        st = serving_from_json(d["serving"])
        if st.fitted_on == want:
            return st, {**d["info"], "resumed": True, "identity": st.identity()}
    ft, info = refit(ctx, template, layout, index_rows, (items, info0))
    st = ft.to_serving()
    if st.fitted_on != want:
        raise AssertionError("refit fitted_on != the readout's calibration content set")
    st.assert_servable()
    info = {**{k: v for k, v in info.items() if k != "spec"}, "template": template, "layout": layout,
            "n_dropped_by_readout_filter": len(info["dropped_by_readout_filter"]),
            "procedure": "train.m2_gates.fit_child_temperature (FeatureTemperature.fit defaults: NLL, L-BFGS 200, "
                         "strong Wolfe, l2 1e-4, t_min 0.05) on the cohort calibration role, scores under this readout",
            "raw_template_temperature": ctx.raw.identity() if ctx.raw is not None else None}
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"serving": serving_json(st), "info": info}, indent=1), encoding="utf-8")
    return st, {**info, "resumed": False, "identity": st.identity()}


# ------------------------------------------------------------------------------------------------ serving
class Capture:
    """Records the backend's last ``ScoreResult`` (logits + features) for the row being decided: wraps the INSTANCE's
    ``score`` (no change to what it returns, so the served decider is untouched)."""

    def __init__(self, backend):
        self.last = None
        inner = backend.score

        def score(*args, **kw):
            res = inner(*args, **kw)
            self.last = res
            return res

        backend.score = score

    def reset(self) -> None:
        self.last = None


def _argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def refit_answers(questions: dict, result, raw_cal, refit_cal, served_answers: dict) -> dict[str, Any]:
    """{"answers": ...} under ``refit_cal`` from the SAME logits the adapter served (``result`` = the captured
    ScoreResult). Asserts (1) the raw-template temperature re-derived here reproduces the adapter's served answers
    exactly, and (2) per question the refit's argmax is the logits' argmax (a temperature never changes it)."""
    from decide.calibration import softmax
    from decide.schema import canonical_options, wire_answer

    out = {}
    feats = getattr(result, "features", None) or {}
    for qid, q in questions.items():
        oids = [o for o, _ in canonical_options(q)]
        vec = [float(result.logits[qid][o]) for o in oids]
        k = len(oids)
        t_raw = raw_cal.temperature_for(k, feats.get(qid))
        if wire_answer(q, dict(zip(oids, softmax(vec, t_raw)))) != served_answers[qid]:
            raise AssertionError(f"{qid}: re-derived raw-template served answer != the adapter's")
        p = softmax(vec, refit_cal.temperature_for(k, feats.get(qid)))
        a, b = _argmax(vec), _argmax(p)
        if a != b and not math.isclose(p[a], p[b], rel_tol=0, abs_tol=0):
            raise AssertionError(f"{qid}: refit temperature changed the argmax ({oids[a]} -> {oids[b]})")
        out[qid] = wire_answer(q, dict(zip(oids, p)))
    return {"answers": out}


def predict_with_refit(decide: Callable, row: dict, cap: Capture, raw_cal, refit_cal, tempered: bool) -> dict:
    """``train.m2_gates.index_predict`` on one row plus ``refit_response`` (ok rows): the readout's refit temperature
    applied to the same forward's logits (no second forward)."""
    from train import m2_gates as G

    cap.reset()
    res = G.index_predict(decide, [row], untempered_from_raw=tempered)[row["_evaluation"]["run_id"]]
    if res.get("status") == "ok":
        if cap.last is None:
            raise AssertionError("no captured score result for an ok row")
        served = (res.get("served_response") or res["response"])["answers"]
        res["refit_response"] = refit_answers(row["questions"], cap.last, raw_cal, refit_cal, served)
    return res


def readout_decider(base, template: str, layout: str):
    """The base (training-readout) served decider's model, encoder, checkpoint record and RAW-template calibration
    under another readout (``readout_ab.child_deciders``' construction)."""
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend

    b0 = base.backend
    be = DecisionModelBackend(b0.model, b0.encode, template=template, layout=layout, checkpoint=b0.checkpoint,
                              shared_prefix=b0.shared_prefix)
    return Decider(be, calibration=base.calibration)


#: extra scoring-minute model of a refit = one predict() pass over the calibration items under the readout
def refit_minutes(n_items: int, tokens_per_item: float, s_per_1k_tokens: float, n_readouts: int) -> float:
    return n_readouts * n_items * tokens_per_item / 1000 * s_per_1k_tokens / 60
