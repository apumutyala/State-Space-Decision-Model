"""Mode ``ragtruth_polarity``: does a trained cell rank RAGTruth responses with INVERTED polarity? (TRAIN split only.)

Hypothesis (agent B, 2026-10-09): D13 (job-hjwpq art-cr9a2, best step 1600) under readout v3w scores RAGTruth with
inverted polarity (proxy600 AUROC 0.18; D1 predicts nearly all-positive). The decision may use ONLY the RAGTruth TRAIN
split (15,090 responses, prevalence 0.445), never Index DEV. Pre-registered rule (``DECISION_RULE``, README
``jobs/m2_first_cell/RAGTRUTH_POLARITY.md``), applied per arm:

    polarity INVERTED if the AUROC 95 % bootstrap upper bound < 0.5 on train -> a statement rephrase is proposed;
    if the CI includes 0.5 -> no change. (Upper bound < 0.5 is the only "inverted" outcome; anything else is no change.)

Items are built EXACTLY as the Index presents RAGTruth (catalog 59) rows: ``state = {"prompt": source_info.prompt,
"response": response}``, the single ``noul`` question ``q`` (``QUESTION``, byte-identical to every DEV RAGTruth row),
``expected.q = bool(labels)`` (true = hallucinated), the kit's metadata keys. ``check_dev_format`` rebuilds all 270
DEV RAGTruth rows from the same source files and requires equality (state, questions, expected, metadata, scoring,
benchmark fields) before any item is written. Scoring goes through ``train.m2_gates.index_predict`` (the Index arm's
own path) on the decider ``fulldev_eval.load_child`` builds for readout v3w (untempered, as the screen scores it), so the
prompt is rendered by the same template code as the Index. p(hallucinated) = ``answers.q.noul`` (p_true).

Selection (CPU, local, before packaging; ``build``): RAGTruth TRAIN responses only, minus
  (a) any response whose source document id (``source_id``) is the ``source_task_id`` of an Index DEV RAGTruth row;
  (b) any response the document-overlap gate fails (``train.doc_overlap_guard`` scan report, rule doc_overlap_guard/v1:
      13-gram overlap with the document text of ANY Index eval row, DEV + TEST + held-out; a superset of DEV);
then at most ONE response per source document (independent items), stratified by (task_type, label) with
proportional allocation to ``N_TARGET``, each stratum ordered by sha256(f"{SEED}:{id}"). The ids sha256 is recorded.

    python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity records --src <ragtruth dir> --out recs.jsonl.gz
    python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity build --src <ragtruth dir> --dev-rows <dev-rows.jsonl.gz> \
        --guard-report <scan.json> --out <items.jsonl.gz> --selection <selection.json>
    python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity estimate --params <job_params> --selection <selection.json>
    # GPU (in the job), once per arm:
    python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity score --arm d13 --items <items> --cell <cell.json> ...
    python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity report --out-dir <out> --params <p> --items <items> --out <r>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ssdm.m2.ragtruth_polarity.v1"
CATALOG_ID = 59
DATASET = "RAGTruth response-level hallucination"
SEED = "ragtruth-polarity-v1"
N_TARGET = 300
N_BOOT = 2000
BOOT_SEED = 20261009
CI_LEVEL = 0.95
NULL_AUROC = 0.5
READOUT = "v3w"
ARM_NAMES = ("d13", "d1")
# Arms whose checkpoint was trained BEFORE the attack guard (train.attack_guard, ca9d482): their cell config pins no
# train_set.attack_clean, so the lineage check must rebuild the UNfiltered draw (train.train_set.draw lineage_replay;
# eval-only, nothing is trained). D1 = job-c7pq2 (trained 2026-10-08 on the 150k set incl. the 8,049 D1-A steer rows;
# drawn_ids_sha256 2bbbf180..., n_pool 111,795). job-vftxa failed exit 10 at score_d1 without this.
LINEAGE_REPLAY_ARMS = ("d1",)
RUN_ID_PREFIX = "ragtruth-train:59:"
DECISION_RULE = ("per arm: polarity is INVERTED if the AUROC 95% percentile-bootstrap upper bound (2000 item "
                 "resamples, seed 20261009) < 0.5 on the RAGTruth TRAIN items; then a statement rephrase is proposed. "
                 "If the CI includes 0.5 (or lies above it), no change.")
#: the Index's RAGTruth question, verbatim (every one of the 270 DEV RAGTruth rows carries exactly this block)
QUESTION = {"q": {"type": "noul",
                  "instructions": "The response contains content that is not supported by the context in the prompt.",
                  "criteria": {"true": "The response contains content that is not supported by the context in the prompt.",
                               "false": "All content of the response is supported by the context in the prompt."}}}
SCORING = {"type": "candidate_decisions", "primary_field": "q"}


# ------------------------------------------------------------------------------------------------ item builder
def read_jsonl(path: str | Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_source(src_dir: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """({response id: response}, {source_id: source_info}) from the RAGTruth release files."""
    d = Path(src_dir)
    resp = {str(r["id"]): r for r in read_jsonl(d / "response.jsonl")}
    srcs = {str(s["source_id"]): s for s in read_jsonl(d / "source_info.jsonl")}
    return resp, srcs


def label_types(labels: list[dict]) -> list[str]:
    return sorted({str(x.get("label_type")) for x in labels or []})


def payload_sha256(state: dict, questions: dict) -> str:
    """Our own content hash of an item (canonical JSON of state + questions); NOT the kit's payload recipe."""
    return hashlib.sha256(json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def build_row(resp: dict, src: dict) -> dict:
    """One RAGTruth response as an Index catalog-59 row (the kit's row shape; run_id prefixed ``ragtruth-train:``)."""
    rid = str(resp["id"])
    if str(resp["source_id"]) != str(src["source_id"]):
        raise ValueError(f"response {rid}: source {resp['source_id']} != {src['source_id']}")
    state = {"prompt": src["prompt"], "response": resp["response"]}
    questions = json.loads(json.dumps(QUESTION))
    return {"id": f"59:{rid}", "benchmark": DATASET, "family": DATASET, "split": "evaluation", "state": state,
            "questions": questions, "expected": {"q": bool(resp["labels"])},
            "metadata": {"source_id": rid, "group_id": f"59:{rid}", "track": src["task_type"],
                         "source_task_id": str(src["source_id"]), "responder_model": resp["model"],
                         "quality": resp["quality"], "label_types": label_types(resp["labels"])},
            "scoring": dict(SCORING),
            "_evaluation": {"run_id": RUN_ID_PREFIX + rid, "catalog_id": CATALOG_ID, "dataset": DATASET,
                            "track": src["task_type"], "group_id": f"59:{rid}",
                            "payload_sha256": payload_sha256(state, questions),
                            "benchmark_origin": "RAGTruth TRAIN split (ragtruth_polarity; not an Index row)",
                            "ragtruth_split": resp["split"]}}


#: row fields compared against the Index DEV rows (``_evaluation`` differs by design: run_id, payload recipe, batch)
FORMAT_KEYS = ("id", "benchmark", "family", "split", "state", "questions", "expected", "metadata", "scoring")


def dev_ragtruth_rows(dev_rows: Iterable[dict]) -> list[dict]:
    return [r for r in dev_rows if r["_evaluation"]["catalog_id"] == CATALOG_ID]


def check_dev_format(dev_rows: list[dict], resp: dict[str, dict], srcs: dict[str, dict]) -> dict[str, Any]:
    """Rebuild every Index DEV RAGTruth row from the release files with ``build_row``; ValueError unless every
    ``FORMAT_KEYS`` field is equal (the item builder reproduces the Index's presentation byte for byte)."""
    rows = dev_ragtruth_rows(dev_rows)
    if not rows:
        raise ValueError("no Index DEV RAGTruth rows to check the format against")
    bad = []
    for r in rows:
        rr = resp.get(str(r["metadata"]["source_id"]))
        built = build_row(rr, srcs[str(rr["source_id"])]) if rr else None
        if built is None or any(built[k] != r[k] for k in FORMAT_KEYS):
            bad.append(r["_evaluation"]["run_id"])
    if bad:
        raise ValueError(f"{len(bad)} of {len(rows)} DEV RAGTruth rows do not rebuild from the sources, e.g. {bad[:3]}")
    return {"dev_rows_checked": len(rows), "fields": list(FORMAT_KEYS), "equal": True}


def guard_records(resp: dict[str, dict], srcs: dict[str, dict]) -> Iterable[dict]:
    """TRAIN responses as doc_overlap_guard scan records (id = response id, source = "ragtruth", state + questions)."""
    for rid, r in sorted(resp.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        if r.get("split") != "train":
            continue
        row = build_row(r, srcs[str(r["source_id"])])
        yield {"id": rid, "source": "ragtruth", "family": "ragtruth", "state": row["state"],
               "questions": row["questions"]}


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _rank(seed: str, rid: str) -> str:
    return hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()


def allocate(counts: dict[tuple, int], n: int) -> dict[tuple, int]:
    """Proportional allocation of ``n`` over strata (largest remainder; ties by stratum key); capped at stratum size."""
    tot = sum(counts.values())
    raw = {k: n * c / tot for k, c in counts.items()}
    out = {k: min(counts[k], int(math.floor(v))) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - math.floor(raw[k])), k)):
        if sum(out.values()) >= n:
            break
        if out[k] < counts[k]:
            out[k] += 1
    return out


def select(resp: dict[str, dict], srcs: dict[str, dict], exclude: set[str], n: int = N_TARGET,
           seed: str = SEED) -> dict[str, Any]:
    """Stratified (task_type x label) TRAIN sample of ``n`` responses, at most one per source document, minus
    ``exclude``. Strata sizes are counted over the ELIGIBLE one-per-source pool (each source's first eligible response in
    seed order), so allocation matches the pool the items are drawn from."""
    elig = [r for rid, r in resp.items() if r.get("split") == "train" and rid not in exclude]
    elig.sort(key=lambda r: _rank(seed, str(r["id"])))
    first: dict[str, dict] = {}
    for r in elig:  # one response per source document: the first in seed order
        first.setdefault(str(r["source_id"]), r)
    pool: dict[tuple, list[dict]] = {}
    for r in first.values():
        pool.setdefault((srcs[str(r["source_id"])]["task_type"], bool(r["labels"])), []).append(r)
    for k in pool:
        pool[k].sort(key=lambda r: _rank(seed, str(r["id"])))
    alloc = allocate({k: len(v) for k, v in pool.items()}, n)
    picked = [r for k in sorted(pool) for r in pool[k][:alloc[k]]]
    ids = sorted(str(r["id"]) for r in picked)
    return {"ids": ids, "ids_sha256": ids_sha256(ids), "n": len(ids), "seed": seed,
            "strata": {f"{k[0]}|{'hallucinated' if k[1] else 'supported'}": {"pool": len(pool[k]), "picked": alloc[k]}
                       for k in sorted(pool)},
            "prevalence": round(sum(bool(r["labels"]) for r in picked) / max(len(picked), 1), 4),
            "n_train_eligible": len(elig), "n_sources_eligible": len(first)}


def dev_exclusions(resp: dict[str, dict], dev_rows: list[dict], guard_fail_ids: Iterable[str]) -> dict[str, Any]:
    """TRAIN response ids excluded from selection: (a) source document id shared with an Index DEV RAGTruth row,
    (b) doc_overlap_guard fails (document text overlapping ANY Index eval row, DEV included)."""
    dev_src = {str(r["metadata"]["source_task_id"]) for r in dev_ragtruth_rows(dev_rows)}
    dev_resp = {str(r["metadata"]["source_id"]) for r in dev_ragtruth_rows(dev_rows)}
    train = {rid for rid, r in resp.items() if r.get("split") == "train"}
    a = {rid for rid in train if str(resp[rid]["source_id"]) in dev_src}
    b = {str(x) for x in guard_fail_ids} & train
    if train & dev_resp:
        raise ValueError(f"{len(train & dev_resp)} TRAIN response ids are Index DEV RAGTruth rows")
    return {"source_id_in_dev": sorted(a), "doc_overlap_guard_fail": sorted(b), "all": sorted(a | b),
            "n_source_id_in_dev": len(a), "n_doc_overlap_guard_fail": len(b), "n_excluded": len(a | b),
            "n_train": len(train), "n_dev_ragtruth_rows": len(dev_resp), "n_dev_sources": len(dev_src)}


# ------------------------------------------------------------------------------------------------ metrics
def auroc(scores: list[float], labels: list[bool]) -> float:
    """Mann-Whitney AUROC (ties = 1/2): P(score of a positive > score of a negative). NaN if a class is empty."""
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rpos = sum(r for r, y in zip(ranks, labels) if y)
    return (rpos - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_auroc(scores: list[float], labels: list[bool], b: int = N_BOOT, seed: int = BOOT_SEED,
                    level: float = CI_LEVEL) -> dict[str, Any]:
    """Point AUROC + percentile bootstrap CI over item resamples (items are independent: one per source document).
    Resamples with a single class are skipped and counted."""
    import random

    rng = random.Random(seed)
    n = len(scores)
    reps, skipped = [], 0
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        v = auroc([scores[i] for i in idx], [labels[i] for i in idx])
        if math.isnan(v):
            skipped += 1
        else:
            reps.append(v)
    reps.sort()
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

    def q(p: float) -> float:
        if not reps:
            return float("nan")
        x = p * (len(reps) - 1)
        f = math.floor(x)
        c = min(f + 1, len(reps) - 1)
        return reps[f] + (reps[c] - reps[f]) * (x - f)

    return {"point": auroc(scores, labels), "lo": q(lo_q), "hi": q(hi_q), "level": level, "B": b, "seed": seed,
            "skipped_single_class": skipped, "n": n, "n_pos": sum(1 for y in labels if y)}


def decide_polarity(ci: dict[str, Any]) -> str:
    """The pre-registered rule (``DECISION_RULE``) on one arm's AUROC CI."""
    hi = ci.get("hi")
    if isinstance(hi, (int, float)) and math.isfinite(hi) and hi < NULL_AUROC:
        return "inverted: propose a statement rephrase"
    return "no change"


def arm_metrics(items: list[dict], results: dict[str, dict]) -> dict[str, Any]:
    """Per-item p(hallucinated) (answers.q.noul) vs expected.q over the ok rows; AUROC + CI, flipped AUROC + CI
    (score 1 - p), F1 / accuracy at 0.5, and the rule's decision."""
    per, sc, lab, status = [], [], [], {}
    for it in items:
        rid = it["_evaluation"]["run_id"]
        r = results.get(rid) or {"status": "missing"}
        status[r["status"]] = status.get(r["status"], 0) + 1
        p = None
        if r["status"] == "ok":
            p = float(r["response"]["answers"]["q"]["noul"])
            sc.append(p)
            lab.append(bool(it["expected"]["q"]))
        per.append({"run_id": rid, "track": it["metadata"]["track"], "hallucinated": bool(it["expected"]["q"]),
                    "p_hallucinated": p, "status": r["status"]})
    ci = bootstrap_auroc(sc, lab)
    flipped = bootstrap_auroc([1 - x for x in sc], lab)
    tp = sum(1 for p, y in zip(sc, lab) if p >= 0.5 and y)
    fp = sum(1 for p, y in zip(sc, lab) if p >= 0.5 and not y)
    fn = sum(1 for p, y in zip(sc, lab) if p < 0.5 and y)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    by_track = {}
    for t in sorted({x["track"] for x in per}):
        s_ = [x["p_hallucinated"] for x in per if x["track"] == t and x["p_hallucinated"] is not None]
        l_ = [x["hallucinated"] for x in per if x["track"] == t and x["p_hallucinated"] is not None]
        by_track[t] = {"n": len(s_), "auroc": auroc(s_, l_)}
    return {"status": status, "n_scored": len(sc), "auroc": ci, "auroc_flipped": flipped,
            "mean_p_hallucinated": (sum(sc) / len(sc)) if sc else float("nan"),
            "frac_pred_hallucinated": (sum(1 for p in sc if p >= 0.5) / len(sc)) if sc else float("nan"),
            "f1_hallucinated_at_0p5": f1, "accuracy_at_0p5": ((sum(1 for p, y in zip(sc, lab) if (p >= 0.5) == y)
                                                                / len(sc)) if sc else float("nan")),
            "by_track_auroc": by_track, "decision": decide_polarity(ci), "per_item": per}


# ------------------------------------------------------------------------------------------------ GPU: score
def score(a) -> int:
    from jobs.m2_first_cell.fulldev_eval import checkpoint_declarer, limited, load_child, readout_spec
    from jobs.m2_pilot.guard_ready import GuardNotReady
    from train import m2_gates as G
    from train.cohort import CohortLeak

    items = read_jsonl(a.items)
    want = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    ids = [it["metadata"]["source_id"] for it in items]
    if ids_sha256(ids) != want["ids_sha256"] or len(ids) != want["n"]:
        print("items: ids sha256 / n != the selection pin", file=sys.stderr)
        return 10
    try:
        cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
        a.lineage_replay = a.arm in LINEAGE_REPLAY_ARMS
        if a.lineage_replay and (cfg.get("train_set") or {}).get("attack_clean"):
            raise ValueError(f"arm {a.arm}: lineage replay is for pre-guard cells, but its config pins attack_clean")
        ro = readout_spec(READOUT, cfg)
        decider, _, ident = load_child(a, temperature=False, readout=(ro["template"], ro["layout"]))
        from decide.ragtruth_rephrase import wording_sha256
        ident = {**ident, "readout": ro, "arm_name": a.arm, "lineage_replay": a.lineage_replay,
                 "wording": a.wording, "wording_sha256": wording_sha256(a.wording)}
    except (ValueError, KeyError, GuardNotReady, CohortLeak) as e:
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"identity-{a.arm}.json").write_text(json.dumps(ident, indent=1, default=str), encoding="utf-8")
    resume = Path(a.resume_dir) / f"{a.arm}.partial.jsonl"
    resume.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if resume.is_file():
        for x in resume.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(x)
                done[r["run_id"]] = r
            except (ValueError, KeyError):
                break  # a torn tail: rescored
    todo = [it for it in items if it["_evaluation"]["run_id"] not in done]
    prog = G.Progress(a.progress)
    prog.plan([G.unit_weight(it) for it in todo])
    from decide.ragtruth_rephrase import rephrased
    # B-PREREG-RAGTRUTH-REPHRASE: --wording rt1 rewrites the catalog-59 question at readout time (original = identity)
    decide = rephrased(G.beat_wrap(prog.wrap(limited(decider.decide, a.max_input_tokens))), a.wording)
    declare = checkpoint_declarer()
    t0 = time.perf_counter()
    with open(resume, "w", encoding="utf-8", newline="\n") as f:  # rewritten from the parsed rows: drops a torn tail
        for r in done.values():
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
        for it in todo:
            res = G.index_predict(decide, [it])
            f.write(json.dumps(res[it["_evaluation"]["run_id"]], separators=(",", ":")) + "\n")
            f.flush()
            declare()
    declare(force=True)
    secs = time.perf_counter() - t0
    done = {json.loads(x)["run_id"]: json.loads(x) for x in resume.read_text(encoding="utf-8").splitlines() if x.strip()}
    rids = [it["_evaluation"]["run_id"] for it in items]
    if set(done) != set(rids):
        print(f"{len(set(rids) - set(done))} items have no result", file=sys.stderr)
        return 9
    with open(out / f"{a.arm}.results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for rid in rids:
            f.write(json.dumps(done[rid], separators=(",", ":")) + "\n")
    (out / f"timing-{a.arm}.json").write_text(json.dumps(
        {"seconds_this_attempt": secs, "rows_this_attempt": len(todo),
         "sum_row_wall_s": sum(float(done[i].get("total_wall_ms") or 0) for i in rids) / 1000}, indent=1),
        encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ CPU: report
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def report(a) -> dict[str, Any]:
    from jobs.common.costs import worst_case_usd

    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    items = read_jsonl(a.items)
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    out: dict[str, Any] = {"schema": SCHEMA, "split": "RAGTruth TRAIN (never Index DEV)", "readout": READOUT,
                           "n_items": len(items), "ids_sha256": sel["ids_sha256"], "items_sha256": _sha(Path(a.items)),
                           "decision_rule": DECISION_RULE, "wording": params.get("wording", "original"),
                           "arms_expected": list(params.get("arms", ARM_NAMES)), "arms": {}}
    for arm in params.get("arms", ARM_NAMES):
        rp = d / f"{arm}.results.jsonl"
        if not rp.is_file():
            continue
        res = {r["run_id"]: r for r in read_jsonl(rp)}
        m = arm_metrics(items, res)
        (d / f"{arm}.per_item.jsonl").write_text("".join(json.dumps(x) + "\n" for x in m.pop("per_item")),
                                                 encoding="utf-8")
        m["results_sha256"] = _sha(rp)
        idf = d / f"identity-{arm}.json"
        if idf.is_file():
            idc = json.loads(idf.read_text(encoding="utf-8"))
            m["checkpoint_loaded"] = idc.get("checkpoint")
            m["readout_loaded"] = idc.get("readout")
            m["wording_loaded"] = idc.get("wording")
        tm = d / f"timing-{arm}.json"
        if tm.is_file():
            m["timing"] = json.loads(tm.read_text(encoding="utf-8"))
        out["arms"][arm] = m
    usd = params["batch_usd_per_min"]
    out["cost"] = {"elapsed_min": a.elapsed_min, "runtime_usd_batch": round(a.elapsed_min * usd, 2),
                   "worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"]}
    out["checkpoints"] = {k: {kk: vv for kk, vv in (params.get(k) or {}).items() if not kk.startswith("_")}
                          for k in ("checkpoint", "checkpoint_d1")}
    out["verdict"] = "measured"
    return out


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: both arms measured on every item, finite AUROC CIs, loaded checkpoints
    equal the pins."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        ok = d.get("schema") == SCHEMA and d.get("verdict") == "measured" and set(d["arms"]) == set(d.get("arms_expected", ARM_NAMES))
        pins = {"d13": d["checkpoints"]["checkpoint"]["ckpt_sha256"], "d1": d["checkpoints"]["checkpoint_d1"]["ckpt_sha256"]}
        for arm, m in d["arms"].items():
            ok = ok and sum(m["status"].values()) == d["n_items"] and m["n_scored"] > 0
            ok = ok and m.get("wording_loaded", "original") == d.get("wording", "original")
            ok = ok and all(fin(m[k][f]) for k in ("auroc", "auroc_flipped") for f in ("point", "lo", "hi"))
            ok = ok and (m.get("checkpoint_loaded") or {}).get("sha256") == pins[arm]
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ CPU: pricing
SLOW_BOX = 1.25  # every cap must still hold on a box this much slower than the measured rates (load AND scoring)


def _arm_minutes(params: dict, sel: dict) -> tuple[float, dict[str, float], dict[str, float]]:
    m = params["measured_rates_reference"]
    toks = sel["est_v1_tokens"] * float(params["readout_length_ratio"][READOUT])
    score = {arm: toks / 1000 * m["child_s_per_1k_tokens"] / 60 for arm in params.get("arms", ARM_NAMES)}
    load = {arm: float(params["evals_fixed_minutes"][arm]) for arm in params.get("arms", ARM_NAMES)}
    return toks, score, load


def estimate(params: dict, sel: dict) -> dict[str, Any]:
    """Setup preflight (exit 15 BEFORE any checkpoint is fetched). Per arm: load = ``evals_fixed_minutes[arm]`` (model
    load + 150k lineage), scoring = item tokens (the selection's committed v1-token estimate x the v3w length ratio) x
    the MEASURED child decide() wall (D13 screen, job-hjwpq; D1's own v1 rate is lower). Pass = on a box SLOW_BOX
    (1.25x) slower than measured, for every arm: load <= its ``score_setup_cap_minutes`` (the rate guard's
    --setup-cap-min, so a slow load stops with 15 before scoring starts), load + scoring <= 95 % of the phase cap, and
    the whole expected schedule (``expected_minutes`` minus build) x 1.25 fits max_duration minus reserve."""
    toks, score, load = _arm_minutes(params, sel)
    caps, scap = params["phase_caps_minutes"], params["score_setup_cap_minutes"]
    phases = {f"score_{a}": load[a] + score[a] for a in params.get("arms", ARM_NAMES)}
    slow = {k: v * SLOW_BOX for k, v in phases.items()}
    sched = sum(v for k, v in params["expected_minutes"].items() if k != "build")
    checks = {
        **{f"score_{a}_slow_le_95pct_cap": slow[f"score_{a}"] <= 0.95 * caps[f"score_{a}"] for a in params.get("arms", ARM_NAMES)},
        **{f"load_{a}_slow_le_setup_cap": load[a] * SLOW_BOX <= scap[a] for a in params.get("arms", ARM_NAMES)},
        **{f"setup_cap_{a}_leaves_scoring_room": scap[a] + score[a] * SLOW_BOX <= 0.95 * caps[f"score_{a}"]
           for a in params.get("arms", ARM_NAMES)},
        "schedule_slow_fits": sched * SLOW_BOX <= params["max_duration_minutes"] - params["reserve_minutes"],
    }
    return {"tokens_per_arm": round(toks), "minutes": {k: round(v, 1) for k, v in phases.items()},
            "minutes_slow_box": {k: round(v, 1) for k, v in slow.items()}, "slow_box_factor": SLOW_BOX,
            "expected_run_minutes": sched, "caps": {k: caps[k] for k in phases}, "setup_caps": dict(scap),
            "checks": checks, "ok": all(checks.values()),
            "usd_batch_scoring": round(sum(phases.values()) * params["batch_usd_per_min"], 2)}


def reprice(params: dict, sel: dict, rate_report: dict, timing: dict, elapsed_min: float) -> dict[str, Any]:
    """Between the arms (exit 15 BEFORE D1's checkpoint is fetched): D1's phase re-projected from D13's MEASURED load
    (``rate-score_d13.json`` scoring_started_min, scaled by the arms' fixed-minute ratio) and MEASURED scoring wall
    (``timing-d13.json``). Pass = projected D1 phase <= 95 % of its cap AND elapsed + checkpoint_d1 cap + projection +
    report cap <= max_duration minus reserve."""
    toks, score, load = _arm_minutes(params, sel)
    caps = params["phase_caps_minutes"]
    load_d13 = float(rate_report.get("scoring_started_min") or load["d13"])
    d1_load = load["d1"] * max(1.0, load_d13 / load["d13"])
    d1_score = max(float(timing["sum_row_wall_s"]) / 60, score["d1"])  # same items + readout: D13's measured wall
    proj = d1_load + d1_score
    left = params["max_duration_minutes"] - params["reserve_minutes"] - elapsed_min
    need = caps["checkpoint_d1"] + proj + caps["report"]
    out = {"d13_load_min": round(load_d13, 1), "d1_load_proj": round(d1_load, 1), "d1_score_proj": round(d1_score, 1),
           "d1_phase_proj": round(proj, 1), "cap": caps["score_d1"], "elapsed_min": elapsed_min,
           "minutes_left": round(left, 1), "minutes_needed": round(need, 1)}
    out["ok"] = proj <= 0.95 * caps["score_d1"] and need <= left
    return out


# ------------------------------------------------------------------------------------------------ CLI
def build(a) -> int:
    resp, srcs = load_source(a.src)
    dev = read_jsonl(a.dev_rows)
    fmt = check_dev_format(dev, resp, srcs)
    rep = json.loads(Path(a.guard_report).read_text(encoding="utf-8"))
    if rep.get("n_records") != sum(1 for r in resp.values() if r.get("split") == "train"):
        raise SystemExit(f"guard report scanned {rep.get('n_records')} records, not every TRAIN response")
    ex = dev_exclusions(resp, dev, rep["fail_ids"])
    sel = select(resp, srcs, set(ex["all"]), a.n, a.seed)
    items = [build_row(resp[i], srcs[str(resp[i]["source_id"])]) for i in sel["ids"]]
    items.sort(key=lambda it: _rank(a.seed, it["metadata"]["source_id"]))  # scoring order: seed order, not by stratum
    # v1-token estimate for pricing: chars x the tokens/char of the 270 DEV RAGTruth rows (fulldev_lengths_rwkv7.json)
    lens = json.loads((ROOT / "jobs/m2_first_cell/fulldev_lengths_rwkv7.json").read_text(encoding="utf-8"))["lengths"]
    dr = dev_ragtruth_rows(dev)
    tpc = sum(lens[r["_evaluation"]["run_id"]] for r in dr) / sum(len(r["state"]["prompt"]) + len(r["state"]["response"])
                                                                  for r in dr)
    est = sum(len(it["state"]["prompt"]) + len(it["state"]["response"]) for it in items) * tpc
    with gzip.GzipFile(a.out, "wb", mtime=0) as g:
        g.write("".join(json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n" for it in items).encode("utf-8"))
    sel_out = {"_comment": "ragtruth_polarity selection (CPU, local): RAGTruth TRAIN only; DEV never read for "
                           "anything but the format check and the exclusions", **sel,
               "items_sha256": _sha(Path(a.out)), "format_check": fmt,
               "exclusions": {k: v for k, v in ex.items() if k != "all"},
               "guard_report": {k: rep.get(k) for k in ("rule", "index_sha256", "allowlist_sha256", "n_records",
                                                        "n_fail", "fail_ids_sha256", "eval_rows_attributed_by_split",
                                                        "per_eval_family_fail")},
               "tokens_per_char_dev_ragtruth": round(tpc, 5), "est_v1_tokens": round(est),
               "source_files_sha256": {n: _sha(Path(a.src) / n) for n in ("response.jsonl", "source_info.jsonl")}}
    Path(a.selection).write_text(json.dumps(sel_out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in sel_out.items() if k not in ("ids", "_comment")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.ragtruth_polarity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records", help="TRAIN responses as doc_overlap_guard scan records")
    r.add_argument("--src", required=True)
    r.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--dev-rows", required=True)
    b.add_argument("--guard-report", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--selection", required=True)
    b.add_argument("--n", type=int, default=N_TARGET)
    b.add_argument("--seed", default=SEED)
    e = sub.add_parser("estimate")
    e.add_argument("--params", required=True)
    e.add_argument("--selection", required=True)
    q = sub.add_parser("reprice", help="between arms: D1 re-projected from D13's measured load + scoring")
    q.add_argument("--params", required=True)
    q.add_argument("--selection", required=True)
    q.add_argument("--rate-report", required=True)
    q.add_argument("--timing", required=True)
    q.add_argument("--elapsed-min", type=float, required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARM_NAMES, required=True)
    s.add_argument("--items", required=True)
    s.add_argument("--selection", required=True)
    s.add_argument("--cell", required=True)
    s.add_argument("--ckpt-dir", required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--records", required=True)
    s.add_argument("--train-dir", required=True)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--guard-readiness", required=True)
    s.add_argument("--max-input-tokens", type=int, default=65_536)
    s.add_argument("--resume-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--progress", default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--wording", default="original", help="decide.ragtruth_rephrase wording (original | rt1)")
    p = sub.add_parser("report")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--elapsed-min", type=float, required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "records":
        resp, srcs = load_source(a.src)
        with gzip.open(a.out, "wt", encoding="utf-8") as f:
            for rec in guard_records(resp, srcs):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return 0
    if a.cmd == "build":
        return build(a)
    if a.cmd == "estimate":
        est = estimate(json.loads(Path(a.params).read_text(encoding="utf-8")),
                       json.loads(Path(a.selection).read_text(encoding="utf-8")))
        print(json.dumps(est, indent=1))
        return 0 if est["ok"] else 15
    if a.cmd == "reprice":
        rr = json.loads(Path(a.rate_report).read_text(encoding="utf-8"))
        rp = reprice(json.loads(Path(a.params).read_text(encoding="utf-8")),
                     json.loads(Path(a.selection).read_text(encoding="utf-8")), rr,
                     json.loads(Path(a.timing).read_text(encoding="utf-8")), a.elapsed_min)
        print(json.dumps(rp, indent=1))
        return 0 if rp["ok"] else 15
    if a.cmd == "score":
        return score(a)
    res = report(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({arm: {"auroc": m["auroc"], "auroc_flipped": m["auroc_flipped"], "decision": m["decision"]}
                      for arm, m in res["arms"].items()}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
