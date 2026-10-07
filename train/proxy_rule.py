"""PRE-REGISTERED decision rule for the frozen-Q9 proxy head loop (PREMORTEM.md blocker #1, S8).

Committed ALONE, before any feature cache exists (no GPU data has been produced or seen when this file was written).
Constants are literal and pinned by ``tests/test_proxy_rule.py``. Any change after the first cached feature exists is a
dated amendment committed BEFORE the run it applies to (LESSONS L-07); it never edits these values in place.

Question. Can an ATTENTION-FREE head on frozen Qwen3.5-9B prefix states (stock, and with 1 attention layer converted
to GDN) approach Clef-flash (55.8 on the 2,008-row DEV subset; stock Q9 readout 38.3) BEFORE we pay for the full-8
conversion (S1, $28-74) and the first Q9-AF cell (S2)?

Splits (DEV only; TEST is never read):
  SELECT   the pre-registered 2,008-row m1 DEV subset (``jobs/common/dev_subset_m1.json``, ids sha 6e4cb8df...). Every
           recipe is scored here and the recipe is CHOSEN here. Because it is used for selection, its score of the
           chosen recipe is optimistically biased (selection over many recipes; winner's curse ~ 1-2 points at the
           subset's SE ~1.6) and is REPORTED ONLY.
  HOLDOUT  a second, disjoint 2,000-row DEV split (``jobs/q9_feature_cache/dev_holdout.json``), built by the same
           stratified procedure (``jobs.common.dev_subset.build_subset``) over the DEV manifest rows NOT in SELECT
           (whole kit groups, so no group straddles the two splits), seed ``HOLDOUT_SEED``. Scored ONCE, for the one
           recipe chosen on SELECT (and its pre-declared seeds). The verdict reads HOLDOUT.
  Domain split (reported for every scored recipe, both splits; Index 0.3 rewards new-domain transfer): since
           amendment A1, the merged harness/families.py families-v1 groups (see AMENDMENT A1 below); v1's hand-defined
           catalog-id list is withdrawn before any data.

Scoring: the real harness / kit scorer (``jobs.common.dev_subset score|compare``: chance-corrected per-benchmark
skills, gold weights, unanswered or errored rows score as wrong, no truncation of the row set). CIs are the harness
cluster bootstrap (B = ``BOOTSTRAP_B``, seed ``BOOTSTRAP_SEED``); paired comparisons are on identical rows.
Comparators: Clef-flash per-row results (job-sgihj, full DEV, so they cover both splits) and the stock Q9 readout
(job-5cmat on SELECT; on HOLDOUT the same readout recomputed from the cached option log-likelihoods, validated against
job-5cmat on SELECT first).

Head families (``train/proxy_heads.py``): (a) the m2 fork MLP head (``heads.heads.FinalStateFork``), (b) attention-free
cross-field / option mixers, (c) an ATTENTION routing head that is an UPPER-BOUND REFERENCE ONLY: it is never a
candidate, never selected and never the verdict's head. "Attention-free" (as in PREMORTEM S2b): no softmax- or
otherwise-normalised weighting over sequence POSITIONS (or over chunk summaries of positions). Mixing over the fixed
set of option / field vectors of one request is allowed and disclosed.

Recipe order (anchor-then-ablate, techniques/10 §10): R00 the Clef-like ANCHOR (family (c) on stock features, clef
loss) first, then the full attention-free mixer R01, then one-factor ablations from R01, then the fork baselines.
``RECIPES`` lists them; each recipe is trained with the same steps budget and seed list, no early stopping on DEV.

Verdict (``verdict``), read from HOLDOUT for the recipe chosen on SELECT:
  go           chosen attention-free recipe's HOLDOUT index point >= GO_THRESHOLD
  escalate     below it: report the measured head ceiling to the user BEFORE any full-8 spend (S1)
  no-measurement  the chosen recipe is not attention-free, a split is incomplete, or scoring failed
The SELECT number, the selection gap (SELECT - HOLDOUT), paired CIs vs Clef-flash and vs stock Q9 on both splits, and
the IN/NEW domain breakdown are always reported; none of them changes the verdict.

AMENDMENT A1 (2026-10-07, before any feature cache exists; no GPU job has run). Requested by the coordinator after
the independent review of PR #101, quoted: "Pre-data amendment A1 [...]: (a) Replace the hand-defined
IN_DOMAIN_CATALOG_IDS with agent-main's harness/families.py families-v1 split. Report three columns: in-domain,
new-domain, and new_domain_frozen (the 9 frozen benchmarks). (b) Equalise optimiser steps across arms, or else state
that conv1/e5 comparisons use a matched-steps stock arm. (c) Pin the verdict to edition 0.2.1 and report 0.3
alongside. (d) Declare the post-job stock-check tolerance now: |delta| <= 0.3 against job-5cmat's 38.27 on SELECT.
(e) State that "attention-free" refers to the head; the stock features come from a backbone with attention. Keep the
go threshold at 50.0 on HOLDOUT, with escalation to the user, unchanged." What A1 changes:
  (a) Domain columns = families-v1 (``FAMILIES_V1_SHA256``) groups evaluated in the verdict edition:
      ``in_domain`` (same_source + same_family tiers), ``new_domain`` (= families ``new_domain_proxy``, the 'none'
      tier) and ``new_domain_frozen`` (heldout-draw-v1's 9 frozen benchmarks). Pinned as literals below (so the rule
      imports without the kit); tests/test_proxy_rule.py checks them against harness.families when the kit is present.
  (b) Every recipe trains exactly ``TRAIN_STEPS`` optimiser steps of ``GRAD_ACCUM`` requests (not epochs), on every
      arm. The arms still differ in data: stock has 20,000 training rows, conv1 / e5 the first 5,000 rows of the same
      draw. So conv1 / e5 recipes (R09, R10, R13) are compared ONLY against the matched-data, matched-steps stock
      controls added here, R14 (= R01) and R15 (= R11) trained on the stock arm's first ``MATCHED_TRAIN_ROWS`` rows.
      R14 / R15 are attention-free candidates like any other.
  (c) The verdict, SELECT choice and every threshold use edition ``VERDICT_EDITION`` = 0.2.1 (GO = 50.0 was set
      against Clef-flash 55.78 on 0.2.1). Every score is also reported under 0.3's public index
      (``REPORT_EDITIONS``), scored without the 0.3 GSM8K twin rows (the cache holds 0.2.1 DEV rows only).
  (d) Before any recipe is trained, ``stock-check`` recomputes the stock readout from the cached option
      log-likelihoods on SELECT; |that - job-5cmat's 38.27| must be <= ``STOCK_CHECK_TOL`` (0.3 Index points), else
      the whole loop is no-measurement (the features do not reproduce the measured readout).
  (e) "Attention-free" refers to the HEAD. The stock features come from Qwen3.5-9B with its 8 softmax-attention layers
      (conv1 replaces 1 of 8). A "go" bounds what an attention-free HEAD can reach on these features; it is not a
      measurement of Q9-AF. The verdict record carries ``scope`` saying so, and the conv1 - stock gap is reported next
      to it.
  Unchanged: GO_THRESHOLD 50.0 on HOLDOUT; below it the verdict is "escalate" to the user; SELECT / HOLDOUT; the
  transfer rule.

AMENDMENT A2 (2026-10-07, before any feature cache exists: the stock cache jobs job-kxc5e (stock_dev) and job-wuw7g
(stock_train) were still BUILDING their images when this was committed; no proxy recipe has been trained or scored and
no SELECT receipt exists anywhere). Adds knowledge distillation from Cloudflare Clef-flash (Apache-2.0 open weights;
PLAN.md allows open-weight teachers; recipe per techniques/10) as two candidate recipes:
  R16 = R01 + KD(alpha 0.5, T 1)        R17 = R01 + KD(alpha 0.9, T 2)
  (a) Loss (``train/losses.py``): per training question WITH a teacher row,
          alpha * T^2 * KL( teacher_T || student_T ) + (1 - alpha) * clef arm (LS-CE 0.1 + Brier + RPS on score qs),
      where x_T renormalises p^(1/T) over the question's VALID options (Hinton scaling T^2 keeps gradient size
      comparable across T); a question WITHOUT a teacher row (exhausted record, or not in the teacher file) takes the
      clef arm only. Nothing else about R01 changes (family, arm, budget, seeds).
  (b) Teacher data, pinned (``TEACHER``): job-tnrwu (clef_baseline teacher_clef_flash @5b478e3, mission
      clef-baseline-safety-spec-s7), artifact art-ggrs8; teacher-clef-flash.jsonl sha256 ``TEACHER["sha256"]``, 24,000
      rows = the m2 first cell's seed-0 TRAINING draw (drawn ids sha256 ``TEACHER["drawn_ids_sha256"]``, the same draw
      the stock_train cache keeps its 20,000 rows from), 23,458 ok / 542 exhausted (> 16,384 tokens, no
      probabilities). ``train/teacher.py`` refuses any other file, identity, draw, or a row whose record content hash
      differs, and never maps a DEV / TEST / cohort row.
  (c) Selection: R16 and R17 are attention-free candidates like any other, so SELECT must score R01-R17 before
      ``choose``; HOLDOUT stays ONE shot for the one recipe chosen (unchanged). Disclosure: 17 candidates instead of 15
      enlarges the SELECT winner's curse slightly; the verdict reads HOLDOUT, so it is unaffected.
      ``KD_CONTROL``: R16 and R17 are compared against R01 (same arm, same data, same budget) and that paired diff is
      reported next to them.
  (d) Validity: A2 holds only because no proxy SELECT data exists at its commit. Every receipt records the amendments
      it was trained under and ``select`` refuses receipts whose amendments differ from ``AMENDMENTS`` (no pre-A2
      receipt can be mixed in). Had SELECT already run, KD would need a separately pre-registered rule version
      (proxy-rule-v2) with a fresh HOLDOUT, never this one's.
  Unchanged: GO_THRESHOLD 50.0 on HOLDOUT; below it the verdict is "escalate" to the user; splits, editions, budget,
  seeds, the transfer rule.

AMENDMENT A3 (2026-10-07, written AFTER the stock-check failed with delta=+0.772; see
receipts/2026-10-07-D-q9-proxy-stock-check; before any recipe, select or holdout data). Changes ONLY the A1(d)
stock-check, a validity check of the cache (not the go/no-go rule).
  Disclosure (decomposition recomputed by the independent review of PR #130; every variant rescored with the kit).
  job-5cmat did not answer 65 of the 2,008 SELECT rows, which the kit scores as wrong; the cache answers all 65:
    * 7 rows "unsupported" (3 ToolRet + 4 BRIGHT): longest model input 17,126-28,497 tokens > the reference's
      max_input_tokens 16,384 (no truncation); the cache's cap is 32,768. An INPUT-LIMIT POLICY difference, not a
      bug. They carry about +0.67 of the gap (masking only them: delta +0.105).
    * 58 POP909 rows "error" (backend_error), believed to be the GPU OOM root-caused in PR #102, whose fix is NOT yet
      confirmed on GPU (L-21 stays partial). They carry about +0.06 (masking only them: delta +0.714, which would
      still FAIL).
    * all 65 masked: delta +0.047 (paired vs job-5cmat +0.048, CI -0.79 .. +0.72); unmasked: +0.772.
  Masking ONLY the bug rows ("error") was a plausible alternative, and it fails. "unsupported" rows are masked as well
  because the reference has no answer on them: comparing an answer to a non-answer measures the serving protocol (the
  input cap), not feature fidelity, which is what A1(d) tests. The masking criterion is the reference's OWN row status,
  never the cache's score. The cache's answers on the 65 rows are near chance level, like the rows around them, not
  favourable (review: POP909 2/58; BRIGHT 3.1% = the ok rows' rate).
  Not validated by this check: rows whose model input is 16,384-32,768 tokens (the reference cannot answer them, so no
  stock-check can); the HOLDOUT report states how many such rows there are and how each comparator scored them.
  No recipe has been trained or scored, and no select_summary or HOLDOUT score exists.
  (a) Rule: the stock-check compares the cached readout to the reference ONLY on rows the reference answered with
      status "ok". A row whose reference status is anything else (error / unsupported / ...) is MASKED from both
      sides: the cached results carry the reference's own row for it (``stock_check_mask``), so it contributes
      identically to the cached index and to the reference's 38.27, and the delta measures only the rows the
      reference answered. The tolerance is unchanged: |delta| <= STOCK_CHECK_TOL (0.3).
  (b) Reference pinned: the reference is job-5cmat's stock results file, sha256 ``STOCK_REFERENCE_SHA256``; it must
      hold exactly the SELECT ids (no missing, extra or duplicate rows). The masked ids are pinned too
      (``STOCK_CHECK_MASKED_N`` = 65, ``STOCK_CHECK_MASKED_IDS_SHA256``); any other masked set fails closed.
  (c) Reference failure cap: the check FAILS CLOSED if more than ``STOCK_CHECK_MAX_MASKED_FRAC`` (5%) of the SELECT
      rows are masked.
  (d) Masked rows must still be valid cache answers: every masked id's CACHED row must have status "ok" and every
      answer must be well-formed (choice: finite, non-negative probabilities summing to 1 within
      ``STOCK_CHECK_PROB_SUM_TOL`` with the choice among them; noul: finite in [0, 1]); else the check fails closed
      (``masked_row_problems``), so masking can never hide a cache crash / NaN / unanswerable row.
  (e) Reporting: stock_check.json carries the masked delta (decisive), the unmasked delta, the masked-row count,
      fraction, ids and their sha256, the per-reference-status counts and each status group's contribution
      (unmasked point - point with only that group masked), any masked-row problems, and both paired comparisons.
  Report-only addition (no rule change): every receipt lists its rows whose cached model input exceeds
  ``REFERENCE_MAX_INPUT_TOKENS`` (16,384), and the HOLDOUT verdict reports how many there are and how the chosen head,
  the stock readout and Clef-flash scored them (index with and without them, statuses on them).
  Unchanged: GO_THRESHOLD 50.0 on HOLDOUT; below it the verdict is "escalate" to the user; STOCK_CHECK_TOL 0.3;
  STOCK_Q9_SELECT 38.27; splits, editions, budget, seeds, recipes, teacher, the transfer rule. A3 is recorded in
  ``AMENDMENTS`` (receipts and select_summary carry it).

Proxy -> real transfer (``adopt_ranking``): the proxy ranking is adopted for the S2 cell recipe only if, over the
recipes that have BOTH a proxy score and a real (head + LoRA) cell score, there are >= MIN_TRANSFER_RECIPES of them,
Kendall tau-a >= TAU_MIN, and on the one real-cell calibration pair (the proxy's best and worst recipe) the real cells
order the pair the same way the proxy does. Otherwise the proxy ranking is a prior only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

RULE_VERSION = "proxy-rule-v1"
#: verdict threshold on the HOLDOUT index point (Index points, 0-100) of the chosen attention-free recipe
GO_THRESHOLD = 50.0
#: transfer rule
TAU_MIN = 0.6
MIN_TRANSFER_RECIPES = 4
#: comparators (reported, not thresholds)
CLEF_FLASH_SELECT = 55.78
STOCK_Q9_SELECT = 38.27
#: splits
SELECT_SUBSET = "jobs/common/dev_subset_m1.json"
SELECT_IDS_SHA256 = "6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3"
HOLDOUT_SUBSET = "jobs/q9_feature_cache/dev_holdout.json"
HOLDOUT_SEED = 20261008
HOLDOUT_N = 2000
#: harness bootstrap
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
#: seeds: every recipe on SELECT with seed 0; the chosen recipe on HOLDOUT with seeds 0, 1, 2 (verdict = seed 0,
#: seeds 1-2 reported as spread)
SELECT_SEEDS = (0,)
HOLDOUT_SEEDS = (0, 1, 2)
#: --- amendment A1 ---
AMENDMENTS = ("A1", "A2", "A3")
#: (a) families-v1 groups in the 0.2.1 DEV edition (harness/families.py members(group, "0.2.1") at families-v1.json
#: sha256 FAMILIES_V1_SHA256 and heldout-draw-v1)
FAMILIES_V1_SHA256 = "e0e4bd5f76ca928f2a05c3f52150592aed462b4018350346eb723e9b40096f46"
DOMAIN_COLUMNS = ("in_domain", "new_domain", "new_domain_frozen")
IN_DOMAIN = (1, 2, 3, 4, 5, 11, 12, 21, 25, 28, 29, 30, 36, 37, 39, 42, 45, 57, 62)
NEW_DOMAIN = (9, 20, 22, 23, 31, 32, 33, 38, 40, 41, 43, 44, 48, 50, 56, 58, 59, 61, 64)
NEW_DOMAIN_FROZEN = (9, 20, 22, 31, 32, 33, 44, 58, 64)
#: (b) optimiser steps per recipe (every arm) and the matched-data control size
TRAIN_STEPS = 2000
GRAD_ACCUM = 8
MATCHED_TRAIN_ROWS = 5000
#: (c) editions
VERDICT_EDITION = "0.2.1"
REPORT_EDITIONS = ("0.2.1", "0.3")
#: (d) post-job stock-check tolerance (Index points, SELECT, vs job-5cmat 38.27)
STOCK_CHECK_TOL = 0.3
#: (e) the verdict's scope statement
VERDICT_SCOPE = ("attention-free HEAD on features of Qwen3.5-9B WITH its 8 attention layers (stock), or with 1 of 8 "
                 "converted (conv1): bounds the head ceiling on these features; not a Q9-AF measurement")
#: the anchor and the one-factor ablations, in run order. family: fork | mixer | attn_ref. Fields not named take the
#: family default (train/proxy_heads.py). "arm" = feature-cache arm. "loss": ce | clef (LS-CE 0.1 + Brier + RPS on
#: score questions) | clef_norps (LS-CE 0.1 + Brier).
RECIPES = (
    {"id": "R00", "family": "attn_ref", "arm": "stock", "loss": "clef", "note": "Clef-like ANCHOR (reference only)"},
    {"id": "R01", "family": "mixer", "arm": "stock", "loss": "clef", "note": "full attention-free mixer"},
    {"id": "R02", "family": "mixer", "arm": "stock", "loss": "clef", "no_lm": True, "note": "R01 - LM logprob (E4)"},
    {"id": "R03", "family": "mixer", "arm": "stock", "loss": "clef", "no_keyclass": True, "note": "R01 - key class"},
    {"id": "R04", "family": "mixer", "arm": "stock", "loss": "clef", "no_route": True, "note": "R01 - evidence routing"},
    {"id": "R05", "family": "mixer", "arm": "stock", "loss": "clef", "no_fields": True, "note": "R01 - field mixer"},
    {"id": "R06", "family": "mixer", "arm": "stock", "loss": "clef", "no_lex": True, "note": "R01 - lexical prior"},
    {"id": "R07", "family": "mixer", "arm": "stock", "loss": "ce", "note": "R01 with CE"},
    {"id": "R08", "family": "mixer", "arm": "stock", "loss": "clef_norps", "note": "R01 with LS-CE+Brier, no RPS"},
    {"id": "R09", "family": "mixer", "arm": "e5", "loss": "clef", "note": "R01 on E5 short-code features"},
    {"id": "R10", "family": "mixer", "arm": "conv1", "loss": "clef", "note": "R01 on 1-layer-converted features"},
    {"id": "R11", "family": "fork", "arm": "stock", "loss": "clef", "note": "m2 fork MLP head (baseline)"},
    {"id": "R12", "family": "fork", "arm": "stock", "loss": "ce", "note": "m2 fork MLP head, CE"},
    {"id": "R13", "family": "fork", "arm": "conv1", "loss": "clef", "note": "fork head on converted features"},
    # A1(b): matched-data, matched-steps stock controls for the 5,000-row arms (R09, R10 vs R14; R13 vs R15)
    {"id": "R14", "family": "mixer", "arm": "stock", "loss": "clef", "train_rows": 5000,
     "note": "R01 on the stock arm's first 5,000 rows (control for R09 / R10)"},
    {"id": "R15", "family": "fork", "arm": "stock", "loss": "clef", "train_rows": 5000,
     "note": "R11 on the stock arm's first 5,000 rows (control for R13)"},
    # A2: knowledge distillation from Clef-flash (teacher TEACHER) on top of R01
    {"id": "R16", "family": "mixer", "arm": "stock", "loss": "clef", "kd": {"alpha": 0.5, "temperature": 1.0},
     "note": "R01 + KD from Clef-flash (alpha 0.5, T 1)"},
    {"id": "R17", "family": "mixer", "arm": "stock", "loss": "clef", "kd": {"alpha": 0.9, "temperature": 2.0},
     "note": "R01 + KD from Clef-flash (alpha 0.9, T 2)"},
)
#: --- amendment A2 ---
#: the pinned teacher (job-tnrwu, artifact art-ggrs8): file sha256, identity fields, and the training draw it covers
TEACHER = {
    "job": "job-tnrwu",
    "artifact": "art-ggrs8",
    "release": "clef-flash",
    "repo": "Cloudflare/clef-flash",
    "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
    "backend_revision": "clef:Cloudflare/clef-flash@17f0b0ad64ef;jsm=0e304cf7c650;enc=untrusted-split-special+cut-leaking-added-v1;cfg=05bdad56;adapter=0.2.0",
    "sha256": "a88b4fe0f6a79a8fc48bdea51391524bd4f5da885705461a543becf81c81e7f7",
    "drawn_ids_sha256": "2bbbf1808c0bdd6cf0ba3ecd236cf9a64110c5e429e22f8b8e9cd2df7507556b",
    "rows": 24000,
    "status_counts": {"ok": 23458, "exhausted": 542},
}
#: --- amendment A3 (stock-check masking; see the module docstring) ---
#: the stock-check reference: job-5cmat's stock results file (SELECT, 2,008 rows)
STOCK_REFERENCE_SHA256 = "646b61490a37b3aa8887a57bb1a6797f8254c0136a63a0c3cf18d00f90497beb"
#: a reference row counts (is not masked) only with this status
STOCK_CHECK_REFERENCE_OK = "ok"
#: fail closed above this fraction of masked SELECT rows
STOCK_CHECK_MAX_MASKED_FRAC = 0.05
#: the masked set itself, pinned (job-5cmat: 58 POP909 "error" + 7 ToolRet/BRIGHT "unsupported");
#: sha256 of the sorted ids joined by "\n" (``masked_ids_sha256``)
STOCK_CHECK_MASKED_N = 65
STOCK_CHECK_MASKED_IDS_SHA256 = "d3465d44c3073b35685cab7bf5e827dfef3d85b36d33004767dc47c115466629"
#: a masked row's cached choice probabilities must sum to 1 within this (wire answers are rounded to 4 decimals)
STOCK_CHECK_PROB_SUM_TOL = 0.01
#: report-only: the reference's model-input cap (job-5cmat max_input_tokens, no truncation); the cache allows 32,768
REFERENCE_MAX_INPUT_TOKENS = 16384
#: KD recipes and the recipe each is compared against (same arm, data and budget)
KD_RECIPES = ("R16", "R17")
KD_CONTROL = "R01"
#: recipes that must ALL have a SELECT score before ``choose`` may pick one (single-shot HOLDOUT; A1 / review B2)
def candidate_ids(recipes=None) -> tuple:
    return tuple(r["id"] for r in (recipes or RECIPES) if r["family"] in ("fork", "mixer"))

#: families that may be chosen; attn_ref is a reference and can never be the verdict's head
CANDIDATE_FAMILIES = ("fork", "mixer")


def domain_of(catalog_id: int) -> tuple[str, ...]:
    """A1(a): the families-v1 columns a benchmark counts in (new_domain_frozen is a subset of new_domain)."""
    n = int(catalog_id)
    out = ("in_domain",) if n in IN_DOMAIN else ("new_domain",) if n in NEW_DOMAIN else ()
    return out + (("new_domain_frozen",) if n in NEW_DOMAIN_FROZEN else ())


def masked_ids_sha256(ids: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def stock_check_mask(cached_rows: Sequence[Mapping], reference_rows: Sequence[Mapping], ids: Sequence[str],
                     statuses: Sequence[str] | None = None) -> tuple[list, list[str]]:
    """A3(a, b): (cached rows with every reference-failed row replaced by the reference's own row, masked ids).
    Both row lists must hold exactly ``ids`` once each; a row is masked iff its reference status != "ok" (and, when
    ``statuses`` is given, only rows with one of those reference statuses: the per-group contribution report)."""
    want = set(ids)
    for name, rows in (("cached", cached_rows), ("reference", reference_rows)):
        got = [r["run_id"] for r in rows]
        if len(got) != len(set(got)) or set(got) != want:
            raise ValueError(f"A3: {name} rows do not hold exactly the {len(want)} SELECT ids once each")
    ref = {r["run_id"]: r for r in reference_rows}
    masked = sorted(k for k, r in ref.items() if r.get("status") != STOCK_CHECK_REFERENCE_OK
                    and (statuses is None or r.get("status") in statuses))
    m = set(masked)
    return [dict(ref[r["run_id"]]) if r["run_id"] in m else dict(r) for r in cached_rows], masked


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def masked_row_problems(cached_rows: Sequence[Mapping], masked_ids: Sequence[str]) -> list[str]:
    """A3(d): why a masked id's CACHED row is not a valid answer (empty list = all valid). Fail closed on anything
    unrecognised."""
    by = {r["run_id"]: r for r in cached_rows}
    out = []
    for rid in sorted(masked_ids):
        r = by.get(rid)
        if r is None or r.get("status") != "ok":
            out.append(f"{rid}: cached status {None if r is None else r.get('status')!r} != 'ok'")
            continue
        answers = (r.get("response") or {}).get("answers")
        if not isinstance(answers, Mapping) or not answers:
            out.append(f"{rid}: no answers")
            continue
        for qid, a in answers.items():
            t = a.get("type") if isinstance(a, Mapping) else None
            if t == "choice":
                p = a.get("probabilities")
                if (not isinstance(p, Mapping) or not p or not all(_finite(v) and v >= 0 for v in p.values())
                        or abs(sum(p.values()) - 1.0) > STOCK_CHECK_PROB_SUM_TOL or a.get("choice") not in p):
                    out.append(f"{rid}/{qid}: invalid choice probabilities")
            elif t == "noul":
                v = a.get("noul")
                if not (_finite(v) and 0.0 <= v <= 1.0):
                    out.append(f"{rid}/{qid}: invalid noul {v!r}")
            else:
                out.append(f"{rid}/{qid}: unrecognised answer type {t!r}")
    return out


def stock_check(point: float, *, unmasked_point: float, masked_ids: Sequence[str], n_rows: int,
                masked_problems: Sequence[str], by_status: Mapping[str, Mapping]) -> dict:
    """A1(d) as amended by A3: ``point`` = the cache-recomputed stock readout on SELECT with the reference's failed rows
    masked (``stock_check_mask``); it must reproduce job-5cmat's 38.27 within STOCK_CHECK_TOL. Fails closed if more
    than STOCK_CHECK_MAX_MASKED_FRAC of the ``n_rows`` SELECT rows are masked, if the masked set is not the pinned one,
    or if any masked row's cached answer is invalid (``masked_problems``). ``by_status``: reference status ->
    {"n", "point_only_this_masked"}; each group's contribution (unmasked - that point) is added."""
    if int(n_rows) <= 0:
        raise ValueError("A3: n_rows must be the SELECT row count")
    d, du = float(point) - STOCK_Q9_SELECT, float(unmasked_point) - STOCK_Q9_SELECT
    frac = len(masked_ids) / int(n_rows)
    sha = masked_ids_sha256(masked_ids)
    within, cap_ok = abs(d) <= STOCK_CHECK_TOL, frac <= STOCK_CHECK_MAX_MASKED_FRAC
    pinned = len(masked_ids) == STOCK_CHECK_MASKED_N and sha == STOCK_CHECK_MASKED_IDS_SHA256
    valid = not masked_problems
    groups = {st: {"n": int(g["n"]), "point_only_this_masked": float(g["point_only_this_masked"]),
                   "contribution": float(unmasked_point) - float(g["point_only_this_masked"])}
              for st, g in sorted(by_status.items())}
    return {"point": float(point), "reference": STOCK_Q9_SELECT, "delta": d, "tol": STOCK_CHECK_TOL,
            "unmasked_point": float(unmasked_point), "unmasked_delta": du,
            "masked": {"n": len(masked_ids), "n_rows": int(n_rows), "frac": frac,
                       "max_frac": STOCK_CHECK_MAX_MASKED_FRAC, "ids_sha256": sha, "ids": sorted(masked_ids)},
            "by_reference_status": groups, "masked_problems": list(masked_problems)[:50],
            "within_tol": within, "mask_cap_ok": cap_ok, "masked_ids_pinned_ok": pinned, "masked_rows_valid": valid,
            "amendment": "A3", "ok": within and cap_ok and pinned and valid}


def choose(select_scores: Mapping[str, float], recipes: Sequence[Mapping] = RECIPES) -> str | None:
    """The recipe chosen on SELECT: highest SELECT index point among attention-free candidate recipes (ties -> the
    earlier recipe id). ``None`` if no candidate has a score."""
    fam = {r["id"]: r["family"] for r in recipes}
    cands = [(-float(s), rid) for rid, s in select_scores.items() if fam.get(rid) in CANDIDATE_FAMILIES and s is not None]
    return min(cands)[1] if cands else None


def verdict(chosen: str | None, holdout_point: float | None, recipes: Sequence[Mapping] = RECIPES,
            holdout_complete: bool = True) -> dict:
    fam = {r["id"]: r["family"] for r in recipes}
    if chosen is None or fam.get(chosen) not in CANDIDATE_FAMILIES:
        return {"verdict": "no-measurement", "reason": "no attention-free candidate chosen", "rule": RULE_VERSION}
    if holdout_point is None or not holdout_complete:
        return {"verdict": "no-measurement", "reason": "HOLDOUT score missing or incomplete", "rule": RULE_VERSION}
    v = "go" if float(holdout_point) >= GO_THRESHOLD else "escalate"
    return {"verdict": v, "chosen": chosen, "holdout_point": float(holdout_point), "threshold": GO_THRESHOLD,
            "rule": RULE_VERSION, "amendments": list(AMENDMENTS), "edition": VERDICT_EDITION, "scope": VERDICT_SCOPE,
            "escalate_to": None if v == "go" else "user"}


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Tau-a (ties count as neither concordant nor discordant); identical to train.proxy.kendall_tau."""
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (a[i] - a[j]) * (b[i] - b[j])
            s += 1 if d > 0 else -1 if d < 0 else 0
    return 2 * s / (n * (n - 1)) if n > 1 else 0.0


def adopt_ranking(proxy: Mapping[str, float], real: Mapping[str, float], pair: tuple[str, str]) -> dict:
    """``proxy`` / ``real``: recipe id -> score (higher is better). ``pair`` = (proxy-best id, proxy-worst id) that ran
    as the real-cell calibration pair."""
    both = sorted(set(proxy) & set(real))
    tau = kendall_tau([proxy[i] for i in both], [real[i] for i in both]) if len(both) > 1 else None
    best, worst = pair
    pair_ok = (best in real and worst in real and best in proxy and worst in proxy
               and (proxy[best] - proxy[worst]) * (real[best] - real[worst]) > 0)
    enough = len(both) >= MIN_TRANSFER_RECIPES
    adopt = bool(enough and tau is not None and tau >= TAU_MIN and pair_ok)
    return {"adopt": adopt, "n_recipes": len(both), "tau": tau, "tau_min": TAU_MIN, "pair_agrees": pair_ok,
            "enough_recipes": enough, "rule": RULE_VERSION}
