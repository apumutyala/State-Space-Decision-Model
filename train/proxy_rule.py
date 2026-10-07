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
  Domain split (reported for every scored recipe, both splits; Index 0.3 rewards new-domain transfer): IN_DOMAIN =
           benchmarks whose own train split, or a proxy named for it in data/SOURCES.md, is in the 150k training mix;
           NEW_DOMAIN = every other DEV benchmark. Defined here from harness catalog ids (the Index-0.3 migration
           branch's family definition did not exist on origin when this was committed; if it lands, its split is
           reported ALONGSIDE this one, never instead of it).

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
#: catalog ids whose own train split (or a proxy named for it in data/SOURCES.md) is in the 150k mix:
#: BFCL (Glaive FC / ToolACE proxies), BANKING77, CLINC150, ANLI, Humicroedit, WinoGrande, HellaSwag, GSM8K,
#: MMLU-Pro (cais/mmlu aux_train proxy), When2Call
IN_DOMAIN_CATALOG_IDS = (1, 4, 5, 12, 21, 28, 29, 30, 57, 62)
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
)
#: families that may be chosen; attn_ref is a reference and can never be the verdict's head
CANDIDATE_FAMILIES = ("fork", "mixer")


def in_domain(catalog_id: int) -> bool:
    return int(catalog_id) in IN_DOMAIN_CATALOG_IDS


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
            "rule": RULE_VERSION}


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
