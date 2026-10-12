# Sym-fork label-semantics check: pre-stated criterion (written BEFORE the per-type slice numbers)

Hypothesis (lead 2026-10-12): the id-agnostic sym-fork head removes meaningful labels where option ids carry
semantics (score/ordinal levels), so relabel invariance is the wrong symmetry for those types.

Data: job-6zs79 (art-xkhwy) jev3 SLICE only (1,200 held-out decisions, ids sha 844233f2; non-public):
out/screen/jevkit/slice_floor_rows.jsonl (trained step-200 top-1 `native` vs stock host top-1 `step0`, per row)
joined to calib_probs.jsonl (type, option_ids). No public item text is used in any decision.

Row classes (fixed before looking):
- SCORE: type == "score".
- ORDINAL-ID choice: type == "choice" and every option id is an integer string, or every id is in
  {none, low, minor, medium, moderate, high, major, severe, critical, urgent, very_low, very_high}.
- NOMINAL: type == "noul", or a choice row that is not ORDINAL-ID.

Pass criterion: the hypothesis is SUPPORTED iff on the slice
  (1) SCORE and ORDINAL-ID rows (pooled; each also reported): trained top-1 <= stock top-1 - 10 pt, AND
  (2) NOMINAL rows: trained top-1 >= stock top-1 - 5 pt.
Otherwise NOT SUPPORTED. Paired bootstrap 95% CIs (B 10000, seed 16) and n reported per class; a class with
n < 30 is reported as UNDERPOWERED and cannot support the hypothesis on its own.

Routing: the jev3 slice has no routing family (families: adequacy, ambiguous, judge_hard, long_policy, policy,
temporal_numeric), so routing cannot be diagnosed on these preds; reported as NOT TESTABLE here.

Disclosure: before writing this, the lead had seen job-6zs79's per-FAMILY slice deltas (adequacy -8.0 pt, the other
five +2.0..+3.5 pt) and the noul band rates; no per-TYPE or per-id-class slice number had been computed.
