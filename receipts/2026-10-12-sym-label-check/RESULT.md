# Result (job-6zs79 slice, criterion c783be49)
- Per the pre-stated rule: SUPPORTED on the point estimate only. SCORE n 43: trained 37.2% vs stock 48.8% (-11.6 pt,
  95% CI [-30.2, +7.0], includes 0); ORDINAL-ID choice n 0; NOMINAL n 1,157: +1.6 pt [-1.6, +4.8] (choice +4.6, noul -0.1).
- Score criteria on the slice are self-describing level lists ("zero / exactly one / ... sentences", "none (0) / one /
  two / three / four or more"): the description alone identifies the level, yet the trained head predicts only the
  extremes (long_policy argmax {0: 23, 3: 5}; temporal_numeric {0: 8, 4: 7} vs gold {1: 5, 2: 7, 4: 3}; 3/15 vs stock 7/15).
  So the loss is not the missing id string; it is the per-option fork (each level scored alone, never next to the scale).
- Routing: not on the jev3 slice and not in jev3 train (coverage); public routing ids are named, K 5-6, every description
  non-empty and > 25 chars (no id-as-content fallback). NOT TESTABLE on these preds.
