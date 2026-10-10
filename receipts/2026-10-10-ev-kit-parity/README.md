# KIT PARITY (EV-PREREG-KIT-PARITY): job-r3uc4, 2026-10-10

**Verdict: kit parity DONE (3 rows).** H2O-Lightning-4B (anchor, job-zbtbj) + decider-12b + decider-12b-v1 pass.
torchcast-decision-12b FAILS the pre-registered count rule (gap attributed below).

The board's I/C cells use the v1.6.1 open draw P300 + 1,200 sealed items, which are not downloadable, so each row is
judged against its author's published public-231 result for the same pinned weights, run end to end through jevkit with
the system's own readout (rule: all 231 answered and top-1 count within 3).

| row (pin) | readout | ours (easy/std/hard) | published | diff | verdict |
|---|---|---|---|---|---|
| h2o-lightning-4b @193ad740 (job-zbtbj) | vendored shim | 205 (48/71/86) | 205 (48/71/86) | 0 | PASS |
| decider-12b v2 @8ac1efa7 | decider-ai 1.8.0 system_one | 197 (48/71/78) | 198 (48/71/79) | -1 | PASS |
| decider-12b-v1 @09e9e338 | same | 200 (48/71/81) | 200 (48/71/81) | 0 | PASS (exact per tier) |
| torchcast-decision-12b @49107b55, primary arm double_bos | vendored shim | 198 (46/68/84), 226/231 answered | 203 (48/70/85) | -5 | FAIL |

jevkit open-only axes (231 items, not board cells): decider-12b I 70.8 [62.1, 80.8], C 77.9, Composite A 71.0;
decider-12b-v1 I 72.5 [64.1, 79.8], C 80.9, A 72.4; torchcast (double_bos) I 76.4 [67.9, 83.6], C 82.6, A 73.1.

**torchcast gap, attributed.** The primary arm (double <bos>) reproduces the author's input exactly: per-item prompt
token count equal on 226/226 answered items (the plain arm is 1 short on all). On the answered items, top-1 correctness
agrees with the author's per-item results on 224/231 and the median TVD is 0.0015. The whole -5 comes from the 5 items
where the shim found no option letter among the top-20 RAW log-probabilities ("could not recover an option-letter
distribution", scored wrong; 4 of them are right in the author's run). The author's vLLM run never hit this, so its
top_logprobs were evidently taken AFTER the structured-output letter mask (processed logprobs), not raw as our
replacement assumed. Fix for a future run: restrict the top-k to the allowed letter tokens before the shim's
letter_probs. Under the pre-registration this row stays FAIL (no re-run with altered readout settings counts).
Cross-check: jevkit scoring the author's own per-item outputs gives I 77.8 [69.5, 84.8], C 80.5; ours 76.4 / 82.6.

Files: decision.json (in-job summary, reference_checks), per-arm preds/report, run_meta, torchcast_compare.json
(analysis/kit_parity_compare.py), the author's results.jsonl (github Torchcast-AI/torchcast-decision-12b@d7f84468
runs/h100-release, sha256 fa2e0ca8...) and its jevkit conversion and report. Artifact art-t4psd (sha256 3c47d833...).
Cost $0.55 (527 billed s).
