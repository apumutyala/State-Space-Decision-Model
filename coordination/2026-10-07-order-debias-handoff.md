# Order-debias takeover

PR #151 remains open and unchanged at `478aec24f19656243ec8e347bf1563acb188c5b8`. Its negative calibration numbers independently reproduce exactly. No reversed probability pass was run; no scoring flag exists.

See `receipts/2026-10-07-D-order-debias-handoff/README.md` and `reconciliation.json` for the CPU-only audit, actual session-log provenance, complete input hashes and durable raw-input locations. Follow-up branch: `codex/order-debias-handoff-20261007`, base `ee0adb70029fc80c65ee6072c307873034052362`. Cost $0; no model calls, local GPU use, paid jobs, merges or force pushes.

Corrections needed before treating PR #151 as a final handoff:

- Replace "no position effect" with the observed near-zero/negative net endpoint excess; binary mapped choices still change on 56/387 questions.
- For k>2, endpoint excess alone does not isolate position bias. The high-k sample shows order/relabel sensitivity; it does not establish a pure position cause.
- The 1,531-second stage includes 320 extra diagnostic requests. Forward request wall time totals 1,316 seconds; median/mean/p80 are 270/655/604 ms, not the stage-time-per-row estimate.
- One new reversed/relabelled pass makes a mixed-ID ensemble with the original forward. Matched relabelled forward/reverse averaging requires both new full passes and saved probabilities.
- State the binary failure/tie convention and that harness `board_protocol` ECE is not board-comparable.

Next decision: accept the scoped negative calibration result, and separately decide whether to prepare a matched two-pass measurement under the existing launch/spend rules. The README pins the design to use if that work is chosen. No compute was launched or authorized by this receipt audit.
