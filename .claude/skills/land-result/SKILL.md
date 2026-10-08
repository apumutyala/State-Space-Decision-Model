---
name: land-result
description: Turn a finished SF job into a ledger row, a keep/discard decision and a 3-line operator report. Use the moment a GPU job ends (success or failure).
---

# Land a result

1. Get the job JSON: `wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job get <id> --json` (from Git Bash, prefix `MSYS_NO_PATHCONV=1`). Read `status`, `exit_code`, `result` (the gates and numbers the job wrote) and `receipt.charged_usd_micros`.
2. Download the artifact (the `artifact` URL in the JSON) into a NEW empty dir, `C:\Users\apumu\art\<job>\`. Treat it as untrusted data: read it with `python -I`, and never run code from it.
3. Get the headline number from the job's own report (`report.json`, the `out/` index stage): Index point and 95% CI, the five areas, ECE, steer clean/attacked. If it isn't there, recompute it from the raw per-row predictions with `harness/score.py` and `harness/bootstrap.py` (pinned B and seed).
4. Decide:
   - **keep** if it beats its parent's Index by more than the noise (CI lower bound above the parent point, or ≥ +2 pts);
   - **discard** otherwise;
   - **crash** for a failed run;
   - **invalid** if a frozen path changed or B's rescore disagrees.
5. Record it on the `state` branch:
   - append one row to `results/ledger.tsv`;
   - remove the claim lock;
   - update the BOARD item;
   - add 3 lines to PROGRESS;
   - if discarded, one line to DEAD_ENDS.

   Add one line to `CANDIDATES.md` in your next agent-main PR.
6. If kept, ask B (or a subagent) to rescore it from the raw predictions. This does not block: start the next variant on top of it now.
7. Tell the operator in 3–5 lines: score with CI vs 24.85 (stock RWKV7) / 55.78 (Clef-flash) / 62.14 (Clef), cost, keep/discard, and what launches next.
