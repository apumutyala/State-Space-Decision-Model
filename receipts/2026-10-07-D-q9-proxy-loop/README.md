# Receipt: q9 proxy loop run (proxy-rule-v1 + A1, A2, A3) at agent-main d7f1290

Runner: `run_proxy_loop.sh` (worktree q9fcl-run detached at d7f12909ee71; CPU; kit = ssdm-data/decision-index).
Cache: `C:\Users\apumu\q9cache\stock_train\x\cache` (art-mtqne train + art-7x7dp select/holdout, every shard re-hashed on load).

## Stage 1: stock-check (A1(d) as amended by A3) - PASS
`run_proxy_loop.sh stock-check --dev-rows <dev-rows.jsonl.gz> --stock <job-5cmat stock results, sha 646b6149...>` -> exit 0.
Masked point 38.317, delta +0.047 (tol 0.3); unmasked delta +0.772; 65 masked (pinned sha d3465d44...; error 58 ->
contribution +0.058, unsupported 7 -> +0.667); masked rows valid; paired vs job-5cmat +0.048 (CI -0.79..+0.72).
