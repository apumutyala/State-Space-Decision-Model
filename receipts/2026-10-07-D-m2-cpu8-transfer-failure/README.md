Issue #171's first CPU-8 attempt failed on the first direct node upload: the 26,208,694-byte bundle announced seven parts of up to 4 MiB, then `error: request failed`. The node was stopped after 37 seconds, disk intact; the immediate available-balance decrease was $0.004958. Subsequent `autor usage --json` reconciliation reports the settled node charge as $0.004800; see `receipts/2026-10-08-D-cpu8-wake-sequence-failure/usage-reconciliation.json`. No file completed and no test ran. The generic error supports a suspected chunk-upload failure, not a proven file-size limit.

`incident.json`, the original byte-exact `package-manifest.json` and `package.sha256` preserve the source, context/data, runner, helper, original supervisor and package hashes. The original supervisor failed before its terminal receipt, so its numeric return code and absolute balances are unknown in the local evidence. The frozen test target remains `4247b340dddecf620de6ff22751256f4127237cb`; this orchestration correction starts at `agent-main@a6b0755`.

The canonical local WSL entry point is `scripts/supervise_m2_cpu8.sh`. After fresh exact-package review, an Issue #171 update and live balance/expiry/commitments/workspace-cap/free-storage preflight, its invocation is:

```sh
CPU8_STORAGE_FREE_BYTES=<live-remaining-free-bytes> \
  bash scripts/supervise_m2_cpu8.sh <frozen-package> <fresh-output> <workspace> <existing-private-bucket> ubuntu-24.04
```

This is documentation, not a launch authorization or an executed command. This PR created no bucket and made no provider calls/uploads. The prior live account evidence showed no buckets, zero storage bytes and a 100GB/month free tier; that is not future remaining-quota proof. A private bucket must be explicitly created and verified empty in a separate later reviewed preflight, and the live remaining free allowance supplied to the controller. Every AUTOR command carries the explicit workspace. The controller generates its own unpredictable node name and exclusively owned prefix, uploads only the nine pinned files one at a time, records each successful object's key/size/hash, then creates CPU-8 and checks `yours`, chip, running status and integer locked rate ≤12,000 USD micros/min. Import plus remote `sha256sum -c package.sha256` establish transferred content integrity; upload acknowledgments alone do not. Import failure/timeout has unknown remote outcome: the controller stops the owned node and never hashes/runs/wakes it. The local import timeout is not a remote cancellation or a server deadline. Storage/import/export stdout, stderr and exit codes are retained in the local receipt with bearer/key/password material redacted; environment variables are never dumped.

The frozen 145m on-wake command, 8610s subreaper with 30s KILL grace, 20m setup, 175m total controller, 135s cleanup reserve and zero workload retries remain. Normal operations retain 10s KILL grace; stop has a 55s ceiling plus 5s KILL grace and at most one cleanup retry. The first attempt's $0.004958 provisional balance decrease later reconciled to $0.004800 usage charge. The combined CPU-8 charge is now reconciled to $0.006400 across both attempts. A failed stop records unknown node state and nonzero status; it is not reported as stopped. Outputs export to the owned prefix, the node stops, then storage downloads the export.

The controller retains storage objects for evidence; it never removes remote data. Preserve `storage-prefix.txt`, `staging.json`, `controller.json` and exported output until review. Any later cleanup must target only the recorded exclusively owned prefix after verified export/retention review. Never delete the shared bucket or another prefix.

Local no-model proof (stdlib only, fake AUTOR subprocesses; no Index/heldout access):

```sh
wsl -d bi-ubuntu -- bash -lc 'cd /mnt/c/Users/apumu/research/CHLabs/ssdm-wt/m2-cpu8-transfer-recovery && python3 -m unittest discover -s tests -p test_m2_cpu8_controller.py -v'
```
