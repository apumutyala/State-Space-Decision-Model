[Issue #188](https://github.com/apumutyala/State-Space-Decision-Model/issues/188), new spend proposal following the consumed [Issue #171](https://github.com/apumutyala/State-Space-Decision-Model/issues/171) attempt: exactly one CPU-8 staged M2 validation attempt, with zero retries, on merged `agent-main@11fb140ce64802222d8abe796267e22e14fe4959` (PR #186 serving metadata and PR #187 staged archive/PID corrections). Issue #171's previous single-attempt authorization is consumed; this is a separate request for one additional attempt. Historical incidents and settled/provisional usage remain recorded in #171. This proposal does not authorize a node, upload or workload; it requires a new acknowledgment, independent exact-package/controller review, merged controller proof and fresh live spend/storage/rate/ownership preflight.

The local prepared package is `ssdm-data/prepared-contexts/m2-cpu8-validation-11fb140-20261008/`. Its exact upload allowlist contains **9 files / 498,078,038 bytes**:

| File | Bytes | SHA-256 |
|---|---:|---|
| agent-main-11fb140.bundle | 26,346,294 | f5ca0afa4380a66ee40598591a9f2c0efd3f508d87b5d8d418f90c9cd4348d55 |
| decision-index-87d4650.bundle | 1,044,541 | 46a9e0ecfff272cc8ffb4d39c0bbe79a320a1bbd455f6ee06b2b20065d73263a |
| m2_first_cell-4247b340ddde.tar.zst | 469,087,698 | 6902a600f541c518db35cd37d80cfa04968441ff2672ad2b98fc3d6a07acc27b |
| package-manifest.json | 6,377 | 4cd83d308309efda0cca767f503b70614ed8dd2a8e945739298d7565da9bb2fc |
| run_deadline.py | 5,723 | ac4b06f61e22c9cb5f4de0e8d61180d502ff31bd96c3277a200b4970ce153702 |
| run_m2_cpu8_tests.sh | 5,186 | b0a46f5ab20ec5634b4fed1b4d9dc6666d991060c21256aeb59f2ab87130754b |
| tokenizer_config.json | 531 | 2b5e075c2c16780061ae272c9f0b52ff620ddd8e87fdf7fa82fc425661667329 |
| tokenizer.json | 1,580,973 | 1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf |
| package.sha256 | 715 | 410dbca12ff3aecf3ecdf613deab023ec470d0e26cd1d010e2afea39d49ffa14 |

The source bundle contains complete history and pins `refs/remotes/origin/agent-main` to `11fb140ce64802222d8abe796267e22e14fe4959`; local reconstruction in a fresh offline repository verified that exact source and the merged test corrections. The original context is intentionally retained as a **data carrier**. The runner copies only its `data/` entries into SSDM_DATA and never executes its old `code/m1`. Seven required params/cohort/data/guard/snapshot pin paths are byte-identical between `4247b340` and `11fb140`. The context's 11 data members and tokenizer/kit/deadline files retain their original exact pins; sealed Index TEST, held-out D1B and pretrained weights remain excluded. Input `CONTEXT_MANIFEST.json` remains SHA-256 `228f46e2e705ee81e8ccc63cbd776f688c6acff440675de88c89fab5d20572aa`. The actual standard launcher will build the new `m2_first_cell-11fb140ce648.tar.zst` from the merged source after the full staged suite passes; its future output hash is not known or claimed now.

The runner changes only five source/bundle/output pins. The original deadline helper, timeout behavior, data layout and exclusions remain unchanged. The controller's compiled manifest/checksum/file/source pins are refreshed to this package, including exact-source post-JUnit completion. Its receipt records the conservative cumulative cost basis below. Local checks passed: 17 fake-AUTOR/static controller tests, 71 launch-lint tests, runner `bash -n`, all nine input hashes/sizes, streamed hashes for all 11 context data members, original context-manifest identity and offline source-bundle reconstruction. The capacity value used to exercise the local package gate was synthetic; it is not live storage evidence. No model tests, provider calls, upload, node or training ran during preparation; local cost $0.

Requested incremental worst-case compute exposure is **(180 active + 15 idle) minutes × $0.012/min = $2.340000**. Prior CPU-8 metered usage is **$0.162800**, with **$0.022780** extra unsettled balance decrease retained conservatively pending reconciliation. Cumulative projected exposure is **$2.525580**, leaving **$0.474420** under the proposed cumulative **$3** hard cap. The settled-only basis is $2.502800. Storage is budgeted at $0 only if a fresh live remaining-free-tier check covers the exact package plus expected export; no current allowance or provider quote is inferred from historical receipts. Locked rate must be at most $0.012/min; otherwise stop before import/workload. Refresh balance, expiry, unsettled commitments, October MTD, workspace cap, node/job ownership and storage before any execution.

The hard stop is unchanged: setup/upload capped at 20 minutes; remote Python subreaper 8610 seconds plus 30-second KILL grace; exact-command log-follow cap 8700 seconds; controller wall 175 minutes with 135-second stop reserve; bounded export up to 10 minutes only while the owned node is running; at most two bounded stop attempts; provider idle fallback 15 minutes. Direct detached execution on the running node replaces on-wake registration; the obsolete on-wake-only 145-minute server limit is not claimed. No workload or upload retry is authorized. Any transfer/hash/rate/registration/completion failure ends the attempt, exports available evidence when possible and stops the exclusively owned node. Full suite failure remains failure; CLI rc0 alone cannot claim success. The controller never submits the generated H100 request.

Reviewable future workload, **not executed**:

```sh
cd /home/dev/m2-cpu8-validation/input
exec python3 ./run_deadline.py --seconds 8610 --kill-grace-seconds 30 -- bash ./run_m2_cpu8_tests.sh
```

The exact local controller invocation, after all gates and a fresh independent authorization, uses the reviewed merged controller, this immutable package, a fresh output path and fresh owned prefix in the existing verified private bucket. `CPU8_STORAGE_FREE_BYTES` must come from the new live storage preflight, never the synthetic local-validation value. The coordinator will retain that invocation and live evidence before launch. No further attempt is authorized by the unused portion of the $3 ceiling alone.

Controller dependency: `codex/d-m2-cpu8-package-refresh` updates the compiled manifest/checksum/bundle/completion-source pins. Its exact local commit will be added here after the new spend issue number is bound into its receipt. The controller PR must be independently approved and merged before execution, followed by a rerun of exact-package validation from that merged controller commit. The workload source stays frozen at `11fb140ce64802222d8abe796267e22e14fe4959`.

Scope is CPU validation only: no GPU, pretrained weights, campaign training or H100 submission. The normal staged suite uses its existing tiny random CPU fixtures. The controller receipt binds this proposed attempt to Issue #188, separate from the consumed Issue #171 authorization.
