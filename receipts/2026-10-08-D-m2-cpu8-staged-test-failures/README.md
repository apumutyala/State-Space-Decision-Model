Issue #171's first completed CPU-8 staged suite failed: 152 tests, 144 passed, 4 failed, 4 skipped. The frozen target was `4247b340dddecf620de6ff22751256f4127237cb`; the local correction starts at `agent-main@cce4387aa60daa17d53b8214816197e40c301509`. The controller correctly retained command `cmd-zk3da`, recorded workload exit 1, exported before stopping the owned node, and downloaded the failure evidence. There was no submit.json or H100 request. The suite exercised the tiny random CPU model; no campaign training or pretrained-model run occurred.

The actual metered node cost was $0.156400. Its provisional available-balance decrease was $0.179180. Including the two previous failed orchestration attempts ($0.006400), cumulative metered CPU-8 charges are $0.162800, with $0.022780 currently unsettled extra. These amounts were supplied by the coordinator's account reconciliation; this correction made no provider/billing calls and incurred $0. Future reconciliation must retain the distinction between usage and the provisional balance delta.

The external 92,211,200-byte archive remains at `C:/Users/apumu/research/CHLabs/ssdm-data/receipts/m2-cpu8-validation-20261008-run2/launch-output.tar`, SHA-256 `cae3106eefb9ef98ecbb0415ec63622c53cd31fd2c196f11cc8a20df4431e1e8`. Only its 45,026-byte JUnit is committed here, byte-exact SHA-256 `30b58207ec4fbd37a24ffed9f0c19b69fdf01a6cf9c3f91f7cddc9a1222ec87b`. Small original command/controller/staging/log files and the pinned package manifest/runner are also retained. No source/data/checkpoint archive is committed. `incident.json` records hashes, counts and the four exact failures.

Three seed tests invoked `git ls-files` from `launch-output/stage/4247b340ddde`, which is intentionally a source archive without `.git`. The fixture now retains tracked/unignored Git selection in a checkout and uses archive files otherwise, excluding Python/pytest caches. An enclosing repository cannot supply a different file list. Synthetic archive and checkout regressions enforce copying both runner and original test source, changing only the seed parameter in the copied parameters, and preserving original bytes.

The fourth failure was finalizer PID 2115 responding to `kill -0` after SIGKILL. The original receipt contains no process-state snapshot: it does not prove whether that PID was live, zombie or reused. A prior local WSL reproduction passed. Linux zombies remain visible to `kill -0` while awaiting reaping, so the corrected fixture records /proc start ticks inside each owned child before TERM, then checks identity and state. Explicit states separate verified live owner, gone/zombie, reused and unknown. The same live process still fails; zombies and different start identities cannot execute as that owned child and are excluded from cleanup. Missing/uninspectable identity fails conservatively if kill -0 still succeeds, but never authorizes a signal. Cleanup signals only a verified same live PID/starttime. Other Bash platforms retain the conservative kill -0 exit-proof fallback if /proc is unavailable, without granting cleanup permission. Regressions exercise a real unreaped zombie, a live owner, simulated PID reuse, a reaped PID and unknown live state. A signal-attempt regression requires unknown/reused/zombie/missing identities to send none and verified same-live ownership to send one. Independent review caught the initial boolean conflation of unknown liveness with cleanup permission; this follow-up corrects it. Runner timeout/finalizer behavior and all source assertions remain intact.

The checkout's old LF test also selected byte-exact historical receipt checksum lists, whose original CRLF is intentionally preserved with `-text`. Its selection now covers `jobs/*.sha256`; the required operational manifests and their `eol: lf`/byte checks remain enforced.

Exact historical entry point (record only; not rerun by this correction): the controller invoked the byte-exact `run_m2_cpu8_tests.sh` retained here under `run_deadline.py` with 8610 seconds and 30-second KILL grace. That runner reconstructed the pinned SSDM_DATA and kit Git identity, then called:

```sh
python -m jobs.common.launch m2_first_cell \
  --sha 4247b340dddecf620de6ff22751256f4127237cb \
  --out /home/dev/m2-cpu8-validation/work/launch-output \
  --data /home/dev/m2-cpu8-validation/work/SSDM_DATA \
  --repo /home/dev/m2-cpu8-validation/work/repo --no-fetch
```

The precise suite reproduction, after restoring that runner's layout and stage, is:

```sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export SSDM_DATA=/home/dev/m2-cpu8-validation/work/SSDM_DATA
export SSDM_DI_KIT="$SSDM_DATA/decision-index"
export HF_HUB_CACHE=/home/dev/m2-cpu8-validation/work/hf-cache
cd /home/dev/m2-cpu8-validation/work/launch-output/stage/4247b340ddde
uv run --no-project --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --with transformers==5.17.0 \
  --with pytest --with torch --with numpy --with pyarrow --with zstandard \
  --with "$SSDM_DI_KIT" --with-editable "$PWD/harness" \
  python -X utf8 -m pytest -q -rs -p no:cacheprovider \
  --junitxml=../junit-m2_first_cell.xml \
  tests/test_jobs_lint.py tests/test_m2_first_cell.py tests/test_m2_first_cell_shell.py
```

This command is documentation of the original full suite, not permission for a provider retry. Full staged qualification has not been rerun for the correction. Local verification uses synthetic CPU fixtures only, no model/data/provider calls, no submit.json and no training. In WSL `bi-ubuntu`, from this worktree:

```sh
export GIT_DIR=/mnt/c/Users/apumu/research/CHLabs/State-Space-Decision-Model/.git/worktrees/m2-test-archive-fix
export GIT_WORK_TREE=/mnt/c/Users/apumu/research/CHLabs/ssdm-wt/m2-test-archive-fix
/root/ssdm-data/.venv/bin/python -m pytest -q tests/test_m2_first_cell_shell.py \
  -k 'owned_process or term_resistant or seeded_copy or cached_parent or long_phase or overrides or sha256'
/root/ssdm-data/.venv/bin/python -m pytest -q tests/test_jobs_lint.py
```

The explicit Git overrides are needed only to read the Windows-created worktree from WSL; temporary regression repositories remove those overrides. Local result: 16 focused tests and 71 launch-lint tests passed. Post-commit source-archive verification is recorded in `validation.json`.
