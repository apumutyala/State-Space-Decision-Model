Prepared locally for independent exact-package review; no node, provider call, upload, staged model suite, submit artifact or training. Source is merged `agent-main@11fb140ce64802222d8abe796267e22e14fe4959`. The previous single-attempt authorization was consumed; `spend-proposal.md` requests a fresh acknowledgment for one attempt with zero retries and the conservative $2.525580 cumulative exposure under a proposed $3 cap. The separate renewed spend proposal is [Issue #188](https://github.com/apumutyala/State-Space-Decision-Model/issues/188); it remains pending acknowledgment. Issue #171 is historical and grants no further attempt. Wrapper changes must also receive independent review and merged proof before paid execution.

The exact 9-file upload package (498,078,038 bytes) remains external at `C:/Users/apumu/research/CHLabs/ssdm-data/prepared-contexts/m2-cpu8-validation-11fb140-20261008/`. `package-summary.json`, `package.sha256` and the byte-exact manifest/runner provide every hash and size. The large context and bundles are not committed. Additional local summary/validation files in that directory are excluded from the controller's compiled 9-file allowlist.

The old `m2_first_cell-4247b340ddde.tar.zst` is safe to reuse as a data carrier: the runner consumes only its 11 allowlisted data members, and fresh source comes from the new complete-history bundle. Offline bundle reconstruction verified `11fb140`, the reviewed archive/PID test fix and equality of all seven required source pin paths. Streaming the context verified the exact original data and context-manifest hashes. Kit, deadline and tokenizer inputs are byte-exact copies, and no data files were regenerated. The runner has precisely five pin substitutions for source/origin/bundle/output/report. It will invoke the normal launcher on `11fb140`, which runs the full staged suite and creates the new source context only after success.

Local checks: 17 fake-AUTOR/static tests; 71 launch-lint tests; shell syntax; the controller's actual package hash gate; all context-data hashes; offline bundle/source verification. `local-validation.json` records details. The package gate's free-byte input was synthetic and grants no live storage or spend authorization. Local cost $0. Exact local validation in WSL `bi-ubuntu`:

```sh
cd /mnt/c/Users/apumu/research/CHLabs/ssdm-wt/m2-cpu8-package-refresh
python3 -m unittest discover -s tests -p test_m2_cpu8_controller.py -v
GIT_DIR=/mnt/c/Users/apumu/research/CHLabs/State-Space-Decision-Model/.git/worktrees/m2-cpu8-package-refresh GIT_WORK_TREE="$PWD" /root/ssdm-data/.venv/bin/python -m pytest -q tests/test_jobs_lint.py
PYTHONPATH=/root/.cache/uv/archive-v0/btWEoG0sPJF4bYYF /root/ssdm-data/.venv/bin/python receipts/2026-10-08-D-cpu8-package-11fb140/validate_package.py \
  /mnt/c/Users/apumu/research/CHLabs/ssdm-data/prepared-contexts/m2-cpu8-validation-11fb140-20261008 \
  /mnt/c/Users/apumu/research/CHLabs/ssdm-data/prepared-contexts/m2-cpu8-validation-20261007 \
  /mnt/c/Users/apumu/research/CHLabs/ssdm-wt/m2-cpu8-package-refresh
```

The existing cached zstandard wheel supplies decompression without installing/downloading dependencies; no model library is imported. The validator uses temporary offline Git repositories, never source credentials or provider APIs. `bash -n` checks the prepared runner without executing it.
