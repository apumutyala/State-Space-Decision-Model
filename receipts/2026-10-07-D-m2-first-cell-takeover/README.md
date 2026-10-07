# First-cell seed/restart fixes (CPU only)

Base: `38ec076dbc3968a9e638d50b43f7083a75aff905` (`origin/stack/rwkv7/first-cell-fix`, PR #147).
Branch: `codex/m2fc-takeover-fixes`.
The eight-file pending diff from `ssdm-wt/m2fc-fix` was copied into this separate worktree; that original worktree was preserved.

The inherited changes add the profiled allocator, full seed identity/guard/revision checks before training, a small reproduction/speed probe, and probe-based repricing.
The takeover adds provenance inside the atomic seed-cache install, recognizes legacy caches lacking the old external marker, checks ordinary cached arms before training, and requires comparator copies even on cached restarts.
L-34 is updated in place; this is a refinement of the existing first-cell incident.

All commands used `C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider` with `CUDA_VISIBLE_DEVICES` empty and `PYTHONDONTWRITEBYTECODE=1`.
Git Bash `C:/Program Files/Git/usr/bin` was first on PATH for the final checks.

| Arguments after the command prefix | Result |
| --- | --- |
| `tests/test_m2_first_cell.py -k 'not dry_run_end_to_end and not smoke_runs_under' --tb=short` | 57 passed, 2 deselected; 16.84 s |
| `tests/test_m2_first_cell.py -k smoke_runs_under --tb=short` | 1 passed, 58 deselected; 5.50 s |
| `tests/test_m2_first_cell_shell.py -k cached_parent_block --tb=short` | 4 passed, 12 deselected; 16.14 s |
| `tests/test_jobs_lint.py --tb=short` | 70 passed; 3.11 s |

The shell regressions execute the real parent-cache/reprice section with scoring and repricing drivers stood in. They cover seeded and ordinary restarts, verification rejection before training, and byte-exact comparator preservation. Unit regressions cover interruption immediately after atomic rename, legacy marker loss, and full item/guard/revision mismatches.

An earlier broader unit invocation selected the WindowsApps WSL launcher and failed two existing rate-env tests; the final Git Bash invocation above passed them. Initial new shell-test attempts exposed PATH/shebang errors in the test stand-in, which were corrected before the passing run.
The private-data end-to-end dry run, full shell suite, and GPU qualification were not run. No job was submitted and no paid resources were used.

## Follow-up: ordinary cache with shipped source present

Seed classification now requires in-cache provenance or an exact match to the configured seed manifest hash. A source directory alone cannot classify a same-job cache as seeded. The added unit case preserves an ordinary destination with a different arm hash; two added real shell-block cases verify that it follows the full ordinary-cache verifier and stops before training on verifier rejection.

Using the same CPU command prefix and environment above:

| Arguments after the command prefix | Result |
| --- | --- |
| `tests/test_m2_first_cell.py -k 'arm_seed or seed_probe or seed_restart or shipped_seed or cache_verifier or reprice_uses_the_seed_probe or runner_verifies_a_seed' --tb=short` | 10 passed, 50 deselected; 1.64 s |
| `tests/test_m2_first_cell_shell.py -k cached_parent_block --tb=short` | 6 passed, 12 deselected; 23.83 s |
| `tests/test_jobs_lint.py --tb=short` | 70 passed; 1.84 s |
