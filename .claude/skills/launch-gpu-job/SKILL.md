---
name: launch-gpu-job
description: Package and submit an SF Compute H100 job for this repo (RWKV7 cells, readout rescoring, comparators) in minutes. Use whenever a variant is ready to measure.
---

# Launch a GPU job

1. **Commit and push** the variant on its branch (`stack/rwkv7/<variant>`, `stack/clef/<x>`). Any pushed sha can launch; merging is not required.
2. **Package** (CPU only; about 2 min; the test gate is skipped unless you pass `--tests`):
   `CUDA_VISIBLE_DEVICES="" uv run --no-project --with zstandard python -X utf8 -m jobs.common.launch <job> [--mode M] --sha <sha> --out C:\Users\apumu\<outdir> --data C:\Users\apumu\research\CHLabs\ssdm-data`
   - A new variant needs `jobs/<job>/job_params_<mode>.json` plus its config. Copy the nearest existing mode, then change what differs.
   - Pinned data that lives outside ssdm-data is found through `src_env` variables (for example `SSDM_KD_*` for the teacher files in `C:\Users\apumu\td\art\clef-flash-teacher\teacher`).
3. **Submit** from an agent-main worktree (`jobs/common/sf_submit.py` exists only on agent-main checkouts, not on every variant branch): run `python -I jobs/common/sf_submit.py C:\Users\apumu\<outdir> --dry` and check the args, then run it again without `--dry`. It prints `JOB <id> ... max_cost`.
   - If the cost-cap preflight blocks, raise the workspace cap within the $525 envelope (`autor api` on the workspace cap), then resubmit with the same idempotency key.
4. **Claim and record:** push `claims/<id>.lock` on the `state` branch (owner, job id, start UTC, and `time_limit` as an absolute UTC timestamp, not a duration). Add the job to the BOARD item.
5. **While it runs:** package the next variant. Don't wait. Poll at most every 10–20 minutes: `job get <id> --json` (status), `job logs <id>` (progress). From Git Bash, prefix `autor` calls with `MSYS_NO_PATHCONV=1` (e.g. `MSYS_NO_PATHCONV=1 wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job ls`), or MSYS rewrites the `/root/...` path.
6. **If it crashes:** read `job logs <id>` and the artifact's `logs/`, fix the cause, push, relaunch. After two failed fixes, log a `crash` row and move on.

**Rates:** the platform quotes $0.063/min per H100 ($3.78/h; this sets `max_cost`) but bills about $0.0495/min ($2.97/h). Budget with the billed rate, cap with the quoted one. A seeded first cell (parent arm reused) is about 5 h, about $15 expected and $31.5 max. An Index-subset rescore is about 30 min, about $1.5.
