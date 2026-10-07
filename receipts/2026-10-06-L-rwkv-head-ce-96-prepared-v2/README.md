# 96-update preparation v2: operational watchdog correction

Prepared, not run. This replaces the 900-second preparation at `a0003c7`;
the earlier receipt remains unchanged in `../2026-10-06-L-rwkv-head-ce-96-prepared/`.

The only amendment to that prepared recipe is the operational watchdog:
**900 to 1200 seconds**. The scientific TrainConfig delta versus the measured
24-update recipe remains solely `max_steps: 24 to 96`. No training step,
evaluation pass, architecture, data, optimizer, seed or sampling change is added.

Preserved CPU timestamps project **1013.67 seconds**: 278.14 for the original
run + 404.13 for 72 additional updates + 331.40 for two teacher-forced scoring
passes. The new ceiling leaves **186.33 seconds (18.38%)** headroom. The
registration recomputes this estimate and freezes both raw timing-log hashes.
The [preregistered plan](../../plans/codex/L0_RWKV_HEAD_CE_96.md) records the
timing spans and the distinction between scoring and load/setup time.

All 96 fixed updates, three complete data passes, final fit/free-running
reporting, failure denominators and paired intervals remain. Full trainer
parity, source/data pins, offline CPU mode, two threads, RAM floors, fresh
output and authenticated child-only cleanup are unchanged.

**91 tests passed in 11.19s**, including registration drift, unchanged sampler
and full trainer config, exact 1200-second stop boundary, timeout/RAM cleanup,
optimizer-free post-evaluation and no model access during preparation.
`verification.json` records exact source/data/config hashes and the completed
dry-run. No real checkpoint access or fit occurred.

Completed preparation command, from the branch checkout:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-prepared-v2-20261006
```

Proposed guarded command, still awaiting independent review and approval:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-run1-20261006
```

No PR was opened. This remains one exploratory exposure variant on reused
synthetic development worlds, with no learned capability or promotion claim.
