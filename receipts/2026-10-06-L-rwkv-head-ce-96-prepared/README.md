# 96-update head exposure variant: prepared, not run

The only scientific training change from the 24-update recipe is **96 fixed
updates**, covering the same 64 slots three times at batch size 2. Parent,
initialization, seed, loss, optimizer, architecture, examples and sampler stay
the same. The [preregistered plan](../../plans/codex/L0_RWKV_HEAD_CE_96.md)
defines outcomes and failure handling. Base is
`bb2eadd798b5d5ae317524767b37115e9a8df168`; branch is `stack/rwkv7/head-ce-96`.

`registration.json` freezes the full recipe, source closure, plan, original
corpus/split hashes and committed step-0/24 measurement references.
`corpus.json.gz` retains the exact dry-run inputs. Earlier preparations and
measured receipts are preserved. This preparation reads only source and
committed JSON; no real checkpoint or snapshot is accessed.

The runtime fails closed if any trainer setting other than `max_steps` differs
from the previous recipe. The actual-sampler regression proves the unchanged
first 24 batches and three complete passes with no dropped item or tail.
These tests use tiny random CPU RWKV and sampler calls; they do not train the
cached parent.

Final teacher-forced train/dev accuracy and CE, free-running usefulness and
kernel acceptance are paired against step 0 and step 24. The final-only
reporter constructs no optimizer, loads/selects no earlier checkpoint, keeps
all failures, and bootstraps worlds with their slots together. Teacher-forced
step-0 fit is measured on the exactly reproduced initial head; step-24 rows
come from the committed independent audit. No real result exists for step 96.

The inherited offline CPU, two-thread, 8 GiB start/3 GiB running RAM floors,
authenticated worker, fresh-output and 900-second watchdog remain in force.
Twelve reused development worlds and one seed cannot qualify L0, establish
convergence or prove adapted-head decision retention.

Completed dry-run command, from the variant checkout:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-prepared-20261006
```

Result: `prepared_not_run`. Exact source/data/config and 90 passing test
receipts are in `verification.json`; no real parent load or fit occurred.

Proposed command, **not executed or authorized by this preparation**:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-run1-20261006
```

The coordinator will independently review the committed branch before
authorizing execution. The preparation owner opens no PR.
