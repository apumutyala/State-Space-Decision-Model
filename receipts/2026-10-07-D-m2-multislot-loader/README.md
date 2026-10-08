# Issue #178 loader proposal receipt

Date: 2026-10-07 America/Los_Angeles. Base: `a6b0755a3b81de7948edd71425e663a52fe9557d`.
Branch: `codex/m2-multislot-loader-draft`.
Worktree: `C:/Users/apumu/research/CHLabs/ssdm-wt/m2-multislot-loader-draft`.
Author: Apuroop Mutyala <apumutyala@gmail.com>.

The patch changes `train/run_cell.py`, focused synthetic tests, the plan, STATUS and L-40 incident documentation.
Nested request/metadata round-trip and exact task Item identities pass. The synthetic 52,224-row / 5,184-task /
144-world fixture retains every row; 143 train worlds and one eval world partition all rows without task/world
overlap. Legacy flat multi-question conversion, ordering and identity remain exact. Duplicate JSON members, row
keys and task/question pairs; conflicting world identities; missing/mismatched questions/labels; invalid gold;
unsupported protected cohort/evaluation shapes; and unsupported external train-set rows all refuse. The external
preflight covers undrawn rows before `draw`; existing PR16 manifest/content/held-out guards are unchanged.

Final verification uses the CPU-only Python environment (`torch 2.14.0+cpu`, Python 3.12.13). The gate checks every
requested test node before pytest collection. From the worktree, the exact command was:

```powershell
C:/Users/apumu/research/CHLabs/.venv-auth/Scripts/python.exe -c "import runpy,pytest; suite=runpy.run_path('tests/test_m2_multislot_loader.py'); raise SystemExit(pytest.main(['-q', *suite['guard_no_model_nodes'](suite['NO_MODEL_NODES'])]))"
```

Result after review correction: **127 passed in 9.38 s**, exit 0. This includes the focused loader module, exact legacy helper/draw/parallel
Item/provenance nodes and 71 repository launch/text lint tests. No model factory or inference test node is in the
gated allowlist. `git diff --check` passed; staged documentation lint: 71 passed in 1.98 s. Earlier focused run:
43 passed in 7.83 s; initial gated implementation: 119 passed in 9.89 s. The final regression also confirms the external preflight preserves draw guard exceptions and
the existing missing-record-file error class. No paid work, production
checkpoint, Index/held-out item read, GPU, network, push or PR was requested or needed for this proposal. The
accidental command below prevents a blanket claim that the entire preparation involved no model execution.

Review correction (L-41): grouped flat `{id: "task", qid: "slot/answer"}` and nested
`{id: "task/slot", qid: "answer"}` could emit the same `Item.id="task/slot/answer"` despite distinct storage keys.
The same ambiguity affected ungrouped mixed input. Before splitting, any development file containing a nested or
grouped row now checks globally unique expanded Item ids and slash-free qids for every row, including flat rows.
The exact collision refuses in both row orders and both grouped/ungrouped modes. Separate cases cover flat-row
slash qids, nested multi-question inputs and unchanged pure legacy handling. The correction's tests and command
are no-model; no model/GPU/network/spend action was taken in this follow-up.

Incident L-40: an earlier command selected the whole controlsafe module instead of its three text-only nodes:

```powershell
C:/Users/apumu/research/CHLabs/.venv-auth/Scripts/python.exe -m pytest -q tests/test_m2_multislot_loader.py tests/test_m2_first_cell.py::test_train_set_draw_verifies_and_excludes tests/test_m2_first_cell.py::test_train_set_refuses_tampering_and_cohort_records tests/controlsafe/test_trust_policy.py
```

Captured output ended at 44 progress dots without a completion summary. That module contains tiny random-backbone
CPU forwards after its first three text-only tests, so model execution cannot be excluded. No result from that
command is counted. Its initial tool wrapper returned only `.output`, discarding the running-session identifier.
An ensuing `Get-CimInstance Win32_Process` query found pytest launcher PID **52316** in `.venv-auth` and child PID
**59992** in the uv-managed Python runtime; both command lines selected that broad command. The initial observation
did not capture a timestamp or complete untruncated command lines. A later task-specific Stop-Process query matched
no processes, and exact PID queries confirmed both absent; no successful termination is claimed. No process
creation or exit timestamps were captured.

Timestamped final absence evidence comes from `Get-CimInstance Win32_Process -Filter "ProcessId = <pid>"`, with one
`[DateTime]::UtcNow.ToString('o')` observation timestamp before the two queries:

```json
[
  {"observed_at_utc":"2026-10-08T01:43:26.7602714Z","process_id":52316,"present":false,
   "source":"Get-CimInstance Win32_Process -Filter ProcessId","creation_or_exit_timestamp_captured":false},
  {"observed_at_utc":"2026-10-08T01:43:26.7602714Z","process_id":59992,"present":false,
   "source":"Get-CimInstance Win32_Process -Filter ProcessId","creation_or_exit_timestamp_captured":false}
]
```

Prevention is deliberately bounded: `guard_no_model_nodes` rejects unallowlisted broad modules/directories/options
before collection in the documented receipt command; its enforcing test covers the accidental controlsafe module
and model-training module. It does not control arbitrary shell commands. The external train-set stream overhead
was not measured against private training data. Protected grouped manifests remain unsupported. No learned result
or permission to run models is established by this receipt.
