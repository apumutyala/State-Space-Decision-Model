# intent16-v1: recover intent supervision within the existing row budget

Unmeasured hypothesis variant from `ea6bfc4`. No job parameters, launch, PR or
spend authorization are supplied. The original M2 set, config and running cell
are unchanged.

The original seed-0 24k draw has 1,143 BANKING77 and 1,480 CLINC150 question rows;
all have K > 16 and are dropped before training. This variant replaces only the
single-question intent records with 16-option training views. It reserves one
of every eight consumed rows for these views, alternating BANKING77 and CLINC.
The other seven rows use the original baseline mixture. Source permutations
cycle only after exhaustion; a checkpoint's global row cursor resumes exactly.

Construction keeps the original gold option and its description, the original
injection target, and CLINC's OOS option on both positive and negative examples.
It selects seven additional negatives by token-set Jaccard similarity to the
utterance, breaking ties by a seeded SHA256; remaining slots use a separate
SHA256 ordering. A label-blind SHA256 permutation controls displayed option
order. The renderer's canonical option axis and all literal IDs remain intact.
Source IDs, original record/line/content hashes, untrusted fields, state,
instructions and injection metadata are preserved in a training sidecar.
No sidecar or gold label enters the model prompt.

Mixed/multi rows keep their payloads and remain baseline rows. The artifact is
24,000 records: 21,422 baseline and 2,578 intent views (1,127 BANKING77 and 1,451
CLINC), including 288 injected intent views. Files are outside Git under
`ssdm-data/m2/build/sample-24000-seed0-intent16-v1/`; the committed pointer pins
the records, manifest and content map. The original 150k directory must be its
sibling, including in any future staged context.

Both the CPU builder and derived loader check original and derived records
against cohort IDs, normalized content, original source items, key-free text,
the pinned SplitGuard/content index, frozen families and held-out D1-B text.
The loader re-verifies the original pointer/config/files/draw and reconstructs
every derived row before returning training records. Changing the source
artifact, labels, option subset/order, lineage or derived files fails closed.
KD and order-consistency variants are refused because their changed payloads
have no matching teacher probabilities.

The new config is `m2_rwkv7_intent16.json`. It keeps rank64, the Clef supervised
loss, learning rates, warmup, batch 1 x accumulation 8, K<=16, T ladder
512/1024/2048, row batching, continuation filtering and the 190-minute cap.
Long views are filtered rather than truncated. Both intent pools must survive
the measured K/T filters; otherwise training refuses the schedule. The memory
ladder still measures worst rows rather than applying the mixture schedule.

The historical 11,056-row projection would allocate approximately 691 rows
to each intent source. K16 raises fork/continuation work, so that row count and
the historical 8.25-second step rate are not verified for this variant. Main
risks are shortlist-to-full-domain transfer and displacement of knowledge/tools
updates; the 1/8 allocation bounds displacement. A future candidate requires a
new preregistered comparison on unchanged, full-option evaluation requests and
all applicable gates. No improvement or qualification claim is made here.

CPU rebuild (use a fresh sibling output directory; an existing output is never
overwritten, and an existing committed pointer must match the rebuild):

```powershell
python -X utf8 -m train.intent16 --parent-dir <build>/sample-150000-seed20261006 --out <build>/intent16-rebuild
```

Focused checks: `python -X utf8 -m pytest -q tests/test_intent16.py` (10 passed),
plus six existing checks for default layout/spans and legacy checkpoint resume.
Aggregate construction evidence is in `receipts/2026-10-07-D-intent16-data/`.
