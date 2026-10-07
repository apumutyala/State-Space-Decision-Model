# RWKV 96-update attempt: partial memory-stop result

The run at `ce1dcfb7d3bb65a5e6997624b17507ede3f0df2e` stopped at **839.938s**
when available RAM fell below the 3 GiB running floor (minimum 3215503360 bytes,
exit 125). All 96 training updates and both free-running phases completed, but
only 95 of 96 final teacher-forced row results were flushed. No clean final
receipt was written. This is a **partial attempt, not a clean success**.

`partial_receipt.json` was recovered solely from frozen corpus JSON, saved
outputs and flushed logs. Recovery imported no Torch/model runtime, loaded
no checkpoint and made no model call or missing-row retry. The missing slot
is `synthetic-development_validation-9202-11 / slot-004-relation`; it occupies
its original denominator as a failed row. Complete development CE is null.

Observed outcomes:

- Useful free-running answers: 2/12 initially, 1/12 after 96 updates; step 24 also
  had 1/12. There is no observed free-running gain. The paired difference versus
  step 24 is 0pp, 95% world-bootstrap interval [−25,+25]pp.
- Kernel acceptance: 12/12→10/12, unchanged from step 24. Accepted wrong-task
  answers remain failures of usefulness.
- Final train slots: 23/64 correct, CE 1.904917, compared with 22/64 and CE 1.369076
  at step 24. Train CE worsened by 0.535842, [0.108097,0.937792].
- Final development slots: 10/32 correct with one missing row counted failed;
  31/32 rows observed. Full-split CE and its paired interval are undefined.

The saved before/after programs and all per-slot results remain in the partial
receipt. Direct-LM parity after reporting, final full-parent-state parity and
the clean final receipt remain unverified. These missing checks cannot be
inferred from the completed optimizer-step logs or checkpoint hash.

`raw.jsonl`, `teacher-before-raw.jsonl`, `teacher-after-raw.jsonl`, `before.json`,
`after.json` and `teacher-before.json` are byte-exact copies of the external
evidence. Their hashes, plus every other preserved external file, are listed
under `preserved_external_artifacts` in the partial receipt. The corpus is
unchanged from the prepared-v2 archive; its canonical/split hashes and executed
source pins are retained in the receipt.

Unique binary checkpoints remain untouched under
`C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-run1-20261006/`:

| Checkpoint | SHA256 |
|---|---|
| `checkpoint-before/ckpt-step0000000.pt` | `cfa66690ecb7ba059505f6dde50aa85b81580633d406fb76574046dac56c1d5b` |
| `checkpoint-after/ckpt-step0000096.pt` | `5b10920d1b5cb3ef07b329c71fb6653e93a22f6fa916a859b6386c9a284cece9` |

Recovery hashes these files as opaque bytes; it never deserializes them.
The parent now invokes this recovery only after an owned worker's timeout or
memory stop, preserving its stop code. The enforcing 95/96-row fixture forbids
model-runtime imports and verifies missing-row failure accounting, null CE,
identity matching and byte preservation. A LESSONS entry accompanies the fix.
The executed preparation remains immutable; this recovery fix does not
authorize a rerun or replace its source registration.

Verification: 121 focused/repository-lint tests passed in 261.65s. The launch
lint passed with its existing legacy waivers, and the diff check passed.
`verification.json` records recovery, copied-evidence and checkpoint hashes.

This remains one seed on reused synthetic development worlds and an
exploratory surrogate. It establishes no L0 capability, parent retention,
strong-comparator parity or promotion. Billed run/recovery cost was $0.
