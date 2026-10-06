# Anonymous DEV receipt packet (for Track L re-measurement without item access)

`packet.json.gz` (127 KB) lets a reviewer recompute the m0 DEV aggregates (every per-benchmark raw and skill, the 0.2.1 index, and paired bootstrap CIs and MDE) **without opening any DEV item, gold label or manifest**.

## Contents
- **Per index benchmark (38):**
  - `spec`: the plain-data value-function spec read by `harness.score.value_fn` (column layout, number of label/key blocks, public chance level and track baseline);
  - `n_clusters`;
  - per system, the source-case × column sufficient-statistics matrix as sparse triplets.
- **Per system (8 fixtures; same seeds as `../reproduce.py`):** the official kit scorer's per-benchmark raw/skill and `balanced_skill` on DEV (`official_reference`).
- `weights`: the published 0.2.1 benchmark weights.

## What is not in it
Item text, options, instructions, gold labels, row/run/group/case ids, label or key names, and any ordering.
- **Shuffling:** clusters are shuffled and per-label / per-key column blocks permuted, per benchmark, with a seed from `os.urandom` that was never recorded. One permutation is shared by all systems, which keeps pairing valid.
- **Audit:** `audit_packet.py` scans the packet against every DEV row. 0 of 43,697 identifiers and 0 of 16,903 option/instruction strings occur in it (`audit.json`).
- **Residual information:** per-cluster size and correctness counts in permuted columns. These are needed for recomputation. Someone who already holds DEV could try to align clusters by size; nobody without DEV can recover items, labels or ids.

## Verify (reviewer)
```sh
python -X utf8 -I receipts/2026-10-06-D-m0-harness/anon-packet/verify_packet.py > my-packet.json
python -c "import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); print('MATCH' if a==b else 'DIFF')" my-packet.json receipts/2026-10-06-D-m0-harness/anon-packet/expected-verify.json
```
Requirements: numpy, the pinned kit (installed `--no-deps`) and `pip install -e ./harness`. No suite, DEV rows or manifests are needed, and it runs on CPU in about a minute.

The checks:
- **Parity:** `parity_ok` holds for all 8 systems. The index and every per-benchmark raw/skill recomputed from summed statistics equal the official kit's outputs to 1e-9: perfect 98.141, uniform 1.0657, noisy-0.7 41.0698, base-0.55 24.1172, independent-0.60 30.5506.
- **Paired bootstrap (B=2000, seed 0):**

| Pair (vs base-0.55) | Δ | SE | MDE |
|---|---:|---:|---:|
| independent-0.60 | +6.4334 | 0.8411 | 2.3565 |
| base + 4% rows fixed | +2.7662 | 0.2375 | 0.6653 |
| base + 11% rows fixed | +7.0104 | 0.3466 | 0.971 |
| base + 11% cases fixed | +6.9786 | 0.3979 | 1.1149 |

  The Δ values equal `../reproduce.py` exactly. SE and MDE depend on the cluster shuffle and agree statistically (within ~3%), not digit for digit.
- **Zero-SE regression:** `base-0.55 - base-0.55` gives SE 0 and `mde_available: false`, the same handling as the `compare` CLI.

## Rebuild (author side, needs DEV)
`make_packet.py` regenerates the packet with a fresh, unrecorded shuffle, which yields a different but equally valid packet. Verify against the committed packet, not a rebuild.
