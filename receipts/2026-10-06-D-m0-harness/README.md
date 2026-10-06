# Receipt: m0-a Decision Index 0.2.1 harness (Track D)

- **Code commit:** `1eb50538b2e0d83660745bfe4769a64f7d4920ff` (branch `claude/m0-harness`; input validation over a3739f4; `pip-freeze.txt` was taken at this commit)
- **Official kit:** `apolinario/decision-index` @ `87d4650b42b377c0291a89c1f1a879f9b31082bf`, installed `--no-deps` from a clone with `core.autocrlf=false`
- **Board data:** HF Space `multimodalart/jev-decision-index` @ `cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d` (methodology sha256 `3724d15e6698d9a5c39fd2fc9ab70c5cc94a03aeea8fdefe2b8b30ec72420ce8`)
- **Hardware / cost:** local CPU only (Windows 11 + WSL2 `bi-ubuntu` for the suite build). Billed $0.
- **Python:** 3.12, run as `python -X utf8 -I` (UTF-8 mode is required; see INDEX_NOTES §10). Package versions: `pip-freeze.txt`.

## Data (complete 0.2.1 suite, byte-identical to the official files)

| File | sha256 (uncompressed) | Status |
|---|---|---|
| `selected-rows.jsonl` (124,971 rows) | `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` | **byte-identical** to official |
| rebuilt 0.1 rows (132,422) | `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99` | **byte-identical** to official |
| `added-rows.jsonl` (30,419 rows) | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` | **byte-identical** to official |
| 43 normalised source files | per-file hashes in kit `hub/manifest.json` | 43/43 match (`cmp_norm.py`) |
| HLE parquet (cais/hle @5a81a4c7) | `6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a` | gated; fetched after the terms were accepted (amutyala3) |
| `excluded-questions.json` | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` | matches |
| 5 read-time subset lists | match `editions.py` pins | `verify.json` `match: true` |

Scoreable 0.2.1 scope: 119,898 base (HLE 501) + 30,419 added = 150,317, the official count. `verify.json` reports `match: true` and no missing index benchmark.

Data location (outside git): `C:\Users\apumu\research\CHLabs\ssdm-data\suite-0.2\` (rows) and `...\ssdm-data\slices\dev-rows.jsonl.gz` (DEV only; TEST is not materialised).

## Commands

```sh
# suite (WSL): rebuild.sh (all catalogs but HLE) -> stage.sh/cut_v2.py; then hle_fetch.py (Windows) + hle_rebuild.sh (WSL: --only 45 --skip-download, official 0.2 cut) -> rebuild-hle.log; cmp_norm.py checks the 43 normalised files
python -X utf8 -I -m harness verify        # -> verify.json
python -X utf8 -I -m harness slice         # seed 20261006, area_target 2400, cap 0.45 -> harness/manifests/
python -X utf8 -I -m harness selfcheck     # -> selfcheck.json
python -X utf8 -I receipts/2026-10-06-D-m0-harness/demo.py > demo-bootstrap-ece.json   # B=2000, seed 0
python -X utf8 -I -m pytest -v harness/tests   # -> pytest.txt (98 passed)
```

## Slices (seed 20261006; disjoint by row, kit group and source case)

| | rows | groups | source cases | benchmarks | knowledge | language | retrieval | tools | arts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | 12,104 | 11,466 | 8,473 | 38 | 2,402 | 2,466 | 2,400 | 2,402 | 2,434 |
| TEST (sealed) | 12,161 | 11,518 | 8,409 | 38 | 2,404 | 2,500 | 2,399 | 2,403 | 2,455 |

DEV by request-length bin (proxy tokens): <128: 1,304 · 128–511: 5,082 · 512–2,047: 4,292 · 2,048–8,191: 1,204 · ≥8,192: 222.

- **Source cases** (`harness/cases.py`) are connected components over shared kit group, per-benchmark source key, identical payload and identical non-trivial state. `source_overlap_dev_test` in `slices.json` is `{}`.
- Cases above 25% of a benchmark's slice quota stay in the pool: NLI4CT 2,202 rows, ForecastBench 6,255, cfcolor 939.
- **HLE added:** 225 rows in each slice, filling exactly its reserved quota. The other 37 benchmarks' manifest entries are identical to the pre-HLE cut (b2edd86), checked entry by entry.
- Manifests are ordered by run_id.
- Manifest sha256 (gz files): DEV `1362c23ebe810cccf8f2ec87533d454cc3aacd6ba6dc6086e14ad36211c8073a`, TEST `76a6b135cfcc11ab3448dd80409567ad86e6edbc4593da2c33d1f67f6f0abe12`. Uncompressed: DEV `26cd6bc0…964a`, TEST `8de5aaed…155a`.
- TEST loads require `SSDM_UNSEAL_TEST=1` plus `SSDM_PREREG=<pre-registration>`, and each load is appended to `receipts/test-unseal-log.jsonl`. No TEST row was loaded for this receipt.

## Reproduce the headline claims (one command)

```sh
python -X utf8 -I receipts/2026-10-06-D-m0-harness/reproduce.py > my-reproduce.json
python -c "import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); print('MATCH' if a==b else 'DIFF')" my-reproduce.json receipts/2026-10-06-D-m0-harness/expected-reproduce.json
```

The expected output is `expected-reproduce.json`, and two runs here were identical. It checks:
1. **Suite:** the official hashes match, base scoreable = 119,898, HLE = 501, and no index benchmark is missing.
2. **DEV:** 12,104 rows with manifest gz `1362c23e…073a`. Every row is re-hashed against the manifest on load.
3. **Parity:** the harness index equals the unmodified kit's `balanced_skill`, with per-benchmark skill gap 0.0, for the perfect (98.141), uniform (1.0657) and noisy-0.7 (41.0698) fixtures.
4. **MDE range:** the paired source-case bootstrap (B=2000, seed 0) gives [0.6455, 2.3786] index points, with each pair's Δ, 95% CI and SE.

Values are rounded to 4 decimals. The run takes a few minutes on CPU, uses the materialised DEV rows only, and never loads TEST.

## Input validation (Codex review)

- **Calibration:** the estimators reject empty input, NaN/inf or out-of-range confidence, non-binary correctness, mismatched shapes and bad weights with `CalibrationInputError`, never an ECE of 0. Empty report blocks are `{"n": 0, "available": false}`. `fields()` runs the kit's own response validator on every `ok` result before reading it.
- **Bootstrap:** `bootstrap.replicate_skills` refuses to pair systems whose benchmark sets or cluster orders differ.
- **MDE helpers:** they require positive finite SE, sample sizes and effects, and α and power strictly inside (0, 1). numpy scalars are accepted.
- **Tests:** `harness/tests/test_validation.py` covers all of this.
- **Zero SE (Track L finding):** `harness compare` on two identical results files used to crash in `mde.table`. Its MDE block is now `mde.report(...)`, which returns `{"available": false, "se": 0.0, "note": ...}` when the paired SE is 0. A regression test runs the CLI end to end on DEV.

## Anonymous packet (Codex request)

`anon-packet/` holds per-source-case sufficient statistics for 8 fixture systems plus the official kit's aggregates. With it, a reviewer can recompute every DEV aggregate, and the paired bootstrap, without seeing any item, label or id: shuffled, unrecorded permutation, and a leak audit finding 0 of 43,697 ids and 0 of 16,903 strings. See `anon-packet/README.md` and `anon-packet/verify_packet.py` with `expected-verify.json`.

Value functions are now built from plain-data specs (`score.value_fn`), so the packet and the harness share one code path. Tests total 98.

## Checks (DEV, with HLE)

| Check | Result |
|---|---|
| Decomposed scorer vs official kit, three predictors | per-benchmark skill gap 0.0 |
| Perfect predictor | 100 on all 36 non-retrieval benchmarks; ToolRet/BRIGHT reach their oracle ceiling. Index 98.14 (board convention = slice index, now that nothing is missing) |
| Uniform predictor | index 1.07: clipping bias |
| Point estimate, case vs group clusters | identical (tested) |

Paired bootstrap (B=2000, seed 0; index points; MDE at α 0.05, power 0.8):

| Fixture pair | Δ | 95% CI (case) | SE (group clusters) | SE (case clusters) | MDE (case) |
|---|---:|---|---:|---:|---:|
| independent noise 0.60 vs 0.55 | +6.43 | [4.82, 8.11] | 0.839 | 0.849 | 2.38 |
| baseline + 11% rows fixed | +7.01 | [6.26, 7.61] | 0.342 | 0.341 | 0.96 |
| baseline + 4% rows fixed | +2.77 | [2.26, 3.17] | 0.240 | 0.230 | 0.65 |
| baseline + 11% of source cases fixed | +6.98 | [6.25, 7.83] | 0.365 | 0.396 | 1.11 |

**MDE at DEV size is about 0.65–2.4 index points, depending on how correlated the two systems' errors are.** Group-level clustering understates the SE once improvements are case-correlated (0.365 vs 0.396). The fixtures are iid within a case, so real systems will sit higher.

Calibration (independent 0.60 fixture, one-hot): 10-bin ECE 0.397, benchmark-equal 0.398, and ACOS supplies 49% of fields. The board-protocol stand-in covers 28 base-edition benchmarks with n = 5,615.

The noisy oracles are fixtures that see the gold answer; they are not models. No model was evaluated in this receipt, and no TEST row was loaded.
