# Receipt: m0-a Decision Index 0.2.1 harness (Track D)

- **Code commit:** `81c4ec96b527d37b1b3b161cf1d887ce39d95a6f` (branch `claude/m0-harness`; review fixes over f0a6f5b; `pip-freeze.txt` was taken at this commit)
- **Official kit:** `apolinario/decision-index` @ `87d4650b42b377c0291a89c1f1a879f9b31082bf`, installed `--no-deps` from a clone with `core.autocrlf=false`
- **Board data:** HF Space `multimodalart/jev-decision-index` @ `cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d` (methodology sha256 `3724d15e6698d9a5c39fd2fc9ab70c5cc94a03aeea8fdefe2b8b30ec72420ce8`)
- **Hardware / cost:** local CPU only (Windows 11 + WSL2 `bi-ubuntu` for the suite build). Billed $0.
- **Python:** 3.12, run as `python -X utf8 -I` (UTF-8 mode is required; see INDEX_NOTES §10). Package versions: `pip-freeze.txt`.

## Data

| File | sha256 (uncompressed) | Status |
|---|---|---|
| `added-rows.jsonl` (30,419 rows) | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` | byte-identical to official |
| `selected-rows.jsonl` (124,458 rows; no HLE) | `6787a1d5cfb693c0d6aaf320057368c10b3b56e23d9c4e4bb98057a8ce6c41c7` | official `b2b56d6f…` needs gated HLE; counts match official minus HLE's 513 rows exactly |
| rebuilt 0.1 rows (131,909) | `3eb7aa1eabd0a3de999008e9a3922068a1fb7499024175020acbf83b326e1fca` | intermediate |
| `excluded-questions.json` | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` | matches |
| 5 read-time subset lists | match `editions.py` pins | `verify.json` `subsets_match: true` |

Scoreable 0.2.1 scope locally: 119,397 base + 30,419 added = 149,816 rows. With HLE's published 501 rows, the base count is 119,898, the official figure.

Data location (outside git): `C:\Users\apumu\research\CHLabs\ssdm-data\suite-0.2\` (rows) and `...\ssdm-data\slices\dev-rows.jsonl.gz` (DEV only; TEST is not materialised).

## Commands

```sh
# suite (WSL): rebuild.sh -> suite rebuild --only <all catalogs except 45>; stage.sh -> cut_v2.py (official release_v2.cut) and copy to suite-0.2/
python -X utf8 -I -m harness verify        # -> verify.json
python -X utf8 -I -m harness slice         # seed 20261006, area_target 2400, cap 0.45 -> harness/manifests/
python -X utf8 -I -m harness selfcheck     # -> selfcheck.json
python -X utf8 -I receipts/2026-10-06-D-m0-harness/demo.py > demo-bootstrap-ece.json   # B=2000, seed 0
python -X utf8 -I -m pytest -v harness/tests   # -> pytest.txt (40 passed)
```

## Slices (seed 20261006; disjoint by row, kit group and source case)

| | rows | groups | source cases | benchmarks | knowledge | language | retrieval | tools | arts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | 11,879 | 11,241 | 8,248 | 37 | 2,177 | 2,466 | 2,400 | 2,402 | 2,434 |
| TEST (sealed) | 11,936 | 11,293 | 8,184 | 37 | 2,179 | 2,500 | 2,399 | 2,403 | 2,455 |

DEV by request-length bin (proxy tokens): <128: 1,260 · 128–511: 4,939 · 512–2,047: 4,256 · 2,048–8,191: 1,202 · ≥8,192: 222.

- **Source cases** (`harness/cases.py`) are connected components over shared kit group, per-benchmark source key, identical payload and identical non-trivial state. `source_overlap_dev_test` in `slices.json` is `{}`.
- Cases above 25% of a benchmark's slice quota stay in the pool: NLI4CT 2,202 rows, ForecastBench 6,255, cfcolor 939.
- HLE's quota (225 rows per slice) is reserved; adding HLE later does not move any other benchmark (tested).
- Manifests are ordered by run_id.
- Manifest sha256 (gz files): DEV `8d83d9a826c64639013b86d5ce1cc418745409054dc133545c155c71754a7836`, TEST `99a253d78667774c0760fd69b95fd8b0970cadd11750c033a52ef9d5931e71f3`. Uncompressed hashes: DEV `78868fdb…23e3`, TEST `c1a01eb2…016`.
- TEST loads require `SSDM_UNSEAL_TEST=1` plus `SSDM_PREREG=<pre-registration>`, and each load is appended to `receipts/test-unseal-log.jsonl`. No TEST row was loaded for this receipt.

## Checks (DEV)

| Check | Result |
|---|---|
| Decomposed scorer vs official kit, three predictors | per-benchmark skill gap 0.0 |
| Perfect predictor | 100 on all 35 non-retrieval benchmarks; ToolRet/BRIGHT reach their oracle ceiling. Board-convention index 95.27 |
| Uniform predictor | index 1.14 (board convention) / 1.20 (renormalised): clipping bias |
| Point estimate, case vs group clusters | identical (tested) |

Paired bootstrap (B=2000, seed 0; index points; MDE at α 0.05, power 0.8):

| Fixture pair | Δ | 95% CI | SE (group clusters) | SE (case clusters) | MDE (case) |
|---|---:|---|---:|---:|---:|
| independent noise 0.60 vs 0.55 | +6.43 | [4.73, 8.13] | 0.891 | 0.868 | 2.43 |
| baseline + 11% rows fixed | +7.28 | [6.46, 8.04] | 0.376 | 0.398 | 1.12 |
| baseline + 4% rows fixed | +2.64 | [2.17, 3.06] | 0.239 | 0.231 | 0.65 |
| baseline + 11% of source cases fixed | +7.09 | [6.33, 7.96] | 0.395 | 0.421 | 1.18 |

**MDE at DEV size is about 0.65–2.4 index points, depending on how correlated the two systems' errors are.** Group-level clustering understates the SE once improvements are case-correlated (0.395 vs 0.421). The fixtures are iid within a case, so real systems will sit higher.

Calibration (independent 0.60 fixture, one-hot): 10-bin ECE 0.396, benchmark-equal 0.395, and ACOS supplies 49% of fields. The board-protocol stand-in covers 27 base-edition benchmarks with n = 5,569.

The noisy oracles are fixtures that see the gold answer; they are not models. No model was evaluated in this receipt, and no TEST row was loaded.
