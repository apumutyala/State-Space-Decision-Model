# Receipt: m0-a Decision Index 0.2.1 harness (Track D)

- **Code commit:** `f0a6f5bc15b00744c819c1fa39ae9c6ed0e0b02b` (branch `claude/m0-harness`)
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
python -X utf8 -I demo.py                  # -> demo-bootstrap-ece.json (B=2000, seed 0)
python -X utf8 -I -m pytest -v harness/tests   # -> pytest.txt (37 passed)
```

## Slices (seed 20261006; groups never split; DEV and TEST disjoint by row and by group)

| | rows | groups | benchmarks | knowledge | language | retrieval | tools | arts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | 11,781 | 11,125 | 37 | 2,177 | 2,406 | 2,399 | 2,399 | 2,400 |
| TEST (sealed) | 11,786 | 11,155 | 37 | 2,178 | 2,409 | 2,399 | 2,400 | 2,400 |

DEV by request-length bin (proxy tokens): <128: 1,243 · 128–511: 4,922 · 512–2,047: 4,211 · 2,048–8,191: 1,183 · ≥8,192: 222.

HLE's quota (225 rows per slice) is reserved: adding HLE later does not move any other benchmark's selection (tested). Manifest sha256 (gz files): DEV `f8cd4ba2c6ceca44b2263070fc9bd2710080464fd1535d9201c13868b927a52c`, TEST `3dc79c0a975cba4c3428a87087e0af8eb149ebc85618a2062fe037eacb8852e8`; uncompressed-content hashes are in `harness/manifests/slices.json`.

## Checks (DEV)

| Check | Result |
|---|---|
| Decomposed scorer vs official kit, three predictors | per-benchmark skill gap 0.0; index identical to 1e-13 |
| Perfect predictor | 100 on all 35 non-retrieval benchmarks. ToolRet/BRIGHT reach their oracle ceiling (<1, because some relevant documents are outside the 32 candidates). Board-convention index 95.18 (HLE counted 0 plus the retrieval ceiling) |
| Uniform predictor | index 1.89 (board convention) / 1.98 (renormalised): the positive bias from clipping chance-level noise at 0 |
| Paired bootstrap, noisy-oracle 0.60 vs 0.55 | +6.90 index points, 95% CI [5.09, 8.64], SE 0.90 |
| MDE (α 0.05, power 0.8) at DEV size | 2.53 points; 1.79 at 2× DEV; 1.26 at 4× DEV |
| Calibration estimators on one-hot predictions (acc 0.602) | 10-bin ECE 0.398, debiased 0.398, SmoothECE 0.398 (expected 1 − acc) |

The noisy oracles are fixtures that see the gold answer; they are not models. No model was evaluated in this receipt, and no TEST row was loaded.
