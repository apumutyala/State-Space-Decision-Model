# Deviations from the planned run order: q9 proxy loop

## D1: stock-arm recipes run out of order with `recipes --only` (2026-10-07)

**What ran.**
1. The full `recipes` stage was started at 09:55Z at agent-main d7f1290 (`recipes.log`).
2. It trained and scored R00–R08 on SELECT.
3. At 11:13Z it stopped with `FileNotFoundError` on `<cache>/e5/train/MANIFEST.jsonl`. R09 (e5 arm) needs a feature cache that has not been built. R10 and R13 (conv1 arm) need one too. Only the stock_dev and stock_train cache parts had been run, so the cache holds only the `stock` arm.
4. Without waiting for coordinator instruction, I started `recipes --only R11,R12,R14,R15,R16,R17` at about 11:14Z (`recipes_stock_only.log`). These are the remaining recipes on the stock arm. They finished at 11:55Z with exit 0.
5. R16 and R17 used `--teacher-dir` with the pinned Clef-flash export.

**Why this does not affect selection.**
- Same code. Every receipt was produced by agent-main d7f1290, recorded as `code_identity` in each receipt.
- Same inputs. All runs used the same pinned budget (`budget_sha256`), seed 0 and cache identity, and the same `--clef`, `--stock` and `--dev-rows` inputs as the full run.
- Same amendments, A1–A3.
- Each recipe's receipt depends only on its own recipe, seed, budget and data, never on run order. The `--only` path writes exactly the receipts the full run writes.
- `--only` never writes `select_summary.json`.
- `select` still refuses until R09, R10 and R13 also have SELECT receipts (`train/proxy_loop.py:452`).
- Nothing was retrained or re-drawn, and no recipe was skipped.
- No SELECT choice, HOLDOUT attempt or verdict exists.

**Still pending.**
1. The e5_dev, e5_train, conv1_dev and conv1_train cache parts.
2. Then `recipes --only R09,R10,R13` into the same output directory, reusing every completed receipt.
3. Then `select`, then the single-shot HOLDOUT.

The coordinator reviewed this order and accepted it on 2026-10-07.
