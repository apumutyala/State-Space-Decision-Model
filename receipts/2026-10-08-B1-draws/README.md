# B1 weighted training draws (BOARD B1, for D4/D8 RWKV7-G1k-13.3B cells)

## b1-d4-draw45k (shipped first)

- Files (outside git): `ssdm-data/m2/build/b1-d4-draw45k/{records.jsonl.gz, b1-d4-draw45k.manifest.jsonl.gz, b1-d4-draw45k.content.jsonl.gz}`,
  the 150k set's format (`train.train_set` reads it unchanged). Pointer `data/manifests/b1-d4-draw45k.pointer.json`
  (n_records 45,000 = lines in the file); raw-gz pins `jobs/m2_first_cell/data_d4_draw45k.sha256`.
- Builder: `data/b1.py` @ 45208f3 (stages: `supplement` = the PR #44 builder re-run with larger glaive/banking/clinc/gsm8k
  quotas, same seed and guards; `draw` = per-family seeded selection). Rebuild:
  `python -X utf8 -I -c "...data.b1.main(['draw','--tokdir',<RWKV7 tokenizer>,'--draws','b1-d4-draw45k','--receipt',...])"`.
- Mix (`build-report.json`): tools 41.9% (Glaive native 6,455 incl. 2,955 backfill, Glaive catalog views 2,000 (K 4-16) +
  3,000 (K 17-27), ToolACE 2,000, When2Call 2,000, fact-change 500, API-Bank train 604, BFCL-synthetic 2,286);
  procedural 14.8% (GSM8K 4,238 = pool exhausted, CRUXEval-synthetic 2,400); BANKING77/CLINC150 21.1% (16-option views
  3,000 + 4,500; BANKING77 full 77-option catalogs 2,000); language 10%; other retrieval 5.6% (NQ 817, BoolQ 1,700);
  facts 4.4% (MMLU-aux 2,000); arts 2.2%. K<=16 rows 40,000, K>16 rows 5,000.
- Every row fits the cell: rendered with template `jrt-rwkv-g1-v1` (ordered layout) and the pinned RWKV7-G1k tokenizer,
  `train.run_cell.keep_item(max_k=255, max_t=2048)`; tokens p50 509 / p95 1,876 / p99 2,006 / max 2,048.
- Gold position: option order + ids re-drawn per row (seeded) so the gold's canonical position is uniform (max deviation
  <= 4.6% for every K with n >= 150); BoolQ true/false 897/897. Injected rows (3,893) keep ids (the injected text names an id).
  Per-tool yes/no rows (BFCL-synthetic, NQ per-candidate) keep the task's natural no-heavy label mix.
- Exclusions: `exclusion-check.json` (data/b1_check.py, independent of the builder's filter) PASS over all 45,000 rows:
  cohort v2 + v3 (498efa61) by id / source item / content / key-free text, Index DEV+TEST rows (155,390 rows: 0.2.1 suite +
  DEV slice + 0.3 GSM8K DEV) by content, key-free text and item text, SplitGuard, `harness.heldout.check_record` per row and
  `check_generator` per (source, family), `assert_draw_disjoint` v2/v3, D1-B scan 0, within-draw near-duplicates 0.
  `heldout-gate.json` + the trainer's own `train.train_set.draw` with the D4 cell config: pass (n_pool 45,000, v3 overlap 0).
- Not sourced: MuSR, SATA-Bench, CLadder (FROZEN held-out families, heldout-draw-v1), full CLINC150 catalogs and long NQ
  lists (> 2,048 tokens), API-Bank rows over 2,048 tokens (1,680 of 2,284).

## clef-flash-teacher-check.json

Exclusion check of the Clef-flash KD teacher export (24,000 rows of the 150k seed-0 draw): frozen families,
check_generator, cohort v2/v3 disjointness, SplitGuard, D1-B all clean; the stricter Index item-text check flags 186 rows
(When2Call 137, MMLU-aux 28, multi 7, other 14) and one key-free full-text match (m2:mmlu_aux:94003).

## b1-draw100k

- Files: `ssdm-data/m2/build/b1-draw100k/` (same format; pointer `data/manifests/b1-draw100k.pointer.json`, n 100,000).
  Raw gz sha256: records c377b24e5348d5745d686e88c8b38e688ebb4b197c5352aaaab6dd401aa8c2a6, manifest
  64a10a6c171b7339015b036284019266c1c818a21ebcce982aa1b76b0e854467, content 720d70adf45888d805abe31ffe39eb033a0f17f735ad7fdba0af06cc66a3acf6.
- Builder `data/b1.py` @ d5f4cbb (adds the Index content / key-free-text rejection to the builder's filter after the first
  100k build failed `b1_check` on 3 MMLU-aux rows; that build was discarded).
- Mix: tools 45.5%, intent 13.2%, facts 11.9%, language 15%, procedural 6.6%, other retrieval 4.8%, arts 3%; backfill
  glaive_native +7,499, mmlu_aux +496 (pools exhausted: API-Bank <=2,048 tokens, NQ, When2Call, fact-change, GSM8K).
  K<=16 90,235 / K>16 9,765; tokens p50 508 / p95 1,891 / max 2,048.
- `exclusion-check.json`: PASS over all 100,000 rows.

## b1-draw100k-v2 (low-positive-rate yes/no + minority-support views)

- Why: D1 over-called majority-positive labels on families missing from its draw (ACOS yes 25% vs gold 0.5%, RAGTruth
  true 91% vs 35%, NLI4CT Entailment 64% vs 33.5%). v2 = the 100k with 10,500 quota moved from glaive_native / mmlu_aux /
  anli to views of ALLOWED, sha-locked train sources (no download), in eval layouts, at natural base rates
  (`data/b1_lowrate.py`, `WEIGHTS_100K_V2`):
  - clinc150_member 3,000 / banking77_member 275 (BANKING77 pool exhausted after its k16/full rows): ACOS-style
    12 yes/no "intent expressed?" questions per message; yes 5.3% / 6.3% of questions; 36% / 24% of records all-no.
  - anli_claim 3,000: HoVer-style SUPPORTED / NOT_SUPPORTED (neutral + contradiction = not supported): 33.2% supported.
  - mmlu_verify 3,000: one yes/no per candidate answer: 25% yes.
  - shortfall (banking77_member and the v1 exhausted pools) backfilled by glaive_native (+9,220; final 13,720).
- Files: `ssdm-data/m2/build/b1-draw100k-v2/`, pointer `data/manifests/b1-draw100k-v2.pointer.json`; raw gz sha256
  records faee5885d0f1c74c9f98562f82b13f087a6142a04b3a55204bd51b4bb2df70d9, manifest
  ae30c2ee9b71281dfdff6139be85925df0bed3131ea62f3bb02986d2d45f6e69, content 101ebfa9552e7f409f3d68dc76609980a65ab9ad84b8403e7c2c7252bb1f1c8b.
  ANLI / MMLU-aux raw rows: `ssdm-data/m2/build/b1-lowrate-src/` (dump of the locked parquet, sha-pinned in
  `dump-manifest.json`; identity in build-report `lowrate_raw`).
- Builder `data/b1.py` + `data/b1_lowrate.py` @ f4cfd23. Tokens p50 532 / p95 1,891 / max 2,048 (0 over).
- Every view row ran the Index fingerprint (ngram13 / exact / minhash / edit / contain) as well as the v1 filter;
  view text audited against the native item (9,275 views, 0 mismatches).
- REFUSED: a Home appliances (catalog 9, `device_state_record`) handling / membership generator. It is FROZEN in
  heldout-draw-v1, and `check_generator` refuses both specs (`generator-specs.json`).
- `exclusion-check.json`: PASS over all 100,000 rows (generators incl. anli/claim_verification, clinc150+banking77/intent_membership, mmlu_aux/answer_verification).
