# Plan for porting sym-fork to the RWKV line (D19 / D20). This is a plan only; no code goes in yet.

Target: `stack/rwkv7/d19-v3b-joint` (D19: template `jrt-rwkv-g1-v3w`, layout `g1v2`, head `joint_record`) and its
successors. **Do not touch those branches while D20 is mid-chain.** Cut a new branch,
`stack/rwkv7/d19-sym-invariant`, from the D19 tip, or from the D20 tip once its chain finishes. Then merge
`stack/qwen35/sym-invariant` into it, or cherry-pick the files listed in step 1.

## What already exists on stack/qwen35/sym-invariant

- **`heads/symmetric.py`.** Its `g1_pieces` renders the RWKV7-G1 vendor format: the `System: …\n\nUser: ` head,
  `g1_clean` on all request text, and a lean `"\n\n"` turn end per option (or the v2 answer tail under
  `sym-fork-chat`). Tokenization is whole-string through the row encoder's `token_bytes`, and id 0 is refused.
  Template detection is `is_g1_template` (`jrt-rwkv-g1-*`).
- **`tests/test_sym_invariance.py::test_g1_format_rows_and_end_to_end_invariance_on_tiny_rwkv7`.** Reorder + relabel
  invariance on a tiny RWKV-7 (state fork) holds bit-exactly for `sym-fork` and `sym-fork-chat`.
- **The eval path and tie rules.** `jobs/sym_eval/perturb_eval.py`, `decide.schema.wire_answer(tie=…)`, and the
  `content_order` softmax in `decide/adapter.py` are all template-agnostic.

## Steps

1. **Bring the files over.** These are `heads/symmetric.py`, plus the three-line hooks in `heads/layout.build_row`,
   `heads/features.py` (e from the fork) and `heads/decide_backend.py` (token count, tie info, sym_version). Also
   `decide/schema.py` and `decide/adapter.py` (tie rules), `train/run_cell.keep_item` (the `SYM_MAX_CONT` cap),
   `jobs/sym_eval/`, and both test files.
   - Expected conflicts: `heads/features.py` (D19 adds `lex` / `groups` to `HeadInput`) and `heads/layout.py`
     (D19's layout is older). Both are additive, so resolve by keeping both sides.
2. **Lexical prior (D19 leak-map row 5).** In `heads/model.lexical_vectors`, add a branch so sym rows read
   `r.option_seqs[k]` (the description tokens alone). Today it reads the schema span
   `"<CODE>. <id>: <desc>"`, which carries both the id and the position code.
   - It is one condition before the `ordered`/`g1v2` branch: `if r.layout in SYM_LAYOUTS: toks = r.option_seqs[k]`.
   - Test: lex vectors are identical under relabel, and they move under relabel in the D19 g1v2 rows (teeth).
3. **Record grouping.** `record_key` groups by the shared schema+state prefix. Sym prefixes are per question (state
   + this question + K), so rows of one record no longer share a prefix. Training already sets `meta["record"]`.
   At serve time, set `meta["record"] = request_fingerprint(request)` in `DecisionModelBackend._rows` for sym
   layouts. That is one line in the backend.
   - Test: a multi-question request forms one group under both paths.
4. **Joint mixer.** Optional; it makes the record mixer independent of question order (D19 row 6). It is invariant
   to option order already: within a field, options are pooled by mean/max. The question axis is a separate claim.
   - If wanted, swap `_scan` for a DeepSets field mixer: `c_i = ρ(φ_p(payload) + Σ_{j≠i} φ(x_j))`. Its sum is
     exclusive, which keeps the "a single-question record sees only itself + payload" property. Zero-init `w` and
     `lex_scale` keep "step 0 = the fork head".
   - Register it as a new head name (`joint_record_sym`) so D19 checkpoints still load.
5. **Cost and clock.** RWKV-7 copies a fixed-size state per option (no KV), so a K-way fork costs O(K) state
   copies, independent of T. Re-run the token census with `scripts/sym_reprice_jev1.py` adapted to the v3b draw
   (the G1k tokenizer, `keep_item` caps). v3b has K up to 79 (BANKING77 K77, API-Bank K40-60). For those rows the
   continuations replace the option lines in the prefix, which is about neutral, but the right padding per 64-chunk
   grows with the longest description. Check the computed-token factor before pricing.
6. **Name-only catalogs (A2).** API-Bank name-only rows (1,787 in v3b) have no descriptions. Their options render by
   id (`id_as_content`), so reorder invariance holds and relabel invariance cannot hold for them (it is not a
   symmetry). Report their share apart in the G1 gate.
7. **Gate.** Same as A-PREREG-JEV1-4B-SYM, G1/G2, with the D19 full-DEV 49.49 cell as comparator. Also run
   `perturb_eval --layout g1v2` on the D19 checkpoint for the flip baseline.

Estimated effort: about 60 lines of code plus tests (steps 1–3), and about 80 more for step 4. CPU-testable on tiny
RWKV-7 throughout.
