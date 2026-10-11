# REGATE_JEV2_FINAL: combined jev2 pointer and the jev1+jev2 mix (evidence lane, 2026-10-11)

**Verdict: CONFIRMED.** The combined pointer and the mix manifest reference exactly the gated builds. Every sha matches
the git blob and the records on disk. Neither contains a superseded build. There are 0 cross-draw duplicates.

**Consolidated mix exclusion list:** `mix_exclude_ids.txt` holds 23 ids, sha256 c0c175f50a7886b53bc647e022bb00705747deadf535a2708fef19c32ab9c2ad.
- All 23 come from jev2a-v2.
- jev1 contributes 0: I ran the verbatim ≥ 6-word-leaf rule over all 96,000 jev1 rows, including the replaced
  long_policy rows (`jev1-exact-leaves.json`).

Ref: claude/ev-jev2-draw @282a4867. Script: `scripts/regate_mix_check.py`, output `mix-check.json`.

## Combined pointer `data/manifests/bschema-draw-jev2.pointer.json` (sha256 bef9ec72…, CONFIRMED)
| component | pointer sha (claimed = git) | records sha (pointer = disk) | n / families |
|---|---|---|---|
| bschema-draw-jev2a-v2 | eba70d2c… ✓ | 00a1d722… ✓ | 7,500 ✓ |
| bschema-draw-jev2b | 35620f5f… ✓ | 9406ae08… ✓ | 7,500 ✓ |
| bschema-draw-jev2c-v2 | 86960d90… ✓ | 0bdb1ac6… ✓ | 1,500 ✓ |
- **Total:** 16,500 rows ✓, 11 families × 1,500.
- **Superseded builds:** jev2a (393d0f0f…) and jev2c (0a297a58…) are listed under `superseded_do_not_mix`, their shas
  match the disk, and neither is a component ✓.
- **Cross-draw duplicates:** 0 among the 3 components by id, by content hash and by normalised text. There are also 0
  within-draw duplicates.
- **Overlap with the superseded builds (information only; those builds are not mixed):**
  - jev2a-v2 shares 4,441 content hashes with jev2a, because it re-samples the same sources.
  - jev2c-v2 reuses 1,345 jev2c row ids but with different content: 0 content-hash matches.

## Mix manifest `data/manifests/bschema-mix-jev1-jev2.json` (sha256 71b503eb…, CONFIRMED)
- **Components:** every component pointer sha equals its git blob, and every records sha equals both its pointer and
  the disk. They are jev1 582ffe1c… / 45735109… (the gated jev1 ✓), jev2a-v2, jev2b and jev2c-v2.
- **Families:** 18 ✓. **Rows:** 100,500 ✓. That is the 84,000 jev1 rows without long_policy, plus 16,500 jev2 rows.
- **long_policy:** jev1's 12,000 long_policy rows are excluded, and the long_policy owner is bschema-draw-jev2c-v2 ✓.
- **Superseded builds** jev2a and jev2c are absent ✓.
- **Cross-draw duplicates** across all 4 mixed components: 0 by id, 0 by content hash, 0 by normalised text ✓.

## Not covered in this round
- I did not re-run the full gate battery on jev1. Only the verbatim-leaf rule and the hash and identity checks were in
  scope.
- The MuSiQue provenance caveat for 104 jev2a-v2 rows is described in REGATE_JEV2A_V2.md, section 4.
