# L0c renderer handoff

The renderer was written separately from the kernel and reviewed by the Track L coordinator. It consumes only a currently valid kernel-minted result, preserving typed predicates, entities, comparison direction, quantities, scope and speech-act kind. Untrusted strings use JSON quoting. Plain-text quotation is not HTML sanitization: a downstream HTML or Markdown consumer must escape for its medium.

The review checked the only public rendering entry point calls `validate_release`; rejected/forged/revoked objects cannot pass it. Regression cases cover exact values and sides, hostile strings, Ask without a negative assertion, conjunction without a causal connective, and changed record version after checking. Derivation output names semantic-program nodes, not neural reasoning. This small grammar has no usefulness or fluency claim beyond its tested productions.

Dependencies are fixture source `2fe814a` in #5 and kernel source `e1dd16b` in #10, plus contract #4. For pre-merge reproduction, copy `fixtures/language_growth` from #5 and the kernel's `language_growth` files from #10 without overwriting this PR's `realize.py` or `test_realize.py`. The receipt includes hashes of the actual dependency files. Merge dependencies before this PR. Development was preserved in `ssdm-codex-l0-renderer`; the isolated publication branch is `codex/l0-realizer` in `ssdm-codex-l0-realizer-pr`.

Claude must rerun the receipt before merge. No GPU or paid job is authorized by these tests.
