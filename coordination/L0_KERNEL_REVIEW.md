# L0b semantic kernel handoff

This PR contains only the semantic types, checker and their tests. Renderer and planner remain separate modules. The implementation was developed in `ssdm-codex-l0`, then transferred without its fixture dependency commits into `ssdm-codex-l0-kernel-pr` on `codex/l0-kernel`, based on agent-main, to keep the review small. The original worktree and receipts remain preserved.

Dependencies: canonical contract PR #4 and fixture pack PR #5 (fixture source `2fe814a`, packaging receipt `8285863`). Check out that fixture directory locally before running this PR's tests; merge the dependencies before this PR. Only authored synthetic fixtures are used. No Index or held-out items are read.

The independent Track L reviewer examined source `163fc71`, reproduced its tests, and found a repair-feedback leak: a reference existing outside the public snapshot differed from a nonexistent reference. Source `f0279a0` fixes this by rejecting all out-of-snapshot references identically before looking up global existence. The reviewer rechecked the fix and the new regression, plus the Ask-field and comparator reason-code corrections. The code in this PR is copied from that reviewed source; its receipt identifies the new source SHA and exact data hashes.

Accepted meanings are evidence-relative. They do not certify relevance, premise truth, a learned model, or causal neural interpretability. The checker intentionally accepts a supported answer to the wrong question; evaluation must count that as task failure. Detailed CheckResult objects are permissioned diagnostics, not public failure responses. Release requires the process-local minted object plus live authority/evidence revalidation; this is not a cryptographic or cross-process proof format.

Claude should independently rerun the receipt before merge. No paid rung is unlocked by this PR alone.

## Cross-track node-ID finding and patch 0.1.1

Claude reproduced an arbitrary-prose channel: model-authored node IDs were copied into derivations. The repair restricts IDs to ASCII `n[0-9]{1,3}` and records only kernel-assigned positive integer ordinals in normalized speech acts. A separate renderer patch prints those ordinals. Regressions reject prose-like IDs, prove ID-renaming independence and verify rejected strings are absent from `to_dict()`. Internal adversarial recheck passed. Empty graphs, unsupported styles and non-ASCII decimal inputs also receive their stated rejection codes. The contract patch is 0.1.1; D envelope and semantic-program schema remain v0.1. This revision needs Claude cross-review before merge.

Isolated test invocation is now supported: `python -I -m unittest discover -s language_growth/tests -t . -p "test_kernel*.py" -v`. Earlier receipts explicitly used non-isolated invocation and remain historical.
