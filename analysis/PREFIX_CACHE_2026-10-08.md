# Cross-row prefix caching: RWKV7 (D1 g1-v1, v2f, v3w) vs Clef (2026-10-08)

**Takeaway.** The ~1,723-token per-row prefix is not a constant preamble. It is the request's own schema and state,
and the trained cell already reads it only once per request (the tree fork).
- **Identical across every DEV row:** only the chat header plus system prompt plus `SCHEMA FIELDS:`. That is 42 tokens
  for g1-v1, 43 for v2f / v3w (token 0 included) and 36 for Clef, under 2.3% of the tokens each model computes.
- **Identical within a family:** 275 tokens per row for D1 (11.1%) against 149 for Clef (9.7%).
- **Ideal prefix (radix) cache over all DEV rows:** −21.8% of D1's computed tokens against −14.8% of Clef's FLOPs.

Both models benefit, so the asymmetry is small. Under the best policy (radix), it moves the RWKV ÷ Clef 27B T1 cost
ratio by about 8% in RWKV's favour. Caching alone does not make 2× vs Clef 27B reachable:
- The best combination (v2f readout + fla WKV + radix cache on both sides) is **0.93×**, i.e. 1.08× cheaper.
- D1 tree + fla + radix is **1.00×**.
- 2× needs RWKV at **≥ 54–58% end-to-end MFU**. fla measures 29%, and Clef runs at 38%.
- At equal MFU, 2× needs ≤ ~1,285 computed RWKV tokens per row. D1 tree + radix computes 1,942.

Reproduce (CPU, tokenizers only, ~25 min for the two whole-string arms):

    python -I -X utf8 analysis/prefix_cache.py tokens --arm g1v1|v2f|v3w|clef --out <dir>
    python -I -X utf8 analysis/prefix_cache.py boundary --out <dir>
    python -I -X utf8 analysis/prefix_cache.py analyze --out <dir> --json analysis/prefix_cache/result.json

Outputs: `analysis/prefix_cache/result.json` (per arm, per family, prices) and `analysis/prefix_cache/boundary.json`.

## What was rendered

Every one of the 12,104 full-DEV requests was rendered with each arm's own builder and tokenizer. None was refused.

| arm | builder | tokenizer | what is read once per request | tokens computed per request |
|---|---|---|---|---|
| D1 (g1-v1) | `heads.layout.build_row` (jrt-rwkv-g1-v1, ordered), per segment (D1's `RwkvSegmentEncoder`) | G1k `tokenizer.json` (ssdm-data/tokenizers) | common token prefix of the question rows (the tree prefix) | `tree_plan` positions |
| v2f / v3w | `decide.prompt_g1v3.render_g1v3` @ `9c3e435` (b-train-v3w; no diff vs r3-catalog `0290fa8` on these files) | `[0]` + whole string, G1k | common prefix of the coded question prompts | the backend's `plan` total |
| Clef | `baselines.clef_backend.ClefEncoder` (Clef's `encode_record` layout) | Clef's own tokenizer.json (sha `06b95093…`, the pin) | the single joint sequence | its length |

Checks on the 2,000 common ok subset rows:
- The Clef lengths equal the landed Clef 27B `model_tokens` on 2,000 of 2,000 rows.
- The v3w totals equal the landed fgu2c v3w `model_tokens` on 2,000 of 2,000 rows.
- The D1 tree positions (mean 2,482.3) equal `analysis/tree_fork/volume.json`.
- The D1 tree prefix averages **1,722.9 tokens**. This is the "~1,723" in the question.
- v2f computes 2,234 tokens per row, more than v2's 1,830. The instruction fix renders the dict instructions of ToolRet
  and BRIGHT, which v2 replaced by the question id.

## 1. Constant-prefix tokens per row (2,000 common rows; full DEV and the 2,008 subset agree within 1 point)

Each cell gives mean tokens per row and, in brackets, the share of the tokens that arm computes.

| policy | D1 g1-v1 (2,482 computed) | v2f (2,234) | v3w (2,401) | Clef (1,543) |
|---|---:|---:|---:|---:|
| global LCP: identical in every row | 42 (1.7%) | 43 (1.9%) | 43 (1.8%) | 36 (2.3%) |
| family LCP: identical in every row of its family | 275 (11.1%) | 218 (9.8%) | 216 (9.0%) | 149 (9.7%) |
| family LCP, cut back to a multiple of 64 | 246 (9.9%) | 188 (8.4%) | 187 (7.8%) | 114 (7.4%) |
| radix, any other row: longest prefix shared with some row | 567 (22.8%) | 468 (20.9%) | 463 (19.3%) | 253 (16.4%) |
| radix, amortised: exact saving of a trie cache, first row pays | 541 (21.8%) | 444 (19.9%) | 439 (18.3%) | 228 (14.8%) |

What the global constant contains:
- RWKV: `System: <system prompt>\n\nUser: SCHEMA FIELDS:\n`. The next text is `FIELD 1\nID: <question id>`.
- Clef: `<|im_start|>system…<|im_end|>\n<|im_start|>user\nSTATE:\n`.

There is no shared task-family instruction block or option-code legend before the first row-specific token. The schema
carries per-row content (question ids, instructions containing the item, and options), so the family LCP usually ends
inside FIELD 1.

**Per family, the layouts cut both ways** (subset; family-LCP share of computed tokens, D1 / v3w / Clef):

| family | D1 | v3w | Clef | why |
|---|---:|---:|---:|---|
| POP909 | 0.65 | 0.30 | 0.01 | constant multi-field schema; RWKV puts the schema first |
| PhishNChips | 0.44 | 0.43 | 0.03 | same |
| ContractNLI | 0.38 | 0.31 | 0.01 | same: 17 fixed hypotheses |
| iSarcasmEval-B-En | 0.51 | 0.44 | 0.07 | same |
| Amazon-ESCI | 0.31 | 0.24 | 0.07 | same |
| API-Bank | 0.01 | 0.01 | **0.78** | constant API catalogue in the **state**; Clef puts the state first |
| NLI4CT, CFColor (radix) | 0.09 / 0.15 | 0.09 / 0.13 | **0.81 / 0.73** | rows share whole states (premise / context) |
| short-prompt QA (ANLI, WinoGrande, CLadder, …) | 0.17–0.26 | 0.18–0.31 | 0.30–0.46 | a ~60–85-token constant over ~250–450-token rows; Clef's rows are shorter |

The full table is `families_subset` / `families_full_dev` in `result.json`.

## 2. $/1k rows with cross-row caching (T1 = max batch on today's kernels; 2,000 common rows)

Method:
- **RWKV:** (computed tokens − cached tokens) × the arm's T1 per-token rate from `throughput_maxbatch.py`. The rate is
  linear in tokens, because WKV is O(1) per token.
- **Clef:** T1 × FLOPs_cached ÷ FLOPs_full, where FLOPs_cached = 2P(n−c) + GDN(n−c) + 2·qd·(n²−c²) per full-attention
  layer. The suffix tokens' attention over the cached prefix stays in. It is small (0.08 TFLOP per row out of 77 for
  Clef 27B under the family policy), because the cached prefixes are short against the rows.

| arm | no cache | global | family LCP | family, 64-aligned | radix (amortised) |
|---|---:|---:|---:|---:|---:|
| Clef 27B | 0.167 | 0.163 | 0.151 | – | 0.143 |
| Clef-flash | 0.061 | 0.059 | 0.055 | – | 0.052 |
| D1 tree, chunked_lean | 0.382 | 0.375 | 0.339 | 0.344 | 0.299 |
| **D1 tree + fla** | 0.183 | 0.180 | 0.163 | 0.165 | **0.143** |
| v2f + fla | 0.165 | 0.162 | 0.149 | 0.151 | **0.132** |
| v3w + fla | 0.177 | 0.174 | 0.161 | 0.163 | 0.145 |

**RWKV ÷ Clef 27B at T1, both sides cached the same way (<1 means RWKV is cheaper):**

| RWKV arm | no cache | family | radix |
|---|---:|---:|---:|
| D1 tree + fla | 1.09 | 1.08 | 1.00 |
| v2f + fla | 0.99 | 0.98 | **0.93** |
| v3w + fla | 1.06 | 1.07 | 1.01 |

Against Clef-flash, every RWKV arm is still 2.5–3.0× more expensive.

## 3. Is there a real asymmetry? Does 2× become reachable?

**The asymmetry is real but small.**
- Under the family policy, RWKV saves 11.1% of its tokens and Clef 9.6% of its FLOPs: a 1–2% swing.
- Under the radix policy, RWKV saves 21.8% and Clef 14.7%: an **~8% swing** (ratio 1.09 → 1.00 for D1, 0.99 → 0.93 for
  v2f).

The swing comes from layout, not from the O(1) state. JRT puts the schema before the state, so RWKV can cache a
constant schema (POP909, ContractNLI, PhishNChips). Clef puts the state first, so it caches constant or shared states
instead (API-Bank, NLI4CT, CFColor). Each model caches whatever its own format puts first.

The O(1) state does not make caching cheaper in compute terms:
- Clef's leftover attention over the cached prefix is ~0.1% of its FLOPs.
- A cached Clef prefix of 149–228 tokens costs only 64 KiB per token of KV (≤ 15 MiB), shared read-only across the
  batch, plus a 147 MiB GDN state per branch. RWKV needs 62 MiB per branch.

**2× vs Clef 27B is still not reachable on caching + tree + fla.** The best combination, v2f + fla + radix on both
sides, gives 0.93× (1.08× cheaper). D1 tree + fla + radix gives 1.00×. At Clef's measured T1 (38% MFU), 2× needs RWKV
end-to-end MFU of:

| RWKV arm | MFU needed for 2× | measured fla T1 MFU |
|---|---:|---:|
| D1 tree + fla, no cache | 63% | 29% |
| D1 tree + fla, family cache | 62% | 29% |
| D1 tree + fla, radix cache | **58%** | 29% |
| v2f + fla, radix cache | **54%** | 29% |

At equal MFU, the RWKV ÷ Clef FLOPs ratio is 0.76 (D1, radix) or 0.70 (v2f, radix): 1.3–1.4×, not 2×. 2× at equal MFU
needs ≤ ~1,285 computed RWKV tokens per row. D1 tree + radix computes 1,942 and v2f + radix 1,791. Caching closes about
a third of that gap, the same third it closes for Clef.

## 4. Implementation feasibility (exactness of a reused state)

- **Token-space LCPs are exact by construction.** Two rows whose token ids agree on the first c ids put an identical
  token stream into the model from the same start, so the state after c tokens is the same function of the weights:
  - token 0 for v2f / v3w is inside every preamble;
  - D1 / g1-v1 has no token 0, and its preamble is the same 42-token head.

  A serving cache keyed on token ids (radix/trie over token blocks) needs no text-boundary reasoning. Tokenize the
  row first, then look up its longest cached token prefix.
- **Text-boundary preamble ("compute the family preamble state once, then append each new row"):**
  - **Whole-string arms (v2f / v3w, F5).** `tok(head + rest) == tok(head) + tok(rest)` at `…SCHEMA FIELDS:\n` on
    844 of 844 sampled rows (20 per family). At `…FIELD 1\nID: ` it holds on **0 of 844**: the trailing space merges
    with the question id (`ID: nda-1` → ` nda`). For every family preamble (the family LCP decoded), the token
    round-trip is exact in 43 of 43 families. Against 404 different continuations, **the last preamble token can
    change (backoff = 1 token)** in 42 of 43 families, and never more than one. Rule: cache the state at the family
    LCP **minus one token**, or at a template boundary that ends in `\n`.
  - **Segment-wise arms (D1 g1-v1, Clef).** Every segment is tokenized on its own, so any segment boundary is stable by
    construction (the 36-token D1 / Clef heads: 844 of 844). A family LCP that ends inside a data segment (for
    example inside a question id) must back off to that segment's start. The whole-string continuation test in
    `boundary.json` does not apply to these two arms, and its large g1-v1 "backoff" numbers are an artefact of
    re-tokenizing segment-wise ids as one string.
  - **The 64-aligned column covers both rules.** Its ≤ 63-token cut is larger than either backoff, and it costs only
    ~1 point of share (family LCP 11.1% → 9.9% for D1).
- **Numerics.** Prefilling c cached tokens and then the rest in a second call changes only where the WKV / GDN kernels
  split chunks:
  - The tree fork already splits at the request prefix. Its fp32 exactness on 606 questions is |Δlogprob| ≤ 4.8e-7.
  - A cache boundary at a multiple of the kernel chunk (64 for fla; 16 for the chunked WKV) reproduces the one-pass
    chunking of the prefix exactly. The remaining differences are bf16 rounding in the state hand-off, the same class
    the tree fork and fla re-gates already accept.
  - Re-gate the cached path with the pre-declared equality gate (argmax ≥ 99.5%, flips only below a 0.01 margin,
    |Δp| ≤ 0.02).
- **State memory (RWKV).**
  - Family policy: one 62 MiB state per family (43 families = 2.6 GiB, resident or host-pinned).
  - Radix policy, offline: process rows sorted by token prefix (DFS) and keep only the stack of branch states along
    the current path.
  - Clef needs KV for the cached tokens (shared blocks) plus GDN/conv state per branch. That is the same engineering,
    which Clef's `use_cache=False` Index path does not have today either.
- **Caveats.**
  - The radix figures are an oracle over this DEV set. Offline batch scoring can reach them by sorting. Online traffic
    gets the family figure only if the traffic mix and order repeat.
  - The T1 per-token rates are the projections from `THROUGHPUT_MAXBATCH_2026-10-08.md`, not measurements of a
    cached path.
  - The v2f / v3w chunked rate reuses the stock-v2 arm's rate.
