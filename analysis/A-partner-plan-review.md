# A — Adversarial review of the partner plan (`plans/partner/`)

Date: 2026-10-06. Reviewer stance: adversarial but fair; every verdict carries a confidence level and the evidence that would change it. Inputs: `plans/partner/{PLAN,SF525_PILOT,ASTRA_REVIEW,EVIDENCE}.md` read in full; judged against `reviews/01–04`; `plans/claude/PLAN.md` skimmed only to locate differences (the claude plan has its own reviewer). Web checks performed today are listed in §6.

Confidence scale: **High** = primary source or arithmetic; **Medium** = consistent with the evidence base but depends on an unmeasured quantity; **Low** = my estimate.

---

## 0. Verdict in one paragraph

The partner plan is the better *specification* of the controlled model — its authority/information-flow contract, its "semantic sufficiency" framing, its oracle-decomposition methodology, its typed program algebra, and its spend gating are all more careful than anything in the claude plan — and it is the worse *route to the stated capability target*. Its only capability experiment is a 50M–100M pure SSM trained from initialization on 100M–300M tokens for ≈$187. Against the verified board (sub-1B entrants on *pretrained* backbones score 6–28; Clef-flash ≈57; Jev 57.9) that experiment cannot say anything about beating Clef or Jev, and the plan's own text concedes this ("not asserted winners", "a prerequisite, not victory"). Its headline target — beat Clef, Clef-flash, Jev, **Astra and Fable** "including difficult knowledge and reasoning tasks" — is a scope error: Astra and Fable are GPT-6 Astra and Claude Fable 5.1 (GPQA Diamond ≈96 / ≈94 vs Clef 48), they are ineligible for the Decision Index board under its >1,000 ms rule, and a full-suite comparison against them would cost roughly $2–8K in API fees that no budget line covers. The plan also contains one stale fact that materially changes its first experiment: official Mamba-3 checkpoints (187M–1.5B, Apache-2.0, no attention) do exist. The right move is to keep the partner plan's contract, interfaces, methodology and gating, and bolt them onto the claude plan's capability route (pretrained/distilled backbones, Index dev slice, head factorial). Details follow.

---

## 1. Strengths — what it gets right that the claude plan misses or underweights

| # | Strength | Why it matters / evidence | Confidence |
|---|---|---|---|
| S1 | **Information-flow and authority contract stated at the right granularity.** Trusted caller metadata mints authority; NL parsing cannot. Public-only outputs cannot read private recurrent state, private semantic records, *private-dependent normalizers, or caches derived from private input*. Training-time non-interference must account for *shared embeddings, global gradient clipping, joint reward normalization, optimizer state, cache reuse*. State is versioned with model/policy/tokenizer/schema/permissions; "never reuse a cached state under a different permission set"; output-distribution isolation is distinguished from timing/error/resource side channels. | This is the correct list; gradient clipping and Adam moments are real cross-source channels that most "source-separated training" claims ignore. The lab's Oct 1 study needed exactly this discipline (exact public-projection invariance held; the GDN2 extension was *not* bit-exact, diff 8.9e-8). The claude plan's "provenance channel embedding + learned gate" is an empirical mitigation, not a contract, and it says nothing about caches, versions or training-time channels. | High |
| S2 | **Renormalization must not manufacture confidence.** "Expose predictive probabilities and any policy-conditioned distribution distinctly: renormalizing away forbidden actions must not make the model appear more certain." | Matches the lab's decomposition: of the −12.8 pp attacked-input loss on Qwen3-32B, 9.2 pp were *renormalized hijacks* that the mask silently handed to second-best (reviews/03 §1.1). The claude plan states masks are exact and stops there. | High |
| S3 | **Semantic sufficiency P(Y\|X) = P(Y\|C(X)) as the central research question**, with the explicit experiment: learned concepts vs oracle concepts vs richer representation on the same head; expand the interface when expressivity is the limit. | This is the honest replacement for v1's vacuous "100% passes through h_sem" (reviews/02 §2). The claude plan has leakage/completeness tests but no sufficiency framing and no oracle ladder. | High |
| S4 | **Oracle decomposition of failures**: oracle evidence / oracle previous state / oracle programs separate perception, memory-update, discovery, expressivity and realization failures; fix the failing component. | A reusable diagnostic protocol the claude plan lacks. It also encodes a lab lesson: teacher-forced state is a diagnostic only, own-history rollout utility is the endpoint (Travel_1 inherited-state 22%, gate failed; reviews/03 §1.4). | High |
| S5 | **Prospective intervention protocol**: predict the effect of a fact edit / source erasure / relation clamp *before* running it; random and output-only controls; report the unexplained remainder. "Probe accuracy and logit attribution are diagnostics, not the acceptance criterion." | Stronger interpretability standard than the claude plan's "causal probes". It is falsifiable (prediction vs outcome) and includes the control arms reviews/02 §2 asked for. | High |
| S6 | **Typed semantic program algebra with obligations** (Assert/Infer/Compare/Explain/Ask/Quote/Hypothesize/Plan), speech-act semantics, module contracts (pre/postconditions incl. audience, freshness, units, epistemic status), composition-level leakage check (a private-dependent normalizer or search branch can leak even when every field type-checks), bounded proof search with explicit unsupported outcomes. The invariant `Check(K,Π,p) ⇒ Authorized(Π,Effects(p)) ∧ K ⊨ AssertedMeaning(p)` is correctly scoped as a theorem target over a formal language, with "a proof from a false premise remains a proof from a false premise." | The claude P2 (DFA/CNL rung A + claim-decomposition rung B) is the right thing to ship first but has no notion of inference obligations, quotation-without-endorsement, hypotheticals, or cross-module composition. The partner's algebra is the right target for L2+. | High |
| S7 | **Ordinal/score contract read correctly** from the released serializer: score = expected raw level index, confidence = max level probability; the draft's "score ÷ levels + entropy confidence" would have broken wire compatibility. | Verified against reviews/01 Claim 2/7. | High |
| S8 | **Spend gating.** Maximum allocations not estimates; first commitment ≤ $75 (stages 0–1); unlock by inspection; "a failed decision model does not earn $90 of language polishing"; 3-week promotional-credit expiry flagged; preflight cost cap; interactive only for qualification; export minimal bundle. | All SF Compute facts verified today (§6). The claude plan's Stage 0 has per-module kill questions but no sequential unlock rule, and it assumed "$2–2.5/h" where the real rates are $2.97 batch / $3.96 interactive, so its "210–260 H100-h" is really ≈133–177 h. The partner's arithmetic ($525/2.97 = 176.8 h; 9×7 h×2.97 = $187.11) is right. | High |
| S9 | **Honest claim scoping.** "No hallucination" decomposed into type validity / evidence entailment / real-world truth / task correctness; alignment split into structural / statistical / empirical targets; abstention cannot silently alter the forced-choice benchmark; "unanswered = wrong" acknowledged; selected risk R = E[AE]/E[A] with the 2,995-case rule-of-three illustration (n = ln 0.05 / ln 0.999 = 2,994.2 → correct). | High |
| S10 | **ASTRA_REVIEW's mathematical corrections independently agree with reviews/02's numerical checks**: Gramians transform as P′=TPT*, Q′=T⁻*QT⁻¹ (so min-eigenvalues are coordinate-dependent); a constant map is contractive and kills SNGP's lower bound; ECE is unsigned/non-monotone; Hankel rank ≠ reduced-rank-regression singularities; no universal ½-unit LLC jumps (reviews/02 found complex modes jump by 1); the √(log n/8n) threshold does not follow. Two reviewers with different methods converged. | High |
| S11 | **Head B is a genuine fourth head design.** Per-option lightweight selective recurrence over shared backbone token features: `r[t,k] = a[t,k]·r[t-1,k] + b[t,k]·v[t]`, `(a,b) = f(v[t], e[k], q, s)`. Attention-free, permutation-equivariant given the symmetric summary `s`, O(Kd) state, shares the backbone prefill, and the option is known while evidence streams. Cost O(KTd) is stated, not hidden. | It is not in reviews/02 §1's design table (the nearest row, "options-first integrator", makes the *backbone* option-aware; Head B keeps the backbone shared and makes only the head option-aware). It belongs in the claude plan's head factorial. | Medium (untested) |

What would lower these: if the partner's authority contract proves unimplementable without the −12 pp capacity split the lab measured (the plan hypothesizes capacity matching fixes it; that is untested).

---

## 2. Weaknesses

### W1. The capability experiment cannot speak to the capability target (High)

- **What is proposed.** SF525 Stage 2: 50M–100M pure SSM from initialization, 100M–300M tokens per fit, 3 conditions × 3 seeds, $210. PLAN.md ladder: 100M–150M on 1–2B tokens → 300M–400M on 6–8B tokens → "1B–3B or larger successor depends on measured scaling."
- **What the board says.** Every sub-1B official entrant sits on a *pretrained* backbone and scores 6–28 (JPT-0.8B 19.22; Bekko 400M encoder 21.87; Dinah-0 150M 27.63; Julia 141M 5.54). The index measures backbone knowledge first (stock Gemma-4-31B with a temperature readout = 57.33 ≈ Jev). Knowledge & Reasoning is 25.8% of the score and includes GPQA Diamond, MMLU-Pro, HLE.
- **Token arithmetic.** Mamba-3-187M (the *smallest* official checkpoint) was pretrained on 100B tokens. The pilot trains a comparable-size model on 300–1,000× fewer tokens; the $5K ladder's 400M/8B-token model is still 12× fewer tokens than Mamba-3-187M at 2× the parameters. My estimate for such a model on the Index: < 10 (Low confidence on the number, High that it is nowhere near 57).
- **The plan knows this** ("affordable hypothesis-testing scales, not asserted winners") but keeps the competitor list as "the target" of this lineage and allocates $1,800 of the $5K to "shared controlled-backbone pilot and matched training" — the largest line — for a model that cannot approach the target. That is a misallocation relative to the stated goal.
- **What would change my mind:** an Index dev-slice score ≥ 25 from any from-scratch ≤400M model trained on ≤10B tokens.

### W2. "Beat Clef, Clef-flash, Jev, Astra, Fable including difficult knowledge and reasoning tasks" is a scope error, not an ambitious goal (High)

- **What Astra and Fable are** (web-verified today): GPT-6 Astra (OpenAI, released 2026-09-03; $10/$50 per M tokens; GPQA Diamond ≈96.0 per secondary sources) and Claude Fable 5.1 (Anthropic; $10/$50 per M; GPQA Diamond ≈93.7). Clef scores 48.0 on GPQA Diamond; Jev 78.3.
- **They are not on the Decision Index and cannot be.** The board lists 70 open entrants + Jev; none is a frontier API. The harness README (fetched today): "Models whose median latency is over 1,000 ms per request on one RTX PRO 6000 … are not added to the board." TypeSafe's own post gives frontier end-to-end latency as "3 to 329 seconds." So "beat Astra/Fable on the Index" is an off-board comparison the plan would have to construct and pay for.
- **Unbudgeted cost.** Index 0.2.1 = 120,340 requests. The README reports no token-length distribution, so this is an estimate: at 1–2.5K input tokens/request → 120M–300M input tokens → $1.2K–3K per model at $10/M (half at batch rates) plus ≈6–24M output tokens at $50/M ($0.3–1.2K). Two models ≈ **$2–8K**, versus the plan's $700 "independent final evaluation and contingency." A 5K-row stratified slice is ≈4% of that ($80–330 for both) and is the only affordable form. SF525 correctly notes credits do not cover these APIs; PLAN.md never adds them back.
- **Even the honest framing concedes the knowledge axis.** No small model beats a 96-GPQA frontier model on "difficult knowledge and reasoning." The axes that are genuinely open against *all* comparators are the ones TypeSafe markets Jev on (latency, cost, typed-contract reliability) plus calibration (best board ECE 0.014) and steer rate under injection. The plan should say so.
- **What would change my mind:** the Index maintainers adding a latency-exempt frontier reference row; or a measured token distribution showing a full run under ~$500.

### W3. One stale fact that changes Stage 1–2: Mamba-3 checkpoints exist (High)

- PLAN.md: "the official repository's enumerated pretrained checkpoints are principally Mamba-1/2. Do not assume an unverified pretrained Mamba-3 checkpoint exists." EVIDENCE.md repeats it.
- The README indeed lists only Mamba-1/2 models and says Mamba-3 code (`modules/mamba3.py`) must be installed from source (verified today). But Hugging Face hosts `state-spaces/mamba3-{siso,mimo}-{187m,443m,893m,1.5b}` (reviews/01 via HF API; I re-fetched `mamba3-mimo-1.5b`: "1.50 B parameters", "pretrained on 100B tokens from FineWeb-Edu", ctx 2,048, Apache-2.0, "contains no attention layers").
- **Consequence.** The plan's rationale for from-initialization training ("a pretrained opaque backbone with readable head … inception claim is false") loses its forcing function: a pure Mamba-3 backbone with 100B tokens of competence is available at every pilot size. The three-condition experiment can be run *initialized from* Mamba-3-187M/443M, keeping "pure SSM" and gaining language competence the pilot otherwise cannot buy. The structural guarantees (support function, checker, permission-bound cache, versioned state) are weight-independent and hold identically. The only thing that truly needs from-init training is "sparse dictionaries jointly trained from the first step", and the plan offers no evidence that from-init concept learning beats post-hoc at matched capability. The plan's own rule already admits this path: "If broader shared pretraining is used, state that its training corpus is outside that isolation claim."

### W4. It misses the cheapest route to an attention-free Clef (High)

EVIDENCE.md describes Clef as "a prefill-only Qwen-based decision system with attention routing" and never notes that Qwen3.5-9B / Qwen3.8-27B are 3:1 Gated-DeltaNet:attention hybrids (24+8 / 48+16 layers; reviews/01 headline 1), nor that Clef is a frozen backbone + rank-256 LoRA + 128M head. "Attention-free Clef-flash" is therefore an 8-layer conversion problem, not a pretraining problem — the claude plan's Q9-AF and the only route reviews/02 §13 judged plausible under $5K. The partner plan also omits RWKV7-G1k 13.3B (2026-09-30, Apache-2.0), the largest pure attention-free open LM, while listing Mamba-2 2.7B and Falcon-Mamba-7B as references.

### W5. Gates are procedurally falsifiable but numerically empty (Medium-High)

Stage gates read "controlled candidate preserves useful decision capability", "causal edits outperform random/output-only controls", "improvement extends to unseen schemas." PLAN.md says "prespecify any practically negligible screening margin" and "use paired confidence intervals" — correct procedure — but no margin, threshold or power statement appears anywhere, and the lab's own rule is that un-powered percentage gates are "planning targets, not power analyses" (reviews/03 §3). The plan avoided fake-precise gates and delivered none. Before Stage 2 unlock it needs: a non-inferiority margin for B/C vs A; a minimum intervention-agreement gap over the random control; a pre-registered seed-SD report.

Also: the plan makes zero-shot the primary Index number and relegates train-split adaptation to "its own table." The Index is not zero-shot (reviews/01 headline 2); nearly every entrant trains on public train splits. Using the zero-shot number as the go/no-go is a self-imposed handicap.

### W6. 3 conditions × 3 seeds is underpowered for the claim that matters (Medium-High)

- The lab's most comparable study (Oct 1, 3 seeds) gave a 95% interval of [−17.71, −6.51] on a −12.24 pp effect: ±5.6 pp half-width with "substantial" seed variation.
- Two-sample comparison at n = 3 per arm (4 df) needs a standardized effect ≈ 3 × seed SD for 80% power. At a plausible seed SD of 2–3 pp the minimum detectable difference is ≈ 6–9 pp. The plan can therefore detect *large* capability losses from the semantic interface; it cannot certify "no tradeoff" at the 2–3 pp level the user's "no tradeoff" framing implies. Case-level SE on 1–2K held-out items (≈1.3 pp) is not the binding term.
- **Cheap fix.** Per-fit compute is ≈1.8e17 FLOP (100M × 300M × 6) ≈ 1–3 GPU-h including intervention/rollout overhead, not 7 h. Five seeds × 3 conditions ≈ 30–45 h ≈ $90–135, inside the $210 cap. Pre-register the non-inferiority margin and report seed SD.
- **What would change my mind:** measured seed SD < 1 pp on the frozen 1–2K set.

### W7. The custom task does not measure Index-relevant ability (Medium)

The catalog/document assistant (record lookup, comparison, eligibility, corrected facts, explicit policy) resembles the Clef blog's *workflow* evals and the Tools area in spirit (≈18% of Index weight), overlaps the Index literally at 0%, and touches none of Knowledge (25.8%), Language/BANKING77/CLINC (25.8%), or Retrieval/BRIGHT (20%). The plan says this honestly ("capability anchors, not a full Decision Index result"), but it means the $525 produces no evidence about the headline target. The symbolic-parser guard ("if the simple parser solves all natural-language cases, improve the task before training") is correct and the lab's A16/S4 lessons are absorbed — but it is an unbounded loop with no time cap. Fix: add a public-data Index dev slice (free) as a secondary readout for every arm.

### W8. Hidden costs (Medium)

- **Data construction labor**: 20K–50K varied examples + a licensed representation corpus + 1–2K independently reviewed held-out cases with paraphrase-family and entity disjointness + paired necessary-fact cases + disputed-label review. SF credits cannot pay reviewers; PLAN.md's $900 data line is for the $5K scenario, not the pilot. This is weeks of work before the first GPU hour and is nowhere on the 8-week schedule.
- **Engineering surface**: nine new modules (`ssm_backbone`, `semantic_memory`, `schema_encoder`, `ssm_decision`, `semantic_program`, `ssm_language`, `verified_emission`, pilot, eval), a typed program language with checker and realizer, plus a from-source Mamba-3 build. The Mamba-2 fallback is prudent; the qualification allowance ($15 / 3.8 interactive hours) is tight for a from-source CUDA build.
- **Comparator serving**: Clef-flash needs `flash-linear-attention` + `causal-conv1d` or a slow fallback (reviews/04); Clef-27B fits one H100 in bf16 (~55 GB) but only for inference.
- **Astra/Fable API**: see W2.
- **Multimodal gate**: correctly flagged, but low stakes — image items are ≈1,196 of 120,340 rows (≈1%).

### W9. The ASTRA_REVIEW row "pretrained opaque backbone + readable head meets the requirement → false" presents a values choice as an error (Medium)

It is a legitimate preference that the *whole* model be controlled from inception. But the plan's own contract says the guarantees live in the checker, support function and state/cache rules, not in the weights; and reviews/03's evidence says the lab's from-scratch models were matched by a frozen 0.8B backbone + linear head and lost to a regex on data-dependent tasks. Treating "pretrained backbone" as disqualifying buys philosophical purity with the entire capability target. The partner plan's own escape hatch (shared pretraining with the corpus excluded from the isolation claim) should be the default, not the exception.

---

## 3. Claims: wrong / unverified / overstated — and correct guarantees (credit)

### 3.1 Wrong or stale
| Claim | Status | Correction |
|---|---|---|
| "Do not assume an unverified pretrained Mamba-3 checkpoint exists" | **Wrong** (stale) | `state-spaces/mamba3-{siso,mimo}-{187m,443m,893m,1.5b}`, Apache-2.0, 100B tokens, no attention. |
| Relative links `../../overnight/LATEST_PILOT_RESULTS_20261001.md`, `../../chl-structured-models/core/native_agency.py` | **Broken in this repo** | From `plans/partner/` they resolve to `State-Space-Decision-Model/overnight/…` (absent). Real files: `C:\Users\apumu\research\chlabs\overnight\LATEST_PILOT_RESULTS_20261001.md`, `C:\Users\apumu\research\chlabs\chl-structured-models\core\native_agency.py` (both exist). Receipts should resolve. |
| ASTRA_REVIEW: BANKING77/CLINC wins differ from "zero-shot general evaluation" | **Half wrong** | The Index is not zero-shot; train-split fine-tuning is normal. The valid point is that 36 other benchmarks must also work. |
| Clef described without its 75% Gated-DeltaNet backbone | **Material omission** | See W4. |

### 3.2 Overstated or unbudgeted
- "Target Clef, Clef-flash, Jev, Astra, Fable … retain difficult knowledge, reasoning" — see W2.
- Acceptance criterion "positive full-index improvement over every comparator" — well-formed, unattainable at this budget; no open-axis definition.
- Evaluation matrix lists "Full frozen Decision Index" with no comparator-API line.

### 3.3 Verified correct (credit where due)
- SF Compute: H100 "$0.066/min ($3.96/hr)" interactive; batch "$0.0495/min per GPU" = $2.97/h; "Promotional credit expires 3 weeks after it is granted"; managed researcher "bills intelligence units … `intelligence-1` at $0.01 per unit"; preflight "adds about 2 seconds and it is free" with a workspace-cap refusal; checkpoint capture is a slot ("Each capture replaces the last"), one per 5 min; Friday 33% / weekend 20% discounts exist ("calendar discounts may improve this"). All arithmetic in SF525 checks.
- Decision Index 0.2.1: 38 counted benchmarks, revised weights, no truncation/option filtering, unanswered = wrong — matches reviews/01.
- Clef cards/serializer: 27B/9B backbones, joint head, score = expected level index, confidence = max probability, 16,384 default max length, 64k context — matches reviews/01.
- Vendor table (BANKING77 94.2/90.9/79.7; CLINC 97.4/66.8/89.3; GPQA 48.0/51.0/78.3; latency 209.3/38.8/524.1) — matches reviews/01 Claim 8, correctly labelled "vendor reported."
- Jev architecture undisclosed; "do not classify Jev as a known transformer or PGM" — correct (reviews/01 headline 7).
- Mamba-in-the-Llama's strongest result keeps ~25% attention — correct (fetched today).
- Citations resolve: CaMeL (Debenedetti et al.), CB-LLMs (Sun et al.), Monet (Park et al.), concept leakage = Havasi, Parbhoo, Doshi-Velez NeurIPS 2022, Learn-then-Test, Decision Hijacking (Wu & Lim: 1.8% base / 3.5% adaptive per reviews/04).
- Lab figures: "21 Qwen retrains and 12 scratch models" = 21 fits + 9 attention + 3 recurrent models in the Oct 1 file; −12.24 pp [−17.71, −6.51]; symbolic baseline 100% — all match.
- FLOP sanity: 6ND for 300M/6B = 1.08e19 → 20–60 GPU-h at 50–150 TFLOP/s — correct.
- Guarantees stated correctly: the support-function invariant is scoped to the formal language; "a proof from a false premise remains a proof from a false premise"; grammar × trie is "insufficient by itself" (copied values can form a false relation) — matches reviews/02 §11; selected-risk denominator; conformal singleton caveat; abstention counts as wrong.

---

## 4. Complementarity with the claude plan, and the strongest combined shots on goal

**Where they differ (noted, not re-reviewed):** the claude plan converts Clef-flash's 8 attention layers to GDN (Q9-AF), runs RWKV7-13.3B and Falcon-Mamba-7B arms, a 3-head × 2-backbone factorial with Clef's head as control, an Index dev slice per area × length bin, the lab's steer/hijack harness as a gate, and a glass-box conversion map; its P2 is DFA/CNL realizer + claim decomposition. It under-prices SF Compute (~$2–2.5/h assumed) and has no information-flow contract, no oracle ladder, no semantic program algebra, and no sequential spend-unlock rule.

**Combination principle:** the partner plan's contract and methodology are backbone-agnostic; the claude plan's backbones are contract-agnostic. Put the partner's interfaces on the claude's backbones. Under the partner's own rule, the pretraining corpus sits outside the isolation claim and everything else is enforced structurally.

### Shots on goal (ordered by expected leverage), with SF credits first

| # | Shot | From | Budget | Kill question |
|---|---|---|---|---|
| 1 | **Harness + stratified Index dev slice + Clef-flash parity + record-length distribution** | claude 0.1 | ~$15 | Does our scorer reproduce Clef-flash per-benchmark? What is the real token volume (also prices W2)? |
| 2 | **Q9-AF attention-removal pilot** (layer-wise swap of 8 attention layers → GDN, ~200M tokens), scored per area × length bin | claude 0.4 | ~$150 (at $2.97) | Does knowledge survive removing attention? Which layers cost? This is the only path to >57 under $5K. |
| 3 | **Head pilot on the better backbone: integrator / final-state fork / snapshot fork / Clef-head control + partner Head B** | claude 0.5 + partner S11 | ~$120 | Does an attention-free head close the retrieval/tools gap? |
| 4 | **Partner's 3-condition experiment, initialized from Mamba-3-443M (and one from-init arm for attribution), 5 seeds, on the partner's catalog task + the Index dev slice; partner's authority/IF contract on arms B/C; prospective intervention protocol with random/output-only controls** | partner Stage 2 + W3/W6 fixes | ~$130 | Does the semantic interface preserve capability within a pre-registered margin? Do predicted interventions land? This is the glass-box end of the Pareto map, at honest scale. |
| 5 | **Steer/hijack baseline on all backbones** (chl-typed-control D1 + GCG-lite), reporting predictive vs policy-conditioned distributions separately | claude 0.3 + partner S2 | ~$20 | Are recurrent backbones more or less steerable? Is renormalized hijack being hidden? |
| 6 | **Partner L0/L1 language on the surviving checkpoint** (Assert/Compare/Explain/Ask via checked templates, then licensed paraphrase), scored with the claude plan's claim-decomposition for rung B | partner Stage 4 + claude P2 | ~$60 | Does a checked program interface extend output without losing decision accuracy? |
| — | Reserve, governed by the partner's unlock rule (first commitment ≤ $75 = shots 1 + 5 + half of 2) | partner S8 | remainder | |

Then the $5K: Q27-AF stretch and the full-suite run only if shot 2 passes; Astra/Fable only on the 5K-row slice (~$100–300 total) as an off-board reference, never as the headline comparator; the partner's $1,800 from-scratch backbone line is cut to the one attribution arm in shot 4.

**What each plan should take from the other, in one line each.** Claude ← partner: the information-flow/versioning contract (S1), expose both distributions (S2), sufficiency + oracle ladder (S3–S4), prospective-intervention protocol (S5), program algebra as the L2+ target (S6), sequential spend gating and the correct SF rates (S8), Head B (S11). Partner ← claude: Clef is 75% GDN so convert rather than pretrain (W4); pretrained pure-SSM inits incl. Mamba-3 and RWKV7 (W3); an Index dev slice per area × length as the primary capability readout (W7); numeric gates with margins and ≥5 seeds (W5–W6); drop Astra/Fable from the capability target and budget them as a slice-only reference (W2).

---

## 5. What would most change this review

1. A from-scratch ≤400M pure SSM scoring ≥25 on a stratified Index slice (reopens W1).
2. Seed SD < 1 pp on the partner's frozen set (weakens W6).
3. A Q9-AF pilot that loses >5 points on Knowledge after converting 8 layers (would raise the partner's "no cheap route" implicit position and push toward a hybrid fallback for both plans).
4. Evidence that from-init concept dictionaries give materially better intervention agreement than post-hoc dictionaries at matched capability (would justify the partner's from-init commitment on its own terms).

---

## 6. Checks performed today (2026-10-06)

- https://autoresearch.sfcompute.com/ — "$0.066/min ($3.96/hr)" H100; 8-GPU "$0.528/min ($31.68/hr)"; batch "$0.0495/min per GPU … 25% below interactive"; CPU-2 $0.18/h, CPU-8 $0.48/h.
- https://autoresearch.sfcompute.com/docs — "Promotional credit expires 3 weeks after it is granted"; purchased credit does not expire; "The managed researcher bills intelligence units at its tier price … `intelligence-1` at $0.01 per unit"; "Preflight runs on every submission. It adds about 2 seconds and it is free"; checkpoint slot "Each capture replaces the last … one per 5 minutes per job"; automatic top-up thresholds; `max_duration` per job.
- https://huggingface.co/state-spaces/mamba3-mimo-1.5b — 1.50B params, 100B FineWeb-Edu tokens, ctx 2,048, Apache-2.0, "contains no attention layers." (Full size list via reviews/01: siso/mimo × 187m/443m/893m/1.5b.)
- https://github.com/state-spaces/mamba — Mamba-3 block at `modules/mamba3.py`; install from source; README lists only Mamba-1/2 pretrained models. Issue #860 ("Will Mamba 3 be released on Hugging Face?", 2026-03-20) unanswered — the checkpoints appeared later.
- https://raw.githubusercontent.com/apolinario/decision-index/main/README.md — ">1,000 ms median … not added to the board"; rules: no truncation, no option filtering, no prompt tuning, unanswered = wrong, linked cases stay whole; no token-length distribution published.
- Web search — Astra = GPT-6 Astra (OpenAI, 2026-09-03; $10/$50 per M; GPQA Diamond ≈96.0, secondary sources); Fable = Claude Fable 5.1 ($10/$50 per M; batch $5/$25; GPQA Diamond ≈93.7, secondary). Not on the Decision Index.
- arXiv 2503.18813 (CaMeL), 2412.07992 (CB-LLMs), 2412.04139 (Monet), 2408.15237 (Mamba in the Llama; "a quarter of the attention layers" retained) — titles/authors/claims as cited. OpenReview `tglniD_fn9` = Havasi, Parbhoo, Doshi-Velez, "Addressing leakage in concept bottleneck models," NeurIPS 2022 (via search; OpenReview blocked direct fetch).
- Local: `C:\Users\apumu\research\chlabs\overnight\LATEST_PILOT_RESULTS_20261001.md` lines 11, 26, 58 confirm "21 Qwen retrains", "nine attention models … three exploratory recurrent-encoder models", "12.24 … [-17.71, -6.51]", symbolic 100%. `core/native_agency.py` and `core/native_register_dialogue.py` exist under `chlabs\chl-structured-models\`. No `overnight/` directory exists under `State-Space-Decision-Model\` (links broken).
