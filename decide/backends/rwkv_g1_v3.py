"""jrt-rwkv-g1-v3 / -v3r / -v3c readout backends (R3): v2 + long-catalog binding, optionally + a top-m re-read or
contextual calibration.

Opt-in readout families, one change each on top of v3 so every gain is attributable (every mode renders
dict-valued instructions as canonical JSON, the v2 instruction fix; ``rwkv7-hf-g1-v2f`` = v2 + that fix only):
  ``rwkv7-hf-g1-v2f``  mode "v2f"        v2 (frozen v2 code table, no code key) + the instruction fix
  ``rwkv7-hf-g1-v3w``  mode "words"      v3 + word option ids scored by their own unique first token
  ``rwkv7-hf-g1-v3``   mode "catalog"    v2 + the v3 code table + the code key (``decide.prompt_g1v3.render_g1v3``)
  ``rwkv7-hf-g1-v3p``  mode "permute"    v3 + deterministic per-question presentation order (codes mapped back)
  ``rwkv7-hf-g1-v3r``  mode "reread"     v3 + a second answer slot over the model's own top-m first-pass codes
  ``rwkv7-hf-g1-v3c``  mode "calibrate"  v3 + contextual calibration
  ``rwkv7-hf-g1-v3j``  mode "joint"      v3 + the joint candidate-block decode (``decide.joint``, score-only, R4 (b))
  ``rwkv7-hf-g1-v3js`` mode "joint_seq"  v3j + sequential own-answer conditioning (R4 (a), extra model calls)
Scoring is the v2 path: [0] + whole-string tokenization, the shared token prefix of every question prompt prefilled
once, one single-token code read per question from packed tail extensions of that state.

Context budget: a question whose v3 prompt is longer than ``budget`` tokens (min(plugin.max_context, the adapter's
max_input_tokens) - 1 answer position) is re-rendered with a shorter catalog label (CATALOG_CHARS_LADDER) and, as the
last resort, without the catalog block (the exact v2 text). The step used is reported per question.

v3r re-read: for a coded question with k >= REREAD_MIN_K, the shortlist S = the m codes with the highest FIRST-PASS
log-probabilities of this very request (m = reread_m(k)); nothing else (no teacher, no data field, no gold, no other
model) can enter S, and the backend asserts it. The second-pass prompt is the first-pass prompt text + reread_suffix
(S re-listed with option text, then "FINAL ANSWER:"); it is scored from the SAME prefilled prefix state (only its tail
is extended, no full re-run), reading log p(code | second-pass prompt) for the codes in S only. The reported
distribution keeps the first-pass mass outside S and redistributes the first-pass mass of S by the second pass:
    p(o) = p1(o)                       o not in S
    p(o) = p1(S) * p2(o | S)           o in S
so it is a proper distribution over all k options (ECE stays meaningful). Logits are log p(o).

v3c contextual calibration (Zhao et al. 2021): for each coded question with k <= CALIBRATE_MAX_K, the content-free
prior p_cf(o) = mean over CONTENT_FREE_INPUTS of softmax_o(log p(code_o | content-free prompt)), where the content-free
prompt is the same v3 template with the state and every instruction replaced by the filler (options kept). The
reported logit is log softmax_o(log p(code_o | prompt) - log p_cf(o)) (W = diag(p_cf)^-1, b = 0). Nothing is fitted:
the prior depends only on the request's own label space, never on gold labels or other rows. Identical content-free
prompts (same schema across rows) are served from an in-process memo, which changes cost, not values.

v3j joint decode: the v3 logits of a request that is a candidate block (``decide.joint.candidate_block``: n >= 2 yes/no
questions over one instruction stem) are replaced by the log max-marginals of the joint under #yes >= 1; every other
request is returned with the v3 logits unchanged. No extra model call.

v3js sequential conditioning: easy-first over the request's coded questions. The first pass is v3; questions are
ordered by their first-pass max probability (ties by id); the first keeps its v3 logits, and question t is re-scored
from the SAME prefilled prefix state with ``answers_so_far`` (the model's own argmax answers to questions 1..t-1 of
this request; nothing else) inserted between END STATE and the re-read question. One extra prefill + one tail call per
re-scored question. The v3j joint decode is applied afterwards.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable

from ..adapter import DEFAULT_LIMITS
from ..joint import JOINT_RULE_ID, joint_logits
from ..prompt import CANARY_REQUEST, G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE
from ..prompt_g1v3 import (CALIBRATE_MAX_K, CATALOG_CHARS_LADDER, CATALOG_MIN_K, CONTENT_FREE_INPUTS, G1V3_ID,
                           G1V3C_ID, G1V3J_ID, G1V3JS_ID, G1V3P_ID, G1V3R_ID, REREAD_LABEL_CHARS, REREAD_MIN_K,
                           RESTATE_CHARS, WORDLIKE, G1V2F_ID, G1V3W_ID, answers_so_far, catalog_label, content_free_request, identity_order, load_v3_codes,
                           presentation_order, question_codes, render_g1v3, reread_m, reread_suffix, wants_catalog,
                           word_ids_ok)
from ..schema import canonical_json, canonical_options, canonical_question_ids, sha256_text
from .base import BackendUnsupported, Plan, ScoreResult
from .hf_readout import sha256_file
from .rwkv_g1_filtered_codes import TABLE_SHA256 as V2_TABLE_SHA256, TokenCode, load_frozen_table
from .rwkv_g1_v2 import BOS_ID, G1V2Backend
from .state_fork import PAD_ID, common_prefix_len

CF_MEMO_MAX = 4096
MODES = {"v2f": (G1V2F_ID, "rwkv_g1_v2f"), "words": (G1V3W_ID, "rwkv_g1_v3w"), "catalog": (G1V3_ID, "rwkv_g1_v3"), "permute": (G1V3P_ID, "rwkv_g1_v3p"), "reread": (G1V3R_ID, "rwkv_g1_v3r"),
         "calibrate": (G1V3C_ID, "rwkv_g1_v3c"), "joint": (G1V3J_ID, "rwkv_g1_v3j"),
         "joint_seq": (G1V3JS_ID, "rwkv_g1_v3js")}
JOINT_MODES = ("joint", "joint_seq")
V3_CODES, V3_TABLE_SHA256 = load_v3_codes()
V3_TABLE = [e["code"] for e in V3_CODES]
ANSWER_CTX = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"


def _table(mode: str) -> list[str]:
    return [c.code for c in load_frozen_table()] if mode == "v2f" else V3_TABLE


def renderer_identity(mode: str) -> dict[str, str]:
    table = _table(mode)
    tid = MODES[mode][0]
    canary = {qid: render_g1v3(CANARY_REQUEST, qid, table, None if mode == "v2f" else 64, permute=mode == "permute")
              for qid in canonical_question_ids(CANARY_REQUEST["questions"])}
    value = {"id": tid, "head": G1V2_TEMPLATE.head, "tail": G1V2_TEMPLATE.tail, "system": G1V2_SYSTEM_PROMPT,
             "bos": BOS_ID, "code_table_sha256": V2_TABLE_SHA256 if mode == "v2f" else V3_TABLE_SHA256,
             "wordlike_excluded": None if mode == "v2f" else sorted(WORDLIKE), "instruction_fix": True,
             "word_ids": "non-opaque ids, unique first token after ANSWER:" if mode == "words" else None,
             "canary": canary, "permute": mode == "permute",
             "catalog": {"min_k": CATALOG_MIN_K, "chars_ladder": list(CATALOG_CHARS_LADDER), "restate_chars": RESTATE_CHARS},
             "reread": ({"min_k": REREAD_MIN_K, "m": "2 if k == 3 else 3", "label_chars": REREAD_LABEL_CHARS,
                         "suffix_canary": reread_suffix({"type": "choice", "criteria": {"a": "x", "b": None, "c": "z"}},
                                                        [2, 0], table)}
                        if mode == "reread" else None),
             "calibration": ({"kind": "contextual (Zhao et al. 2021), diag(p_cf)^-1", "content_free": list(CONTENT_FREE_INPUTS),
                              "max_k": CALIBRATE_MAX_K} if mode == "calibrate" else None)}
    if mode in JOINT_MODES:  # keys only for the R4 modes: every pre-R4 identity hash is unchanged
        value["joint"] = {"rule": JOINT_RULE_ID, "decode": "map", "output": "log max-marginals"}
        value["sequential"] = ({"order": "easy-first (first-pass max p, ties by id)", "block": "answers_so_far",
                                "canary": _answers_canary()} if mode == "joint_seq" else None)
    return {"id": tid, "sha256": sha256_text(canonical_json(value))}


def _answers_canary() -> str:
    qids = canonical_question_ids(CANARY_REQUEST["questions"])
    return answers_so_far(CANARY_REQUEST, [(q, canonical_options(CANARY_REQUEST["questions"][q])[0][0]) for q in qids])


def _norm(d: dict[str, float]) -> dict[str, float]:
    m = max(d.values())
    z = sum(math.exp(v - m) for v in d.values())
    return {o: math.exp(v - m) / z for o, v in d.items()}


def calibrate_logits(logp: dict[str, float], priors: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """(normalized log p(o)/p_cf(o) over the options, p_cf). ``logp`` and each prior entry are log-probabilities of
    the option codes (any per-question constant cancels); p_cf is the mean of the priors' per-option softmaxes."""
    opts = list(logp)
    ps = [_norm({o: p[o] for o in opts}) for p in priors]
    p_cf = {o: sum(p[o] for p in ps) / len(ps) for o in opts}
    raw = {o: logp[o] - math.log(max(p_cf[o], 1e-300)) for o in opts}
    m = max(raw.values())
    lz = m + math.log(sum(math.exp(v - m) for v in raw.values()))
    return {o: raw[o] - lz for o in opts}, p_cf


def shortlist(first_pass: dict[str, float], options: list[str], m: int) -> list[int]:
    """Indices (canonical option order) of the m highest first-pass scores; ties broken by canonical order. The ONLY
    input is the model's own first-pass score dict for this question."""
    order = sorted(range(len(options)), key=lambda j: (-first_pass[options[j]], j))
    return sorted(order[:m])


def combine_reread(first_pass: dict[str, float], second_pass: dict[str, float]) -> dict[str, float]:
    """log p over all options: first-pass mass outside the shortlist kept, the shortlist's first-pass mass
    redistributed by the second pass (``second_pass`` = log-probs of the shortlisted options only)."""
    p1 = _norm(first_pass)
    p2 = _norm(second_pass)
    mass = sum(p1[o] for o in second_pass)
    out = {o: (mass * p2[o] if o in second_pass else p1[o]) for o in first_pass}
    return {o: math.log(max(v, 1e-300)) for o, v in out.items()}


class G1V3Backend(G1V2Backend):
    name = "rwkv_g1_v3"

    def __init__(self, plugin, mode: str = "catalog", budget: int | None = None, **kw) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        self.mode = mode
        self.calibrate = mode == "calibrate"
        self.reread = mode == "reread"
        self.permute = mode == "permute"
        self.words = mode == "words"
        self.joint = mode in JOINT_MODES
        self.sequential = mode == "joint_seq"
        self._first_tok: dict[str, int | None] = {}
        self.name = MODES[mode][1]
        limit = DEFAULT_LIMITS["max_input_tokens"]
        ctx = plugin.max_context or limit
        self.budget = (budget if budget is not None else min(ctx, limit)) - 1  # 1 = the answer position
        self._cf_memo: dict[str, dict[str, dict[str, float]]] = {}
        self.cf_stats = {"memo_hits": 0, "memo_misses": 0}
        super().__init__(plugin, **kw)
        if mode != "v2f":  # the v3 code table replaces v2's frozen one (proved below as v2's init proved its own)
            self.codes = tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i)
                               for i, e in enumerate(V3_CODES))
            self.code_table = list(V3_TABLE)
        self.code_token = {c.code: c.token_id for c in self.codes}
        ctx = ANSWER_CTX
        self._ctx_ids = base = plugin.encode(ctx, True)
        for c in self.codes:
            if plugin.encode(ctx + " " + c.code, True) != base + [c.token_id]:
                raise ValueError(f"v3 code {c.code!r} is not one token after the v2 tail")

    def _build_revision(self) -> dict[str, Any]:
        revision = super()._build_revision()
        config = dict(revision["inference_config"])
        here = Path(__file__)
        config.update({
            "template_variant": MODES[self.mode][0],
            "codebook_table_sha256": V2_TABLE_SHA256 if self.mode == "v2f" else None,
            "v3_code_table_sha256": None if self.mode == "v2f" else V3_TABLE_SHA256,
            "instruction_fix": "non-string instructions as canonical JSON (v1's render)",
            "word_ids": "first token of ' <id>' when ids are non-opaque and first tokens unique" if self.words else None,
            "presentation": "sha256(question id, type, option ids) order; codes mapped back" if self.permute else "canonical",
            "catalog": None if self.mode == "v2f" else (
                f"code key for coded k >= {CATALOG_MIN_K}; label chars ladder {list(CATALOG_CHARS_LADDER)} "
                f"under a {self.budget}-token budget"),
            "reread": (f"k >= {REREAD_MIN_K}: model's own first-pass top-m (m = 2 if k == 3 else 3) re-listed after "
                       f"the first answer context, second pass from the same prefix state, p = p1 outside S, "
                       f"p1(S) * p2(.|S) inside") if self.reread else None,
            "calibration": (f"contextual: log softmax(log p - log mean_cf softmax(log p_cf)), cf inputs "
                            f"{list(CONTENT_FREE_INPUTS)}, k <= {CALIBRATE_MAX_K}") if self.calibrate else None,
            "source_sha256": {**config.get("source_sha256", {}),
                              **{p.name: sha256_file(p) for p in (here, here.parents[1] / "prompt_g1v3.py")}},
        })
        if self.joint:  # keys only for the R4 modes
            config["joint"] = (f"{JOINT_RULE_ID}: candidate block (n >= 2 yes/no questions, one instruction stem) -> "
                               f"joint MAP under #yes >= 1, reported as log max-marginals; other requests unchanged")
            config["sequential"] = ("easy-first own-answer conditioning: question t re-scored from the shared prefix "
                                    "state with the model's own argmax answers to questions 1..t-1 between END STATE "
                                    "and the re-read question") if self.sequential else None
        cfg_hash = sha256_text(canonical_json(config))
        tmpl = renderer_identity(self.mode)
        revision.update({
            "backend": self.name, "prompt_template": tmpl, "inference_config": config, "inference_config_sha256": cfg_hash,
            "revision_id": (f"{self.plugin.family}:{self.plugin.model_id}@{(self.plugin.snapshot or 'local')[:12]}"
                            f";w={self.plugin.files.get('weights', 'unknown')[:12]}"
                            f";tok={self.plugin.files.get('tokenizer', 'unknown')[:12]}"
                            f";prompt={tmpl['id']}.{tmpl['sha256'][:8]};cfg={cfg_hash[:8]}"
                            f";adapter={revision['adapter_version']}"),
        })
        return revision

    def revision(self) -> dict[str, Any]:
        return copy.deepcopy(self._revision)

    # ---------------------------------------------------------------- prompts
    def first_token(self, oid: str) -> int | None:
        """The answer token of " <oid>" right after "ANSWER:" (None if the tokenization does not extend the context)."""
        if oid not in self._first_tok:
            ids = self.plugin.encode(ANSWER_CTX + " " + oid, True)
            n = len(self._ctx_ids)
            self._first_tok[oid] = ids[n] if len(ids) > n and ids[:n] == self._ctx_ids and ids[n] != BOS_ID else None
        return self._first_tok[oid]

    def word_qids(self, request: dict[str, Any]) -> frozenset[str]:
        if not self.words:
            return frozenset()
        return frozenset(k for k in canonical_question_ids(request["questions"])
                         if word_ids_ok(request["questions"][k], self.first_token))

    def _render(self, request: dict[str, Any], qid: str, words: frozenset[str] = frozenset(), block: str | None = None):
        """(text, token ids, label chars used; None = no block) of the longest-label v3 prompt within the budget;
        ``block`` = the v3js answers-so-far block (None: the v3 prompt)."""
        rq = request["questions"][qid]
        ladder = CATALOG_CHARS_LADDER if (self.mode != "v2f" and qid not in words
                                          and wants_catalog(rq, self.code_table)) else (None,)
        text, ids = "", []
        for n in ladder:
            text = render_g1v3(request, qid, self.code_table, n, self.permute, True, words, block)
            ids = self._full(text)
            if len(ids) <= self.budget:
                return text, ids, n
        return text, ids, None  # v2 text, over budget: the adapter refuses it exactly as it refuses v2

    def _questions(self, request: dict[str, Any], label_counts: dict[str, int] | None = None):
        out = []
        words = self.word_qids(request)
        for qid in canonical_question_ids(request["questions"]):
            text, prompt, n = self._render(request, qid, words)
            if label_counts is not None:
                label_counts["total"] = label_counts.get("total", 0) + len(prompt)
            rq = request["questions"][qid]
            opts = [o for o, _ in canonical_options(rq)]
            coded = len(opts) <= len(self.codes)
            order = presentation_order(rq, qid) if self.permute else identity_order(rq)
            q = {"qid": qid, "text": text, "prompt": prompt, "options": opts, "coded": coded, "catalog_chars": n,
                 "catalog_wanted": self.mode != "v2f" and qid not in words and wants_catalog(rq, self.code_table),
                 "words": qid in words}
            if qid in words:  # v3w: each option scored by the (unique) first token of its own id
                q["coded"], q["codes"] = True, None
                q["tokens"] = [self.first_token(o) for o in opts]
            elif coded:
                # codes[j] = the code shown for CANONICAL option j, so every score maps back to its option id
                q["codes"] = question_codes(rq, order, self.code_table)
                q["tokens"] = [self.code_token[c] for c in q["codes"]]
            else:
                q["conts"] = [self.plugin.encode(" " + o + G1V2_TEMPLATE.terminator, True) for o in opts]
            out.append(q)
        return out

    def _calibrated_qids(self, request: dict[str, Any]) -> list[str]:
        out = []
        for qid in canonical_question_ids(request["questions"]):
            k = len(canonical_options(request["questions"][qid]))
            if 2 <= k <= min(CALIBRATE_MAX_K, len(self.codes)):
                out.append(qid)
        return out

    def plan(self, request: dict[str, Any]) -> Plan:
        p = super().plan(request)
        calls, longest, total = p.inference_calls, p.max_call_input_tokens, p.total_input_tokens
        if self.calibrate and self._calibrated_qids(request):
            for f in CONTENT_FREE_INPUTS:  # upper bound: the memo can only remove calls
                cp = G1V2Backend.plan(self, content_free_request(request, f))
                calls += cp.inference_calls
                longest = max(longest, cp.max_call_input_tokens)
                total += cp.total_input_tokens
        if self.reread:  # upper bound: the longest possible suffix (labels at full length) for every eligible question
            qs = [q for q in self._questions(request) if q["coded"] and reread_m(len(q["options"]))]
            if qs:
                L = common_prefix_len([q["prompt"] for q in self._questions(request) if q["coded"]])
                extra = 0
                for q in qs:
                    m = reread_m(len(q["options"]))
                    rq = request["questions"][q["qid"]]
                    opts = canonical_options(rq)  # the m longest labels bound the suffix (+8 tokens slack)
                    longest_m = sorted(range(len(opts)), key=lambda j: -len(catalog_label(*opts[j], REREAD_LABEL_CHARS)))[:m]
                    worst = len(self._full(q["text"] + reread_suffix(rq, longest_m, q["codes"]))) + 8
                    longest = max(longest, worst + 1)
                    extra += worst - L
                calls += max(1, -(-extra // self.pack_tokens), -(-len(qs) // self.pack_rows))
                total += extra
        if self.sequential:  # upper bound: every other coded question answered with its longest option label
            qs = [q for q in self._questions(request) if q["coded"]]
            if len(qs) >= 2:
                worst = []
                for q in qs:
                    opts = dict(canonical_options(request["questions"][q["qid"]]))
                    worst.append((q["qid"], max(sorted(opts), key=lambda o: len(catalog_label(o, opts[o], 64)))))
                seq = [self._render(request, q["qid"], block=answers_so_far(request, [w for w in worst if w[0] != q["qid"]]))[1]
                       for q in qs]
                L = common_prefix_len([q["prompt"] for q in qs] + seq)
                extra = 0
                for ids in seq:
                    longest = max(longest, len(ids) + 1)
                    extra += len(ids) - L + 8
                calls += len(qs)  # one prefill + one tail call per re-scored question (all but the first)
                total += L + extra
        return Plan(inference_calls=calls, max_call_input_tokens=longest, total_input_tokens=total,
                    per_question=p.per_question, prompt_token_labels=p.prompt_token_labels, provenance_applied=False)

    # ---------------------------------------------------------------- scoring
    def _extend_coded(self, state, L: int, items, check_deadline):
        """items: [(key, ids, token_ids)] whose ids share ``state``'s L-token prefix. Packs their tails exactly as v2
        and returns ({key: log p of each token id}, calls, tokens, call ms)."""
        import torch

        ops, dev = self.plugin.state_ops, self.plugin.device
        tails = sorted(((it, it[1][L:]) for it in items), key=lambda t: (-len(t[1]), t[0][0]))
        packs, cur = [], []
        for t in tails:
            if cur and (len(cur) >= self.pack_rows or len(cur[0][1]) * (len(cur) + 1) > self.pack_tokens):
                packs.append(cur)
                cur = []
            cur.append(t)
        if cur:
            packs.append(cur)
        out, ms, total = {}, [], 0
        for pack in packs:
            check_deadline()
            B, W = len(pack), len(pack[0][1])
            ids = torch.full((B, W), PAD_ID, dtype=torch.long)
            last = torch.zeros(B, dtype=torch.long)
            for b, (_, tail) in enumerate(pack):
                ids[b, : len(tail)] = torch.tensor(tail, dtype=torch.long)
                last[b] = len(tail) - 1
            t0 = time.perf_counter()
            with torch.inference_mode():
                lg, _ = ops.extend(state, ids.to(dev), W)
                sel = lg[torch.arange(B, device=lg.device), last.to(lg.device)]
                if not bool(torch.isfinite(sel).all()):
                    raise BackendUnsupported("non_finite_logits", "nonfinite next-token logits")
                lp = torch.log_softmax(sel.float(), dim=-1)
                for b, ((key, _, toks), _) in enumerate(pack):
                    out[key] = lp[b].index_select(0, torch.tensor(toks, device=lp.device)).tolist()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            total += sum(len(t) for _, t in pack)
        return out, len(packs), total, ms

    def _score_reread(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        """First pass exactly as v3 (shared prefix state, one code read per question), then the second pass over the
        model's own first-pass top-m from the SAME prefix state."""
        import torch

        from .state_fork import fork_score

        ops, dev = self.plugin.state_ops, self.plugin.device
        qs = self._questions(request)
        coded = [q for q in qs if q["coded"]]
        logits: dict[str, dict[str, float]] = {}
        ms: list[float] = []
        total, calls = 0, 0
        extras: dict[str, Any] = {"shortlist": {}, "first_pass_logits": {}, "reread_skipped": {}}
        if coded:
            L = common_prefix_len([q["prompt"] for q in coded])
            check_deadline()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _, state = ops.prefill(torch.tensor([coded[0]["prompt"][:L]], dtype=torch.long, device=dev))
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            calls, total = 1, L
            first, c, t, m = self._extend_coded(state, L, [(q["qid"], q["prompt"], q["tokens"]) for q in coded],
                                                check_deadline)
            calls, total, ms = calls + c, total + t, ms + m
            items, short = [], {}
            for q in coded:
                fp = dict(zip(q["options"], first[q["qid"]]))
                logits[q["qid"]] = fp
                k = len(q["options"])
                mm = reread_m(k)
                if not mm:
                    continue
                # Candidates come ONLY from this request's own first-pass scores (fp): never a teacher, data field,
                # gold label or another model. The assertion re-derives the shortlist from fp alone.
                s_idx = shortlist(fp, q["options"], mm)
                top = sorted(fp.values(), reverse=True)[:mm]
                assert len(s_idx) == mm and all(fp[q["options"][j]] >= top[-1] for j in s_idx), "shortlist not first-pass top-m"
                ids2 = self._full(q["text"] + reread_suffix(request["questions"][q["qid"]], s_idx, q["codes"]))
                if len(ids2) > self.budget:
                    extras["reread_skipped"][q["qid"]] = "over_budget"
                    continue
                if ids2[:L] != q["prompt"][:L] or len(ids2) <= L:
                    extras["reread_skipped"][q["qid"]] = "prefix_mismatch"
                    continue
                short[q["qid"]] = s_idx
                items.append((q["qid"], ids2, [q["tokens"][j] for j in s_idx]))
            if items:
                second, c, t, m = self._extend_coded(state, L, items, check_deadline)
                calls, total, ms = calls + c, total + t, ms + m
                for qid, s_idx in short.items():
                    q = next(x for x in coded if x["qid"] == qid)
                    sp = {q["options"][j]: v for j, v in zip(s_idx, second[qid])}
                    extras["first_pass_logits"][qid] = logits[qid]
                    extras["shortlist"][qid] = [q["options"][j] for j in s_idx]
                    logits[qid] = combine_reread(logits[qid], sp)
                    if not all(math.isfinite(v) for v in logits[qid].values()):
                        raise BackendUnsupported("non_finite_logits", "nonfinite re-read logits")
        for q in qs:  # fallback: multi-token option ids (k > codebook), as v2
            if q["coded"]:
                continue
            rows = [{"qid": q["qid"], "option": o, "prompt": q["prompt"], "cont": c}
                    for o, c in zip(q["options"], q["conts"])]
            lg, n, tok, m = fork_score(rows, ops, self.pack_rows, dev, check_deadline, self.pack_tokens, None)
            logits[q["qid"]] = lg[q["qid"]]
            ms += m
            total += tok
            calls += len(m)
        extras.update({"call_latency_ms": ms, "scoring_path": "g1-v3r-prefix-fork",
                       "catalog_chars": {q["qid"]: q["catalog_chars"] for q in qs if q["catalog_wanted"]},
                       "coded": {q["qid"]: q["coded"] for q in qs}})
        return ScoreResult(logits=logits, inference_calls=len(ms), total_input_tokens=total, extras=extras)

    def _cf_logits(self, request: dict[str, Any], filler: str, check_deadline: Callable[[], None]):
        cf = content_free_request(request, filler)
        key = hashlib.sha256(canonical_json({"q": cf["questions"], "s": cf["state"]}).encode("utf-8")).hexdigest()
        hit = self._cf_memo.get(key)
        if hit is not None:
            self.cf_stats["memo_hits"] += 1
            return hit, 0, 0, []
        self.cf_stats["memo_misses"] += 1
        r = G1V2Backend.score(self, cf, check_deadline)
        if len(self._cf_memo) >= CF_MEMO_MAX:
            self._cf_memo.clear()
        self._cf_memo[key] = r.logits
        return r.logits, r.inference_calls, r.total_input_tokens, r.extras.get("call_latency_ms", [])

    def _score_sequential(self, request: dict[str, Any], first: dict[str, dict[str, float]],
                          check_deadline: Callable[[], None]):
        """v3js: (logits, calls, tokens, call ms, diagnostics). ``first`` = this request's v3 logits."""
        import torch

        qs = [q for q in self._questions(request) if q["coded"]]
        diag: dict[str, Any] = {"order": [], "skipped": {}, "first_pass_logits": {}}
        if len(qs) < 2:
            return first, 0, 0, [], diag
        by_id = {q["qid"]: q for q in qs}

        def pmax(lg):
            m = max(lg.values())
            return 1.0 / sum(math.exp(v - m) for v in lg.values())

        def argmax(lg):
            return max(sorted(lg), key=lambda o: lg[o])  # ties -> smallest id, as decide.schema

        order = sorted(by_id, key=lambda k: (-pmax(first[k]), k))
        diag["order"] = order
        chosen = [(order[0], argmax(first[order[0]]))]
        # the shared prefix ends where the answers block starts (every block opens with the same header)
        probe = self._render(request, order[1], block=answers_so_far(request, chosen))[1]
        L = common_prefix_len([q["prompt"] for q in qs] + [probe])
        prefix = qs[0]["prompt"][:L]
        ops, dev = self.plugin.state_ops, self.plugin.device
        check_deadline()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _, state = ops.prefill(torch.tensor([prefix], dtype=torch.long, device=dev))
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ms, calls, total = [round((time.perf_counter() - t0) * 1000.0, 3)], 1, L
        logits = dict(first)
        for qid in order[1:]:
            q = by_id[qid]
            _, ids, _ = self._render(request, qid, block=answers_so_far(request, chosen))
            if len(ids) > self.budget:
                diag["skipped"][qid] = "over_budget"
            elif ids[:L] != prefix or len(ids) <= L:
                diag["skipped"][qid] = "prefix_mismatch"
            else:
                out, c, t, m = self._extend_coded(state, L, [(qid, ids, q["tokens"])], check_deadline)
                calls, total, ms = calls + c, total + t, ms + m
                diag["first_pass_logits"][qid] = first[qid]
                logits[qid] = dict(zip(q["options"], out[qid]))
                if not all(math.isfinite(v) for v in logits[qid].values()):
                    raise BackendUnsupported("non_finite_logits", "nonfinite sequential logits")
            chosen.append((qid, argmax(logits[qid])))
        return logits, calls, total, ms, diag

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        if self.reread:
            return self._score_reread(request, check_deadline)
        res = super().score(request, check_deadline)
        if self.joint:
            extras = dict(res.extras)
            logits, calls, total = res.logits, res.inference_calls, res.total_input_tokens
            ms = list(extras.get("call_latency_ms", []))
            if self.sequential:
                logits, c, t, m, extras["sequential"] = self._score_sequential(request, logits, check_deadline)
                calls, total, ms = calls + c, total + t, ms + m
            joint, extras["joint"] = joint_logits(request, logits, "map")
            if extras["joint"]["applied"]:
                extras["independent_logits"] = {qid: logits[qid] for qid in extras["joint"]["changed"]}
            extras.update({"call_latency_ms": ms,
                           "scoring_path": "g1-v3js-prefix-fork" if self.sequential else "g1-v3j-prefix-fork"})
            return ScoreResult(logits=joint, inference_calls=calls, total_input_tokens=total, extras=extras)
        qs = {q["qid"]: q for q in self._questions(request)}
        extras = dict(res.extras)
        extras["scoring_path"] = "g1-v3c-prefix-fork" if self.calibrate else "g1-v3-prefix-fork"
        extras["catalog_chars"] = {qid: q["catalog_chars"] for qid, q in qs.items() if q["catalog_wanted"]}
        if not self.calibrate:
            return ScoreResult(logits=res.logits, inference_calls=res.inference_calls,
                               total_input_tokens=res.total_input_tokens, extras=extras)
        targets = self._calibrated_qids(request)
        logits = dict(res.logits)
        calls, total, ms = res.inference_calls, res.total_input_tokens, list(extras.get("call_latency_ms", []))
        priors: dict[str, list[dict[str, float]]] = {qid: [] for qid in targets}
        if targets:
            for f in CONTENT_FREE_INPUTS:
                lg, c, t, m = self._cf_logits(request, f, check_deadline)
                calls, total, ms = calls + c, total + t, ms + m
                for qid in targets:
                    priors[qid].append(lg[qid])
        cal = {}
        for qid in targets:
            logits[qid], p_cf = calibrate_logits(res.logits[qid], priors[qid])
            if not all(math.isfinite(v) for v in logits[qid].values()):
                raise BackendUnsupported("non_finite_logits", "nonfinite calibrated logits")
            cal[qid] = {o: round(v, 6) for o, v in p_cf.items()}
        extras.update({"call_latency_ms": ms, "calibration_prior": cal, "cf_memo": dict(self.cf_stats),
                       "uncalibrated_logits": {qid: res.logits[qid] for qid in targets}})
        return ScoreResult(logits=logits, inference_calls=calls, total_input_tokens=total, extras=extras)
