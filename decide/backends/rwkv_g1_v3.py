"""jrt-rwkv-g1-v3 / -v3r / -v3c readout backends (R3): v2 + long-catalog binding, optionally + a top-m re-read or
contextual calibration.

Opt-in readout families, one change each on top of v3 so every gain is attributable:
  ``rwkv7-hf-g1-v3``   mode "catalog"    v2 + the v3 code table + the code key (``decide.prompt_g1v3.render_g1v3``)
  ``rwkv7-hf-g1-v3p``  mode "permute"    v3 + deterministic per-question presentation order (codes mapped back)
  ``rwkv7-hf-g1-v3r``  mode "reread"     v3 + a second answer slot over the model's own top-m first-pass codes
  ``rwkv7-hf-g1-v3c``  mode "calibrate"  v3 + contextual calibration
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
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable

from ..adapter import DEFAULT_LIMITS
from ..prompt import CANARY_REQUEST, G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE
from ..prompt_g1v3 import (CALIBRATE_MAX_K, CATALOG_CHARS_LADDER, CATALOG_MIN_K, CONTENT_FREE_INPUTS, G1V3_ID,
                           G1V3C_ID, G1V3P_ID, G1V3R_ID, REREAD_LABEL_CHARS, REREAD_MIN_K, RESTATE_CHARS, WORDLIKE,
                           content_free_request, identity_order, load_v3_codes, presentation_order, question_codes,
                           render_g1v3, reread_m, reread_suffix, wants_catalog)
from ..schema import canonical_json, canonical_options, canonical_question_ids, sha256_text
from .base import BackendUnsupported, Plan, ScoreResult
from .hf_readout import sha256_file
from .rwkv_g1_filtered_codes import TokenCode
from .rwkv_g1_v2 import BOS_ID, G1V2Backend
from .state_fork import PAD_ID, common_prefix_len

CF_MEMO_MAX = 4096
MODES = {"catalog": (G1V3_ID, "rwkv_g1_v3"), "permute": (G1V3P_ID, "rwkv_g1_v3p"), "reread": (G1V3R_ID, "rwkv_g1_v3r"),
         "calibrate": (G1V3C_ID, "rwkv_g1_v3c")}
V3_CODES, V3_TABLE_SHA256 = load_v3_codes()
V3_TABLE = [e["code"] for e in V3_CODES]


def renderer_identity(mode: str) -> dict[str, str]:
    table = V3_TABLE
    tid = MODES[mode][0]
    canary = {qid: render_g1v3(CANARY_REQUEST, qid, table, permute=mode == "permute")
              for qid in canonical_question_ids(CANARY_REQUEST["questions"])}
    value = {"id": tid, "head": G1V2_TEMPLATE.head, "tail": G1V2_TEMPLATE.tail, "system": G1V2_SYSTEM_PROMPT,
             "bos": BOS_ID, "v3_code_table_sha256": V3_TABLE_SHA256, "wordlike_excluded": sorted(WORDLIKE),
             "canary": canary, "permute": mode == "permute",
             "catalog": {"min_k": CATALOG_MIN_K, "chars_ladder": list(CATALOG_CHARS_LADDER), "restate_chars": RESTATE_CHARS},
             "reread": ({"min_k": REREAD_MIN_K, "m": "2 if k == 3 else 3", "label_chars": REREAD_LABEL_CHARS,
                         "suffix_canary": reread_suffix({"type": "choice", "criteria": {"a": "x", "b": None, "c": "z"}},
                                                        [2, 0], table)}
                        if mode == "reread" else None),
             "calibration": ({"kind": "contextual (Zhao et al. 2021), diag(p_cf)^-1", "content_free": list(CONTENT_FREE_INPUTS),
                              "max_k": CALIBRATE_MAX_K} if mode == "calibrate" else None)}
    return {"id": tid, "sha256": sha256_text(canonical_json(value))}


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
        self.name = MODES[mode][1]
        limit = DEFAULT_LIMITS["max_input_tokens"]
        ctx = plugin.max_context or limit
        self.budget = (budget if budget is not None else min(ctx, limit)) - 1  # 1 = the answer position
        self._cf_memo: dict[str, dict[str, dict[str, float]]] = {}
        self.cf_stats = {"memo_hits": 0, "memo_misses": 0}
        super().__init__(plugin, **kw)
        # the v3 code table replaces v2's frozen one (v2's init proved its own table; prove this one the same way)
        self.codes = tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i) for i, e in enumerate(V3_CODES))
        self.code_table = list(V3_TABLE)
        self.code_token = {c.code: c.token_id for c in self.codes}
        ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"
        base = plugin.encode(ctx, True)
        for c in self.codes:
            if plugin.encode(ctx + " " + c.code, True) != base + [c.token_id]:
                raise ValueError(f"v3 code {c.code!r} is not one token after the v2 tail")

    def _build_revision(self) -> dict[str, Any]:
        revision = super()._build_revision()
        config = dict(revision["inference_config"])
        here = Path(__file__)
        config.update({
            "template_variant": MODES[self.mode][0],
            "codebook_table_sha256": None, "v3_code_table_sha256": V3_TABLE_SHA256,
            "presentation": "sha256(question id, type, option ids) order; codes mapped back" if self.permute else "canonical",
            "catalog": f"code key for coded k >= {CATALOG_MIN_K}; label chars ladder {list(CATALOG_CHARS_LADDER)} "
                       f"under a {self.budget}-token budget",
            "reread": (f"k >= {REREAD_MIN_K}: model's own first-pass top-m (m = 2 if k == 3 else 3) re-listed after "
                       f"the first answer context, second pass from the same prefix state, p = p1 outside S, "
                       f"p1(S) * p2(.|S) inside") if self.reread else None,
            "calibration": (f"contextual: log softmax(log p - log mean_cf softmax(log p_cf)), cf inputs "
                            f"{list(CONTENT_FREE_INPUTS)}, k <= {CALIBRATE_MAX_K}") if self.calibrate else None,
            "source_sha256": {**config.get("source_sha256", {}),
                              **{p.name: sha256_file(p) for p in (here, here.parents[1] / "prompt_g1v3.py")}},
        })
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
    def _render(self, request: dict[str, Any], qid: str) -> tuple[str, list[int], int | None]:
        """(text, token ids, label chars used; None = no block) of the longest-label v3 prompt within the budget."""
        ladder = CATALOG_CHARS_LADDER if wants_catalog(request["questions"][qid], self.code_table) else (None,)
        text, ids = "", []
        for n in ladder:
            text = render_g1v3(request, qid, self.code_table, n, self.permute)
            ids = self._full(text)
            if len(ids) <= self.budget:
                return text, ids, n
        return text, ids, None  # v2 text, over budget: the adapter refuses it exactly as it refuses v2

    def _questions(self, request: dict[str, Any], label_counts: dict[str, int] | None = None):
        out = []
        for qid in canonical_question_ids(request["questions"]):
            text, prompt, n = self._render(request, qid)
            if label_counts is not None:
                label_counts["total"] = label_counts.get("total", 0) + len(prompt)
            rq = request["questions"][qid]
            opts = [o for o, _ in canonical_options(rq)]
            coded = len(opts) <= len(self.codes)
            order = presentation_order(rq, qid) if self.permute else identity_order(rq)
            q = {"qid": qid, "text": text, "prompt": prompt, "options": opts, "coded": coded, "catalog_chars": n,
                 "catalog_wanted": wants_catalog(rq, self.code_table)}
            if coded:
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
                    worst = max((self._full(q["text"] + reread_suffix(rq, list(range(i, i + m)), q["codes"]))
                                 for i in range(len(q["options"]) - m + 1)), key=len)
                    longest = max(longest, len(worst) + 1)
                    extra += len(worst) - L
                calls += max(1, -(-extra // self.pack_tokens), -(-len(qs) // self.pack_rows))
                total += extra
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

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        if self.reread:
            return self._score_reread(request, check_deadline)
        res = super().score(request, check_deadline)
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
