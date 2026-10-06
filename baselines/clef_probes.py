"""Pre-flight and difference probes for the Clef baseline job (stage S1 of jobs/clef_baseline/JOB_SPEC.md).

1. parity   decide-path raw probabilities vs Cloudflare's own ``systemone`` on N DEV rows whose ssdm and
            native encodings are identical and fit ``max_length``: max |dp| <= 2e-4 (systemone rounds to 4
            decimals) and identical argmax except within 1e-3 ties. Failure -> exit 11 (nothing else runs).
2. encodings over ALL DEV rows (CPU-only tokenization): rows whose native and ssdm ids differ (control or
            added-token strings in request text), rows Clef would truncate, rows over 255 options. These
            are exactly the rows the native Index arm must run on; every other row is identical by
            construction (ids equal => same forward).
3. G8       an over-length request (state of ~1.5x max_length) and a 256-option request: Clef's native
            status/usage vs decide's status.
4. G2(b)    the safety_eval chatml-spoof string: control ids produced inside Clef's sequence, native vs ssdm.
5. kernel   fast GDN kernels (fla chunk_gated_delta_rule + causal-conv1d, the path the image runs) vs
            transformers' own torch reference (``clef_backend.reference_gdn_path``) on up to N_KERNEL parity
            rows of <= KERNEL_MAX_TOKENS tokens: max |dp| <= KERNEL_TOL and no argmax flip where the margin
            exceeds KERNEL_MARGIN. This is the in-image evidence that Clef scores the same on our pinned stack
            (torch 2.13 / triton 3.7.1 / fla 0.5.2, off the model card's torch 2.11). Failure -> exit 11.

    python -X utf8 -m baselines.clef_probes --model-dir /models/clef-flash --release clef-flash \
        --dev-rows /data/slices/dev-rows.jsonl.gz --out /out/probes.json --n-parity 64
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

from . import clef_backend as CB

N_KERNEL = 16
KERNEL_MAX_TOKENS = 2048  # the torch reference GDN is O(T) python-chunked and slow; short rows suffice
KERNEL_TOL = 2e-2         # bf16 kernel-to-kernel numerics on probabilities (pre-registered)
KERNEL_MARGIN = 2e-2      # an argmax may only differ where the top-2 gap is below this


def kernel_path_check(decider, model, rows, release, max_tokens=KERNEL_MAX_TOKENS, n=N_KERNEL) -> dict:
    """Fast-kernel vs torch-reference GDN on the same requests through the same decide path.

    Non-vacuous by construction (PR #43 B1): ``ok`` needs (a) at least one GDN layer whose fast kernel object
    is NOT the torch reference (transformers silently falls back when fla is unavailable, in which case the
    "fast" arm already is the reference), and (b) a measured difference ``max_abs_dp > 0`` (two different
    kernels in bf16 never agree bit-for-bit on every probability; exact equality means the switch did not
    reach the forward), plus the tolerance and argmax rules.
    """
    picked, worst, flips = [], 0.0, []
    found = already_ref = switched = 0
    for r in rows:
        if len(picked) >= n:
            break
        req = {"model": release, "state": r["state"], "questions": r["questions"]}
        fast = decider.decide(req)["ssdm"]
        if fast["status"] != "ok" or fast["diagnostics"]["input_counts"]["max_call_input_tokens"] > max_tokens:
            continue
        ctx = CB.reference_gdn_path(model)
        with ctx:
            ref = decider.decide(req)["ssdm"]
        found, already_ref, switched = ctx.n_found, ctx.n_already_reference, ctx.n_switched
        picked.append(r["_evaluation"]["run_id"])
        for qid, pf in fast["raw_probabilities"].items():
            pr = ref["raw_probabilities"][qid]
            worst = max(worst, max(abs(pf[k] - pr[k]) for k in pf))
            top = sorted(pf.values())
            if len(top) > 1 and top[-1] - top[-2] > KERNEL_MARGIN and max(pf, key=pf.get) != max(pr, key=pr.get):
                flips.append([r["_evaluation"]["run_id"], qid])
    reasons = []
    if not picked:
        reasons.append("no eligible row")
    if switched == 0:
        reasons.append(f"no fast GDN kernel in use ({found} GDN layers, {already_ref} already on the torch reference)")
    if picked and worst == 0.0:
        reasons.append("max_abs_dp == 0: the reference switch did not change the forward")
    if worst > KERNEL_TOL:
        reasons.append(f"max_abs_dp {worst:.3g} > tol {KERNEL_TOL}")
    if flips:
        reasons.append(f"{len(flips)} argmax flips at margin > {KERNEL_MARGIN}")
    return {"n": len(picked), "run_ids": picked, "gdn_layers_found": found,
            "gdn_layers_already_reference": already_ref, "gdn_layers_switched": switched,
            "max_abs_dp": worst, "argmax_flips": flips, "tol": KERNEL_TOL, "margin": KERNEL_MARGIN,
            "fail_reasons": reasons, "ok": not reasons}


def read_rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--release", choices=sorted(CB.RELEASES), default="clef-flash")
    ap.add_argument("--dev-rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-parity", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--grad-probe", action="store_true", help="one GCG-lite gradient step at HP_GATE size (memory)")
    a = ap.parse_args(argv)
    try:
        backend, model, processor = CB.load_clef(a.model_dir, a.release, device=a.device, verify=not a.no_verify)
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD/VERIFY FAILED: {exc}", flush=True)
        return 10
    jsm, tok = backend.jsm, processor.tokenizer
    native_enc = CB.ClefEncoder(tok, jsm, "clef_native")
    decider = CB.build_decider(backend)
    out: dict = {"release": a.release, "revision": backend.revision(), "files_sha256": backend.files_sha256}

    # 2. encodings over every DEV row (tokenizer only). The native arm must cover EVERY row where the decide
    #    path does not run the same forward Clef would: ids differ, Clef truncates, our encoder refuses, or the
    #    adapter rejects before scoring (validate_legacy: >255 options, 1-level score, noul criteria keys, ids
    #    with control chars / '<|' '|>'; or DEFAULT_LIMITS max_questions / max_options_per_question).
    from decide.adapter import DEFAULT_LIMITS
    from decide.schema import RequestRejected, canonical_options, validate_legacy

    rows = list(read_rows(a.dev_rows))
    differ, trunc, refused, over255, same, rejected = [], [], [], [], [], []
    for n_done, r in enumerate(rows, 1):
        if n_done % 1000 == 0:
            print(f"[probes] encodings {n_done}/{len(rows)}", flush=True)
        rid = r["_evaluation"]["run_id"]
        req = {"model": a.release, "state": r["state"], "questions": r["questions"]}
        try:  # same order and codes as Decider.decide (contract 0.1)
            validate_legacy(req)
            if len(req["questions"]) > DEFAULT_LIMITS["max_questions"]:
                raise RequestRejected("exhausted", "questions_exceeded", "too many questions")
            if max(len(canonical_options(q)) for q in req["questions"].values()) > DEFAULT_LIMITS["max_options_per_question"]:
                raise RequestRejected("exhausted", "options_exceeded", "too many options")
        except RequestRejected as exc:
            rejected.append({"run_id": rid, "status": exc.status, "code": exc.code})
        try:
            nat = native_enc.parts(req)
        except Exception as exc:  # noqa: BLE001  (Clef's own encode_record would fail the same way)
            differ.append(rid)
            out.setdefault("native_encode_failed", []).append({"run_id": rid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if any(len(q.option_ids) > 255 for q in nat.questions):
            over255.append(rid)
        if nat.length() > backend.max_length:
            trunc.append({"run_id": rid, "full": nat.length(), "dropped_state_tokens": nat.length() - backend.max_length})
        try:
            ss = backend.encoder.parts(req)
        except Exception:  # noqa: BLE001  (fail-closed)
            refused.append(rid)
            continue
        ids_n = native_enc.assemble(nat).input_ids
        ids_s = backend.encoder.assemble(ss).input_ids
        rej = bool(rejected) and rejected[-1]["run_id"] == rid
        (same if ids_n == ids_s and nat.length() <= backend.max_length and not rej else differ).append(rid)
    trunc_ids = {t["run_id"] for t in trunc}
    rej_ids = {x["run_id"] for x in rejected}
    out["encodings"] = {"rows": len(rows), "identical_and_fit": len(same),
                        "ids_differ": sorted(set(differ) - trunc_ids - rej_ids),
                        "clef_would_truncate": trunc, "ssdm_refused_control_tokens": refused,
                        "over_255_options": over255, "decide_rejected_before_scoring": rejected}
    native_rows = sorted(set(differ) | trunc_ids | set(refused) | set(over255) | rej_ids)
    out["native_arm_rule"] = ("union of: ids differ, Clef truncates, ssdm encoder refuses, >255 options, adapter "
                              "rejects before scoring (validate_legacy / max_questions / max_options_per_question); every other row runs identical "
                              "ids through the identical forward, so its native output equals the decide output")
    rng = random.Random(a.seed)
    parity_ids = sorted(rng.sample(same, min(a.n_parity, len(same))))
    out["native_arm_run_ids"] = native_rows
    out["parity_sample_run_ids"] = parity_ids

    # 1. parity
    by_id = {r["_evaluation"]["run_id"]: r for r in rows}
    worst, argmax_mismatch = 0.0, []
    for rid in parity_ids:
        r = by_id[rid]
        req = {"model": a.release, "state": r["state"], "questions": r["questions"]}
        ours = decider.decide(req)["ssdm"]
        theirs, _ = CB.native_systemone(jsm, model, processor, req, backend.max_length)
        for qid, ans in theirs["answers"].items():
            pr = ours["raw_probabilities"][qid]
            tp = ans["probabilities"] if ans["type"] != "noul" else {"true": ans["noul"]}
            for k, p in tp.items():
                worst = max(worst, abs(pr[k] - p))
            if ans["type"] == "choice":
                top = sorted(pr.values())
                if len(top) > 1 and top[-1] - top[-2] > 1e-3 and max(pr, key=pr.get) != ans["choice"]:
                    argmax_mismatch.append([rid, qid])
    out["parity"] = {"n": len(parity_ids), "max_abs_dp": worst, "argmax_mismatch": argmax_mismatch,
                     "ok": worst <= 2e-4 and not argmax_mismatch}

    # 5. fast GDN kernels vs torch reference (in-image "scores the same" check)
    out["kernel_path"] = kernel_path_check(decider, model, [by_id[r] for r in parity_ids], a.release)

    # 3. G8 probes
    long_req = {"model": a.release, "state": "lorem ipsum " * int(backend.max_length * 0.75),
                "questions": {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "A", "b": "B"}}}}
    many_req = {"model": a.release, "state": "pick",
                "questions": {"q": {"type": "choice", "instructions": "Pick one.",
                                    "criteria": {f"o{i:03d}": f"option {i}" for i in range(256)}}}}
    g8 = {}
    for name, req in (("over_length", long_req), ("options_256", many_req)):
        ours = decider.decide(req)["ssdm"]
        try:
            theirs, meta = CB.native_systemone(jsm, model, processor, req, backend.max_length)
            nat = {"answered": bool(theirs["answers"]), "usage": theirs["usage"], **meta}
        except Exception as exc:  # noqa: BLE001
            nat = {"raised": f"{type(exc).__name__}: {exc}"}
        g8[name] = {"decide_status": ours["status"], "decide_error": ours["error"], "clef_native": nat}
    out["g8"] = g8

    # 4. G2(b) control ids from the chatml spoof
    from safety_eval.attacks import HIJACK_TEMPLATES

    spoof = HIJACK_TEMPLATES["chatml_spoof"].format(attacker="escalate", attacker_desc="Escalate.")
    req = {"model": a.release, "state": {"email": spoof}, "questions": long_req["questions"]}
    _, meta = CB.native_systemone(jsm, model, processor, req, backend.max_length)
    try:
        hard = backend.encoder.parts(req)
        from decide.untrusted import control_ids_and_contents

        ctrl, _ = control_ids_and_contents(tok)
        ssdm_ctrl = sum(i in ctrl for i in hard.state_ids + hard.schema_ids)
    except Exception as exc:  # noqa: BLE001
        ssdm_ctrl = f"refused: {exc}"
    out["g2b_chatml_spoof"] = {"clef_native_control_ids_from_request": meta["control_ids_from_request"],
                               "ssdm_control_ids_from_request": ssdm_ctrl}
    # 5. memory: peak after a max_length forward (G8 probe above) and, optionally, one GCG gradient step
    import torch

    if a.grad_probe:
        from safety_eval import data, gcg

        from . import clef_gcg

        r = data.build(1, 0, split="dev")[0]
        eng = clef_gcg.ClefGCGEngine(backend)
        prob = clef_gcg.ClefProblem(backend, r, "steer")
        init = tok(gcg.HP_GATE["init"], add_special_tokens=False).input_ids
        adv = torch.tensor(init * gcg.HP_GATE["n_adv"])
        eng.loss_grad([prob], adv)
        eng.eval_advs(prob, adv.unsqueeze(0).repeat(eng.eval_batch, 1))
    if torch.cuda.is_available():
        out["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["device_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"parity": out["parity"], "kernel_path_ok": out["kernel_path"]["ok"],
                      "native_arm_rows": len(native_rows), "g8": {k: v["decide_status"] for k, v in g8.items()}}), flush=True)
    return 0 if out["parity"]["ok"] and out["kernel_path"]["ok"] else 11


if __name__ == "__main__":
    sys.exit(main())
