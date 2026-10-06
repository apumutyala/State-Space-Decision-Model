"""Optional controlsafe arms (G4 snapshot reuse, G14 monitor, G15 trust gate, persistence, Gramians).

    python -m safety_eval.controlsafe_arms --out receipts/<dir> [--arms g15,persistence,gramian,g4snap,g14]
    python -m safety_eval.run_guarantees_smoke --out <dir> --controlsafe     (same arms, embedded in that receipt)

CPU only, tiny RANDOM backbones (vendored RWKV-7 and all-GDN Q9-AF layouts, ``heads.backbone.tiny_hf_backbone``)
and the byte encoder: PIPELINE QUALIFICATION, never model evidence (PREREG section 0). The G4 snapshot arm runs
on the real dev-split request generator (``safety_eval.data``) through the real layout tokenisation, so its
token/call savings are a property of the request shapes and k, not of the random weights; its decisions are
verified identical to the C(n, k) full-recompute baseline on every origin.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

ARMS = ("g15", "persistence", "gramian", "g4snap", "g14", "g16")
FAMILIES = ("rwkv7", "q9af")


def _ids(n: int, seed: int) -> list[int]:
    import torch

    return torch.randint(4, 260, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def arm_g15(fam: str, seed: int = 0) -> dict[str, Any]:
    import torch

    from controlsafe import gramian, trust_gate
    from heads.backbone import tiny_hf_backbone

    g = trust_gate.TrustGatedBackbone(tiny_hf_backbone(fam, seed=seed), bridge_rank=4)
    with torch.no_grad():
        for p in g.bridges.parameters():  # an arbitrary trained-like bridge (the guarantee is weight-independent)
            p.normal_(0, 0.3)
    ids = _ids(48, seed)
    trust = [True] * 12 + [False] * 24 + [True] * 12
    r = trust_gate.reachability_check(g, ids, trust, n_trials=16, seed=seed)
    r["protected_gramian_fro"] = gramian.protected_gramian(g, ids, trust)
    g.tokens_processed = 0
    with torch.no_grad():
        g.features(torch.tensor([ids]), None, torch.tensor([trust]))
    r["compute_multiplier"] = trust_gate.compute_multiplier(g, len(ids))
    return r


def arm_persistence(fam: str, seed: int = 0) -> dict[str, Any]:
    import torch

    from controlsafe import adapters, persistence
    from heads.backbone import tiny_hf_backbone

    m = tiny_hf_backbone(fam, seed=seed).model.double()
    ids = _ids(40, seed)
    batch = torch.tensor([_ids(40, seed + i) for i in range(8)])
    local = {l: persistence.layer_local_curve(m, ids, l, t=5, horizon=30, eps=1e-2, n_dirs=4)
             for l in adapters.recurrent_layer_ids(m)}
    e2e = persistence.end_to_end_curve(m, ids, t=5, horizon=30, n_dirs=4)
    pick = lambda v: {tau: v[tau] for tau in (0, 1, 5, 10, 20, 30) if tau < len(v)}  # noqa: E731
    return {"universal": adapters.universal_decay_bound(m),
            "observed": persistence.observed_transition_norms(m, batch),
            "local": {l: {"rho_obs": v["rho_obs"], "bound_holds": v["bound_holds"], "window": v["window"],
                          "state_influence_max": pick(v["state_influence_max"]),
                          "bound_product": pick(v["bound_product"]), "status": v["status"]} for l, v in local.items()},
            "end_to_end_final_feature": pick(e2e["hidden_influence"][max(e2e["hidden_influence"])]),
            "end_to_end_state": {l: pick(v) for l, v in e2e["state_influence"].items()},
            "end_to_end_status": e2e["status"]}


def arm_gramian(fam: str, seed: int = 0) -> dict[str, Any]:
    from controlsafe import gramian
    from heads.backbone import tiny_hf_backbone

    m = tiny_hf_backbone(fam, seed=seed).model.double()
    ids = _ids(32, seed)
    unt = [False] * 8 + [True] * 16 + [False] * 8
    out = {}
    for mode in ("end_to_end", "local"):
        res = gramian.gramians(m, ids, unt, mode=mode)
        out[mode] = {l: {k: v[k] for k in ("trace", "eff_rank_entropy", "participation_ratio", "numerical_rank",
                                           "dim", "readout_overlap", "isotropic_overlap", "status") if k in v}
                     for l, v in res.items()}
    return out


def arm_g4snap(fam: str, seed: int = 0, n_route: int = 6, n_pay: int = 4, ks=(1, 2)) -> dict[str, Any]:
    from controlsafe import adapters, snapshot
    from heads import ByteEncoder
    from heads.backbone import tiny_hf_backbone
    from heads.layout import build_row

    from . import data

    enc = ByteEncoder()
    reqs = [data.strip_meta(r) for r in data.build(n_route, n_pay, split="dev")]
    m = tiny_hf_backbone(fam, seed=seed).model
    readout = snapshot.random_readout(m.config.hidden_size, 4, seed=seed)
    out: dict[str, Any] = {}
    for k in ks:
        rows = []
        for r in reqs:
            qid = sorted(r["questions"])[0]
            o = snapshot.request_origins(r, k, lambda q: build_row(q, qid, enc).ids)
            rec = snapshot.snapshot_certify(adapters.Runner(m), readout, o, verify=True)
            rows.append(rec)
        tot = lambda key: sum(x[key] for x in rows)  # noqa: E731
        out[f"k{k}"] = {
            "n_requests": len(rows), "n_origins": tot("n_origins"),
            "decisions_identical_all": all(x["decisions_identical"] for x in rows),
            "max_abs_logit_diff": max(x["max_abs_logit_diff"] for x in rows),
            "baseline_tokens": tot("baseline_tokens"), "snapshot_tokens": tot("snapshot_tokens"),
            "token_savings": 1 - tot("snapshot_tokens") / tot("baseline_tokens"),
            "baseline_calls": tot("baseline_calls"), "snapshot_calls": tot("snapshot_calls"),
            "forward_equivalents_baseline_mean": tot("forward_equivalents_baseline") / len(rows),
            "forward_equivalents_snapshot_mean": tot("forward_equivalents_snapshot") / len(rows),
            "max_states_held": max(x["max_states_held"] for x in rows),
        }
    return out


G14_SEEDS = 10


def arm_g14(fam: str, seed: int = 0, n_seeds: int = G14_SEEDS) -> dict[str, Any]:
    """Seed distribution (seeds ``seed .. seed + n_seeds - 1``, model + trigger jointly), never one seed."""
    from controlsafe import monitor
    from heads.backbone import tiny_hf_backbone

    def make(sd):
        m = tiny_hf_backbone(fam, seed=sd).model
        monitor.long_memory_(m, seed=sd)
        return m

    return monitor.seed_sweep(make, range(seed, seed + n_seeds), n=24)


def arm_g16(fam: str, seed: int = 0, n_route: int = 6, n_pay: int = 4, ks=(1, 2)) -> dict[str, Any]:
    """G16 on the real dev-split request generator: random (seeded) judge and evidence weights on a tiny
    long-memory backbone. Reports the certificate mix (independent / G4-over-E certified / escalated), the radii,
    and the cost (origins evaluated, P reused exactly). Every origin is verified against a full G16 recompute."""
    import torch

    from controlsafe import g16, monitor, trust_gate
    from heads import ByteEncoder
    from heads.backbone import tiny_hf_backbone
    from heads.layout import build_row

    from . import data

    enc = ByteEncoder()
    bb = tiny_hf_backbone(fam, seed=seed)
    monitor.long_memory_(bb.model, seed=seed)
    head = g16.G16Head(trust_gate.TrustGatedBackbone(bb, bridge_rank=4), r=8, c=1.0, max_options=16, seed=seed)
    gen = torch.Generator().manual_seed(seed + 17)
    with torch.no_grad():
        for p in (head.A.weight, head.w.weight, head.u, head.beta, head.evidence.W_e.weight):
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    reqs = [data.strip_meta(r) for r in data.build(n_route, n_pay, split="dev")]
    out: dict[str, Any] = {"status": "PIPELINE QUALIFICATION (random judge); certificate soundness is the tested part"}
    for k in ks:
        rows = []
        for r in reqs:
            qid = sorted(r["questions"])[0]
            rows.append(g16.certify(head, r, qid, k, lambda q: build_row(q, qid, enc), verify=True))
        n = len(rows)
        out[f"k{k}"] = {
            "n_requests": n,
            "independent_rate": sum(x["independent_of_untrusted"] for x in rows) / n,
            "certified_rate": sum(x["certified"] for x in rows) / n,
            "origins_evaluated": sum(x["n_origins_evaluated"] for x in rows),
            "p_cache_misses": sum(x["p_cache_misses"] for x in rows),
            "rho_joint_l2_median": float(torch.tensor([x["rho_joint_l2"] for x in rows]).median()),
            "verified_against_full_recompute": True,
        }
    return out


FUNCS = {"g15": arm_g15, "persistence": arm_persistence, "gramian": arm_gramian, "g4snap": arm_g4snap, "g14": arm_g14,
         "g16": arm_g16}


def run_arms(arms=ARMS, families=FAMILIES, seed: int = 0) -> dict[str, Any]:
    import torch

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    res: dict[str, Any] = {"meta": {"status": "PIPELINE QUALIFICATION on tiny random backbones (CPU); not model evidence",
                                    "torch": torch.__version__, "python": platform.python_version(),
                                    "seed": seed, "arms": list(arms), "families": list(families)}}
    for arm in arms:
        res[arm] = {}
        for fam in families:
            t = time.perf_counter()
            res[arm][fam] = FUNCS[arm](fam, seed)
            res[arm][fam]["wall_s"] = round(time.perf_counter() - t, 2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    res = run_arms([x for x in a.arms.split(",") if x], [x for x in a.families.split(",") if x], a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "controlsafe.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps({arm: {f: {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                            for f, r in res[arm].items()} for arm in res if arm != "meta"}, indent=1, default=float))


if __name__ == "__main__":
    main()
