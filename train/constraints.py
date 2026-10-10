"""Constrained training as feedback control: nuPI-controlled Lagrange multipliers (docs/NUPI.md).

Each constraint ``g_k(theta) <= eps_k`` gets a multiplier ``lambda_k`` set by a PI law on the constraint error
``e_t = g_t - eps`` (Sohrabi et al., ICML 2024, "On PI controllers for updating Lagrange multipliers in constrained
optimization"), in its positional form:

    xi_t     = nu * xi_{t-1} + (1 - nu) * e_t                       (EMA of the error; nu = 0 -> xi_t = e_t)
    I_t      = clip(I_{t-1} + e_t, integral_min, integral_max)       (integral with anti-windup clamp)
    lambda_t = clip(lambda_0 + kappa_p * xi_t + kappa_i * I_t, 0, lambda_max)

``kappa_p = 0, integral_min = 0`` is exactly projected gradient ascent on the multiplier (``lambda <- max(0, lambda +
kappa_i e)``): the baseline nuPI generalises. The bang-bang collapse guard (``Trainer.collapse``) is the other extreme.

The trained objective is ``L = L_task + sum_k lambda_k (g_k - eps_k)`` with ``lambda`` held fixed (detached) within a
step; the controller updates ``lambda`` after the step from a measurement of ``g_k`` on either the step's own training
rows (``source: "batch"``, smoothed by ``nu``) or a fixed TRAIN-split monitor sample (``source: "monitor"``, never the
eval items). OFF by default: a ``TrainConfig`` without ``constraints`` runs the original code path, bit-identically.

Config (``TrainConfig.constraints`` / the cell config's ``constraints:`` block)::

    {"kappa_p": 1.0, "kappa_i": 0.05, "nu": 0.0, "lambda_max": 10.0, "integral_min": 0.0, "integral_max": 200.0,
     "source": "batch", "monitor_rows": 64, "monitor_every": 25, "monitor_seed": 0,
     "items": [
       {"name": "cal", "kind": "calibration", "metric": "brier", "setpoint": 0.18},
       {"name": "ece", "kind": "calibration", "metric": "ece", "setpoint": 0.05, "bins": 10, "source": "monitor"},
       {"name": "fam", "kind": "family_regression", "delta": 0.05, "reference": "init"},
       {"name": "flip", "kind": "consistency", "setpoint": 0.01},
       {"name": "recall", "kind": "recall", "setpoint": 0.7, "probe_path": "...", "probe_batch": 8}]}

Every controller key (kappa_p ... integral_max, lambda_init) can be overridden per item.
"""

from __future__ import annotations

from typing import Any

import torch

from .pi_control import PIController, gradient_ascent  # noqa: F401  (pure Python, no torch)

KINDS = ("calibration", "consistency", "family_regression", "recall")
CONTROLLER_KEYS = ("kappa_p", "kappa_i", "nu", "lambda_init", "lambda_max", "integral_min", "integral_max")
DEFAULTS = {"kappa_p": 1.0, "kappa_i": 0.05, "nu": 0.0, "lambda_init": 0.0, "lambda_max": 10.0,
            "integral_min": 0.0, "integral_max": 200.0}
TOP_KEYS = set(CONTROLLER_KEYS) | {"source", "monitor_rows", "monitor_every", "monitor_seed", "items"}
ITEM_KEYS = set(CONTROLLER_KEYS) | {"name", "kind", "setpoint", "metric", "bins", "source", "delta", "reference",
                                    "family_key", "probe_path", "probe_batch"}


# ---------------------------------------------------------------------------------------------- metrics
def brier(lp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-row multi-class Brier score over the row's options (padded options: p = 0 and target 0, so they add 0)."""
    p = lp.exp()
    t = torch.zeros_like(p)
    t[torch.arange(len(y), device=p.device), y] = 1.0
    return ((p - t) ** 2).sum(-1)


def ece_from(conf: torch.Tensor, correct: torch.Tensor, bins: int) -> torch.Tensor:
    """Top-1 expected calibration error with ``bins`` equal-width bins (differentiable through ``conf``)."""
    idx = torch.clamp((conf.detach() * bins).long(), max=bins - 1)
    out = conf.new_zeros(())
    for b in range(bins):
        m = idx == b
        if m.any():
            out = out + m.float().mean() * (correct[m].float().mean() - conf[m].mean()).abs()
    return out


def top1(lp: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    conf, pred = lp.exp().max(-1)
    return conf, pred == y


# ---------------------------------------------------------------------------------------------- the constraint set
def validate(spec: dict) -> None:
    if not isinstance(spec, dict) or not spec.get("items"):
        raise ValueError("constraints: needs a non-empty 'items' list")
    bad = set(spec) - TOP_KEYS
    if bad:
        raise ValueError(f"constraints: unknown keys {sorted(bad)}")
    names = set()
    for it in spec["items"]:
        bad = set(it) - ITEM_KEYS
        if bad:
            raise ValueError(f"constraints item {it.get('name')}: unknown keys {sorted(bad)}")
        if it.get("kind") not in KINDS:
            raise ValueError(f"constraints item {it.get('name')}: kind must be one of {KINDS}")
        n = it.get("name") or it["kind"]
        if n in names:
            raise ValueError(f"constraints: duplicate item name {n!r}")
        names.add(n)
        src = it.get("source", spec.get("source", "batch"))
        if src not in ("batch", "monitor"):
            raise ValueError(f"constraints item {n}: source must be 'batch' or 'monitor'")
        if it["kind"] == "calibration" and it.get("metric", "brier") not in ("brier", "ece"):
            raise ValueError(f"constraints item {n}: calibration metric must be 'brier' or 'ece'")
        if it["kind"] == "family_regression":
            ref = it.get("reference", "init")
            if not (ref == "init" or isinstance(ref, dict)):
                raise ValueError(f"constraints item {n}: reference must be 'init' or a {{family: loss}} dict")
            if float(it.get("delta", 0.05)) < 0:
                raise ValueError(f"constraints item {n}: delta must be >= 0")
            if ref == "init" and src != "monitor" and not spec.get("monitor_rows", 64):
                raise ValueError(f"constraints item {n}: reference 'init' needs monitor_rows > 0")
        elif it["kind"] == "recall":
            if src != "batch":
                raise ValueError(f"constraints item {n}: recall is measured on its own probe batch (source 'batch')")
            if "setpoint" not in it and "delta" not in it:
                raise ValueError(f"constraints item {n}: recall needs a 'setpoint' (probe loss) or 'delta' (x init)")
        elif it["kind"] == "consistency" and src != "batch":
            raise ValueError(f"constraints item {n}: consistency is measured on the step's paired views ('batch')")
        elif "setpoint" not in it:
            raise ValueError(f"constraints item {n}: needs a 'setpoint'")


class ConstraintSet:
    """The trainer-side half: per-step penalties, measurements, controller updates, logging and checkpoint state."""

    def __init__(self, spec: dict, train_items: list, probe_items: list | None = None):
        validate(spec)
        self.spec = spec
        self.source_default = spec.get("source", "batch")
        self.monitor_every = int(spec.get("monitor_every", 25))
        self.items: list[dict] = []
        for it in spec["items"]:
            c = {k: float(it.get(k, spec.get(k, DEFAULTS[k]))) for k in CONTROLLER_KEYS}
            self.items.append({**it, "name": it.get("name") or it["kind"], "ctl": c,
                               "source": it.get("source", self.source_default)})
        self.kinds = {it["kind"] for it in self.items}
        if "consistency" in self.kinds and not any(getattr(i, "view", None) is not None for i in train_items):
            raise ValueError("constraints: a consistency item needs training items with order views (view/view_perm)")
        if "recall" in self.kinds and not probe_items:
            raise ValueError("constraints: a recall item needs probe items (Trainer(probe_items=...) / probe_path)")
        self.probe_items = list(probe_items or [])
        from .trainer import dev_card_sample

        needs_monitor = any(it["source"] == "monitor" or (it["kind"] == "family_regression"
                                                         and it.get("reference", "init") == "init")
                            for it in self.items)
        self.monitor_items = (dev_card_sample(train_items, int(spec.get("monitor_rows", 64)),
                                              int(spec.get("monitor_seed", 0))) if needs_monitor else [])
        #: controllers by key: "<name>" or "<name>/<family>"; family controllers are made once the reference is known
        self.ctl: dict[str, PIController] = {}
        self.bounds: dict[str, float] = {}  # per-family bounds and recall-from-init bounds (stored in the checkpoint)
        for it in self.items:
            if it["kind"] == "family_regression" and isinstance(it.get("reference", "init"), dict):
                self._make_family(it, {str(k): float(v) for k, v in it["reference"].items()})
            elif it["kind"] != "family_regression":
                self.ctl[it["name"]] = PIController(**it["ctl"])
        self.initialised = False
        self.probe_cursor = 0
        self.last: dict[str, dict[str, float]] = {}
        self._acc: dict[str, list[float]] = {}
        self.step_penalty = 0.0
        self.probe_penalty = 0.0

    # ------------------------------------------------------------------ setup
    def _make_family(self, it: dict, ref: dict[str, float]) -> None:
        d = float(it.get("delta", 0.05))
        for fam, r in sorted(ref.items()):
            key = f"{it['name']}/{fam}"
            self.bounds[key] = (1.0 + d) * r
            self.ctl[key] = PIController(**it["ctl"])

    def initialise(self, tr) -> None:
        """Step-0 references (family 'init' references from the monitor sample, recall 'delta' from the probe set).
        Run once before the first step; the results live in the checkpoint, so a resume does not re-measure."""
        if self.initialised:
            return
        need_fam = [it for it in self.items if it["kind"] == "family_regression"
                    and it.get("reference", "init") == "init"]
        if need_fam:
            per = self._measure(tr, self.monitor_items)["family_rows"]
            for it in need_fam:
                ref = {f: sum(v) / len(v) for f, v in per.items() if v}
                self._make_family(it, ref)
        for it in self.items:
            if it["kind"] == "recall":
                if "setpoint" in it:
                    self.bounds[it["name"]] = float(it["setpoint"])
                else:
                    init = self._measure(tr, self.probe_items)["ce_mean"]
                    self.bounds[it["name"]] = (1.0 + float(it["delta"])) * init
        self.initialised = True

    def family_of(self, item, it: dict) -> str:
        from .trainer import item_source

        return item_source(item.id) if it.get("family_key", "source") == "source" else str(getattr(item, "family", ""))

    def bound(self, it: dict) -> float:
        return self.bounds.get(it["name"], float(it.get("setpoint", 0.0)))

    # ------------------------------------------------------------------ per step
    def begin_step(self) -> None:
        self._acc = {}
        self.step_penalty = 0.0  # micro-batch penalties (inside the task-loss totals the trainer sums)
        self.probe_penalty = 0.0  # the recall probe term (backpropagated separately)

    def _add(self, key: str, val: float, w: float) -> None:
        a = self._acc.setdefault(key, [0.0, 0.0])
        a[0] += val
        a[1] += w

    def penalty(self, tr, lp: torch.Tensor, y: torch.Tensor, batch: list, weight: float,
                view: tuple | None = None) -> torch.Tensor:
        """sum_k lambda_k (g_k(batch) - eps_k) for this micro-batch, in the same per-row normalisation as the task
        loss (the caller scales it by ``weight`` like the loss). Also accumulates the step's measurements."""
        from .losses import aligned_order_js, arm_loss

        pen = lp.new_zeros(())
        qtypes = [i.row.qtype for i in batch]
        B = len(batch)
        for it in self.items:
            k, name = it["kind"], it["name"]
            if k == "calibration":
                if it.get("metric", "brier") == "brier":
                    g = brier(lp, y).mean()
                else:
                    conf, cor = top1(lp, y)
                    g = ece_from(conf, cor, int(it.get("bins", 10)))
                self._add(name, float(g.detach()) * weight, weight)
                pen = pen + self.ctl[name].lam * (g - self.bound(it))
            elif k == "family_regression":
                if not any(key.startswith(name + "/") for key in self.ctl):
                    continue
                per = arm_loss(lp, y, qtypes, tr.cfg.loss)
                fams = [self.family_of(i, it) for i in batch]
                for fam in sorted(set(fams)):
                    key = f"{name}/{fam}"
                    if key not in self.ctl:
                        continue  # a family without a reference is unconstrained
                    m = torch.tensor([f == fam for f in fams], device=lp.device)
                    s = per[m].sum()
                    n = int(m.sum())
                    self._add(key, float(s.detach()) * weight / B, n * weight / B)
                    pen = pen + self.ctl[key].lam * (s - n * self.bounds[key]) / B
            elif k == "consistency":
                if view is None:
                    view = tr._constraint_view(batch, lp)
                if view is None:
                    continue
                js = view
                self._add(name, float(js.detach()) * weight, weight)
                pen = pen + self.ctl[name].lam * (js - self.bound(it))
        self.step_penalty += float(pen.detach()) * weight
        return pen

    def probe_backward(self, tr) -> float:
        """Recall items: one probe batch per step (deterministic cycle over the probe set), lambda (CE - bound)
        backpropagated into the step's gradient. Returns the probe penalty's value."""
        rec = [it for it in self.items if it["kind"] == "recall"]
        if not rec:
            return 0.0
        total = 0.0
        for it in rec:
            n = int(it.get("probe_batch", 8))
            idx = [(self.probe_cursor + j) % len(self.probe_items) for j in range(n)]
            batch = [self.probe_items[j] for j in idx]
            with tr._autocast():
                fw = tr.model([i.row for i in batch])
            lp = fw.log_probs.float()
            y = torch.tensor([i.label for i in batch], device=lp.device)
            from .losses import arm_loss

            ce = arm_loss(lp, y, [i.row.qtype for i in batch], tr.cfg.loss).mean()
            self._add(it["name"], float(ce.detach()), 1.0)
            lam = self.ctl[it["name"]].lam
            if lam:
                (lam * (ce - self.bound(it))).backward()
            total += float(lam * (float(ce.detach()) - self.bound(it)))
        self.probe_cursor = (self.probe_cursor + max(int(it.get("probe_batch", 8)) for it in rec)) % len(self.probe_items)
        self.probe_penalty = total
        return total

    def _measure(self, tr, items: list) -> dict[str, Any]:
        """No-grad pooled measurements over ``items`` (eval mode, chunks of batch_size): Brier, ECE inputs,
        per-family row losses, mean CE."""
        from .losses import arm_loss

        was = tr.model.training
        tr.model.eval()
        conf_all, cor_all, br, ce_all = [], [], [], []
        fam_rows: dict[str, list[float]] = {}
        fam_items = [it for it in self.items if it["kind"] == "family_regression"]
        try:
            with torch.no_grad():
                bs = max(1, tr.cfg.batch_size)
                for s in range(0, len(items), bs):
                    batch = items[s: s + bs]
                    with tr._autocast():
                        fw = tr.model([i.row for i in batch])
                    lp = fw.log_probs.float()
                    y = torch.tensor([i.label for i in batch], device=lp.device)
                    conf, cor = top1(lp, y)
                    conf_all.append(conf)
                    cor_all.append(cor)
                    br.append(brier(lp, y))
                    per = arm_loss(lp, y, [i.row.qtype for i in batch], tr.cfg.loss)
                    ce_all.append(per)
                    if fam_items:
                        for i, row_loss in zip(batch, per.tolist()):
                            fam_rows.setdefault(self.family_of(i, fam_items[0]), []).append(row_loss)
        finally:
            tr.model.train(was)
        cat = (lambda xs: torch.cat(xs) if xs else torch.zeros(0))
        conf, cor, br_t, ce = cat(conf_all), cat(cor_all), cat(br), cat(ce_all)
        return {"conf": conf, "correct": cor, "brier": float(br_t.mean()) if len(br_t) else float("nan"),
                "family_rows": fam_rows, "ce_mean": float(ce.mean()) if len(ce) else float("nan")}

    def end_step(self, tr) -> dict[str, dict[str, float]]:
        """Measure, update the controllers (lambda for the NEXT step), return the step's log entry."""
        log: dict[str, dict[str, float]] = {}
        monitor_due = self.monitor_every > 0 and tr.step % self.monitor_every == 0
        mon = None
        for it in self.items:
            keys = ([k for k in self.ctl if k.startswith(it["name"] + "/")] if it["kind"] == "family_regression"
                    else [it["name"]])
            for key in keys:
                g = None
                if it["source"] == "batch":
                    a = self._acc.get(key)
                    if a and a[1] > 0:
                        g = a[0] / a[1]
                elif monitor_due:
                    if mon is None:
                        mon = self._measure(tr, self.monitor_items)
                    if it["kind"] == "calibration":
                        g = (mon["brier"] if it.get("metric", "brier") == "brier" else
                             float(ece_from(mon["conf"], mon["correct"], int(it.get("bins", 10)))))
                    elif it["kind"] == "family_regression":
                        rows = mon["family_rows"].get(key.split("/", 1)[1])
                        g = sum(rows) / len(rows) if rows else None
                bnd = self.bounds[key] if key in self.bounds else self.bound(it)
                if g is not None:
                    self.ctl[key].update(g - bnd)
                    self.last[key] = {"g": g, "e": g - bnd}
                log[key] = {"lam": self.ctl[key].lam, **({"g": g, "e": g - bnd} if g is not None else {})}
        return log

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict[str, Any]:
        return {"ctl": {k: c.state_dict() for k, c in sorted(self.ctl.items())}, "bounds": dict(self.bounds),
                "initialised": self.initialised, "probe_cursor": self.probe_cursor}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.bounds = {k: float(v) for k, v in st["bounds"].items()}
        for k in st["ctl"]:
            if k not in self.ctl:  # family controllers created from an 'init' reference
                name = k.split("/", 1)[0]
                it = next(i for i in self.items if i["name"] == name)
                self.ctl[k] = PIController(**it["ctl"])
            self.ctl[k].load_state_dict(st["ctl"][k])
        self.initialised = bool(st["initialised"])
        self.probe_cursor = int(st["probe_cursor"])

    def lambdas(self) -> dict[str, float]:
        return {k: c.lam for k, c in sorted(self.ctl.items())}
