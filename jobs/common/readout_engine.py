"""Decision Index kit engine over the ``decide`` HF readout (``python -m decision_index run --engine
jobs.common.readout_engine:ReadoutIndexEngine --option family=... --option model_dir=...``).

families
  ``q9af-probe``  ``convert.readout.load_converted_plugin``; options ``gdn_ckpt``, ``converted`` (comma list; empty =
                  the stock model through the same code path, the paired baseline arm of jobs/probe9b)
  ``rwkv7-hf``    ``decide.backends.hf_readout.load_rwkv7_hf`` (RWKV7-G1k, vendored bundled code)
  ``qwen3.5``     ``decide.backends.hf_readout.load_qwen35`` (stock transformers path)

Mapping as ``baselines.clef_index_engine``: ``exhausted``/``unsupported`` -> kit ``Unsupported`` (scored zero by the Index
rules), any other non-ok status raises.
"""
from __future__ import annotations

from typing import Any

TEMPLATES = {"q9af-probe": "jrt-chatml-v1", "qwen3.5": "jrt-chatml-v1", "rwkv7-hf": "jrt-rwkv-g1-v1"}


def build_plugin(family: str, options: dict[str, Any]):
    dev, dt = options.get("device", "cuda"), options.get("dtype", "bfloat16")
    if family == "q9af-probe":
        from convert.readout import load_converted_plugin

        return load_converted_plugin(options["model_dir"], options.get("gdn_ckpt") or None,
                                     options.get("converted", ""), device=dev, dtype=dt)
    from decide.backends import hf_readout as H

    if family == "rwkv7-hf":
        return H.load_rwkv7_hf(options["model_dir"], device=dev, dtype=dt)
    if family == "qwen3.5":
        return H.load_qwen35(options["model_dir"], device=dev, dtype=dt)
    raise ValueError(f"unknown family {family!r}; known: {sorted(TEMPLATES)}")


def map_response(resp: dict[str, Any], unsupported_exc=ValueError):
    """decide response -> (kit response, raw) or raise; pure so it is testable without the kit."""
    ss = resp["ssdm"]
    if ss["status"] in ("exhausted", "unsupported"):
        raise unsupported_exc(f"{ss['status']}/{ss['error']['code']}: {ss['error']['message']}")
    if ss["status"] != "ok":
        raise RuntimeError(f"{ss['status']}/{ss['error']['code']}: {ss['error']['message']}")
    raw = {"raw_probabilities": ss.get("raw_probabilities"), "sequence_tokens": resp["usage"]["input_tokens"],
           "latency_ms": (ss.get("diagnostics") or {}).get("latency_ms")}
    return {"model": resp["model"], "answers": resp["answers"], "usage": resp["usage"]}, raw


try:  # the kit is installed in the job images; CPU tests import map_response without it
    from decision_index.engines.base import Engine, Unsupported
except Exception:  # noqa: BLE001
    Engine, Unsupported = object, ValueError


class ReadoutIndexEngine(Engine):
    name = "ssdm-readout"
    latency = ("In-process wall time of one request through decide.Decider (prompt build, option log-likelihood "
               "forwards, calibration), CUDA-synchronised by the kit; excludes model loading. Batch 1 request.")

    def __init__(self, **options: Any):
        super().__init__(**options)
        from decide import Decider
        from decide.backends.hf_readout import HFReadoutBackend

        self.family = options.get("family", "q9af-probe")
        plug = build_plugin(self.family, options)
        self.backend = HFReadoutBackend(plug, TEMPLATES[self.family], max_batch_rows=int(options.get("max_batch_rows", 4)))
        self.decider = Decider(self.backend, enable_v0_1_deltas=True, backend_max_input_tokens=plug.max_context)
        rev = self.backend.revision()
        self.provenance = {"family": self.family, "model_id": plug.model_id, "backend_revision": rev["revision_id"],
                           "notes": plug.notes}

    def __call__(self, state, questions):
        resp = self.decider.decide({"model": self.family, "state": state, "questions": questions})
        return map_response(resp, Unsupported)

    def runtime(self):
        import torch

        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        return {}
