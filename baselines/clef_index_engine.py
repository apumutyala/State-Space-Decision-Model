"""Decision Index kit engine for Clef / Clef-flash (``python -m decision_index run --engine
baselines.clef_index_engine:ClefIndexEngine --option model_dir=... --option release=clef-flash``).

``path=decide`` (default, pre-registered primary): every row goes through ``decide.Decider`` over
``ClefBackend`` (ssdm encoding, no truncation). ``exhausted``/``unsupported`` become the kit's
``Unsupported`` (scored as zero, per the Index house rules); ``error`` raises.

``path=native``: Cloudflare's own ``systemone`` unchanged (silent truncation at ``max_length``); the
truncation and control-token counts are kept in ``raw_output`` so the receipt can show which rows
Clef answered on a cut state. Secondary arm only.
"""

from __future__ import annotations

from typing import Any

from decision_index.engines.base import Engine, Unsupported

from . import clef_backend as CB


class ClefIndexEngine(Engine):
    name = "clef"
    latency = ("In-process wall time of one request (encode + one joint forward + head + softmax), CUDA-synchronised; "
               "excludes model loading. One request at a time, batch 1.")

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.release = options.get("release", "clef-flash")
        self.path = options.get("path", "decide")
        if self.path not in ("decide", "native"):
            raise ValueError("path must be decide or native")
        self.max_length = int(options.get("max_length", CB.CLEF_MAX_LENGTH))
        self.backend, self.model, self.processor = CB.load_clef(
            options["model_dir"], self.release, encoding=options.get("encoding", "ssdm"),
            device=options.get("device", "cuda"), dtype=options.get("dtype", "bfloat16"),
            max_length=self.max_length, verify=str(options.get("verify", "true")).lower() != "false")
        self.decider = CB.build_decider(self.backend)
        self.provenance = {"repo": CB.RELEASES[self.release]["repo"], "revision": CB.RELEASES[self.release]["revision"],
                           "path": self.path, "backend_revision": self.backend.revision()["revision_id"],
                           "vendored_jsm_sha256": CB.JSM_SHA256}

    def __call__(self, state, questions):
        request = {"model": self.release, "state": state, "questions": questions}
        if self.path == "native":
            resp, meta = CB.native_systemone(self.backend.jsm, self.model, self.processor, request, self.max_length)
            return {"model": resp["model"], "answers": resp["answers"], "usage": resp["usage"]}, meta
        resp = self.decider.decide(request)
        ss = resp["ssdm"]
        if ss["status"] in ("exhausted", "unsupported"):
            raise Unsupported(f"{ss['status']}/{ss['error']['code']}: {ss['error']['message']}")
        if ss["status"] != "ok":
            raise RuntimeError(f"{ss['status']}/{ss['error']['code']}: {ss['error']['message']}")
        raw = {"raw_probabilities": ss["raw_probabilities"], "sequence_tokens": resp["usage"]["input_tokens"],
               "latency_ms": ss["diagnostics"]["latency_ms"]}
        return {"model": resp["model"], "answers": resp["answers"], "usage": resp["usage"]}, raw

    def runtime(self):
        import torch

        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        return {}
