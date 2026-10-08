"""Decision Index kit engine over the ``decide`` HF readout (``python -m decision_index run --engine
jobs.common.readout_engine:ReadoutIndexEngine --option family=... --option model_dir=...``).

families
  ``q9af-probe``  ``convert.readout.load_converted_plugin``; options ``gdn_ckpt``, ``converted`` (comma list; empty =
                  the stock model through the same code path, the paired baseline arm of jobs/probe9b)
  ``rwkv7-hf``    ``decide.backends.hf_readout.load_rwkv7_hf`` (RWKV7-G1k, vendored bundled code)
  ``rwkv7-hf-code-label-v1`` same plugin, explicit G1CodeLabelBackend (frozen filtered codes)
  ``rwkv7-hf-code-label-v2`` same plugin, all-field labels and bounded prefix-fork scorer
  ``rwkv7-hf-g1-v2`` same plugin, ``decide.backends.rwkv_g1_v2.G1V2Backend`` (vendor-compliant jrt-rwkv-g1-v2: no
                  blank lines in the turn, token 0 first, G1k no-think tail, single-token option codes, whole-string
                  tokenization)
  ``rwkv7-hf-g1-v3`` same plugin, ``decide.backends.rwkv_g1_v3.G1V3Backend`` (R3: v2 + v3 code table + code key)
  ``rwkv7-hf-g1-v3p`` / ``-v3r`` / ``-v3c`` / ``-v3w`` G1V3Backend modes permute / reread / calibrate / words (one
                  change each on v3); ``rwkv7-hf-g1-v2f`` mode v2f (v2 + the dict-instruction fix only)
  ``rwkv7-hf-g1-v3j`` / ``-v3js`` G1V3Backend modes joint / joint_seq (R4: v3 + joint candidate-block decode;
                  + sequential own-answer conditioning)
  ``qwen3.5``     ``decide.backends.hf_readout.load_qwen35`` (stock transformers path)

Mapping as ``baselines.clef_index_engine``: ``exhausted``/``unsupported`` -> kit ``Unsupported`` (scored zero by the Index
rules), any other non-ok status raises.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CODE_LABEL_FAMILY = "rwkv7-hf-code-label-v1"
CODE_LABEL_V2_FAMILY = "rwkv7-hf-code-label-v2"
G1V2_FAMILY = "rwkv7-hf-g1-v2"
G1V3_FAMILY = "rwkv7-hf-g1-v3"
G1V3C_FAMILY = "rwkv7-hf-g1-v3c"
G1V3_MODES = {"rwkv7-hf-g1-v2f": "v2f", G1V3_FAMILY: "catalog", "rwkv7-hf-g1-v3w": "words",
              "rwkv7-hf-g1-v3p": "permute", "rwkv7-hf-g1-v3r": "reread", G1V3C_FAMILY: "calibrate",
              "rwkv7-hf-g1-v3j": "joint", "rwkv7-hf-g1-v3js": "joint_seq"}
#: RWKV7 plugins loaded in this process, keyed by (model_dir, device, dtype): one process that runs several readout
#: variants (jobs/rwkv7 JOB=readout_g1v3) loads the 26 GB model once. The plugin is read-only for every backend.
_RWKV7_PLUGINS: dict[tuple[str, str, str], Any] = {}
TEMPLATES = {"q9af-probe": "jrt-chatml-v1", "qwen3.5": "jrt-chatml-v1", "rwkv7-hf": "jrt-rwkv-g1-v1",
             CODE_LABEL_FAMILY: "jrt-rwkv-g1-code-label-v1", CODE_LABEL_V2_FAMILY: "jrt-rwkv-g1-code-label-v2",
             G1V2_FAMILY: "jrt-rwkv-g1-v2", G1V3_FAMILY: "jrt-rwkv-g1-v3", "rwkv7-hf-g1-v3p": "jrt-rwkv-g1-v3p",
             "rwkv7-hf-g1-v3r": "jrt-rwkv-g1-v3r", G1V3C_FAMILY: "jrt-rwkv-g1-v3c",
             "rwkv7-hf-g1-v2f": "jrt-rwkv-g1-v2f", "rwkv7-hf-g1-v3w": "jrt-rwkv-g1-v3w",
             "rwkv7-hf-g1-v3j": "jrt-rwkv-g1-v3j", "rwkv7-hf-g1-v3js": "jrt-rwkv-g1-v3js"}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_code_label_decode(model_dir):
    """Recreate only the reviewed local tokenizer's decoder; never use remote code.

    Validate this inexpensive prerequisite before the model/plugin is loaded.
    The backend separately binds/proves the actual plugin encoder and code table.
    """
    import third_party.rwkv7_hf as V
    from decide.backends.rwkv_g1_filtered_codes import TOKENIZER_SHA256, TOKENIZER_SOURCE_SHA256

    vocab = Path(model_dir) / "tokenizer.json"
    source = V.HERE / "tokenization_rwkv7.py"
    if (_sha256_file(vocab) != TOKENIZER_SHA256
            or _sha256_file(source) != TOKENIZER_SOURCE_SHA256
            or V.CODE_SHA256["tokenization_rwkv7.py"] != TOKENIZER_SOURCE_SHA256):
        raise ValueError("code-label engine requires the pinned G1k vocabulary and reviewed tokenizer source")
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    tokenizer = Rwkv7Tokenizer(tokenizer_file=str(vocab))
    return tokenizer.decode


def build_plugin(family: str, options: dict[str, Any]):
    dev, dt = options.get("device", "cuda"), options.get("dtype", "bfloat16")
    if family == "q9af-probe":
        from convert.readout import load_converted_plugin

        return load_converted_plugin(options["model_dir"], options.get("gdn_ckpt") or None,
                                     options.get("converted", ""), device=dev, dtype=dt)
    from decide.backends import hf_readout as H

    if family in ("rwkv7-hf", CODE_LABEL_FAMILY, CODE_LABEL_V2_FAMILY, G1V2_FAMILY, *G1V3_MODES):
        key = (str(options["model_dir"]), dev, dt)
        if key not in _RWKV7_PLUGINS:
            _RWKV7_PLUGINS[key] = H.load_rwkv7_hf(options["model_dir"], device=dev, dtype=dt)
        return _RWKV7_PLUGINS[key]
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
        decode = _load_code_label_decode(options["model_dir"]) if self.family in (CODE_LABEL_FAMILY, CODE_LABEL_V2_FAMILY) else None
        plug = build_plugin(self.family, options)
        if self.family in G1V3_MODES:
            from decide.backends.rwkv_g1_v3 import G1V3Backend

            self.backend = G1V3Backend(plug, mode=G1V3_MODES[self.family])
        elif self.family == G1V2_FAMILY:
            from decide.backends.rwkv_g1_v2 import G1V2Backend

            self.backend = G1V2Backend(plug)
        elif self.family == CODE_LABEL_V2_FAMILY:
            from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend

            self.backend = G1CodeLabelV2Backend(plug, tokenizer_decode=decode)
        elif self.family == CODE_LABEL_FAMILY:
            from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend

            self.backend = G1CodeLabelBackend(plug, tokenizer_decode=decode)
        else:
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
