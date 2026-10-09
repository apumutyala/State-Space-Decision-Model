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
  ``rwkv7-hf-g1-v3w-r`` v3w-r = v3w (mode words) + the opt-in parameter-free readout rules
                  (``decide.readout_rules`` r1: BFCL exactly-one, SATA top-m) + a 32,768-token default input cap
                  for the adapter and the v3 render budget (the plugin's 25,600 trained-ctx cap is not applied: RWKV
                  is recurrent). 32k is a readout-family setting, not the claim cap (R3: RWKV arms read every row
                  in full at 65,536, as fulldev_eval.py; Clef: 16,384 + the 51 over-length rows substituted at
                  32,768)
  ``qwen3.5``     ``decide.backends.hf_readout.load_qwen35`` (stock transformers path)
  ``qrwkv6-chatml`` / ``qrwkv6-chatml-nothink`` ``decide.backends.rwkv6qwen2.load_rwkv6qwen2`` (attention-free RWKV6-Qwen2
                  conversions, vendored pinned code; option ``repo`` names the pinned repo), stock LM-logit readout
                  (HFReadoutBackend, prefix fork, ``fork_max_rows`` default 16) under jrt-chatml-v1 (QwQ no-think tail)
                  or jrt-chatml-nothink-v1 (Qwen2.5-Instruct tail). Option ``selfcheck_out``: before the first row,
                  score the canary request by fork AND by recompute and refuse (raise) unless they agree.
  ``qrwkv6-v3w-generic`` the same RWKV6-Qwen2 plug-in, scored by the v3w-generic readout
                  (``decide.backends.v3w_generic_backend.V3WGenericBackend`` over ``decide.v3w_generic.make_v3w``: the
                  model's own chat template, its own single-token code table, word ids, whole-string tokenization);
                  options ``backbone_id`` (a ``decide.v3w_generic.BACKBONES`` key), ``v3w_sha256`` (the pinned readout
                  identity: refused unless equal), ``selfcheck_out`` (canary: prefix fork == full recompute).
  ``rwkv07c-chatml-nothink`` ``decide.backends.rwkv07c.load_rwkv07c`` (OpenMOSE RWKV-Qwen3-30B-A3B, attention-free
                  RWKV hxa07c + Qwen3-MoE), same readout / options / self-check, jrt-chatml-nothink-v1.

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
              "rwkv7-hf-g1-v3p": "permute", "rwkv7-hf-g1-v3r": "reread", G1V3C_FAMILY: "calibrate"}
#: readout families served with ``Decider(readout_rules=True)`` (decide.readout_rules): family -> its base G1V3 family
RULES_FAMILIES = {"rwkv7-hf-g1-v3w-r": "rwkv7-hf-g1-v3w"}
#: RWKV7 plugins loaded in this process, keyed by (model_dir, device, dtype): one process that runs several readout
#: variants (jobs/rwkv7 JOB=readout_g1v3) loads the 26 GB model once. The plugin is read-only for every backend.
_RWKV7_PLUGINS: dict[tuple[str, str, str], Any] = {}
TEMPLATES = {"q9af-probe": "jrt-chatml-v1", "qwen3.5": "jrt-chatml-v1", "rwkv7-hf": "jrt-rwkv-g1-v1",
             CODE_LABEL_FAMILY: "jrt-rwkv-g1-code-label-v1", CODE_LABEL_V2_FAMILY: "jrt-rwkv-g1-code-label-v2",
             G1V2_FAMILY: "jrt-rwkv-g1-v2", G1V3_FAMILY: "jrt-rwkv-g1-v3", "rwkv7-hf-g1-v3p": "jrt-rwkv-g1-v3p",
             "rwkv7-hf-g1-v3r": "jrt-rwkv-g1-v3r", G1V3C_FAMILY: "jrt-rwkv-g1-v3c",
             "rwkv7-hf-g1-v2f": "jrt-rwkv-g1-v2f", "rwkv7-hf-g1-v3w": "jrt-rwkv-g1-v3w",
             "rwkv7-hf-g1-v3w-r": "jrt-rwkv-g1-v3w",
             "qrwkv6-chatml": "jrt-chatml-v1", "qrwkv6-chatml-nothink": "jrt-chatml-nothink-v1",
             "rwkv07c-chatml-nothink": "jrt-chatml-nothink-v1"}
#: attention-free conversions scored by the stock LM-logit readout with the canary fork self-check (backbone scout)
QRWKV6_FAMILIES = ("qrwkv6-chatml", "qrwkv6-chatml-nothink", "rwkv07c-chatml-nothink")
#: the same attention-free plug-ins under the v3w-generic readout (backbone scout readout-artifact check)
V3W_GENERIC_FAMILIES = {"qrwkv6-v3w-generic": "qrwkv6-chatml-nothink"}
#: fork vs recompute on the canary request (bf16 on GPU): max |delta sum-logprob| allowed, and same argmax per question
SELFCHECK_TOL = 0.25


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

    if family in ("rwkv7-hf", CODE_LABEL_FAMILY, CODE_LABEL_V2_FAMILY, G1V2_FAMILY, *G1V3_MODES, *RULES_FAMILIES):
        key = (str(options["model_dir"]), dev, dt)
        if key not in _RWKV7_PLUGINS:
            _RWKV7_PLUGINS[key] = H.load_rwkv7_hf(options["model_dir"], device=dev, dtype=dt)
        return _RWKV7_PLUGINS[key]
    if family == "qwen3.5":
        return H.load_qwen35(options["model_dir"], device=dev, dtype=dt)
    if family == "rwkv07c-chatml-nothink":
        from decide.backends.rwkv07c import load_rwkv07c

        cap = options.get("input_cap")
        return load_rwkv07c(options["model_dir"], options["repo"], device=dev, dtype=dt, input_cap=int(cap) if cap else None)
    if family in QRWKV6_FAMILIES or family in V3W_GENERIC_FAMILIES:
        from decide.backends.rwkv6qwen2 import load_rwkv6qwen2

        cap = options.get("input_cap")
        return load_rwkv6qwen2(options["model_dir"], options["repo"], device=dev, dtype=dt,
                               input_cap=int(cap) if cap else None)
    raise ValueError(f"unknown family {family!r}; known: {sorted(TEMPLATES)}")


def template_of(family: str):
    """Template id -> the object HFReadoutBackend takes (jrt-chatml-nothink-v1 lives outside decide.prompt.TEMPLATES)."""
    if TEMPLATES[family] == "jrt-chatml-nothink-v1":
        from decide.backends.rwkv6qwen2 import CHATML_NOTHINK

        return CHATML_NOTHINK
    return TEMPLATES[family]


def fork_selfcheck(plug, template, fork_max_rows: int, tol: float = SELFCHECK_TOL) -> dict[str, Any]:
    """Score decide.prompt.CANARY_REQUEST by the prefix fork and by full recompute on the loaded model; the fork's
    state ops are only trusted if both agree (max |delta| <= tol nats per option, same argmax per question)."""
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.prompt import CANARY_REQUEST

    out = {}
    for name, fork in (("fork", True), ("recompute", False)):
        b = HFReadoutBackend(plug, template, max_batch_rows=4, fork=fork, fork_max_rows=fork_max_rows)
        out[name] = b.score(CANARY_REQUEST, lambda: None).logits
    deltas = [abs(out["fork"][q][o] - out["recompute"][q][o]) for q in out["fork"] for o in out["fork"][q]]
    same = all(max(out["fork"][q], key=out["fork"][q].get) == max(out["recompute"][q], key=out["recompute"][q].get)
               for q in out["fork"])
    rep = {"max_abs_delta": max(deltas), "options": len(deltas), "same_argmax": same, "tol": tol,
           "ok": bool(max(deltas) <= tol and same), "logits": out}
    return rep


def v3w_tokenizers_root(backbone_id: str, model_dir) -> Path:
    """A tokenizers root holding the pinned snapshot under the registry's directory name (a symlink), so make_v3w
    reads the snapshot's own tokenizer files and its identity (which names the directory) equals the CPU-pinned one."""
    import tempfile

    from decide.v3w_generic import BACKBONES

    root = Path(tempfile.mkdtemp(prefix="v3w-tok-"))
    (root / BACKBONES[backbone_id].tokenizer_dir).symlink_to(Path(model_dir).resolve(), target_is_directory=True)
    return root


def v3w_selfcheck(backend, tol: float = SELFCHECK_TOL) -> dict[str, Any]:
    """The v3w-generic backend's prefix-fork scores on decide.prompt.CANARY_REQUEST vs a full recompute of every
    question's whole prompt (one forward, the next-token read after "ANSWER:"); same tolerance as fork_selfcheck."""
    import torch

    from decide.prompt import CANARY_REQUEST

    fork = backend.score(CANARY_REQUEST, lambda: None).logits
    plug, rec = backend.plugin, {}
    for q in backend.v3w.questions(CANARY_REQUEST):
        if not q["coded"]:
            continue
        with torch.inference_mode():
            lg = plug.forward(torch.tensor([q["prompt"]], dtype=torch.long, device=plug.device), 1)
            lp = torch.log_softmax(lg[0, -1].float(), dim=-1)
        rec[q["qid"]] = dict(zip(q["options"], lp.index_select(0, torch.tensor(q["tokens"], device=lp.device)).tolist()))
    deltas = [abs(fork[q][o] - rec[q][o]) for q in rec for o in rec[q]]
    same = all(max(fork[q], key=fork[q].get) == max(rec[q], key=rec[q].get) for q in rec)
    return {"max_abs_delta": max(deltas), "options": len(deltas), "same_argmax": same, "tol": tol,
            "ok": bool(deltas and max(deltas) <= tol and same), "logits": {"fork": fork, "recompute": rec}}


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
        rules = self.family in RULES_FAMILIES
        if rules:
            from decide.backends.rwkv_g1_v3 import G1V3Backend
            from decide.readout_rules import READOUT_RULES_CAP

            self.backend = G1V3Backend(plug, mode=G1V3_MODES[RULES_FAMILIES[self.family]], budget=READOUT_RULES_CAP)
        elif self.family in G1V3_MODES:
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
        elif self.family in V3W_GENERIC_FAMILIES:
            import json

            from decide.backends.v3w_generic_backend import V3WGenericBackend
            from decide.v3w_generic import make_v3w

            v3w = make_v3w(options["backbone_id"], tokenizers_root=v3w_tokenizers_root(options["backbone_id"],
                                                                                 options["model_dir"]))
            got = v3w.identity()["sha256"]
            if options.get("v3w_sha256") and got != options["v3w_sha256"]:
                raise RuntimeError(f"v3w-generic identity {got} != pinned {options['v3w_sha256']}")
            self.backend = V3WGenericBackend(plug, v3w)
            if options.get("selfcheck_out"):
                rep = v3w_selfcheck(self.backend)
                Path(options["selfcheck_out"]).write_text(json.dumps(rep, indent=1), encoding="utf-8")
                if not rep["ok"]:
                    raise RuntimeError(f"v3w fork != recompute on the canary: {json.dumps({k: rep[k] for k in ('max_abs_delta', 'same_argmax')})}")
        elif self.family in QRWKV6_FAMILIES:
            from decide.backends.rwkv6qwen2 import FORK_MAX_ROWS

            fmr = int(options.get("fork_max_rows", FORK_MAX_ROWS))
            if options.get("selfcheck_out"):
                import json

                rep = fork_selfcheck(plug, template_of(self.family), fmr)
                Path(options["selfcheck_out"]).write_text(json.dumps(rep, indent=1), encoding="utf-8")
                if not rep["ok"]:
                    raise RuntimeError(f"fork != recompute on the canary: {json.dumps({k: rep[k] for k in ('max_abs_delta', 'same_argmax')})}")
            self.backend = HFReadoutBackend(plug, template_of(self.family), max_batch_rows=int(options.get("max_batch_rows", 4)),
                                            fork_max_rows=fmr)
        else:
            self.backend = HFReadoutBackend(plug, TEMPLATES[self.family], max_batch_rows=int(options.get("max_batch_rows", 4)))
        if rules:
            self.decider = Decider(self.backend, enable_v0_1_deltas=True, readout_rules=True)
        else:
            self.decider = Decider(self.backend, enable_v0_1_deltas=True, backend_max_input_tokens=plug.max_context)
        rev = self.decider.revision() if rules else self.backend.revision()
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
