"""Inference-only reconstruction of pinned native RWKV7 M2 Trainer artifacts.

No Trainer.load, optimizer restoration, calibration fitting, training records,
evaluation data, downloads, or permissive state loading. Runtime model loading
is intentionally separate from metadata-only qualification preflight.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .native_parent import LoadedNativeParent, NativeParentPin


def _config(payload):
    from train.losses import LossConfig
    from train.trainer import TrainConfig
    if not isinstance(payload, dict) or not isinstance(payload.get("loss"), dict):
        raise ValueError("checkpoint needs its complete saved TrainConfig")
    try:
        cfg = TrainConfig(**{**payload, "loss": LossConfig(**payload["loss"])})
    except TypeError as exc:
        raise ValueError("unsupported checkpoint TrainConfig") from exc
    from .qualified_parent import canonical_sha256
    if canonical_sha256(cfg.to_json()) != canonical_sha256(payload):
        raise ValueError("checkpoint config has missing or noncanonical fields")
    if cfg.controlsafe or cfg.head == "g16":
        raise ValueError("trust-gated reconstruction requires a separately reviewed loader")
    return cfg


def validate_reconstruction_metadata(q: dict, pin: NativeParentPin):
    from .qualified_parent import _digest, canonical_sha256
    cfg = _config(q["checkpoint_config"])
    if cfg.hash() != q["checkpoint_config_hash"]:
        raise ValueError("checkpoint config hash mismatch")
    if cfg.layout != pin.revision.get("layout") or cfg.fork_mode != pin.runtime["fork_mode"]:
        raise ValueError("checkpoint layout/fork runtime differs from serving bundle")
    checkpoint = pin.revision["checkpoint"]
    if (type(checkpoint.get("step")) is not int or checkpoint["step"] < 0
            or not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]):
        raise ValueError("serving revision needs the exact checkpoint step/path")
    lineage = q["checkpoint_lineage"]
    if not isinstance(lineage, dict) or set(lineage) != {"backbone", "tokenizer", "layout", "data"}:
        raise ValueError("incomplete checkpoint lineage")
    if canonical_sha256(lineage["backbone"]) != canonical_sha256(pin.revision.get("backbone")):
        raise ValueError("checkpoint frozen backbone lineage differs from serving revision")
    from decide.prompt import BODY_VERSION
    from heads.layout import LAYOUT_VERSION
    expected_layout = {"layouts": [[cfg.layout, pin.revision["prompt_template"]["id"]]],
                       "body_version": BODY_VERSION, "layout_version": LAYOUT_VERSION}
    if canonical_sha256(lineage["layout"]) != canonical_sha256(expected_layout):
        raise ValueError("checkpoint prompt layout lineage differs from serving revision")
    expected_tokenizer = [json.dumps(pin.revision["encoder"], sort_keys=True, default=str)]
    if lineage["tokenizer"] != expected_tokenizer:
        raise ValueError("checkpoint tokenizer lineage differs from serving revision")
    if (not isinstance(lineage["data"], dict) or set(lineage["data"]) != {"n", "sha256", "order_fn"}
            or type(lineage["data"]["n"]) is not int or lineage["data"]["n"] < 1
            or not isinstance(lineage["data"]["order_fn"], str) or not lineage["data"]["order_fn"]):
        raise ValueError("incomplete training-data lineage identity")
    _digest(lineage["data"]["sha256"])
    if (lineage["backbone"].get("weights_digest_mode") not in {"full", "sampled"}
            or lineage["backbone"].get("spec") != "rwkv7-13.3b"):
        raise ValueError("unsupported frozen-backbone lineage")
    _digest(lineage["backbone"].get("weights_digest"))
    if q["rwkv7_wkv"] not in (None, "eager", "chunked", "chunked_lean", "fla", "fla_safe_gate"):
        raise ValueError("unsupported explicit WKV implementation")
    modules = q["lora_modules"]
    if (not isinstance(modules, list) or any(not isinstance(n, str) or not n for n in modules)
            or modules != sorted(set(modules)) or bool(modules) != (cfg.lora_rank > 0)):
        raise ValueError("invalid complete LoRA module inventory")
    files = q["snapshot_files_sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("missing full local snapshot inventory")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (not isinstance(name, str) or "\\" in name or ":" in name or path.is_absolute()
                or ".." in path.parts or path.as_posix() != name):
            raise ValueError("snapshot inventory has unsafe/noncanonical path")
        _digest(digest)
    from third_party.rwkv7_hf import CODE_SHA256
    if not {"config.json", "tokenizer.json", *CODE_SHA256}.issubset(files):
        raise ValueError("snapshot inventory omits config/tokenizer/implementation")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("snapshot inventory omits backbone weights")
    return cfg


def verify_snapshot(snapshot: Path, inventory: dict) -> None:
    from .qualified_parent import file_sha256
    actual = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    if set(actual) != set(inventory):
        raise ValueError("local snapshot file inventory mismatch")
    if any(file_sha256(path) != inventory[name] for name, path in actual.items()):
        raise ValueError("local snapshot content hash mismatch")


def _load_checkpoint(path: Path):
    # Trainer saves NumPy's MT19937 RNG state. Permit only the inert ndarray
    # reconstruction primitives needed for that payload; never unsafe pickle.
    import numpy as np
    import torch
    reconstruct = np._core.multiarray._reconstruct if hasattr(np, "_core") else np.core.multiarray._reconstruct
    with torch.serialization.safe_globals([reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def restore_trainables(model, state: dict, q: dict, cfg, lora_modules: list, *, step: int | None = None) -> None:
    """Preflight every key/tensor before the first in-place copy; CPU-testable."""
    import torch
    from .qualified_parent import canonical_sha256
    if step is not None and (type(state.get("step")) is not int or state["step"] != step):
        raise ValueError("checkpoint step differs from serving revision")
    for key, expected in (("config", q["checkpoint_config"]), ("config_hash", cfg.hash()),
                          ("lineage", q["checkpoint_lineage"]), ("lora_modules", lora_modules),
                          ("init_seed", cfg.seed), ("backbone", "rwkv7-13.3b")):
        if canonical_sha256(state.get(key)) != canonical_sha256(expected):
            raise ValueError(f"checkpoint {key} differs from pinned reconstruction")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    weights = state.get("trainable")
    if not isinstance(weights, dict) or set(weights) != set(params) or not params:
        raise ValueError("checkpoint trainable key set is incomplete or mismatched")
    for name, param in params.items():
        value = weights[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != param.shape or value.dtype != param.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"checkpoint trainable tensor mismatch: {name}")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(weights[name])


def native_rwkv7_loader(snapshot: str | Path, qualification: dict, pin: NativeParentPin, *,
                       device: str = "cpu", dtype: str = "bfloat16"):
    """Create a strict callable; only invoking it hashes/loads local model weights.

    Keep snapshot/checkpoint immutable during loading. A new runtime dtype, WKV
    implementation or relocated config identity must have a matching qualified
    serving export; it cannot silently reuse another revision's qualification.
    """
    q, pin = deepcopy(qualification), deepcopy(pin)
    cfg = validate_reconstruction_metadata(q, pin)
    if dtype not in {"float32", "bfloat16"} or device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("unsupported explicit device/dtype policy")

    def load(artifact: Path, *, artifact_sha256: str) -> LoadedNativeParent:
        import torch
        from decide.calibration import FeatureTemperature
        from heads.backbone import BACKBONES, HiddenStateBackbone
        from heads.decide_backend import DecisionModelBackend
        from heads.model import DecisionModel
        from third_party.rwkv7_hf import verify_snapshot_code
        from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM
        from train.lineage import backbone_identity
        from train.run_cell import parent_encoder
        from train.trainer import prepare_model
        from .qualified_parent import canonical_sha256
        if artifact_sha256 != pin.artifact_sha256:
            raise ValueError("loader artifact pin mismatch")
        local = Path(snapshot).resolve(strict=True)
        if not local.is_dir():
            raise ValueError("snapshot must be a local directory")
        verify_snapshot(local, q["snapshot_files_sha256"])
        verify_snapshot_code(local)
        state = _load_checkpoint(artifact)
        if not isinstance(state, dict):
            raise ValueError("checkpoint must contain a Trainer state object")
        # Validate metadata before constructing the backbone, then validate all
        # tensors before any checkpoint state is installed.
        for key in ("config", "config_hash", "lineage", "lora_modules"):
            if canonical_sha256(state.get(key)) != canonical_sha256(q["checkpoint_" + key] if key != "lora_modules" else q[key]):
                raise ValueError(f"checkpoint {key} pin mismatch")
        if type(state.get("step")) is not int or state["step"] != pin.revision["checkpoint"]["step"]:
            raise ValueError("checkpoint step differs from serving revision")
        lm, loading = Rwkv7ForCausalLM.from_pretrained(
            str(local), dtype=getattr(torch, dtype), device_map={"": device},
            local_files_only=True, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("pretrained RWKV7 snapshot did not load its complete exact state")
        from heads.rwkv7_wkv import select_wkv
        select_wkv(lm, q["rwkv7_wkv"])
        backbone = HiddenStateBackbone(lm.rwkv7, spec=BACKBONES["rwkv7-13.3b"],
                                       pad_id=0, fork_mode=cfg.fork_mode)
        del lm  # release the unused language vocabulary head before reconstruction
        # Use exactly the implementation in the served config, with no inferred
        # alternative. The unchanged pinned config defaults to chunked.
        expected_config = q["checkpoint_lineage"]["backbone"]
        enc = parent_encoder("rwkv7", str(local))
        model = DecisionModel(backbone, cfg.head, cfg.head_kwargs, pad_id=0, seed=cfg.seed)
        model.to(backbone.device)
        modules = prepare_model(model, cfg)
        if canonical_sha256(backbone_identity(backbone, full=expected_config["weights_digest_mode"] == "full")) != canonical_sha256(expected_config):
            raise ValueError("reconstructed frozen backbone lineage mismatch")
        restore_trainables(model, state, q, cfg, modules, step=pin.revision["checkpoint"]["step"])
        backend = DecisionModelBackend(model, enc, template=pin.revision["prompt_template"]["id"],
                                       layout=cfg.layout, checkpoint=deepcopy(pin.revision["checkpoint"]),
                                       shared_prefix=pin.runtime["shared_prefix"])
        verify_snapshot(local, q["snapshot_files_sha256"])
        return LoadedNativeParent(backend, FeatureTemperature.from_dict(pin.calibration),
                                  pin.runtime["backend_max_input_tokens"],
                                  pin.runtime["enable_experimental_provenance"],
                                  pin.runtime["enable_v0_1_deltas"])
    return load
