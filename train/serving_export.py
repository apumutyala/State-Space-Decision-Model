"""Additive M2 artifact metadata for exact serving reconstruction; no inference.

This exports the existing calibrated decider, never a replacement calibration or
context limit. Strict weight reconstruction and promotion remain separate gates.
See interfaces/M2_SERVING_BUNDLE_V1.md and interface issue #185.
"""

from __future__ import annotations

import json

from decide.calibration import FeatureTemperature

SCHEMA = "ssdm.m2.serving_bundle.v1"


def export_serving_bundle(decider) -> dict:
    """Snapshot the full fitted calibration, served revision and runtime settings.

    The backend revision is already computed by the M2 report path. This function
    does not open checkpoint files, reconstruct weights, fit or score anything.
    Returned objects are JSON copies, so consumers cannot mutate the live scorer.
    """
    calibration = decider.calibration
    if not isinstance(calibration, FeatureTemperature):
        raise TypeError("M2 serving export requires explicit FeatureTemperature")
    calibration.assert_servable()
    revision = decider.revision()
    # A temperature identity omits the calibration content needed by from_dict.
    payload = calibration.to_dict()
    restored = FeatureTemperature.from_dict(payload)
    if restored.identity() != revision.get("calibration"):
        raise ValueError("M2 served revision does not bind the exported calibration")
    digest = revision.get("checkpoint", {}).get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("M2 serving export requires the checkpoint sha256")
    backend = decider.backend
    runtime = {
        "fork_mode": backend.model.backbone.fork_mode,
        "shared_prefix": backend.shared_prefix,
        "backend_max_input_tokens": decider.backend_max_input_tokens,
        "enable_experimental_provenance": decider.enable_experimental_provenance,
        "enable_v0_1_deltas": decider.enable_v0_1_deltas,
    }
    return json.loads(json.dumps({"schema": SCHEMA, "artifact_sha256": digest,
                                 "revision": revision, "calibration": payload,
                                 "runtime": runtime}, allow_nan=False))
