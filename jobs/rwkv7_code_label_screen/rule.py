"""Preregistered integrity screen only; never a promotion or Clef verdict."""
RULE_VERSION = "rwkv7-code-label-v2-screen-v1"
EDITION = "0.2.1"
ROWS = 2008
MIN_OK_ROWS = 2000
ROWS_SHA256 = "406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2"
IDS_SHA256 = "6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3"
MANIFEST_SHA256 = "26cd6bc0194efdb7a344b9313909483ad0bc9ab739beb26cbae698fc980b964a"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261007
CI_LEVEL = 0.95
PROBABILITY_TOLERANCE = 1e-9
REQUIRED_FLOORS = ("safety", "calibration", "abstention", "control", "interpretability", "clef", "clef_flash")


def verdict(integrity_ok):
    # Even positive v2-v1 CIs and a complete artifact do not qualify this model.
    return {"screen_status": "complete" if integrity_ok else "invalid",
            "promotable": False, "beats_clef": False, "spend_authorized": False,
            "floors": {name: "unmeasured" for name in REQUIRED_FLOORS},
            "reason": "Integrity screen only; matched Clef and all decision/safety floors remain unmeasured."}
