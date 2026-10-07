"""TEST-ONLY launcher for the first-cell CPU dry run (scripts/m2_first_cell_dryrun.py): ``python shim.py <module> args``
runs ``python -m <module> args`` after these dry-run substitutions that never exist in a paid run:

1. the pinned Decision Index kit clone is put on ``sys.path`` (the job image pip-installs it);
2. ``third_party.rwkv7_hf.verify_snapshot_code`` accepts the TINY random RWKV7 snapshot: its code files are the vendored
   ones, but its config.json is a 2-layer config, so the 13.3B config pin cannot match;
3. with ``DRYRUN_ROLE_N`` set, every cohort role keeps only its first N entries (CPU time; the cohort file, its sha256
   and every identity check are unchanged);
4. without ``accelerate`` installed locally, ``device_map={"": "cpu"}`` is dropped (it means the same placement);
5. the 13.3B spec's LoRA group counts (61 layers) are scaled to the tiny model's layers, as apply_lora checks them.
Nothing in the job's code paths reads these; the shim only wraps the entry points.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m2_e2e_dryrun import ensure_kit  # noqa: E402

ensure_kit()
import third_party.rwkv7_hf as V  # noqa: E402

V.verify_snapshot_code = lambda model_dir: {f: "tiny-dryrun (pin check bypassed by the test shim)" for f in V.CODE_SHA256}
try:  # 4. local venvs without `accelerate` (the job image pins accelerate 1.15.0): device_map {"": "cpu"} == no map
    import accelerate  # noqa: F401
except ImportError:
    from transformers import PreTrainedModel

    _fp = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, *a, **k):
        if k.get("device_map") == {"": "cpu"}:
            k.pop("device_map")
        return _fp(cls, *a, **k)

    PreTrainedModel.from_pretrained = _from_pretrained
import dataclasses  # noqa: E402

from heads import backbone as _B  # noqa: E402

_s = _B.BACKBONES["rwkv7-13.3b"]  # 5. LoRA group counts (apply_lora checks them) scaled from 61 layers to the tiny 2
_TINY_LAYERS = int(os.environ.get("DRYRUN_TINY_LAYERS") or 2)
_B.BACKBONES["rwkv7-13.3b"] = dataclasses.replace(
    _s, lora_groups=[(g, rx, c * _TINY_LAYERS // _s.n_layers) for g, rx, c in _s.lora_groups])
_n = int(os.environ.get("DRYRUN_ROLE_N") or 0)
if _n:
    from train import cohort as C

    _init = C.Cohort.__init__

    def _truncated(self, *a, **k):
        _init(self, *a, **k)
        for r in self.roles:
            self.roles[r] = self.roles[r][:_n]

    C.Cohort.__init__ = _truncated

if __name__ == "__main__":
    mod = sys.argv[1]
    sys.argv = [mod] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
