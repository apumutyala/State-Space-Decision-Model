"""(c) golden: the base branch's (stack/qwen35/jev1-4b @ 5e159437) losses for tests/test_nupi.py's ``base()`` config on
its ``toy`` fixture (12 steps, seed 0), as float.hex. Run with cwd = a git archive of 5e159437; prints one JSON list.
Mirrors test_off_mode_is_bit_identical_to_the_base_branch exactly: toy items first, then torch.manual_seed(0) + model."""
import json
import tempfile

import torch

from heads import ByteEncoder, DecisionModel
from heads.backbone import tiny_backbone
from train.data import TOY_TEMPLATE, items_from_records, toy_records
from train.guard import SplitGuard
from train.trainer import Trainer, TrainConfig

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("nupi golden: train.data.toy_records only")
toy = items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)
torch.manual_seed(0)
model = DecisionModel(tiny_backbone(seed=0, hidden=16), "snapshot")
with tempfile.TemporaryDirectory() as d:
    cfg = TrainConfig(head="snapshot", lora_rank=4, batch_size=8, max_steps=12, lr_head=1e-2, lr_lora=1e-3,
                      ckpt_dir=d + "/a", ckpt_every=6)
    tr = Trainer(model, cfg, toy, guard=G)
    tr.train()
    assert len(tr.history) == 12, tr.history
    print(json.dumps([float.hex(x["loss"]) for x in tr.history]))
