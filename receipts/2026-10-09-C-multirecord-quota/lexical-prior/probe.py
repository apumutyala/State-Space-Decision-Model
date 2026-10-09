"""Actual D13 lexical function and C world, tiny frozen embedding, no backbone/model load."""
import ast
from copy import deepcopy
import difflib
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
HERE = Path(__file__).resolve().parent
D13 = Path("C:/Users/apumu/research/CHLabs/ssdm-wt/d13")
DATA = Path("C:/Users/apumu/research/CHLabs/ssdm-data")
sys.path.insert(0, str(D13))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


PINS = {"heads/model.py": "8b5129071642c7dc8230c7fff811d160714fadc985047da96f9f12db2fe90c7c",
        "heads/heads.py": "5aa120cdf62ce9981c8827d6ecb20673a7e4f4013cd71061a7a009500d31d918",
        "heads/layout.py": "edc66905324f38e26579bc22f16c7ffddb54faa23b46c2c520d50752e7d51ee1",
        "train/run_cell.py": "d406d8281765e8919d61591d66f08ffc5c2af506bc1db53775fb657e0ab1f82e"}
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=D13, text=True).strip() == "8d92f04f6bbc46ec40cf19b2ab5786435e139266"
for name, pin in PINS.items():
    assert sha((D13 / name).read_bytes()) == pin, name

import torch
from torch import nn
from heads.layout import G1V2_LAYOUT, build_row
from heads.features import HeadInput
from heads.heads import JointRecordHead
import heads.model as model_module
from train.losses import LossConfig, decision_loss
from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

torch.set_num_threads(1)
tokenizer_dir = DATA / "tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba"
tokenizer = Rwkv7Tokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
tree = ast.parse((D13 / "train/run_cell.py").read_text(encoding="utf-8"))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RwkvSegmentEncoder")
environment = {"hashlib": hashlib, "json": json}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(D13 / "train/run_cell.py"), "exec"), environment)
encoder = environment["RwkvSegmentEncoder"](tokenizer, set(tokenizer.all_special_ids))

artifact = DATA / "c-multirecord-7k-v2-allk-quota/train_set"
pointer_raw = (artifact / "pointer.json").read_bytes()
assert sha(pointer_raw) == "2c4659464e3ddaab28145b45628c59081964300f64d855f402edc1f6e1b81618"
pointer = json.loads(pointer_raw)
with gzip.open(artifact / "records.jsonl.gz", "rb") as stream:
    for raw in stream:
        record = json.loads(raw)
        choices = [(qid, role) for qid, role in record["aug"]["question_roles"].items() if role["kind"] == "top_profile"]
        if choices[0][1]["k"] == 77:
            break
    else:
        raise AssertionError("missing K77 C world")
qid = choices[0][0]
with gzip.open(artifact / pointer["manifest"], "rb") as stream:
    entry = next(json.loads(line) for line in stream if json.loads(line)["id"] == record["id"])
assert sha(raw[:-1]) == entry["sha256"]
request = {"model": "m2", "state": record["state"], "questions": record["questions"]}
rows = [build_row(request, question, encoder, layout="g1v2", template="jrt-rwkv-g1-v3w") for question in sorted(request["questions"])]
choice = next(row for row in rows if row.qid == qid)
assert choice.n_options == 77 and all(row.option_spans for row in rows)

# Tiny deterministic frozen table: 65,536 token identities x eight scalars (~2 MiB).
tokens = torch.arange(65536, dtype=torch.float32)[:, None]
dimensions = torch.arange(1, 9, dtype=torch.float32)[None, :]
table = nn.Embedding.from_pretrained(torch.sin(tokens * dimensions / 97.0), freeze=True)
backbone = SimpleNamespace(model=nn.Sequential(table))
baseline = model_module.lexical_vectors
source_text = (D13 / "heads/model.py").read_text(encoding="utf-8")
proposed = source_text.replace("from .layout import EncodeFn, LayoutRow, build_row", "from .layout import G1V2_LAYOUT, EncodeFn, LayoutRow, build_row")
proposed = proposed.replace('if r.layout == "ordered" and r.option_spans:', 'if r.layout in ("ordered", G1V2_LAYOUT) and r.option_spans:')
proposed = proposed.replace("(ordered layout: the option's schema span; else", "(ordered/g1v2 layouts: the option's schema span; else")
assert proposed != source_text
patch_text = "".join(difflib.unified_diff(source_text.splitlines(keepends=True), proposed.splitlines(keepends=True), fromfile="a/heads/model.py", tofile="b/heads/model.py"))
(HERE / "lexical-prior.patch").write_text(patch_text, encoding="utf-8", newline="\n")
node = next(n for n in ast.parse(proposed).body if isinstance(n, ast.FunctionDef) and n.name == "lexical_vectors")
environment = {**model_module.__dict__, "G1V2_LAYOUT": G1V2_LAYOUT}
exec(compile(ast.Module(body=[node], type_ignores=[]), "proposed-heads-model.py", "exec"), environment)
corrected = environment["lexical_vectors"]


def expected(rows, kmax):
    output = torch.zeros(len(rows), kmax, 8)
    for i, row in enumerate(rows):
        for k, (a, b) in enumerate(row.option_spans):
            output[i, k] = table(torch.tensor(row.ids[a:b])).mean(0)
    return output


ks = [row.n_options for row in rows]
kmax = max(ks)
old, new, wanted = baseline(backbone, rows, kmax), corrected(backbone, rows, kmax), expected(rows, kmax)
assert torch.equal(new, wanted) and not torch.allclose(old, wanted)
changed = deepcopy(request)
oid = choice.option_ids[0]
changed["questions"][qid]["criteria"][oid] = str(changed["questions"][qid]["criteria"][oid]) + "; independent lexical intervention fixture"
changed_choice = build_row(changed, qid, encoder, layout="g1v2", template="jrt-rwkv-g1-v3w")
assert choice.option_ids == changed_choice.option_ids and choice.continuations == changed_choice.continuations
old_changed = baseline(backbone, [changed_choice], 77)
new_changed = corrected(backbone, [changed_choice], 77)
old_choice = baseline(backbone, [choice], 77)
new_choice = corrected(backbone, [choice], 77)
assert torch.equal(old_choice, old_changed) and not torch.equal(new_choice, new_changed)
# Isolate answer-ID features at representation level while keeping the actual C schema spans fixed.
id_only = deepcopy(choice)
id_only.continuations = [[(token + 1) % 65536 for token in seq] for seq in choice.continuations]
old_id_only = baseline(backbone, [id_only], 77)
new_id_only = corrected(backbone, [id_only], 77)
assert not torch.equal(old_choice, old_id_only) and torch.equal(new_choice, new_id_only)

length = max(len(row.ids) for row in rows)
h, mask, evidence, q_pre, q_end = torch.zeros(7, length, 8), torch.zeros(7, length, dtype=torch.bool), torch.zeros(7, length, dtype=torch.bool), [], []
fork = torch.zeros(7, kmax, 8)
for i, row in enumerate(rows):
    h[i, :len(row.ids)] = table(torch.tensor(row.ids))
    mask[i, :len(row.ids)] = True
    evidence[i, row.state_span[0]:row.state_span[1]] = True
    q_pre.append(h[i, row.field_span[0]:row.field_span[1]].mean(0)); q_end.append(h[i, row.answer_pos])
    for k, seq in enumerate(row.continuations):
        fork[i, k] = table(torch.tensor(seq)).mean(0)
option_mask = torch.arange(kmax)[None, :] < torch.tensor(ks)[:, None]
x = HeadInput(h=h, token_mask=mask, evidence_mask=evidence, q_pre=torch.stack(q_pre), q_end=torch.stack(q_end), e=wanted,
              option_mask=option_mask, fork=fork, ks=ks, lex=new, groups=[list(range(7))])
torch.manual_seed(19)
head = JointRecordHead(8, hidden=16, width=8)
typed = model_module.TypedOutput(8)
lp = typed(head(x).logits, [row.qtype for row in rows], ks, wanted)
labels = torch.tensor([row.option_ids.index(str(record["labels"][row.qid]).lower() if row.qtype == "noul" else str(record["labels"][row.qid])) for row in rows])
loss = decision_loss(lp, labels, [row.qtype for row in rows], LossConfig(arm="clef"))
loss.backward()
assert torch.isfinite(lp[option_mask]).all() and torch.isneginf(lp[~option_mask]).all()
assert torch.isfinite(head.lex_scale.grad) and head.lex_scale.grad.abs() > 0
assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters() if parameter.grad is not None)
assert table.weight.requires_grad is False and table.weight.grad is None
report = {"schema": "ssdm.c_joint_lexical_prior_probe.v1", "status": "pass", "source_commit": "8d92f04f6bbc46ec40cf19b2ab5786435e139266",
          "source_sha256": PINS, "proposed_model_source_sha256": sha(proposed.encode()), "patch_sha256": sha((HERE / "lexical-prior.patch").read_bytes()),
          "record_id": record["id"], "record_sha256": entry["sha256"], "pointer_sha256": sha(pointer_raw),
          "questions": 7, "qtypes": [row.qtype for row in rows], "options": ks, "layout": "g1v2", "template": "jrt-rwkv-g1-v3w",
          "baseline_option_span_max_abs_difference": float((old-wanted).abs().max()), "corrected_option_span_max_abs_difference": float((new-wanted).abs().max()),
          "description_change_fixed_ids_baseline_vector_changed": not torch.equal(old_choice, old_changed),
          "description_change_fixed_ids_corrected_vector_changed": not torch.equal(new_choice, new_changed),
          "id_representation_change_fixed_schema_baseline_vector_changed": not torch.equal(old_choice, old_id_only),
          "id_representation_change_fixed_schema_corrected_vector_changed": not torch.equal(new_choice, new_id_only),
          "mixed_question_loss": float(loss.detach()), "lex_scale_gradient": float(head.lex_scale.grad),
          "valid_log_probs_finite": True, "padding_negative_infinity": True, "head_gradients_finite": True, "frozen_table_gradient_absent": True,
          "table": "deterministic sine features, 65536x8; numerical plumbing fixture only", "model_downloaded_or_loaded": False,
          "gpu_used": False, "b_inputs_read": False, "protected_rows_read": False, "index_gain_measured": False,
          "probe_sha256": sha(Path(__file__).read_bytes()), "scope": "lexical vectors and mixed-qtype JointRecordHead backward only; does not fix separate features ordinal path or establish model performance"}
(HERE / "proof.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
