"""Proposed regression tests to copy into the C-owned integration branch."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from heads.layout import LayoutRow
from heads.model import lexical_vectors


def fixture(layout):
    table = nn.Embedding.from_pretrained(torch.arange(128, dtype=torch.float32).reshape(32, 4) / 17, freeze=True)
    backbone = SimpleNamespace(model=nn.Sequential(table))
    row = LayoutRow(qid="profile", qtype="choice", option_ids=["a", "b"], ids=[1, 2, 3, 4, 5],
                    state_span=(0, 1), field_span=(1, 5), option_spans=[(1, 3), (3, 5)],
                    option_seqs=[], continuations=[[20], [21]], layout=layout)
    return backbone, table, row


def test_g1v2_lexical_vectors_use_option_spans_and_ignore_answer_id_tokens():
    backbone, table, row = fixture("g1v2")
    expected = torch.stack([table(torch.tensor(row.ids[a:b])).mean(0) for a, b in row.option_spans])
    actual = lexical_vectors(backbone, [row], 3)
    assert torch.equal(actual[0, :2], expected)
    assert torch.equal(actual[0, 2], torch.zeros(4))
    assert not actual.requires_grad and table.weight.grad is None
    changed_ids = deepcopy(row)
    changed_ids.continuations = [[22], [23]]
    assert torch.equal(actual, lexical_vectors(backbone, [changed_ids], 3))
    changed_description = deepcopy(row)
    changed_description.ids[1:3] = [6, 7]
    changed = lexical_vectors(backbone, [changed_description], 3)
    assert not torch.equal(actual[0, 0], changed[0, 0])
    assert torch.equal(actual[0, 1:], changed[0, 1:])


def test_ordered_lexical_span_path_remains_identical():
    backbone, _, row = fixture("ordered")
    ordered = lexical_vectors(backbone, [row], 2)
    row.layout = "g1v2"
    assert torch.equal(ordered, lexical_vectors(backbone, [row], 2))


def test_symmetric_lexical_continuation_fallback_is_unchanged():
    backbone, table, row = fixture("symmetric")
    row.option_spans = []
    expected = table(torch.tensor([20, 21]))
    assert torch.equal(lexical_vectors(backbone, [row], 2)[0], expected)
