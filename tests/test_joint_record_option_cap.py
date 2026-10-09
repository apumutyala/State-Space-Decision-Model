"""The joint continuation call must respect its cap across prefix groups."""
from types import SimpleNamespace
import pytest

from train.trainer import Trainer, item_record


def sampler(records, cap):
    trainer = object.__new__(Trainer)
    trainer.cfg = SimpleNamespace(row_options=cap)
    trainer.train_items = [SimpleNamespace(id=f"r{r}/q{i}", row=SimpleNamespace(n_options=k))
                           for r, ks in enumerate(records) for i, k in enumerate(ks)]
    trainer.epoch = trainer.cursor = 0
    trainer._order = lambda epoch: list(range(len(trainer.train_items)))
    return trainer


def test_c_seven_question_record_remains_whole_without_169_continuation_call():
    trainer = sampler([[77, 5, 2, 2, 2, 2, 2], [77]], 160)
    first = trainer._next_record_rows(8)
    assert len(first) == 7 and sum(i.row.n_options for i in first) == 92
    assert {item_record(i) for i in first} == {"r0"}
    assert trainer.cursor == 7
    resumed = sampler([[77, 5, 2, 2, 2, 2, 2], [77]], 160)
    resumed.cursor = trainer.cursor
    second = trainer._next_record_rows(8)
    assert [i.id for i in second] == [i.id for i in resumed._next_record_rows(8)] == ["r1/q0"]
    assert trainer.cursor == 8
    assert [i.id for i in trainer._next_record_rows(8)] == [i.id for i in first]
    assert trainer.epoch == 1


@pytest.mark.parametrize("cap", [0, 169, 1000])
def test_unlimited_or_fitting_steps_keep_all_existing_rows(cap):
    trainer = sampler([[77, 5, 2, 2, 2, 2, 2], [77]], cap)
    rows = trainer._next_record_rows(8)
    assert len(rows) == 8 and sum(i.row.n_options for i in rows) == 169
    assert trainer.cursor == 8 and trainer.epoch == 0


def test_oversized_record_refuses_before_consumption_and_does_not_erase_prior_batch():
    trainer = sampler([[2], [100, 100]], 160)
    assert [i.id for i in trainer._next_record_rows(8)] == ["r0/q0"]
    assert trainer.cursor == 1
    with pytest.raises(ValueError, match="joint record exceeds row_options"):
        trainer._next_record_rows(8)
    assert trainer.cursor == 1 and trainer.epoch == 0
