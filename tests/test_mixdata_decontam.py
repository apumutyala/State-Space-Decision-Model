"""convert.mixdata decontamination filters: dev13-v1 (job-w3c2e, reproducible) vs full-eval-v1 (the new default).

Tiny synthetic mix (tests/q9af_corpus_check_stub.py sources: planted TEST / frozen / DEV / attack docs) and a synthetic
eval-hash index; CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("pyarrow")

from convert import mixdata as M  # noqa: E402
from jobs.q9af_corpus_check.corpus_check import FakeTok  # noqa: E402
from tests import q9af_corpus_check_stub as S  # noqa: E402
from train import doc_overlap_guard as G  # noqa: E402


@pytest.fixture()
def fx(tmp_path):
    rows = S.eval_rows()
    idx = G.build_index(None, G.load_allowlist(), rows_factory=lambda: iter(rows), n_parts=2)
    idx.meta["index_sha256"] = "synthetic"
    G.set_index_for_tests(idx)
    dev = tmp_path / "dev-rows.jsonl.gz"
    S.write_dev_rows(dev)
    roots = S.write_sources(tmp_path / "src")
    yield tmp_path, idx, str(dev), roots
    G.set_index_for_tests(None)


def _cfg(decontam="__absent__"):
    c = S.tiny_config()
    c["mix"].pop("decontam")
    if decontam != "__absent__":
        c["mix"]["decontam"] = decontam
    return c


def test_default_is_the_full_filter_and_w3c2e_records_the_old_one():
    assert M.DECONTAM_DEFAULT == M.DECONTAM_FULL == "full-eval-v1"
    assert M.decontam_of({}) == M.DECONTAM_FULL
    cfg = json.loads((ROOT / "configs/q9af_v2_h100.json").read_text(encoding="utf-8"))
    assert M.decontam_of(cfg["mix"]) == M.DECONTAM_DEV == "dev13-v1"
    with pytest.raises(SystemExit):
        M.decontam_of({"decontam": "dev-only"})


def test_dev13_build_is_unchanged_by_the_hook_and_keeps_w3c2es_shape(fx):
    tmp, _, dev, roots = fx
    a = M.build(_cfg("dev13-v1"), dev, str(tmp / "a"), tok=FakeTok(), roots=roots)
    seen = []
    b = M.build(_cfg("dev13-v1"), dev, str(tmp / "b"), tok=FakeTok(), roots=roots, on_doc=lambda *x: seen.append(x))
    strip = lambda m: {k: v for k, v in m.items() if k != "seconds"}
    assert strip(a) == strip(b) and "decontam" not in a
    assert set(a["sources"]["web"]) == {"tokens", "quota", "row_groups", "docs_seen", "docs_kept",
                                        "docs_dropped_dev_ngram", "files"}
    assert a["sources"]["web"]["docs_dropped_dev_ngram"] == 1
    kept = [x for x in seen if x[0] == "train" and x[7] is None]
    assert sum(x[6] for x in kept) == a["train_tokens"] == len(np.fromfile(tmp / "b/train.u32", dtype=np.uint32))
    assert [x[7] for x in seen if x[7]] == ["dev_ngram"]


def test_full_filter_drops_test_frozen_and_attack_docs(fx):
    tmp, idx, dev, roots = fx
    meta = M.build(_cfg(), dev, str(tmp / "full"), tok=FakeTok(), roots=roots)
    assert meta["decontam"] == "full-eval-v1" and meta["eval_index_sha256"] == "synthetic"
    assert meta["attack_templates_sha256"] and meta["decontam_rule"] == M.DECONTAM_RULES[M.DECONTAM_FULL]
    web, math, chat = (meta["sources"][s] for s in ("web", "math", "chat"))
    assert web["docs_dropped_dev_ngram"] == 1
    assert web["docs_dropped_eval_index"] == 2  # the copied TEST premise and the doc with 13 DEV words
    assert math["docs_dropped_eval_index"] == 1  # the frozen (BBH) input
    assert chat["docs_dropped_attack"] == 1
    # nothing planted survives in the stream
    tok = FakeTok()
    train = np.fromfile(tmp / "full/train.u32", dtype=np.uint32)
    for text in (S.TEST_DOC, S.FROZEN_DOC):
        ids = tok([text])["input_ids"][0][:13]
        assert not any((train[i:i + 13] == ids).all() for i in range(len(train) - 13))
    with pytest.raises(SystemExit):  # an existing build under another filter is never reused silently
        M.build(_cfg("dev13-v1"), dev, str(tmp / "full"), tok=FakeTok(), roots=roots)


def test_eval_flags_ignore_eval_side_templates_and_allowlisted_corpora(fx):
    _, idx, _, _ = fx
    tmpl = S.TEMPLATE  # 15 words in >= 100 eval rows (every synthetic row's instructions): a frame, not a document
    f = M.eval_flags([tmpl, "unrelated words " * 10, S.TEST_DOC], idx)
    assert f.tolist() == [0, 0, M.FLAG_EVAL]
    assert int(idx.df[idx.lookup(G.text_grams([tmpl]))].min()) >= G.TEMPLATE_EVAL_DF


def test_cli_override(fx, monkeypatch):
    tmp, _, dev, roots = fx
    cfgp = tmp / "cfg.json"
    cfgp.write_text(json.dumps(_cfg()))
    got = {}
    monkeypatch.setattr(M, "build", lambda cfg, d, o, decontam=None: got.update(decontam=decontam) or
                        {"sources": {}, "x": 1})
    assert M.main(["build", "--config", str(cfgp), "--dev-rows", dev, "--out", str(tmp / "o"), "--decontam",
                   "dev13-v1"]) == 0
    assert got["decontam"] == "dev13-v1"
