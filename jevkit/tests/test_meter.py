import json

import pytest

from jevkit import meter as M
from jevkit import scoring as SC

from ._fixtures import choice, mixed_set

SAGE = M.PriceSource("list_tariff", 0.05, 10.0, "https://levanto.ai/pricing (read 2026-10-05)", "2026-10-05",
                     listed_since="2026-08-01")


def test_price_rules():
    SAGE.validate()
    with pytest.raises(M.PriceError, match="30"):
        M.PriceSource("list_tariff", 0.05, 10, "x", "2026-10-05", listed_since="2026-09-20").validate()
    with pytest.raises(M.PriceError, match="source"):
        M.PriceSource("size_class_estimate", 0.1, 0.1, " ", "2026-10-05").validate()
    with pytest.raises(M.PriceError):
        M.PriceSource("free", 0, 0, "x", "2026-10-05").validate()


def test_parse_price_string_and_json(tmp_path):
    p = M.parse_price("kind=base_model_reference,in=0.42,out=3.0,source=openrouter qwen3.8-27b,as_of=2026-10-05")
    assert p.kind == "base_model_reference" and p.usd_per_m_output == 3.0
    f = tmp_path / "price.json"
    f.write_text(json.dumps({"price": SAGE.to_dict()}), encoding="utf-8")
    assert M.parse_price(str(f)) == SAGE
    with pytest.raises(M.PriceError, match="missing"):
        M.parse_price("kind=list_tariff,in=1,out=1")


def test_cost_per_1000_and_floor():
    recs = [{"id": "a", "input_tokens": 400, "output_tokens": 1}, {"id": "b", "input_tokens": 600, "output_tokens": 3},
            {"id": "long", "input_tokens": 80000, "output_tokens": 1}]
    usd = M.cost_per_1000(recs, SAGE, exclude_ids={"long"})
    assert usd == pytest.approx(1000 * ((400 * 0.05 + 10) / 1e6 + (600 * 0.05 + 30) / 1e6) / 2)
    ref = M.PriceSource("base_model_reference", 0.2, 0.2, "openrouter base", "2026-10-05")
    chosen, rule = M.scoring_price(recs[:2], SAGE, ref)
    assert chosen is ref and "floor" in rule  # 0.2/M input beats 0.05/M + tiny output
    cs = M.cost_summary(recs, SAGE, None, exclude_ids={"long"})
    assert cs["score"] == pytest.approx(SC.cost_score(usd)) and cs["n_decisions"] == 2 and cs["kind"] == "list"


def test_measure_serial_with_fake_clock_and_warmup():
    t = {"now": 0.0}
    calls = []

    def clock():
        return t["now"]

    def answer(it):
        calls.append(it.id)
        t["now"] += 0.1 if it.id.endswith("0") else 0.3
        return {"status": "ok", "usage": {"input_tokens": 10, "output_tokens": 0}}

    its = [choice(f"c{i}") for i in range(4)]
    recs = M.measure(answer, its, clock=clock, warmup=2)
    assert calls[:2] == ["c0", "c1"] and len(recs) == 4  # warm-up requests are not recorded
    assert [r["latency_s"] for r in recs] == pytest.approx([0.1, 0.3, 0.3, 0.3])
    lat = M.latency_summary([r["latency_s"] for r in recs], self_hosted=True)
    assert lat["p50_s_raw"] == pytest.approx(0.3) and lat["p50_s_adjusted"] == pytest.approx(0.75)
    assert lat["score"] == pytest.approx((SC.speed_f(0.75) + SC.speed_f(lat["p95_s_adjusted"])) / 2)


def test_timed_subset_is_open_standard_and_judge():
    its = mixed_set("public") + mixed_set("private")
    ts = M.timed_subset(its)
    assert ts and all(i.side == "open" and i.tier in ("standard", "judge") for i in ts)
    assert len(ts) == 2 * 3 * 3
