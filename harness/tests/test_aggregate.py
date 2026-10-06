"""Our index weights reproduce the live 0.2.1 board from published per-benchmark values."""

import json

import numpy as np
import pytest

from decision_index.scoring import index02 as X

from harness import config, score
from harness.suite import spec

FIXTURE = config.KIT_CLONE / "tests" / "fixtures" / "board-0.2.1.json"
# Live board values used by the kit's own tests (tests/test_index021.py @ KIT_COMMIT).
LIVE = {"jev": 57.89, "rune-26b-a4b-v3": 57.44, "autojev-27b": 56.4, "jebadiah-27b": 54.67, "reflex-27b-v2": 52.16, "jpt-9b": 46.89, "hopper": 39.67, "lev": 38.54, "pngwn-space-uncapped-full": 29.91, "jpt-0.8b": 19.22, "lavoir": 8.69, "verdict-8af2496e": 1.87, "lumma-fev-0.1b": 1.78}


def test_panel_is_38_benchmarks_with_published_area_weights():
    s = spec()
    ids = [n for a in s["areas"] for n in a["benchmarks"]]
    assert len(ids) == len(set(ids)) == 38
    w = score.weights()
    assert sum(w.values()) == pytest.approx(1.0)
    areas = {a["id"]: sum(w[n] for n in a["benchmarks"]) for a in s["areas"]}
    assert {k: round(v, 4) for k, v in areas.items()} == {"knowledge": 0.2585, "language": 0.2585, "retrieval": 0.2002, "tools": 0.1828, "arts": 0.1}
    # gold benchmarks weigh 1.2x a plain one inside the same area
    assert w[57] / w[30] == pytest.approx(1.2)


def test_renormalised_weights_drop_missing_benchmark_inside_its_area():
    present = set(n for a in spec()["areas"] for n in a["benchmarks"]) - {45}
    w = score.weights(present, renormalize=True)
    assert 45 not in w and sum(w.values()) == pytest.approx(1.0)
    k = [a for a in spec()["areas"] if a["id"] == "knowledge"][0]["benchmarks"]
    assert sum(w[n] for n in k if n != 45) == pytest.approx(0.258494, abs=1e-6)


@pytest.mark.skipif(not FIXTURE.exists(), reason="kit clone not present")
@pytest.mark.parametrize("engine", sorted(LIVE))
def test_index_from_skills_reproduces_live_board(engine):
    board = json.loads(FIXTURE.read_text())["entrants"][engine]
    s = spec()
    skills = {}
    for n, b in board["benchmarks"].items():
        n = int(n)
        if n in s["track_scored"]:
            v = X.benchmark_value(n, s, track={"catalog_id": n, "raw": b["board"]["raw"], "skill": b["board"]["skill"], "coverage": b["board"]["coverage"], "tracks": []})
        else:
            v = X.benchmark_value(n, s, native=b["result"])
        skills[n] = np.array([v["skill"]])
    assert float(score.index_from_skills(skills, renormalize=False)[0]) == pytest.approx(LIVE[engine], abs=0.01)
