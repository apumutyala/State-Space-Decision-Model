"""jobs/jev_host_scan mode h2o_eval (EV-PREREG-H2O-HOST): CPU checks of the runner, slice rule, pins and T fit."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "jobs/jev_host_scan/h2o_eval.sh"
PARAMS = ROOT / "jobs/jev_host_scan/job_params_h2o_eval.json"


def test_runner_parses():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    r = subprocess.run([bash, "-n", str(ROOT / "jobs/jev_host_scan/h2o_eval.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_params_pins_and_slice_file():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    c = p["h2o_eval"]
    assert c["revision"] == "0940cbd7f3d2c9ee618e5dfaa3c74bfb765b12de"
    assert p["model"].endswith("@" + c["revision"])
    ids = (ROOT / c["slice"]["ids_file"]).read_bytes()
    assert hashlib.sha256(ids).hexdigest() == c["slice"]["ids_sha256"]
    lines = ids.decode().split()
    assert len(lines) == c["slice"]["n"] == len(set(lines)) and lines == sorted(lines)
    assert "LICENSE" in c["files"] and "model.safetensors" in c["files"]
    from jobs.common import costs

    assert costs.worst_case_usd(p) <= 12.0 <= p["hard_cap_usd"] + 1e-9


def test_slice_rule_deterministic(tmp_path):
    from jobs.jev_host_scan import h2o_eval as H

    recs = [{"id": f"r{i}", "aug": {"jev1": {"family8": "fa" if i % 2 else "fb"}}} for i in range(40)]
    f = tmp_path / "r.jsonl.gz"
    with gzip.open(f, "wt", encoding="utf-8") as g:
        for r in recs:
            g.write(json.dumps(r) + "\n")
    a = H.select_slice(f, "salt", 5)
    b = H.select_slice(f, "salt", 5)
    assert a == b and a["n"] == 10 and set(a["families"]) == {"fa", "fb"}
    want = sorted((hashlib.sha256(("salt\n" + r["id"]).encode()).hexdigest(), r["id"]) for r in recs
                  if r["aug"]["jev1"]["family8"] == "fa")[:5]
    assert a["families"]["fa"] == [i for _, i in want]


def test_git_blob(tmp_path):
    from jobs.jev_host_scan import h2o_eval as H

    f = tmp_path / "x"
    f.write_bytes(b"hello\n")
    assert H._git_blob(f) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object` of "hello\n"


def test_fit_t_recovers_temperature():
    from jobs.jev_host_scan import h2o_eval as H

    rnd = random.Random(0)
    pairs = []
    for _ in range(3000):
        lp = [rnd.gauss(0, 2) for _ in range(4)]
        p = [math.exp(x / 2.0) for x in lp]  # gold drawn from softmax(lp / 2)
        s = sum(p)
        u, acc, g = rnd.random() * s, 0.0, 0
        for k, x in enumerate(p):
            acc += x
            if u <= acc:
                g = k
                break
        pairs.append((lp, g))
    fit = H.fit_t(pairs)
    assert 1.7 < fit["T"] < 2.3 and fit["nll"] <= fit["nll_T1"]


def test_gold_index():
    from jobs.jev_host_scan import h2o_eval as H

    assert H.gold_index({"labels": {"q": "true"}}, {"qid": "q", "type": "noul", "names": ["true", "false"],
                                                    "descs": ["", ""]}) == 0
    assert H.gold_index({"labels": {"q": "b"}}, {"qid": "q", "type": "choice", "names": ["a", "b"],
                                                 "descs": ["", ""]}) == 1
    assert H.gold_index({"labels": {"q": 2}}, {"qid": "q", "type": "score", "names": ["0", "1", "2"],
                                               "descs": ["x", "y", "z"]}) == 2
    assert H.gold_index({"labels": {}}, {"qid": "q", "type": "choice", "names": ["a"], "descs": [""]}) is None
