import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = ROOT / "fixtures" / "replay" / "v0"


def base_request():
    return {
        "model": "test",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        },
    }


def shuffled(request, seed):
    """Same request with question order and choice-option order permuted."""
    rng = random.Random(seed)
    qitems = list(request["questions"].items())
    rng.shuffle(qitems)
    out = dict(request)
    out["questions"] = {}
    for qid, q in qitems:
        q = dict(q)
        if q["type"] == "choice":
            items = list(q["criteria"].items())
            rng.shuffle(items)
            q["criteria"] = dict(items)
        out["questions"][qid] = q
    return out


@pytest.fixture
def request_dict():
    return base_request()


def pytest_configure(config):
    config.addinivalue_line("markers", "shell_runner: executes a jobs/*.sh runner under bash on CPU (minutes; "
                                       "deselect with -m 'not shell_runner')")
