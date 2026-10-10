"""Small synthetic JevBench items for CPU tests (no network, no model)."""

from __future__ import annotations

from jevkit.items import Item


def choice(i: str, k: int = 4, gold: int = 0, tier: str = "standard", split: str = "public") -> Item:
    keys = [f"o{j}" for j in range(k)]
    return Item(id=i, family="fam_choice", state=f"state for {i}. It is not urgent.",
                question={"type": "choice", "instructions": "Which option applies? Do not guess.",
                          "criteria": {kk: f"Option {kk} description." for kk in keys}},
                labels=keys, expected=keys[gold], split=split, tier=tier).validate()


def noul(i: str, gold: str = "yes", tier: str = "standard", split: str = "public") -> Item:
    return Item(id=i, family="fam_noul", state={"note": f"record {i}", "detail": "It is approved."},
                question={"type": "noul", "instructions": "Is it approved?",
                          "criteria": {"true": "It is approved.", "false": "It is not approved."}},
                labels=["no", "yes"], expected=gold, split=split, tier=tier).validate()


def score(i: str, k: int = 4, gold: int = 1, tier: str = "standard", split: str = "public") -> Item:
    return Item(id=i, family="fam_score", state=f"incident {i}: cannot log in",
                question={"type": "score", "instructions": "Rate impact.",
                          "criteria": [f"level {j}" for j in range(k)]},
                labels=[str(j) for j in range(k)], expected=gold, split=split, tier=tier).validate()


def ch_ans(probs: dict) -> dict:
    return {"type": "choice", "probabilities": probs}


def nl_ans(p: float) -> dict:
    return {"type": "noul", "noul": p}


def sc_ans(probs: list[float]) -> dict:
    return {"type": "score", "probabilities": {str(i): p for i, p in enumerate(probs)}}


def mixed_set(split: str = "public") -> list[Item]:
    out = []
    for t in ("easy", "standard", "judge", "hard"):
        for n in range(3):
            out.append(choice(f"{split}-c-{t}-{n}", k=4, gold=n % 4, tier=t, split=split))
            out.append(noul(f"{split}-n-{t}-{n}", gold="yes" if n % 2 == 0 else "no", tier=t, split=split))
            out.append(score(f"{split}-s-{t}-{n}", k=4, gold=n % 4, tier=t, split=split))
    return out


def good_answers(items: list[Item], conf: float = 0.85) -> dict:
    """Mostly-right, roughly calibrated answers (deterministic)."""
    ans = {}
    for n, it in enumerate(items):
        wrong = n % 5 == 4
        if it.type == "choice":
            tgt = it.labels[(it.labels.index(it.expected) + (1 if wrong else 0)) % it.k]
            rest = (1 - conf) / (it.k - 1)
            ans[it.id] = ch_ans({lab: (conf if lab == tgt else rest) for lab in it.labels})
        elif it.type == "noul":
            yes = (it.expected == "yes") != wrong
            ans[it.id] = nl_ans(0.9 if yes else 0.1)
        else:
            g = int(it.expected) if not wrong else (int(it.expected) + 1) % it.k
            rest = (1 - conf) / (it.k - 1)
            ans[it.id] = sc_ans([conf if j == g else rest for j in range(it.k)])
    return ans
