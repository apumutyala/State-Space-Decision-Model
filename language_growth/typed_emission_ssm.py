"""Random-initialized CPU development diagnostic; checked L0 emission only.

Architecture reference: PR #78's local factorized query scorer was read only.
This minimal implementation has no dependency on that unmerged implementation.
Only the first, prefix-free development request supplies inference evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

from .development_data import generate_development_tasks, comparison_swap_counterfactual
from .kernel import check_program
from .realize import render_checked

KINDS = ("read", "explain", "compare")
RELATIONS = ("greater_than", "less_than", "equal")
CONFIG = {"seeds": [20261006, 20261016, 20261026], "width": 32,
          "steps": 300, "batch_size": 32, "lr": .003, "clip_norm": 1.,
          "train_seed": 9101, "train_count": 600, "development_seed": 9102,
          "development_clusters": 180, "cpu_threads": 2, "max_seconds": 180,
          "numeric_scales": {"price": 200., "warranty_months": 36.},
          "relation_features": ["left/scale", "right/scale", "delta", "delta_squared"],
          "loss": "operation CE + conditional target CE + comparison relation CE",
          "inference": "argmax operation, conditional argmax target, relation argmax on predicted operands",
          "scope": "One condition; final step only; iterative synthetic development; no validation tuning; random initialization; no parent retention claim"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def project(state):
    """Enumerate from permitted visible records, independent of question or gold.

    Every registered field has read/explain candidates and every ordered pair
    of distinct records with that field has a comparison candidate. All three
    relations are always offered by the learned relation head, with no truth
    filtering. No request IDs, option IDs, prefix, or context hash is encoded.
    """
    candidates = []
    for kind in KINDS:
        for left in sorted(state["records"], key=lambda r: r["record_ref"]):
            for field in sorted(left["fields"]):
                if field not in state["registry"]["registered_fields"]:
                    continue
                rights = ([None] if kind != "compare" else
                          [r for r in sorted(state["records"], key=lambda r: r["record_ref"])
                           if r["record_ref"] != left["record_ref"] and field in r["fields"]])
                for right in rights:
                    target = {"kind": kind, "field": field, "left": left["record_ref"],
                              "right": right["record_ref"] if right else None}
                    parts = (field.replace("_", " "), left["entity_ref"],
                             right["entity_ref"] if right else "", left["fields"][field]["temporal_scope"])
                    numeric = [0., 0., 0., 0.]
                    if right:
                        scale = CONFIG["numeric_scales"][field]
                        lv = float(left["fields"][field]["value"]) / scale
                        rv = float(right["fields"][field]["value"]) / scale
                        delta = lv - rv
                        numeric = [lv, rv, delta, delta * delta]
                    candidates.append({"target": target, "parts": parts, "numeric": numeric})
    return candidates


def compile_target(state, target):
    """Pure syntax compilation of model selections, without consulting labels."""
    nodes = [{"id": "n1", "op": "ReadFact", "record_ref": target["left"], "field": target["field"]}]
    if target["kind"] == "compare":
        nodes += [{"id": "n2", "op": "ReadFact", "record_ref": target["right"], "field": target["field"]},
                  {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"},
                   "right": {"node": "n2", "project": "object"}, "relation": target["relation"]},
                  {"id": "n4", "op": "Assert", "proposition": "n3"}]
    else:
        nodes += [{"id": "n2", "op": "Assert", "proposition": "n1"}]
        if target["kind"] == "explain":
            nodes += [{"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"}]
    return {"schema": "ssdm.semantic_program.v0.1", **{k: state[k] for k in ("context_ref", "policy_ref", "registry_ref")},
            "nodes": nodes, "outputs": [nodes[-1]["id"]], "style": "plain"}


def gold_target(program):
    """Supervision/scoring only, never called by predict or project."""
    nodes = program["nodes"]
    kind = "compare" if any(n["op"] == "Compare" for n in nodes) else (
        "explain" if any(n["op"] == "ExplainDerivation" for n in nodes) else "read")
    target = {"kind": kind, "field": nodes[0]["field"], "left": nodes[0]["record_ref"],
              "right": nodes[1]["record_ref"] if kind == "compare" else None}
    if kind == "compare":
        target["relation"] = nodes[2]["relation"]
    return target


def byte_batch(texts):
    encoded = [list(t.encode("utf-8")) for t in texts]
    ids = torch.zeros((len(texts), max(1, max(map(len, encoded)))), dtype=torch.long)
    for i, row in enumerate(encoded):
        if row:
            ids[i, :len(row)] = torch.tensor(row) + 1
    return ids


class TypedEmissionSSM(nn.Module):
    """Shared diagonal recurrence, factorized operation/target/relation heads.

    h_t = a*h_(t-1) + (1-a)*tanh(W*embedding(byte_t)); concatenated terminal
    and mean states. Vectorized geometric sums compute that recurrence exactly
    up to floating point. Relation MLP receives raw arithmetic features only.
    """
    def __init__(self):
        super().__init__()
        w = CONFIG["width"]
        self.embedding = nn.Embedding(257, w, padding_idx=0)
        self.input = nn.Linear(w, w)
        self.decay_logit = nn.Parameter(torch.logit(torch.linspace(.5, .999, w)))
        self.operation = nn.Linear(2 * w, 3)
        self.scorer = nn.Sequential(nn.Linear(18 * w, 32), nn.Tanh(), nn.Linear(32, 1))
        self.relation = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 3))

    def encode(self, ids):
        mask = ids.ne(0)
        length = mask.sum(1)
        pos = torch.arange(ids.shape[1])[None, :]
        lag = (length[:, None] - 1 - pos).clamp_min(0)
        remaining = (length[:, None] - pos).clamp_min(0)
        a = self.decay_logit.sigmoid()[None, None, :]
        u = self.input(self.embedding(ids)).tanh() * mask[:, :, None]
        terminal = (u * a.pow(lag[:, :, None]) * (1 - a)).sum(1)
        mean = (u * (1 - a.pow(remaining[:, :, None]))).sum(1) / length[:, None].clamp_min(1)
        return torch.cat((terminal, mean), 1)

    def forward(self, query_ids, component_ids, part_indices, numeric):
        query = self.encode(query_ids)
        parts = self.encode(component_ids)[part_indices].unbind(2)
        q = query[:, None, :].expand(-1, part_indices.shape[1], -1)
        scores = self.scorer(torch.cat((q, *parts, *(q * p for p in parts)), -1)).squeeze(-1)
        return self.operation(query), scores, self.relation(numeric)


def tensors(states):
    """Inference batch accepts visible states exclusively; no target argument."""
    candidates = [project(s) for s in states]
    assert len({len(cs) for cs in candidates}) == 1
    components = sorted({part for cs in candidates for c in cs for part in c["parts"]})
    ids = {part: i for i, part in enumerate(components)}
    indices = torch.tensor([[[ids[p] for p in c["parts"]] for c in cs] for cs in candidates])
    numeric = torch.tensor([[c["numeric"] for c in cs] for cs in candidates])
    kinds = torch.tensor([[KINDS.index(c["target"]["kind"]) for c in cs] for cs in candidates])
    return byte_batch([s["task_input"] for s in states]), byte_batch(components), indices, numeric, kinds, candidates


@torch.no_grad()
def predict(model, states):
    model.eval()
    q, components, indices, numeric, kinds, candidates = tensors(states)
    operation, score, relation = model(q, components, indices, numeric)
    op = operation.argmax(1)
    selected = score.masked_fill(kinds.ne(op[:, None]), -torch.inf).argmax(1)
    result = []
    for i, j in enumerate(selected.tolist()):
        target = dict(candidates[i][j]["target"])
        if target["kind"] == "compare":
            target["relation"] = RELATIONS[relation[i, j].argmax().item()]
        result.append(target)
    return result


def row(example, cluster, member):
    # First request only. Later generator requests contain authored gold prefixes.
    state = json.loads(example["model_inputs"][0]["state"])
    assert state["accepted_partial_program"] == []
    tr = example["training_only"]
    return {"id": example["task_id"] + ":" + member, "cluster": cluster, "member": member,
            "state": state, "evaluation_only": {"program": tr["program"], "host": tr["host"], "catalog": tr["catalog"]}}


def prepare(out):
    assert not out.exists()
    out.mkdir(parents=True)
    train = [row(t, t["task_id"], "original") for t in generate_development_tasks(CONFIG["train_seed"], CONFIG["train_count"])]
    development = []
    for task in generate_development_tasks(CONFIG["development_seed"], CONFIG["development_clusters"]):
        original = row(task, task["task_id"], "original")
        development.append(original)
        if task["training_only"]["kind"] == "compare":
            partner = row(comparison_swap_counterfactual(task), task["task_id"], "value_swap")
            assert partner["state"]["task_input"] == original["state"]["task_input"]
        else:
            partner = copy.deepcopy(original)
            partner["id"] += ":wording"
            partner["member"] = "wording"
            target = gold_target(partner["evaluation_only"]["program"])
            record = next(r for r in partner["state"]["records"] if r["record_ref"] == target["left"])
            field, name = target["field"].replace("_", " "), record["entity_ref"]
            partner["state"]["task_input"] = (f"Give the October 2026 {field} for {name}." if target["kind"] == "read"
                                                 else f"Show the checked derivation for {name}'s {field}.")
            # Ensure genuinely different paired wording using the other generator template.
            if partner["state"]["task_input"] == original["state"]["task_input"]:
                partner["state"]["task_input"] = (f"What is {name}'s {field} in October 2026?" if target["kind"] == "read"
                                                     else f"Explain the checked {field} fact for {name}.")
        development.append(partner)
    write(out / "frozen_rows.json", {"train": train, "development": development})
    sources = [__file__, str(Path(__file__).with_name("development_data.py")),
               str(Path(__file__).with_name("kernel.py")), str(Path(__file__).with_name("realize.py")),
               str(Path(__file__).with_name("parent_bridge.py")), str(Path(__file__).with_name("types.py"))]
    write(out / "registration.json", {"config": CONFIG, "source_hashes": {Path(p).name: sha(p) for p in sources},
        "source": str(Path(__file__).resolve()), "base_commit": "24f164e9cfff1abe8f7eecef68a1771d357ca465",
        "frozen_data_sha256": sha(out / "frozen_rows.json"), "paired_task_clusters": CONFIG["development_clusters"],
        "architecture_reference_only": "PR #78 local source at 59e6622ba121676ea75cdbc8e580d0623657de01; no unmerged imports",
        "candidate_contract": "12 structural candidates (4 each operation), three relation options for each comparison, no truth filtering",
        "initialization": "random; no teacher or parent checkpoint", "cost_usd": 0})


def summary(observations):
    def aggregate(rows):
        return {"count": len(rows), **{k: sum(r[k] for r in rows) for k in
            ("target_exact", "kernel_accepted", "whole_answer_correct", "operation_correct", "structure_exact", "relation_correct")}}
    return {"all": aggregate(observations), "by_kind": {kind: aggregate([r for r in observations if r["gold"]["kind"] == kind]) for kind in KINDS}}


def evaluate(model, rows):
    observations = []
    for start in range(0, len(rows), CONFIG["batch_size"]):
        batch = rows[start:start + CONFIG["batch_size"]]
        predictions = predict(model, [r["state"] for r in batch])
        for r, predicted in zip(batch, predictions):
            ev = r["evaluation_only"]
            gold = gold_target(ev["program"])
            program = compile_target(r["state"], predicted)
            checked = check_program(program, ev["host"], ev["catalog"])
            rendered = render_checked(checked, ev["host"], ev["catalog"]) if checked.status == "accepted" else None
            reference = check_program(ev["program"], ev["host"], ev["catalog"])
            reference_text = render_checked(reference, ev["host"], ev["catalog"])
            exact = predicted == gold
            observations.append({"id": r["id"], "cluster": r["cluster"], "member": r["member"],
                "predicted": predicted, "gold": gold, "program": program,
                "kernel_status": checked.status, "kernel_reason": checked.reason, "rendered": rendered,
                "target_exact": exact, "kernel_accepted": checked.status == "accepted",
                "whole_answer_correct": checked.status == "accepted" and rendered == reference_text,
                "operation_correct": predicted["kind"] == gold["kind"],
                "structure_exact": {k: v for k, v in predicted.items() if k != "relation"} == {k: v for k, v in gold.items() if k != "relation"},
                "relation_correct": gold["kind"] == "compare" and predicted.get("relation") == gold.get("relation")})
            assert exact == observations[-1]["whole_answer_correct"]
    return summary(observations), observations


def adversarial(rows):
    comparison = next(r for r in rows if gold_target(r["evaluation_only"]["program"])["kind"] == "compare")
    ev = comparison["evaluation_only"]
    correct = copy.deepcopy(ev["program"])
    wrong_relation = copy.deepcopy(correct)
    wrong_relation["nodes"][2]["relation"] = next(rel for rel in RELATIONS if rel != correct["nodes"][2]["relation"])
    wrong = check_program(wrong_relation, ev["host"], ev["catalog"])
    assert (wrong.status, wrong.reason) == ("rejected", "unsupported_assertion")
    mismatch = copy.deepcopy(correct)
    mismatch["nodes"][1]["field"] = "price" if mismatch["nodes"][0]["field"] == "warranty_months" else "warranty_months"
    unit = check_program(mismatch, ev["host"], ev["catalog"])
    assert (unit.status, unit.reason) == ("rejected", "type_unit_mismatch")
    read = next(r for r in rows if gold_target(r["evaluation_only"]["program"])["kind"] == "read")
    rev = read["evaluation_only"]
    wrong_record = copy.deepcopy(rev["program"])
    old = wrong_record["nodes"][0]["record_ref"]
    wrong_record["nodes"][0]["record_ref"] = next(r["record_ref"] for r in read["state"]["records"] if r["record_ref"] != old)
    unrelated = check_program(wrong_record, rev["host"], rev["catalog"])
    gold = check_program(rev["program"], rev["host"], rev["catalog"])
    assert unrelated.status == "accepted"
    assert render_checked(unrelated, rev["host"], rev["catalog"]) != render_checked(gold, rev["host"], rev["catalog"])
    # Existing shipped fixture 09 independently exercises the same type/unit boundary.
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "language_growth"
    fixture_path = next((fixtures / "cases").glob("09_*.json"))
    fixture = json.loads(fixture_path.read_text())
    catalog = json.loads((fixtures / "catalog.json").read_text())
    fixture_result = check_program(fixture["proposal"], fixture["host"], catalog)
    assert fixture_result.reason == "type_unit_mismatch"
    return {"wrong_relation": wrong.to_dict(), "unit_mismatch": unit.to_dict(),
            "wrong_record": {"kernel_status": unrelated.status, "whole_answer_correct": False,
                             "interpretation": "Kernel checks evidence-relative validity, not question relevance; scorer rejects wrong requested record."},
            "existing_fixture_09": {"path": str(fixture_path), "sha256": sha(fixture_path), "status": fixture_result.status, "reason": fixture_result.reason}}


def paired(observations):
    clusters = {}
    for r in observations:
        clusters.setdefault(r["cluster"], []).append(r)
    assert all(len(rs) == 2 for rs in clusters.values())
    return {kind: {"clusters": len(rs), "both_target_exact": sum(all(x["target_exact"] for x in pair) for pair in rs),
                   "both_kernel_accepted": sum(all(x["kernel_accepted"] for x in pair) for pair in rs),
                   "both_whole_answer_correct": sum(all(x["whole_answer_correct"] for x in pair) for pair in rs)}
            for kind in ("all", *KINDS)
            for rs in [[pair for pair in clusters.values() if kind == "all" or pair[0]["gold"]["kind"] == kind]]}


def run(out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    registration = json.loads((out / "registration.json").read_text())
    assert registration["config"] == CONFIG and registration["source_hashes"][Path(__file__).name] == sha(__file__)
    for name, expected in registration["source_hashes"].items():
        assert sha(Path(__file__).with_name(name)) == expected
    assert registration["frozen_data_sha256"] == sha(out / "frozen_rows.json")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    data = json.loads((out / "frozen_rows.json").read_text())
    train = data["train"]
    results, raw = {}, {}
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed)
        model = TypedEmissionSSM().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        rng = random.Random(seed)
        history = []
        for step in range(CONFIG["steps"]):
            if time.monotonic() - started >= CONFIG["max_seconds"]:
                raise TimeoutError("Frozen total CPU diagnostic runtime exceeded")
            batch = rng.sample(train, CONFIG["batch_size"])
            states = [r["state"] for r in batch]
            q, components, indices, numeric, kinds, candidates = tensors(states)
            gold = [gold_target(r["evaluation_only"]["program"]) for r in batch]
            structures = [{k: v for k, v in g.items() if k != "relation"} for g in gold]
            labels = torch.tensor([next(i for i, c in enumerate(cs) if c["target"] == g) for cs, g in zip(candidates, structures)])
            op_labels = torch.tensor([KINDS.index(g["kind"]) for g in gold])
            compare = op_labels.eq(2)
            relation_labels = torch.tensor([RELATIONS.index(g.get("relation", RELATIONS[0])) for g in gold])
            model.train()
            optimizer.zero_grad()
            operation, scores, relations = model(q, components, indices, numeric)
            op_loss = nn.functional.cross_entropy(operation, op_labels)
            target_loss = nn.functional.cross_entropy(scores.masked_fill(kinds.ne(op_labels[:, None]), -torch.inf), labels)
            rel_loss = nn.functional.cross_entropy(relations[torch.arange(len(batch)), labels][compare], relation_labels[compare]) if compare.any() else scores.sum() * 0
            loss = op_loss + target_loss + rel_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_norm"])
            optimizer.step()
            if (step + 1) % 100 == 0:
                history.append({"step": step + 1, "operation_loss": op_loss.item(), "target_loss": target_loss.item(), "relation_loss": rel_loss.item()})
        checkpoint = out / f"seed-{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "seed": seed, "config": CONFIG}, checkpoint)
        restored = TypedEmissionSSM().cpu()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        metrics, observations = evaluate(restored, data["development"])
        train_metrics, _ = evaluate(restored, train)
        raw[str(seed)] = observations
        results[str(seed)] = {"metrics": metrics, "paired": paired(observations), "train": train_metrics,
                              "history": history, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint)}
        write(out / "predictions.json", raw)
        write(out / "partial_results.json", results)
        print(json.dumps({"seed": seed, "metrics": metrics, "elapsed_seconds": time.monotonic() - started}), flush=True)
    qa = adversarial(data["development"])
    elapsed = time.monotonic() - started
    assert elapsed <= CONFIG["max_seconds"]
    pooled = [r for rows in raw.values() for r in rows]
    receipt = {"registration": registration, "results": results, "pooled": summary(pooled), "adversarial": qa,
               "wall_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
               "runtime": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
               "raw_predictions_sha256": sha(out / "predictions.json"), "data_sha256": sha(out / "frozen_rows.json"),
               "scope": CONFIG["scope"], "cost_usd": 0}
    write(out / "receipt.json", receipt)
    print(json.dumps({"pooled": receipt["pooled"], "wall_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.out)
