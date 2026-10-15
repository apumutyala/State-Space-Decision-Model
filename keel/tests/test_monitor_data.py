"""Regression for the actual APPS line-list representation encountered in acquisition."""
import json

from keel.monitor.data import load_problems, stdio_text


def test_stdio_line_lists_and_nontext_values():
    assert stdio_text(["82", "28"]) == "82\n28\n"
    assert stdio_text("82\n28\n") == "82\n28\n"
    assert stdio_text([82, 28]) is None
    assert stdio_text([["82"]]) is None


def test_loader_retains_line_lists_without_admitting_function_calls(tmp_path, monkeypatch):
    source = tmp_path / "train.jsonl"
    base = {"difficulty": "interview", "solutions": json.dumps(["print(1)"]), "question": "Subtract two inputs."}
    records = [
        {**base, "id": 1, "input_output": json.dumps({"inputs": [["82", "28"]], "outputs": [["54"]]})},
        {**base, "id": 2, "input_output": json.dumps({"fn_name": "subtract", "inputs": [[82, 28]], "outputs": [54]})},
        {**base, "id": 3, "input_output": json.dumps({"inputs": ["82\n28\n"], "outputs": ["54\n"]})},
    ]
    source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(source))
    problems, report = load_problems(tmp_path, 10)
    assert report["eligible_problems"] == 2
    assert report["exclusions"] == {"not_stdio_with_reference": 1}
    assert {p.problem_id for p in problems} == {"apps-train-1", "apps-train-3"}
    assert all(p.inputs == ("82\n28\n",) and p.outputs == ("54\n",) for p in problems)
