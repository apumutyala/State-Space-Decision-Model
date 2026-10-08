"""Fixed comparator answer protocol; templates contain no task labels."""


def answer_format() -> dict:
    return {"type": "JSON object only", "keys": ["claims", "explanations"],
                 "claim_types": {
                     "fact": {"kind": "fact", "entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT", "scope": "SCOPE"},
                     "comparison": {"kind": "comparison", "left": {"entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT"},
                                    "right": {"entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT"}, "relation": "less_than|equal|greater_than", "scope": "SCOPE"},
                     "exact_difference": {"kind": "exact_difference", "left": "typed operand object", "right": "typed operand object", "value": 0, "unit": "UNIT", "scope": "SCOPE"},
                     "hypothesis": {"kind": "hypothesis", "assumption_ref": "HOST_ASSUMPTION_ID", "premise": {"entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT"}, "alternate_value": 0, "right": {"entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT"}, "relation": "less_than|equal|greater_than", "scope": "SCOPE"},
                     "plan": {"kind": "plan", "workflow_ref": "WORKFLOW_ID", "steps": [{"step_ref": "STEP_ID", "action_ref": "ACTION_ID", "after": ["PRIOR_STEP_ID"], "preconditions": [{"kind": "fact", "entity": "ENTITY", "field": "FIELD", "value": 0, "unit": "UNIT", "scope": "SCOPE"}]}]}},
                 "rules": "Return {claims:[claim objects],explanations:integer}. Include all and only requested claims. Claims may appear in any order; plan step order matters. Inverse comparisons and equivalent exact numeric strings are accepted. explanations counts requested checked derivation explanations, otherwise 0. A hypothesis keeps its alternate value conditional, never factual; a plan proposes actions without claiming execution."}
