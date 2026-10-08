"""Small deterministic checks for the synthetic C multirecord builder."""
from collections import Counter
import unittest

from data import RECORD_SCHEMA
from data.augment import validate
from data.c_multirecord_synth import (
    FIELDS, GENERATOR_SPEC, _parse_state, make_record, option_ids, verify_record,
)


class CMultirecordSynthTests(unittest.TestCase):
    def test_synthetic_generator_spec_and_multi_question_case_exact_records(self):
        from harness.heldout import check_generator, check_record
        check_generator(GENERATOR_SPEC)
        for index in range(30):
            row = make_record(20261008, index)
            check_record(row)
            validate(row)
            verify_record(row)
            self.assertEqual(row["schema"], RECORD_SCHEMA)
            self.assertEqual(row["source_split"], "synthetic")
            self.assertEqual(row["family"], "synthetic_multirecord_telemetry")
            self.assertEqual(len(row["questions"]), 7)
            self.assertEqual(set(row["questions"]), set(row["labels"]))
            kinds = Counter(q["type"] for q in row["questions"].values())
            self.assertEqual(kinds, {"choice": 1, "noul": 5, "score": 1})
            roles = row["aug"]["question_roles"]
            self.assertEqual(sum(m["kind"] == "multi_select_member" for m in roles.values()), 4)
            self.assertEqual(sum(m["kind"] == "case_exact" for m in roles.values()), 1)
            self.assertEqual(sum(m["kind"] == "top_profile" for m in roles.values()), 1)
            self.assertEqual(sum(m["kind"] == "joint_score" for m in roles.values()), 1)
            case_qid = next(q for q, m in roles.items() if m["kind"] == "case_exact")
            case_label = row["labels"][case_qid]
            self.assertEqual(case_label, "true" if index % 2 == 0 else "false")
            self.assertEqual(len(FIELDS), 4)
            parsed, order = _parse_state(row["state"])
            self.assertEqual(len(parsed), 4)
            self.assertEqual(set(order), set(FIELDS))
            selected = [q for q, meta in roles.items() if meta["kind"] == "multi_select_member"
                        and row["labels"][q] == "true"]
            self.assertEqual(len(selected), 2)

    def test_k_sweep_styles_and_option_ids(self):
        counts = Counter()
        for index in range(3 * 76):
            row = make_record(71, index)
            for meta in row["aug"]["question_roles"].values():
                if meta["kind"] == "top_profile":
                    counts[meta["k"]] += 1
        self.assertEqual(set(counts), set(range(2, 78)))
        for style in ("letters", "words", "opaque"):
            ids = option_ids(style, 77, seed=1, record_index=1, slot=1)
            self.assertEqual(len(ids), len(set(ids)))

    def test_deterministic_records_and_schema_variants(self):
        first = make_record(20261008, 91)
        again = make_record(20261008, 91)
        self.assertEqual(first, again)
        self.assertNotEqual(first, make_record(20261008, 92))
        shapes = {make_record(20261008, i)["aug"]["schema_shape"] for i in range(12)}
        self.assertEqual(shapes, {"column_table_v1", "register_map_v1", "sample_frame_v1"})


if __name__ == "__main__":
    unittest.main()
