"""Small deterministic checks for the synthetic C multirecord builder."""
from collections import Counter
import unittest

from data import RECORD_SCHEMA
from data.augment import validate
from data.c_multirecord_synth import (
    FIELDS, GENERATOR_SPEC, BoundedEncodingCache, _parse_state, make_record, option_ids, verify_record,
)


class CMultirecordSynthTests(unittest.TestCase):
    def test_encoding_cache_preserves_trust_mutation_and_memory_limits(self):
        calls = []
        def encode(text, trusted):
            calls.append((text, trusted))
            if text == "control" and not trusted:
                raise ValueError("untrusted control")
            return [len(text), int(trusted)]
        cached = BoundedEncodingCache(encode, max_entries=2, max_bytes=1024)
        first = cached("sample", True)
        first[0] = -1
        self.assertEqual(cached("sample", True), [6, 1])
        self.assertEqual(cached("sample", False), [6, 0])
        self.assertEqual(len(calls), 2)
        cached("control", True)
        with self.assertRaises(ValueError):
            cached("control", False)
        for i in range(20):
            self.assertEqual(cached(str(i), True), [len(str(i)), 1])
            self.assertLessEqual(len(cached.entries), 2)
            self.assertLessEqual(cached.bytes, 1024)
        cached("x" * 2048)
        self.assertLessEqual(cached.bytes, 1024)

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

    def test_v2_case_labels_are_not_revealed_by_wording_or_option_parity(self):
        from harness.heldout import check_record
        labels_by_wording, labels_by_parity = {}, {0: set(), 1: set()}
        score_targets = set()
        for index in range(304):
            row = make_record(20261008, index, revision=2)
            verify_record(row)
            check_record(row)
            self.assertIn("very low < -500", row["state"]["rule"])
            roles = row["aug"]["question_roles"]
            case = next(q for q, m in roles.items() if m["kind"] == "case_exact")
            choice = next(q for q, m in roles.items() if m["kind"] == "top_profile")
            score = next(q for q, m in roles.items() if m["kind"] == "joint_score")
            wording = row["questions"][case]["instructions"].split(": ", 1)[0]
            labels_by_wording.setdefault(wording, set()).add(row["labels"][case])
            labels_by_parity[len(row["questions"][choice]["criteria"]) % 2].add(row["labels"][case])
            score_targets.add(roles[score]["node_id"] == roles[choice]["node_id"])
        self.assertTrue(all(v == {"true", "false"} for v in labels_by_wording.values()))
        self.assertTrue(all(v == {"true", "false"} for v in labels_by_parity.values()))
        self.assertEqual(score_targets, {True, False})


if __name__ == "__main__":
    unittest.main()
