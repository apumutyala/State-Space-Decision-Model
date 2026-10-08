"""Tiny synthetic Index rows only; never load a dataset, model or suite."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from language_growth.gate_decision_retention import SCHEMA, _canonical, _digest, decision_index_retention


def publish(path, value, *, lines=False):
    path.write_bytes((b"".join(_canonical(row) + b"\n" for row in value) if lines else _canonical(value)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetentionTests(unittest.TestCase):
    def fixture(self, root):
        parent_id = {"artifact_sha256": "b" * 64, "revision_id": "qualified-parent"}
        candidate_id = {"artifact_sha256": "c" * 64, "revision_id": "actual-child"}
        rows, items, parent, child = [], [], [], []
        for i in range(20):
            rid, request = f"synthetic-{i}", hashlib.sha256(str(i).encode()).hexdigest()
            row = {"_evaluation": {"run_id": rid, "catalog_id": 3, "group_id": f"group-{i}", "payload_sha256": request},
                   "questions": {"q": {"type": "choice", "criteria": {"yes": "yes", "no": "no"}}},
                   "expected": {"q": "yes"}, "metadata": {}}
            rows.append(row)
            items.append({"run_id": rid, "request_sha256": request, "scoring_row_sha256": _digest(row), "source_case_id": f"case-{i}"})
            for target, identity, choice in ((parent, parent_id, "no"), (child, candidate_id, "yes")):
                target.append({"run_id": rid, "request_sha256": request, "checkpoint_sha256": identity["artifact_sha256"],
                               "revision_id": identity["revision_id"], "status": "ok", "response": {"answers": {
                                   "q": {"type": "choice", "choice": choice, "probabilities": {"yes": float(choice == "yes"), "no": float(choice == "no")}}}}})
        paths = {key: root / key for key in ("authorization", "rows", "parent", "child")}
        auth = {"schema": SCHEMA, "split": "DEV", "purpose": "language_parent_decision_regression", "edition": "0.2.1",
                "bootstrap": {"draws": 100, "seed": 20261008, "confidence": .95},
                "parent_identity": parent_id, "candidate_identity": candidate_id,
                "scoring_rows_sha256": publish(paths["rows"], rows, lines=True),
                "parent_predictions_sha256": publish(paths["parent"], parent, lines=True),
                "candidate_predictions_sha256": publish(paths["child"], child, lines=True),
                "cohort": {"scope": "dev_subset", "expected_rows": 20, "complete_groups": True,
                           "benchmark_rows": {"3": 20}, "unavailable_benchmarks": [],
                           "complete_source_cases": True, "minimum_cases_per_benchmark": 20}, "items": items}
        return paths, auth, rows, parent, child

    def run_fixture(self, paths, auth):
        sha = publish(paths["authorization"], auth)
        return decision_index_retention(paths["authorization"], sha, paths["rows"], paths["parent"], paths["child"],
                                        qualified_parent_identity=auth["parent_identity"], candidate_identity=auth["candidate_identity"], draws=100)

    def test_actual_index_weighted_pairing_and_incomplete_panel_cannot_pass(self):
        from harness.score import weights
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, *_ = self.fixture(Path(directory))
            report = self.run_fixture(paths, auth)
            expected = 100 * weights(renormalize=False)[3]
            self.assertAlmostEqual(report["paired"]["diff"]["index"]["point"], expected)
            self.assertNotEqual(expected, 100)  # accuracy difference would be 100
            self.assertTrue(report["criterion_observed"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["n_items"], 20)
            auth["cohort"]["scope"] = "full_dev"
            self.assertFalse(self.run_fixture(paths, auth)["passed"])

    def test_error_malformed_and_missing_predictions_keep_or_refuse_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, _rows, _parent, child = self.fixture(Path(directory))
            perfect = self.run_fixture(paths, auth)["paired"]["candidate"]["index"]["point"]
            child[0]["status"] = "exhausted"
            child[1]["response"]["answers"]["q"]["probabilities"] = {"yes": .7, "no": .7}
            auth["candidate_predictions_sha256"] = publish(paths["child"], child, lines=True)
            report = self.run_fixture(paths, auth)
            self.assertEqual(report["error_rows"]["candidate"], 2)
            self.assertEqual(report["n_items"], 20)
            self.assertLess(report["paired"]["candidate"]["index"]["point"], perfect)
            auth["candidate_predictions_sha256"] = publish(paths["child"], child[:-1], lines=True)
            with self.assertRaisesRegex(ValueError, "identical complete"):
                self.run_fixture(paths, auth)

    def test_request_revision_checkpoint_and_row_hash_mismatches_refuse(self):
        for key in ("request_sha256", "revision_id", "checkpoint_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                paths, auth, _rows, _parent, child = self.fixture(Path(directory))
                child[0][key] = "wrong"
                auth["candidate_predictions_sha256"] = publish(paths["child"], child, lines=True)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.run_fixture(paths, auth)
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, rows, *_ = self.fixture(Path(directory))
            rows[0]["expected"]["q"] = "no"
            auth["scoring_rows_sha256"] = publish(paths["rows"], rows, lines=True)
            with self.assertRaisesRegex(ValueError, "row identity"):
                self.run_fixture(paths, auth)

    def test_original_input_nondev_old_schema_and_incomplete_cases_refuse(self):
        for kind in ("state", "TEST", "old", "cases"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                paths, auth, rows, *_ = self.fixture(Path(directory))
                if kind == "state":
                    rows[0]["state"] = "original task text is not permitted"
                    auth["scoring_rows_sha256"] = publish(paths["rows"], rows, lines=True)
                elif kind == "TEST": auth["split"] = "TEST"
                elif kind == "old": auth["schema"] = "ssdm.language_authorized_dev_predictions.v1"
                else: auth["cohort"]["complete_source_cases"] = False
                with self.assertRaises(ValueError): self.run_fixture(paths, auth)

    def test_legacy_probabilities_and_prediction_map_use_same_index(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, _rows, parent, child = self.fixture(Path(directory))
            expected = self.run_fixture(paths, auth)["paired"]["diff"]["index"]["point"]
            for row in child:
                row["probabilities"] = row.pop("response")["answers"]["q"]["probabilities"]
            auth["candidate_predictions_sha256"] = publish(paths["child"], {r["run_id"]: r for r in child})
            self.assertEqual(self.run_fixture(paths, auth)["paired"]["diff"]["index"]["point"], expected)

    def test_actual_identity_cannot_be_replaced_by_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, *_ = self.fixture(Path(directory))
            sha = publish(paths["authorization"], auth)
            actual = deepcopy(auth["parent_identity"])
            actual["artifact_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "actual qualified"):
                decision_index_retention(paths["authorization"], sha, paths["rows"], paths["parent"], paths["child"],
                                         qualified_parent_identity=actual, candidate_identity=auth["candidate_identity"], draws=100)


if __name__ == "__main__":
    unittest.main()
