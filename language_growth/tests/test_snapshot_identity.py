"""Byte-preserving snapshot relocation metadata checks; no Torch or weights."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from language_growth.snapshot_identity import bind_qualified_snapshot_location


class SnapshotIdentityTests(unittest.TestCase):
    def setUp(self):
        self.revision = "a" * 40
        self.actual_path = f"/hfcache/hub/models--RWKV--G1k/snapshots/{self.revision}"
        self.saved_path = f"/root/.cache/huggingface/hub/models--RWKV--G1k/snapshots/{self.revision}"
        self.actual = {"spec": "rwkv7-13.3b", "class": "Rwkv7Model", "config_sha256": "b" * 64,
                       "name_or_path": self.actual_path, "commit_hash": self.revision,
                       "capture": "hidden_states", "layer": None,
                       "weights_digest": "c" * 64, "weights_digest_mode": "full"}
        self.expected = {**self.actual, "name_or_path": self.saved_path}
        self.config = SimpleNamespace(_name_or_path=self.actual_path)

    def live(self):
        return {key: (self.config._name_or_path if key == "name_or_path" else value)
                for key, value in self.actual.items() if key not in ("weights_digest", "weights_digest_mode")}

    def bind(self, actual=None, expected=None, live=None):
        return bind_qualified_snapshot_location(self.config, self.actual if actual is None else actual,
            self.expected if expected is None else expected, live_config_identity=self.live if live is None else live)

    def test_identical_hf_snapshot_in_another_cache_preserves_saved_identity(self):
        pins = deepcopy(self.expected)
        self.assertTrue(self.bind())
        self.assertEqual(self.config._name_or_path, self.saved_path)
        self.assertEqual(self.expected, pins)
        self.assertEqual(self.live(), {k: v for k, v in self.expected.items()
                                      if k not in ("weights_digest", "weights_digest_mode")})

    def test_windows_cache_root_and_exact_ordinary_path(self):
        self.config._name_or_path = f"C:\\hf\\models--RWKV--G1k\\snapshots\\{self.revision}"
        self.actual["name_or_path"] = self.config._name_or_path
        self.assertTrue(self.bind())
        same = {**self.actual, "name_or_path": "/local/ordinary-directory"}
        self.config._name_or_path = same["name_or_path"]
        self.assertFalse(self.bind(same, deepcopy(same)))

    def test_every_non_location_change_refuses_before_mutation(self):
        for key in self.actual:
            if key == "name_or_path":
                continue
            modified = {**self.expected, key: "different"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bind(expected=modified)
            self.assertEqual(self.config._name_or_path, self.actual_path)
        for modified in ({k: v for k, v in self.expected.items() if k != "layer"},
                         {**self.expected, "extra": None}):
            with self.assertRaises(ValueError):
                self.bind(expected=modified)
            self.assertEqual(self.config._name_or_path, self.actual_path)

    def test_malformed_or_different_locations_refuse_before_mutation(self):
        invalid = [self.saved_path.replace("RWKV--G1k", "Other--G1k"),
                   self.saved_path.replace(self.revision, "d" * 40),
                   self.saved_path.replace(self.revision, "A" * 40),
                   self.saved_path[:-1], self.saved_path + "/weights", self.saved_path + "\n",
                   "/ordinary/local-copy", self.saved_path.replace("/root/", "/root/../"),
                   "https://example.com" + self.saved_path,
                   self.saved_path.lstrip("/")]
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.bind(expected={**self.expected, "name_or_path": path})
            self.assertEqual(self.config._name_or_path, self.actual_path)

    def test_failed_live_recheck_rolls_back_location(self):
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.bind(live=lambda: {**self.live(), "config_sha256": "d" * 64})
        self.assertEqual(self.config._name_or_path, self.actual_path)


if __name__ == "__main__":
    unittest.main()
