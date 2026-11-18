"""Checkpoint and evidence contracts independent of the core gradient suite."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tiny_lm as lm
import workbench as wb


def resign(document):
    document["sha256"] = hashlib.sha256(wb.canonical(document["payload"]).encode()).hexdigest()
    return document


class CheckpointTests(unittest.TestCase):
    def test_trained_round_trip_and_inference(self):
        for attention in (True, False):
            model, _ = lm.fit(seed=7, attention=attention)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.json"
                wb.save(path, model)
                restored = wb.load(path)
                for name in model.p:
                    np.testing.assert_array_equal(model.p[name], restored.p[name])
                train, held = lm.corpus()
                self.assertEqual(lm.evaluate(model, held), lm.evaluate(restored, held))
                self.assertEqual(lm.generate(model), lm.generate(restored))
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    wb.save(path, restored)
                self.assertEqual(path.read_bytes(), before)
                inspection = wb.inspect(restored, "the red robot near the blue lamp stays")
                self.assertAlmostEqual(sum(inspection["next_token_probability"].values()), 1)
                if attention:
                    self.assertAlmostEqual(sum(inspection["attention_weights"]), 1)
                    self.assertGreater(inspection["next_token_probability"]["red"], .9)
                else:
                    self.assertIsNone(inspection["attention_weights"])

    def test_corruption_and_schema_rejection(self):
        valid = wb.pack(lm.Model())
        cases = []
        for key, value in (("format", True), ("format", 2), ("source_sha256", "bad"),
                           ("vocabulary", []), ("attention", 1)):
            d = copy.deepcopy(valid)
            d["payload"][key] = value
            cases.append(resign(d))
        for value in ([0], [["1"]]*18, [[True]*16]*18, [[1e200]*16]*18):
            d = copy.deepcopy(valid)
            d["payload"]["parameters"]["embed"] = value
            cases.append(resign(d))
        d = copy.deepcopy(valid)
        del d["payload"]["parameters"]["bias"]
        cases.append(resign(d))
        d = copy.deepcopy(valid)
        d["payload"]["parameters"]["bias"][0] += 1
        cases.append(d)  # no resign, catches accidental corruption
        cases.extend([{}, [], {**valid, "extra": 1}])
        d = copy.deepcopy(valid)
        d["payload"]["parameters"]["embed"][0][0] = True
        cases.append(resign(d))
        for document in cases:
            with self.assertRaises(ValueError):
                wb.unpack(document)

    def test_bounded_json_and_failed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{', ' '*(wb.LIMIT+1)):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    wb.load(path)
            model = lm.Model()
            model.p["bias"][0] = float("nan")
            missing = Path(directory) / "missing.json"
            with self.assertRaises(ValueError):
                wb.save(missing, model)
            self.assertFalse(missing.exists())

    def test_evidence_numeric_and_structure_contract(self):
        wb.compare({"x": [1., True], "python": "old"}, {"x": [1.+1e-12, True], "python": "new"})
        for changed in ({"x": [2., True]}, {"x": [1., False]}, {"x": []}, {"y": [1., True]}):
            with self.assertRaises(ValueError):
                wb.compare({"x": [1., True]}, changed)
        with self.assertRaises(ValueError):
            wb.inspect(lm.Model(), "notinvocabulary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
