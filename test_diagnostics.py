"""Tool correctness only; no model search and no paid API requests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from diagnostics import Diagnostics
from solution import discover_files, run_research


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "work"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        train = pd.DataFrame({"id": range(6), "period": [1, 1, 2, 2, 3, 3],
                              "category": ["Education_College", "Education_Other"] * 3,
                              "target": [1., 2., 3., 4., 5., 6.]})
        train.to_csv(self.folder / "train.csv", index=False)
        pd.DataFrame({"id": [9, 10], "period": [4, 4], "category": ["Unseen", "Education_Other"]}).to_csv(
            self.folder / "test.csv", index=False)
        (self.folder / "sample_submission.csv").write_text("id,target\n9,0\n10,0\n")
        (self.folder / "data_dictionary.csv").write_text("feature,role\ncategory,feature\n")
        (self.folder / "submission.csv").write_text("id,target\n9,1\n10,2\n")
        self.run = self.folder / "logs"
        root = self.run / "research"
        root.mkdir(parents=True)
        self.state = {"features": ["period", "category"], "task_type": "regression", "target": "target",
                      "metric": "rmse", "experiments": [{"id": 1, "model_path": "saved.joblib",
                      "feature_code": "def transform(X): return X", "score": 1.}], "best": None}
        (root / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.files = discover_files(self.folder)
        self.worker = Diagnostics(root, self.files, self.folder)

    def test_categories_and_worker_protocol(self):
        result = self.worker.inspect_categories(["category"])["categories"]["category"]
        self.assertEqual(result["unseen_test_rows"], 1)
        self.assertEqual(result["seven_byte_collision_groups"], 1)
        self.assertIn("feature_code", self.worker.read_experiment(1)["experiment"])
        result = run_research("inspect_categories", {"columns": ["category"]}, self.folder,
                              self.files, self.run, 1, 30, module_name="diagnostics.py")
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual(json.loads(result["stdout"])["categories"]["category"]["train_unique"], 2)

    def test_regression_errors_and_comparison_preserve_artifacts(self):
        root = self.run / "research"
        np.savez(root / "split.npz", train=[0, 1, 2, 3], validation=[4, 5])
        np.save(root / "validation_1.npy", [4., 8.])
        np.save(root / "validation_2.npy", [5., 7.])
        self.state["experiments"].append(dict(self.state["experiments"][0], id=2))
        (root / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        before = (root / "state.json").read_bytes(), (self.folder / "submission.csv").read_bytes()
        with patch("diagnostics.joblib.load", return_value={}):
            result = self.worker.analyze_errors(1, ["period", "category"], 1)
            comparison = self.worker.compare_errors([1, 2])
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["mean_prediction_minus_target"], .5)
        self.assertEqual(comparison["second_lower_absolute_error"], 2)
        self.assertTrue(result["groups"]["period"])
        self.assertEqual(before, ((root / "state.json").read_bytes(), (self.folder / "submission.csv").read_bytes()))
        from solution import experiment_memory, reviewer_context
        self.assertEqual(len(experiment_memory(self.run)["error_analysis"]), 2)
        # Supply normal experiment metadata required by reviewer ranking.
        for record in self.state["experiments"]:
            record.update(algorithm="catboost", params={})
        (root / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.assertEqual(len(json.loads(reviewer_context(self.run, "task"))["error_analysis"]), 2)

    def test_classification_confusions_and_ensemble_fallback(self):
        root = self.run / "research"
        self.state.update(task_type="classification", metric="f1_macro")
        self.state["experiments"].append(dict(self.state["experiments"][0], id=2))
        (root / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        np.savez(root / "split.npz", train=[0, 1, 2, 3], validation=[4, 5])
        np.save(root / "validation_1.npy", [[.9, .1], [.8, .2]])
        with patch("diagnostics.joblib.load", return_value={"classes": [5., 6.]}), \
             patch.object(self.worker.engine, "predict_raw", return_value=np.array([[.1, .9], [.1, .9]])) as predict:
            result = self.worker.analyze_errors(1)
            comparison = self.worker.compare_errors([1, 2])
        self.assertEqual(result["confusions"], [{"actual": "6.0", "predicted": "5.0", "count": 1}])
        self.assertEqual(sum(r["count"] for r in result["confidence_bins"]), 2)
        self.assertEqual(comparison["only_first_correct"], 1)
        self.assertEqual(comparison["only_second_correct"], 1)
        predict.assert_called_once()

    def test_temporal_check_preserves_winner_and_uses_only_past_rows(self):
        before = (self.folder / "submission.csv").read_bytes()
        state_before = (self.run / "research/state.json").read_bytes()
        with patch("diagnostics.joblib.load", return_value={"recipe": "saved"}), \
             patch.object(self.worker.engine, "refit", return_value={}) as refit, \
             patch.object(self.worker.engine, "predict_raw", return_value=np.array([5., 6.])):
            result = self.worker.validate_experiment(1, "period", 3)
        self.assertEqual(result["score"], 0.)
        self.assertTrue((refit.call_args.args[1].period < 3).all())
        self.assertEqual(result["train_rows"], 4)
        self.assertEqual(result["validation_rows"], 2)
        self.assertEqual((self.folder / "submission.csv").read_bytes(), before)
        self.assertEqual((self.run / "research/state.json").read_bytes(), state_before)
        self.assertEqual(len(json.loads((self.run / "diagnostics.json").read_text(encoding="utf-8"))), 1)
        with self.assertRaises(ValueError):
            self.worker.validate_experiment(1, "period", 1)


if __name__ == "__main__":
    unittest.main()
