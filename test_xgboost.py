"""Synthetic integration checks, no real-data research and no Qwen requests."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd

import test_research
from solution import compatible_manifest, run_research, validate_submission
from xgb_backend import ExtendedResearch


class XGBoostTests(unittest.TestCase):
    def setUp(self):
        test_research.ResearchTests.setUp(self)
        self.worker = ExtendedResearch(self.folder / "research", self.files, self.folder)

    def test_regression_metrics_serialization_and_full_refit(self):
        for metric in ("rmse", "mae"):
            worker = ExtendedResearch(self.folder / metric, self.files, self.folder)
            worker.configure("target", "regression", metric)
            result = worker.train("xgb", {"iterations": 40, "depth": 3}, "xgboost", early_stopping_rounds=5)
            e = result["experiment"]
            bundle = joblib.load(e["model_path"])
            trace = bundle["model"].evaluation_history["validation"][metric]
            self.assertEqual(e["selected_iterations"], int(np.argmin(trace)) + 1)
            self.assertEqual(bundle["model"].booster.num_boosted_rounds(), e["selected_iterations"])
            worker.finalize()
            self.assertTrue(validate_submission(self.folder, self.files)[0])
            final = joblib.load(worker.root / "final_model.joblib")
            self.assertEqual(final["model"].selected_iterations, e["selected_iterations"])

    def test_classification_probability_ensemble_and_unknown_category(self):
        path = self.folder / "train.csv"
        frame = pd.read_csv(path)
        frame["target"] = np.where(frame.x < -.5, "A", np.where(frame.x > .5, "B", "C"))
        frame.to_csv(path, index=False)
        test = pd.read_csv(self.folder / "test.csv")
        test.loc[0, "category"] = "UNSEEN"
        test.to_csv(self.folder / "test.csv", index=False)
        self.worker.configure("target", "classification", "f1_macro")
        result = self.worker.train("weighted xgb", {"iterations": 60, "depth": 3,
                                   "auto_class_weights": "SqrtBalanced"}, "xgboost", early_stopping_rounds=5)
        e = result["experiment"]
        bundle = joblib.load(e["model_path"])
        trace = bundle["model"].evaluation_history["validation"]["f1_macro"]
        self.assertEqual(e["selected_iterations"], int(np.argmax(trace)) + 1)
        p = self.worker.predict_raw(bundle, test, self.worker.state())
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(bundle["classes"], ["A", "B", "C"])
        self.worker.train("cat", {"iterations": 15, "depth": 3})
        blend = self.worker.blend([1, 2])
        b = joblib.load(blend["experiment"]["model_path"])
        self.worker.publish(self.worker.refit(b, frame, self.worker.state()))
        self.assertTrue(validate_submission(self.folder, self.files)[0])

    def test_subprocess_model_reload_and_finalize(self):
        self.worker.configure("target", "regression", "rmse")
        result = run_research("train", {"hypothesis": "protocol", "algorithm": "xgboost",
                              "params": {"iterations": 10}}, self.folder, self.files, self.folder, 1, 30)
        self.assertEqual(result["exit_code"], 0, result)
        result = run_research("finalize", {}, self.folder, self.files, self.folder, 2, 30)
        self.assertEqual(result["exit_code"], 0, result)

    def test_no_early_stopping_binary_accuracy_and_parameter_validation(self):
        frame = pd.read_csv(self.folder / "train.csv")
        frame["target"] = np.where(frame.x > 0, "yes", "no")
        frame.to_csv(self.folder / "train.csv", index=False)
        self.worker.configure("target", "classification", "accuracy")
        result = self.worker.train("binary", {"iterations": 10, "auto_class_weights": "Balanced"},
                                   "xgboost", early_stopping_rounds=0)
        self.assertEqual(result["experiment"]["selected_iterations"], 10)
        for params in ({"num_leaves": 31}, {"subsample": 0}, {"min_child_weight": -1}):
            with self.assertRaises(ValueError):
                self.worker.model(self.worker.state(), params, "xgboost")

    def test_manifest_extension_never_allows_changed_inputs_or_changed_backend(self):
        old_lgb = {"train.csv": "data", "xgb_code": "backend",
                   "training_code": "b0698d3a3bf584c678da363d89b29b7c77b907b6ae7a1b2539d5dc9f9a3f72c9"}
        new_lgb = dict(old_lgb, training_code="3475c51db744d59104e53e443462efdfa43931821d19e87a843c1214962230dc")
        self.assertTrue(compatible_manifest(old_lgb, new_lgb, {}))
        self.assertFalse(compatible_manifest(old_lgb, dict(new_lgb, **{"train.csv": "changed"}), {}))
        self.assertFalse(compatible_manifest(old_lgb, dict(new_lgb, training_code="future"), {}))
        state = {"experiments": [{"algorithm": "catboost"}]}
        old = {"train.csv": "a", "training_code": "b"}
        new = dict(old, xgb_code="c")
        self.assertTrue(compatible_manifest(old, new, state))
        self.assertFalse(compatible_manifest(old, dict(new, **{"train.csv": "changed"}), state))
        self.assertFalse(compatible_manifest(new, dict(new, xgb_code="changed"), state))
        self.assertFalse(compatible_manifest(old, new, {"experiments": [{"algorithm": "xgboost"}]}))


if __name__ == "__main__":
    unittest.main()
