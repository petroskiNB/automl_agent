import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from classification_research import ClassificationResearch, fit_decision_rule
from final_output import FinalOutput
from research import write_json
from solution import discover_files, run_research, experiment_memory


class ClassificationOOFTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "work"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rng = np.random.default_rng(42)
        x = rng.normal(size=240)
        train = pd.DataFrame(dict(id=np.arange(240), x=x, category=np.where(x > 0, "a", "b"),
                                 target=np.where(x < -.4, "A", np.where(x > .4, "C", "B"))))
        train.to_csv(self.folder / "train.csv", index=False)
        test = train.drop(columns="target").iloc[:20].copy()
        test["id"] += 1000
        test.loc[0, "category"] = "unknown"
        test.to_csv(self.folder / "test.csv", index=False)
        pd.DataFrame(dict(id=test.id, target="A")).to_csv(self.folder / "sample_submission.csv", index=False)
        (self.folder / "data_dictionary.csv").write_text("feature,role\ntarget,target\n")
        (self.folder / "submission.csv").write_text("old untouched")
        self.files = discover_files(self.folder)
        self.run = self.folder / "logs"
        self.root = self.run / "research"
        self.root.mkdir(parents=True)
        recipes = [dict(id=1, algorithm="catboost", params=dict(iterations=10, depth=2), feature_code="", score=.5),
                   dict(id=2, algorithm="xgboost", params=dict(iterations=10, depth=2), feature_code="", score=.5)]
        self.state = dict(task_type="classification", metric="f1_macro", target="target", features=["x", "category"],
                          ids=["id"], best=recipes[0], experiments=recipes)
        write_json(self.root / "state.json", self.state)
        self.worker = ClassificationResearch(self.root, self.files, self.folder)

    def test_nested_suite_adversarial_and_final_publication(self):
        old_state = (self.root / "state.json").read_bytes()
        shift = self.worker.adversarial_validation()
        self.assertGreaterEqual(shift["auc"], 0)
        self.assertEqual(shift["excluded_ids"], ["id"])
        self.assertNotIn("target", [r["feature"] for r in shift["important_features"]])
        with patch.object(self.worker, "_fit", wraps=self.worker._fit) as fit:
            result = self.worker.run_classification_suite([1, 2], folds=3, shift_id=shift["diagnostic_id"])
        directory = Path(result["report_path"]).parent
        self.assertEqual(fit.call_count, 16)  # 2 recipes x 2 fit roles x (3 CV + 1 stress).
        for i in range(4):
            split = np.load(directory / f"fold_{i}/indices.npz")
            inner = np.load(directory / f"fold_{i}/calibration_indices.npz")
            for call in fit.call_args_list[i * 4:(i + 1) * 4]:
                self.assertFalse(set(call.args[0].index) & set(split["validation"]))
            self.assertFalse(set(inner["train"]) & set(inner["validation"]))
        predictions = np.load(directory / "oof.npz")
        self.assertEqual(predictions["macro_decision"].shape, (240, 3))
        self.assertTrue(np.isfinite(predictions["macro_decision"]).all())
        self.assertEqual(len(result["results"]), 5)
        self.assertTrue(all(r["stress_f1"] is not None for r in result["results"]))
        self.assertEqual((self.folder / "submission.csv").read_text(), "old untouched")
        with patch.object(self.worker, "_fit", side_effect=AssertionError("cache")):
            self.assertEqual(self.worker.run_classification_suite([1, 2], 3, shift["diagnostic_id"]), result)
        # Explicitly exercise multiplier + weighted fold ensemble export.
        result["selected_candidate"] = "macro_decision"
        write_json(directory / "report.json", result)
        finalizer = FinalOutput(self.root, self.files, self.folder)
        finalizer.select_final_candidate("classification_oof", result["suite_id"])
        response = run_research("finalize_selected", {}, self.folder, self.files, self.run,
                                700, 30, module_name="final_output.py")
        self.assertEqual(response["exit_code"], 0, response)
        output = pd.read_csv(self.folder / "submission.csv")
        package = joblib.load(directory / "final_model.joblib")
        test = pd.read_csv(self.files["test.csv"])
        raw = self.worker.predict_package(package, test)
        self.assertEqual(output.target.tolist(), np.array(package["folds"][0]["classes"])[raw.argmax(axis=1)].tolist())
        self.assertEqual(output.id.tolist(), test.id.tolist())
        self.assertEqual((self.root / "state.json").read_bytes(), old_state)
        self.assertTrue(list(self.run.glob("submission_before_selected_*.csv")))
        self.assertIn("classification_summary", experiment_memory(self.run))
        Path(self.files["train.csv"]).write_text(Path(self.files["train.csv"]).read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            finalizer.finalize_selected()

    def test_rule_is_bounded_and_does_not_mutate_probabilities(self):
        p = np.array([[.6, .4], [.55, .45], [.2, .8], [.8, .2]])
        before = p.copy()
        w, m = fit_decision_rule([p, p[:, ::-1]], np.array(["A", "B", "B", "A"]), ["A", "B"])
        self.assertAlmostEqual(sum(w), 1)
        self.assertTrue(all(v >= 0 for v in w))
        self.assertTrue(all(.75 <= v <= 1.5 for v in m))
        np.testing.assert_array_equal(p, before)

    def test_invalid_recipe_and_id(self):
        with self.assertRaises(ValueError):
            self.worker.run_classification_suite([1, 1])
        with self.assertRaises(ValueError):
            self.worker.run_classification_suite([1, 2], 2)
        with self.assertRaises(ValueError):
            self.worker.export_classification("../outside")


if __name__ == "__main__":
    unittest.main()
