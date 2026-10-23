import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from research import write_json
from solution import discover_files, experiment_memory, reviewer_context, run_research
from temporal_research import HistoricalFeatures, TemporalResearch, temporal_folds


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(year=[1, 2, 2, 3, 3], key=["a", "a", "a", "a", "b"],
                                       group=["g"] * 5, target=[10., 20., 40., 999., 888.]))

    def make(self, frame=None):
        return HistoricalFeatures("year", "target", ["key"], ["group"], "history_groups").fit(
            self.frame if frame is None else frame)

    def test_strict_past_and_same_period_exclusion(self):
        result = self.make().transform(self.frame)
        self.assertTrue(np.isnan(result.loc[0, "hist_baseline"]))
        self.assertEqual(result.loc[1, "hist_exact_mean"], 10.)
        self.assertEqual(result.loc[2, "hist_exact_mean"], 10.)
        self.assertAlmostEqual(result.loc[3, "hist_exact_mean"], 70 / 3)
        self.assertEqual(result.loc[3, "hist_exact_last"], 30.)
        self.assertEqual(result.loc[3, "hist_exact_count"], 3)
        self.assertEqual(result.loc[4, "hist_exact_seen"], 0)
        self.assertAlmostEqual(result.loc[4, "hist_baseline"], 70 / 3)

    def test_query_targets_and_future_targets_never_change_past(self):
        reference = self.make().transform(self.frame)
        changed = self.frame.copy()
        changed["target"] = -123456.
        pd.testing.assert_frame_equal(reference, self.make().transform(changed))
        changed = self.frame.copy()
        changed.loc[changed.year == 3, "target"] = -123456.
        pd.testing.assert_frame_equal(reference, self.make(changed).transform(self.frame))
        pd.testing.assert_frame_equal(reference, self.make().transform(self.frame.drop(columns="target")))

    def test_order_unknown_and_missing_keys(self):
        shuffled = self.frame.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(self.make(shuffled).transform(shuffled).sort_index(),
                                      self.make().transform(self.frame))
        query = pd.DataFrame(dict(year=[3, 3], key=[None, "new"], group=["g", "new"]))
        result = self.make().transform(query)
        self.assertEqual(result.hist_exact_seen.tolist(), [0., 0.])
        self.assertAlmostEqual(result.loc[1, "hist_baseline"], 70 / 3)

    def test_fold_contract(self):
        folds = temporal_folds(pd.DataFrame({"year": [1, 2, 3, 4]}), "year", [3, 4])
        for period, tr, va in folds:
            self.assertFalse(set(tr) & set(va))
            self.assertTrue(np.all(tr < va.min()))
        with self.assertRaises(ValueError):
            temporal_folds(self.frame, "year", [2, 2])


class TemporalIntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "work"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        frame = pd.DataFrame([dict(id=year * 100 + k, year=year, key=str(k), group=str(k % 2),
            x=float(k), target=float(k * 10 + year)) for year in range(1, 6) for k in range(12)])
        frame.to_csv(self.folder / "train.csv", index=False)
        test = frame.iloc[:12].drop(columns="target").copy()
        test["year"], test["id"] = 6, np.arange(1000, 1012)
        test.to_csv(self.folder / "test.csv", index=False)
        pd.DataFrame(dict(id=test.id, target=0.)).to_csv(self.folder / "sample_submission.csv", index=False)
        (self.folder / "data_dictionary.csv").write_text("feature,role\ntarget,target\n")
        (self.folder / "submission.csv").write_text("legacy output untouched")
        self.files = discover_files(self.folder)
        self.run = self.folder / "logs"
        self.root = self.run / "research"
        self.root.mkdir(parents=True)
        recipe = dict(id=1, algorithm="catboost", params=dict(iterations=12, depth=2), feature_code="", score=10.)
        self.state = dict(task_type="regression", metric="rmse", target="target", ids=["id"],
                          features=["year", "key", "group", "x"], best=recipe, experiments=[recipe])
        write_json(self.root / "state.json", self.state)
        self.worker = TemporalResearch(self.root, self.files, self.folder)
        self.args = dict(base_experiment_id=1, time_column="year", key_columns=["key"],
                         group_columns=["group"], validation_periods=[3, 4, 5], history_gate=True)

    def test_suite_exports_oof_and_reload_without_changing_legacy(self):
        before = (self.root / "state.json").read_bytes()
        with patch.object(self.worker, "_iterations", wraps=self.worker._iterations) as inner:
            result = self.worker.run_temporal_suite(**self.args)
        self.assertEqual(inner.call_count, 9)
        self.assertEqual([call.args[0].year.max() for call in inner.call_args_list], [2, 3, 4] * 3)
        directory = Path(result["report_path"]).parent
        self.assertEqual(len(result["results"]), 4)
        for row in result["results"]:
            oof = np.load(directory / (row["mode"] + "_oof.npy"))
            self.assertTrue(np.isnan(oof[:24]).all())
            self.assertTrue(np.isfinite(oof[24:]).all())
            self.assertEqual(len(row["folds"]), 3)
        gate = joblib.load(directory / "history_gate_fold_2.joblib")
        query = pd.read_csv(self.files["test.csv"])
        query["key"] = query["key"].astype(float)
        query.loc[0, "key"] = 999
        query.loc[1, "key"] = np.nan
        query = query.sample(frac=1, random_state=1)
        base, hist = gate["gate_models"]
        seen = hist["history"].transform(query).hist_exact_seen.to_numpy().astype(bool)
        self.assertTrue(seen.any())
        self.assertTrue((~seen).any())
        np.testing.assert_allclose(self.worker._predict(gate, query),
            np.where(seen, self.worker._predict(hist, query), self.worker._predict(base, query)))
        with patch.object(self.worker, "_fit", side_effect=AssertionError("cache must avoid retraining")):
            self.assertEqual(self.worker.run_temporal_suite(**self.args), result)
        # Exercise two-model final export even when tiny synthetic data selects baseline.
        result["selected_mode"] = "history_gate"
        write_json(directory / "report.json", result)
        exported = self.worker.finalize_temporal(result["suite_id"])
        bundle = joblib.load(directory / "final_model.joblib")
        test = pd.read_csv(self.files["test.csv"])
        np.testing.assert_allclose(self.worker._predict(bundle, test), pd.read_csv(exported["submission_path"]).target)
        self.assertEqual((self.root / "state.json").read_bytes(), before)
        self.assertEqual((self.folder / "submission.csv").read_text(), "legacy output untouched")
        self.assertIn("temporal_comparison", experiment_memory(self.run))
        self.assertIn("temporal_comparison", json.loads(reviewer_context(self.run, "task")))
        response = run_research("finalize_temporal", {"suite_id": result["suite_id"]}, self.folder,
            self.files, self.run, 99, 30, module_name="temporal_research.py")
        self.assertEqual(response["exit_code"], 0, response)
        from final_output import FinalOutput
        finalizer = FinalOutput(root=self.root, files=self.files, folder=self.folder)
        finalizer.select_final_candidate("temporal", result["suite_id"])
        published = finalizer.finalize_selected()
        self.assertTrue(published["published"])
        np.testing.assert_allclose(pd.read_csv(self.folder / "submission.csv").target,
                                   pd.read_csv(exported["submission_path"]).target)
        # Changed data must invalidate final export of the saved suite.
        with Path(self.files["train.csv"]).open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.worker.finalize_temporal(result["suite_id"])

    def test_reject_target_id_keys_and_noncatboost(self):
        for key in ("target", "id", "year"):
            with self.assertRaises(ValueError):
                self.worker.run_temporal_suite(**dict(self.args, key_columns=[key]))
        with self.assertRaises(ValueError):
            self.worker.run_temporal_suite(**dict(self.args, base_experiment_id=999))


if __name__ == "__main__":
    unittest.main()
