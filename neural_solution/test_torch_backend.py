"""Meaningful leakage, persistence, deadline, parallel and nested-validation checks."""
# Source checkout delegates to the complete, versioned runtime.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    if not _Path(__file__).with_name("solution.py").exists():
        from runtime import main as _launch
        _sys.argv.insert(1, 'test')
        _launch()

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from threading import Barrier
from unittest.mock import patch

from torch_resources import configure
configure(2)

import joblib
import numpy as np
import pandas as pd

from benchmark_torch import smoke_data, make_splits
from solution import discover_files, validate_submission
from torch_backend import TorchResearch
from torch_model import TabularPreprocessor, TorchModel


class TorchTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent/"work"
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.task = smoke_data(self.folder/"data", False)
        self.files = discover_files(self.task.parent)
        self.log = self.folder/"logs"
        self.log.mkdir()
        self.worker = TorchResearch(self.log/"research", self.files, self.task.parent)

    def test_preprocessing_is_train_fitted_and_row_independent(self):
        train = pd.DataFrame({"value": [1., 2., np.nan, 3.], "cat": ["known", "__MISSING__", None, "known"]})
        prep = TabularPreprocessor().fit(train)
        query = pd.DataFrame({"value": [10000., np.nan], "cat": ["novel", None]})
        medians, vocab = prep.medians.copy(), prep.vocabularies.copy()
        num, cat = prep.transform(query)
        self.assertEqual(prep.medians["value"], 2)
        np.testing.assert_array_equal(cat[:, 0], [0, 1])
        self.assertEqual(num[1, 0], 0)
        self.assertEqual(num[1, 1], 1)
        pd.testing.assert_series_equal(prep.medians, medians)
        self.assertEqual(prep.vocabularies, vocab)
        for i in range(2):
            one = prep.transform(query.iloc[[i]])
            np.testing.assert_array_equal(one[0], num[i:i+1])
            np.testing.assert_array_equal(one[1], cat[i:i+1])

    def test_regression_roundtrip_best_epoch_and_full_refit(self):
        self.worker.configure("target", "regression", "rmse", "time", "period")
        record = self.worker.train("network", {"iterations": 12, "width": 32, "layers": 2},
                                   "torch_resnet", early_stopping_rounds=3)["experiment"]
        bundle = joblib.load(record["model_path"])
        self.assertEqual(record["selected_iterations"], np.argmin(bundle["model"].evaluation_history)+1)
        train, test = self.worker.load()
        a = self.worker.predict_raw(bundle, test, self.worker.state())
        joblib.dump(bundle, self.folder/"roundtrip.joblib")
        b = self.worker.predict_raw(joblib.load(self.folder/"roundtrip.joblib"), test, self.worker.state())
        np.testing.assert_array_equal(a, b)
        self.worker.finalize()
        final = joblib.load(self.worker.root/"final_model.joblib")
        self.assertEqual(final["model"].selected_iterations, record["selected_iterations"])
        self.assertEqual(self.worker.state()["finalized_experiment"], record["id"])
        self.assertEqual(final["model"].preprocessor.medians["period"], train.period.median())
        self.assertTrue(validate_submission(self.task.parent, self.files)[0])

    def test_classification_weights_order_unknown_categories_and_blend(self):
        folder = self.folder/"classification"
        smoke_data(folder, True)
        files = discover_files(folder)
        worker = TorchResearch(self.log/"classification", files, folder)
        worker.configure("target", "classification", "f1_macro")
        for weight in (None, "SqrtBalanced"):
            worker.train("classification", {"iterations": 6, "width": 32, "layers": 1,
                "auto_class_weights": weight}, "torch_mlp", early_stopping_rounds=2)
        bundle = joblib.load(worker.state()["experiments"][0]["model_path"])
        self.assertEqual(bundle["classes"], ["A", "M", "Z"])
        p = worker.predict_raw(bundle, pd.read_csv(files["test.csv"]), worker.state())
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        worker.blend([1, 2])
        worker.finalize()
        self.assertTrue(validate_submission(folder, files)[0])

    def test_deadline_and_invalid_parameters_preserve_fallback(self):
        self.worker.configure("target", "regression", "rmse")
        self.worker.train("tree fallback", {"iterations": 5, "depth": 2}, "catboost")
        before = (self.task.parent/"submission.csv").read_bytes()
        state_before = (self.worker.root/"state.json").read_bytes()
        with patch("torch_backend.deadline", return_value=time.monotonic()-1):
            with self.assertRaises(TimeoutError):
                self.worker.train("out of time", {"iterations": 4, "width": 16}, "torch_mlp")
        self.assertEqual((self.task.parent/"submission.csv").read_bytes(), before)
        self.assertEqual((self.worker.root/"state.json").read_bytes(), state_before)
        for params in ({"iterations": 1.5}, {"depth": 4}, {"learning_rate": float("nan")}, {"width": 0}):
            with self.assertRaises(ValueError):
                TorchModel("regression", "rmse", params)
        train, _ = self.worker.load()
        with self.assertRaises(TimeoutError):
            TorchModel("regression", "rmse", {"iterations": 1, "width": 16}).fit(
                train[["x"]], train.target, deadline=time.monotonic()-1)

    def test_parallel_workers_respect_total_budget_and_do_not_publish(self):
        from parallel_research import run_batch
        self.worker.configure("target", "regression", "rmse")
        submission = self.task.parent/"submission.csv"
        submission.write_text("original")
        calls = []
        for role, algorithm, params in (("models", "torch_mlp", {"iterations": 3, "width": 16}),
                ("rare", "catboost", {"iterations": 5, "depth": 2})):
            calls.append(NS(id="team_1_"+role, function=NS(name="train_model", arguments=json.dumps(
                dict(hypothesis=role, algorithm=algorithm, params=params, feature_code="")))))
        with patch("parallel_research.available_memory", return_value=16*1024**3):
            results = run_batch(calls, self.task.parent, self.files, self.log, 90, jobs=2,
                                backend="torch_backend", total_threads=4)
        for result in results.values():
            self.assertEqual(result["exit_code"], 0, result)
            self.assertEqual(result["threads"], 2)
        self.assertEqual(submission.read_text(), "original")
        self.assertEqual(len(self.worker.state()["experiments"]), 2)
        self.worker.finalize()
        self.assertTrue(validate_submission(self.task.parent, self.files)[0])

    def test_staging_and_temporal_splits_preserve_originals(self):
        from solution_torch import stage_task
        snapshot = {p.name: p.read_bytes() for p in self.task.parent.iterdir()}
        stage_task(self.task, self.folder/"isolated")
        stage_task(self.task, self.folder/"isolated")
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.task.parent.iterdir()})
        frame = pd.read_csv(self.files["train.csv"])
        for tr, va in make_splits(frame, "target", "regression", "time", "period", 3):
            self.assertLess(frame.iloc[tr].period.max(), frame.iloc[va].period.min())
        (self.folder/"isolated"/"train.csv").write_text("changed")
        with self.assertRaises(ValueError):
            stage_task(self.task, self.folder/"isolated")

    def test_archive_entry_point_and_allowlist(self):
        from build_torch_submission import build
        from zipfile import ZipFile
        path = self.folder/"torch.zip"
        build(path)
        with self.assertRaises(FileExistsError):
            build(path)
        with ZipFile(path) as archive:
            self.assertIn("torch_model.py", archive.namelist())
            self.assertIn("legacy_solution.py", archive.namelist())
            self.assertFalse(any(n.endswith((".csv", ".joblib", ".pt")) for n in archive.namelist()))
            archive.extractall(self.folder/"package")
        result = subprocess.run([sys.executable, str(self.folder/"package"/"solution.py"), str(self.task), "--check"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No API calls made", result.stdout)
        self.assertFalse((self.task.parent/"submission.csv").exists())

    def test_nested_oof_torch_and_selected_pipeline_subprocess(self):
        from solution_torch import limited_run_python
        from solution import run_research
        folder = self.folder/"classification"
        smoke_data(folder, True)
        files = discover_files(folder)
        worker = TorchResearch(self.log/"research", files, folder)
        worker.configure("target", "classification", "f1_macro")
        worker.train("baseline", {"iterations": 5, "depth": 2}, "catboost")
        worker.train("network", {"iterations": 2, "width": 16, "layers": 1}, "torch_mlp")
        before = (folder/"submission.csv").read_bytes()
        with patch("solution.run_python", limited_run_python):
            result = run_research("run_classification_suite", {"experiment_ids": [1, 2], "folds": 3},
                folder, files, self.log, 1, 90, module_name="torch_suites.py")
            self.assertEqual(result["exit_code"], 0, result)
            summary = json.loads((self.log/"classification_summary.json").read_text(encoding="utf-8"))
            self.assertEqual((folder/"submission.csv").read_bytes(), before)
            sid = summary["suite_id"]
            for action, arguments in (("select_final_candidate", dict(kind="classification_oof", suite_id=sid)),
                                      ("finalize_selected", {})):
                result = run_research(action, arguments, folder, files, self.log,
                                     2 if arguments else 3, 60, module_name="torch_suites.py")
                self.assertEqual(result["exit_code"], 0, result)
        self.assertTrue(validate_submission(folder, files)[0])
        self.assertTrue((self.log/"final_output.json").exists())

    def test_three_researcher_controller_uses_torch_queue(self):
        import solution
        import parallel_research
        from solution_torch import install_hooks
        run = self.task.parent/"agent_runs"/"integration"
        worker = TorchResearch(run/"research", self.files, self.task.parent)
        worker.configure("target", "regression", "rmse")
        worker.train("baseline", {"iterations": 5, "depth": 2}, "catboost")
        names = ("run_research", "run_python", "fingerprint", "compatible_manifest", "TOOLS", "SYSTEM_PROMPT", "research_requirements")
        originals = {name: getattr(solution, name) for name in names}
        original_batch = parallel_research.run_batch
        try:
            install_hooks()
            (run/"manifest.json").write_text(json.dumps(solution.fingerprint(self.task, self.files)))
            args = NS(task=self.task, resume=run, researchers=3, training_jobs=2, reviewer=False,
                analyst=False, max_steps=1, min_experiments=3, time_budget=360, final_reserve=60,
                step_timeout=180, new_selection=False, require_selection=False)
            barrier = Barrier(3)
            def completion(**kwargs):
                text = kwargs["messages"][0]["content"]
                role = "features" if "Ты исследователь признаков." in text else "rare" if "Ты исследователь редких классов." in text else "models"
                barrier.wait(timeout=10)
                recipe = dict(hypothesis=role, feature_code="")
                if role == "features":
                    recipe.update(base_experiment_id=1, feature_code='def transform(X):\n    X["x2"] = X["x"] ** 2\n    return X')
                else:
                    recipe.update(algorithm="torch_mlp" if role == "models" else "torch_resnet",
                                  params=dict(iterations=2, width=16, layers=1))
                call = NS(function=NS(name="train_model", arguments=json.dumps(recipe)))
                return NS(choices=[NS(message=NS(content="Measured experiment requested", tool_calls=[call]))])
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "OPENAI_BASE_URL": "http://unused.invalid"}), \
                 patch("openai.OpenAI") as client, \
                 patch("parallel_research.available_memory", return_value=16*1024**3):
                client.return_value.__enter__.return_value.chat.completions.create.side_effect = completion
                solution.run_agent(args, self.task.parent, self.files)
            self.assertEqual(len(worker.state()["experiments"]), 4)
            self.assertTrue(worker.state()["finalized"])
            self.assertTrue(validate_submission(self.task.parent, self.files)[0])
        finally:
            for name, value in originals.items():
                setattr(solution, name, value)
            for name in ("_torch_hooks", "run_research_original"):
                if hasattr(solution, name):
                    delattr(solution, name)
            parallel_research.run_batch = original_batch


if __name__ == "__main__":
    unittest.main()
