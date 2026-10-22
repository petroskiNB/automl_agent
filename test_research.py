import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from research import Research
from solution import discover_files, run_agent, run_research, validate_submission, fingerprint, experiment_memory


class ResearchTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / "work"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rng = np.random.default_rng(42)
        x = rng.normal(size=240)
        data = pd.DataFrame({"id": np.arange(240), "x": x,
                             "category": np.where(x > 0, "yes", "no"),
                             "period": np.repeat([1, 2, 3, 4], 60), "target": x * 3})
        data.loc[0, "category"] = None
        data.to_csv(self.folder / "train.csv", index=False)
        test = data.drop(columns="target").iloc[:20].copy()
        test["id"] += 1000
        test.to_csv(self.folder / "test.csv", index=False)
        pd.DataFrame({"id": test.id, "target": 0}).to_csv(self.folder / "sample_submission.csv", index=False)
        pd.DataFrame({"feature": ["target"], "role": ["target"]}).to_csv(self.folder / "data_dictionary.csv", index=False)
        self.files = discover_files(self.folder)
        self.worker = Research(self.folder / "research", self.files, self.folder)

    def test_regression_winner_and_refit(self):
        self.worker.configure("target", "regression", "rmse", "time", "period")
        split = np.load(self.folder / "research" / "split.npz")
        self.assertEqual(len(split["validation"]), 60)
        a = self.worker.train("baseline", {"iterations": 10, "depth": 2})
        b = self.worker.train("more trees", {"iterations": 20, "depth": 2})
        self.assertAlmostEqual(self.worker.state()["best"]["score"],
                               min(a["experiment"]["score"], b["experiment"]["score"]))
        self.assertTrue(validate_submission(self.folder, self.files)[0])
        self.worker.finalize()
        self.assertTrue(self.worker.state()["finalized"])
        self.assertTrue(validate_submission(self.folder, self.files)[0])
        with self.assertRaises(ValueError):
            self.worker.configure("target", "regression", "mae")
        with self.assertRaises(ValueError):
            self.worker.train("duplicate", {"iterations": 10, "depth": 2})

    def test_lightgbm_sampling_validation_and_refit(self):
        import joblib
        self.worker.configure("target", "regression", "rmse", "time", "period")
        state = self.worker.state()
        for params in ({"subsample": .8}, {"subsample_freq": 1}, {"subsample": 0},
                       {"colsample_bytree": 1.1}, {"reg_alpha": -1}, {"min_split_gain": float("nan")},
                       {"subsample_freq": 1.5}, {"subsample": True}):
            with self.assertRaises(ValueError):
                self.worker.model(state, params, "lightgbm")
        with self.assertRaises(ValueError):
            self.worker.model(state, {"subsample": .8}, "catboost")
        from lightgbm import LGBMRegressor
        baseline = self.worker.model(state, {}, "lightgbm")
        expected = LGBMRegressor(n_estimators=200, max_depth=5, num_leaves=31,
            min_child_samples=20, learning_rate=.1, reg_lambda=0, n_jobs=8, random_state=42, verbosity=-1)
        self.assertEqual(baseline.get_params(), expected.get_params())
        options = dict(subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_alpha=.1, min_split_gain=.01)
        result = self.worker.train("sampling", dict(iterations=20, depth=3, **options), "lightgbm")
        bundle = joblib.load(result["experiment"]["model_path"])
        final = self.worker.refit(bundle, self.worker.load()[0], state)
        for model in (bundle["model"], final["model"]):
            for key, value in options.items():
                self.assertEqual(model.get_params()[key], value)
        self.assertEqual(final["model"].n_estimators, result["experiment"]["selected_iterations"])

    def test_classification_labels_and_weights(self):
        path = self.folder / "train.csv"
        df = pd.read_csv(path)
        df["target"] = np.where(df.x > .8, "rare", "common")
        df.to_csv(path, index=False)
        self.worker.configure("target", "classification", "f1_macro")
        result = self.worker.train("balanced", {"iterations": 10, "depth": 2,
                                               "auto_class_weights": "SqrtBalanced"})
        self.assertGreaterEqual(result["experiment"]["score"], 0)
        self.assertIn("rare", result["experiment"]["class_metrics"])
        self.worker.finalize()
        self.assertTrue(validate_submission(self.folder, self.files)[0])

    def test_worker_protocol(self):
        logs = self.folder / "logs"
        logs.mkdir()
        result = run_research("configure", {"target": "target", "task_type": "regression",
                                            "metric": "rmse"}, self.folder, self.files, logs, 1, 30)
        self.assertEqual(result["exit_code"], 0, result)
        result = run_research("train", {"hypothesis": "smoke", "params": {"iterations": 10}},
                              self.folder, self.files, logs, 2, 30)
        self.assertEqual(result["exit_code"], 0, result)
        self.assertTrue(json.loads(result["stdout"])["improved"])

    def test_lightgbm_parameters(self):
        self.worker.configure("target", "regression", "rmse")
        state = self.worker.state()
        model = self.worker.model(state, {"num_leaves": 63, "depth": 8,
                                         "min_child_samples": 50}, "lightgbm")
        self.assertEqual(model.get_params()["num_leaves"], 63)
        self.assertEqual(model.get_params()["min_child_samples"], 50)
        with self.assertRaises(ValueError):
            self.worker.model(state, {"num_leaves": 63, "depth": 4}, "lightgbm")
        with self.assertRaises(ValueError):
            self.worker.model(state, {"num_leaves": 63}, "catboost")

    def test_native_categories_preserve_full_strings_and_unknowns(self):
        train = pd.DataFrame({"facility": ["Education_College", "Education_Other", "Office"]})
        encoded, levels = self.worker.encode(train, ["facility"], "lightgbm")
        self.assertEqual(len(levels["facility"]), 3)
        self.assertNotEqual(encoded.facility.cat.codes.iloc[0], encoded.facility.cat.codes.iloc[1])
        validation = pd.DataFrame({"facility": ["Education_Other", "Unseen"]})
        encoded, _ = self.worker.encode(validation, ["facility"], "lightgbm", levels)
        self.assertEqual(encoded.facility.iloc[0], "Education_Other")
        self.assertTrue(pd.isna(encoded.facility.iloc[1]))

    def test_features_early_stopping_ensemble(self):
        self.worker.configure("target", "regression", "rmse")
        code = "def transform(X):\n    X['square'] = X['x'] ** 2\n    return X.drop(columns=['period'])"
        a = self.worker.train("features", {"iterations": 100, "depth": 3}, feature_code=code,
                              early_stopping_rounds=10)
        b = self.worker.train("other algorithm", {"iterations": 100, "depth": 3}, algorithm="lightgbm",
                              feature_code=code, early_stopping_rounds=10)
        self.assertLessEqual(a["experiment"]["selected_iterations"], 100)
        self.assertLessEqual(b["experiment"]["selected_iterations"], 100)
        blend = self.worker.blend([1, 2])
        import joblib
        bundle = joblib.load(blend["experiment"]["model_path"])
        train, _ = self.worker.load()
        state = self.worker.state()
        refit = self.worker.refit(bundle, train, state)
        self.worker.publish(refit)
        self.assertTrue(validate_submission(self.folder, self.files)[0])
        for invalid in ("def transform(X):\n    return X.iloc[::-1]",
                        "def transform(X):\n    X['mean'] = X['x'].mean()\n    return X",
                        "import os\ndef transform(X):\n    return X"):
            with self.assertRaises(ValueError):
                self.worker.prepare(train, state, invalid)

    def test_probability_ensemble(self):
        path = self.folder / "train.csv"
        df = pd.read_csv(path)
        df["target"] = np.where(df.x < -.5, "A", np.where(df.x > .5, "B", "C"))
        df.to_csv(path, index=False)
        self.worker.configure("target", "classification", "f1_macro")
        self.worker.train("cat", {"iterations": 30, "depth": 3}, early_stopping_rounds=5)
        self.worker.train("lgb", {"iterations": 30, "depth": 3}, algorithm="lightgbm", early_stopping_rounds=5)
        result = self.worker.blend([1, 2], [.4, .6])
        import joblib
        bundle = joblib.load(result["experiment"]["model_path"])
        train, test = self.worker.load()
        state = self.worker.state()
        probabilities = self.worker.predict_raw(bundle, test, state)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.worker.publish(self.worker.refit(bundle, train, state))
        self.assertTrue(validate_submission(self.folder, self.files)[0])

    def test_controller_with_fake_api(self):
        # Real worker training; only the remote Qwen responses are replaced.
        from openai.types.chat import ChatCompletionMessage
        messages = [
            ChatCompletionMessage(role="assistant", tool_calls=[{
                "id": "one", "type": "function", "function": {"name": "configure_task",
                "arguments": json.dumps({"target": "target", "task_type": "regression", "metric": "rmse"})}}]),
            ChatCompletionMessage(role="assistant", content="Already done"),
            ChatCompletionMessage(role="assistant", tool_calls=[{
                "id": "two", "type": "function", "function": {"name": "train_model",
                "arguments": json.dumps({"hypothesis": "baseline", "params": {"iterations": 10}})}}]),
            ChatCompletionMessage(role="assistant", content="Finished"),
        ]
        task = self.folder / "task.md"
        task.write_text("Predict target, RMSE", encoding="utf-8")
        args = SimpleNamespace(task=task, time_budget=180, final_reserve=50,
                               max_steps=6, step_timeout=30, min_experiments=1)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake", "OPENAI_BASE_URL": "http://unused.invalid"}), \
             patch("openai.OpenAI") as client, patch("builtins.print"):
            create = client.return_value.__enter__.return_value.chat.completions.create
            create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(message=m)]) for m in messages]
            run_agent(args, self.folder, self.files)
            self.assertEqual(create.call_count, 4)
        self.assertTrue(validate_submission(self.folder, self.files)[0])
        state_path = next(self.folder.glob("agent_runs/*/research/state.json"))
        self.assertTrue(json.loads(state_path.read_text())["finalized"])

    def test_resume_with_reviewer_and_changed_data(self):
        from openai.types.chat import ChatCompletionMessage
        task = self.folder / "task.md"
        task.write_text("Regression RMSE", encoding="utf-8")
        run = self.folder / "agent_runs" / "original"
        worker = Research(run / "research", self.files, self.folder)
        worker.configure("target", "regression", "rmse")
        worker.train("old", {"iterations": 10, "depth": 2})
        (run / "step_001.py").write_text("# old step", encoding="utf-8")
        (run / "conversation.json").write_text("[]")
        (run / "manifest.json").write_text(json.dumps(fingerprint(task, self.files)))
        args = SimpleNamespace(task=task, resume=run, reviewer=True, review_every=6,
                               time_budget=300, final_reserve=50, max_steps=3,
                               step_timeout=30, min_experiments=1)
        args.instructions = self.folder / "plan.md"
        args.instructions.write_text("Check full categories before hash scans", encoding="utf-8")
        responses = [ChatCompletionMessage(role="assistant", content="Check stability on another period"),
                     ChatCompletionMessage(role="assistant", tool_calls=[{
                         "id": "new", "type": "function", "function": {"name": "train_model",
                         "arguments": json.dumps({"hypothesis": "new", "params": {"iterations": 20, "depth": 2}})}}]),
                     ChatCompletionMessage(role="assistant", content="Done")]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake", "OPENAI_BASE_URL": "http://unused.invalid"}), \
             patch("openai.OpenAI") as client, patch("builtins.print"):
            create = client.return_value.__enter__.return_value.chat.completions.create
            research_responses = iter(responses[1:] + [responses[-1]])
            def fake_completion(**kwargs):
                message = next(research_responses) if "tools" in kwargs else responses[0]
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])
            create.side_effect = fake_completion
            run_agent(args, self.folder, self.files)
            self.assertGreaterEqual(create.call_count, 3)
            self.assertEqual(sum("tools" not in call.kwargs for call in create.call_args_list), 1)
        self.assertEqual((run / "step_001.py").read_text(), "# old step")
        self.assertEqual(len(experiment_memory(run)["experiments"]), 2)
        self.assertTrue((run / "reviews.jsonl").exists())
        self.assertTrue(list(run.glob("conversation_before_*.json")))
        self.assertEqual(next(run.glob("instructions_*.md")).read_text(encoding="utf-8"),
                         "Check full categories before hash scans")
        with (self.folder / "train.csv").open("a") as stream:
            stream.write("\n")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake", "OPENAI_BASE_URL": "http://unused.invalid"}):
            with self.assertRaisesRegex(ValueError, "changed"):
                run_agent(args, self.folder, self.files)
        self.assertFalse((self.folder / ".agent.lock").exists())


if __name__ == "__main__":
    unittest.main()
