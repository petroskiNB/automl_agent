import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from research import Research
from solution import discover_files, run_agent, run_research, validate_submission


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


if __name__ == "__main__":
    unittest.main()
