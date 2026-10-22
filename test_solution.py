"""Offline checks: python -m unittest -v"""
import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

from solution import discover_files, run_python, validate_submission, review_experiments, reviewer_context


class AgentTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "work"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, text in {
            "train.csv": "id,x,target\n1,2,A\n2,3,B\n",
            "test.csv": "id,x\n3,4\n4,5\n",
            "sample_submission.csv": "id,target\n3,A\n4,A\n",
            "data_dictionary.csv": "feature,role\ntarget,target\n",
        }.items():
            (self.root / name).write_text(text, encoding="utf-8")
        self.files = discover_files(self.root)

    def test_submission(self):
        path = self.root / "submission.csv"
        self.assertFalse(validate_submission(self.root, self.files)[0])
        path.write_text("id,target\n3,A\n4,B\n", encoding="utf-8")
        self.assertTrue(validate_submission(self.root, self.files)[0])
        for text in ("id,target\n4,A\n3,B\n", "id,target\n3,A\n4,C\n",
                     "id,target\n3,A\n4,\n", "id,target\n3,A\n3,B\n"):
            path.write_text(text, encoding="utf-8")
            self.assertFalse(validate_submission(self.root, self.files)[0])

    def test_numeric_nonfinite(self):
        (self.root / "train.csv").write_text("id,x,target\n1,2,1.5\n2,3,2.5\n")
        path = self.root / "submission.csv"
        path.write_text("id,target\n3,2.0\n4,3.0\n")
        self.assertTrue(validate_submission(self.root, self.files)[0])
        path.write_text("id,target\n3,nan\n4,inf\n")
        self.assertFalse(validate_submission(self.root, self.files)[0])

    def test_execution_error_and_timeout(self):
        result = run_python("from pathlib import Path; print(Path('train.csv').exists())",
                            self.root, self.root, 1, 5)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("True", result["stdout"])
        result = run_python("raise ValueError('test error')", self.root, self.root, 2, 5)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("test error", result["stderr"])
        result = run_python("import time; time.sleep(10)", self.root, self.root, 3, 0.2)
        self.assertEqual(result["exit_code"], -1)

    def test_reviewer_retries_empty_and_truncated_responses(self):
        for content in (None, "   ", "Незавершённая проверка"):
            with self.subTest(content=content):
                client = Mock()
                def response(text, reason):
                    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                        message=SimpleNamespace(content=text), finish_reason=reason)])
                create = client.chat.completions.create
                create.side_effect = [response(content, "length"), response("Проверь категории.", "stop")]
                self.assertEqual(review_experiments(client, self.root, "Задача", 200), "Проверь категории.")
                self.assertEqual([c.kwargs["max_tokens"] for c in create.call_args_list], [8000, 12000])
                self.assertEqual(create.call_args_list[0].kwargs["reasoning_effort"], "none")
                self.assertNotIn("reasoning_effort", create.call_args_list[1].kwargs)
                logs = [json.loads(s) for s in (self.root / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(logs[-2]["status"], "invalid_response")
                self.assertEqual(logs[-2]["finish_reason"], "length")
                self.assertEqual(logs[-1]["status"], "ok")

    def test_reviewer_failure_is_not_feedback(self):
        client = Mock()
        create = client.chat.completions.create
        create.return_value = SimpleNamespace(usage=None, choices=[])
        with self.assertRaisesRegex(RuntimeError, "Проверка не выполнена"):
            review_experiments(client, self.root, "Задача", 200)
        self.assertEqual(create.call_count, 2)
        create.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "Недостаточно времени"):
            review_experiments(client, self.root, "Задача", 0)
        create.assert_not_called()

    def test_reviewer_recovers_from_api_error(self):
        client = Mock()
        client.chat.completions.create.side_effect = [TimeoutError("secret"), SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="Замечание"), finish_reason="stop")])]
        self.assertEqual(review_experiments(client, self.root, "Задача", 200), "Замечание")
        log = (self.root / "reviews.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("secret", log)
        self.assertIn("TimeoutError", log)

    def test_reviewer_context_keeps_best_members_and_limits_history(self):
        root = self.root / "research"
        root.mkdir()
        experiments = [{"id": i, "algorithm": "lightgbm", "params": {}, "score": float(i),
                        "feature_code": "def transform(X): return X", "hypothesis": "x" * 5000}
                       for i in range(1, 201)]
        experiments[99]["feature_code"] = "def _h(s): return s.encode().hex()[:14]"
        best = {"id": 201, "algorithm": "ensemble", "params": {"ids": [1, 100, 101]}, "score": .5}
        experiments.append(best)
        (root / "state.json").write_text(json.dumps({"metric": "rmse", "experiments": experiments,
                                                     "best": best}), encoding="utf-8")
        text = reviewer_context(self.root, "task" * 10000)
        context = json.loads(text)
        self.assertLess(len(text), 18000)
        self.assertEqual(context["experiment_count"], 201)
        self.assertIn("100", context["best_feature_code"])
        self.assertTrue(context["controller_warnings"])
        self.assertLessEqual(len(context["selected_experiments"]), 13)


if __name__ == "__main__":
    unittest.main()
