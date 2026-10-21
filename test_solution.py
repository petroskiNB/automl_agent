"""Offline checks: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path

from solution import discover_files, run_python, validate_submission


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


if __name__ == "__main__":
    unittest.main()
