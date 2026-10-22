import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from solution import Advisors


class AdvisorTests(unittest.TestCase):
    def test_parallel_snapshot_and_delivery(self):
        barrier = Barrier(3)
        seen = []

        def answer(client, log, task, remaining, role, context, count):
            seen.append((role, context, count))
            barrier.wait(timeout=5)
            return role + " recommendation"

        with ThreadPoolExecutor(max_workers=2) as pool, \
             patch("solution.reviewer_context", return_value="immutable snapshot"), \
             patch("solution.review_experiments", side_effect=answer):
            worker = Advisors(pool, None, Path("unused"), "task", ["reviewer", "analyst"])
            worker.start(300, 28)
            messages = []
            self.assertFalse(worker.collect(messages))
            self.assertEqual(len(worker.pending), 2)
            barrier.wait(timeout=5)  # Both API workers started without blocking the controller.
            self.assertTrue(worker.collect(messages, wait=True))
            self.assertEqual(len(messages), 2)
            self.assertTrue(all("28" in m["content"] for m in messages))
            self.assertEqual({v[0] for v in seen}, {"reviewer", "analyst"})
            self.assertTrue(all(v[1:] == ("immutable snapshot", 28) for v in seen))

    def test_failure_disables_only_failed_role(self):
        def answer(client, log, task, remaining, role, context, count):
            if role == "analyst":
                raise RuntimeError("provider unavailable")
            return "valid review"

        with ThreadPoolExecutor(max_workers=2) as pool, \
             patch("solution.reviewer_context", return_value="snapshot"), \
             patch("solution.review_experiments", side_effect=answer):
            worker = Advisors(pool, None, Path("unused"), "task", ["reviewer", "analyst"])
            worker.start(300, 28)
            messages = []
            worker.collect(messages, wait=True)
            self.assertEqual(worker.roles, {"reviewer"})
            self.assertEqual(len(messages), 1)
            self.assertFalse(worker.pending)


if __name__ == "__main__":
    unittest.main()
