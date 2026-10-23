import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace as NS
from unittest.mock import Mock

from research_team import propose_round, ResearchTeamUnavailable


class TeamTests(unittest.TestCase):
    def test_all_timeouts_remain_retryable_and_errors_are_logged(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError()
        root = Path(__file__).resolve().parent / 'work'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder, ThreadPoolExecutor(3) as pool:
            with self.assertRaises(ResearchTeamUnavailable) as caught:
                propose_round(client, pool, 'qwen', 'system', 'task', '', {}, [], [],
                              600, 1, Path(folder), lambda *args: None, researchers=3)
            self.assertTrue(caught.exception.retryable)
            record = json.loads((Path(folder) / 'research_team.jsonl').read_text())
            self.assertEqual(len(record['failures']), 3)
            client.chat.completions.create.side_effect = ValueError('bad request')
            with self.assertRaises(ResearchTeamUnavailable) as caught:
                propose_round(client, pool, 'qwen', 'system', 'task', '', {}, [], [],
                              600, 2, Path(folder), lambda *args: None, researchers=3)
            self.assertFalse(caught.exception.retryable)

    def test_sequential_recovery_has_only_one_active_request(self):
        import time
        from threading import Lock
        lock, active, peak = Lock(), [0], [0]
        def create(**kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(.01)
            with lock:
                active[0] -= 1
            self.assertGreater(kwargs['timeout'], 90)
            return NS(choices=[NS(message=NS(content='done', tool_calls=[]))])
        client = Mock()
        client.chat.completions.create.side_effect = create
        root = Path(__file__).resolve().parent / 'work'
        with tempfile.TemporaryDirectory(dir=root) as folder, ThreadPoolExecutor(3) as pool:
            propose_round(client, pool, 'qwen', 'system', 'task', '', {}, [], [],
                          600, 1, Path(folder), lambda *args: None, researchers=3, sequential=True)
        self.assertEqual(peak[0], 1)
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_parallel_proposals_deduplicated_serial_queue(self):
        barrier = Barrier(2)
        def create(**kwargs):
            barrier.wait(timeout=5)
            call = NS(function=NS(name="train_model", arguments=json.dumps(dict(
                hypothesis=kwargs["messages"][0]["content"], algorithm="catboost", params={"depth": 3}))))
            return NS(choices=[NS(message=NS(content="idea", tool_calls=[call]))])
        client = Mock()
        client.chat.completions.create.side_effect = create
        root = Path(__file__).resolve().parent / "work"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder, ThreadPoolExecutor(2) as pool:
            message = propose_round(client, pool, "qwen", "system", "task", "plan", {}, [], [],
                                    200, 1, Path(folder), lambda *args: None)
            self.assertEqual(len(message.tool_calls), 1)
            self.assertEqual(message.model_dump()["tool_calls"][0]["function"]["name"], "train_model")
            self.assertTrue((Path(folder) / "research_team.jsonl").exists())

    def test_one_failed_branch_keeps_other_proposal(self):
        def create(**kwargs):
            if "исследователь признаков" in kwargs["messages"][0]["content"]:
                raise TimeoutError()
            return NS(choices=[NS(message=NS(content="done", tool_calls=[]))])
        client = Mock()
        client.chat.completions.create.side_effect = create
        root = Path(__file__).resolve().parent / "work"
        with tempfile.TemporaryDirectory(dir=root) as folder, ThreadPoolExecutor(2) as pool:
            result = propose_round(client, pool, "qwen", "system", "task", "plan", {}, [], [],
                                   200, 1, Path(folder), lambda *args: None)
            self.assertIn("models: done", result.content)


if __name__ == "__main__":
    unittest.main()
