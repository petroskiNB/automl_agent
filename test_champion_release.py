"""Two small controller checks; no training and no network calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('champion_' + name, ROOT / 'champion_solution' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ControllerTests(unittest.TestCase):
    def test_empty_retry_preserves_valid_tool_calls(self):
        team = module('research_team')
        for text, reason in [(None, 'stop'), ('  ', 'stop'), ('unfinished', 'length')]:
            response = NS(choices=[NS(message=NS(content=text, tool_calls=[]), finish_reason=reason)])
            with self.assertRaises(team.EmptyResearchResponse) as caught:
                team.validate_response(response)
            self.assertTrue(team.transient_error(caught.exception))
        message = NS(content=None, tool_calls=[NS()])
        self.assertIs(team.validate_response(NS(choices=[NS(message=message, finish_reason='tool_calls')])), message)

    def test_failure_audit_does_not_claim_old_submission_is_success(self):
        audit = module('release_audit')
        scratch = ROOT / 'work' / 'champion_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder = Path(directory)
            run = folder / 'logs'
            (run / 'research').mkdir(parents=True)
            (folder / 'submission.csv').write_text('id,y\n1,0\n', encoding='utf-8')
            state = dict(experiments=[dict(id=1, algorithm='xgboost', score=.6)], best=dict(id=1))
            (run / 'research/state.json').write_text(json.dumps(state), encoding='utf-8')
            audit.write_summary(run, NS(researchers=1), folder, 3., 'TimeoutError')
            result = json.loads((run / 'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error_type'], 'TimeoutError')
            self.assertIsNone(result['final_output'])
            self.assertTrue(result['submission_exists'])
            self.assertEqual(len(result['submission_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
