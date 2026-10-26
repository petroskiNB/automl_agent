"""Small parameter-schema and repeat-rejection checks, no training or API."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
import solution_torch
from training_contract import TRAINING_TOOLS, canonical_arguments, validate_arguments


class TrainingContractTests(unittest.TestCase):
    def test_split_schemas_match_algorithms(self):
        with patch.dict(os.environ, AUTOML_AUTOGLUON_BUDGET='0', AUTOML_FLAML_BUDGET='0', AUTOML_LIGHTAUTOML_BUDGET='0'):
            solution_torch.install_hooks('regression')
        functions = {t['function']['name']: t['function'] for t in solution_torch.controller().TOOLS}
        self.assertNotIn('train_model', functions)
        for name in TRAINING_TOOLS:
            self.assertIn(name, functions)
        cat = functions['train_catboost']['parameters']['properties']['params']['properties']
        net = functions['train_torch_mlp']['parameters']['properties']['params']['properties']
        self.assertFalse({'width', 'loss', 'target_transform'} & set(cat))
        self.assertTrue({'width', 'loss', 'target_transform'} <= set(net))
        self.assertEqual(canonical_arguments('train_xgboost', {'hypothesis': 'test'})['algorithm'], 'xgboost')

    def test_irrelevant_parameters_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'unsupported parameters'):
            validate_arguments(dict(algorithm='catboost', params={'width': 1024, 'loss': 'auto'}))
        valid = validate_arguments(dict(algorithm='catboost', params={'depth': 7, 'auto_class_weights': None}), 'regression')
        self.assertEqual(valid['params'], {'depth': 7})

    def test_failed_request_repeats_do_not_start_another_worker(self):
        scratch = ROOT/'work'/'training_contract_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            folder = Path(temporary)
            args = dict(hypothesis='test', algorithm='catboost', params={'width': 1024}, feature_code='')
            controller = solution_torch.controller()
            with patch.object(controller, 'run_research_original', create=True, side_effect=AssertionError('No worker should start')):
                first = solution_torch.run_worker('train', args, folder, {}, folder, 1, 10)
                second = solution_torch.run_worker('train', dict(args, hypothesis='different prose'), folder, {}, folder, 2, 10)
            self.assertEqual(first['exit_code'], 1)
            self.assertTrue(second['repeated_failure'])


if __name__ == '__main__':
    unittest.main()
