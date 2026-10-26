"""Short routing/comparison checks: no network and no model training."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


class RegressionIsolationTests(unittest.TestCase):
    def test_comparison_requires_identical_temporal_rows(self):
        import numpy as np
        import compare_regression as comparison
        scratch = ROOT / 'work/regression_isolation_checks'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder = Path(directory)
            (folder / 'inputs').mkdir()
            periods = np.repeat(np.arange(1, 6), 2)
            (folder / 'inputs/train.csv').write_text('year\n' + ''.join(f'{v}\n' for v in periods), encoding='utf-8')
            hashes = {name: 'test_hash' for name in comparison.INPUTS}
            manifest = dict(arms=dict(control={}, isolated={}), input_hashes=hashes)
            for arm in manifest['arms']:
                run = folder / arm / 'agent_runs/test'
                suite = run / ('temporal/' + 'a' * 16)
                suite.mkdir(parents=True)
                for i, period in enumerate([3, 4, 5]):
                    np.savez(suite / f'fold_{i}.npz', train=np.flatnonzero(periods < period), validation=np.flatnonzero(periods == period))
                comparison.write(suite / 'plan.json', dict(config=dict(task=dict(task_type='regression', metric='rmse'),
                    time_column='year', validation_periods=[3, 4, 5]), hashes=hashes))
                comparison.write(suite / 'report.json', dict(selected_mode='baseline', results=[dict(mode='baseline',
                    mean_rmse=2., latest_rmse=1., worst_rmse=3., folds=[dict(period=p, rmse=float(6-p), rows=2) for p in [3, 4, 5]])]))
                submission = folder / arm / 'submission.csv'
                submission.write_text('id,y\n0,1\n', encoding='utf-8')
                comparison.write(run / 'run_summary.json', dict(status='completed',
                    final_selection=dict(kind='temporal', suite_id='a' * 16, candidate='baseline'),
                    final_output=dict(published=True, mode='baseline', selected_suite='a' * 16),
                    submission_sha256=comparison.digest(submission.read_bytes()), prediction_bounds=dict(lower=0., upper=None)))
            self.assertTrue(comparison.report(folder, manifest)['comparable'])
            # Both folds are causal and have the same year, but evaluate different rows.
            candidate = folder / 'isolated/agent_runs/test/temporal' / ('a' * 16) / 'fold_0.npz'
            np.savez(candidate, train=np.flatnonzero(periods < 3), validation=np.flatnonzero(periods == 3)[:1])
            result = comparison.report(folder, manifest)
            self.assertFalse(result['comparable'])
            self.assertIsNone(result['delta_isolated_minus_control'])

    def test_task_routing_and_platform_defaults(self):
        spec = importlib.util.spec_from_file_location('regression_router', ROOT / 'champion_solution/solution_torch.py')
        router = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / 'champion_solution'))
        try:
            spec.loader.exec_module(router)
        finally:
            sys.path.pop(0)
        self.assertEqual(router.task_kind(ROOT / 'tasks/regression/task.md'), 'regression')
        self.assertEqual(router.task_kind(ROOT / 'tasks/classification/task.md'), 'classification')
        with self.assertRaises(ValueError):
            router.task_kind(ROOT / 'tasks/regression/task.md', 'classification')
        self.assertEqual(router.effective_autogluon_budget('regression', 3600, True, 60), 0)
        self.assertEqual(router.effective_autogluon_budget('classification', None, True, 9900), 3600)
        self.assertEqual(router.effective_autogluon_budget('classification', None, False, 9900), 0)
        self.assertEqual(router.effective_autogluon_budget('unknown', None, True, 9900), 0)
        with self.assertRaises(ValueError):
            router.effective_autogluon_budget('unknown', 1800, True, 9900)

    def test_regression_keeps_autogluon_off_and_exposes_specific_training_tools(self):
        scratch = ROOT / 'work/regression_isolation_checks'
        scratch.mkdir(parents=True, exist_ok=True)
        runner = '''
import json, sys
from torch_resources import configure
configure(8)
import solution_torch as entry
entry.install_hooks(*(['regression'] if sys.argv[1] == 'isolated' else []))
agent = entry.controller()
print(json.dumps(dict(prompt=agent.SYSTEM_PROMPT, tools=agent.TOOLS), ensure_ascii=False, sort_keys=True))
'''
        env = dict(os.environ, AUTOML_AUTOGLUON_BUDGET='0', AUTOML_FLAML_BUDGET='0',
                   AUTOML_LIGHTAUTOML_BUDGET='0', PYTHONPATH='', PYTHONIOENCODING='utf-8')
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            control = Path(directory)
            metadata = json.loads((ROOT / 'controls/champion_v5_1_2157_225.json').read_text(encoding='utf-8'))
            with ZipFile(ROOT / metadata['archive']) as archive:
                for name in archive.namelist():
                    self.assertEqual(Path(name).name, name)
                    (control / name).write_bytes(archive.read(name))
            profiles = []
            for kind, folder in [('control', control), ('isolated', ROOT / 'champion_solution')]:
                process = subprocess.run([sys.executable, '-c', runner, kind], cwd=folder, env=env,
                                         capture_output=True, text=True, encoding='utf-8', timeout=15)
                self.assertEqual(process.returncode, 0, process.stderr)
                profiles.append(json.loads(process.stdout))
            old = {t['function']['name'] for t in profiles[0]['tools']}
            new = {t['function']['name'] for t in profiles[1]['tools']}
            self.assertTrue(old - {'train_model'} <= new)
            self.assertNotIn('train_model', new)
            self.assertTrue({'train_catboost', 'train_lightgbm', 'train_xgboost',
                             'train_torch_mlp', 'train_torch_resnet', 'run_regression_anchors'} <= new)
            self.assertFalse({'run_autogluon_suite', 'run_lightautoml_suite', 'audit_classification_shift'} & new)


if __name__ == '__main__':
    unittest.main()
