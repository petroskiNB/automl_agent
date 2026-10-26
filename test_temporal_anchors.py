"""Small anchor orchestration tests; no model fitting, API, or network."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
from reference_recipes import REFERENCE_RECIPES, REGRESSION_TEMPORAL_ANCHORS
from research import write_json
from research_policy import recipe_key
from temporal_anchors import TemporalAnchors


class FakeEngine:
    def __init__(self, state, train, test):
        self.data, self.train_frame, self.test_frame = state, train, test
        self.calls = []

    def state(self):
        return self.data

    def load(self):
        return self.train_frame, self.test_frame

    def train(self, **arguments):
        self.calls.append(arguments)
        record = dict(id=len(self.data['experiments']) + 1, score=41., **arguments)
        self.data['experiments'].append(record)
        return {'experiment': record}


class FakeAnchors(TemporalAnchors):
    def __init__(self, run, engine):
        self.run, self.engine, self.version = run, engine, 'same_data'
        self.suite_calls, self.blends = [], []
        self.fail_source = None

    def _identity(self, config):
        return dict(config=config, hashes={'fixture_data_and_code': self.version})

    def _make(self, config, selected):
        identity = self._identity(config)
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / 'temporal' / sid
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'plan.json', identity)
        results = []
        for mode in config.get('modes', [selected]):
            rows = [dict(period=p, rmse=40. + i) for i, p in enumerate(config['validation_periods'])]
            results.append(dict(mode=mode, folds=rows, mean_rmse=41., latest_rmse=42., worst_rmse=42.))
            np.save(directory / f'{mode}_oof.npy', np.zeros(len(self.engine.train_frame)))
            for i in range(len(rows)):
                np.save(directory / f'{mode}_fold_{i}.npy', np.zeros(2))
                np.savez(directory / f'fold_{i}.npz', train=[0, 1], validation=[2, 3])
        report = dict(suite_id=sid, selected_mode=selected, results=results)
        write_json(directory / 'report.json', report)
        return report

    def run_temporal_suite(self, **arguments):
        self.suite_calls.append(arguments)
        if self.fail_source == len(self.suite_calls):
            raise RuntimeError('Simulated interrupted source')
        state = self.engine.state()
        recipe = next(e for e in state['experiments'] if e['id'] == arguments['base_experiment_id'])
        config = dict(arguments, recipe={k: recipe[k] for k in ('params', 'feature_code', 'algorithm')},
            task={k: state[k] for k in ('target', 'task_type', 'metric', 'features', 'ids')},
            modes=['baseline', 'history_exact', 'history_groups', 'history_hierarchy'])
        return self._make(config, 'history_hierarchy')

    def blend_temporal_suites(self, suite_ids, member_modes=None):
        self.blends.append((suite_ids, member_modes))
        identity = json.loads((self.run / 'temporal' / suite_ids[0] / 'plan.json').read_text())
        common = identity['config']
        config = dict(kind='ensemble', task=common['task'], time_column=common['time_column'],
            validation_periods=common['validation_periods'],
            members=[dict(suite_id=sid, mode=mode) for sid, mode in zip(suite_ids, member_modes)])
        return self._make(config, 'equal_temporal_ensemble')


class TemporalAnchorTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'work' / 'temporal_anchor_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        base = next(r for r in REFERENCE_RECIPES if r['reference_id'] == 'regression_231')
        features = base['required_columns']
        train = pd.DataFrame({c: np.ones(10) for c in features})
        train['Year_Factor'] = np.repeat(np.arange(1, 6), 2)
        train['site_eui'] = 20. + np.arange(10)
        test = train.iloc[:2].drop(columns='site_eui').copy()
        test['Year_Factor'] = 6
        state = dict(task_type='regression', target='site_eui', metric='rmse', features=features,
            ids=['id'], split='time', split_column='Year_Factor', experiments=[])
        self.worker = FakeAnchors(self.folder, FakeEngine(state, train, test))
        environment = patch.dict(os.environ, AUTOML_WORKER_DEADLINE='9999999999')
        environment.start()
        self.addCleanup(environment.stop)

    def test_exact_frozen_recipes_modes_and_current_cache(self):
        original = deepcopy(REFERENCE_RECIPES)
        result = self.worker.run_regression_anchors()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.worker.engine.calls), 2)
        self.assertEqual(len(self.worker.suite_calls), 3)
        self.assertNotIn('iterations', self.worker.engine.calls[1]['params'])
        self.assertEqual(self.worker.blends[0][1], ['history_hierarchy', 'history_exact', 'history_hierarchy'])
        self.assertEqual(result['validation_periods'], [3., 4., 5.])
        self.assertEqual(REFERENCE_RECIPES, original)
        again = self.worker.run_regression_anchors()
        self.assertEqual(again['ensemble_id'], result['ensemble_id'])
        self.assertTrue(all(s['cached'] for s in again['sources']))
        self.assertEqual(len(self.worker.engine.calls), 2)
        self.assertEqual(len(self.worker.suite_calls), 3)
        self.assertNotEqual(again['ensemble_measurements']['mean_rmse'],
                            REGRESSION_TEMPORAL_ANCHORS['historical_mean_rmse'])

    def test_nonmatching_schema_and_deadline_never_train(self):
        self.worker.engine.data['target'] = 'unrelated'
        result = self.worker.run_regression_anchors()
        self.assertFalse(result['applicable'])
        self.assertFalse(self.worker.engine.calls)
        self.worker.engine.data['target'] = 'site_eui'
        with patch.dict(os.environ, AUTOML_WORKER_DEADLINE='1'):
            result = self.worker.run_regression_anchors()
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(self.worker.engine.calls)
        self.assertFalse(self.worker.suite_calls)

    def test_partial_recovery_and_stale_sources_not_reused(self):
        self.worker.fail_source = 2
        partial = self.worker.run_regression_anchors()
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual(len(partial['sources']), 2)
        self.worker.fail_source = None
        recovered = self.worker.run_regression_anchors()
        self.assertEqual(recovered['status'], 'completed')
        self.assertEqual(len(self.worker.engine.calls), 2)
        self.assertEqual(len(self.worker.suite_calls), 4)
        self.worker.version = 'changed_data_or_implementation'
        fresh = self.worker.run_regression_anchors()
        self.assertEqual(fresh['status'], 'completed')
        self.assertTrue(all(not s['cached'] for s in fresh['sources']))
        self.assertEqual(len(self.worker.suite_calls), 7)
        self.assertNotEqual(fresh['ensemble_id'], recovered['ensemble_id'])


if __name__ == '__main__':
    unittest.main()
