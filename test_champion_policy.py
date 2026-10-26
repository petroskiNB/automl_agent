"""Small synthetic policy/leakage checks; no model training and no API calls."""
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))


def module(name):
    spec = importlib.util.spec_from_file_location('policy_test_' + name, ROOT / 'champion_solution' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


policy = module('research_policy')


class PolicyTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'work' / 'policy_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_future_evidence_rejects_random_but_not_overlapping_data(self):
        train = pd.DataFrame({'Year_Factor': [1, 2, 3, 4], 'year_built': [1980]*4})
        test = pd.DataFrame({'Year_Factor': [5, 5], 'year_built': [1980]*2})
        files = {n: self.root / n for n in ('train.csv', 'test.csv')}
        train.to_csv(files['train.csv'], index=False)
        test.to_csv(files['test.csv'], index=False)
        profile = policy.validation_profile(files)
        self.assertEqual(profile['recommended']['validation_periods'], [3., 4.])
        with self.assertRaises(ValueError):
            policy.guard_split(profile, 'random', None)
        policy.guard_split(profile, 'time', 'Year_Factor')
        test['Year_Factor'] = [2, 4]
        test.to_csv(files['test.csv'], index=False)
        policy.guard_split(policy.validation_profile(files), 'random', None)

    def test_old_recipe_is_not_a_current_measurement_and_goal_is_bounded(self):
        ref = dict(reference_id='anchor', target='y', task_type='regression', required_columns=['x'],
                   algorithm='catboost', params={'depth': 7}, feature_code='', historical_score=.01)
        state = dict(target='y', task_type='regression', metric='rmse', features=['x'], split='random',
                     experiments=[], best=None)
        with patch.object(policy, 'library', return_value=[ref]):
            self.assertTrue(any('Повторно измерить' in s for s in policy.progress(self.root, state, {})['pending']))
            state['experiments'] = [dict(id=1, algorithm='catboost', params={'depth': 7}, feature_code='', score=10.)]
            state['best'] = state['experiments'][0]
            result = policy.progress(self.root, state, {})
            self.assertEqual(result['improvement_goal']['current_reference_score'], 10.)
            self.assertTrue(any('трёх новых' in s for s in result['pending']))
            state['experiments'] += [dict(id=i+2, algorithm='lightgbm', params={'iterations': i+10},
                                         feature_code='', score=11.) for i in range(3)]
            result = policy.progress(self.root, state, {})
            self.assertFalse(result['improvement_goal']['target_achieved'])
            self.assertFalse(any('трёх новых' in s for s in result['pending']))

    def test_temporal_stages_require_two_recipes_on_identical_periods(self):
        profile = dict(future_columns=[dict(column='period', validation_periods=[3., 4.])])
        state = dict(target='y', task_type='regression', metric='rmse', split='time', split_column='period',
                     features=['period', 'x'], experiments=[dict(id=i, algorithm=a, params={}, feature_code='', score=10.)
                     for i, a in enumerate(('catboost', 'lightgbm', 'xgboost'), 1)], best=None)
        with patch.object(policy, 'library', return_value=[]):
            self.assertFalse(policy.progress(self.root, state, profile)['complete'])
            for sid, algorithm in [('aaa', 'catboost'), ('bbb', 'lightgbm')]:
                folder = self.root / 'temporal' / sid
                folder.mkdir(parents=True)
                plan = dict(config=dict(time_column='period', validation_periods=[3., 4.],
                    recipe=dict(algorithm=algorithm, params={}, feature_code='')))
                (folder / 'plan.json').write_text(json.dumps(plan))
                (folder / 'report.json').write_text(json.dumps(dict(suite_id=sid,
                    selected_mode='baseline',
                    results=[dict(mode=m, mean_rmse=10., latest_rmse=10., worst_rmse=10.)
                             for m in ('baseline', 'history_exact', 'history_groups')])))
            (self.root / 'final_selection.json').write_text(json.dumps(dict(kind='temporal', suite_id='bbb')))
            self.assertFalse(any('два разных финалиста' in s for s in policy.progress(self.root, state, profile)['pending']))
            path = self.root / 'temporal/bbb/plan.json'
            plan = json.loads(path.read_text()); plan['config']['validation_periods'] = [2., 3.]
            path.write_text(json.dumps(plan))
            self.assertTrue(any('два разных финалиста' in s for s in policy.progress(self.root, state, profile)['pending']))

    def test_hybrid_inner_selection_never_sees_outer_future(self):
        temporal = module('temporal_research')
        seen = []
        class Model:
            selected_iterations = 7
            def predict(self, x): return np.ones(len(x))
        class Engine:
            def prepare(self, frame, state, code): return frame[state['features']].copy(), []
            def encode(self, x, cats, algorithm, levels=None): return x, None
            def model(self, state, params, algorithm): return Model()
            def _fit(self, model, algorithm, state, x, y, cats, xv=None, yv=None, patience=0):
                seen.append((x['period'].tolist(), None if xv is None else xv['period'].tolist()))
        worker = temporal.TemporalResearch.__new__(temporal.TemporalResearch)
        worker.engine = Engine()
        frame = pd.DataFrame(dict(period=[1., 2., 3., 4.], x=[1., 2., 3., 4.], y=[2., 3., 4., 999999.]))
        state = dict(features=['period', 'x'], target='y')
        recipe = dict(algorithm='xgboost', params={}, feature_code='')
        config = dict(time_column='period')
        past = frame.loc[frame.period < 4]
        iterations = worker._iterations(past, state, recipe, config, 'baseline')
        worker._fit(past, state, recipe, config, 'baseline', iterations)
        self.assertEqual(seen, [([1., 2.], [3.]), ([1., 2., 3.], None)])


if __name__ == '__main__':
    unittest.main()
