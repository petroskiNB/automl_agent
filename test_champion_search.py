"""Focused causality/search/export checks with dummy estimators; no ML training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'champion_solution'))
import numpy as np
import pandas as pd
import joblib
from hierarchical_history import HierarchicalHistory
from temporal_research import TemporalResearch
from temporal_search import catalogue, search_status
from temporal_ensemble import blend_suites
from prediction_policy import ResidualRegressor, bounded_model
from final_output import FinalOutput


class DummyModel:
    def __init__(self, offset=0.): self.offset = offset
    def predict(self, x): return x['x'].to_numpy() + self.offset


class DummyEngine:
    def __init__(self, train, test): self.train, self.test, self.labels = train, test, []
    def load(self): return self.train, self.test
    def prepare(self, frame, state, code): return frame[state['features']].copy(), []
    def encode(self, x, cats, algorithm, levels=None): return x, None
    def model(self, state, params, algorithm): return DummyModel(params.get('offset', 0.))
    def _fit(self, model, algorithm, state, x, y, cats, xv=None, yv=None, patience=0): self.labels.append(y.to_numpy())


class FixedHistory:
    def transform(self, frame): return pd.DataFrame({'hist_baseline': frame['baseline']}, index=frame.index)


class SearchTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'work/search_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.state = dict(task_type='regression', target='y', metric='rmse', split='time', split_column='period',
                          features=['period', 'x'], ids=['id'])

    def report(self, sid, score, kind='single', offset=0.):
        folder = self.run/'temporal'/sid
        folder.mkdir(parents=True)
        config = dict(task=self.state, time_column='period', validation_periods=[3., 4.],
                      recipe=dict(algorithm='catboost', params=dict(iterations=1, offset=offset), feature_code=''),
                      key_columns=['x'], group_columns=['x'], modes=['baseline'])
        if kind == 'ensemble': config['kind'] = 'ensemble'
        report = dict(suite_id=sid, selected_mode='baseline', results=[dict(mode='baseline', mean_rmse=score,
            latest_rmse=score, worst_rmse=score,
            folds=[dict(iterations=1, period=period, rmse=score) for period in (3., 4.)])])
        (folder/'plan.json').write_text(json.dumps(dict(config=config)))
        (folder/'report.json').write_text(json.dumps(report))
        return config, report

    def test_hierarchical_matches_are_past_only_and_widen_coverage(self):
        frame = pd.DataFrame(dict(period=[1., 2., 3., 4.], region=['A']*4, kind=['Office']*4,
            floor_area=[1000.1, 1000.4, 1000.2, 1000.3], built=[1990, 1990, 2010, 2010], y=[10., 20., 999., 9999.]))
        args = ('period', 'y', ['region', 'kind', 'floor_area', 'built'], ['region', 'kind'])
        kwargs = dict(key_variants=[['region', 'kind', 'floor_area']], group_hierarchy=[['kind']],
                      smoothing=5., area_rounding=100.)
        history = HierarchicalHistory(*args, **kwargs).fit(frame)
        query = frame.iloc[[2]]
        first = history.transform(query)
        changed = frame.copy(); changed.loc[changed.period >= 3, 'y'] = -999999.
        second = HierarchicalHistory(*args, **kwargs).fit(changed).transform(query.assign(y=-12345))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.hist_exact_seen.iloc[0], 0.)
        self.assertEqual(first.hist_any_key_seen.iloc[0], 1.)
        self.assertTrue(np.isfinite(first.hist_baseline.iloc[0]))
        self.assertTrue(np.isnan(history.transform(frame.iloc[[0]]).hist_baseline.iloc[0]))

    def test_residual_targets_drop_warmup_and_add_back_before_clipping(self):
        frame = pd.DataFrame(dict(period=[1., 2., 3.], x=[1., 2., 3.], baseline=[np.nan, 10., 20.], y=[999., 20., 30.]))
        engine = DummyEngine(frame, frame)
        worker = TemporalResearch.__new__(TemporalResearch); worker.engine = engine
        worker._history = lambda *args: FixedHistory()
        state = dict(self.state, features=['period', 'x'])
        recipe = dict(algorithm='catboost', params={'offset': -3.}, feature_code='')
        bundle = worker._fit(frame, state, recipe, {'time_column': 'period'}, 'residual_history', 10)
        np.testing.assert_array_equal(engine.labels[-1], [10., 10.])
        self.assertEqual(bundle['residual_warmup_rows_dropped'], 1)
        query = pd.DataFrame(dict(period=[4.], x=[0.], baseline=[10.]))
        self.assertEqual(worker._predict(bundle, query)[0], 7.)
        physical = dict(task_type='regression', prediction_bounds={'lower': 0., 'upper': None})
        model = bounded_model(ResidualRegressor(DummyModel(-3.)), physical)
        self.assertEqual(model.predict(pd.DataFrame({'x': [0.], 'hist_baseline': [10.]}))[0], 7.)
        self.assertEqual(worker._iterations(frame.iloc[:2], state, dict(recipe, params={'iterations': 1000}),
                                          {'time_column': 'period'}, 'residual_history'), 200)

    def test_equal_ensemble_checks_rows_and_exports_member_pipelines(self):
        train = pd.DataFrame(dict(id=range(8), period=np.repeat([1., 2., 3., 4.], 2), x=np.arange(8.)+10))
        train['y'] = train.x
        test = pd.DataFrame(dict(id=[100, 101], period=[5., 5.], x=[20., 30.]))
        worker = TemporalResearch.__new__(TemporalResearch)
        worker.run, worker.engine = self.run, DummyEngine(train, test)
        worker._identity = lambda config: dict(config=config)
        ids = [f'{i:016x}' for i in (1, 2, 3)]
        for sid, offset in zip(ids, [1., -1., 2.]):
            self.report(sid, abs(offset), offset=offset)
            folder = self.run/'temporal'/sid
            for i, period in enumerate([3., 4.]):
                tr = np.flatnonzero(train.period.to_numpy() < period)
                va = np.flatnonzero(train.period.to_numpy() == period)
                np.savez(folder/f'fold_{i}.npz', train=tr, validation=va)
                np.save(folder/f'baseline_fold_{i}.npy', train.iloc[va].y.to_numpy()+offset)
        result = blend_suites(worker, ids[:2])
        self.assertEqual(result['results'][0]['mean_rmse'], 0.)
        final = worker.finalize_temporal(result['suite_id'])
        out = pd.read_csv(final['submission_path'])
        np.testing.assert_array_equal(out.y, test.x)
        restored = joblib.load(Path(final['submission_path']).with_name('final_model.joblib'))
        np.testing.assert_array_equal(worker._predict(restored, test), test.x)
        folder = self.run/'temporal'/ids[2]
        saved = dict(np.load(folder/'fold_0.npz')); saved['validation'] = saved['validation'][::-1]
        np.savez(folder/'fold_0.npz', **saved)
        with self.assertRaisesRegex(ValueError, 'row indices'):
            blend_suites(worker, [ids[0], ids[2]])

    def test_plateau_counts_new_trials_and_ranks_all_suites(self):
        scores = [50., 49., 48., 47., 47.01, 47.02, 47.03, 47.04]
        for i, score in enumerate(scores): self.report(f'{i+1:016x}', score)
        status = search_status(self.run, self.state)
        self.assertTrue(status['plateau'])
        self.assertEqual(status, search_status(self.run, self.state))
        self.report('e'*16, 46., kind='ensemble')
        self.assertEqual(catalogue(self.run, self.state)['recommended_suite'], 'e'*16)
        status = search_status(self.run, self.state)
        self.assertEqual(status['trials'], 8)
        self.assertTrue(status['continue_search'])

    def test_final_selection_accepts_global_winner_outside_encoding_pair(self):
        for sid, score in [('a'*16, 45.), ('b'*16, 47.), ('c'*16, 43.)]: self.report(sid, score)
        root = self.run/'research'; root.mkdir()
        (root/'state.json').write_text(json.dumps(self.state))
        publisher = FinalOutput(root, {}, self.run)
        publisher._validate_selection = lambda kind, sid: (None, json.loads((self.run/'temporal'/sid/'report.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'global best'):
            publisher.select_final_candidate('temporal', 'a'*16)
        self.assertEqual(publisher.select_final_candidate('temporal', 'c'*16)['suite_id'], 'c'*16)


if __name__ == '__main__':
    unittest.main()
