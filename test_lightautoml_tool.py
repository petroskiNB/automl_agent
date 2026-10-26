"""Mocked LightAutoML boundaries and export checks; no model training or API."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
import lightautoml_research as lama
from research import write_json
from torch_suites import install_suite_engines


class ReferenceRegressor:
    def __init__(self, offset=1.):
        self.offset = offset

    def predict(self, frame):
        return 2 * frame['x'].to_numpy() + 3 + self.offset


class DummyLama:
    calls = []

    def __init__(self, offset=0., **kwargs):
        self.offset, self.kwargs = offset, kwargs
        self.levels = []

    def fit_predict(self, frame, **kwargs):
        self.calls.append((frame.copy(), kwargs, self.kwargs))
        return self.predict(kwargs['valid_data'])

    def predict(self, frame):
        return SimpleNamespace(data=(2 * frame['x'].to_numpy() + 3 + self.offset).reshape(-1, 1))


class LightAutoMLToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.temporal, _ = install_suite_engines()

    def setUp(self):
        scratch = ROOT / 'work' / 'lightautoml_tool_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.run = self.folder / 'agent_runs' / 'test'
        self.root = self.run / 'research'
        train = pd.DataFrame(dict(id=np.arange(20), period=np.repeat(np.arange(1, 6), 4),
            x=np.tile(np.arange(4), 5), cat=np.tile(['a', 'c', 'a', 'c'], 5)))
        train['y'] = 2 * train['x'] + 3
        test = pd.DataFrame(dict(id=[20, 21], period=[6, 6], x=[1, 2], cat=['c', 'z']))
        train.to_csv(self.folder / 'train.csv', index=False)
        test.to_csv(self.folder / 'test.csv', index=False)
        test[['id']].assign(y=0.).to_csv(self.folder / 'sample_submission.csv', index=False)
        self.files = {name: str(self.folder / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
        self.source = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        state = self.source.engine.configure('y', 'regression', 'rmse', 'time', 'period')
        state['experiments'] = [dict(id=1, algorithm='catboost', params={}, feature_code='')]
        write_json(self.root / 'state.json', state)
        def fitted(worker, fit, state, recipe, config, mode, iterations):
            return dict(model=ReferenceRegressor(2. if mode == 'baseline' else 1.), state=state, recipe=recipe,
                history=worker._history(fit, state, config, mode), levels=None)
        self.fake_fit = patch.object(self.temporal.TemporalResearch, '_fit', autospec=True, side_effect=fitted)
        self.fake_fit.start()
        self.addCleanup(self.fake_fit.stop)
        with patch.object(self.source, '_iterations', return_value=4):
            self.reference = self.source.run_temporal_suite(1, 'period', ['x'], ['cat'], [3., 4., 5.])
        self.worker = lama.LightAutoMLResearch(self.root, self.files, self.folder)
        self.payloads, self.offset = [], 0.
        self.env = patch.dict(os.environ, AUTOML_LIGHTAUTOML_BUDGET='360', AUTOML_WORKER_DEADLINE='9999999999')
        self.env.start()
        self.addCleanup(self.env.stop)

    def child(self, worker, directory, payload, seconds):
        self.assertGreater(seconds, 0)
        inner, validation = payload['inner_x'], payload['inner_validation_x']
        self.assertLess(inner['period'].max(), validation['period'].min())
        self.assertFalse(set(inner.index) & set(validation.index))
        self.assertNotIn('y', inner)
        self.assertNotIn('outer_x', payload)
        warmup = inner['period'] == inner['period'].min()
        self.assertTrue(inner.loc[warmup, 'hist_exact_mean'].isna().all())
        self.assertTrue(np.isfinite(validation['hist_exact_mean']).all())
        self.payloads.append(payload)
        return dict(model=DummyLama(self.offset), selected=dict(requested_algorithms=payload['algorithms'],
            fitted_algorithms=['DummyLama'], inner_rmse=abs(self.offset),
            estimator_training_rows=len(inner), inner_validation_rows=len(validation), full_train_refit=False))

    def run_suite(self):
        with patch.object(lama.LightAutoMLResearch, '_child', autospec=True, side_effect=self.child), \
                patch('lightautoml_research.importlib.metadata.version', return_value='0.4.2'):
            return self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120)

    def test_temporal_boundaries_budgeted_final_fit_cached_export_and_reference_fallback(self):
        report = self.run_suite()
        self.assertEqual(report['selected_mode'], 'lightautoml')
        self.assertEqual(len(self.payloads), 4)  # Three folds PLUS final fit in one budget.
        self.assertEqual([float(p['inner_validation_x'].period.max()) for p in self.payloads], [2., 3., 4., 5.])
        self.assertEqual(report['final_pipeline']['estimator_training_periods'], [1., 2., 3., 4.])
        self.assertEqual(report['final_pipeline']['inner_validation_period'], 5.)
        self.assertFalse(report['final_pipeline']['full_train_refit'])
        self.assertEqual(report['results'][1]['mean_rmse'], 0.)
        ledger = (self.run / 'lightautoml_budget.json').read_bytes()
        with patch.object(lama.LightAutoMLResearch, '_child', side_effect=AssertionError('No fit during cache/export')):
            again = self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120)
            final = self.worker.finalize_temporal(report['suite_id'])
        self.assertEqual(again['suite_id'], report['suite_id'])
        self.assertEqual(ledger, (self.run / 'lightautoml_budget.json').read_bytes())
        output = pd.read_csv(final['submission_path'])
        self.assertEqual(output.id.tolist(), [20, 21])
        self.assertEqual(output.y.tolist(), [5., 7.])
        bundle = joblib.load(final['model_path'])
        # Full past history includes year5 even though estimator fitting stops at4.
        self.assertEqual(float(bundle['history'].reference.period.max()), 5.)
        self.assertTrue(np.array_equal(self.worker._predict(bundle, pd.read_csv(self.files['test.csv'])), output.y))
        directory = self.run / 'temporal' / report['suite_id']
        modified = dict(report, selected_mode='reference')
        write_json(directory / 'report.json', modified)
        with patch.object(lama.LightAutoMLResearch, '_child', side_effect=AssertionError('Reference must not fit LAMA')):
            fallback = self.worker.finalize_temporal(report['suite_id'])
        self.assertEqual(pd.read_csv(fallback['submission_path']).y.tolist(), [6., 8.])

    def test_off_budget_and_failed_training_leave_existing_submission(self):
        destination = self.folder / 'submission.csv'
        destination.write_text('keep previous submission', encoding='utf-8')
        with patch.dict(os.environ, AUTOML_LIGHTAUTOML_BUDGET='0'):
            with self.assertRaisesRegex(ValueError, 'Enable LightAutoML'):
                self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120)
        with self.assertRaisesRegex(ValueError, 'distinct LightAutoML'):
            self.worker.run_lightautoml_suite(self.reference['suite_id'], algorithms=['nn'])
        write_json(self.run / 'lightautoml_budget.json', dict(charged_seconds=250, calls=[]))
        with patch('lightautoml_research.importlib.metadata.version', return_value='0.4.2'):
            with self.assertRaisesRegex(ValueError, 'remaining allocation'):
                self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120)
        write_json(self.run / 'lightautoml_budget.json', dict(charged_seconds=0, calls=[]))
        with patch('lightautoml_research.importlib.metadata.version', return_value='0.4.2'), \
                patch.object(lama.LightAutoMLResearch, '_child', side_effect=RuntimeError('dummy LAMA failure')):
            with self.assertRaisesRegex(RuntimeError, 'dummy LAMA failure'):
                self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120)
        self.assertEqual(destination.read_text(encoding='utf-8'), 'keep previous submission')
        self.assertFalse((self.run / '.lightautoml.lock').exists())
        self.assertFalse((self.run / 'lightautoml_summary.json').exists())
        self.assertEqual(json.loads((self.run / 'lightautoml_budget.json').read_text())['calls'][-1]['status'], 'failed')

    def test_paired_refit_mode_uses_all_past_and_exports_without_fit(self):
        def child(worker, directory, payload, seconds):
            inner, validation, outer = payload['inner_x'], payload['inner_validation_x'], payload['outer_x']
            self.assertLess(inner.period.max(), validation.period.min())
            self.assertEqual(set(outer.index), set(inner.index) | set(validation.index))
            self.assertEqual(outer.period.max(), validation.period.max())
            self.assertNotIn('y', outer)
            return dict(model=DummyLama(-2.), selected=dict(full_train_refit=False),
                        refit_model=DummyLama(0.), refit_metadata=dict(training_rows=len(outer), weights_frozen=True))
        with patch.object(lama.LightAutoMLResearch, '_child', autospec=True, side_effect=child), \
                patch('lightautoml_research.importlib.metadata.version', return_value='0.4.2'):
            report = self.worker.run_lightautoml_suite(self.reference['suite_id'], time_limit=120,
                                                     tuning=True, max_tuning_trials=2, refit_full=True)
        self.assertEqual(report['selected_mode'], 'lightautoml_refit')
        self.assertEqual(report['refit_pipeline']['estimator_training_rows'], 20)
        self.assertTrue(report['refit_pipeline']['full_train_refit'])
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        with patch.object(lama.LightAutoMLResearch, '_child', side_effect=AssertionError('No export fit')):
            result = generic.finalize_temporal(report['suite_id'])
            blend = generic.blend_temporal_suites([self.reference['suite_id'], report['suite_id']],
                                                member_modes=[None, 'lightautoml_refit'])
            mixed = generic.finalize_temporal(blend['suite_id'])
        self.assertEqual(pd.read_csv(result['submission_path']).y.tolist(), [5., 7.])
        self.assertEqual(pd.read_csv(mixed['submission_path']).y.tolist(), [5.5, 7.5])

    def test_child_calls_real_api_shape_with_only_inner_rows_and_cpu_whitelist(self):
        inner = pd.DataFrame(dict(x=[0, 1], period=[1, 1]))
        validation = pd.DataFrame(dict(x=[0, 1], period=[2, 2]))
        payload = dict(inner_x=inner, inner_y=np.array([3., 5.]), inner_validation_x=validation,
            inner_validation_y=np.array([3., 5.]), algorithms=['linear_l2', 'lgb', 'cb'], fit_seconds=20)
        request, result = self.folder / 'request.joblib', self.folder / 'result.joblib'
        joblib.dump(payload, request)
        preset = ModuleType('lightautoml.automl.presets.tabular_presets')
        preset.TabularAutoML = DummyLama
        tasks = ModuleType('lightautoml.tasks')
        tasks.Task = lambda *args, **kwargs: ('reg', kwargs)
        DummyLama.calls = []
        with patch.dict(sys.modules, {'lightautoml.automl.presets.tabular_presets': preset, 'lightautoml.tasks': tasks}), \
                patch('lightautoml_research.importlib.metadata.version', return_value='0.4.2'):
            lama.fit_child(request, result)
        train, arguments, parameters = DummyLama.calls[0]
        self.assertEqual(train.period.tolist(), [1, 1])
        self.assertEqual(arguments['valid_data'].period.tolist(), [2, 2])
        self.assertEqual(arguments['roles'], {'target': lama.TARGET_COLUMN})
        self.assertIsNone(parameters['gpu_ids'])
        self.assertEqual(parameters['general_params']['use_algos'], [['linear_l2', 'lgb', 'cb']])
        self.assertFalse(parameters['general_params']['nested_cv'])
        self.assertFalse(parameters['reader_params']['advanced_roles'])
        self.assertEqual(parameters['selection_params']['mode'], 0)
        self.assertFalse(joblib.load(result)['selected']['full_train_refit'])

    def test_generic_dispatch_nonwinning_raw_mode_mixed_ensemble_and_alias_rejection(self):
        self.offset = -2.
        report = self.run_suite()
        self.assertEqual(report['selected_mode'], 'reference')
        self.assertEqual([row['mean_rmse'] for row in report['results']], [1., 2.])
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        directory = self.run / 'temporal' / report['suite_id']
        identity = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        self.assertEqual(generic._identity(identity['config']), identity)
        test = pd.read_csv(self.files['test.csv'])
        ids = [self.reference['suite_id'], report['suite_id']]
        with patch.object(lama.LightAutoMLResearch, '_child', side_effect=AssertionError('No LightAutoML fit during export')):
            default = generic.finalize_temporal(report['suite_id'])
            default_bytes = Path(default['submission_path']).read_bytes()
            source = generic.finalize_temporal(self.reference['suite_id'])
            source_bytes = Path(source['submission_path']).read_bytes()
            self.assertEqual(pd.read_csv(default['submission_path']).y.tolist(), [6., 8.])
            with self.assertRaisesRegex(ValueError, 'Duplicate temporal pipelines'):
                generic.blend_temporal_suites(ids)
            raw = generic.finalize_temporal(report['suite_id'], mode='lightautoml')
            self.assertEqual(pd.read_csv(raw['submission_path']).y.tolist(), [3., 5.])
            raw_bundle = joblib.load(raw['model_path'])
            self.assertTrue(np.array_equal(generic._predict(raw_bundle, test), [3., 5.]))
            blend = generic.blend_temporal_suites(ids, member_modes=[None, 'lightautoml'])
            self.assertEqual(blend['results'][0]['mean_rmse'], .5)
            final = generic.finalize_temporal(blend['suite_id'])
        mixed = pd.read_csv(final['submission_path'])
        self.assertEqual(mixed.y.tolist(), [4.5, 6.5])
        bundle = joblib.load(final['model_path'])
        self.assertTrue(np.array_equal(generic._predict(bundle, test), mixed.y))
        self.assertEqual(Path(default['submission_path']).read_bytes(), default_bytes)
        self.assertEqual(Path(source['submission_path']).read_bytes(), source_bytes)
        self.assertEqual(json.loads((directory / 'report.json').read_text())['selected_mode'], 'reference')


if __name__ == '__main__':
    unittest.main()
