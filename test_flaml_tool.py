"""Tiny temporal boundaries, budget and export checks; no search or API calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
from flaml_research import FlamlResearch, categorical_design
from research import write_json
from torch_suites import install_suite_engines


class FixedRegressor:
    def __init__(self, offset=0.):
        self.offset = offset

    def predict(self, frame):
        return 2 * frame['x'].to_numpy() + 3 + self.offset


class FlamlToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.temporal, _ = install_suite_engines()

    def setUp(self):
        scratch = ROOT / 'work' / 'flaml_tool_tests'
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
            return dict(model=FixedRegressor(2. if mode == 'baseline' else 1.),
                state=state, recipe=recipe, history=worker._history(fit, state, config, mode), levels=None)
        self.fake_fit = patch.object(self.temporal.TemporalResearch, '_fit', autospec=True, side_effect=fitted)
        self.fake_fit.start()
        self.addCleanup(self.fake_fit.stop)
        with patch.object(self.source, '_iterations', return_value=4):
            self.source_report = self.source.run_temporal_suite(1, 'period', ['x'], ['cat'], [3., 4., 5.])
        self.assertEqual(self.source_report['selected_mode'], 'history_exact')
        self.worker = FlamlResearch(self.root, self.files, self.folder)
        self.payloads = []
        self.child_offset = 0.
        self.env = patch.dict(os.environ, AUTOML_FLAML_BUDGET='360', AUTOML_WORKER_DEADLINE='9999999999')
        self.env.start()
        self.addCleanup(self.env.stop)

    def child(self, worker, directory, payload, seconds):
        self.assertGreater(seconds, 0)
        self.payloads.append(payload)
        if 'selected' not in payload:
            inner, validation, outer = payload['inner_x'], payload['inner_validation_x'], payload['outer_x']
            self.assertLess(inner['period'].max(), validation['period'].min())
            self.assertEqual(validation['period'].max(), outer['period'].max())
            self.assertFalse(set(inner.index) & set(validation.index))
            self.assertEqual(set(inner.index) | set(validation.index), set(outer.index))
            self.assertNotIn('y', outer)
            # Training history cannot see same-period peers or the future inner holdout.
            warmup = inner['period'] == inner['period'].min()
            self.assertTrue(inner.loc[warmup, 'hist_exact_mean'].isna().all())
            self.assertTrue(np.isfinite(validation['hist_exact_mean']).all())
            Path(payload['logfile']).write_text('dummy trial log', encoding='utf-8')
        return dict(model=FixedRegressor(self.child_offset), selected=payload.get('selected',
                    dict(record_id=0, learner='extra_tree', config={'n_estimators': 4}, inner_rmse=0.)))

    def run_suite(self):
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child), \
                patch('flaml_research.importlib.metadata.version', return_value='2.3.6'):
            return self.worker.run_flaml_suite(self.source_report['suite_id'], time_limit=120)

    def test_temporal_boundaries_categorical_mapping_and_cache(self):
        report = self.run_suite()
        self.assertEqual(report['selected_mode'], 'flaml')
        self.assertEqual(len(self.payloads), 3)
        self.assertEqual(report['results'][1]['mean_rmse'], 0.)
        ledger = (self.run / 'flaml_budget.json').read_bytes()
        with patch.object(FlamlResearch, '_child', side_effect=AssertionError('Cached suite must not fit')):
            again = self.worker.run_flaml_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(again['suite_id'], report['suite_id'])
        self.assertEqual(ledger, (self.run / 'flaml_budget.json').read_bytes())
        _, levels = categorical_design(pd.DataFrame(dict(cat=['a', 'c'])))
        batch, _ = categorical_design(pd.DataFrame(dict(cat=['c', 'z'])), levels)
        singleton, _ = categorical_design(pd.DataFrame(dict(cat=['c'])), levels)
        self.assertEqual(batch.cat.cat.codes.tolist(), [1, -1])
        self.assertEqual(singleton.cat.cat.codes.tolist(), [1])

    def test_off_exhausted_budget_and_failure_keep_existing_output(self):
        destination = self.folder / 'submission.csv'
        destination.write_text('keep previous submission', encoding='utf-8')
        with patch.dict(os.environ, AUTOML_FLAML_BUDGET='0'):
            with self.assertRaisesRegex(ValueError, 'Enable FLAML'):
                self.worker.run_flaml_suite(self.source_report['suite_id'], time_limit=120)
        write_json(self.run / 'flaml_budget.json', dict(charged_seconds=250, calls=[]))
        with patch('flaml_research.importlib.metadata.version', return_value='2.3.6'):
            with self.assertRaisesRegex(ValueError, 'remaining allocation'):
                self.worker.run_flaml_suite(self.source_report['suite_id'], time_limit=120)
        write_json(self.run / 'flaml_budget.json', dict(charged_seconds=0, calls=[]))
        with patch('flaml_research.importlib.metadata.version', return_value='2.3.6'), \
                patch.object(FlamlResearch, '_child', side_effect=RuntimeError('dummy training failure')):
            with self.assertRaisesRegex(RuntimeError, 'dummy training failure'):
                self.worker.run_flaml_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(destination.read_text(encoding='utf-8'), 'keep previous submission')
        self.assertFalse((self.run / '.flaml.lock').exists())
        self.assertFalse((self.run / 'flaml_summary.json').exists())
        self.assertEqual(len(list((self.run / 'temporal').glob('*/report.json'))), 1)
        self.assertEqual(json.loads((self.run / 'flaml_budget.json').read_text())['calls'][-1]['status'], 'failed')

    def test_generic_dispatch_export_reload_and_mixed_temporal_ensemble(self):
        report = self.run_suite()
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        directory = self.run / 'temporal' / report['suite_id']
        identity = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        self.assertEqual(generic._identity(identity['config']), identity)
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            final = generic.finalize_temporal(report['suite_id'])
        output = pd.read_csv(final['submission_path'])
        self.assertEqual(output.id.tolist(), [20, 21])
        self.assertEqual(output.y.tolist(), [5., 7.])
        bundle = joblib.load(directory / 'final_model.joblib')
        self.assertTrue(np.array_equal(generic._predict(bundle, pd.read_csv(self.files['test.csv'])), output.y))
        blend = generic.blend_temporal_suites([self.source_report['suite_id'], report['suite_id']])
        self.assertAlmostEqual(blend['results'][0]['mean_rmse'], .5)
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            result = generic.finalize_temporal(blend['suite_id'])
        mixed = pd.read_csv(result['submission_path'])
        self.assertEqual(mixed.y.tolist(), [5.5, 7.5])
        blend_bundle = joblib.load(self.run / 'temporal' / blend['suite_id'] / 'final_model.joblib')
        self.assertTrue(np.array_equal(generic._predict(blend_bundle, pd.read_csv(self.files['test.csv'])), mixed.y))
        from final_output import FinalOutput
        publisher = FinalOutput(self.root, self.files, self.folder)
        publisher.select_final_candidate('temporal', report['suite_id'])
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            published = publisher.finalize_selected()
        self.assertTrue(published['published'])
        self.assertEqual(pd.read_csv(self.folder / 'submission.csv').y.tolist(), [5., 7.])

    def test_nonwinning_flaml_mode_can_improve_blend_and_exports_its_exact_recipe(self):
        self.child_offset = -2.
        report = self.run_suite()
        self.assertEqual(report['selected_mode'], 'reference')
        self.assertEqual([row['mean_rmse'] for row in report['results']], [1., 2.])
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        ids = [self.source_report['suite_id'], report['suite_id']]
        with self.assertRaisesRegex(ValueError, 'Duplicate temporal pipelines'):
            generic.blend_temporal_suites(ids)
        blend = generic.blend_temporal_suites(ids, member_modes=[None, 'flaml'])
        self.assertEqual(blend['results'][0]['mean_rmse'], .5)
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            previous = generic.finalize_temporal(report['suite_id'])
            previous_bytes = Path(previous['submission_path']).read_bytes()
            final = generic.finalize_temporal(blend['suite_id'])
        mixed = pd.read_csv(final['submission_path'])
        self.assertEqual(mixed.y.tolist(), [4.5, 6.5])
        bundle = joblib.load(final['model_path'])
        self.assertTrue(np.array_equal(generic._predict(bundle, pd.read_csv(self.files['test.csv'])), mixed.y))
        self.assertEqual(Path(previous['submission_path']).read_bytes(), previous_bytes)
        flaml_dir = self.run / 'temporal' / report['suite_id']
        raw = json.loads((flaml_dir / 'finalized__flaml.json').read_text())
        self.assertEqual(raw['mode'], 'flaml')
        self.assertEqual(raw['iterations']['record_id'], 0)
        self.assertEqual(self.payloads[-1]['selected']['record_id'], 0)
        self.assertEqual(json.loads((flaml_dir / 'report.json').read_text())['selected_mode'], 'reference')

    def test_explicit_nonwinning_history_mode_is_distinct_and_rejects_corrupt_predictions(self):
        self.child_offset = -2.
        report = self.run_suite()
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        ids = [self.source_report['suite_id'], report['suite_id']]
        selected = generic.finalize_temporal(ids[0])
        previous_bytes = Path(selected['submission_path']).read_bytes()
        blend = generic.blend_temporal_suites(ids, member_modes=['baseline', 'flaml'])
        self.assertEqual(blend['results'][0]['mean_rmse'], 0.)
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            final = generic.finalize_temporal(blend['suite_id'])
        self.assertEqual(pd.read_csv(final['submission_path']).y.tolist(), [5., 7.])
        self.assertEqual(Path(selected['submission_path']).read_bytes(), previous_bytes)
        source = self.run / 'temporal' / ids[0]
        self.assertEqual(json.loads((source / 'report.json').read_text())['selected_mode'], 'history_exact')
        np.save(source / 'baseline_fold_0.npy', np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'reported metrics'):
            generic.blend_temporal_suites(ids, member_modes=['baseline', 'flaml'])
        with self.assertRaisesRegex(ValueError, 'present in'):
            generic.finalize_temporal(ids[0], mode='made_up')
        with self.assertRaisesRegex(ValueError, 'align'):
            generic.blend_temporal_suites(ids, member_modes=['baseline'])

    def test_four_single_sources_include_nonwinning_flaml_without_nested_ensembles(self):
        self.child_offset = -2.
        report = self.run_suite()
        state = self.source.engine.state()
        state['experiments'] += [dict(id=i, algorithm='catboost', params={}, feature_code='') for i in (2, 3)]
        write_json(self.root / 'state.json', state)
        ids = [self.source_report['suite_id']]
        with patch.object(self.source, '_iterations', return_value=4):
            for experiment in (2, 3):
                ids.append(self.source.run_temporal_suite(experiment, 'period', ['x'], ['cat'],
                    [3., 4., 5.])['suite_id'])
        ids.append(report['suite_id'])
        generic = self.temporal.TemporalResearch(self.root, self.files, self.folder)
        blend = generic.blend_temporal_suites(ids, member_modes=[None, None, None, 'flaml'])
        self.assertEqual(blend['results'][0]['mean_rmse'], .25)
        with patch.object(FlamlResearch, '_child', autospec=True, side_effect=self.child):
            final = generic.finalize_temporal(blend['suite_id'])
        self.assertEqual(pd.read_csv(final['submission_path']).y.tolist(), [5.25, 7.25])
        with self.assertRaisesRegex(ValueError, 'Nested'):
            generic.blend_temporal_suites([blend['suite_id'], ids[0]])


if __name__ == '__main__':
    unittest.main()
