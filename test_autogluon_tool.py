"""Small boundary/export tests with fake AutoGluon predictors; no API or ML search."""
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
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
from autogluon_research import AutoGluonResearch, normalise, recover_completed_suite
from classification_research import ClassificationResearch
from final_output import FinalOutput
from research import write_json


class FakePredictor:
    def predict_proba(self, frame):
        indices = frame['x'].astype(int).to_numpy() % 3
        return pd.DataFrame((np.eye(3)[indices] * .97 + .01)[:, [2, 0, 1]],
                            columns=['C', 'A', 'B'], index=frame.index)

    def model_names(self):
        return ['LightGBM_BAG_L1', 'CatBoost_BAG_L1', 'XGBoost_BAG_L1', 'WeightedEnsemble_L2']


class AutoGluonToolTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'work' / 'autogluon_tool_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.run = self.folder / 'agent_runs' / 'test'
        self.root = self.run / 'research'
        x = np.arange(180)
        train = pd.DataFrame(dict(id=x, x=x, y=np.asarray(['A', 'B', 'C'])[x % 3]))
        test = pd.DataFrame(dict(id=[180, 181, 182], x=[180, 181, 182]))
        train.to_csv(self.folder / 'train.csv', index=False)
        test.to_csv(self.folder / 'test.csv', index=False)
        test[['id']].assign(y='A').to_csv(self.folder / 'sample_submission.csv', index=False)
        self.files = {n: str(self.folder / n) for n in ('train.csv', 'test.csv', 'sample_submission.csv')}
        source = ClassificationResearch(self.root, self.files, self.folder)
        state = source.engine.configure('y', 'classification', 'f1_macro')
        state['experiments'] = [dict(id=i, algorithm='catboost', params={'depth': i}, feature_code='') for i in (1, 2)]
        write_json(self.root / 'state.json', state)
        def dummy(frame, state, recipe, seed):
            model = DummyClassifier(strategy='prior').fit(frame[['x']], frame['y'])
            return dict(model=model, classes=model.classes_.tolist(), algorithm='catboost', levels=None, feature_code='')
        with patch.object(source, '_fit', side_effect=dummy):
            self.source_report = source.run_classification_suite([1, 2], folds=3)
        self.worker = AutoGluonResearch(self.root, self.files, self.folder)
        self.calls = []
        self.env = patch.dict(os.environ, AUTOML_AUTOGLUON_BUDGET='360', AUTOML_WORKER_DEADLINE='9999999999')
        self.env.start()
        self.addCleanup(self.env.stop)

    def fit(self, frame, state, path, code, seconds, bag_folds, stack_levels, model_profile='boosting'):
        self.calls.append(set(frame.index))
        path.mkdir(parents=True, exist_ok=True)
        # Native predictor-file structure without invoking AutoGluon training.
        for name in ('predictor.pkl', 'learner.pkl', 'models/trainer.pkl'):
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'fake native artifact')
        for name in FakePredictor().model_names():
            target = path / 'models' / name / 'model.pkl'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'fake native model')
        return FakePredictor()

    def test_recovery_rebuilds_complete_checkpoints_without_fit_or_identity_migration(self):
        with patch.object(self.worker, '_fit_auto', side_effect=self.fit):
            first = self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        directory = self.run / 'autogluon_oof' / first['suite_id']
        original_plan = (directory / 'plan.json').read_bytes()
        budget = json.loads((self.run / 'autogluon_budget.json').read_text(encoding='utf-8'))
        budget['calls'][-1]['status'] = 'running'
        charged = budget['charged_seconds']
        write_json(self.run / 'autogluon_budget.json', budget)
        (directory / 'report.json').unlink()
        (directory / 'oof.npz').unlink()
        (self.run / 'autogluon_summary.json').unlink()
        with patch.object(self.worker, '_fit_auto', side_effect=AssertionError('Recovery must never train')):
            recovered = recover_completed_suite(self.worker, first['suite_id'])
        self.assertEqual(recovered['results'], first['results'])
        self.assertEqual((directory / 'plan.json').read_bytes(), original_plan)
        audit = json.loads((directory / 'recovery.json').read_text(encoding='utf-8'))
        self.assertFalse(audit['training_performed'])
        self.assertTrue(any(key.endswith('models/trainer.pkl') for key in audit['artifacts_sha256']))
        after = json.loads((self.run / 'autogluon_budget.json').read_text(encoding='utf-8'))
        self.assertEqual(after['charged_seconds'], charged)
        self.assertEqual(after['calls'][-1]['status'], 'completed_recovered')
        with patch('autogluon.tabular.TabularPredictor.load', return_value=FakePredictor()), \
             patch.object(self.worker, '_fit_auto', side_effect=AssertionError('Export must never train')):
            exported = self.worker.export_autogluon(first['suite_id'])
        self.assertEqual(pd.read_csv(exported['submission_path'])['y'].tolist(), ['A', 'B', 'C'])

    def test_recovery_rejects_stale_identity_missing_native_files_and_changed_scores(self):
        with patch.object(self.worker, '_fit_auto', side_effect=self.fit):
            first = self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        directory = self.run / 'autogluon_oof' / first['suite_id']
        (directory / 'report.json').unlink()
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        changed = json.loads(json.dumps(plan))
        changed['hashes']['autogluon_research.py'] = '0' * 64
        write_json(directory / 'plan.json', changed)
        with self.assertRaisesRegex(ValueError, 'exact original runtime'):
            recover_completed_suite(self.worker, first['suite_id'])
        write_json(directory / 'plan.json', plan)
        fold = directory / 'fold_0'
        pipe = joblib.load(fold / 'pipeline.joblib')
        native = fold / pipe['auto_path'] / 'predictor.pkl'
        saved_bytes = native.read_bytes()
        native.unlink()
        with self.assertRaisesRegex(ValueError, 'predictor artifact'):
            recover_completed_suite(self.worker, first['suite_id'])
        native.write_bytes(saved_bytes)
        scores = json.loads((fold / 'scores.json').read_text(encoding='utf-8'))
        scores['scores']['autogluon'] -= .01
        write_json(fold / 'scores.json', scores)
        with self.assertRaisesRegex(ValueError, 'fold score differs'):
            recover_completed_suite(self.worker, first['suite_id'])
        self.assertFalse((directory / 'report.json').exists())

    def test_wall_deadline_is_converted_to_monotonic_before_fitting(self):
        captured = []
        def run(directory, config, source, report, worker, end):
            captured.append(end)
            return dict(selected_candidate='reference', results=[], folds=[])
        with patch.dict(os.environ, AUTOML_WORKER_DEADLINE='1700000045'), \
             patch('autogluon_research.time.time', return_value=1700000000.), \
             patch('autogluon_research.time.monotonic', return_value=1000.), \
             patch.object(self.worker, '_run', side_effect=run):
            self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(captured, [1030.])

    def test_nested_boundaries_alignment_cache_and_final_export(self):
        with patch.object(self.worker, '_fit_auto', side_effect=self.fit):
            report = self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(len(self.calls), 6)
        source = self.run / 'classification_oof' / self.source_report['suite_id']
        for i in range(3):
            with np.load(source / f'fold_{i}/indices.npz') as f:
                outer_train, outer_valid = set(f['train']), set(f['validation'])
            with np.load(source / f'fold_{i}/calibration_indices.npz') as f:
                inner_train, calibration = set(f['train']), set(f['validation'])
            self.assertEqual(self.calls[2*i], inner_train)
            self.assertEqual(self.calls[2*i+1], outer_train)
            self.assertFalse(self.calls[2*i] & calibration)
            self.assertFalse((self.calls[2*i] | self.calls[2*i+1]) & outer_valid)
        self.assertEqual(report['selected_candidate'], 'autogluon')
        before = (self.run / 'autogluon_budget.json').read_bytes()
        with patch.object(self.worker, '_fit_auto', side_effect=AssertionError('Cache must not fit again')):
            cached = self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(cached['suite_id'], report['suite_id'])
        self.assertEqual(before, (self.run / 'autogluon_budget.json').read_bytes())
        final = FinalOutput(self.root, self.files, self.folder)
        final.select_final_candidate('classification_oof', self.source_report['suite_id'])
        recovered = final.select_best_classification()
        self.assertTrue(recovered['selected'])
        self.assertEqual(recovered['selection']['kind'], 'autogluon_oof')
        self.assertEqual(recovered['selection']['suite_id'], report['suite_id'])
        self.assertEqual(len(recovered['candidates']), 2)
        with patch('autogluon.tabular.TabularPredictor.load', return_value=FakePredictor()):
            result = final.finalize_selected()
            self.assertTrue(result['published'])
            output = pd.read_csv(self.folder / 'submission.csv')
            self.assertEqual(output['id'].tolist(), [180, 181, 182])
            self.assertEqual(output['y'].tolist(), ['A', 'B', 'C'])
            package = joblib.load(result['model_path'])
            self.assertEqual(self.worker.predict_package(package, pd.read_csv(self.files['test.csv'])).shape, (3, 3))

    def test_failure_and_budget_do_not_replace_previous_submission(self):
        path = self.folder / 'submission.csv'
        path.write_text('previous submission', encoding='utf-8')
        with patch.object(self.worker, '_fit_auto', side_effect=RuntimeError('simulated fit failure')):
            with self.assertRaises(RuntimeError):
                self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        self.assertEqual(path.read_text(encoding='utf-8'), 'previous submission')
        self.assertFalse((self.run / '.autogluon.lock').exists())
        self.assertFalse((self.run / 'autogluon_summary.json').exists())
        write_json(self.run / 'autogluon_budget.json', dict(charged_seconds=250, calls=[]))
        with self.assertRaisesRegex(ValueError, 'remaining allocation'):
            self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)

    def test_different_outer_rows_are_not_compared_automatically(self):
        final = FinalOutput(self.root, self.files, self.folder)
        other_id = 'f' * 16
        other = self.run / 'classification_oof' / other_id
        other.mkdir()
        write_json(other / 'report.json', self.source_report)
        write_json(other / 'plan.json', dict(config=dict(folds=3)))
        source = self.run / 'classification_oof' / self.source_report['suite_id']
        # A rotation preserves full OOF coverage while changing the ordered folds.
        for i in range(3):
            dest = other / f'fold_{i}'
            dest.mkdir()
            with np.load(source / f'fold_{(i+1)%3}/indices.npz') as split:
                np.savez(dest / 'indices.npz', train=split['train'], validation=split['validation'])
        original = final._validate_selection
        def validate(kind, sid):
            return (None, self.source_report) if sid == other_id else original(kind, sid)
        with patch.object(final, '_validate_selection', side_effect=validate):
            result = final.select_best_classification()
        self.assertFalse(result['selected'])
        self.assertFalse((self.run / 'final_selection.json').exists())

    def test_rejects_outer_rows_in_calibration_and_invalid_scores(self):
        source = self.run / 'classification_oof' / self.source_report['suite_id'] / 'fold_0'
        with np.load(source / 'indices.npz') as f:
            outer_valid = f['validation'].copy()
        with np.load(source / 'calibration_indices.npz') as f:
            inner, calibration = f['train'].copy(), f['validation'].copy()
        inner[0] = outer_valid[0]
        np.savez(source / 'calibration_indices.npz', train=inner, validation=calibration)
        with patch.object(self.worker, '_fit_auto', side_effect=AssertionError('Must reject before fitting')):
            with self.assertRaisesRegex(ValueError, 'boundaries'):
                self.worker.run_autogluon_suite(self.source_report['suite_id'], time_limit=120)
        for invalid in (np.array([[0., 0.]]), np.array([[np.nan, 1.]]), np.array([[-1., 2.]])):
            with self.assertRaises(ValueError): normalise(invalid)


if __name__ == '__main__':
    unittest.main()
