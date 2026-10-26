"""Focused synthetic checks for TabM, nested OVR export and causal fuzzy history."""
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
import torch
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'champion_solution'))
from tabm_model import TabMModel
from autogluon_research import model_hyperparameters
from classification_experts import ClassificationExperts, fuse, calibrate
from classification_research import ClassificationResearch
from regression_specialists import RegressionSpecialists, NearestAreaHistory, choose_strength
from research import write_json
from training_contract import validate_arguments
from torch_suites import install_suite_engines
from final_output import FinalOutput


class SmallRegressor:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, frame):
        return 2 * frame['floor_area'].to_numpy() + self.offset


class UpgradeTests(unittest.TestCase):
    def test_tabm_member_loss_probabilities_and_reload(self):
        x = pd.DataFrame(dict(x=np.linspace(-2, 2, 48), constant=1.,
                              category=['a', 'b']*24))
        y = np.asarray(['N', 'Y']*24)
        model = TabMModel('classification', 'f1_macro', dict(iterations=2, width=16,
            layers=1, ensemble_size=2, embedding_dim=4, n_bins=4, batch_size=32))
        model.fit(x.iloc[:32], y[:32], eval_set=(x.iloc[32:], y[32:]), early_stopping_rounds=1)
        tensors = model.preprocessor.transform(x.iloc[32:])
        with torch.inference_mode():
            raw = model.network(*tensors)
            expected = raw.softmax(-1).mean(1).numpy()
        self.assertEqual(tuple(raw.shape), (16, 2, 2))
        np.testing.assert_allclose(model.predict_proba(x.iloc[32:]), expected, rtol=1e-6)
        criterion = torch.nn.CrossEntropyLoss()
        target = torch.tensor([0, 1]*8)
        self.assertTrue(torch.allclose(model._loss(criterion, raw, target),
            torch.stack([criterion(raw[:, k], target) for k in range(2)]).mean()))
        scratch = ROOT/'work'/'upgrade_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            path = Path(temp)/'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(x.iloc[32:]), expected, rtol=1e-6)
        # Extreme query values cannot change fitted train-only bins or vocabulary.
        before = [b.copy() for b in model.preprocessor.tabm_bins]
        model.predict_proba(x.assign(x=1e9, category='unseen'))
        for a, b in zip(before, model.preprocessor.tabm_bins):
            np.testing.assert_array_equal(a, b)

    def test_tabm_numeric_only_regression_and_none_embedding(self):
        x = pd.DataFrame(dict(x=np.linspace(-1, 1, 32)))
        for mode in ('none', 'ple'):
            model = TabMModel('regression', 'rmse', dict(iterations=1, width=8, layers=1,
                ensemble_size=2, embedding_dim=4, n_bins=4, batch_size=32, num_embedding=mode))
            model.fit(x, 2*x.x.to_numpy())
            self.assertEqual(model.predict(x).shape, (32,))
            self.assertTrue(np.isfinite(model.predict(x)).all())
        validate_arguments(dict(algorithm='torch_tabm', params={'ensemble_size': 8, 'num_embedding': 'ple'}))
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            validate_arguments(dict(algorithm='torch_mlp', params={'ensemble_size': 8}))

    def test_fuzzy_history_excludes_current_future_and_query_targets(self):
        history = pd.DataFrame(dict(period=[1, 2, 3], area=[100., 100.5, 101.],
            state=['a']*3, y=[10., 20., 99999.])).rename(columns={'area': 'floor_area'})
        transformer = NearestAreaHistory('period', 'y', ['state'], .01).fit(history)
        query = pd.DataFrame(dict(period=[1, 2, 3, 3], floor_area=[100., 100.1, 100.6, 200.],
                                 state=['a']*4, y=[-999]*4))
        result = transformer.transform(query)
        self.assertEqual(result.fuzzy_seen.tolist(), [0., 1., 1., 0.])
        self.assertEqual(result.fuzzy_mean.iloc[1], 10.)
        self.assertEqual(result.fuzzy_mean.iloc[2], 20.)
        pd.testing.assert_frame_equal(result, transformer.transform(query.assign(y=1e12)))
        self.assertEqual(choose_strength(np.array([1., 2.]), np.array([50., 60.]), np.array([1., 2.])), 0.)

    def test_profiles_and_zero_strength_reference(self):
        self.assertEqual(set(model_hyperparameters('boosting')), {'GBM', 'CAT', 'XGB'})
        self.assertEqual(set(model_hyperparameters('diverse')), {'GBM', 'CAT', 'XGB', 'RF', 'XT', 'NN_TORCH'})
        p = np.array([[.8, .2], [.2, .8]])
        np.testing.assert_allclose(fuse(p, [np.array([.99, .99])], [1], [0.]), p)
        strengths, _ = calibrate(p, [np.array([.99, .99])], [1], ['A', 'B'], ['A', 'B'])
        self.assertEqual(strengths, [0.])

    def test_expert_nested_boundaries_and_no_fit_export(self):
        install_suite_engines()
        scratch = ROOT/'work'/'upgrade_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            folder = Path(temp)
            run, root = folder/'agent_runs'/'test', folder/'agent_runs'/'test'/'research'
            x = np.arange(180)
            train = pd.DataFrame(dict(id=x, x=x, y=np.asarray(['A', 'B', 'C'])[x % 3]))
            test = pd.DataFrame(dict(id=[180, 181, 182], x=[180, 181, 182]))
            train.to_csv(folder/'train.csv', index=False)
            test.to_csv(folder/'test.csv', index=False)
            test[['id']].assign(y='A').to_csv(folder/'sample_submission.csv', index=False)
            files = {n: str(folder/n) for n in ('train.csv', 'test.csv', 'sample_submission.csv')}
            source = ClassificationResearch(root, files, folder)
            state = source.engine.configure('y', 'classification', 'f1_macro')
            state['experiments'] = [dict(id=i, algorithm='catboost', params={'depth': i}, feature_code='') for i in (1, 2)]
            write_json(root/'state.json', state)
            def dummy(frame, state, recipe, seed):
                model = DummyClassifier(strategy='prior').fit(frame[['x']], frame['y'])
                return dict(model=model, classes=model.classes_.tolist(), algorithm='catboost', levels=None, feature_code='')
            with patch.object(source, '_fit', side_effect=dummy):
                baseline = source.run_classification_suite([1, 2], folds=3)
            worker = ClassificationExperts(root, files, folder)
            calls = []
            def experts(frame, state, labels, params, code, seed):
                calls.append(set(frame.index))
                return [None]*len(labels)
            with patch.object(worker, '_fit_experts', side_effect=experts):
                report = worker.run_classification_experts(baseline['suite_id'], class_labels=['A', 'B'], time_limit=120)
            source_dir = run/'classification_oof'/baseline['suite_id']
            for i in range(3):
                with np.load(source_dir/f'fold_{i}/indices.npz') as f:
                    outer, validation = set(f['train']), set(f['validation'])
                with np.load(source_dir/f'fold_{i}/calibration_indices.npz') as f:
                    inner, calibration = set(f['train']), set(f['validation'])
                self.assertEqual(calls[2*i], inner)
                self.assertEqual(calls[2*i+1], outer)
                self.assertFalse((calls[2*i] | calls[2*i+1]) & validation)
                self.assertFalse(calls[2*i] & calibration)
            final = FinalOutput(root, files, folder)
            final.select_final_candidate('classification_oof', report['suite_id'])
            with patch.object(ClassificationExperts, '_fit_experts', side_effect=AssertionError('Export must never fit')):
                result = final.finalize_selected()
            self.assertTrue(result['published'])
            self.assertEqual(pd.read_csv(folder/'submission.csv').id.tolist(), [180, 181, 182])

    def test_specialist_frozen_years_full_past_and_saved_export(self):
        _, temporal, _ = install_suite_engines()
        scratch = ROOT/'work'/'upgrade_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            folder = Path(temp)
            root = folder/'agent_runs'/'test'/'research'
            train = pd.DataFrame(dict(id=np.arange(40), period=np.repeat(np.arange(1, 6), 8),
                floor_area=np.tile(np.arange(100., 108.), 5), State_Factor='S', facility_type='Office',
                building_class='Commercial', year_built=1950.))
            train['site_eui'] = 2*train.floor_area+1
            test = train.iloc[:2].drop(columns=['site_eui']).assign(id=[40, 41], period=6)
            train.to_csv(folder/'train.csv', index=False)
            test.to_csv(folder/'test.csv', index=False)
            test[['id']].assign(site_eui=0.).to_csv(folder/'sample_submission.csv', index=False)
            files = {n: str(folder/n) for n in ('train.csv', 'test.csv', 'sample_submission.csv')}
            source = temporal.TemporalResearch(root, files, folder)
            state = source.engine.configure('site_eui', 'regression', 'rmse', 'time', 'period')
            state['experiments'] = [dict(id=1, algorithm='catboost', params={}, feature_code='')]
            write_json(root/'state.json', state)
            def original_fit(worker, frame, state, recipe, config, mode, iterations):
                return dict(model=SmallRegressor(2. if mode == 'baseline' else 1.), state=state,
                    recipe=recipe, levels=None, history=worker._history(frame, state, config, mode))
            main_calls, tail_calls = [], []
            def main_fit(worker, frame, config, iterations, fuzzy=False):
                main_calls.append((frame.period.max(), set(frame.index), fuzzy))
                return dict(model=SmallRegressor(.5 if fuzzy else 1.), state=config['task'], levels=None,
                    recipe=config['recipe'], history=worker._combined_history(frame, config['task'], config)
                    if fuzzy else worker._history(frame, config['task'], config, config['history_mode']))
            def tail_fit(worker, frame, config):
                tail_calls.append(set(frame.index))
                return dict(gate=None, expert=None, threshold=float(frame.site_eui.quantile(.95)), high_rows=2)
            with patch.object(temporal.TemporalResearch, '_fit', autospec=True, side_effect=original_fit):
                with patch.object(source, '_iterations', return_value=4):
                    baseline = source.run_temporal_suite(1, 'period', ['floor_area'], ['facility_type'], [3., 4., 5.])
                worker = RegressionSpecialists(root, files, folder)
                with patch.object(RegressionSpecialists, '_fit_main', autospec=True, side_effect=main_fit), \
                     patch.object(RegressionSpecialists, '_fit_tail', autospec=True, side_effect=tail_fit):
                    report = worker.run_regression_specialists(baseline['suite_id'], time_limit=120)
            self.assertEqual({r['mode'] for r in report['results']}, {'reference', 'fuzzy_history', 'tail_expert', 'tail_fuzzy'})
            for i, period in enumerate((3, 4, 5)):
                self.assertEqual(tail_calls[2*i], set(train.index[train.period < period-1]))
                self.assertEqual(tail_calls[2*i+1], set(train.index[train.period < period]))
            self.assertEqual(tail_calls[-1], set(train.index))
            generic = temporal.TemporalResearch(root, files, folder)
            with patch.object(RegressionSpecialists, '_fit_main', side_effect=AssertionError('Export must not fit')):
                exported = generic.finalize_temporal(report['suite_id'], mode='fuzzy_history')
            np.testing.assert_allclose(pd.read_csv(exported['submission_path']).site_eui, 2*test.floor_area+.5)
            package = joblib.load(exported['model_path'])
            np.testing.assert_allclose(generic._predict(package, test), 2*test.floor_area+.5)


if __name__ == '__main__':
    unittest.main()
