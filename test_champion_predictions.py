"""Four focused checks of prediction bounds and paired encoding; no model fits."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
import joblib
import numpy as np
import pandas as pd
from prediction_policy import bound_predictions, bounded_model
from temporal_research import TemporalResearch
from research import Research
from final_output import FinalOutput
import research_policy as policy


class RawRegressor:
    def __init__(self, offset=0.): self.offset = offset
    def predict(self, x): return x['raw'].to_numpy() + self.offset


class History:
    def transform(self, frame): return pd.DataFrame({'hist_exact_seen': frame['seen'].to_numpy()}, index=frame.index)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(task_type='regression', target='site_eui', metric='rmse', ids=['id'],
                          features=['floor_area', 'heating_degree_days_c', 'cooling_degree_days_c'])

    def test_bounds_are_domain_specific_and_do_not_hide_invalid_values(self):
        raw = np.array([-4., -1., 13.])
        clipped = bound_predictions(raw, self.state)
        np.testing.assert_array_equal(clipped, [0., 0., 13.])
        truth = np.array([0., 1., 10.])
        self.assertLessEqual(np.mean((clipped-truth)**2), np.mean((raw-truth)**2))
        np.testing.assert_array_equal(bound_predictions(raw, dict(self.state, target='signed_target')), raw)
        np.testing.assert_array_equal(bound_predictions(raw, dict(self.state, task_type='classification')), raw)
        invalid = bound_predictions([np.nan, -np.inf, np.inf], self.state)
        self.assertTrue(np.isnan(invalid[0]) and np.isneginf(invalid[1]) and np.isposinf(invalid[2]))

    def test_serialized_temporal_gate_and_final_publisher_preserve_clipping(self):
        scratch = ROOT / 'work' / 'prediction_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder = Path(directory)
            run = folder / 'logs'
            root = run / 'research'
            root.mkdir(parents=True)
            frame = pd.DataFrame(dict(id=[1, 2, 3], raw=[-5., 4., 8.], seen=[1, 0, 1]))
            base = dict(model=bounded_model(RawRegressor(), self.state), state=self.state,
                        recipe=dict(feature_code='', algorithm='catboost'), history=None)
            historical = dict(base, model=bounded_model(RawRegressor(-10.), self.state), history=History())
            path = folder / 'model.joblib'
            joblib.dump(dict(gate_models=[base, historical]), path)
            restored = joblib.load(path)
            worker = TemporalResearch.__new__(TemporalResearch)
            worker._design = lambda frame, *args: (frame, [])
            worker.engine = NS()
            prediction = worker._predict(restored, frame)
            np.testing.assert_array_equal(prediction, [0., 4., 0.])
            truth = np.array([1., 3., 2.])
            score = Research.score(None, self.state, truth, prediction)
            (root / 'state.json').write_text(json.dumps(self.state), encoding='utf-8')
            (run / 'final_selection.json').write_text(json.dumps(dict(kind='temporal', suite_id='a'*16)))
            files = {n: folder / n for n in ('train.csv', 'test.csv', 'sample_submission.csv')}
            frame[['id']].to_csv(files['test.csv'], index=False)
            pd.DataFrame(dict(id=frame.id, site_eui=0.)).to_csv(files['sample_submission.csv'], index=False)
            def export(sid):
                source = run / 'selected.csv'
                pd.DataFrame(dict(id=frame.id, site_eui=worker._predict(restored, frame))).to_csv(source, index=False)
                return dict(submission_path=str(source))
            publisher = FinalOutput(root, files, folder)
            publisher._validate_selection = lambda *args: (NS(finalize_temporal=export), {})
            self.assertTrue(publisher.finalize_selected()['published'])
            actual = pd.read_csv(folder / 'submission.csv').site_eui.to_numpy()
            np.testing.assert_array_equal(actual, prediction)
            self.assertEqual(Research.score(None, self.state, truth, actual), score)

    def test_full_categories_change_only_category_representation(self):
        recipes = policy.library()
        base = next(r for r in recipes if r['reference_id'] == 'regression_231')
        full = next(r for r in recipes if r['reference_id'] == 'regression_231_full_categories')
        self.assertEqual(base['params'], full['params'])
        frame = pd.DataFrame(dict(Year_Factor=[1., 2., 3.], State_Factor=['State_1', 'State_10', 'State_11'],
            building_class=['Residential']*3, facility_type=['Warehouse_Distribution_or_Shipping_center',
            'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage'], floor_area=[10., 20., 30.],
            year_built=[1980., 0., 2000.], energy_star_rating=[50., np.nan, 60.], ELEVATION=[1., 2., 3.],
            heating_degree_days_c=[10., 20., 30.], cooling_degree_days_c=[2., 3., 4.]))
        state = dict(features=list(frame))
        engine = Research.__new__(Research)
        old, _ = engine.prepare(frame, state, base['feature_code'])
        new, categories = engine.prepare(frame, state, full['feature_code'])
        cats = ['State_Factor', 'building_class', 'facility_type']
        self.assertEqual(old.State_Factor.nunique(), 1)
        self.assertEqual(new.State_Factor.nunique(), 3)
        self.assertEqual(new.facility_type.nunique(), 3)
        self.assertEqual(set(categories), set(cats))
        pd.testing.assert_frame_equal(old.drop(columns=cats), new.drop(columns=cats))

    def test_encoding_comparison_rejects_different_keys_and_keeps_fixed_criterion(self):
        base = dict(reference_id='base', task_type='regression', target='site_eui', required_columns=[],
                    algorithm='catboost', params={'depth': 7}, feature_code='legacy')
        full = dict(base, reference_id='full', comparison_of='base', feature_code='strings')
        common = dict(time_column='year', validation_periods=[3, 4, 5], key_columns=['entity'],
                      group_columns=['kind'], history_gate=True, modes=['history_gate'], task={'target': 'site_eui'})
        a = dict(suite_id='old', selected_mode='history_gate', results=[dict(mode='history_gate', mean_rmse=45., latest_rmse=37., worst_rmse=55.)])
        b = dict(suite_id='new', selected_mode='history_gate', results=[dict(mode='history_gate', mean_rmse=44., latest_rmse=38., worst_rmse=54.)])
        reports = [(dict(common, recipe=base), a), (dict(common, recipe=full), b)]
        with patch.object(policy, 'library', return_value=[base, full]):
            result = policy.encoding_comparison(self.state, reports)
            self.assertTrue(result['temporal_complete'])
            self.assertEqual(result['temporal_pair']['recommended_suite'], 'new')
            reports[1][0]['key_columns'] = ['different_entity']
            self.assertFalse(policy.encoding_comparison(self.state, reports)['temporal_complete'])


if __name__ == '__main__':
    unittest.main()
