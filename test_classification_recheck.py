"""Small mocked OOF checks: frozen boundaries and new measurements of old recipes."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
from classification_research import ClassificationResearch
from classification_fold_plan import freeze_plan, load_plan
from final_output import FinalOutput
from research import write_json


def dummy_fit(frame, state, recipe, seed):
    model = DummyClassifier(strategy='prior').fit(frame[['x']], frame[state['target']])
    return dict(model=model, classes=model.classes_.tolist(), algorithm=recipe['algorithm'],
                levels=None, feature_code=recipe.get('feature_code', ''))


class ClassificationRecheckTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'work' / 'classification_recheck_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        old, new = base / 'old', base / 'new'
        old.mkdir()
        new.mkdir()
        x = np.arange(180)
        train = pd.DataFrame(dict(id=x, x=x, y=np.asarray(['A', 'B', 'C'])[x % 3]))
        test = pd.DataFrame(dict(id=[180, 181, 182], x=[180, 181, 182]))
        train.to_csv(old / 'train.csv', index=False)
        test.to_csv(old / 'test.csv', index=False)
        test[['id']].assign(y='A').to_csv(old / 'sample_submission.csv', index=False)
        for name in ('train.csv', 'test.csv', 'sample_submission.csv'):
            shutil.copy2(old / name, new / name)
        old_files = {name: str(old / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
        self.files = {name: str(new / name) for name in old_files}
        old_run = old / 'agent_runs' / 'old'
        source = ClassificationResearch(old_run / 'research', old_files, old)
        state = source.engine.configure('y', 'classification', 'f1_macro')
        state['experiments'] = [dict(id=i, algorithm='catboost', params={'depth': i}, feature_code='') for i in (1, 2)]
        state['experiments'].append(dict(id=5, algorithm='xgboost', params={'depth': 4}, feature_code='',
                                        metric='f1_macro', score=.99))
        write_json(source.engine.root / 'state.json', state)
        with patch.object(source, '_fit', side_effect=dummy_fit):
            original = source.run_classification_suite([1, 2], folds=3)
        self.folder, self.run = new, new / 'agent_runs' / 'new'
        self.worker = ClassificationResearch(self.run / 'research', self.files, new)
        current = self.worker.engine.configure('y', 'classification', 'f1_macro')
        self.plan = freeze_plan(old_run, original['suite_id'], self.run, current,
                                files=self.files, recipe_ids=[5])

    def test_saved_recipe_is_measured_on_exact_folds_and_exports_without_fit(self):
        calls = []
        def fitted(frame, state, recipe, seed):
            calls.append((set(frame.index), recipe['id'], seed))
            return dummy_fit(frame, state, recipe, seed)
        before = (self.worker.engine.root / 'state.json').read_bytes()
        with patch.object(self.worker, '_fit', side_effect=fitted):
            report = self.worker.run_saved_recipe_oof(self.plan['split_plan_id'], [5])
        _, splits = load_plan(self.run, self.plan['split_plan_id'], self.worker._state(), self.files)
        self.assertEqual(len(calls), 6)
        for i, (tr, va, inner, cal) in enumerate(splits):
            self.assertEqual(calls[2*i], (set(inner), 5, 200+i))
            self.assertEqual(calls[2*i+1], (set(tr), 5, 300+i))
            self.assertFalse(calls[2*i][0] & set(cal))
            self.assertFalse((calls[2*i][0] | calls[2*i+1][0]) & set(va))
        self.assertEqual(before, (self.worker.engine.root / 'state.json').read_bytes())
        self.assertFalse(report['historical_scores_imported'])
        self.assertFalse(report['historical_models_imported'])
        self.assertEqual(report['recipe_origin'], 'historical_specification')
        self.assertTrue(all(row['mean_f1'] < .99 for row in report['results']))
        directory = self.run / 'classification_oof' / report['suite_id']
        config = json.loads((directory / 'plan.json').read_text())['config']
        self.assertEqual(config['recipes'][0]['id'], 5)
        self.assertNotIn('historical_holdout_score', config['recipes'][0])
        self.assertTrue((directory / 'fold_0/calibration_indices.npz').is_file())
        with patch.object(self.worker, '_fit', side_effect=AssertionError('Cached suite must not train')):
            cached = self.worker.run_saved_recipe_oof(self.plan['split_plan_id'], [5])
        self.assertEqual(cached['suite_id'], report['suite_id'])
        final = FinalOutput(self.worker.engine.root, self.files, self.folder)
        final.select_final_candidate('classification_oof', report['suite_id'])
        with patch.object(ClassificationResearch, '_fit', side_effect=AssertionError('Export must not train')):
            exported = final.finalize_selected()
        self.assertTrue(exported['published'])
        self.assertEqual(pd.read_csv(self.folder / 'submission.csv').id.tolist(), [180, 181, 182])

    def test_current_single_recipe_requires_frozen_plan_and_keeps_registry(self):
        state = self.worker.engine.state()
        state['experiments'] = [dict(id=9, algorithm='catboost', params={'depth': 2}, feature_code='')]
        write_json(self.worker.engine.root / 'state.json', state)
        with self.assertRaisesRegex(ValueError, '2..3'):
            self.worker.run_classification_suite([9])
        with patch.object(self.worker, '_fit', side_effect=dummy_fit):
            report = self.worker.run_classification_suite([9], split_plan_id=self.plan['split_plan_id'])
        self.assertEqual(report['recipe_origin'], 'current_experiments')
        self.assertEqual(self.worker.engine.state()['experiments'], state['experiments'])
        with self.assertRaisesRegex(ValueError, 'Fold count'):
            self.worker.run_classification_suite([9], folds=4, split_plan_id=self.plan['split_plan_id'])

    def test_changed_split_or_data_is_rejected_before_any_fit(self):
        directory = self.run / 'classification_fold_plans' / self.plan['split_plan_id']
        path = directory / 'calibration_0.npz'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with patch.object(self.worker, '_fit', side_effect=AssertionError('Invalid plan must not fit')):
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                self.worker.run_saved_recipe_oof(self.plan['split_plan_id'], [5])


if __name__ == '__main__':
    unittest.main()
