import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from classification_checks import ClassificationChecks, decision_rule, duplicate_groups
from classification_research import macro
from final_output import FinalOutput
from research import write_json
from test_classification_oof import ClassificationOOFTests


class CheckTests(unittest.TestCase):
    setUp = ClassificationOOFTests.setUp

    def test_pair_subset_omits_harmful_third_model(self):
        p = [np.array([[.9,.1],[.6,.4]]), np.array([[.4,.6],[.1,.9]]), np.array([[0.,1.],[1.,0.]])]
        weights, _ = decision_rule(p, np.array(['a','b']), np.array(['a','b']))
        self.assertEqual(weights[2], 0)
        self.assertEqual(macro(['a','b'], sum(w*x for w,x in zip(weights,p)), ['a','b']), 1)

    def test_reblend_reuses_predictions_export_and_ablation(self):
        state = self.worker.engine.state()
        state['experiments'][1]['feature_code'] = 'def transform(X):\n    X = X.copy()\n    X["aa"] = X["x"] ** 2\n    X["bb"] = X["x"] > 0\n    return X'
        write_json(self.root / 'state.json', state)
        shift = self.worker.adversarial_validation()
        source = self.worker.run_classification_suite([1,2], 3, shift['diagnostic_id'])
        checks = ClassificationChecks(self.root, self.files, self.folder)
        with patch.object(checks, '_fit', side_effect=AssertionError('must reuse saved models')):
            new = checks.reblend_suite(source['suite_id'], [1,2])
        self.assertAlmostEqual(new['results'][0]['mean_f1'], source['results'][0]['mean_f1'])
        self.assertEqual((self.folder / 'submission.csv').read_text(), 'old untouched')
        final = FinalOutput(self.root, self.files, self.folder)
        final._validate_selection('classification_oof', source['suite_id'])
        final._validate_selection('classification_oof', new['suite_id'])
        exported = checks.export_classification(new['suite_id'])
        self.assertEqual(exported['rows'], 20)
        errors = checks.stress_errors(source['suite_id'], [1,2])
        self.assertGreater(errors['rows'], 0)
        result = checks.stress_ablation(source['suite_id'], 2, {'squared':['aa'], 'positive':['bb']})
        self.assertEqual(len(result['results']), 2)
        self.assertEqual((self.folder / 'submission.csv').read_text(), 'old untouched')

    def test_duplicate_groups_do_not_cross_outer_folds(self):
        frame = pd.read_csv(self.files['train.csv'])
        # Repeat all feature rows, preserve unique IDs; group on features only.
        frame.loc[120:, ['x','category','target']] = frame.loc[:119, ['x','category','target']].to_numpy()
        frame.to_csv(self.files['train.csv'], index=False)
        checks = ClassificationChecks(self.root, self.files, self.folder)
        result = checks.grouped_validation([1,2], 3)
        self.assertEqual(result['duplicate_rows_beyond_first'], 120)
        groups = duplicate_groups(frame[['x','category']])
        from pathlib import Path
        directory = Path(result['report_path']).parent
        for i in range(3):
            split = np.load(directory / f'fold_{i}/indices.npz')
            self.assertFalse(set(groups[split['train']]) & set(groups[split['validation']]))
        self.assertEqual((self.folder / 'submission.csv').read_text(), 'old untouched')


if __name__ == '__main__':
    unittest.main()
