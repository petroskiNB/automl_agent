"""Small leakage, weighted-loss and saved preprocessing checks; no real data/API."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'champion_solution'))
from tabm_model import QuantileTabMPreprocessor, TabMModel


class TabMF1Tests(unittest.TestCase):
    def test_quantiles_are_train_only_and_missing_channels_are_informative(self):
        x = pd.DataFrame(dict(heavy=np.geomspace(1, 1e8, 32), clean=np.arange(32.),
            partial=np.where(np.arange(32) % 5 == 0, np.nan, np.arange(32.)),
            constant=1., empty=np.nan, category=['A', 'B']*16))
        prep = QuantileTabMPreprocessor().fit(x)
        quantiles = prep.quantile_transformer.quantiles_.copy()
        transformed, _ = prep.transform(x)
        self.assertEqual(prep.missing_indicators, ['partial'])
        self.assertEqual(transformed.shape, (32, 6))
        self.assertTrue(np.isfinite(transformed.numpy()).all())
        self.assertEqual(transformed[:, 3].abs().sum().item(), 0.)
        self.assertEqual(transformed[:, 4].abs().sum().item(), 0.)
        # Rank normalization retains distinctions in the upper tail instead of
        # flattening all values above a fixed robust-scale clipping bound.
        self.assertGreater(torch.unique(transformed[-10:, 0]).numel(), 5)
        query = x.assign(heavy=1e30, clean=np.nan, category='unknown')
        values, cats = prep.transform(query)
        self.assertTrue(np.isfinite(values.numpy()).all())
        self.assertTrue((cats == 0).all())
        np.testing.assert_array_equal(prep.quantile_transformer.quantiles_, quantiles)
        # A single row and the same row in a batch have identical transforms.
        np.testing.assert_allclose(prep.transform(query.iloc[[0]])[0], values[:1])

    def test_weighted_members_use_mean_of_weighted_losses(self):
        model = TabMModel('classification', 'f1_macro', dict(ensemble_size=3))
        logits = torch.tensor([[[1., 0., -1.], [0., 2., 1.], [2., 1., 0.]],
                               [[2., 1., 0.], [1., 0., 2.], [0., 1., 2.]],
                               [[0., 2., 1.], [2., 0., 1.], [1., 2., 0.]]])
        target = torch.tensor([0, 0, 2])
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 7.]))
        expected = torch.stack([criterion(logits[:, k], target) for k in range(3)]).mean()
        self.assertTrue(torch.allclose(model._loss(criterion, logits, target), expected))
        self.assertFalse(torch.allclose(expected, criterion(logits.mean(1), target)))

    def test_quantile_weighted_fit_reload_and_legacy_shape(self):
        x = pd.DataFrame(dict(heavy=np.geomspace(1, 1e6, 32), constant=1., category=['a', 'b']*16))
        y = np.asarray(['major']*24 + ['minor']*8)
        model = TabMModel('classification', 'f1_macro', dict(iterations=1, width=8, layers=1,
            ensemble_size=2, embedding_dim=4, n_bins=4, batch_size=32,
            numeric_preprocessing='quantile', auto_class_weights='SqrtBalanced'))
        captured = []
        original = TabMModel._loss
        def record(worker, criterion, logits, target):
            captured.append(criterion.weight.detach().numpy().copy())
            return original(worker, criterion, logits, target)
        with patch.object(TabMModel, '_loss', autospec=True, side_effect=record):
            model.fit(x, y)
        expected = np.sqrt(np.array([1., 3.]))
        expected /= expected.mean()
        np.testing.assert_allclose(captured[0], expected, rtol=1e-6)
        self.assertEqual(model.preprocessor.transform(x)[0].shape[1], 2)
        prediction = model.predict_proba(x)
        scratch = ROOT/'work'/'tabm_f1_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            path = Path(temp)/'model.joblib'
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(x), prediction, rtol=1e-6)
            np.testing.assert_array_equal(restored.preprocessor.quantile_transformer.quantiles_,
                                          model.preprocessor.quantile_transformer.quantiles_)
        legacy = TabMModel('classification', 'f1_macro', {})
        prep = legacy._make_preprocessor(x)
        self.assertEqual(prep.transform(x)[0].shape[1], 4)
        # Saved pre-upgrade params lack numeric_preprocessing; reload uses the
        # original robust preprocessor's exact input dimensionality.
        legacy.params.pop('numeric_preprocessing')
        legacy.preprocessor, legacy.output_size = prep, 2
        self.assertEqual(legacy._build()(*prep.transform(x)).shape, (32, 8, 2))


if __name__ == '__main__':
    unittest.main()
