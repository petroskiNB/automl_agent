"""Author TabM 0.0.3 with train-only PLE bins and the shared CPU training loop."""
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch_model import TorchModel, TabularPreprocessor
from tabm_vendor import TabM
from num_embeddings_vendor import PiecewiseLinearEmbeddings


class QuantileTabMPreprocessor(TabularPreprocessor):
    """Train-fitted rank normalization; discard constant missing indicators only.

    This is intentionally a separate class: legacy robust preprocessors keep
    their original two-channels-per-numeric-feature shape when unpickled.
    """
    def fit(self, frame):
        super().fit(frame)
        numeric = frame[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        filled = numeric.fillna(self.medians)
        self.variable_numeric = [c for c in self.numeric if filled[c].nunique() > 1]
        self.missing_indicators = [c for c in self.numeric if numeric[c].isna().any() and numeric[c].notna().any()]
        self.quantile_transformer = None
        if self.variable_numeric:
            # Full training rows only. No validation values, targets or noisy
            # query-dependent statistics participate in fitting the transform.
            n_quantiles = min(len(frame), max(10, min(1000, len(frame) // 30)))
            self.quantile_transformer = QuantileTransformer(
                n_quantiles=n_quantiles, output_distribution='normal',
                subsample=10**9, random_state=42).fit(filled[self.variable_numeric].to_numpy())
        self.tabm_numeric_size = len(self.numeric) + len(self.missing_indicators)
        self.numeric_output_columns = self.numeric + ['missing:' + c for c in self.missing_indicators]
        return self

    def transform(self, frame):
        if list(frame.columns) != self.columns:
            raise ValueError('Feature names/order differ from fitted preprocessing')
        numeric = frame[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        values = pd.DataFrame(0., index=frame.index, columns=self.numeric)
        if self.quantile_transformer is not None:
            filled = numeric[self.variable_numeric].fillna(self.medians[self.variable_numeric])
            values.loc[:, self.variable_numeric] = self.quantile_transformer.transform(filled.to_numpy())
        missing = numeric[self.missing_indicators].isna().to_numpy(dtype=np.float32)
        values = np.concatenate([values.to_numpy(dtype=np.float32), missing], axis=1)
        categories = np.empty((len(frame), len(self.categorical)), dtype=np.int64)
        for i, c in enumerate(self.categorical):
            encoded = frame[c].astype(str).map(self.vocabularies[c]).fillna(0).to_numpy(dtype=np.int64)
            encoded[frame[c].isna().to_numpy()] = 1
            categories[:, i] = encoded
        return torch.from_numpy(values), torch.from_numpy(categories)


class TabMNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TabM.make(**kwargs)

    def forward(self, numeric, categorical):
        numeric = numeric if numeric is not None and numeric.shape[1] else None
        categorical = categorical if categorical is not None and categorical.shape[1] else None
        return self.model(numeric, categorical)


class TabMModel(TorchModel):
    def __init__(self, task_type, metric, params):
        params = dict(params)
        extra = {key: params.pop(key, default) for key, default in
                 [('ensemble_size', 8), ('num_embedding', 'ple'), ('embedding_dim', 16), ('n_bins', 32),
                  ('numeric_preprocessing', 'robust')]}
        for key, low, high in [('ensemble_size', 2, 32), ('embedding_dim', 4, 64), ('n_bins', 2, 128)]:
            if type(extra[key]) is not int or not low <= extra[key] <= high:
                raise ValueError(f'{key} must be an integer in {low}..{high}')
        if extra['num_embedding'] not in ('none', 'ple'):
            raise ValueError('num_embedding must be none or ple')
        if extra['numeric_preprocessing'] not in ('robust', 'quantile'):
            raise ValueError('numeric_preprocessing must be robust or quantile')
        params = {
            **dict(iterations=80, width=256, layers=2, batch_size=256,
                   learning_rate=.002, weight_decay=.0003), **params}
        super().__init__(task_type, metric, params)
        self.params.update(extra)

    def _make_preprocessor(self, frame):
        cls = QuantileTabMPreprocessor if self.params.get('numeric_preprocessing', 'robust') == 'quantile' else TabularPreprocessor
        prep = cls(self.params['max_categories']).fit(frame)
        numeric, _ = prep.transform(frame)
        # Quantiles see this fit's rows only; neither query values nor labels are used.
        # Constant features (including missing indicators) need two distinct endpoints.
        prep.tabm_bins = []
        for column in numeric.T:
            bins = torch.unique(torch.quantile(column, torch.linspace(0, 1, self.params['n_bins'] + 1)))
            if len(bins) == 1:
                bins = torch.stack((bins[0] - .5, bins[0] + .5))
            prep.tabm_bins.append(bins.numpy().copy())
        return prep

    def _build(self):
        prep, p = self.preprocessor, self.params
        cardinalities = [len(prep.vocabularies[c]) + 2 for c in prep.categorical]
        if sum(cardinalities) > 50000:
            raise ValueError('TabM one-hot input exceeds 50000 categories; exclude identifiers or reduce max_categories')
        embeddings = None
        numeric_size = getattr(prep, 'tabm_numeric_size', len(prep.numeric) * 2)
        if numeric_size and p['num_embedding'] == 'ple':
            with warnings.catch_warnings():
                # Binary and constant columns legitimately have a single PLE
                # interval. Avoid repeating this benign message for every
                # missing indicator on each fit/reload; keep other warnings.
                warnings.filterwarnings('ignore', message=r'The .* feature has just two bin edges, which means only one bin\..*',
                                        category=UserWarning, module=r'num_embeddings_vendor')
                embeddings = PiecewiseLinearEmbeddings([torch.tensor(v) for v in prep.tabm_bins],
                                                       p['embedding_dim'], activation=False, version='B')
        return TabMNetwork(n_num_features=numeric_size, cat_cardinalities=cardinalities or None,
                         d_out=self.output_size, num_embeddings=embeddings, k=p['ensemble_size'],
                         n_blocks=p['layers'], d_block=p['width'], dropout=p['dropout'])

    def _loss(self, criterion, output, target):
        # The mean of per-member losses, NEVER the loss of an averaged prediction.
        if self.task_type == 'classification':
            return criterion(output.flatten(0, 1), target.repeat_interleave(output.shape[1]))
        return criterion(output, target[:, None, :].expand_as(output))

    def _predict_tensors(self, tensors):
        self.network.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(tensors[0]), self.params['batch_size']):
                num, cat = (t[start:start+self.params['batch_size']] for t in tensors)
                raw = self.network(num if num.shape[1] else None, cat if cat.shape[1] else None)
                if self.task_type == 'classification':
                    values.append(raw.softmax(dim=-1).mean(dim=1).numpy())
                else:
                    pred = raw[..., 0].numpy().astype(float) * self.target_scale + self.target_mean
                    if self.params['target_transform'] == 'log1p':
                        pred = np.expm1(np.clip(pred, -30, 30))
                    values.append(pred.mean(axis=1))
        return np.concatenate(values) if values else np.empty((0, self.output_size)) if self.task_type == 'classification' else np.empty(0)
