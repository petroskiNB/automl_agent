"""Pinned LightAutoML 0.4.2 adapter: inner selection, fixed full-past replay.

No fit receives outer validation. Feature pipelines are rebuilt on the refit
training data with target encoding disabled, and inner blend weights stay fixed.
"""
from contextlib import contextmanager
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd


def input_roles(frame, target):
    roles = {'target': target}
    cats = [c for c in frame if c != target and not pd.api.types.is_numeric_dtype(frame[c])]
    if cats:
        from lightautoml.dataset.roles import CategoryRole
        roles[CategoryRole(dtype=str, encoding_type='int')] = cats
    return roles


@contextmanager
def record_linear_regularization():
    """Record the C selected by the pinned LBFGS inner holdout, without replaying search."""
    from lightautoml.ml_algo.torch_based.linear_model import TorchBasedLinearEstimator
    original = TorchBasedLinearEstimator.fit
    def fit(model, *args, **kwargs):
        optimize, metric = model._optimize, model.metric
        best, current = -np.inf, None
        def tracked_optimize(*values, **options):
            nonlocal current
            current = float(options.get('c', values[-1]))
            return optimize(*values, **options)
        def tracked_metric(*values, **options):
            nonlocal best
            score = metric(*values, **options)
            if float(score) > best:
                best = float(score)
                model.refit_regularization = current
            return score
        model._optimize, model.metric = tracked_optimize, tracked_metric
        try:
            return original(model, *args, **kwargs)
        finally:
            model.__dict__.pop('_optimize', None)
            model.metric = metric
    TorchBasedLinearEstimator.fit = fit
    try:
        yield
    finally:
        TorchBasedLinearEstimator.fit = original


def _pool(dataset, label=None):
    import catboost as cb
    frame = dataset.to_pandas().data.copy()
    cats = [c for c in dataset.features if dataset.roles[c].name == 'Category']
    for column in cats:
        if getattr(dataset.roles[column], 'label_encoded', False):
            frame[column] = frame[column].astype('int64')
        else:
            frame[column] = frame[column].fillna('__MISSING__').astype(str)
    return cb.Pool(frame, label=label, cat_features=cats, feature_names=dataset.features)


def _predict_component(kind, model, dataset):
    if kind == 'cb':
        return np.asarray(model.predict(_pool(dataset))).reshape(-1)
    if kind == 'linear_l2':
        return np.asarray(model.predict(dataset.to_numpy().data)).reshape(-1)
    return np.asarray(model.predict(dataset.data)).reshape(-1)


class RefitPipeline:
    def __init__(self, reader, groups, metadata):
        self.reader, self.groups, self.metadata = reader, groups, metadata

    def predict(self, frame):
        dataset = self.reader.read(frame, add_array_attrs=False)
        prediction = np.zeros(len(frame))
        for group in self.groups:
            transformed = group['features'].transform(dataset)
            for component in group['components']:
                prediction += component['weight'] * _predict_component(
                    component['kind'], component['model'], transformed)
        return SimpleNamespace(data=prediction.reshape(-1, 1))


def full_past_refit(selected, frame, target, threads):
    """Replay every retained component with frozen inner choices, no validation set."""
    from lightautoml.reader.base import PandasToPandasReader
    if len(selected.levels) != 1:
        raise ValueError('Full refit supports exactly one LightAutoML level')
    weights = np.asarray(selected.blender.wts, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid frozen inner LightAutoML blend weights')
    reader = PandasToPandasReader(selected.task, cv=None, random_state=42, n_jobs=threads, advanced_roles=False)
    dataset = reader.fit_read(frame, roles=input_roles(frame, target))
    if dataset.folds is not None:
        raise ValueError('Refit preprocessing must not construct random target-encoding folds')
    truth = np.asarray(dataset.target)
    groups, details, position = [], [], 0
    for pipe in selected.levels[0]:
        kinds = {type(algo).__name__ for algo in pipe.ml_algos}
        if not kinds <= {'LinearLBFGS', 'BoostLGBM', 'BoostCB'}:
            raise ValueError('Unsupported LightAutoML component during fixed refit: ' + str(kinds))
        family = 'linear_l2' if kinds == {'LinearLBFGS'} else 'gbm'
        if family == 'gbm' and 'LinearLBFGS' in kinds:
            raise ValueError('Mixed preprocessing families are not supported')
        features = selected.get_feature_pipeline(family)
        transformed = features.fit_transform(dataset)
        group = dict(features=features, components=[])
        for algo in pipe.ml_algos:
            if len(algo.models) != 1 or position >= len(weights):
                raise ValueError('Full refit requires one inner holdout model per component')
            native, weight = algo.models[0], float(weights[position])
            position += 1
            kind = {'LinearLBFGS': 'linear_l2', 'BoostLGBM': 'lgb', 'BoostCB': 'cb'}[type(algo).__name__]
            detail = dict(kind=kind, inner_name=algo.name, weight=weight, training_rows=len(frame),
                          feature_count=transformed.shape[1], validation_rows=0)
            if kind == 'lgb':
                import lightgbm as lgb
                params = copy.deepcopy(native.params)
                rounds = int(native.best_iteration or native.current_iteration())
                for key in ('num_iterations', 'num_iteration', 'num_trees', 'n_estimators', 'num_boost_round',
                            'early_stopping_round', 'early_stopping_rounds', 'categorical_column'):
                    params.pop(key, None)
                params.update(num_threads=threads, verbosity=-1)
                cats = [i for i, c in enumerate(transformed.features) if transformed.roles[c].name == 'Category']
                train = lgb.Dataset(transformed.data, label=truth, categorical_feature=cats)
                fitted = lgb.train(params, train, num_boost_round=rounds)
                detail.update(iterations=rounds, parameters=params)
            elif kind == 'cb':
                import catboost as cb
                params = copy.deepcopy(native.get_params())
                rounds = int(native.tree_count_)
                for key in ('num_trees', 'num_boost_round', 'n_estimators', 'iterations', 'od_wait',
                            'od_type', 'early_stopping_rounds', 'use_best_model', 'best_model_min_trees',
                            'verbose_eval', 'logging_level', 'verbose'):
                    params.pop(key, None)
                params.update(iterations=rounds, thread_count=threads, task_type='CPU', verbose=False,
                              allow_writing_files=False, use_best_model=False)
                fitted = cb.CatBoost(params)
                fitted.fit(_pool(transformed, truth))
                detail.update(iterations=rounds, parameters=params)
            else:
                import torch
                from lightautoml.ml_algo.torch_based.linear_model import TorchBasedLinearRegression
                regularization = getattr(native, 'refit_regularization', None)
                if regularization is None or not np.isfinite(regularization) or regularization <= 0:
                    raise ValueError('Inner-selected linear regularization was not recorded')
                design = transformed.to_numpy()
                cats = [i for i, c in enumerate(design.features) if design.roles[c].name == 'Category']
                sizes = tuple((design.data[:, cats].max(axis=0).astype(int) + 1).tolist()) if cats else ()
                torch.set_num_threads(threads)
                torch.manual_seed(42)
                fitted = TorchBasedLinearRegression(data_size=design.shape[1], categorical_idx=cats,
                    embed_sizes=sizes, output_size=1, cs=[regularization], max_iter=native.max_iter,
                    tol=native.tol, loss=selected.task.losses['torch'].loss,
                    metric=selected.task.losses['torch'].metric_func)
                numeric, categorical = fitted._prepare_data(design.data)
                labels = torch.from_numpy(truth.astype(np.float32).reshape(-1, 1))
                fitted._optimize(numeric, categorical, labels, None, regularization)
                detail.update(regularization=float(regularization), max_iter=native.max_iter)
            if not np.isfinite(_predict_component(kind, fitted, transformed)).all():
                raise ValueError('Nonfinite fixed refit training predictions')
            group['components'].append(dict(kind=kind, model=fitted, weight=weight))
            details.append(detail)
        groups.append(group)
    if position != len(weights):
        raise ValueError('Frozen blend axis differs from the refitted components')
    metadata = dict(full_train_refit=True, training_rows=len(frame), components=details,
        target_encoding=False, weights_frozen=True, hyperparameters_frozen=True,
        note='Fresh target-free preprocessing on all past rows; inner-selected iterations/C/weights; no refit validation set')
    # Tuners can emit numpy scalars; reports must remain ordinary JSON.
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError('Nonportable refit metadata: ' + type(value).__name__)
    metadata = json.loads(json.dumps(metadata, default=scalar))
    return RefitPipeline(reader, groups, metadata), metadata
