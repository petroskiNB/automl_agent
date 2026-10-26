"""Persisted prediction-domain rules; never infer target bounds from sample minima."""
import numpy as np


def prediction_bounds(state):
    if state.get('task_type') != 'regression':
        return None
    explicit = state.get('prediction_bounds')
    if explicit is not None:
        return explicit
    if state.get('target') == 'site_eui' and {'floor_area', 'heating_degree_days_c',
                                             'cooling_degree_days_c'}.issubset(state.get('features', [])):
        return {'lower': 0., 'upper': None, 'reason': 'Nonnegative physical energy-use intensity'}
    return None


def bound_predictions(values, state):
    bounds = prediction_bounds(state)
    if bounds is None:
        return np.asarray(values)
    # Keep invalid predictions invalid so that normal validation rejects NaN/Inf.
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], bounds.get('lower'), bounds.get('upper'))
    return result


class BoundedRegressor:
    """A pickled bundle's model.predict applies the same domain rule as its score."""
    def __init__(self, estimator, bounds):
        self.estimator = estimator
        self.bounds = bounds

    def __getattr__(self, name):
        if name in {'estimator', 'bounds'}:
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def predict(self, x, *args, **kwargs):
        return bound_predictions(self.estimator.predict(x, *args, **kwargs),
                                 {'task_type': 'regression', 'prediction_bounds': self.bounds})


def bounded_model(model, state):
    bounds = prediction_bounds(state)
    return BoundedRegressor(model, bounds) if bounds is not None and not isinstance(model, BoundedRegressor) else model


class ResidualRegressor:
    """Add the causal baseline back before any domain clipping."""
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, x, *args, **kwargs):
        return np.asarray(self.estimator.predict(x, *args, **kwargs)).reshape(-1) + x['hist_baseline'].to_numpy()

    def __getattr__(self, name):
        if name == 'estimator':
            raise AttributeError(name)
        return getattr(self.estimator, name)
