"""Past-only fuzzy entity history and gated high-target specialists on frozen years."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flaml_research import FlamlResearch
from temporal_research import TemporalResearch, temporal_folds, metrics
from prediction_policy import bound_predictions, bounded_model, prediction_bounds
from research import write_json


class NearestAreaHistory:
    """Nearest area within stable groups; no same-period or query target access."""
    def __init__(self, time_column, target, group_columns, tolerance=.01):
        self.time_column, self.target = time_column, target
        self.groups, self.tolerance = list(group_columns), tolerance

    def fit(self, frame):
        self.reference = frame[list(dict.fromkeys(self.groups + ['floor_area', self.time_column, self.target]))].copy()
        return self

    def transform(self, frame):
        query = frame.drop(columns=[self.target], errors='ignore').reset_index(drop=True)
        output = pd.DataFrame(np.nan, index=np.arange(len(query)), columns=
            ['fuzzy_mean', 'fuzzy_std', 'fuzzy_count', 'fuzzy_area_error', 'fuzzy_gap'])
        periods = pd.to_numeric(query[self.time_column], errors='coerce').to_numpy()
        if not np.isfinite(periods).all():
            raise ValueError('Fuzzy history requires complete numeric time values')
        for period in np.unique(periods):
            past = self.reference.loc[self.reference[self.time_column] < period].copy()
            past['floor_area'] = pd.to_numeric(past['floor_area'], errors='coerce')
            past = past.replace([np.inf, -np.inf], np.nan).dropna(subset=self.groups + ['floor_area'])
            past = past.loc[past.floor_area > 0]
            if past.empty:
                continue
            tables = {}
            for key, group in past.groupby(self.groups, sort=False, observed=True):
                key = key if isinstance(key, tuple) else (key,)
                stats = group.groupby('floor_area')[self.target].agg(['mean', 'std', 'count']).sort_index()
                stats['last_period'] = group.groupby('floor_area')[self.time_column].max()
                tables[key] = stats
            positions = np.flatnonzero(periods == period)
            part = query.iloc[positions].copy()
            part['_position'] = positions
            part = part.dropna(subset=self.groups + ['floor_area'])
            for key, group in part.groupby(self.groups, sort=False, observed=True):
                key = key if isinstance(key, tuple) else (key,)
                stats = tables.get(key)
                if stats is None:
                    continue
                area = pd.to_numeric(group.floor_area, errors='coerce').to_numpy(dtype=float)
                known = stats.index.to_numpy(dtype=float)
                right = np.searchsorted(known, area).clip(0, len(known)-1)
                left = (np.searchsorted(known, area)-1).clip(0, len(known)-1)
                chosen = np.where(np.abs(known[left]-area) <= np.abs(known[right]-area), left, right)
                error = np.abs(known[chosen]-area) / np.maximum(np.abs(area), 1)
                valid = np.isfinite(area) & (area > 0) & (error <= self.tolerance)
                values = stats.iloc[chosen].copy()
                destination = group['_position'].to_numpy()[valid]
                output.loc[destination, :] = np.column_stack((values['mean'], values['std'].fillna(0),
                    values['count'], error, period-values['last_period']))[valid]
        output['fuzzy_count'] = output['fuzzy_count'].fillna(0)
        output['fuzzy_seen'] = (output.fuzzy_count > 0).astype(float)
        output.index = frame.index
        return output


class CombinedHistory:
    def __init__(self, base, fuzzy):
        self.base, self.fuzzy = base, fuzzy

    def transform(self, frame):
        extra = self.fuzzy.transform(frame)
        return extra if self.base is None else pd.concat([self.base.transform(frame), extra], axis=1)


def choose_strength(reference, candidate, truth):
    return min((0., .25, .5, .75, 1.), key=lambda a: float(np.mean((reference + a*(candidate-reference)-truth)**2)))


class RegressionSpecialists(FlamlResearch):
    def _identity(self, config):
        result = super()._identity(config)
        result['hashes']['regression_specialists.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    def _combined_history(self, frame, state, config):
        base = self._history(frame, state, config, config['history_mode'])
        fuzzy = NearestAreaHistory(config['time_column'], state['target'], config['fuzzy_groups'],
                                   config['area_tolerance']).fit(frame)
        return CombinedHistory(base, fuzzy)

    def _fit_main(self, fit, config, iterations, fuzzy=False):
        state, recipe = config['task'], config['recipe']
        history = self._combined_history(fit, state, config) if fuzzy else self._history(fit, state, config, config['history_mode'])
        x, cats = self._design(fit, state, recipe, history)
        x, levels = self.engine.encode(x, cats, recipe['algorithm'])
        model = self.engine.model(state, dict(recipe['params'], iterations=int(iterations)), recipe['algorithm'])
        self.engine._fit(model, recipe['algorithm'], state, x, fit[state['target']], cats)
        return dict(model=bounded_model(model, state), history=history, recipe=recipe,
                    state=state, levels=levels, prediction_bounds=prediction_bounds(state))

    def _fit_tail(self, frame, config):
        from catboost import CatBoostClassifier, CatBoostRegressor
        from torch_resources import configure
        state = config['task']
        history = self._combined_history(frame, state, config)
        x, cats = self._design(frame, state, config['recipe'], history)
        x, _ = self.engine.encode(x, cats, 'catboost')
        y = frame[state['target']].to_numpy()
        threshold = float(np.quantile(y, config['tail_quantile']))
        high = y > threshold
        result = dict(history=history, threshold=threshold, gate=None, expert=None, high_rows=int(high.sum()))
        if high.sum() < 30 or (~high).sum() < 30:
            return result
        params = dict(iterations=250, depth=5, learning_rate=.05, l2_leaf_reg=10,
                      thread_count=configure(), random_seed=42, verbose=False, allow_writing_files=False)
        # Gate sees only features/history at inference, never the query's true target.
        gate = CatBoostClassifier(**params, loss_function='Logloss')
        gate.fit(x, high.astype(int), cat_features=cats)
        expert = CatBoostRegressor(**dict(params, iterations=350, depth=6), loss_function='RMSE')
        expert.fit(x.iloc[np.flatnonzero(high)], y[high], cat_features=cats)
        result.update(gate=gate, expert=expert)
        return result

    def _tail_predictions(self, tail, frame, config, fallback):
        if tail['gate'] is None:
            return np.zeros(len(frame)), np.asarray(fallback)
        x, cats = self._design(frame, config['task'], config['recipe'], tail['history'])
        x, _ = self.engine.encode(x, cats, 'catboost')
        return tail['gate'].predict_proba(x)[:, 1], np.asarray(tail['expert'].predict(x))

    def _predict(self, bundle, frame):
        if bundle.get('backend') != 'specialists_v1':
            return TemporalResearch(self.engine.root, self.engine.files, self.engine.folder)._predict(bundle, frame)
        base = self._predict(bundle['base'], frame)
        if bundle['mode'] == 'reference':
            return base
        fuzzy = self._predict(bundle['fuzzy'], frame)
        if bundle['mode'] == 'fuzzy_history':
            return fuzzy
        probability, high = self._tail_predictions(bundle['tail'], frame, bundle['config'], base)
        initial = fuzzy if bundle['mode'] == 'tail_fuzzy' else base
        alpha = bundle['strengths'][bundle['mode']]
        return bound_predictions(initial + alpha * probability * (high-initial), bundle['config']['task'])

    def run_regression_specialists(self, reference_suite_id, area_tolerance=.01, tail_quantile=.95, time_limit=1800):
        if not np.isfinite(area_tolerance) or not 0 < area_tolerance <= .05:
            raise ValueError('area_tolerance must be positive and <= .05')
        if not np.isfinite(tail_quantile) or not .8 <= tail_quantile <= .99:
            raise ValueError('tail_quantile must be .8.. .99')
        if type(time_limit) is not int or not 120 <= time_limit <= 2400:
            raise ValueError('time_limit must be 120..2400 seconds including final fits')
        lock = self.run / '.temporal.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(handle)
        try:
            source, original, sr, worker = self._reference(reference_suite_id)
            required = {'floor_area', 'State_Factor', 'facility_type', 'building_class', 'year_built'}
            if not required <= set(original['task']['features']):
                raise ValueError('Fuzzy building comparison requires the energy-use schema')
            config = {k: original[k] for k in ('task', 'time_column', 'key_columns', 'group_columns',
                'validation_periods', 'key_variants', 'group_hierarchy', 'smoothing', 'area_rounding') if k in original}
            config.update(backend='specialists_v1', reference_suite_id=reference_suite_id,
                recipe=original['recipe'], history_mode=sr['selected_mode'], area_tolerance=float(area_tolerance),
                tail_quantile=float(tail_quantile), time_limit=time_limit,
                fuzzy_groups=['State_Factor', 'facility_type', 'building_class', 'year_built'],
                modes=['reference', 'fuzzy_history', 'tail_expert', 'tail_fuzzy'])
            identity = self._identity(config)
            sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            directory = self.run / 'temporal' / sid
            if (directory / 'report.json').exists():
                return json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', identity)
            result = self._run_specialists(directory, config, source, sr, worker, self._end(time_limit))
            result.update(suite_id=sid, report_path=str(directory / 'report.json'))
            write_json(directory / 'report.json', result)
            write_json(self.run / 'regression_specialists_summary.json', result)
            return result
        finally:
            lock.unlink(missing_ok=True)

    def _run_specialists(self, directory, config, source, sr, worker, end):
        train, test = self.engine.load()
        state, mode = config['task'], config['history_mode']
        folds = temporal_folds(train, config['time_column'], config['validation_periods'])
        source_rows = next(r for r in sr['results'] if r['mode'] == mode)['folds']
        rows = {name: [] for name in config['modes']}
        oof = {name: np.full(len(train), np.nan) for name in rows}
        for i, (period, tr, va) in enumerate(folds):
            if end - time.monotonic() < 15:
                raise TimeoutError('Specialist suite exhausted its budget')
            with np.load(source / f'fold_{i}.npz') as saved:
                if not np.array_equal(tr, saved['train']) or not np.array_equal(va, saved['validation']):
                    raise ValueError('Specialist temporal indices differ')
            fit, validation = train.iloc[tr], train.iloc[va]
            column = config['time_column']
            last = fit[column].max()
            inner, calibration = fit.loc[fit[column] < last], fit.loc[fit[column] == last]
            iterations = int(source_rows[i]['iterations'])
            base_cal = self._fit_main(inner, config, iterations)
            fuzzy_cal = self._fit_main(inner, config, iterations, fuzzy=True)
            tail_cal = self._fit_tail(inner, config)
            reference_cal, fuzzy_prediction = self._predict(base_cal, calibration), self._predict(fuzzy_cal, calibration)
            gate, high = self._tail_predictions(tail_cal, calibration, config, reference_cal)
            truth = calibration[state['target']].to_numpy()
            strengths = dict(tail_expert=choose_strength(reference_cal, reference_cal + gate*(high-reference_cal), truth),
                tail_fuzzy=choose_strength(fuzzy_prediction, fuzzy_prediction + gate*(high-fuzzy_prediction), truth))
            base = joblib.load(source / f'{mode}_fold_{i}.joblib')
            fuzzy = self._fit_main(fit, config, iterations, fuzzy=True)
            tail = self._fit_tail(fit, config)
            reference = np.load(source / f'{mode}_fold_{i}.npy')
            if not np.allclose(self._predict(base, validation), reference, atol=1e-10, rtol=1e-10):
                raise ValueError('Reference model/predictions differ')
            np.savez(directory / f'fold_{i}.npz', train=tr, validation=va)
            for name in config['modes']:
                bundle = dict(backend='specialists_v1', mode=name, base=base, fuzzy=fuzzy, tail=tail,
                              config=config, state=state, strengths=strengths)
                prediction = self._predict(bundle, validation)
                row = dict(period=float(period), train_rows=len(tr), iterations=iterations,
                    **metrics(validation[state['target']], prediction), strengths=strengths,
                    tail_threshold=tail['threshold'], tail_training_rows=tail['high_rows'])
                high_mask = validation[state['target']].to_numpy() > tail['threshold']
                row['segments'] = {label: metrics(validation[state['target']].to_numpy()[mask], prediction[mask])
                    for label, mask in (('high_target_diagnostic', high_mask), ('other', ~high_mask)) if mask.any()}
                row['fuzzy_coverage'] = float(fuzzy['history'].transform(validation).fuzzy_seen.mean())
                joblib.dump(bundle if name != 'reference' else base, directory / f'{name}_fold_{i}.joblib')
                np.save(directory / f'{name}_fold_{i}.npy', prediction)
                rows[name].append(row)
                oof[name][va] = prediction
            latest_iterations, latest_strengths = iterations, strengths
        # Prepare every deployable candidate inside the suite budget, before recording success.
        if end - time.monotonic() < 15:
            raise TimeoutError('No time left for full-past specialist refits')
        exported = worker.finalize_temporal(config['reference_suite_id'], mode=mode)
        base = joblib.load(exported['model_path'])
        fuzzy, tail = self._fit_main(train, config, latest_iterations, fuzzy=True), self._fit_tail(train, config)
        if time.monotonic() > end:
            raise TimeoutError('Specialist final refit exceeded its allocation')
        for name in config['modes']:
            bundle = dict(backend='specialists_v1', mode=name, base=base, fuzzy=fuzzy,
                          tail=tail, config=config, state=state, strengths=latest_strengths)
            prediction = self._predict(bundle, test)
            joblib.dump(bundle, directory / f'final_pipeline_{name}.joblib')
            if not np.allclose(self._predict(joblib.load(directory / f'final_pipeline_{name}.joblib'), test), prediction):
                raise ValueError('Specialist reload changed predictions')
            np.save(directory / f'final_predictions_{name}.npy', prediction)
        results = []
        for name, fold_rows in rows.items():
            np.save(directory / f'{name}_oof.npy', oof[name])
            scores = [r['rmse'] for r in fold_rows]
            mask = np.isfinite(oof[name])
            results.append(dict(mode=name, folds=fold_rows, mean_rmse=float(np.mean(scores)),
                latest_rmse=scores[-1], worst_rmse=float(max(scores)), std_rmse=float(np.std(scores)),
                pooled=metrics(train.loc[mask, state['target']], oof[name][mask])))
        winner = min(results, key=lambda r: (r['mean_rmse'], r['latest_rmse'], r['worst_rmse']))['mode']
        return dict(selected_mode=winner, results=results, reference_suite_id=config['reference_suite_id'],
            reference_mode=mode, prediction_bounds=prediction_bounds(state),
            note='Frozen outer years. Fuzzy matching excludes current/future periods and query targets. '
                 'Tail threshold uses training targets; inference gate uses features only. Strengths fit last inner past year. '
                 'All full-past pipelines saved inside suite budget; export performs no fitting. Adaptive validation.')

    def finalize_temporal(self, suite_id, mode=None):
        self._check_id(suite_id)
        directory = self.run / 'temporal' / suite_id
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        if plan != self._identity(plan['config']):
            raise ValueError('Stale regression specialists')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        mode = report['selected_mode'] if mode is None else mode
        if mode not in plan['config']['modes']:
            raise ValueError('Unknown specialist mode')
        _, test = self.engine.load()
        bundle = joblib.load(directory / f'final_pipeline_{mode}.joblib')
        prediction = self._predict(bundle, test)
        saved = np.load(directory / f'final_predictions_{mode}.npy')
        if not np.allclose(prediction, saved, atol=1e-10, rtol=1e-10):
            raise ValueError('Saved specialist export differs from its pipeline')
        output = test[plan['config']['task']['ids']].copy()
        output[plan['config']['task']['target']] = prediction
        suffix = '__' + mode
        output.to_csv(directory / f'submission{suffix}.csv', index=False)
        shutil.copy2(directory / f'final_pipeline_{mode}.joblib', directory / f'final_model{suffix}.joblib')
        return dict(suite_id=suite_id, mode=mode, rows=len(test),
            submission_path=str(directory / f'submission{suffix}.csv'),
            model_path=str(directory / f'final_model{suffix}.joblib'), note='Saved full-past specialist pipeline; no export-time fit')
