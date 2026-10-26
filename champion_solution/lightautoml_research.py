"""Budgeted LightAutoML regression candidate on a frozen temporal outer plan.

The preset selects parameters/iterations/blend weights on an INNER past holdout.
An optional paired mode replays these fixed choices on every available past row
through the pinned native adapter, without a validation set or further search.
Outer validation is never passed to fitting. Historical transforms exclude the
query period and later periods. All work, including final replay, is budgeted.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext

from torch_resources import configure
configure()

import joblib
import numpy as np
import pandas as pd

from flaml_research import FlamlResearch, _kill_tree
from prediction_policy import bound_predictions, prediction_bounds
from research import write_json
from temporal_research import metrics, temporal_folds


VERSION = '0.4.2'
ALGORITHMS = {'linear_l2', 'lgb', 'cb', 'lgb_tuned', 'cb_tuned'}
TARGET_COLUMN = '__automl_lama_target__'


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalise_design(frame):
    """Keep column identities and raw categories; LAMA fits its own encoders."""
    result = frame.copy()
    if TARGET_COLUMN in result:
        raise ValueError('LightAutoML reserved target column collides with a feature')
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].fillna('__MISSING__').astype(str)
    return result


def tabular_parameters(algorithms, seconds, threads, tuning=False, max_tuning_trials=8):
    """One CPU level, target-free preprocessing and bounded inner-only tuning."""
    return dict(timeout=max(1, int(seconds * .95)), cpu_limit=threads,
        gpu_ids=None, debug=True,
        general_params=dict(use_algos=[list(algorithms)], nested_cv=False,
                            return_all_predictions=False),
        reader_params=dict(cv=None, random_state=42, n_jobs=threads, advanced_roles=False),
        nested_cv_params=dict(cv=1), selection_params=dict(mode=0),
        timing_params=dict(mode=2, tuning_rate=.35 if tuning else 0),
        tuning_params=dict(max_tuning_iter=max_tuning_trials,
                           max_tuning_time=max(1, min(90, int(seconds*.2))), fit_on_holdout=True),
        gbm_pipeline_params=dict(top_intersections=0),
        linear_pipeline_params=dict(top_intersections=0),
        lgb_params=dict(default_params=dict(num_threads=threads)),
        cb_params=dict(default_params=dict(task_type='CPU', thread_count=threads)))


def fit_child(request_path, output_path):
    """Private bounded fit. Payload contains inner train/holdout, never outer rows."""
    if importlib.metadata.version('lightautoml') != VERSION:
        raise ValueError('LightAutoML 0.4.2 is required')
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    payload = joblib.load(request_path)
    train = payload['inner_x'].copy()
    validation = payload['inner_validation_x'].copy()
    train[TARGET_COLUMN] = np.asarray(payload['inner_y'])
    validation[TARGET_COLUMN] = np.asarray(payload['inner_validation_y'])
    refit = payload.get('refit_full', False)
    selection_seconds = payload['fit_seconds'] * (.65 if refit else 1)
    model = TabularAutoML(task=Task('reg', loss='mse', metric='mse'),
        **tabular_parameters(payload['algorithms'], selection_seconds, configure(),
                             payload.get('tuning', False), payload.get('max_tuning_trials', 8)))
    from lightautoml_refit import input_roles, record_linear_regularization, full_past_refit
    with record_linear_regularization() if refit else nullcontext():
        predictions = np.asarray(model.fit_predict(train, roles=input_roles(train, TARGET_COLUMN),
            valid_data=validation, verbose=0).data).reshape(-1)
    if predictions.shape != (len(validation),) or not np.isfinite(predictions).all():
        raise ValueError('LightAutoML returned invalid inner holdout predictions')
    fitted = [type(algo).__name__ for level in getattr(model, 'levels', [])
              for pipeline in level for algo in getattr(pipeline, 'ml_algos', [])]
    metadata = dict(requested_algorithms=payload['algorithms'], fitted_algorithms=fitted,
        retained_algorithms=[algo.name for level in getattr(model, 'levels', [])
                             for pipeline in level for algo in getattr(pipeline, 'ml_algos', [])],
        tuning=payload.get('tuning', False), max_tuning_trials=payload.get('max_tuning_trials', 8),
        categorical_columns=[c for c, role in getattr(getattr(model, 'reader', None), 'roles', {}).items()
                             if getattr(role, 'name', None) == 'Category'],
        inner_rmse=metrics(payload['inner_validation_y'], predictions)['rmse'],
        estimator_training_rows=len(train), inner_validation_rows=len(validation),
        full_train_refit=False)
    result = dict(model=model, selected=metadata)
    if refit:
        full = payload['outer_x'].copy()
        full[TARGET_COLUMN] = np.asarray(payload['outer_y'])
        result['refit_model'], result['refit_metadata'] = full_past_refit(model, full, TARGET_COLUMN, configure())
    joblib.dump(result, output_path)


class LightAutoMLResearch(FlamlResearch):
    def _identity(self, config):
        # This must not call super: the generic hybrid identity dispatches here.
        paths = dict(self.engine.files)
        for name in ('research.py', 'temporal_research.py', 'prediction_policy.py',
                     'hierarchical_history.py', 'temporal_ensemble.py', 'torch_suites.py',
                     'torch_backend.py', 'torch_model.py', 'torch_resources.py',
                     'research_policy.py', 'reference_recipes.py', 'xgb_backend.py',
                     'temporal_search.py', 'flaml_research.py', 'lightautoml_research.py', 'lightautoml_refit.py'):
            paths[name] = Path(__file__).with_name(name)
        source = self.run / 'temporal' / config['reference_suite_id']
        for name in ('plan.json', 'report.json'):
            paths['reference/' + name] = source / name
        hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for name, path in paths.items()}
        hashes.update(cpu_threads=str(configure()), lightautoml_version=VERSION)
        for name in ('tabm_model.py', 'tabm_vendor.py', 'num_embeddings_vendor.py'):
            hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return dict(config=config, hashes=hashes)

    def _reference(self, suite_id):
        try:
            return super()._reference(suite_id)
        except ValueError as exc:
            raise ValueError(str(exc).replace('FLAML', 'LightAutoML')) from exc

    def _child(self, directory, payload, seconds):
        end = time.monotonic() + seconds
        if seconds < 3:
            raise TimeoutError('LightAutoML allocation exhausted before a bounded fit')
        job = directory / ('fit_' + str(time.time_ns()))
        request, result, log = job.with_suffix('.joblib'), job.with_suffix('.result.joblib'), job.with_suffix('.txt')
        joblib.dump(payload, request)
        try:
            if end-time.monotonic() < 1:
                raise TimeoutError('LightAutoML allocation exhausted while preparing fit data')
            with log.open('w', encoding='utf-8') as output:
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = ''
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--fit', str(request), str(result)], stdout=output, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent), env=environment)
                try:
                    code = process.wait(timeout=max(.01, end-time.monotonic()))
                except subprocess.TimeoutExpired:
                    _kill_tree(process)
                    raise TimeoutError('LightAutoML exceeded its whole-suite allocation; partial logs were kept')
            if code != 0:
                tail = log.read_text(encoding='utf-8', errors='replace')[-4000:]
                raise RuntimeError(f'LightAutoML child failed (exit {code}): {tail}')
            return joblib.load(result)
        finally:
            request.unlink(missing_ok=True)
            if result.exists() and 'code' in locals() and code == 0:
                result.unlink()

    def _fit_fold(self, fit, state, config, directory, seconds):
        end = time.monotonic() + seconds
        values = sorted(pd.to_numeric(fit[config['time_column']], errors='raise').unique())
        if len(values) < 2:
            raise ValueError('LightAutoML needs two strictly past periods for its inner holdout')
        inner = fit.loc[fit[config['time_column']] < values[-1]]
        validation = fit.loc[fit[config['time_column']] == values[-1]]
        # Causal transforms exclude >=query period, including for training rows.
        # Known targets from the latest past period are available at prediction.
        history = self._history(fit, state, config, config['history_mode'])
        x, _ = self._design(inner, state, config['recipe'], history)
        xv, _ = self._design(validation, state, config['recipe'], history)
        payload = dict(inner_x=normalise_design(x), inner_y=inner[state['target']],
            inner_validation_x=normalise_design(xv), inner_validation_y=validation[state['target']],
            algorithms=config['algorithms'], fit_seconds=max(1, end-time.monotonic()-1),
            tuning=config.get('tuning', False), max_tuning_trials=config.get('max_tuning_trials', 8),
            refit_full=config.get('refit_full', False))
        if config.get('refit_full'):
            outer_x, _ = self._design(fit, state, config['recipe'], history)
            payload.update(outer_x=normalise_design(outer_x), outer_y=fit[state['target']])
        result = self._child(directory, payload, end-time.monotonic())
        selected = dict(result['selected'], estimator_training_periods=[float(v) for v in values[:-1]],
            inner_validation_period=float(values[-1]), historical_target_cutoff=float(values[-1]),
            full_train_refit=False)
        bundle = dict(backend='lightautoml_v1', model=result['model'], history=history,
            recipe=config['recipe'], state=state, prediction_bounds=prediction_bounds(state),
            selected=selected)
        if config.get('refit_full'):
            refit_selected = dict(selected, full_train_refit=True, estimator_training_periods=[float(v) for v in values],
                                  estimator_training_rows=len(fit), refit_validation_rows=0,
                                  refit=result['refit_metadata'])
            bundle['_refit_bundle'] = dict(bundle, model=result['refit_model'], selected=refit_selected)
        return bundle, selected

    def _predict(self, bundle, frame):
        x, _ = self._design(frame, bundle['state'], bundle['recipe'], bundle['history'])
        output = bundle['model'].predict(normalise_design(x))
        # LAMA returns NumpyDataset, whose .data contains the prediction matrix.
        return bound_predictions(np.asarray(output.data).reshape(-1), bundle['state'])

    def run_lightautoml_suite(self, reference_suite_id, time_limit=1800, algorithms=None, feature_code=None,
                             feature_mode='inherit', tuning=False, max_tuning_trials=8, refit_full=False):
        cap = int(os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0'))
        if cap <= 0:
            raise ValueError('Enable LightAutoML with --lightautoml-budget first')
        if type(time_limit) is not int or not 120 <= time_limit <= 3600:
            raise ValueError('time_limit must be 120..3600 seconds for ALL folds and the saved final fit')
        algorithms = ['linear_l2', 'lgb', 'cb'] if algorithms is None else algorithms
        if type(tuning) is not bool or type(refit_full) is not bool:
            raise ValueError('tuning and refit_full must be booleans')
        if type(max_tuning_trials) is not int or not 1 <= max_tuning_trials <= 12:
            raise ValueError('max_tuning_trials must be in 1..12')
        if (not isinstance(algorithms, list) or not 1 <= len(algorithms) <= 5 or
                any(not isinstance(value, str) or value not in ALGORITHMS for value in algorithms) or
                len(set(algorithms)) != len(algorithms)):
            raise ValueError('Choose distinct LightAutoML algorithms from linear_l2/lgb/cb/lgb_tuned/cb_tuned')
        if not tuning and any(a.endswith('_tuned') for a in algorithms):
            raise ValueError('Tuned algorithms require tuning=True')
        if tuning:
            algorithms = list(dict.fromkeys(algorithms + [a+'_tuned' for a in algorithms if a in {'lgb', 'cb'}]))
        if feature_mode not in {'inherit', 'compact_strings'} or (feature_mode != 'inherit' and feature_code is not None):
            raise ValueError('Choose feature_mode inherit or compact_strings; do not combine compact_strings with feature_code')
        if feature_code is not None and not isinstance(feature_code, str):
            raise ValueError('feature_code must be a row-wise transform string or omitted')
        lock = self.run / '.lightautoml.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(handle)
        started, charged, budget = time.monotonic(), False, None
        ledger = self.run / 'lightautoml_budget.json'
        try:
            source, source_config, source_report, reference_worker = self._reference(reference_suite_id)
            if feature_mode == 'compact_strings':
                from research_policy import reference_recipe
                feature_code = reference_recipe('regression_control_lgbm_compact', self.engine.state())['feature_code']
            inherited = {key: source_config[key] for key in ('task', 'time_column', 'key_columns', 'group_columns',
                'validation_periods', 'key_variants', 'group_hierarchy', 'smoothing', 'area_rounding')
                if key in source_config}
            config = dict(inherited, backend='lightautoml_v1', reference_suite_id=reference_suite_id,
                history_mode=source_report['selected_mode'], time_limit=time_limit,
                algorithms=algorithms, modes=['reference', 'lightautoml'] + (['lightautoml_refit'] if refit_full else []),
                recipe=dict(algorithm='lightautoml', params={}, feature_code=source_config['recipe'].get('feature_code', '')
                            if feature_code is None else feature_code),
                selection='lowest outer-fold mean RMSE; ties latest, worst, unchanged reference',
                inner_selection='last available past period; no outer rows or targets; one CPU level',
                final_fit='saved inner pipeline and optional fixed full-past replay; all work inside suite budget',
                full_train_refit=refit_full, refit_full=refit_full, nested_cv=False, feature_selection=False,
                tuning=tuning, max_tuning_trials=max_tuning_trials, feature_mode=feature_mode,
                target_encoding=False)
            identity = self._identity(config)
            sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            directory = self.run / 'temporal' / sid
            if (directory / 'report.json').exists():
                if json.loads((directory / 'plan.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('Cached LightAutoML plan differs from the current identity')
                self._ready_final(directory, config, self.engine.load()[1])
                if refit_full:
                    self._ready_final(directory, config, self.engine.load()[1], mode='lightautoml_refit')
                return json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            if importlib.metadata.version('lightautoml') != VERSION:
                raise ValueError('LightAutoML 0.4.2 is required')
            budget = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else dict(charged_seconds=0, calls=[])
            if budget['charged_seconds'] + time_limit > cap:
                raise ValueError(f'LightAutoML remaining allocation {cap-budget["charged_seconds"]}s; request a smaller suite budget')
            budget['charged_seconds'] += time_limit
            budget['calls'].append(dict(suite_id=sid, reserved_seconds=time_limit, status='running'))
            write_json(ledger, budget)
            charged = True
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', identity)
            result = self._run(directory, config, source, source_report, reference_worker,
                               self._end(time_limit-(time.monotonic()-started)))
            result.update(suite_id=sid, report_path=str(directory / 'report.json'))
            write_json(directory / 'report.json', result)
            write_json(self.run / 'lightautoml_summary.json', result)
            budget['calls'][-1]['status'] = 'completed'
            return result
        finally:
            if charged:
                elapsed = min(time_limit, max(1, int(np.ceil(time.monotonic()-started))))
                budget['charged_seconds'] -= time_limit-elapsed
                budget['calls'][-1]['charged_seconds'] = elapsed
                if budget['calls'][-1]['status'] == 'running':
                    budget['calls'][-1]['status'] = 'failed'
                write_json(ledger, budget)
            lock.unlink(missing_ok=True)

    def _run(self, directory, config, source, source_report, reference_worker, end):
        train, test = self.engine.load()
        state, mode = config['task'], source_report['selected_mode']
        if not (test[config['time_column']] > train[config['time_column']].max()).all():
            raise ValueError('LightAutoML requires strictly future test periods')
        folds = temporal_folds(train, config['time_column'], config['validation_periods'])
        rows = {name: [] for name in config['modes']}
        oof = {name: np.full(len(train), np.nan) for name in rows}
        source_result = next(row for row in source_report['results'] if row['mode'] == mode)
        for index, (period, tr, va) in enumerate(folds):
            with np.load(source / f'fold_{index}.npz') as indices:
                if not np.array_equal(tr, indices['train']) or not np.array_equal(va, indices['validation']):
                    raise ValueError('Reference temporal rows differ from the frozen plan')
            reference = np.load(source / f'{mode}_fold_{index}.npy')
            if reference.shape != (len(va),) or not np.isfinite(reference).all():
                raise ValueError('Invalid saved reference predictions')
            measured = metrics(train.iloc[va][state['target']], reference)
            previous = source_result['folds'][index]
            if (float(previous['period']) != float(period) or
                    not np.isclose(previous['rmse'], measured['rmse'], rtol=0, atol=1e-10)):
                raise ValueError('Saved reference metrics do not match validation rows')
            np.savez(directory / f'fold_{index}.npz', train=tr, validation=va)
            fit_dir = directory / f'lightautoml_fold_{index}'
            fit_dir.mkdir(exist_ok=True)
            started = time.monotonic()
            allocation = (end-time.monotonic()-3)/(len(folds)-index+1)
            bundle, selected = self._fit_fold(train.iloc[tr], state, config, fit_dir, allocation)
            refit_bundle = bundle.pop('_refit_bundle', None)
            pred = self._predict(bundle, train.iloc[va])
            self._save_checked(bundle, directory / f'lightautoml_fold_{index}.joblib', train.iloc[va], pred)
            predictions = [('reference', reference), ('lightautoml', pred)]
            if refit_bundle is not None:
                values = self._predict(refit_bundle, train.iloc[va])
                self._save_checked(refit_bundle, directory / f'lightautoml_refit_fold_{index}.joblib', train.iloc[va], values)
                predictions.append(('lightautoml_refit', values))
            for name, values in predictions:
                row = dict(period=float(period), train_rows=len(tr), **metrics(train.iloc[va][state['target']], values))
                if name == 'reference':
                    row.update(iterations=previous['iterations'], source_mode=mode)
                    shutil.copy2(source / f'{mode}_fold_{index}.joblib', directory / f'reference_fold_{index}.joblib')
                else:
                    row.update(iterations=refit_bundle['selected'] if name == 'lightautoml_refit' else selected,
                               seconds=round(time.monotonic()-started, 2))
                rows[name].append(row)
                oof[name][va] = values
                np.save(directory / f'{name}_fold_{index}.npy', values)
                write_json(directory / f'{name}_fold_{index}.json', row)
        # Reserve one share for export-ready final selection and optional replay.
        fit_dir = directory / 'lightautoml_final'
        fit_dir.mkdir(exist_ok=True)
        final_bundle, selected = self._fit_fold(train, state, config, fit_dir, end-time.monotonic()-2)
        final_refit = final_bundle.pop('_refit_bundle', None)
        final_pred = self._predict(final_bundle, test)
        self._save_checked(final_bundle, directory / 'final_pipeline.joblib', test, final_pred)
        write_json(directory / 'final_pipeline.json', selected)
        np.save(directory / 'final_predictions.npy', final_pred)
        write_json(directory / 'final_artifact.json', dict(
            model_sha256=_file_hash(directory / 'final_pipeline.joblib'),
            predictions_sha256=_file_hash(directory / 'final_predictions.npy')))
        if final_refit is not None:
            prefix = 'final_pipeline__lightautoml_refit'
            pred = self._predict(final_refit, test)
            self._save_checked(final_refit, directory / (prefix + '.joblib'), test, pred)
            write_json(directory / (prefix + '.json'), final_refit['selected'])
            np.save(directory / (prefix + '.npy'), pred)
            write_json(directory / (prefix + '_artifact.json'), dict(
                model_sha256=_file_hash(directory / (prefix + '.joblib')),
                predictions_sha256=_file_hash(directory / (prefix + '.npy'))))
        results = []
        for name in rows:
            np.save(directory / f'{name}_oof.npy', oof[name])
            scores = [row['rmse'] for row in rows[name]]
            mask = np.isfinite(oof[name])
            results.append(dict(mode=name, folds=rows[name], mean_rmse=float(np.mean(scores)),
                latest_rmse=scores[-1], worst_rmse=float(max(scores)), std_rmse=float(np.std(scores)),
                pooled=metrics(train.loc[mask, state['target']], oof[name][mask]),
                fold_rmse_delta_vs_baseline=[row['rmse']-base['rmse'] for row, base in zip(rows[name], rows['reference'])]))
        winner = min(results, key=lambda row: (row['mean_rmse'], row['latest_rmse'], row['worst_rmse']))['mode']
        coverage = reference_worker._history(train, state, config, 'history_exact').transform(test)
        return dict(selected_mode=winner, results=results, reference_suite_id=config['reference_suite_id'],
            reference_mode=mode, test_key_coverage=float(coverage.hist_exact_seen.mean()),
            prediction_bounds=prediction_bounds(state), final_pipeline=selected,
            refit_pipeline=final_refit['selected'] if final_refit else None,
            note='Identical frozen temporal outer rows. LightAutoML receives only inner past train/holdout. '
                 'One CPU level linear/LGB/CB, optional bounded inner tuning, no selection or nested CV. '
                 'lightautoml retains its inner holdout; lightautoml_refit, when requested, replays frozen inner '
                 'parameters/iterations/weights on all past rows with fresh target-free preprocessing. '
                 'Causal history uses all targets strictly before each query period. Final fit is included in '
                 'the whole-suite budget; export reuses its saved pipeline. Development outer years are not an untouched test.')

    def _save_checked(self, bundle, path, frame, pred):
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError('LightAutoML produced invalid predictions')
        joblib.dump(bundle, path)
        if not np.allclose(self._predict(joblib.load(path), frame), pred, rtol=1e-10, atol=1e-10):
            raise ValueError('Reloaded LightAutoML pipeline changed predictions')

    def _ready_final(self, directory, config, test, mode='lightautoml'):
        """Verify the export-ready artifact without changing it or fitting again."""
        names = ('final_pipeline.joblib', 'final_pipeline.json', 'final_predictions.npy', 'final_artifact.json')
        if mode == 'lightautoml_refit':
            prefix = 'final_pipeline__lightautoml_refit'
            names = (prefix+'.joblib', prefix+'.json', prefix+'.npy', prefix+'_artifact.json')
        paths = [directory / name for name in names]
        if not all(path.exists() for path in paths):
            raise ValueError('Cached LightAutoML suite is missing its final pipeline artifacts')
        manifest = json.loads(paths[3].read_text(encoding='utf-8'))
        if (manifest.get('model_sha256') != _file_hash(paths[0]) or
                manifest.get('predictions_sha256') != _file_hash(paths[2])):
            raise ValueError('Cached LightAutoML final pipeline artifact hash differs')
        bundle = joblib.load(paths[0])
        if (bundle.get('backend') != 'lightautoml_v1' or bundle.get('state') != config['task'] or
                bundle.get('recipe') != config['recipe'] or
                bundle.get('selected') != json.loads(paths[1].read_text(encoding='utf-8'))):
            raise ValueError('Saved final LightAutoML pipeline differs from the frozen plan')
        saved, predicted = np.load(paths[2]), self._predict(bundle, test)
        if (saved.shape != (len(test),) or predicted.shape != saved.shape or not np.isfinite(saved).all() or
                not np.allclose(saved, predicted, rtol=1e-10, atol=1e-10)):
            raise ValueError('Cached LightAutoML final pipeline changed predictions')
        return bundle, predicted

    def finalize_temporal(self, suite_id, mode=None):
        self._check_id(suite_id)
        directory = self.run / 'temporal' / suite_id
        identity = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        if identity != self._identity(identity['config']):
            raise ValueError('Data or LightAutoML implementation changed; run a new suite')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        config = identity['config']
        explicit_mode = mode is not None
        mode = report['selected_mode'] if mode is None else mode
        if (not isinstance(mode, str) or mode not in config['modes'] or
                sum(row.get('mode') == mode for row in report['results']) != 1):
            raise ValueError('Choose a completed LightAutoML mode present in the temporal report')
        train, test = self.engine.load()
        if not (test[config['time_column']] > train[config['time_column']].max()).all():
            raise ValueError('LightAutoML final export requires strictly future test periods')
        if explicit_mode:
            from temporal_ensemble import _read_member, _member_predictions
            _member_predictions(_read_member(self, suite_id, mode),
                temporal_folds(train, config['time_column'], config['validation_periods']), train, config['task']['target'])
        suffix = '__' + mode if explicit_mode else ''
        model_path = directory / ('final_model' + suffix + '.joblib')
        submission_path = directory / ('submission' + suffix + '.csv')
        if mode == 'reference':
            _, _, _, reference_worker = self._reference(config['reference_suite_id'])
            previous = reference_worker.finalize_temporal(config['reference_suite_id'], mode=config['history_mode'])
            shutil.copy2(previous['model_path'], model_path)
            shutil.copy2(previous['submission_path'], submission_path)
            note, selected = 'Unchanged reference won; exact reference pipeline export, no LightAutoML fit', None
        else:
            bundle, pred = self._ready_final(directory, config, test, mode=mode)
            self._save_checked(bundle, model_path, test, pred)
            output = test[config['task']['ids']].copy()
            output[config['task']['target']] = pred
            output.to_csv(submission_path, index=False)
            selected = bundle['selected']
            note = ('Saved final pipeline; no fitting during export. Full-past fixed refit: ' +
                    str(bool(selected.get('full_train_refit'))) + '; historical features use strictly past targets.')
        result = dict(suite_id=suite_id, mode=mode, rows=len(test), iterations=selected,
            submission_path=str(submission_path), model_path=str(model_path),
            prediction_bounds=prediction_bounds(config['task']), note=note)
        write_json(directory / ('finalized' + suffix + '.json'), result)
        return result


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--fit':
        raise ValueError('Private LightAutoML worker requires --fit request.joblib result.joblib')
    fit_child(sys.argv[2], sys.argv[3])
