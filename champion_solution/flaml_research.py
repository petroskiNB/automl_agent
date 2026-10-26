"""Budgeted FLAML regression search inside an existing frozen temporal plan.

Only inner past periods reach AutoML.fit. Each search/refit runs in a bounded
child process because FLAML's fixed-record retrain has no hard time limit.
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

from torch_resources import configure
configure()

import joblib
import numpy as np
import pandas as pd

from prediction_policy import bound_predictions, prediction_bounds
from research import write_json
from temporal_research import TemporalResearch, metrics, temporal_folds


VERSION = '2.3.6'
SUPPORTED_MODES = {'baseline', 'history_exact', 'history_groups', 'history_hierarchy'}
ESTIMATORS = {'lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'}


def categorical_design(frame, levels=None):
    """Freeze categorical codes using training rows, including forest learners."""
    result = frame.copy()
    if levels is None:
        levels = {c: list(pd.unique(result[c].fillna('__MISSING__').astype(str)))
                  for c in result if not pd.api.types.is_numeric_dtype(result[c])}
    for column, categories in levels.items():
        values = result[column].fillna('__MISSING__').astype(str)
        result[column] = pd.Categorical(values, categories=categories)
    return result, levels


def _sanitise(config):
    result = dict(config.get('ml', config))
    for name in ('FLAML_sample_size', 'learner', '_choice_'):
        result.pop(name, None)
    return result


def _best_record(automatic, logfile):
    from flaml.automl.training_log import training_log_reader
    with training_log_reader(str(logfile)) as reader:
        for record in reader.records():
            if (record.learner == automatic.best_estimator and
                    _sanitise(record.config) == automatic.best_config and
                    np.isclose(record.validation_loss, automatic.best_loss, rtol=0, atol=1e-12)):
                config = json.loads(json.dumps(automatic.best_config,
                    default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()))
                return dict(record_id=int(record.record_id), learner=record.learner,
                            config=config, inner_rmse=float(record.validation_loss))
    raise ValueError('The selected FLAML trial was not found in its training log')


def fit_child(request_path, output_path):
    """Private subprocess entry point; it never receives outer validation rows."""
    if importlib.metadata.version('flaml') != VERSION:
        raise ValueError('FLAML 2.3.6 is required')
    from flaml import AutoML, tune
    payload = joblib.load(request_path)
    threads = configure()
    metadata = payload.get('selected')
    if metadata is None:
        search = AutoML(mlflow_logging=False)
        search.fit(X_train=payload['inner_x'], y_train=payload['inner_y'],
            X_val=payload['inner_validation_x'], y_val=payload['inner_validation_y'],
            task='regression', metric='rmse', eval_method='holdout',
            estimator_list=payload['estimators'], time_budget=payload['search_seconds'],
            n_jobs=threads, n_concurrent_trials=1, use_ray=False, use_spark=False,
            ensemble=False, sample=False, retrain_full=False, seed=42, verbose=0,
            mlflow_logging=False, starting_points='static',
            log_file_name=payload['logfile'], log_type='all',
            custom_hp={name: {'n_estimators': {'domain': tune.lograndint(4, 2001),
                'init_value': 32, 'low_cost_init_value': 4}} for name in payload['estimators']})
        metadata = _best_record(search, payload['logfile'])
        del search
    # An exact log record refits the winning recipe once; it does not search.
    model = AutoML(mlflow_logging=False)
    model.retrain_from_log(payload['logfile'], X_train=payload['outer_x'],
        y_train=payload['outer_y'], task='regression', train_full=True,
        record_id=metadata['record_id'], n_jobs=threads, preserve_checkpoint=False)
    joblib.dump(dict(model=model, selected=metadata), output_path)


def _kill_tree(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        psutil.wait_procs(children + [parent], timeout=3)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


class FlamlResearch(TemporalResearch):
    def _identity(self, config):
        # Do not call super: the hybrid worker dispatches this method by backend.
        paths = dict(self.engine.files)
        for name in ('research.py', 'temporal_research.py', 'prediction_policy.py',
                     'hierarchical_history.py', 'temporal_ensemble.py', 'torch_suites.py',
                     'torch_backend.py', 'torch_model.py', 'torch_resources.py',
                     'research_policy.py', 'reference_recipes.py', 'xgb_backend.py',
                     'temporal_search.py', 'flaml_research.py'):
            paths[name] = Path(__file__).with_name(name)
        source = self.run / 'temporal' / config['reference_suite_id']
        for name in ('plan.json', 'report.json'):
            paths['reference/' + name] = source / name
        hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for name, path in paths.items()}
        hashes.update(cpu_threads=str(configure()), flaml_version=VERSION)
        for name in ('tabm_model.py', 'tabm_vendor.py', 'num_embeddings_vendor.py'):
            hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return dict(config=config, hashes=hashes)

    def _reference(self, suite_id):
        self._check_id(suite_id)
        directory = self.run / 'temporal' / suite_id
        identity = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        config = identity['config']
        worker = TemporalResearch(self.engine.root, self.engine.files, self.engine.folder)
        if config.get('backend') or config.get('kind') == 'ensemble' or identity != worker._identity(config):
            raise ValueError('Select a completed, current single run_temporal_suite as reference')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        if report.get('suite_id') != suite_id:
            raise ValueError('Reference report ID differs')
        state = self.engine.state()
        if (state.get('task_type') != 'regression' or state.get('metric') != 'rmse' or
                any(state.get(k) != v for k, v in config['task'].items())):
            raise ValueError('FLAML requires the same regression/RMSE task as its reference')
        if state.get('split') != 'time' or state.get('split_column') != config['time_column']:
            raise ValueError('FLAML temporal comparison requires the frozen time split and the same time column')
        mode = report['selected_mode']
        if mode not in SUPPORTED_MODES:
            raise ValueError('Initial FLAML backend supports baseline/history_exact/history_groups/history_hierarchy; '
                             'select a reference without residual_history or history_gate as winner')
        if not any(row['mode'] == mode for row in report['results']):
            raise ValueError('Reference winner is missing')
        return directory, config, report, worker

    @staticmethod
    def _check_id(suite_id):
        if not isinstance(suite_id, str) or len(suite_id) != 16 or any(c not in '0123456789abcdef' for c in suite_id):
            raise ValueError('Invalid temporal suite ID')

    def _end(self, seconds):
        end = time.monotonic() + seconds
        if os.getenv('AUTOML_WORKER_DEADLINE'):
            end = min(end, time.monotonic() + float(os.environ['AUTOML_WORKER_DEADLINE']) - time.time() - 15)
        return end

    def _child(self, directory, payload, seconds):
        end = time.monotonic() + seconds
        if seconds < 3:
            raise TimeoutError('FLAML time exhausted before a bounded fit/refit')
        job = directory / ('fit_' + str(time.time_ns()))
        request, result, log = job.with_suffix('.joblib'), job.with_suffix('.result.joblib'), job.with_suffix('.txt')
        joblib.dump(payload, request)
        try:
            if end - time.monotonic() < 1:
                raise TimeoutError('FLAML allocation exhausted while preparing fit data')
            with log.open('w', encoding='utf-8') as output:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--fit', str(request), str(result)], stdout=output, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent), env=os.environ.copy())
                try:
                    code = process.wait(timeout=max(.01, end-time.monotonic()))
                except subprocess.TimeoutExpired:
                    _kill_tree(process)
                    raise TimeoutError('FLAML search/refit exceeded its whole-suite allocation; partial logs were kept')
            if code != 0:
                tail = log.read_text(encoding='utf-8', errors='replace')[-4000:]
                raise RuntimeError(f'FLAML child failed (exit {code}): {tail}')
            return joblib.load(result)
        finally:
            # Data payloads are reproducible and large; retain diagnostic logs/models.
            request.unlink(missing_ok=True)
            if result.exists() and 'code' in locals() and code == 0:
                result.unlink()

    def _fit_fold(self, fit, state, config, directory, seconds):
        end = time.monotonic() + seconds
        time_column = config['time_column']
        values = sorted(pd.to_numeric(fit[time_column], errors='raise').unique())
        if len(values) < 2:
            raise ValueError('FLAML needs at least two strictly past periods for its inner holdout')
        inner = fit.loc[fit[time_column] < values[-1]]
        validation = fit.loc[fit[time_column] == values[-1]]
        history = self._history(inner, state, config, config['history_mode'])
        x, _ = self._design(inner, state, config['recipe'], history)
        xv, _ = self._design(validation, state, config['recipe'], history)
        x, inner_levels = categorical_design(x)
        xv, _ = categorical_design(xv, inner_levels)
        full_history = self._history(fit, state, config, config['history_mode'])
        full_x, _ = self._design(fit, state, config['recipe'], full_history)
        full_x, levels = categorical_design(full_x)
        logfile = directory / ('search_' + str(time.time_ns()) + '.jsonl')
        remaining = end-time.monotonic()
        payload = dict(inner_x=x, inner_y=inner[state['target']], inner_validation_x=xv,
            inner_validation_y=validation[state['target']], outer_x=full_x,
            outer_y=fit[state['target']], estimators=config['estimator_list'],
            search_seconds=max(1, remaining * .65 - 3), logfile=str(logfile.resolve()))
        result = self._child(directory, payload, remaining)
        selected = dict(result['selected'], log_file=logfile.name,
                        inner_train_periods=[float(v) for v in values[:-1]],
                        inner_validation_period=float(values[-1]))
        bundle = dict(backend='flaml_v1', model=result['model'], history=full_history,
            recipe=config['recipe'], state=state, levels=levels,
            prediction_bounds=prediction_bounds(state), selected=selected)
        return bundle, selected

    def _predict(self, bundle, frame):
        x, _ = self._design(frame, bundle['state'], bundle['recipe'], bundle['history'])
        x, _ = categorical_design(x, bundle['levels'])
        return bound_predictions(np.asarray(bundle['model'].predict(x)).reshape(-1), bundle['state'])

    def run_flaml_suite(self, reference_suite_id, time_limit=1800, estimator_list=None, feature_code=None):
        cap = int(os.getenv('AUTOML_FLAML_BUDGET', '0'))
        if cap <= 0:
            raise ValueError('Enable FLAML with --flaml-budget first')
        if type(time_limit) is not int or not 120 <= time_limit <= 3600:
            raise ValueError('time_limit must be 120..3600 seconds for the entire suite, including refits')
        estimators = ['lgbm', 'xgboost', 'extra_tree'] if estimator_list is None else estimator_list
        if (not isinstance(estimators, list) or not 1 <= len(estimators) <= 5 or
                any(not isinstance(v, str) or v not in ESTIMATORS for v in estimators) or
                len(set(estimators)) != len(estimators)):
            raise ValueError('Choose distinct FLAML estimators from lgbm/xgboost/rf/extra_tree/catboost')
        if feature_code is not None and not isinstance(feature_code, str):
            raise ValueError('feature_code must be a row-wise transform string or omitted to inherit the reference')
        lock = self.run / '.flaml.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(handle)
        started, charged, budget = time.monotonic(), False, None
        ledger = self.run / 'flaml_budget.json'
        try:
            source, source_config, source_report, reference_worker = self._reference(reference_suite_id)
            inherited = {k: source_config[k] for k in ('task', 'time_column', 'key_columns', 'group_columns',
                'validation_periods', 'key_variants', 'group_hierarchy', 'smoothing', 'area_rounding') if k in source_config}
            config = dict(inherited, backend='flaml_v1', reference_suite_id=reference_suite_id,
                history_mode=source_report['selected_mode'], time_limit=time_limit,
                estimator_list=estimators, modes=['reference', 'flaml'],
                recipe=dict(algorithm='flaml', params={}, feature_code=source_config['recipe'].get('feature_code', '')
                            if feature_code is None else feature_code),
                selection='lowest outer-fold mean RMSE; ties latest, worst, unchanged reference',
                inner_selection='last complete past period; no outer targets; exact log record refit',
                inner_metric='raw RMSE; frozen prediction bounds apply to outer validation and final export',
                search_limits=dict(max_n_estimators=2000, search_fraction=.65, seed=42),
                refit_rule='latest outer fold winning record is reused on all training rows without a new search')
            identity = self._identity(config)
            sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            directory = self.run / 'temporal' / sid
            if (directory / 'report.json').exists():
                return json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            if importlib.metadata.version('flaml') != VERSION:
                raise ValueError('FLAML 2.3.6 is required')
            budget = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else dict(charged_seconds=0, calls=[])
            if budget['charged_seconds'] + time_limit > cap:
                raise ValueError(f'FLAML remaining allocation {cap-budget["charged_seconds"]}s; request a smaller whole-suite budget')
            budget['charged_seconds'] += time_limit
            budget['calls'].append(dict(suite_id=sid, reserved_seconds=time_limit, status='running'))
            write_json(ledger, budget)
            charged = True
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', identity)
            result = self._run(directory, config, source, source_report, reference_worker,
                               self._end(time_limit - (time.monotonic()-started)))
            result.update(suite_id=sid, report_path=str(directory / 'report.json'))
            write_json(directory / 'report.json', result)
            write_json(self.run / 'flaml_summary.json', result)
            budget['calls'][-1]['status'] = 'completed'
            return result
        finally:
            if charged:
                elapsed = min(time_limit, max(1, int(np.ceil(time.monotonic() - started))))
                budget['charged_seconds'] -= time_limit-elapsed
                budget['calls'][-1]['charged_seconds'] = elapsed
                if budget['calls'][-1]['status'] == 'running':
                    budget['calls'][-1]['status'] = 'failed'
                write_json(ledger, budget)
            lock.unlink(missing_ok=True)

    def _run(self, directory, config, source, source_report, reference_worker, end):
        train, test = self.engine.load()
        state, mode = config['task'], source_report['selected_mode']
        folds = temporal_folds(train, config['time_column'], config['validation_periods'])
        rows = {'reference': [], 'flaml': []}
        oof = {name: np.full(len(train), np.nan) for name in rows}
        source_result = next(r for r in source_report['results'] if r['mode'] == mode)
        for i, (period, tr, va) in enumerate(folds):
            with np.load(source / f'fold_{i}.npz') as indices:
                if not np.array_equal(tr, indices['train']) or not np.array_equal(va, indices['validation']):
                    raise ValueError('Reference temporal row indices differ from the frozen plan')
            reference = np.load(source / f'{mode}_fold_{i}.npy')
            if reference.shape != (len(va),) or not np.isfinite(reference).all():
                raise ValueError('Invalid saved reference predictions')
            measured = metrics(train.iloc[va][state['target']], reference)
            previous = source_result['folds'][i]
            if (float(previous['period']) != float(period) or
                    not np.isclose(previous['rmse'], measured['rmse'], rtol=0, atol=1e-10)):
                raise ValueError('Saved reference metrics do not match these validation rows')
            np.savez(directory / f'fold_{i}.npz', train=tr, validation=va)
            fold_dir = directory / f'flaml_fold_{i}'
            fold_dir.mkdir(exist_ok=True)
            started = time.monotonic()
            allocation = (end - time.monotonic() - 3) / (len(folds)-i)
            bundle, selected = self._fit_fold(train.iloc[tr], state, config, fold_dir, allocation)
            pred = self._predict(bundle, train.iloc[va])
            if pred.shape != (len(va),) or not np.isfinite(pred).all():
                raise ValueError('FLAML produced invalid outer predictions')
            model_path = directory / f'flaml_fold_{i}.joblib'
            joblib.dump(bundle, model_path)
            reloaded = self._predict(joblib.load(model_path), train.iloc[va])
            if not np.allclose(reloaded, pred, rtol=1e-10, atol=1e-10):
                raise ValueError('Reloaded FLAML pipeline changed predictions')
            for name, values in (('reference', reference), ('flaml', pred)):
                row = dict(period=float(period), train_rows=len(tr), **metrics(train.iloc[va][state['target']], values))
                if name == 'reference':
                    row.update(iterations=previous['iterations'], source_mode=mode)
                    shutil.copy2(source / f'{mode}_fold_{i}.joblib', directory / f'reference_fold_{i}.joblib')
                else:
                    row.update(iterations=selected, selected_estimator=selected['learner'],
                               seconds=round(time.monotonic()-started, 2))
                rows[name].append(row)
                oof[name][va] = values
                np.save(directory / f'{name}_fold_{i}.npy', values)
                write_json(directory / f'{name}_fold_{i}.json', row)
        results = []
        for name in rows:
            np.save(directory / f'{name}_oof.npy', oof[name])
            scores = [r['rmse'] for r in rows[name]]
            mask = np.isfinite(oof[name])
            results.append(dict(mode=name, folds=rows[name], mean_rmse=float(np.mean(scores)),
                latest_rmse=scores[-1], worst_rmse=float(max(scores)), std_rmse=float(np.std(scores)),
                pooled=metrics(train.loc[mask, state['target']], oof[name][mask]),
                fold_rmse_delta_vs_baseline=[r['rmse']-b['rmse'] for r, b in zip(rows[name], rows['reference'])]))
        winner = min(results, key=lambda r: (r['mean_rmse'], r['latest_rmse'], r['worst_rmse']))['mode']
        coverage = reference_worker._history(train, state, config, 'history_exact').transform(test)
        return dict(selected_mode=winner, results=results, reference_suite_id=config['reference_suite_id'],
            reference_mode=mode, test_key_coverage=float(coverage.hist_exact_seen.mean()),
            prediction_bounds=prediction_bounds(state),
            note='Identical frozen temporal outer rows. FLAML search and preprocessing use an inner past holdout only; '
                 'the selected trial is refit on outer training without further search. These adaptively explored '
                 'outer years are development validation, not an untouched test. Inner search optimizes raw RMSE; '
                 'outer validation and export use frozen prediction bounds. No task submission changes.')

    def finalize_temporal(self, suite_id, mode=None):
        self._check_id(suite_id)
        directory = self.run / 'temporal' / suite_id
        identity = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        if identity != self._identity(identity['config']):
            raise ValueError('Data or FLAML implementation changed; run a new suite')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        config = identity['config']
        explicit_mode = mode is not None
        mode = report['selected_mode'] if mode is None else mode
        if (not isinstance(mode, str) or mode not in {'reference', 'flaml'} or
                sum(row.get('mode') == mode for row in report['results']) != 1):
            raise ValueError('Choose a completed FLAML mode present in the temporal report')
        suffix = '__' + mode if explicit_mode else ''
        model_path = directory / ('final_model' + suffix + '.joblib')
        submission_path = directory / ('submission' + suffix + '.csv')
        source, _, _, reference_worker = self._reference(config['reference_suite_id'])
        train, test = self.engine.load()
        if not (test[config['time_column']] > train[config['time_column']].max()).all():
            raise ValueError('FLAML final export requires strictly future test periods')
        if explicit_mode:
            from temporal_ensemble import _read_member, _member_predictions
            _member_predictions(_read_member(self, suite_id, mode),
                temporal_folds(train, config['time_column'], config['validation_periods']), train, config['task']['target'])
        if mode == 'reference':
            previous = reference_worker.finalize_temporal(config['reference_suite_id'], mode=config['history_mode'])
            shutil.copy2(previous['model_path'], model_path)
            shutil.copy2(previous['submission_path'], submission_path)
            note = 'Unchanged reference won; exact reference pipeline export, no FLAML refit'
            selected = None
        elif mode == 'flaml':
            latest = len(config['validation_periods'])-1
            row = json.loads((directory / f'flaml_fold_{latest}.json').read_text(encoding='utf-8'))
            selected = row['iterations']
            history = self._history(train, config['task'], config, config['history_mode'])
            x, _ = self._design(train, config['task'], config['recipe'], history)
            x, levels = categorical_design(x)
            payload = dict(selected=selected, logfile=str((directory / f'flaml_fold_{latest}' /
                selected['log_file']).resolve()), outer_x=x, outer_y=train[config['task']['target']])
            remaining = self._end(1800)-time.monotonic()
            fitted = self._child(directory, payload, remaining)
            bundle = dict(backend='flaml_v1', model=fitted['model'], history=history,
                recipe=config['recipe'], state=config['task'], levels=levels,
                prediction_bounds=prediction_bounds(config['task']), selected=selected)
            pred = self._predict(bundle, test)
            if pred.shape != (len(test),) or not np.isfinite(pred).all():
                raise ValueError('Invalid final FLAML predictions')
            joblib.dump(bundle, model_path)
            if not np.allclose(self._predict(joblib.load(model_path), test), pred,
                               rtol=1e-10, atol=1e-10):
                raise ValueError('Final FLAML reload changed predictions')
            output = test[config['task']['ids']].copy()
            output[config['task']['target']] = pred
            output.to_csv(submission_path, index=False)
            note = 'Full-train refit of the latest outer fold exact winning log record; no new search'
        else:
            raise ValueError('Unknown FLAML temporal candidate')
        result = dict(suite_id=suite_id, mode=mode, rows=len(test), iterations=selected,
            submission_path=str(submission_path), model_path=str(model_path),
            prediction_bounds=prediction_bounds(config['task']), note=note)
        write_json(directory / ('finalized' + suffix + '.json'), result)
        return result


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--fit':
        raise ValueError('This private FLAML worker requires --fit request.joblib result.joblib')
    fit_child(sys.argv[2], sys.argv[3])
