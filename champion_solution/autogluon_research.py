"""Bounded AutoGluon classification comparison on an existing outer OOF plan."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

from classification_research import ClassificationResearch, fit_decision_rule, macro
from research import write_json
from torch_resources import configure


def model_hyperparameters(profile):
    if profile not in ('boosting', 'diverse'):
        raise ValueError('model_profile must be boosting or diverse')
    result = {'GBM': {}, 'CAT': {}, 'XGB': {}}
    if profile == 'diverse':
        # Give the new families a chance before familiar boosters consume the fit.
        result.update(RF={'n_estimators': 300, 'ag_args': {'priority': 105},
                          'ag_args_fit': {'max_time_limit_ratio': .15}},
                      XT={'n_estimators': 300, 'ag_args': {'priority': 100},
                          'ag_args_fit': {'max_time_limit_ratio': .15}},
                      NN_TORCH={'num_epochs': 80, 'ag_args': {'priority': 110},
                                'ag_args_fit': {'max_time_limit_ratio': .25}})
    return result


def normalise(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid class scores')
    sums = values.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError('Empty class scores')
    return values / sums


def views(reference, automatic, weights, multipliers):
    mixed = [normalise(reference), normalise(automatic)]
    weighted = sum(w * p for w, p in zip(weights, mixed))
    return dict(reference=reference, autogluon=mixed[1], equal_ensemble=np.mean(mixed, axis=0),
                weighted_ensemble=weighted, macro_decision=weighted * np.asarray(multipliers))


def compact(report):
    summary = {k: v for k, v in report.items() if k != 'folds'}
    summary['results'] = [{k: v for k, v in row.items() if k != 'per_class'} for row in report['results']]
    return summary


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(time.time_ns()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _aggregate(config, rows, oof, y, classes):
    """Integer labels avoid repeatedly sorting long string arrays during reporting."""
    encoded = np.searchsorted(classes, y)
    labels = np.arange(len(classes))
    results = []
    for name, probabilities in oof.items():
        scores = [row['scores'][name] for row in rows]
        prediction = probabilities.argmax(axis=1)
        results.append(dict(candidate=name, mean_f1=float(np.mean(scores)), worst_f1=float(min(scores)),
            std_f1=float(np.std(scores)), fold_f1=scores,
            oof_f1=float(f1_score(encoded, prediction, labels=labels, average='macro', zero_division=0)),
            per_class=classification_report(encoded, prediction, labels=labels,
                target_names=[str(c) for c in classes], output_dict=True, zero_division=0)))
    return dict(selected_candidate=max(results, key=lambda row: row['mean_f1'])['candidate'],
        results=results, folds=rows, classes=classes.tolist(), reference_suite_id=config['reference_suite_id'],
        model_profile=config.get('model_profile', 'boosting'),
        requested_model_families=list(model_hyperparameters(config.get('model_profile', 'boosting'))),
        note='Identical outer rows. AutoGluon fits receive training subsets only; blend rules fit inner calibration only. '
             'Previously searched reference and adaptive outer selection are not an untouched test. Shift stress not evaluated here.')


def _saved_fold(worker, source, directory, config, source_report, i, y, classes):
    """Validate a complete checkpoint against its exact original reference fold."""
    src, dest = source / f'fold_{i}', directory / f'fold_{i}'
    required = ('indices.npz', 'calibration_indices.npz', 'predictions.npz',
                'calibration_predictions.npz', 'pipeline.joblib', 'scores.json')
    if any(not (dest / name).is_file() for name in required):
        raise ValueError(f'Incomplete AutoGluon checkpoint fold_{i}')
    with np.load(src / 'indices.npz', allow_pickle=False) as expected, np.load(dest / 'indices.npz', allow_pickle=False) as saved:
        tr, va = np.asarray(saved['train']), np.asarray(saved['validation'])
        if not np.array_equal(tr, expected['train']) or not np.array_equal(va, expected['validation']):
            raise ValueError('Saved AutoGluon outer indices differ from reference')
    with np.load(src / 'calibration_indices.npz', allow_pickle=False) as expected, np.load(dest / 'calibration_indices.npz', allow_pickle=False) as saved:
        inner, calibration = np.asarray(saved['train']), np.asarray(saved['validation'])
        if not np.array_equal(inner, expected['train']) or not np.array_equal(calibration, expected['validation']):
            raise ValueError('Saved AutoGluon calibration indices differ from reference')
    if (not np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(y))) or
            not np.array_equal(np.sort(np.r_[inner, calibration]), np.sort(tr))):
        raise ValueError('Saved AutoGluon fold boundaries are not disjoint nested partitions')
    pipeline = joblib.load(dest / 'pipeline.joblib')
    if pipeline.get('reference', {}).get('classes') != classes.tolist():
        raise ValueError('Saved AutoGluon reference class order differs')
    weights, multipliers = np.asarray(pipeline['weights']), np.asarray(pipeline['multipliers'])
    if (weights.shape != (2,) or not np.isfinite(weights).all() or (weights < 0).any() or
            not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0) or
            multipliers.shape != (len(classes),) or not np.isfinite(multipliers).all() or (multipliers <= 0).any()):
        raise ValueError('Invalid saved AutoGluon calibration rule')
    row = json.loads((dest / 'scores.json').read_text(encoding='utf-8'))
    if (row.get('fold') != i or row.get('rows') != len(va) or
            not np.array_equal(row.get('weights'), weights) or
            not np.array_equal(row.get('multipliers'), multipliers)):
        raise ValueError('Saved AutoGluon score metadata differs from its pipeline')
    name = pipeline['auto_path']
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Invalid saved AutoGluon predictor path')
    predictor = dest / name
    for relative in ('predictor.pkl', 'learner.pkl', 'models/trainer.pkl'):
        if not (predictor / relative).is_file():
            raise ValueError(f'Missing saved AutoGluon predictor artifact: {relative}')
    if not row.get('models') or any(not isinstance(model, str) or Path(model).name != model or
            not (predictor / 'models' / model / 'model.pkl').is_file() for model in row['models']):
        raise ValueError('Saved AutoGluon predictor model files are incomplete')
    candidate = source_report['selected_candidate']
    with np.load(src / 'predictions.npz', allow_pickle=False) as saved:
        reference = saved[candidate]
    with np.load(dest / 'predictions.npz', allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in saved.files}
    expected = views(reference, predictions['autogluon'], weights, multipliers)
    if set(predictions) != set(expected) or set(row['scores']) != set(expected):
        raise ValueError('Saved AutoGluon candidate set differs')
    for key, probabilities in predictions.items():
        if (probabilities.shape != (len(va), len(classes)) or not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or (probabilities.sum(axis=1) <= 0).any() or
                not np.allclose(probabilities, expected[key], rtol=1e-12, atol=1e-12)):
            raise ValueError('Saved AutoGluon predictions differ from calibrated views')
        encoded = np.searchsorted(classes, y[va])
        score = f1_score(encoded, probabilities.argmax(axis=1), labels=np.arange(len(classes)),
                         average='macro', zero_division=0)
        if not np.isclose(score, row['scores'][key], atol=1e-12, rtol=0):
            raise ValueError('Saved AutoGluon fold score differs from predictions')
    # Calibration arrays must use the same class order and preserve the source scores.
    with np.load(dest / 'calibration_predictions.npz', allow_pickle=False) as saved:
        cal_views = {key: saved[key] for key in saved.files}
    source_pipeline = joblib.load(src / 'pipeline.joblib')
    cp = [np.load(src / f'calibration_{j}.npy', allow_pickle=False) for j in range(len(source_pipeline['models']))]
    reference_cal = worker._views(cp, source_pipeline['weights'], source_pipeline['multipliers'])[candidate]
    expected_cal = views(reference_cal, cal_views['autogluon'], weights, multipliers)
    if set(cal_views) != set(expected_cal) or any(
            cal_views[key].shape != (len(calibration), len(classes)) or
            not np.isfinite(cal_views[key]).all() or
            not np.allclose(cal_views[key], expected_cal[key], rtol=1e-12, atol=1e-12)
            for key in expected_cal):
        raise ValueError('Saved AutoGluon calibration predictions differ')
    return row, predictions, va


def recover_completed_suite(worker, suite_id):
    """Recover using an EXACT matching runtime, including an archived frozen worker.

    No hash allowlist or identity migration is performed. The original plan is
    untouched. A frozen worker may be supplied by an isolated recovery script.
    """
    worker._check_id(suite_id)
    directory = worker.run / 'autogluon_oof' / suite_id
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    if plan != worker.identity(plan['config']):
        raise ValueError('Recovery requires the exact original runtime/data identity; use a verified frozen code snapshot')
    config = plan['config']
    if config.get('backend') != 'autogluon_v1' or config.get('folds') not in (3, 4, 5) or config['task'] != worker._state():
        raise ValueError('Recovery requires a matching AutoGluon classification task')
    source, original, source_report, _ = worker.source(config['reference_suite_id'])
    if config['folds'] != original['folds']:
        raise ValueError('Recovered AutoGluon and reference fold counts differ')
    train, _ = worker.engine.load()
    y = train[config['task']['target']].to_numpy()
    classes, rows, oof = np.unique(y), [], {}
    coverage = np.zeros(len(train), dtype=int)
    for i in range(config['folds']):
        row, predictions, va = _saved_fold(worker, source, directory, config, source_report, i, y, classes)
        coverage[va] += 1
        rows.append(row)
        for name, probabilities in predictions.items():
            oof.setdefault(name, np.full((len(train), len(classes)), np.nan))[va] = probabilities
    if not (coverage == 1).all() or any(not np.isfinite(p).all() for p in oof.values()):
        raise ValueError('Recovery requires complete exactly-once OOF coverage')
    # Hash every copied native file: no weights are refit or silently substituted.
    artifacts = {}
    base = directory.resolve()
    for fold in sorted(directory.glob('fold_*')):
        for path in sorted(fold.rglob('*')):
            if path.is_file():
                if not path.resolve().is_relative_to(base):
                    raise ValueError('AutoGluon recovery artifact escapes its suite directory')
                artifacts[str(path.relative_to(directory)).replace('\\', '/')] = _digest(path)
    report = _aggregate(config, rows, oof, y, classes)
    report.update(suite_id=suite_id, report_path=str(directory / 'report.json'), recovered=True,
                  recovery_audit_path=str(directory / 'recovery.json'))
    audit = dict(kind='completed_autogluon_fold_recovery_v1', original_plan_sha256=_digest(directory / 'plan.json'),
        original_identity=plan, recovery_implementation_sha256=_digest(__file__),
        artifacts_sha256=artifacts, folds=config['folds'], rows=len(y), training_performed=False,
        note='Original plan and frozen pipeline choices preserved. Complete saved predictions remeasured; no model fit, blend optimization or new label selection.')
    if (directory / 'report.json').exists():
        previous = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        if previous.get('results') != report['results'] or previous.get('selected_candidate') != report['selected_candidate']:
            raise ValueError('Existing AutoGluon report conflicts with recovered measurements')
    temporary = directory / ('.recovery_oof_' + str(time.time_ns()) + '.npz')
    np.savez(temporary, **oof)
    temporary.replace(directory / 'oof.npz')
    _atomic_json(directory / 'recovery.json', audit)
    _atomic_json(directory / 'report.json', report)
    _atomic_json(worker.run / 'autogluon_summary.json', compact(report))
    budget_path = worker.run / 'autogluon_budget.json'
    if budget_path.exists():
        budget = json.loads(budget_path.read_text(encoding='utf-8'))
        for call in budget.get('calls', []):
            if call.get('suite_id') == suite_id:
                call['status'] = 'completed_recovered'
                call['charged_seconds'] = call.get('charged_seconds', call.get('reserved_seconds', 0))
                call['recovery_training_seconds'] = 0
        _atomic_json(budget_path, budget)
    return compact(report)


class AutoGluonResearch(ClassificationResearch):
    def recover_autogluon_suite(self, suite_id):
        return recover_completed_suite(self, suite_id)

    def identity(self, config):
        result = super().identity(config)
        result['hashes']['autogluon_research.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source = self.run / 'classification_oof' / config['reference_suite_id']
        for name in ('plan.json', 'report.json'):
            result['hashes']['reference/' + name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        return result

    def source(self, suite_id):
        self._check_id(suite_id)
        source = self.run / 'classification_oof' / suite_id
        plan = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
        worker = ClassificationResearch(self.engine.root, self.engine.files, self.folder)
        if plan['config'].get('backend') or plan != worker.identity(plan['config']):
            raise ValueError('Use a completed, current run_classification_suite as reference')
        if plan['config']['task'] != self._state():
            raise ValueError('Reference task differs')
        report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
        return source, plan['config'], report, worker

    def _features(self, frame, state, code):
        return self.engine.prepare(frame, state, code)[0]

    def _fit_auto(self, frame, state, path, code, seconds, bag_folds, stack_levels, model_profile='boosting'):
        from autogluon.tabular import TabularPredictor
        if importlib.metadata.version('autogluon.tabular') != '1.3.1':
            raise ValueError('AutoGluon 1.3.1 is required for this backend')
        features = self._features(frame, state, code).copy()
        features[state['target']] = frame[state['target']].to_numpy()
        predictor = TabularPredictor(label=state['target'], problem_type='binary' if frame[state['target']].nunique() == 2 else 'multiclass',
                                    eval_metric='f1_macro', path=str(path), verbosity=1)
        # All feature processing, bagging and model selection see this frame only.
        # No outer or calibration labels are passed as tuning_data.
        predictor.fit(train_data=features, time_limit=seconds,
            hyperparameters=model_hyperparameters(model_profile),
            presets='best_quality' if model_profile == 'diverse' else 'medium_quality',
            num_bag_folds=bag_folds, num_bag_sets=1, num_stack_levels=stack_levels,
            ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
            fit_weighted_ensemble=True, dynamic_stacking=False,
            fit_strategy='sequential', num_cpus=configure(), num_gpus=0)
        return predictor

    def _auto_prob(self, predictor, frame, state, code, classes):
        p = predictor.predict_proba(self._features(frame, state, code))
        if set(p.columns) != set(classes):
            raise ValueError('AutoGluon class labels differ from reference')
        p = normalise(p.loc[:, list(classes)].to_numpy())
        if p.shape != (len(frame), len(classes)):
            raise ValueError('Invalid AutoGluon probability shape')
        return p

    def run_autogluon_suite(self, reference_suite_id, time_limit=1800, bag_folds=3,
                           stack_levels=0, feature_code='', model_profile='boosting'):
        model_hyperparameters(model_profile)
        cap = int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0'))
        if cap <= 0:
            raise ValueError('Enable AutoGluon with --autogluon-budget first')
        if type(time_limit) is not int or not 120 <= time_limit <= 3600:
            raise ValueError('time_limit must be 120..3600 seconds for the entire comparison')
        if bag_folds not in (3, 5) or stack_levels not in (0, 1):
            raise ValueError('Use bag_folds 3/5 and stack_levels 0/1')
        lock = self.run / '.autogluon.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(handle)
        budget_path = self.run / 'autogluon_budget.json'
        started, charged, budget = time.monotonic(), False, None
        try:
            source, source_config, report, reference_worker = self.source(reference_suite_id)
            state = self._state()
            if self.engine.state().get('split') != 'random':
                raise ValueError('Initial AutoGluon integration requires random stratified classification, not time/group splits')
            config = dict(backend='autogluon_v1', task=state, reference_suite_id=reference_suite_id,
                folds=source_config['folds'], time_limit=time_limit, bag_folds=bag_folds,
                stack_levels=stack_levels, feature_code=feature_code, model_profile=model_profile,
                selection='highest outer mean Macro F1; exact ties prefer unchanged reference')
            identity = self.identity(config)
            sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            directory = self.run / 'autogluon_oof' / sid
            if (directory / 'report.json').exists():
                return compact(json.loads((directory / 'report.json').read_text(encoding='utf-8')))
            if all((directory / f'fold_{i}/scores.json').exists() for i in range(config['folds'])):
                return self.recover_autogluon_suite(sid)
            budget = json.loads(budget_path.read_text(encoding='utf-8')) if budget_path.exists() else dict(charged_seconds=0, calls=[])
            if budget['charged_seconds'] + time_limit > cap:
                raise ValueError(f'AutoGluon remaining allocation: {cap - budget["charged_seconds"]}s; request a smaller whole-suite time_limit')
            budget['charged_seconds'] += time_limit  # Reservation survives a killed worker.
            budget['calls'].append(dict(suite_id=sid, reserved_seconds=time_limit, status='running'))
            write_json(budget_path, budget)
            charged = True
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', identity)
            end = started + time_limit
            if os.getenv('AUTOML_WORKER_DEADLINE'):
                # The subprocess deadline is a wall timestamp, not a monotonic value.
                end = min(end, time.monotonic() + max(0., float(os.environ['AUTOML_WORKER_DEADLINE']) - time.time()) - 15)
            result = self._run(directory, config, source, report, reference_worker, end)
            result.update(suite_id=sid, report_path=str(directory / 'report.json'))
            write_json(directory / 'report.json', result)
            write_json(self.run / 'autogluon_summary.json', compact(result))
            budget['calls'][-1]['status'] = 'completed'
            return compact(result)
        finally:
            if charged:
                elapsed = min(time_limit, max(1, int(np.ceil(time.monotonic() - started))))
                budget['charged_seconds'] -= time_limit - elapsed
                budget['calls'][-1].update(charged_seconds=elapsed)
                if budget['calls'][-1]['status'] == 'running':
                    budget['calls'][-1]['status'] = 'failed'
                write_json(budget_path, budget)
            lock.unlink(missing_ok=True)

    def _run(self, directory, config, source, source_report, reference_worker, end):
        train, _ = self.engine.load()
        state, code = config['task'], config['feature_code']
        y = train[state['target']].to_numpy()
        classes = np.unique(y)
        folds, rows, oof, coverage = config['folds'], [], {}, np.zeros(len(train), dtype=int)
        reference_candidate = source_report['selected_candidate']
        report_reserve = min(max(90., config['time_limit'] * .1), config['time_limit'] * .5)
        fit_end = end - report_reserve
        for i in range(folds):
            src, dest = source / f'fold_{i}', directory / f'fold_{i}'
            dest.mkdir(exist_ok=True)
            if (dest / 'scores.json').exists():
                row, fold_views, va = _saved_fold(self, source, directory, config, source_report, i, y, classes)
                coverage[va] += 1
                rows.append(row)
                for name, probabilities in fold_views.items():
                    oof.setdefault(name, np.full((len(train), len(classes)), np.nan))[va] = probabilities
                continue
            with np.load(src / 'indices.npz') as data:
                tr, va = data['train'], data['validation']
            with np.load(src / 'calibration_indices.npz') as data:
                inner, calibration = data['train'], data['validation']
            if (len(np.intersect1d(tr, va)) or len(np.intersect1d(inner, calibration)) or
                    not np.array_equal(np.sort(np.r_[inner, calibration]), np.sort(tr)) or
                    not np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(train)))):
                raise ValueError('Reference fold or calibration boundaries are invalid')
            coverage[va] += 1
            pipeline = joblib.load(src / 'pipeline.joblib')
            if list(pipeline['classes']) != classes.tolist():
                raise ValueError('Reference class order differs')
            cp = [np.load(src / f'calibration_{j}.npy') for j in range(len(pipeline['models']))]
            reference_cal = reference_worker._views(cp, pipeline['weights'], pipeline['multipliers'])[reference_candidate]
            with np.load(src / 'predictions.npz') as saved:
                reference_outer = saved[reference_candidate]
            def fit(frame, label, jobs_left):
                allocation = (fit_end - time.monotonic()) / jobs_left
                if allocation < 8:
                    raise TimeoutError('AutoGluon suite time exhausted; existing selected output is preserved')
                path = dest / (label + '_' + str(time.time_ns()))
                return self._fit_auto(frame, state, path, code, allocation,
                                      config['bag_folds'], config['stack_levels'], config.get('model_profile', 'boosting')), path
            calibration_model, _ = fit(train.iloc[inner], 'calibration_predictor', (folds - i) * 2)
            auto_cal = self._auto_prob(calibration_model, train.iloc[calibration], state, code, classes)
            weights, multipliers = fit_decision_rule([normalise(reference_cal), auto_cal], y[calibration], classes)
            del calibration_model
            model, model_path = fit(train.iloc[tr], 'predictor', (folds - i) * 2 - 1)
            automatic = self._auto_prob(model, train.iloc[va], state, code, classes)
            fold_views = views(reference_outer, automatic, weights, multipliers)
            np.savez(dest / 'calibration_predictions.npz',
                     **views(reference_cal, auto_cal, weights, multipliers))
            row = dict(fold=i, rows=len(va), weights=weights, multipliers=multipliers,
                       models=model.model_names(), scores={k: macro(y[va], p, classes) for k, p in fold_views.items()})
            np.savez(dest / 'indices.npz', train=tr, validation=va)
            np.savez(dest / 'calibration_indices.npz', train=inner, validation=calibration)
            np.savez(dest / 'predictions.npz', **fold_views)
            joblib.dump(dict(reference=pipeline, weights=weights, multipliers=multipliers,
                             auto_path=model_path.name), dest / 'pipeline.joblib')
            write_json(dest / 'scores.json', row)
            rows.append(row)
            for name, p in fold_views.items():
                if name not in oof:
                    oof[name] = np.full((len(train), len(classes)), np.nan)
                oof[name][va] = p
            del model
        if not (coverage == 1).all() or any(not np.isfinite(p).all() for p in oof.values()):
            raise ValueError('Outer folds must cover each row exactly once')
        np.savez(directory / 'oof.npz', **oof)
        report = _aggregate(config, rows, oof, y, classes)
        report['time_management'] = dict(whole_suite_seconds=config['time_limit'], report_reserve_seconds=report_reserve)
        return report

    def export_autogluon(self, suite_id):
        self._check_id(suite_id)
        directory = self.run / 'autogluon_oof' / suite_id
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        if plan != self.identity(plan['config']):
            raise ValueError('Stale AutoGluon suite')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        config = plan['config']
        source, _, source_report, worker = self.source(config['reference_suite_id'])
        package = dict(kind='autogluon_oof', artifact_root=str(directory), config=config,
                       candidate=report['selected_candidate'], classes=report['classes'],
                       reference_candidate=source_report['selected_candidate'])
        if package['candidate'] == 'reference':
            result = worker.export_classification(config['reference_suite_id'])
            result.update(kind='autogluon_oof', suite_id=suite_id, candidate='reference',
                          note='Unchanged reference export won this comparison; no additional fitting')
            return result
        _, test = self.engine.load()
        p = self.predict_package(package, test)
        output = test[config['task']['ids']].copy()
        output[config['task']['target']] = np.asarray(report['classes'])[p.argmax(axis=1)]
        output.to_csv(directory / 'submission.csv', index=False)
        joblib.dump(package, directory / 'final_model.joblib')
        return dict(kind='autogluon_oof', suite_id=suite_id, candidate=package['candidate'], rows=len(test),
            submission_path=str(directory / 'submission.csv'), model_path=str(directory / 'final_model.joblib'),
            note='Saved outer fold predictors; no export-time training. Keep the complete suite directory with model metadata.')

    def predict_package(self, package, frame):
        root, config = Path(package['artifact_root']), package['config']
        classes = np.asarray(package['classes'])
        return np.mean([self.predict_fold(root, config, package['reference_candidate'], i, frame, classes)[package['candidate']]
                        for i in range(config['folds'])], axis=0)

    def predict_fold(self, root, config, reference_candidate, i, frame, classes):
        from autogluon.tabular import TabularPredictor
        reference_worker = ClassificationResearch(self.engine.root, self.engine.files, self.folder)
        pipeline = joblib.load(Path(root) / f'fold_{i}/pipeline.joblib')
        reference = reference_worker.predict_package(dict(candidate=reference_candidate,
                                                     folds=[pipeline['reference']]), frame)
        predictor = TabularPredictor.load(str(Path(root) / f'fold_{i}' / pipeline['auto_path']), verbosity=0)
        automatic = self._auto_prob(predictor, frame, config['task'], config['feature_code'], classes)
        return views(reference, automatic, pipeline['weights'], pipeline['multipliers'])
