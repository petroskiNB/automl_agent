"""Equal-weight temporal ensembles over exactly matching frozen folds."""
import hashlib
import json
import os

import joblib
import numpy as np
from research import write_json
from prediction_policy import prediction_bounds


def _read_member(worker, suite_id, mode=None):
    if not isinstance(suite_id, str) or len(suite_id) != 16 or any(c not in '0123456789abcdef' for c in suite_id):
        raise ValueError('Invalid temporal suite ID')
    folder = worker.run / 'temporal' / suite_id
    identity = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if identity['config'].get('kind') == 'ensemble':
        raise ValueError('Nested temporal ensembles are not supported')
    if identity != worker._identity(identity['config']):
        raise ValueError('Temporal member is stale')
    if report.get('suite_id') != suite_id:
        raise ValueError('Temporal member report ID differs')
    mode = report.get('selected_mode') if mode is None else mode
    candidates = [row for row in report.get('results', []) if row.get('mode') == mode]
    if (not isinstance(mode, str) or not mode or any(c not in 'abcdefghijklmnopqrstuvwxyz_' for c in mode)
            or len(candidates) != 1 or mode not in identity['config'].get('modes', [])):
        raise ValueError('Choose a completed mode present in the temporal member report')
    canonical = (suite_id, mode)
    if identity['config'].get('backend') in {'flaml_v1', 'lightautoml_v1', 'specialists_v1'} and mode == 'reference':
        config = identity['config']
        if report.get('reference_mode') != config['history_mode']:
            raise ValueError('AutoML reference mode differs from its frozen configuration')
        canonical = (config['reference_suite_id'], config['history_mode'])
    return folder, identity, report, mode, canonical


def _member_predictions(member, folds, train, target):
    folder, _, report, mode, _ = member
    result = next(row for row in report['results'] if row['mode'] == mode)
    rows = result.get('folds', [])
    if len(rows) != len(folds):
        raise ValueError('Temporal member must contain every frozen fold')
    from temporal_research import metrics
    predictions = []
    for i, (period, tr, va) in enumerate(folds):
        with np.load(folder / f'fold_{i}.npz', allow_pickle=False) as saved:
            if not np.array_equal(tr, saved['train']) or not np.array_equal(va, saved['validation']):
                raise ValueError('Member fold row indices differ')
        pred = np.load(folder / f'{mode}_fold_{i}.npy', allow_pickle=False)
        if pred.shape != (len(va),) or not np.isfinite(pred).all():
            raise ValueError('Invalid temporal member predictions')
        measured = metrics(train.iloc[va][target], pred)
        if (float(rows[i]['period']) != float(period) or
                not np.isclose(measured['rmse'], rows[i]['rmse'], rtol=0, atol=1e-10)):
            raise ValueError('Temporal member predictions differ from their reported metrics')
        predictions.append(pred)
    return predictions


def blend_suites(worker, suite_ids, member_modes=None):
    lock = worker.run / '.temporal.lock'
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.close(handle)
        return _blend_suites(worker, suite_ids, member_modes)
    finally:
        lock.unlink(missing_ok=True)


def _blend_suites(worker, suite_ids, member_modes=None):
    from temporal_research import metrics, temporal_folds
    if (not isinstance(suite_ids, list) or not 2 <= len(suite_ids) <= 4 or
            any(not isinstance(sid, str) for sid in suite_ids) or len(set(suite_ids)) != len(suite_ids)):
        raise ValueError('Choose two to four distinct non-ensemble temporal suites')
    if member_modes is None:
        member_modes = [None] * len(suite_ids)
    if not isinstance(member_modes, list) or len(member_modes) != len(suite_ids):
        raise ValueError('member_modes must align with suite_ids; use null for a selected mode')
    members = [_read_member(worker, sid, mode) for sid, mode in zip(suite_ids, member_modes)]
    if len({member[4] for member in members}) != len(members):
        raise ValueError('Duplicate temporal pipelines: an AutoML reference aliases its source mode')
    common = members[0][1]['config']
    for _, identity, _, _, _ in members[1:]:
        for key in ('task', 'time_column', 'validation_periods'):
            if identity['config'][key] != common[key]:
                raise ValueError('Temporal ensembles require identical tasks and validation periods')
    config = dict(kind='ensemble', task=common['task'], time_column=common['time_column'],
        validation_periods=common['validation_periods'],
        members=[dict(suite_id=sid, mode=member[3]) for sid, member in zip(suite_ids, members)],
        weights=[1/len(members)]*len(members), selection='Fixed equal weights; no outer-fold weight optimization')
    identity = worker._identity(config)
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = worker.run / 'temporal' / sid
    train, _ = worker.engine.load()
    folds = temporal_folds(train, config['time_column'], config['validation_periods'])
    saved_predictions = [_member_predictions(member, folds, train, config['task']['target']) for member in members]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'report.json').exists():
        return json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    write_json(folder / 'plan.json', identity)
    out = np.full(len(train), np.nan)
    member_oof = [np.full(len(train), np.nan) for _ in members]
    rows = []
    for i, (period, tr, va) in enumerate(folds):
        predictions = []
        for j in range(len(members)):
            pred = saved_predictions[j][i]
            predictions.append(pred)
            member_oof[j][va] = pred
        pred = np.mean(predictions, axis=0)
        out[va] = pred
        rows.append(dict(period=float(period), train_rows=len(tr), **metrics(train.iloc[va][config['task']['target']], pred)))
        np.savez(folder / f'fold_{i}.npz', train=tr, validation=va)
        np.save(folder / f'equal_temporal_ensemble_fold_{i}.npy', pred)
    np.save(folder / 'equal_temporal_ensemble_oof.npy', out)
    mask = np.isfinite(out)
    truth = train.loc[mask, config['task']['target']].to_numpy()
    diversity = []
    for a in range(len(members)):
        for b in range(a+1, len(members)):
            ea, eb = member_oof[a][mask]-truth, member_oof[b][mask]-truth
            correlation = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 1e-12 and eb.std() > 1e-12 else None
            diversity.append(dict(suites=[suite_ids[a], suite_ids[b]],
                                  modes=[members[a][3], members[b][3]], error_correlation=correlation,
                                  prediction_rmse_difference=float(np.sqrt(np.mean((ea-eb)**2)))))
    scores = [r['rmse'] for r in rows]
    result = dict(mode='equal_temporal_ensemble', folds=rows, mean_rmse=float(np.mean(scores)),
        worst_rmse=float(max(scores)), latest_rmse=scores[-1], std_rmse=float(np.std(scores)), pooled=metrics(truth, out[mask]))
    report = dict(suite_id=sid, selected_mode='equal_temporal_ensemble', results=[result],
        report_path=str(folder/'report.json'), prediction_bounds=prediction_bounds(config['task']),
        diversity=diversity, note='Cached identical temporal folds, fixed equal weights. Repeated outer selection remains adaptive development.')
    write_json(folder / 'report.json', report)
    return report


def finalize_ensemble(worker, folder, config, report):
    from temporal_research import temporal_folds
    train, test = worker.engine.load()
    if not (test[config['time_column']] > train[config['time_column']].max()).all():
        raise ValueError('Temporal ensemble export requires future test periods')
    bundles = []
    members = [_read_member(worker, member['suite_id'], member['mode']) for member in config['members']]
    if len({member[4] for member in members}) != len(members):
        raise ValueError('Duplicate temporal pipelines in the ensemble')
    folds = temporal_folds(train, config['time_column'], config['validation_periods'])
    for member in members:
        _member_predictions(member, folds, train, config['task']['target'])
    for member in config['members']:
        finalized = worker.finalize_temporal(member['suite_id'], mode=member['mode'])
        bundles.append(joblib.load(finalized['model_path']))
    bundle = dict(ensemble_models=bundles, weights=config['weights'], state=config['task'])
    pred = worker._predict(bundle, test)
    if pred.shape != (len(test),) or not np.isfinite(pred).all():
        raise ValueError('Invalid temporal ensemble export')
    output = test[config['task']['ids']].copy()
    output[config['task']['target']] = pred
    joblib.dump(bundle, folder/'final_model.joblib')
    output.to_csv(folder/'submission.csv', index=False)
    result = dict(suite_id=report['suite_id'], mode='equal_temporal_ensemble', rows=len(output),
        members=config['members'], submission_path=str(folder/'submission.csv'),
        model_path=str(folder/'final_model.joblib'),
        prediction_bounds=prediction_bounds(config['task']),
        note='Member-specific validated exports, then fixed equal-weight predictions. '
             'LightAutoML reuses its saved inner-holdout pipeline; other members use their own refit protocol.')
    write_json(folder/'finalized.json', result)
    return result
