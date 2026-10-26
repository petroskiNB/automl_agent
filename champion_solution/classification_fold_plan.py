"""Audited import of classification split indices, never historical fitted models.

Old training code need not match: only the data, task contract, original saved
indices and their scored predictions are read. New models are trained by the
current trusted backend. Historical holdout scores remain provenance only.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


VERSION = 'frozen_classification_folds_v1'
TASK_KEYS = ('task_type', 'metric', 'target', 'features', 'ids')
REQUIRED_DATA = ('train.csv', 'test.csv', 'sample_submission.csv')


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected an object: ' + str(path))
    return value


def _identifier(value):
    if not isinstance(value, str) or len(value) != 16 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid classification split or suite ID')
    return value


def _task(state):
    result = {key: state[key] for key in TASK_KEYS}
    if result['task_type'] != 'classification' or result['metric'] != 'f1_macro':
        raise ValueError('Frozen classification plans require classification/f1_macro')
    return result


def _data_files(run, files=None):
    if files is None:
        if Path(run).parent.name != 'agent_runs':
            raise ValueError('Supply files for a run outside an agent_runs directory')
        folder = Path(run).parent.parent
        files = {name: folder / name for name in REQUIRED_DATA + ('data_dictionary.csv',)}
    result = {name: Path(files[name]).resolve() for name in REQUIRED_DATA}
    if any(not path.is_file() for path in result.values()):
        raise ValueError('Missing train/test/sample data for the frozen plan')
    if files.get('data_dictionary.csv') and Path(files['data_dictionary.csv']).is_file():
        result['data_dictionary.csv'] = Path(files['data_dictionary.csv']).resolve()
    return result


def _indices(values, rows, name):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values) or
            (values < 0).any() or (values >= rows).any() or len(np.unique(values)) != len(values)):
        raise ValueError('Invalid ' + name + ' indices')
    return values.astype(np.int64, copy=False)


def _partition(outer_path, inner_path, rows):
    with np.load(outer_path, allow_pickle=False) as saved:
        tr = _indices(saved['train'], rows, 'outer train')
        va = _indices(saved['validation'], rows, 'outer validation')
    with np.load(inner_path, allow_pickle=False) as saved:
        inner = _indices(saved['train'], rows, 'inner train')
        calibration = _indices(saved['validation'], rows, 'inner calibration')
    if not np.array_equal(np.sort(np.r_[tr, va]), np.arange(rows)):
        raise ValueError('Outer train/validation must be a disjoint complete partition')
    if not np.array_equal(np.sort(np.r_[inner, calibration]), np.sort(tr)):
        raise ValueError('Inner train/calibration must partition outer training only')
    return tr, va, inner, calibration


def _plan_id(plan):
    content = {key: value for key, value in plan.items() if key != 'split_plan_id'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()[:16]


def _validate_data(plan, state, files):
    if plan['task'] != _task(state):
        raise ValueError('Frozen plan task, target, features or ID contract differs')
    for name, expected in plan['data_hashes'].items():
        if name not in files or _digest(files[name]) != expected:
            raise ValueError('Frozen plan data changed: ' + name)
    train_columns = pd.read_csv(files['train.csv'], nrows=0).columns.tolist()
    test_columns = pd.read_csv(files['test.csv'], nrows=0).columns.tolist()
    if (not set(plan['task']['features'] + plan['task']['ids'] + [plan['task']['target']]).issubset(train_columns)
            or not set(plan['task']['features'] + plan['task']['ids']).issubset(test_columns)):
        raise ValueError('Frozen plan feature/ID columns are absent from the task data')
    frame = pd.read_csv(files['train.csv'], usecols=[plan['task']['target']])
    y = frame[plan['task']['target']].to_numpy()
    if pd.isna(y).any():
        raise ValueError('Missing classification target')
    classes = np.unique(y).tolist()
    if len(y) != plan['rows'] or classes != plan['classes']:
        raise ValueError('Frozen plan row count or class labels differ')
    return y


def load_plan(destination_run, split_plan_id, current_engine_state, files=None):
    """Return (manifest, [(outer_train, outer_val, inner_train, calibration), ...]).

    The optional last tuple is a diagnostic stress fold, excluded from OOF ranking.
    Loading checks local copied artifacts and current data, not current source-code
    hashes, because this contract imports indices and recipe descriptions only.
    """
    run = Path(destination_run).resolve()
    directory = run / 'classification_fold_plans' / _identifier(split_plan_id)
    plan = _read(directory / 'plan.json')
    if (plan.get('version') != VERSION or plan.get('split_plan_id') != split_plan_id or
            _plan_id(plan) != split_plan_id):
        raise ValueError('Frozen classification plan identity changed')
    for name, expected in plan['files_sha256'].items():
        if Path(name).name != name or _digest(directory / name) != expected:
            raise ValueError('Frozen classification plan artifact changed: ' + name)
    _validate_data(plan, current_engine_state, _data_files(run, files))
    splits, coverage = [], np.zeros(plan['rows'], dtype=np.int64)
    for i in range(plan['folds'] + int(plan['has_stress'])):
        split = _partition(directory / f'fold_{i}.npz', directory / f'calibration_{i}.npz', plan['rows'])
        splits.append(split)
        if i < plan['folds']:
            coverage[split[1]] += 1
    if not (coverage == 1).all():
        raise ValueError('Frozen outer folds must cover each row exactly once')
    return plan, splits


def freeze_plan(source_run, source_suite_id, destination_run, current_engine_state,
                files=None, recipe_ids=(5,)):
    """Copy a verified old split plan into an independent run without model reuse.

    Source code hashes are retained as historical provenance, not relabelled as
    current. The old trained artifacts are not imported or loaded. Supplied recipe
    IDs are specifications for new OOF fits; their old scores must not enter the new
    experiment registry as current measurements.
    """
    source_run, destination_run = Path(source_run).resolve(), Path(destination_run).resolve()
    source = source_run / 'classification_oof' / _identifier(source_suite_id)
    original = _read(source / 'plan.json')
    report = _read(source / 'report.json')
    old_state = _read(source_run / 'research' / 'state.json')
    config, task = original['config'], _task(current_engine_state)
    if config.get('backend'):
        raise ValueError('Import splits from an original classification OOF suite')
    if config['task'] != task or _task(old_state) != task:
        raise ValueError('Original and destination task contracts differ')
    if report.get('suite_id') != source_suite_id:
        raise ValueError('Original classification report ID differs')
    folds, has_stress = config['folds'], bool(config.get('shift_id'))
    if folds not in (3, 4, 5):
        raise ValueError('Use three to five frozen outer folds')
    current_files = _data_files(destination_run, files)
    data_hashes = {name: original['hashes'][name] for name in REQUIRED_DATA}
    if 'data_dictionary.csv' in original['hashes']:
        data_hashes['data_dictionary.csv'] = original['hashes']['data_dictionary.csv']
    manifest = dict(version=VERSION, task=task, classes=report['classes'],
        rows=int(pd.read_csv(current_files['train.csv'], usecols=[task['target']]).shape[0]),
        folds=folds, has_stress=has_stress, shift_id=config.get('shift_id'),
        data_hashes=data_hashes, source_run=str(source_run), source_suite_id=source_suite_id,
        source_plan_sha256=_digest(source / 'plan.json'), source_report_sha256=_digest(source / 'report.json'),
        source_state_sha256=_digest(source_run / 'research' / 'state.json'),
        source_code_hashes={name: value for name, value in original['hashes'].items()
                            if name.endswith('.py') or name == 'cpu_threads'},
        files_sha256={}, source_prediction_sha256={}, recipes=[],
        note='Split indices and recipe specifications only; no historical fitted models imported. '
             'Historical holdout scores are provenance, never new measurements. '
             'The frozen outer folds are adaptive development validation, not an untouched test.')
    y = _validate_data(manifest, current_engine_state, current_files)
    classes = np.asarray(manifest['classes'])
    encoded = np.searchsorted(classes, y)
    coverage, copies = np.zeros(len(y), dtype=np.int64), {}
    original_fold_scores, original_oof = [], {}
    for i in range(folds + int(has_stress)):
        fold = source / f'fold_{i}'
        tr, va, inner, calibration = _partition(fold / 'indices.npz', fold / 'calibration_indices.npz', len(y))
        if i < folds:
            coverage[va] += 1
        saved_scores = _read(fold / 'scores.json')
        if saved_scores.get('fold') != i or saved_scores.get('rows') != len(va):
            raise ValueError('Original saved fold metadata differs')
        with np.load(fold / 'predictions.npz', allow_pickle=False) as predictions:
            if set(saved_scores['scores']) != set(predictions.files):
                raise ValueError('Original prediction candidates differ from saved scores')
            for name in predictions.files:
                p = predictions[name]
                if (p.shape != (len(va), len(classes)) or not np.isfinite(p).all() or
                        (p < 0).any() or (p.sum(axis=1) <= 0).any()):
                    raise ValueError('Invalid original class scores')
                measured = f1_score(encoded[va], p.argmax(axis=1), labels=np.arange(len(classes)),
                                    average='macro', zero_division=0)
                if not np.isclose(measured, saved_scores['scores'][name], rtol=0, atol=1e-12):
                    raise ValueError('Original predictions and saved metrics differ')
                if i < folds:
                    original_oof.setdefault(name, np.full(len(y), -1, dtype=np.int64))[va] = p.argmax(axis=1)
        original_fold_scores.append(saved_scores['scores'])
        for dest_name, source_path in ((f'fold_{i}.npz', fold / 'indices.npz'),
                                       (f'calibration_{i}.npz', fold / 'calibration_indices.npz')):
            copies[dest_name] = source_path
            manifest['files_sha256'][dest_name] = _digest(source_path)
        manifest['source_prediction_sha256'][f'fold_{i}'] = _digest(fold / 'predictions.npz')
    if not (coverage == 1).all():
        raise ValueError('Original outer folds must cover each row exactly once')
    reported = {row['candidate']: row for row in report.get('results', [])}
    if set(reported) != set(original_oof) or report.get('selected_candidate') not in reported:
        raise ValueError('Original overall report candidates differ from saved predictions')
    for name, pred in original_oof.items():
        row = reported[name]
        scores = [values[name] for values in original_fold_scores[:folds]]
        if (not np.allclose(scores, row['fold_f1'], rtol=0, atol=1e-12) or
                not np.isclose(np.mean(scores), row['mean_f1'], rtol=0, atol=1e-12) or
                not np.isclose(f1_score(encoded, pred, labels=np.arange(len(classes)), average='macro', zero_division=0),
                               row['oof_f1'], rtol=0, atol=1e-12)):
            raise ValueError('Original overall report differs from its scored OOF predictions')
        if has_stress and not np.isclose(original_fold_scores[-1][name], row['stress_f1'], rtol=0, atol=1e-12):
            raise ValueError('Original stress metric differs from saved predictions')
    if not isinstance(recipe_ids, (list, tuple)) or not recipe_ids or len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError('Choose distinct historical recipe IDs')
    for recipe_id in recipe_ids:
        recipe = next((e for e in old_state.get('experiments', []) if e.get('id') == recipe_id), None)
        if not recipe or recipe.get('algorithm') == 'ensemble' or not isinstance(recipe.get('params'), dict):
            raise ValueError('Historical recipe must be a saved single model')
        code = recipe.get('feature_code', '')
        if not isinstance(code, str):
            raise ValueError('Historical feature code must be a string')
        compile(code, '<historical-feature-specification>', 'exec')
        manifest['recipes'].append(dict(source_experiment_id=recipe_id, algorithm=recipe['algorithm'],
            params=recipe['params'], feature_code=code, historical_holdout_score=recipe.get('score'),
            historical_metric=recipe.get('metric'), note='Specification only; train/evaluate again on frozen folds'))
    copies['source_plan.json'], copies['source_report.json'] = source / 'plan.json', source / 'report.json'
    for name in ('source_plan.json', 'source_report.json'):
        manifest['files_sha256'][name] = _digest(copies[name])
    split_plan_id = _plan_id(manifest)
    manifest['split_plan_id'] = split_plan_id
    destination = destination_run / 'classification_fold_plans' / split_plan_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, source_path in copies.items():
        target = destination / name
        if target.exists() and _digest(target) != manifest['files_sha256'][name]:
            raise ValueError('Refuse to overwrite a changed frozen plan artifact: ' + name)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copy2(source_path, temporary)
            temporary.replace(target)
    target = destination / 'plan.json'
    if target.exists():
        if _read(target) != manifest:
            raise ValueError('Refuse to overwrite a changed frozen plan manifest')
    else:
        temporary = destination / 'plan.json.tmp'
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)
    checked, _ = load_plan(destination_run, split_plan_id, current_engine_state, current_files)
    return checked
