"""Freeze and sequentially compare the 2157.225 control with isolated regression.

Preparation/reporting never call an LLM or train a model. --run explicitly starts
Qwen with the caller's environment; keys are never written to the manifest.
"""
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'outputs/regression_comparison_20261025'
INPUTS = ('train.csv', 'test.csv', 'data_dictionary.csv', 'sample_submission.csv')
PACKAGES = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost', 'torch', 'openai')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), packages=packages)


def prepare(folder, task, threads=8, time_budget=9900, max_steps=80, step_timeout=1800):
    """Create two immutable code snapshots and a single immutable input snapshot."""
    if folder.exists():
        raise ValueError(f'Comparison directory already exists: {folder}. Use --run/--report or a new --folder.')
    if not 1 <= threads <= 8 or not 0 < time_budget <= 10800 or min(max_steps, step_timeout) <= 0:
        raise ValueError('Invalid CPU/time/step limits')
    task = task.resolve()
    sys.path.insert(0, str(ROOT / 'champion_solution'))
    from solution_torch import task_kind
    from legacy_solution import discover_files
    if task_kind(task, 'regression') != 'regression':
        raise ValueError('This comparison supports regression only')
    paths = {**discover_files(task.parent), 'task.md': task}
    data = {name: Path(path).read_bytes() for name, path in paths.items()}
    metadata = read(ROOT / 'controls/champion_v5_1_2157_225.json')
    archive = ROOT / metadata['archive']
    if digest(archive.read_bytes()) != metadata['sha256']:
        raise ValueError('Frozen control ZIP hash mismatch')
    with ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(metadata['files_sha256']):
            raise ValueError('Frozen control file list differs')
        control = {name: bundle.read(name) for name in bundle.namelist()}
    if {name: digest(value) for name, value in control.items()} != metadata['files_sha256']:
        raise ValueError('Frozen control member hash mismatch')
    candidate = {p.name: p.read_bytes() for p in (ROOT / 'champion_solution').iterdir()
                 if p.is_file() and (p.suffix == '.py' or p.name == 'requirements.txt')}
    from build_champion import validate_payload
    validate_payload(control)
    validate_payload(candidate)
    folder.mkdir(parents=True)
    (folder / 'inputs').mkdir()
    for name, value in data.items():
        (folder / 'inputs' / name).write_bytes(value)
    arms = {}
    for name, contents in (('control', control), ('isolated', candidate)):
        code = folder / f'{name}_code'
        code.mkdir()
        for filename, value in contents.items():
            (code / filename).write_bytes(value)
        args = [str(code / 'solution_torch.py'), str(folder / 'inputs/task.md'),
                '--output-dir', str(folder / name), '--threads', str(threads),
                '--researchers', '1', '--training-jobs', '1', '--reviewer', '--no-analyst',
                '--time-budget', str(time_budget), '--max-steps', str(max_steps),
                '--step-timeout', str(step_timeout), '--require-selection']
        if name == 'isolated':
            args += ['--task-type', 'regression', '--autogluon-budget', '0']
        arms[name] = dict(arguments=args, files={filename: digest(value) for filename, value in contents.items()})
    result = dict(created=datetime.now().isoformat(timespec='seconds'), control_sha256=metadata['sha256'],
                  control_platform_aggregate=metadata['platform_score_user_reported'], environment=environment(),
                  input_hashes={name: digest(value) for name, value in data.items()}, arms=arms,
                  note='Fresh independent searches; no resume/history import. Qwen sampling and runtime can vary.')
    write(folder / 'comparison_manifest.json', result)
    return result


def verify(folder, manifest):
    if environment() != manifest['environment']:
        raise ValueError('Python or ML package versions differ from preparation; use the same interpreter/environment')
    for name, expected in manifest['input_hashes'].items():
        if digest((folder / 'inputs' / name).read_bytes()) != expected:
            raise ValueError(f'Comparison input changed: {name}')
    for arm, details in manifest['arms'].items():
        for name, expected in details['files'].items():
            if digest((folder / f'{arm}_code' / name).read_bytes()) != expected:
                raise ValueError(f'Frozen comparison code changed: {arm}/{name}; prepare a new comparison')


def run_arm(folder, manifest, arm):
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('OPENAI_BASE_URL'):
        raise ValueError('Set OPENAI_API_KEY and OPENAI_BASE_URL in this PowerShell session before --run')
    api = dict(model=os.getenv('OPENAI_MODEL', 'qwen/qwen3.8-27b'), base_url=os.environ['OPENAI_BASE_URL'])
    api_path = folder / 'api_identity.json'
    if api_path.exists() and read(api_path) != api:
        raise ValueError('Qwen model/base URL changed between arms; use the same API configuration')
    attempt = folder / f'{arm}_attempt.json'
    if attempt.exists() or (folder / arm / 'agent_runs').exists():
        raise ValueError(f'{arm} was already attempted; use --run for the other arm or prepare a new --folder')
    write(api_path, api)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', PYTHONPATH='',
               AUTOML_AUTOGLUON_BUDGET='0')
    started = time.monotonic()
    result = dict(started=datetime.now().isoformat(timespec='seconds'), arm=arm, status='running')
    write(attempt, result)
    print(f'\n=== Qwen regression: {arm}; one training process ===', flush=True)
    try:
        with (folder / f'{arm}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable] + manifest['arms'][arm]['arguments'], cwd=ROOT,
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                result.update(status='completed' if process.wait() == 0 else 'failed', returncode=process.returncode)
            except BaseException:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                process.wait()
                raise
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error_type=type(error).__name__)
        raise
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write(attempt, result)
    return result


def temporal_result(folder, arm, manifest):
    """Only published temporal results with exact saved row indices are comparable."""
    import numpy as np
    runs = sorted((folder / arm / 'agent_runs').glob('*/run_summary.json'))
    if not runs:
        return dict(status='not_run', comparable=False)
    if len(runs) != 1:
        raise ValueError(f'Multiple runs in {arm}; fresh independent runs were expected')
    path = runs[0]
    summary = read(path)
    result = {k: summary.get(k) for k in ('status', 'elapsed_seconds', 'final_selection', 'final_output')}
    result.update(run=str(path.parent), comparable=False)
    selection, output = summary.get('final_selection') or {}, summary.get('final_output') or {}
    if summary.get('status') != 'completed' or selection.get('kind') != 'temporal' or not output.get('published'):
        result['reason'] = 'No successfully published temporal winner; holdout scores are not substituted'
        return result
    sid = selection['suite_id']
    if not re.fullmatch('[0-9a-f]{16}', sid):
        raise ValueError('Invalid temporal suite ID')
    directory = path.parent / 'temporal' / sid
    plan, report = read(directory / 'plan.json'), read(directory / 'report.json')
    cfg = plan['config']
    for name in INPUTS:
        if plan['hashes'][name] != manifest['input_hashes'][name]:
            raise ValueError(f'{arm} temporal suite uses different inputs: {name}')
    chosen = next(r for r in report['results'] if r['mode'] == report['selected_mode'])
    if selection['candidate'] != chosen['mode'] or output['mode'] != chosen['mode'] or output['selected_suite'] != sid:
        raise ValueError('Published selection does not match the temporal report')
    if digest((folder / arm / 'submission.csv').read_bytes()) != summary['submission_sha256']:
        raise ValueError(f'{arm} published submission changed after the run')
    with (folder / 'inputs/train.csv').open(encoding='utf-8-sig', newline='') as stream:
        periods = np.array([float(row[cfg['time_column']]) for row in csv.DictReader(stream)])
    rows = []
    for i, period in enumerate(cfg['validation_periods']):
        with np.load(directory / f'fold_{i}.npz', allow_pickle=False) as saved:
            pair = {key: np.asarray(saved[key], dtype='<i8') for key in ('train', 'validation')}
        for values in pair.values():
            if values.ndim != 1 or not len(values) or values.min() < 0 or values.max() >= len(periods) or len(np.unique(values)) != len(values):
                raise ValueError('Invalid temporal row indices')
        if np.intersect1d(pair['train'], pair['validation']).size or not np.all(periods[pair['train']] < float(period)) or not np.all(periods[pair['validation']] == float(period)):
            raise ValueError('Temporal fold is not strictly past-only')
        rows.append(dict(period=period, **{key: digest(values.tobytes()) for key, values in pair.items()}))
    metrics = {k: float(chosen[k]) for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')}
    if not all(math.isfinite(v) and v >= 0 for v in metrics.values()):
        raise ValueError('Invalid RMSE in the temporal report')
    result.update(comparable=True, metrics=metrics, mode=chosen['mode'],
                  folds=[{k: row[k] for k in ('period', 'rmse', 'rows')} for row in chosen['folds']],
                  comparison_contract=dict(task=cfg['task'], time_column=cfg['time_column'],
                      rows=rows, prediction_bounds=summary.get('prediction_bounds')))
    return result


def report(folder, manifest):
    arms = {arm: temporal_result(folder, arm, manifest) for arm in manifest['arms']}
    a, b = arms['control'], arms['isolated']
    comparable = a.get('comparable') and b.get('comparable') and a['comparison_contract'] == b['comparison_contract']
    result = dict(arms=arms, comparable=bool(comparable), delta_isolated_minus_control=None,
                  note='Lower RMSE is better. Development folds are reused for selection; one Qwen run per arm does not establish significance.')
    if comparable:
        result['delta_isolated_minus_control'] = {k: b['metrics'][k] - a['metrics'][k] for k in a['metrics']}
    else:
        result['reason'] = 'Need two completed temporal exports with identical task, inputs, train/validation rows and prediction bounds'
    write(folder / 'comparison_report.json', result)
    for arm, row in arms.items():
        print(f"{arm}: {row.get('metrics', row.get('reason', row['status']))}")
    print(f"Comparable: {bool(comparable)}; delta isolated - control: {result['delta_isolated_minus_control']}")
    print(f"Report: {folder / 'comparison_report.json'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', choices=('control', 'isolated', 'both'))
    action.add_argument('--report', action='store_true')
    parser.add_argument('--folder', type=Path, default=DEFAULT)
    parser.add_argument('--task', type=Path, default=ROOT / 'tasks/regression/task.md')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--time-budget', type=int, default=9900)
    parser.add_argument('--max-steps', type=int, default=80)
    parser.add_argument('--step-timeout', type=int, default=1800)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.prepare:
        prepare(folder, args.task, args.threads, args.time_budget, args.max_steps, args.step_timeout)
        print(f'Prepared: {folder}\nNo API requests or training. Next: python compare_regression.py --run both --folder "{folder}"')
        return
    if any(arg.split('=', 1)[0] in ('--threads', '--time-budget', '--max-steps', '--step-timeout', '--task') for arg in sys.argv[1:]):
        parser.error('Search settings are frozen by --prepare; use a new --folder to change them')
    manifest = read(folder / 'comparison_manifest.json')
    verify(folder, manifest)
    failed = False
    if args.run:
        arms = ('control', 'isolated') if args.run == 'both' else (args.run,)
        for arm in arms:
            if (folder / f'{arm}_attempt.json').exists() or (folder / arm / 'agent_runs').exists():
                raise ValueError(f'{arm} was already attempted; run only the unattempted arm or prepare a new --folder')
        for arm in arms:
            failed |= run_arm(folder, manifest, arm)['status'] != 'completed'
    report(folder, manifest)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
