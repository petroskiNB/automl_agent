"""Isolated training processes; only the parent merges the experiment registry."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time 
import uuid
from concurrent.futures import ThreadPoolExecutor


def available_memory():
    try:
        if os.name == 'nt':
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ('total', 'available', 'page_total', 'page_available', 'virtual', 'virtual_available', 'extended')]
            status = Status()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return 0
            return status.available
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        return 0


def resolve_recipe(arguments, state, role):
    args = copy.deepcopy(arguments)
    base_id = args.pop('base_experiment_id', None)
    if base_id is not None:
        base = next(e for e in state['experiments'] if e['id'] == base_id)
        if base['algorithm'] == 'ensemble':
            raise ValueError('Base recipe must be a single model')
        if args.get('algorithm', base['algorithm']) != base['algorithm']:
            if 'params' not in args or 'feature_code' not in args:
                raise ValueError('Changing algorithm requires explicit params and feature_code')
        else:
            args['params'] = dict(base['params'], **args.get('params', {}))
        args.setdefault('algorithm', base['algorithm'])
        args.setdefault('feature_code', base.get('feature_code', ''))
    if 'feature_code' not in args:
        raise ValueError('Explicit feature_code (empty for raw features) or base_experiment_id required')
    if role == 'features':
        if base_id is None or 'feature_code' not in arguments:
            raise ValueError('Feature experiment requires base_experiment_id and complete feature_code')
        if args['feature_code'] == base.get('feature_code', ''):
            raise ValueError('Feature experiment did not change feature_code')
        if args['algorithm'] != base['algorithm'] or args['params'] != base['params']:
            raise ValueError('Feature branch must keep base algorithm and parameters unchanged')
    return args


def recipe_key(args):
    return json.dumps([args.get('algorithm', 'catboost'), args.get('params', {}),
                       args.get('feature_code', ''), args.get('early_stopping_rounds', 50)], sort_keys=True)


def run_batch(calls, folder, files, log_dir, timeout, jobs=2):
    """Wait for a bounded batch, then commit successes serially in proposal order."""
    from research import write_json
    root = log_dir / 'research'
    state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
    results, pending = {}, []
    seen = {recipe_key(e) for e in state['experiments'] if e['algorithm'] != 'ensemble'}
    jobs = min(jobs, 2)
    # Conservative admission check, not an OS-enforced memory limit.
    if jobs == 2 and available_memory() < 8 * 1024**3:
        jobs = 1
    threads = max(1, min(8, os.cpu_count() or 1) // jobs)
    print(f'Training queue: {len(calls)} proposals, up to {jobs} processes, {threads} CPU threads each', flush=True)
    deadline = time.monotonic() + timeout
    for call in calls:
        try:
            role = call.id.rsplit('_', 1)[-1]
            args = resolve_recipe(json.loads(call.function.arguments), state, role)
            key = recipe_key(args)
            if key in seen:
                raise ValueError('Duplicate recipe: completed or already queued')
            seen.add(key)
            directory = log_dir / 'parallel_jobs' / uuid.uuid4().hex
            directory.mkdir(parents=True)
            write_json(directory / 'state.json', state)
            shutil.copy2(root / 'split.npz', directory / 'split.npz')
            request = dict(root=str(directory), folder=str(folder), files={k: str(v) for k, v in files.items()},
                           arguments=args, threads=threads)
            write_json(directory / 'request.json', request)
            pending.append((call, directory, args, role))
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            results[call.id] = dict(exit_code=1, error=str(error) or 'Unknown base experiment')

    def execute(item):
        call, directory, args, role = item
        env = {k: v for k, v in os.environ.items()
               if not any(w in k.upper() for w in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        env.update({k: str(threads) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        started = time.monotonic()
        with (directory / 'stdout.txt').open('w', encoding='utf-8') as out, (directory / 'stderr.txt').open('w', encoding='utf-8') as err:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return dict(exit_code=-1, error='Batch deadline exhausted before start')
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(directory / 'request.json')],
                                         cwd=folder, env=env, stdout=out, stderr=err, timeout=remaining)
                result = dict(exit_code=process.returncode)
            except subprocess.TimeoutExpired:
                result = dict(exit_code=-1, error='Training exceeded batch deadline')
            except OSError as error:
                result = dict(exit_code=1, error=f'Could not start worker: {type(error).__name__}')
        result.update(seconds=round(time.monotonic()-started, 2), started_at=started,
                      ended_at=time.monotonic(), job_path=str(directory), threads=threads)
        if result['exit_code']:
            result['stderr'] = (directory / 'stderr.txt').read_text(encoding='utf-8')[-4000:]
        return result

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(execute, item) for item in pending]
        for item, future in zip(pending, futures):
            call, directory, args, role = item
            result = future.result()
            if result['exit_code'] == 0:
                try:
                    record = json.loads((directory / 'result.json').read_text(encoding='utf-8'))['experiment']
                    old_id = record['id']
                    state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
                    eid = len(state['experiments']) + 1
                    path = root / f'model_{eid}.joblib'
                    shutil.copy2(record['model_path'], path)
                    shutil.copy2(directory / f'validation_{old_id}.npy', root / f'validation_{eid}.npy')
                    record.update(id=eid, model_path=str(path), researcher=role, worker_threads=threads)
                    best = state['best']
                    improved = best is None or (record['score'] < best['score'] if state['metric'] in ('rmse', 'mae') else record['score'] > best['score'])
                    state['experiments'].append(record)
                    if improved:
                        state['best'] = record
                    write_json(root / 'state.json', state)
                    result.update(experiment=record, improved=improved, published=False)
                except (OSError, ValueError, KeyError) as error:
                    result = dict(exit_code=1, error=f'Could not commit worker result: {error}')
            results[call.id] = result
            with (log_dir / 'parallel_training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(call_id=call.id, jobs=jobs, result=result), ensure_ascii=False) + '\n')
    return results


def worker(request):
    # Resource overrides are process-local; saved bundles retain standard model classes.
    from unittest.mock import patch
    import xgboost as xgb
    from xgb_backend import ExtendedResearch
    from research import write_json
    threads = request['threads']
    original_matrix, original_train = xgb.DMatrix, xgb.train
    class BoundedMatrix(original_matrix):
        def __init__(self, *args, **kwargs):
            kwargs['nthread'] = threads
            super().__init__(*args, **kwargs)
    def bounded_train(params, *args, **kwargs):
        return original_train(dict(params, nthread=threads), *args, **kwargs)
    class IsolatedResearch(ExtendedResearch):
        def publish(self, bundle):
            pass  # Only the controller's finalizer may publish.
        def model(self, state, params, algorithm):
            model = super().model(state, params, algorithm)
            if algorithm == 'catboost':
                model.set_params(thread_count=threads)
            elif algorithm == 'lightgbm':
                model.set_params(n_jobs=threads)
            return model
    engine = IsolatedResearch(request['root'], request['files'], request['folder'])
    with patch.object(xgb, 'DMatrix', BoundedMatrix), patch.object(xgb, 'train', bounded_train):
        result = engine.train(**request['arguments'])
    write_json(Path(request['root']) / 'result.json', result)


if __name__ == '__main__':
    worker(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
