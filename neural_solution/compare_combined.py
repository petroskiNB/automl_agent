"""Retrain a saved tree recipe and a neural candidate on the exact frozen split.

No old scores or trained weights enter the new registry. Optional nested OOF
rechecks both candidates. Never publishes into the original task directory.
"""
# Source checkout delegates to the complete, versioned runtime.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    if not _Path(__file__).with_name("solution.py").exists():
        from runtime import main as _launch
        _sys.argv.insert(1, 'compare')
        _launch()

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--experiment', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--time-budget', type=int, default=1800)
    parser.add_argument('--oof', action='store_true')
    args = parser.parse_args()
    from torch_resources import configure
    configure(args.threads)
    if not 1 <= args.time_budget <= 10800:
        raise ValueError('time-budget must be in 1..10800 seconds')
    os.environ['AUTOML_WORKER_DEADLINE'] = str(time.time() + args.time_budget)
    import numpy as np
    from solution import discover_files, fingerprint
    from solution_torch import stage_task, install_hooks
    from torch_backend import TorchResearch
    from research import write_json
    original = args.source_run.resolve()
    saved = json.loads((original / 'research/state.json').read_text(encoding='utf-8'))
    manifest = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
    task = args.task.resolve()
    files = discover_files(task.parent)
    for name, path in dict(files, task=task).items():
        if manifest.get(name) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError(f'Source data mismatch: {name}')
    recipe = next(e for e in saved['experiments'] if e['id'] == args.experiment)
    if recipe['algorithm'] not in {'catboost', 'lightgbm', 'xgboost'}:
        raise ValueError('Choose a single tree recipe, not an ensemble')
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError('Use a new output directory for this comparison')
    staged = stage_task(task, destination)
    files = discover_files(staged.parent)
    run = destination / 'agent_runs' / 'combined_comparison'
    engine = TorchResearch(run / 'research', files, destination)
    engine.configure(**{k: saved[k] for k in ('target', 'task_type', 'metric', 'split', 'split_column')})
    with np.load(original / 'research/split.npz') as old, np.load(run / 'research/split.npz') as new:
        for key in ('train', 'validation'):
            np.testing.assert_array_equal(old[key], new[key], err_msg='Frozen split differs')
    install_hooks()
    import solution
    write_json(run / 'manifest.json', solution.fingerprint(staged, files))
    write_json(run / 'imported_recipe.json', dict(source_run=str(original), recipe=recipe,
               note='Recipe only; old metrics are historical, all candidates retrained.'))
    result = engine.train('Reproduce previous tree recipe', recipe['params'], recipe['algorithm'],
                          recipe.get('feature_code', ''), recipe.get('early_stopping_rounds', 50))
    print('Tree:', result['experiment']['score'], flush=True)
    params = dict(iterations=args.epochs, width=128, layers=2, batch_size=512,
                  learning_rate=.001, dropout=.1, weight_decay=.0001, seed=42)
    if saved['task_type'] == 'classification':
        params['auto_class_weights'] = 'SqrtBalanced'
    result = engine.train('First neural candidate, raw features', params, 'torch_mlp', '', 8)
    print('MLP:', result['experiment']['score'], flush=True)
    result = engine.blend([1, 2], [.5, .5])
    print('Equal blend:', result['experiment']['score'], flush=True)
    report = dict(source_experiment=args.experiment, split_verified=True,
                  evidence='Development holdout, not independent evidence of leaderboard improvement',
                  experiments=engine.state()['experiments'])
    write_json(destination / 'comparison.json', report)
    if args.oof:
        if saved['task_type'] != 'classification' or saved['metric'] != 'f1_macro':
            raise ValueError('Nested OOF currently supports classification Macro F1')
        from torch_suites import install_suite_engines
        classification, _, _ = install_suite_engines()
        suite = classification.ClassificationResearch(run / 'research', files, destination)
        report['oof'] = suite.run_classification_suite([1, 2], folds=3)
        write_json(destination / 'comparison.json', report)
    print(f'Report: {destination / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
