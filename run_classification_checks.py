"""Run explicit diagnostics against an existing run, keeping its selection/submission intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from classification_checks import ClassificationChecks
from solution import discover_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--recipes', type=int, nargs=2, required=True)
    parser.add_argument('--ablation-recipe', type=int)
    parser.add_argument('--feature-groups', type=Path)
    args = parser.parse_args()
    folder, run = args.task.resolve().parent, args.resume.resolve()
    if not run.is_relative_to(folder / 'agent_runs'):
        raise ValueError('Run must be within task agent_runs')
    lock = folder / '.agent.lock'
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        tracked = [folder/'submission.csv', run/'final_selection.json', run/'research/state.json']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.exists()}
        checks = ClassificationChecks(run/'research', discover_files(folder), folder)
        output = {}
        print('Reblending saved pair', flush=True)
        output['pair'] = checks.reblend_suite(args.suite, args.recipes)
        _, source = checks.source(args.suite)
        if len(source['config']['recipes']) == 3:
            print('Reblending three recipes with pair subsets', flush=True)
            output['triple'] = checks.reblend_suite(args.suite, [r['id'] for r in source['config']['recipes']])
        print('Comparing stress errors', flush=True)
        output['stress'] = checks.stress_errors(args.suite, args.recipes)
        if args.ablation_recipe:
            if not args.feature_groups:
                raise ValueError('Provide --feature-groups JSON for ablation')
            print('Training stress feature ablations', flush=True)
            output['ablation'] = checks.stress_ablation(args.suite, args.ablation_recipe,
                json.loads(args.feature_groups.read_text(encoding='utf-8')))
        print('Training duplicate-group outer validation', flush=True)
        output['grouped'] = checks.grouped_validation(args.recipes, 3)
        for path, digest in before.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise AssertionError('Protected output changed: ' + path)
        path = run/'classification_checks_result.json'
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = {'suites': [{k: v for k, v in output[name].items() if k != 'results'} | {
            'results': [{k: v for k, v in r.items() if k != 'per_class'} for r in output[name]['results']]}
            for name in ('pair', 'triple') if name in output]}
        (run/'classification_checks_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Completed; selection, submission and experiment registry unchanged. Report: ' + str(path), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
