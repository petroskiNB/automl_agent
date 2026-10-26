"""Verify the frozen winner and build the separate development submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def validate_payload(contents):
    """Reject files the platform will silently discard, before publishing a ZIP."""
    if not {'solution.py', 'requirements.txt', 'reference_recipes.py'} <= set(contents):
        raise ValueError('Missing entry point, requirements, or embedded reference recipes')
    for name, data in contents.items():
        if '/' in name or '\\' in name or (name != 'requirements.txt' and not name.endswith('.py')):
            raise ValueError(f'Platform does not transfer this payload member: {name}')
        if name.endswith('.py'):
            compile(data.decode('utf-8-sig'), name, 'exec')


def smoke_platform_payload(contents, autogluon_budget=0, kinds=('regression', 'classification'), flaml_budget=0, extended_research=False):
    """Exercise the actual entry point up to its first mocked LLM call, twice.

    No API requests or model fits. The code directory contains only files the
    platform transfers; task CSVs are generated in separate fixture directories.
    """
    runner = r'''
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import openai

class ReachedFirstLLM(BaseException):
    pass

def stop_before_network(**kwargs):
    assert kwargs['messages'] and kwargs['tools']
    assert ('run_autogluon_suite' in [t['function']['name'] for t in kwargs['tools']]) == bool(expected_budget)
    assert ('run_flaml_suite' in [t['function']['name'] for t in kwargs['tools']]) == bool(flaml_budget)
    import os
    assert int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0')) == expected_budget
    assert int(os.getenv('AUTOML_FLAML_BUDGET', '0')) == flaml_budget
    assert os.getenv('AUTOML_EXTENDED_RESEARCH', '0') == ('1' if expected_extended else '0')
    if expected_extended:
        names = {t['function']['name'] for t in kwargs['tools']}
        assert 'train_torch_tabm' in names
        assert ('run_regression_specialists' if task.parent.name == 'regression' else 'run_classification_experts') in names
    if task.parent.name == 'regression':
        assert 'autogluon' not in kwargs['messages'][0]['content'].lower()
        selector = next(t['function'] for t in kwargs['tools'] if t['function']['name'] == 'select_final_candidate')
        assert 'autogluon_oof' not in selector['parameters']['properties']['kind']['enum']
    raise ReachedFirstLLM()

class MockOpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=stop_before_network))
    def __enter__(self): return self
    def __exit__(self, *args): return False

openai.OpenAI = MockOpenAI
task = Path(sys.argv[1]).resolve()
auto_budget = int(sys.argv[2])
requested_flaml_budget = int(sys.argv[3])
expected_extended = sys.argv[4] == '1'
flaml_budget = (1800 if task.parent.name == 'regression' else 0) if requested_flaml_budget < 0 else requested_flaml_budget
expected_budget = 0 if task.parent.name == 'regression' else (3600 if auto_budget < 0 else auto_budget)
entry = Path.cwd() / 'solution.py'
sys.argv = [str(entry), str(task), '--time-budget', '9900' if auto_budget < 0 or requested_flaml_budget < 0 or auto_budget or flaml_budget else '60',
            '--max-steps', '1', '--no-reviewer', '--no-analyst']
if auto_budget >= 0:
    sys.argv += ['--autogluon-budget', str(auto_budget)]
if requested_flaml_budget >= 0:
    sys.argv += ['--flaml-budget', str(requested_flaml_budget)]
try:
    runpy.run_path(str(entry), run_name='__main__')
except ReachedFirstLLM:
    from research_policy import library
    recipes = library()
    assert {'classification', 'regression'} <= {r['task_type'] for r in recipes}
    first = recipes[0]['reference_id']
    recipes[0]['reference_id'] = 'mutated'
    assert library()[0]['reference_id'] == first
    run = next((task.parent / 'agent_runs').iterdir())
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    assert 'reference_recipes.py' in manifest
    assert 'reference_recipes.json' not in manifest
    assert (run / 'research_progress.json').is_file()
    print('PLATFORM_STARTUP_OK ' + json.dumps({'task': task.parent.name,
          'reached_mock_llm': True, 'autogluon_enabled': bool(expected_budget),
          'autogluon_budget': expected_budget, 'flaml_budget': flaml_budget, 'recipes': len(library())}))
else:
    raise AssertionError('Entry point did not reach the mocked LLM request')
'''
    scratch = ROOT / 'work' / 'platform_payload_checks'
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        payload = root / 'payload'
        payload.mkdir()
        # Deliberately emulate the organizer's transfer filter.
        for name, data in contents.items():
            if '/' not in name and '\\' not in name and (name.endswith('.py') or name == 'requirements.txt'):
                (payload / name).write_bytes(data)
        assert not list(payload.glob('*.json'))
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        env.update(OPENAI_API_KEY='local-startup-check', OPENAI_BASE_URL='http://127.0.0.1:1/v1',
                   PYTHONPATH='', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        for kind in kinds:
            task = root / kind
            task.mkdir()
            (task / 'task.md').write_text(f'{kind}; target y; write submission.csv with id,y.\n', encoding='utf-8')
            labels = [str(i * 2 + 1) if kind == 'regression' else ('A' if i % 2 else 'B') for i in range(12)]
            (task / 'train.csv').write_text('id,period,x,y\n' + ''.join(
                f'{i},{i // 4 + 1},{i / 10},{labels[i]}\n' for i in range(12)), encoding='utf-8')
            (task / 'test.csv').write_text('id,period,x\n12,4,1.2\n13,4,1.3\n', encoding='utf-8')
            (task / 'sample_submission.csv').write_text('id,y\n12,0\n13,0\n', encoding='utf-8')
            (task / 'data_dictionary.csv').write_text(
                'feature,role,description\nid,id,row identifier\nperiod,feature,time period\nx,feature,value\ny,target,label\n',
                encoding='utf-8')
            process = subprocess.run([sys.executable, '-c', runner, str(task / 'task.md'), str(-1 if autogluon_budget is None else autogluon_budget), str(-1 if flaml_budget is None else flaml_budget), str(int(extended_research))], cwd=payload,
                                     env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
            marker = [line for line in process.stdout.splitlines() if line.startswith('PLATFORM_STARTUP_OK ')]
            if process.returncode or len(marker) != 1:
                raise ValueError(f'{kind} platform startup failed:\n{process.stdout[-2000:]}\n{process.stderr[-3000:]}')
            results.append(json.loads(marker[0].split(' ', 1)[1]))
    return results


def verify_control():
    registry = json.loads((ROOT / 'controls/platform_results.json').read_text(encoding='utf-8'))
    path = ROOT / registry.get('active_control', 'controls/pytorch_v1_1862_567.json')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    archive = ROOT / metadata['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Control ZIP differs from the registered winner')
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(metadata['files_sha256']):
            raise ValueError('Control archive contents are invalid')
        for name, digest in metadata['files_sha256'].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError(f'Control member mismatch: {name}')
    return metadata


def build(destination, run_startup_checks=True, extended_research=False):
    metadata = verify_control()
    folder = ROOT / 'champion_solution'
    added = [n for n in ['release_audit.py', 'research_policy.py', 'reference_recipes.py', 'prediction_policy.py',
             'hierarchical_history.py', 'temporal_ensemble.py', 'temporal_search.py', 'autogluon_research.py', 'flaml_research.py',
             'temporal_anchors.py', 'classification_shift_audit.py', 'lightautoml_research.py',
             'lightautoml_refit.py', 'training_contract.py', 'tabm_model.py', 'tabm_vendor.py',
             'num_embeddings_vendor.py', 'classification_experts.py', 'regression_specialists.py', 'extended_research.py',
             'classification_fold_plan.py']
             if n not in metadata['files_sha256']]
    names = list(metadata['files_sha256']) + added
    contents = {name: (folder / name).read_bytes() for name in names}
    if extended_research:
        entry = contents['solution.py'].decode('utf-8-sig')
        anchor = '    main()\n'
        if entry.count(anchor) != 1:
            raise ValueError('Unexpected entry point; cannot enable extended research safely')
        defaults = '''    # This experimental ZIP prioritizes the four new research directions.
    if '--extended-research' not in sys.argv:
        sys.argv.append('--extended-research')
    for flag, value in (('--flaml-budget', '0'), ('--lightautoml-budget', '0'),
                        ('--max-steps', '80'), ('--api-timeout', '180'), ('--api-retries', '4')):
        if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv):
            sys.argv.extend([flag, value])
'''
        contents['solution.py'] = entry.replace(anchor, defaults + anchor).encode('utf-8')
    validate_payload(contents)
    changed = [name for name, digest in metadata['files_sha256'].items()
               if hashlib.sha256(contents[name]).hexdigest() != digest]
    allowed = {'legacy_solution.py', 'solution_torch.py', 'research_team.py', 'torch_backend.py',
               'torch_suites.py', 'temporal_research.py', 'final_output.py', 'solution.py',
               'release_audit.py', 'research_policy.py', 'reference_recipes.py', 'temporal_ensemble.py', 'parallel_research.py',
               'torch_model.py', 'classification_research.py'}
    if set(changed) - allowed:
        raise ValueError(f'Unexpected changes to baseline model code: {set(changed) - allowed}')
    startup_checks = []
    if run_startup_checks:
        if extended_research:
            startup_checks = smoke_platform_payload(contents, autogluon_budget=None, flaml_budget=0, extended_research=True)
        else:
            startup_checks = smoke_platform_payload(contents, autogluon_budget=None, flaml_budget=None)
            startup_checks += smoke_platform_payload(contents, autogluon_budget=0, kinds=('classification',), flaml_budget=0)
            startup_checks += smoke_platform_payload(contents, autogluon_budget=0, kinds=('regression',), flaml_budget=300)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None or archive.namelist() != names:
            raise ValueError('Development archive verification failed')
        for name, data in contents.items():
            if archive.read(name) != data:
                raise ValueError(f'Archive mismatch: {name}')
    result = dict(path=str(destination), bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        baseline_sha256=metadata['sha256'], changed=changed, added=added,
        platform_startup_checks=startup_checks,
        platform_startup_checks_status='passed' if run_startup_checks else 'not_run (--skip-startup-checks)',
        tests_run=False,
        validation_note='Python syntax, root-only payload and archive/control hashes verified; no unit tests, API calls or model training by builder',
        local_defaults=dict(autogluon_budget=0, flaml_budget=0, lightautoml_budget=0),
        platform_defaults=dict(autogluon_budget={'regression': 0, 'classification': 3600, 'unknown': 0},
            flaml_budget={'regression': 0 if extended_research else 1800, 'classification': 0, 'unknown': 0},
            lightautoml_budget=0,
            threads=8, time_budget=9900, max_steps=80 if extended_research else 60, step_timeout=3600,
            extended_research=extended_research, api_timeout=180, api_retries=4,
            classification_selection='controller picks completed comparable OOF suite after API failures',
            regression='isolated temporal/history path; no AutoGluon prompt, tools or dependency check'),
        platform_score=None, status='experimental; no quality measurement yet')
    destination.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-control', action='store_true')
    parser.add_argument('--extended-research', action='store_true',
                        help='Enable the four-direction experimental platform profile; preserve source defaults')
    parser.add_argument('--skip-startup-checks', action='store_true',
                        help='Build without mocked entry-point startup checks; retain syntax/root/hash validation')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/automl_champion_v10_classification_recovery_20261026.zip')
    options = parser.parse_args()
    print(json.dumps({'verified_control_sha256': verify_control()['sha256']} if options.verify_control
                     else build(options.output, run_startup_checks=not options.skip_startup_checks,
                                extended_research=options.extended_research), ensure_ascii=False, indent=2))
