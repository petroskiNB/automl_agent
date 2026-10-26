"""Recover completed AutoGluon folds using their exact archived runtime, without fitting."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(source_run, archive, destination):
    source_run, archive, destination = map(lambda p: Path(p).resolve(), (source_run, archive, destination))
    if destination.exists():
        raise ValueError('Use a new recovery output directory; existing results are never overwritten')
    expected = '9ed59599e0aca03ebce41fe205d596a43430fe39401b7ca34b43a57cc5620150'
    if digest(archive) != expected:
        raise ValueError('The archived v9 runtime differs from the known submission snapshot')
    source_folder = source_run.parent.parent
    destination.mkdir(parents=True)
    runtime = destination/'frozen_runtime'
    runtime.mkdir()
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if '/' in name or '\\' in name or not (name.endswith('.py') or name == 'requirements.txt'):
                raise ValueError('Unexpected member in root-only runtime snapshot')
            (runtime/name).write_bytes(bundle.read(name))
    for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
        shutil.copy2(source_folder/name, destination/name)
    run = destination/'agent_runs'/'recovered_20261026'
    run.mkdir(parents=True)
    # Mutable metadata is copied. Large saved native estimators are linked to avoid
    # copying gigabytes; recovery never calls fit/save on those estimator files.
    linked, copied = [], 0
    for top in ('research', 'classification_oof', 'autogluon_oof'):
        for path in (source_run/top).rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(source_run)
            target = run/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            native_model = top == 'autogluon_oof' and 'models' in relative.parts and path.suffix == '.pkl'
            if native_model:
                try:
                    os.link(path, target)
                    linked.append(str(relative))
                    continue
                except OSError:
                    pass
            shutil.copy2(path, target)
            copied += 1
    for path in source_run.glob('*.json'):
        shutil.copy2(path, run/path.name)
    # Verify the archived code/data against every original AG identity before
    # accepting any old model. A current-code fingerprint bypass is never used.
    for plan_path in (run/'autogluon_oof').glob('*/plan.json'):
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        for name, expected_hash in plan['hashes'].items():
            if name == 'cpu_threads':
                if expected_hash != '8':
                    raise ValueError('Unexpected original CPU contract')
                continue
            if name.startswith('reference/'):
                path = run/'classification_oof'/plan['config']['reference_suite_id']/name.split('/', 1)[1]
            else:
                path = (runtime if name.endswith('.py') else destination)/name
            if digest(path) != expected_hash:
                raise ValueError('Archived provenance mismatch: ' + name)
    manifest = dict(source_run=str(source_run), archive=str(archive), archive_sha256=expected,
        recovered_run=str(run), runtime=str(runtime), linked_native_model_files=linked,
        copied_files=copied, note='For recovery/export only. Do not train or save native predictors in this linked copy.')
    (destination/'recovery_source.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def recover(manifest):
    # This function is intentionally executed in a fresh subprocess. All pickle
    # classes and source identity methods resolve against the archived runtime.
    runtime, run = Path(manifest['runtime']), Path(manifest['recovered_run'])
    folder = run.parent.parent
    sys.path.insert(0, str(runtime))
    os.environ['AUTOML_CPU_THREADS'] = '8'
    from torch_resources import configure
    configure(8)
    from torch_suites import install_suite_engines
    install_suite_engines()
    from autogluon_research import AutoGluonResearch
    from final_output import FinalOutput
    spec = importlib.util.spec_from_file_location('_current_ag_recovery', ROOT/'champion_solution/autogluon_research.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    files = {name: str(folder/name) for name in ('train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv')}
    worker = AutoGluonResearch(run/'research', files, folder)
    reports = []
    for path in sorted((run/'autogluon_oof').glob('*/plan.json')):
        report = helper.recover_completed_suite(worker, path.parent.name)
        reports.append(report)
        print(json.dumps(dict(recovered_suite=path.parent.name, selected_candidate=report['selected_candidate'],
            results=[{k: r[k] for k in ('candidate', 'mean_f1', 'worst_f1')} for r in report['results']]), ensure_ascii=False), flush=True)
    final = FinalOutput(run/'research', files, folder)
    selection = final.select_best_classification()
    if not selection.get('selected'):
        raise ValueError('No unique comparable recovered OOF plan')
    exported = final.finalize_selected()
    result = dict(selection=selection, export=exported, source=manifest,
                  note='Recovered from complete saved folds with their original code; no fitting or API requests.')
    (folder/'recovery_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(selection=selection['selection'], exported=exported), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--archive', type=Path, default=ROOT/'outputs/automl_champion_v9_extended_20261025.zip')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/champion_classification_recovered_20261026')
    parser.add_argument('--execute-manifest', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_manifest:
        recover(json.loads(args.execute_manifest.read_text(encoding='utf-8')))
    else:
        if not args.source_run:
            parser.error('--source-run is required')
        manifest = prepare(args.source_run, args.archive, args.output_dir)
        path = Path(args.output_dir).resolve()/'recovery_source.json'
        environment = dict(os.environ, PYTHONUNBUFFERED='1', AUTOML_CPU_THREADS='8')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--execute-manifest', str(path)],
                       check=True, env=environment)
