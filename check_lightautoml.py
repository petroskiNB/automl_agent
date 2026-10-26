"""Read-only LightAutoML 0.4.2 dependency/import check; no fit, data or LLM API."""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys


VERSION = '0.4.2'
CORE = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'lightgbm', 'catboost',
        'xgboost', 'torch', 'optuna', 'statsmodels', 'joblib', 'threadpoolctl')
AUXILIARY = ('autowoe', 'holidays', 'json2html', 'networkx', 'cmaes',
             'SQLAlchemy', 'tabm', 'tabicl', 'poetry-core')


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true',
                        help='Report installed distributions without importing LightAutoML')
    parser.add_argument('--output', type=Path, help='Optional JSON diagnostic output file')
    args = parser.parse_args()
    result = {'python': sys.version.split()[0], 'executable': sys.executable,
              'expected_lightautoml': VERSION, 'lightautoml': installed('lightautoml'),
              'versions': {name: installed(name) for name in CORE + AUXILIARY},
              'training_performed': False, 'llm_api_used': False,
              'metadata_only': args.metadata_only, 'preset_imported': False}
    try:
        if result['lightautoml'] != VERSION:
            raise RuntimeError('LightAutoML 0.4.2 is not installed in this interpreter. '
                               'Prepare .venv_lightautoml with constrained dependencies first.')
        if not args.metadata_only:
            # Imports should not need the network; pretrained algorithms are not invoked.
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
            parameters = inspect.signature(TabularAutoML.fit_predict).parameters
            required = {'train_data', 'roles', 'valid_data', 'cv_iter', 'path_to_save'}
            if not required.issubset(parameters):
                raise RuntimeError('Installed preset has an unexpected fit_predict interface')
            Task('reg', loss='mse', metric='mse')
            sys.path.insert(0, str(Path(__file__).resolve().parent / 'champion_solution'))
            from lightautoml_research import tabular_parameters
            parameters_for_fit = tabular_parameters(['linear_l2', 'lgb', 'cb'], 60, 8)
            TabularAutoML(task=Task('reg', loss='mse', metric='mse'), **parameters_for_fit)
            result.update(preset_imported=True, regression_task_created=True,
                          supported_fit_arguments=sorted(required),
                          candidate_constructed=True, candidate_parameters=parameters_for_fit)
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded)
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
