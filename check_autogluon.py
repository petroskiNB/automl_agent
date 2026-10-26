"""Short local AutoGluon 1.3.1 integration check; no API key or hackathon data."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-limit', type=int, default=45)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 15 <= args.time_limit <= 120 or not 1 <= args.threads <= 8:
        parser.error('Use 15..120 seconds and 1..8 CPU threads for this small check')
    root = Path(__file__).resolve().parent
    output = (args.output_dir or root / 'outputs' / ('autogluon_check_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    if output.exists():
        parser.error('Output directory already exists; choose a new one')
    output.mkdir(parents=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    started = time.perf_counter()
    try:
        from autogluon.tabular import TabularPredictor
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        if importlib.metadata.version('autogluon.tabular') != '1.3.1':
            raise RuntimeError('This check targets platform version AutoGluon 1.3.1')
        values, labels = make_classification(n_samples=600, n_features=8, n_informative=5,
            n_redundant=0, n_classes=3, random_state=42)
        data = pd.DataFrame(values, columns=[f'x{i}' for i in range(8)])
        data['segment'] = np.where(data['x0'] > 0, 'A', 'B')
        data.loc[::17, 'x1'] = np.nan
        data['target'] = np.asarray(['class_' + str(v) for v in labels])
        train, outer = train_test_split(data, test_size=.25, random_state=42, stratify=data['target'])
        model_path = output / 'predictor'
        predictor = TabularPredictor(label='target', problem_type='multiclass', eval_metric='f1_macro',
                                    path=str(model_path), verbosity=1)
        predictor.fit(train_data=train, time_limit=args.time_limit,
            hyperparameters={'GBM': {'num_boost_round': 40, 'num_leaves': 15},
                             'CAT': {'iterations': 40, 'depth': 4},
                             'XGB': {'n_estimators': 40, 'max_depth': 4}},
            num_bag_folds=3, num_bag_sets=1, num_stack_levels=0,
            ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
            fit_weighted_ensemble=True, dynamic_stacking=False,
            fit_strategy='sequential', num_cpus=args.threads, num_gpus=0)
        features = outer.drop(columns='target')
        probabilities = predictor.predict_proba(features)
        classes = list(predictor.class_labels)
        assert list(probabilities.columns) == classes
        assert probabilities.shape == (len(outer), 3)
        assert np.isfinite(probabilities.to_numpy()).all()
        assert np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)
        restored = TabularPredictor.load(str(model_path), verbosity=0)
        assert np.allclose(probabilities.to_numpy(), restored.predict_proba(features).to_numpy())
        names = predictor.model_names()
        for family in ('LightGBM', 'CatBoost', 'XGBoost', 'WeightedEnsemble'):
            assert any(name.startswith(family) for name in names), f'Model family did not complete: {family}'
        result = {'status': 'passed', 'autogluon_version': '1.3.1', 'threads': args.threads,
            'gpu_count': 0, 'train_rows': len(train), 'outer_rows': len(outer),
            'outer_data_passed_to_fit': False, 'bagging_folds': 3, 'models': names,
            'best_model': predictor.model_best, 'reload_predictions_equal': True,
            'synthetic_outer_macro_f1': float(f1_score(outer['target'], predictor.predict(features), average='macro')),
            'versions': {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost')},
            'note': 'Synthetic integration check only; this is not hackathon validation or evidence of a score gain.'}
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        raise
    finally:
        if 'result' in locals():
            result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            (output / 'check_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(result, report=str(output / 'check_result.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
