"""Versioned ensemble reanalysis and duplicate-group diagnostics; no automatic publication."""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from classification_research import ClassificationResearch, macro
from research import write_json


def weight_candidates(count):
    result = [np.full(count, 1 / count)] + list(np.eye(count))
    for a, b in itertools.combinations(range(count), 2):
        for value in (.25, .5, .75):
            weights = np.zeros(count)
            weights[a], weights[b] = value, 1 - value
            result.append(weights)
    return result


def decision_rule(probabilities, y, classes):
    def objective(w, m):
        return macro(y, sum(a * b for a, b in zip(w, probabilities)) * m, classes) - .002 * np.mean(np.log(m)**2)
    multipliers = np.ones(len(classes))
    weights = max(weight_candidates(len(probabilities)), key=lambda w: objective(w, multipliers))
    for _ in range(2):
        for k in range(len(classes)):
            best, value = objective(weights, multipliers), multipliers[k]
            for candidate in (.75, 1., 1.25, 1.5):
                trial = multipliers.copy()
                trial[k] = candidate
                score = objective(weights, trial)
                if score > best + 1e-10:
                    best, value = score, candidate
            multipliers[k] = value
    return weights.tolist(), multipliers.tolist()


def duplicate_groups(frame):
    # Factorize actual tuples instead of relying on truncated row hashes.
    return pd.MultiIndex.from_frame(frame).factorize()[0]


class ClassificationChecks(ClassificationResearch):
    def identity(self, config):
        result = super().identity(config)
        result['hashes']['classification_checks.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    def source(self, suite_id):
        self._check_id(suite_id)
        directory = self.run / 'classification_oof' / suite_id
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        worker = self if plan['config'].get('backend') == 'checks_v1' else ClassificationResearch(self.engine.root, self.engine.files, self.folder)
        if plan != worker.identity(plan['config']):
            raise ValueError('Source suite is stale')
        if plan['config']['task'] != self._state():
            raise ValueError('Source task differs')
        return directory, plan

    def reblend_suite(self, source_suite_id, experiment_ids):
        source, plan = self.source(source_suite_id)
        old = plan['config']
        mapping = {r['id']: i for i, r in enumerate(old['recipes'])}
        if not 2 <= len(experiment_ids) <= 3 or len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError('Choose 2..3 distinct recipes from the source suite')
        indices = [mapping[eid] for eid in experiment_ids]
        config = dict(old, recipes=[old['recipes'][i] for i in indices], backend='checks_v1',
                      source_suite_id=source_suite_id, rule='inner-calibration pair subsets and bounded multipliers')
        identity = self.identity(config)
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / 'classification_oof' / sid
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'report.json').exists():
            return json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        write_json(directory / 'plan.json', identity)
        train, _ = self.engine.load()
        classes = np.unique(train[config['task']['target']])
        y = train[config['task']['target']].to_numpy()
        rows, oof, stress = [], {}, {}
        for i in range(config['folds'] + bool(config.get('shift_id'))):
            src, dest = source / f'fold_{i}', directory / f'fold_{i}'
            dest.mkdir(exist_ok=True)
            with np.load(src / 'indices.npz') as split:
                va = split['validation']
            with np.load(src / 'calibration_indices.npz') as split:
                calibration = split['validation']
            cp = [np.load(src / f'calibration_{j}.npy') for j in indices]
            with np.load(src / 'predictions.npz') as predictions:
                outer = [predictions[f'model_{j}'] for j in indices]
            weights, multipliers = decision_rule(cp, y[calibration], classes)
            views = self._views(outer, weights, multipliers)
            row = dict(fold=i, rows=len(va), weights=weights, multipliers=multipliers,
                       scores={name: macro(y[va], p, classes) for name, p in views.items()})
            pipeline = joblib.load(src / 'pipeline.joblib')
            pipeline.update(models=[pipeline['models'][j] for j in indices], weights=weights, multipliers=multipliers)
            joblib.dump(pipeline, dest / 'pipeline.joblib')
            for name in ('indices.npz', 'calibration_indices.npz'):
                shutil.copy2(src / name, dest / name)
            for j, original in enumerate(indices):
                shutil.copy2(src / f'model_{original}.joblib', dest / f'model_{j}.joblib')
                np.save(dest / f'calibration_{j}.npy', cp[j])
            np.savez(dest / 'predictions.npz', **views)
            write_json(dest / 'scores.json', row)
            if i == config['folds']:
                stress = row['scores']
            else:
                rows.append(row)
                for name, pred in views.items():
                    oof.setdefault(name, np.full((len(y), len(classes)), np.nan))[va] = pred
        results = []
        for name, pred in oof.items():
            if not np.isfinite(pred).all():
                raise ValueError('Incomplete OOF predictions')
            scores = [row['scores'][name] for row in rows]
            results.append(dict(candidate=name, mean_f1=float(np.mean(scores)), worst_f1=min(scores),
                fold_f1=scores, oof_f1=macro(y, pred, classes), stress_f1=stress.get(name),
                per_class=classification_report(y, classes[pred.argmax(axis=1)], labels=classes, output_dict=True, zero_division=0)))
        np.savez(directory / 'oof.npz', **oof)
        report = dict(suite_id=sid, selected_candidate=max(results, key=lambda r: r['mean_f1'])['candidate'],
                      results=results, report_path=str(directory / 'report.json'),
                      note='Reused identical fitted models and splits. Rules fit inner calibration only. No publication.')
        write_json(directory / 'report.json', report)
        write_json(self.run / 'classification_checks_summary.json', dict(report, results=[{k: v for k, v in r.items() if k != 'per_class'} for r in results]))
        return report

    def stress_errors(self, suite_id, experiment_ids):
        directory, plan = self.source(suite_id)
        config = plan['config']
        if not config.get('shift_id') or len(experiment_ids) != 2:
            raise ValueError('Requires a stress suite and two recipe IDs')
        mapping = {r['id']: i for i, r in enumerate(config['recipes'])}
        fold = directory / f"fold_{config['folds']}"
        va = np.load(fold / 'indices.npz')['validation']
        frame, _ = self.engine.load()
        frame = frame.iloc[va]
        y = frame[config['task']['target']].to_numpy()
        classes = np.unique(self.engine.load()[0][config['task']['target']])
        predictions = np.load(fold / 'predictions.npz')
        labels = [classes[predictions[f'model_{mapping[e]}'].argmax(axis=1)] for e in experiment_ids]
        results = []
        for eid, predicted in zip(experiment_ids, labels):
            groups = {}
            for column in ('proto', 'service', 'state'):
                if column not in frame:
                    continue
                tmp = pd.DataFrame({'group': frame[column].fillna('<missing>').astype(str).to_numpy(), 'wrong': predicted != y})
                groups[column] = tmp.groupby('group').wrong.agg(['size', 'mean']).sort_values('size', ascending=False).head(20).reset_index().to_dict('records')
            results.append(dict(experiment_id=eid, per_class=classification_report(y, predicted, labels=classes, output_dict=True, zero_division=0),
                                confusion=confusion_matrix(y, predicted, labels=classes).tolist(), groups=groups))
        wrong = [p != y for p in labels]
        report = dict(suite_id=suite_id, rows=len(y), classes=classes.tolist(), results=results,
                      both_wrong=int((wrong[0] & wrong[1]).sum()), only_first_correct=int((~wrong[0] & wrong[1]).sum()),
                      only_second_correct=int((wrong[0] & ~wrong[1]).sum()), report_path=str(directory / 'stress_errors.json'))
        write_json(directory / 'stress_errors.json', report)
        return report

    def grouped_validation(self, experiment_ids, folds=3):
        if not 1 <= len(experiment_ids) <= 3 or len(set(experiment_ids)) != len(experiment_ids) or folds not in (3, 4, 5):
            raise ValueError('Use 1..3 recipes and 3..5 folds')
        state, full = self._state(), self.engine.state()
        recipes = [next(e for e in full['experiments'] if e['id'] == eid) for eid in experiment_ids]
        if any(e['algorithm'] == 'ensemble' for e in recipes):
            raise ValueError('Requires single-model recipes')
        config = dict(task=state, recipes=[{k: r[k] for k in ('id', 'algorithm', 'params', 'feature_code')} for r in recipes],
                      folds=folds, diagnostic='identical-feature-groups', seed=42)
        identity = self.identity(config)
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / 'grouped_validation' / sid
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'report.json').exists():
            return json.loads((directory / 'report.json').read_text())
        write_json(directory / 'plan.json', identity)
        train, _ = self.engine.load()
        y, classes = train[state['target']].to_numpy(), np.unique(train[state['target']])
        groups = duplicate_groups(train[state['features']])
        group_stats = pd.DataFrame(dict(group=groups, label=y)).groupby('group').label.agg(['size', 'nunique'])
        predictions = {str(r['id']): np.full((len(y), len(classes)), np.nan) for r in recipes}
        scores = {str(r['id']): [] for r in recipes}
        for i, (tr, va) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(train, y, groups)):
            if set(groups[tr]) & set(groups[va]):
                raise AssertionError('Group leakage')
            if set(y[tr]) != set(classes):
                raise ValueError('Training fold missing a class; fewer folds or different grouping needed')
            dest = directory / f'fold_{i}'
            dest.mkdir(exist_ok=True)
            np.savez(dest / 'indices.npz', train=tr, validation=va)
            for recipe in recipes:
                key = str(recipe['id'])
                path = dest / f'model_{key}.joblib'
                if path.exists():
                    model = joblib.load(path)
                else:
                    model = self._fit(train.iloc[tr], state, recipe, 300 + i)
                    joblib.dump(model, path)
                p = self._prob(model, train.iloc[va], state, classes)
                predictions[key][va] = p
                scores[key].append(macro(y[va], p, classes))
        if any(not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError('Incomplete grouped OOF')
        np.savez(directory / 'oof.npz', **predictions)
        report = dict(diagnostic_id=sid, unique_groups=len(group_stats),
            duplicate_rows_beyond_first=int(len(train)-len(group_stats)),
            conflicting_label_groups=int((group_stats['nunique'] > 1).sum()),
            rows_in_conflicting_groups=int(group_stats.loc[group_stats['nunique'] > 1, 'size'].sum()),
            results=[dict(experiment_id=int(key), mean_f1=float(np.mean(scores[key])), fold_f1=scores[key],
                          oof_f1=macro(y, p, classes)) for key, p in predictions.items()],
            report_path=str(directory / 'report.json'),
            note='Diagnostic only; identical feature tuples never cross outer folds. Inner early stopping uses random rows within outer train. No publication.')
        write_json(directory / 'report.json', report)
        write_json(self.run / 'grouped_validation_summary.json', report)
        return report

    def stress_ablation(self, suite_id, experiment_id, feature_groups):
        source, plan = self.source(suite_id)
        config = plan['config']
        if not config.get('shift_id') or not 2 <= len(feature_groups) <= 4:
            raise ValueError('Requires stress suite and 2..4 feature groups')
        if any(not isinstance(v, list) or not v or any(not isinstance(c, str) for c in v) for v in feature_groups.values()):
            raise ValueError('Feature groups must contain column names')
        recipe = next(r for r in config['recipes'] if r['id'] == experiment_id)
        # Use the final return only: transformations remain identical apart from removed output columns.
        prefix, marker, suffix = recipe['feature_code'].rpartition('    return X')
        if not marker or suffix.strip():
            raise ValueError('Ablation requires transform ending in return X')
        identity = self.identity(dict(task=config['task'], source_suite_id=suite_id, recipe=recipe,
                                      feature_groups=feature_groups, diagnostic='stress-feature-ablation'))
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / 'stress_ablation' / sid
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'plan.json', identity)
        if (directory / 'report.json').exists():
            return json.loads((directory / 'report.json').read_text())
        train, _ = self.engine.load()
        state = config['task']
        y, classes = train[state['target']].to_numpy(), np.unique(train[state['target']])
        split = np.load(source / f"fold_{config['folds']}/indices.npz")
        tr, va = split['train'], split['validation']
        results = []
        for index, (group, keep) in enumerate(feature_groups.items()):
            drop = sorted(set(c for value in feature_groups.values() for c in value) - set(keep))
            ablated = dict(recipe, feature_code=prefix + f'    X = X.drop(columns={drop!r}, errors="ignore")\n    return X\n')
            path = directory / f'model_{index}.joblib'
            if path.exists():
                model = joblib.load(path)
            else:
                model = self._fit(train.iloc[tr], state, ablated, 300 + config['folds'])
                joblib.dump(model, path)
            p = self._prob(model, train.iloc[va], state, classes)
            np.save(directory / f'predictions_{index}.npy', p)
            results.append(dict(kept_group=group, removed_columns=drop, stress_f1=macro(y[va], p, classes),
                                selected_iterations=model['refit_params']['iterations']))
        report = dict(diagnostic_id=sid, results=results, report_path=str(directory / 'report.json'),
                      note='Diagnostic ablations on reused stress split; needs outer OOF confirmation before selection. No publication.')
        write_json(directory / 'report.json', report)
        write_json(self.run / 'stress_ablation_summary.json', report)
        return report


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    worker = ClassificationChecks(request['root'], request['files'], request['folder'])
    if request['action'] not in {'reblend_suite', 'stress_errors', 'grouped_validation', 'stress_ablation'}:
        raise ValueError('Unknown check')
    result = getattr(worker, request['action'])(**request.get('arguments', {}))
    if 'results' in result:
        result = dict(result, results=[{k: v for k, v in r.items() if k not in {'per_class', 'confusion', 'groups'}} for r in result['results']])
    print(json.dumps(result, ensure_ascii=False))
