"""One-vs-rest specialists with decisions fitted only on nested calibration rows."""
import hashlib
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report
from classification_research import ClassificationResearch, fit_decision_rule, macro
from autogluon_research import normalise
from research import write_json


def fuse(reference, expert_probabilities, class_indices, strengths):
    result = normalise(reference).copy()
    for index, probability, strength in zip(class_indices, expert_probabilities, strengths):
        result[:, index] = (1 - strength) * result[:, index] + strength * np.clip(probability, 0, 1)
    return normalise(result)


def calibrate(reference, probabilities, indices, y, classes):
    strengths = [0.] * len(indices)
    def score(values):
        return macro(y, fuse(reference, probabilities, indices, values), classes) - .001 * np.mean(np.square(values))
    for _ in range(2):
        for i in range(len(indices)):
            best = score(strengths)
            for value in (.25, .5, .75, 1.):
                trial = strengths.copy()
                trial[i] = value
                measured = score(trial)
                if measured > best + 1e-10:
                    best, strengths = measured, trial
    fused = fuse(reference, probabilities, indices, strengths)
    _, multipliers = fit_decision_rule([fused], y, classes)
    return strengths, multipliers


class ClassificationExperts(ClassificationResearch):
    def source(self, kind, suite_id):
        if kind not in ('classification_oof', 'autogluon_oof'):
            raise ValueError('Use classification_oof or autogluon_oof as expert reference')
        self._check_id(suite_id)
        source = self.run / kind / suite_id
        plan = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
        report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
        if kind == 'autogluon_oof':
            from autogluon_research import AutoGluonResearch
            worker = AutoGluonResearch(self.engine.root, self.engine.files, self.folder)
        else:
            worker = ClassificationResearch(self.engine.root, self.engine.files, self.folder)
            if plan['config'].get('backend'):
                raise ValueError('Use an original classification OOF suite, not nested expert/check suites')
        if worker.identity(plan['config']) != plan or plan['config']['task'] != self._state():
            raise ValueError('Expert reference is stale or has a different task')
        return source, plan['config'], report, worker

    def identity(self, config):
        result = super().identity(config)
        for name in ('classification_experts.py', 'autogluon_research.py'):
            result['hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        source = self.run / config['reference_kind'] / config['reference_suite_id']
        for name in ('plan.json', 'report.json'):
            result['hashes']['reference/' + name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        return result

    def _reference_cal(self, source, kind, candidate, pipeline):
        if kind == 'autogluon_oof':
            with np.load(source / 'calibration_predictions.npz') as saved:
                return np.asarray(saved[candidate])
        probabilities = [np.load(source / f'calibration_{j}.npy') for j in range(len(pipeline['models']))]
        return self._views(probabilities, pipeline['weights'], pipeline['multipliers'])[candidate]

    def _fit_experts(self, frame, state, labels, params, code, seed):
        bundles = []
        for label in labels:
            binary = frame.copy()
            binary[state['target']] = (binary[state['target']] == label).astype(int)
            counts = binary[state['target']].value_counts()
            if len(counts) != 2 or counts.min() < 10:
                bundles.append(None)
                continue
            recipe = dict(algorithm='catboost', params=params, feature_code=code)
            bundles.append(self._fit(binary, state, recipe, seed))
        return bundles

    def _expert_prob(self, bundles, frame, state, fallback, indices):
        fallback = normalise(fallback)
        return [fallback[:, index] if bundle is None else self._prob(bundle, frame, state, [0, 1])[:, 1]
                for bundle, index in zip(bundles, indices)]

    def run_classification_experts(self, reference_suite_id, reference_kind='classification_oof',
                                   class_labels=None, time_limit=1200, auto_class_weights='SqrtBalanced'):
        if type(time_limit) is not int or not 120 <= time_limit <= 2400:
            raise ValueError('time_limit must be 120..2400 seconds for all expert fits')
        if auto_class_weights not in (None, 'Balanced', 'SqrtBalanced'):
            raise ValueError('Unsupported expert class weighting')
        lock = self.run / '.classification.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(handle)
        try:
            source, original, report, _ = self.source(reference_kind, reference_suite_id)
            train, _ = self.engine.load()
            state = self._state()
            classes = np.unique(train[state['target']])
            if class_labels is None:
                class_labels = [v for v in ('Backdoor', 'Analysis', 'DoS') if v in classes]
            if not isinstance(class_labels, list) or not 1 <= len(class_labels) <= 3 or len(set(class_labels)) != len(class_labels):
                raise ValueError('Choose one to three distinct class_labels present in training')
            if any(label not in classes for label in class_labels):
                raise ValueError('Expert class label is absent from the task')
            # Match the reference feature representation, without source target or ID columns.
            if reference_kind == 'autogluon_oof':
                code = original['feature_code']
            else:
                candidate = report['selected_candidate']
                index = int(candidate[6:]) if candidate.startswith('model_') else 0
                code = original['recipes'][index].get('feature_code', '')
            config = dict(backend='experts_v1', task=state, folds=original['folds'],
                          reference_kind=reference_kind, reference_suite_id=reference_suite_id,
                          class_labels=class_labels, time_limit=time_limit, feature_code=code,
                          params=dict(iterations=400, depth=6, learning_rate=.05, l2_leaf_reg=6,
                                      auto_class_weights=auto_class_weights, seed=42))
            identity = self.identity(config)
            sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            directory = self.run / 'classification_oof' / sid
            if (directory / 'report.json').exists():
                return json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', identity)
            end = time.monotonic() + time_limit
            result = self._run(directory, config, source, report, end)
            result.update(suite_id=sid, report_path=str(directory / 'report.json'))
            write_json(directory / 'report.json', result)
            write_json(self.run / 'classification_experts_summary.json', result)
            return result
        finally:
            lock.unlink(missing_ok=True)

    def _run(self, directory, config, source, source_report, end):
        train, _ = self.engine.load()
        state, folds = config['task'], config['folds']
        y = train[state['target']].to_numpy()
        classes, rows, oof = np.unique(y), [], {}
        indices = [list(classes).index(label) for label in config['class_labels']]
        candidate = source_report['selected_candidate']
        coverage = np.zeros(len(train), dtype=int)
        for i in range(folds):
            if time.monotonic() >= end - 10:
                raise TimeoutError('Expert suite exhausted its budget; previous candidates remain available')
            src, dest = source / f'fold_{i}', directory / f'fold_{i}'
            dest.mkdir(exist_ok=True)
            with np.load(src / 'indices.npz') as data:
                tr, va = data['train'], data['validation']
            with np.load(src / 'calibration_indices.npz') as data:
                inner, cal = data['train'], data['validation']
            if (not np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(train))) or
                not np.array_equal(np.sort(np.r_[inner, cal]), np.sort(tr))):
                raise ValueError('Expert reference is not a disjoint nested partition')
            coverage[va] += 1
            pipeline = joblib.load(src / 'pipeline.joblib')
            ref_cal = self._reference_cal(src, config['reference_kind'], candidate, pipeline)
            with np.load(src / 'predictions.npz') as data:
                ref_outer = np.asarray(data[candidate])
            calibration_models = self._fit_experts(train.iloc[inner], state, config['class_labels'],
                config['params'], config['feature_code'], 700 + i)
            cp = self._expert_prob(calibration_models, train.iloc[cal], state, ref_cal, indices)
            strengths, multipliers = calibrate(ref_cal, cp, indices, y[cal], classes)
            models = self._fit_experts(train.iloc[tr], state, config['class_labels'],
                config['params'], config['feature_code'], 800 + i)
            probabilities = self._expert_prob(models, train.iloc[va], state, ref_outer, indices)
            fused = fuse(ref_outer, probabilities, indices, strengths)
            predictions = dict(reference=ref_outer, expert_fusion=fused,
                               macro_decision=fused * np.asarray(multipliers))
            row = dict(fold=i, rows=len(va), strengths=strengths, multipliers=multipliers,
                       unavailable_experts=[str(label) for label, m in zip(config['class_labels'], models) if m is None],
                       scores={name: macro(y[va], p, classes) for name, p in predictions.items()})
            np.savez(dest / 'indices.npz', train=tr, validation=va)
            np.savez(dest / 'calibration_indices.npz', train=inner, validation=cal)
            np.savez(dest / 'predictions.npz', **predictions)
            joblib.dump(dict(models=models, strengths=strengths, multipliers=multipliers), dest / 'pipeline.joblib')
            write_json(dest / 'scores.json', row)
            rows.append(row)
            for name, p in predictions.items():
                oof.setdefault(name, np.full((len(train), len(classes)), np.nan))[va] = p
        if not (coverage == 1).all():
            raise ValueError('Incomplete expert OOF coverage')
        results = []
        for name, p in oof.items():
            scores = [r['scores'][name] for r in rows]
            results.append(dict(candidate=name, mean_f1=float(np.mean(scores)), fold_f1=scores,
                worst_f1=float(min(scores)), std_f1=float(np.std(scores)), oof_f1=macro(y, p, classes),
                per_class=classification_report(y, classes[p.argmax(1)], labels=classes, output_dict=True, zero_division=0)))
        np.savez(directory / 'oof.npz', **oof)
        return dict(selected_candidate=max(results, key=lambda r: r['mean_f1'])['candidate'],
                    results=results, folds=rows, classes=classes.tolist(),
                    reference_candidate=candidate, reference_suite_id=config['reference_suite_id'],
                    note='Same outer rows; OVR specialists train on inner/outer training only. Fusion strengths and class multipliers '
                         'fit inner calibration only. Zero strengths preserve the reference. Adaptive development validation.')

    def predict_package(self, package, frame):
        root, config = Path(package['artifact_root']), package['config']
        source, original, _, worker = self.source(config['reference_kind'], config['reference_suite_id'])
        classes = np.asarray(package['classes'])
        indices = [list(classes).index(label) for label in config['class_labels']]
        outputs = []
        for i in range(config['folds']):
            if config['reference_kind'] == 'autogluon_oof':
                _, _, sr, _ = worker.source(original['reference_suite_id'])
                reference = worker.predict_fold(source, original, sr['selected_candidate'], i, frame, classes)[package['reference_candidate']]
            else:
                pipeline = joblib.load(source / f'fold_{i}/pipeline.joblib')
                reference = worker.predict_package(dict(candidate=package['reference_candidate'], folds=[pipeline]), frame)
            pipeline = joblib.load(root / f'fold_{i}/pipeline.joblib')
            p = self._expert_prob(pipeline['models'], frame, config['task'], reference, indices)
            fused = fuse(reference, p, indices, pipeline['strengths'])
            outputs.append(dict(reference=reference, expert_fusion=fused,
                macro_decision=fused * np.asarray(pipeline['multipliers']))[package['candidate']])
        return np.mean(outputs, axis=0)

    def export_classification(self, suite_id):
        self._check_id(suite_id)
        directory = self.run / 'classification_oof' / suite_id
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        if plan != self.identity(plan['config']):
            raise ValueError('Stale classification experts')
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        if report['selected_candidate'] == 'reference':
            _, _, _, worker = self.source(plan['config']['reference_kind'], plan['config']['reference_suite_id'])
            method = worker.export_autogluon if plan['config']['reference_kind'] == 'autogluon_oof' else worker.export_classification
            result = method(plan['config']['reference_suite_id'])
            result.update(kind='classification_oof', suite_id=suite_id, candidate='reference',
                          note='Unchanged reference won; exact original fold export, no specialist fitting')
            return result
        package = dict(kind='classification_experts', artifact_root=str(directory), config=plan['config'],
                       candidate=report['selected_candidate'], classes=report['classes'],
                       reference_candidate=report['reference_candidate'])
        _, test = self.engine.load()
        p = self.predict_package(package, test)
        output = test[plan['config']['task']['ids']].copy()
        output[plan['config']['task']['target']] = np.asarray(report['classes'])[p.argmax(1)]
        output.to_csv(directory / 'submission.csv', index=False)
        joblib.dump(package, directory / 'final_model.joblib')
        return dict(kind='classification_oof', suite_id=suite_id, candidate=package['candidate'], rows=len(test),
                    submission_path=str(directory / 'submission.csv'), model_path=str(directory / 'final_model.joblib'),
                    note='Saved fold specialists and reference predictors; no export-time training. Keep source suite directories.')
