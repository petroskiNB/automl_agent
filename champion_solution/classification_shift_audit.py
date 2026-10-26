"""Read-only evaluation of saved outer OOF scores on a train/test-shift slice.

This is a conditional OOF diagnostic, not an independent stress-fold experiment.
No fitted model, decision multiplier or ensemble weight is learned here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from autogluon_research import AutoGluonResearch
from classification_research import ClassificationResearch
from research import write_json


CANDIDATES = ('reference', 'autogluon', 'equal_ensemble', 'weighted_ensemble', 'macro_decision')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _indices(values, count, name):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values) or
            (values < 0).any() or (values >= count).any() or len(np.unique(values)) != len(values)):
        raise ValueError('Invalid ' + name + ' indices')
    return values


def _scores(values, rows, classes):
    values = np.asarray(values, dtype=float)
    if (values.shape != (rows, classes) or not np.isfinite(values).all() or
            (values < 0).any() or (values.sum(axis=1) <= 0).any()):
        raise ValueError('Invalid cached OOF class scores')
    return values


class ClassificationShiftAudit(ClassificationResearch):
    def audit_classification_shift(self, suite_id, shift_id):
        """Inspect completed, fresh AG/reference OOF predictions; never refit or select."""
        self._check_id(suite_id)
        self._check_id(shift_id)
        state = self._state()
        if self.engine.state().get('split') != 'random':
            raise ValueError('Conditional OOF shift audit requires random stratified classification')
        train, test = self.engine.load()
        y = train[state['target']].to_numpy()
        classes = np.unique(y)
        count = len(train)
        worker = AutoGluonResearch(self.engine.root, self.engine.files, self.folder)
        directory = self.run / 'autogluon_oof' / suite_id
        plan, report = _read(directory / 'plan.json'), _read(directory / 'report.json')
        config = plan['config']
        if (config.get('backend') != 'autogluon_v1' or config.get('task') != state or
                plan != worker.identity(config)):
            raise ValueError('AutoGluon suite is stale or belongs to another task')
        if (report.get('suite_id') != suite_id or report.get('classes') != classes.tolist() or
                report.get('selected_candidate') not in CANDIDATES):
            raise ValueError('AutoGluon report is incomplete or has a different class axis')
        source, source_config, source_report, _ = worker.source(config['reference_suite_id'])
        if (source_report.get('classes') != classes.tolist() or
                report.get('reference_suite_id') != config['reference_suite_id'] or
                config.get('folds') != source_config.get('folds') or config.get('folds') not in (3, 4, 5)):
            raise ValueError('Source and AutoGluon OOF contracts differ')

        shift_directory = self.run / 'adversarial' / shift_id
        shift_plan, shift_report = _read(shift_directory / 'plan.json'), _read(shift_directory / 'report.json')
        expected_shift = dict(task=state, diagnostic='train_vs_test', folds=3, iterations=120, seed=42)
        if (shift_plan.get('config') != expected_shift or shift_plan != self.identity(expected_shift) or
                shift_report.get('diagnostic_id') != shift_id):
            raise ValueError('Shift diagnostic is stale or belongs to another task')
        origin = np.load(shift_directory / 'origin_oof.npy', allow_pickle=False)
        if (origin.shape != (count + len(test),) or not np.isfinite(origin).all() or
                (origin < 0).any() or (origin > 1).any()):
            raise ValueError('Invalid saved origin OOF probabilities')
        subset = _indices(np.load(shift_directory / 'stress_validation.npy', allow_pickle=False), count, 'shift')
        expected_subset = []
        for label in classes:
            rows = np.flatnonzero(y == label)
            number = max(1, int(np.floor(.2 * len(rows))))
            expected_subset.extend(rows[np.argsort(origin[rows], kind='stable')[-number:]].tolist())
        if not np.array_equal(subset, np.array(sorted(expected_subset), dtype=int)):
            raise ValueError('Shift rows do not match the frozen per-class top-20% rule')

        assets = [directory / 'plan.json', directory / 'report.json', directory / 'oof.npz',
                  source / 'plan.json', source / 'report.json', source / 'oof.npz',
                  shift_directory / 'plan.json', shift_directory / 'report.json',
                  shift_directory / 'origin_oof.npy', shift_directory / 'stress_validation.npy']
        origin_coverage = np.zeros(count + len(test), dtype=int)
        for i in range(3):
            path = shift_directory / f'fold_{i}.npz'
            assets.append(path)
            with np.load(path, allow_pickle=False) as saved:
                tr = _indices(saved['train'], len(origin), 'origin training')
                va = _indices(saved['validation'], len(origin), 'origin validation')
            if (len(np.intersect1d(tr, va)) or
                    not np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(origin)))):
                raise ValueError('Invalid origin OOF partition')
            origin_coverage[va] += 1
        if not (origin_coverage == 1).all():
            raise ValueError('Origin OOF folds must cover every row exactly once')
        with np.load(directory / 'oof.npz', allow_pickle=False) as data:
            if set(data.files) != set(CANDIDATES):
                raise ValueError('Incomplete AutoGluon OOF candidate cache')
            predictions = {name: _scores(data[name], count, len(classes)) for name in CANDIDATES}
        reference_candidate = source_report['selected_candidate']
        with np.load(source / 'oof.npz', allow_pickle=False) as data:
            reference = _scores(data[reference_candidate], count, len(classes))
        if not np.array_equal(predictions['reference'], reference):
            raise ValueError('Cached reference scores differ from the source suite')
        coverage, validation_folds = np.zeros(count, dtype=int), []
        for i in range(config['folds']):
            fold = directory / f'fold_{i}'
            original = source / f'fold_{i}'
            assets.extend([fold / 'indices.npz', fold / 'predictions.npz',
                           original / 'indices.npz', original / 'predictions.npz'])
            with np.load(fold / 'indices.npz', allow_pickle=False) as saved:
                tr = _indices(saved['train'], count, 'outer training')
                va = _indices(saved['validation'], count, 'outer validation')
            with np.load(original / 'indices.npz', allow_pickle=False) as saved:
                if not np.array_equal(tr, saved['train']) or not np.array_equal(va, saved['validation']):
                    raise ValueError('Source and AutoGluon outer rows differ')
            if (len(np.intersect1d(tr, va)) or
                    not np.array_equal(np.sort(np.r_[tr, va]), np.arange(count))):
                raise ValueError('Invalid outer fold partition')
            with np.load(fold / 'predictions.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(CANDIDATES):
                    raise ValueError('Incomplete outer-fold candidate cache')
                for name in CANDIDATES:
                    p = _scores(saved[name], len(va), len(classes))
                    if not np.array_equal(p, predictions[name][va]):
                        raise ValueError('OOF cache differs from outer validation scores')
            with np.load(original / 'predictions.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved[reference_candidate], reference[va]):
                    raise ValueError('Reference OOF cache differs from its validation scores')
            coverage[va] += 1
            validation_folds.append(va)
        if not (coverage == 1).all():
            raise ValueError('Outer folds must cover each training row exactly once')

        identity = dict(suite_id=suite_id, shift_id=shift_id, task=state,
            hashes={str(p.relative_to(self.run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
            audit_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        audit_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        output = self.run / 'classification_shift_audit' / audit_id
        output.mkdir(parents=True, exist_ok=True)
        def metrics(scores, rows):
            predicted = classes[scores[rows].argmax(axis=1)]
            per_class = classification_report(y[rows], predicted, labels=classes,
                output_dict=True, zero_division=0)
            return dict(rows=len(rows), macro_f1=float(f1_score(y[rows], predicted,
                labels=classes, average='macro', zero_division=0)),
                per_class=[dict(label=str(label), **per_class[str(label)],
                    training_support=int(np.sum(y == label)),
                    absent=bool(np.sum(y[rows] == label) == 0),
                    low_slice_support=bool(np.sum(y[rows] == label) < 20),
                    rare_training_class=bool(np.sum(y == label) < 100)) for label in classes])
        results = []
        for name in CANDIDATES:
            p = predictions[name]
            folded = []
            for i, va in enumerate(validation_folds):
                selected_rows = np.intersect1d(va, subset)
                folded.append(dict(fold=i, rows=len(selected_rows), macro_f1=(
                    metrics(p, selected_rows)['macro_f1'] if len(selected_rows) else None)))
            results.append(dict(candidate=name, full_oof=metrics(p, np.arange(count)),
                test_like_slice=metrics(p, subset), slice_folds=folded))
        result = dict(audit_id=audit_id, suite_id=suite_id, shift_id=shift_id,
            diagnostic='conditional_outer_oof_slice', selection_changed=False,
            selected_candidate_unchanged=report['selected_candidate'], classes=classes.tolist(),
            train_rows=count, test_like_rows=len(subset), origin_auc=shift_report['auc'], results=results,
            report_path=str(output / 'report.json'),
            note='Cached outer OOF scores only; no training, ranking, threshold or ensemble fitting. '
                 'Slice is the frozen top 20% origin score within each training class, not the hidden test '
                 'or a new untouched holdout. Other slice rows may occur in each outer training fold. '
                 'Full-class Macro F1 includes absent labels as zero; inspect per-class support. '
                 'Previously searched recipes and outer candidate selection remain adaptive.')
        write_json(output / 'plan.json', identity)
        write_json(output / 'report.json', result)
        write_json(self.run / 'classification_shift_audit.json', result)
        return {k: ([dict(candidate=row['candidate'], full_oof_f1=row['full_oof']['macro_f1'],
                         test_like_f1=row['test_like_slice']['macro_f1']) for row in value]
                    if k == 'results' else value) for k, value in result.items()}
