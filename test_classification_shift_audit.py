"""Two small cache-only shift-audit checks; no ML fitting or API calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'champion_solution'))
from autogluon_research import AutoGluonResearch, views
from classification_research import ClassificationResearch
from classification_shift_audit import ClassificationShiftAudit
from research import write_json


def identifier(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


class ClassificationShiftAuditTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'work' / 'classification_shift_audit_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.root = self.folder / 'agent_runs' / 'test' / 'research'
        self.run = self.root.parent
        x = np.arange(180)
        train = pd.DataFrame(dict(id=x, x=x, y=np.asarray(['A', 'B', 'C'])[x % 3]))
        train.to_csv(self.folder / 'train.csv', index=False)
        test = pd.DataFrame(dict(id=[180, 181, 182], x=[180, 181, 182]))
        test.to_csv(self.folder / 'test.csv', index=False)
        test[['id']].assign(y='A').to_csv(self.folder / 'sample_submission.csv', index=False)
        self.files = {name: str(self.folder / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
        worker = ClassificationResearch(self.root, self.files, self.folder)
        worker.engine.configure('y', 'classification', 'f1_macro')
        state = worker._state()
        config = dict(task=state, recipes=[], folds=3, seed=42, shift_id=None, selection='test fixture')
        identity = worker.identity(config)
        source_id = identifier(identity)
        source = self.run / 'classification_oof' / source_id
        source.mkdir(parents=True)
        write_json(source / 'plan.json', identity)
        write_json(source / 'report.json', dict(suite_id=source_id, selected_candidate='model_0', classes=['A', 'B', 'C']))
        reference = np.full((180, 3), 1 / 3)
        automatic = .97 * np.eye(3)[x % 3] + .01
        predictions = views(reference, automatic, [.5, .5], [1., 1., 1.])
        np.savez(source / 'oof.npz', model_0=reference)
        auto = AutoGluonResearch(self.root, self.files, self.folder)
        config = dict(backend='autogluon_v1', task=state, reference_suite_id=source_id, folds=3,
                      time_limit=120, bag_folds=3, stack_levels=0, feature_code='', selection='test fixture')
        identity = auto.identity(config)
        self.suite_id = identifier(identity)
        self.directory = self.run / 'autogluon_oof' / self.suite_id
        self.directory.mkdir(parents=True)
        write_json(self.directory / 'plan.json', identity)
        write_json(self.directory / 'report.json', dict(suite_id=self.suite_id, classes=['A', 'B', 'C'],
            selected_candidate='autogluon', reference_suite_id=source_id))
        np.savez(self.directory / 'oof.npz', **predictions)
        for i, va in enumerate(np.array_split(x, 3)):
            tr = np.setdiff1d(x, va)
            for directory in (source, self.directory):
                fold = directory / f'fold_{i}'
                fold.mkdir()
                np.savez(fold / 'indices.npz', train=tr, validation=va)
            np.savez(source / f'fold_{i}' / 'predictions.npz', model_0=reference[va])
            np.savez(self.directory / f'fold_{i}' / 'predictions.npz',
                     **{name: values[va] for name, values in predictions.items()})
        config = dict(task=state, diagnostic='train_vs_test', folds=3, iterations=120, seed=42)
        identity = worker.identity(config)
        self.shift_id = identifier(identity)
        self.shift = self.run / 'adversarial' / self.shift_id
        self.shift.mkdir(parents=True)
        write_json(self.shift / 'plan.json', identity)
        write_json(self.shift / 'report.json', dict(diagnostic_id=self.shift_id, auc=.8))
        origin = np.linspace(0, 1, 183)
        np.save(self.shift / 'origin_oof.npy', origin)
        np.save(self.shift / 'stress_validation.npy', np.arange(144, 180))
        for i, va in enumerate(np.array_split(np.arange(183), 3)):
            np.savez(self.shift / f'fold_{i}.npz', train=np.setdiff1d(np.arange(183), va), validation=va)
        self.worker = ClassificationShiftAudit(self.root, self.files, self.folder)

    def test_only_cached_outer_predictions_and_no_selection_change(self):
        selection = self.run / 'final_selection.json'
        selection.write_text('unchanged', encoding='utf-8')
        result = self.worker.audit_classification_shift(self.suite_id, self.shift_id)
        report = json.loads(Path(result['report_path']).read_text(encoding='utf-8'))
        self.assertFalse(result['selection_changed'])
        self.assertEqual(report['test_like_rows'], 36)
        self.assertEqual(selection.read_text(encoding='utf-8'), 'unchanged')
        by_candidate = {r['candidate']: r for r in report['results']}
        self.assertEqual(by_candidate['autogluon']['full_oof']['macro_f1'], 1.)
        self.assertEqual(by_candidate['autogluon']['test_like_slice']['macro_f1'], 1.)
        self.assertTrue(all(r['low_slice_support'] for r in by_candidate['autogluon']['test_like_slice']['per_class']))
        self.assertTrue(all(r['support'] == 12 for r in by_candidate['autogluon']['test_like_slice']['per_class']))
        self.assertEqual(report['diagnostic'], 'conditional_outer_oof_slice')

    def test_rejects_invalid_cache_and_stale_task(self):
        with np.load(self.directory / 'fold_0' / 'predictions.npz') as saved:
            changed = dict(saved)
        changed['autogluon'] = changed['autogluon'][:, ::-1]
        np.savez(self.directory / 'fold_0' / 'predictions.npz', **changed)
        with self.assertRaisesRegex(ValueError, 'OOF cache differs'):
            self.worker.audit_classification_shift(self.suite_id, self.shift_id)
        # Freshness is checked before reading any prediction cache.
        with (self.folder / 'train.csv').open('a', encoding='utf-8') as handle:
            handle.write('180,180,A\n')
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.worker.audit_classification_shift(self.suite_id, self.shift_id)
        self.assertFalse((self.run / 'classification_shift_audit.json').exists())


if __name__ == '__main__':
    unittest.main()
