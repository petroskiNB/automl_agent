"""Check source composition, immutable runtime caching and clean packaging."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from neural_solution.runtime import prepare, SHARED, OVERRIDES


class RuntimeTests(unittest.TestCase):
    def test_neural_checklist_requires_actual_completed_variants(self):
        from neural_solution.solution_torch import neural_requirements
        state = dict(task_type='classification', experiments=[])
        self.assertTrue(neural_requirements(state))
        state['experiments'] = [
            dict(algorithm='torch_mlp', params={}, feature_code='def transform(X): return X'),
            dict(algorithm='torch_mlp', params={'auto_class_weights': 'SqrtBalanced'}),
            dict(algorithm='torch_resnet', params={}, timed_out=True)]
        self.assertEqual(neural_requirements(state), ['Обучить компактную torch_resnet'])
        state['experiments'][-1]['timed_out'] = False
        self.assertEqual(neural_requirements(state), [])

    def setUp(self):
        self.scratch = Path(__file__).resolve().parent / 'work' / 'runtime_tests'
        self.scratch.mkdir(parents=True, exist_ok=True)

    def test_runtime_composes_shared_sources_and_overrides(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            root = Path(directory)
            (root / 'neural_solution').mkdir()
            for name in SHARED:
                (root / name).write_text('# shared\n')
            for name in OVERRIDES:
                (root / 'neural_solution' / name).write_text('# neural\n')
            runtime = prepare(root)
            self.assertEqual(prepare(root), runtime)
            self.assertEqual((runtime / 'solution.py').read_text(), '# shared\n')
            self.assertEqual((runtime / 'research.py').read_text(), '# neural\n')
            (root / 'solution.py').write_text('# changed\n')
            self.assertNotEqual(prepare(root), runtime)
            self.assertEqual((runtime / 'solution.py').read_text(), '# shared\n')
            (runtime / 'research.py').write_text('# tampered\n')
            (root / 'solution.py').write_text('# shared\n')
            with self.assertRaisesRegex(ValueError, 'modified'):
                prepare(root)

    def test_packaged_runtime_contains_both_backends_without_data(self):
        runtime = prepare()
        spec = importlib.util.spec_from_file_location('packager', runtime / 'build_torch_submission.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            archive = Path(directory) / 'combined.zip'
            module.build(archive)
            with ZipFile(archive) as bundle:
                self.assertEqual(bundle.read('legacy_solution.py'), (runtime / 'solution.py').read_bytes())
                for name in ('research.py', 'torch_backend.py', 'classification_checks.py', 'xgb_backend.py'):
                    self.assertEqual(bundle.read(name), (runtime / name).read_bytes())
                self.assertTrue(all(Path(n).suffix in {'.py', '.txt'} for n in bundle.namelist()))
                self.assertIn(b'--platform', bundle.read('solution.py'))


if __name__ == '__main__':
    unittest.main()
