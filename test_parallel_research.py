import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from threading import Barrier, Lock

import joblib
import numpy as np
import pandas as pd
from parallel_research import run_batch, resolve_recipe
from research import Research
from solution import discover_files


def call(name, algorithm, **extra):
    args = dict(hypothesis=name, algorithm=algorithm, params=dict(iterations=10, depth=2), feature_code='')
    args.update(extra)
    return NS(id='team_1_' + name, function=NS(name='train_model', arguments=json.dumps(args)))


class ParallelTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / 'work'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        x = np.linspace(-2, 2, 180)
        frame = pd.DataFrame(dict(id=np.arange(len(x)), x=x, target=np.where(x > 0, 'yes', 'no')))
        frame.to_csv(self.folder / 'train.csv', index=False)
        frame.drop(columns='target').iloc[:10].to_csv(self.folder / 'test.csv', index=False)
        frame.iloc[:10][['id', 'target']].to_csv(self.folder / 'sample_submission.csv', index=False)
        (self.folder / 'data_dictionary.csv').write_text('feature,role\ntarget,target\n')
        self.files = discover_files(self.folder)
        self.log = self.folder / 'logs'
        self.engine = Research(self.log / 'research', self.files, self.folder)
        self.engine.configure('target', 'classification', 'f1_macro', 'random')
        (self.folder / 'submission.csv').write_text('unchanged')

    def test_real_workers_merge_without_publishing_and_reload(self):
        calls = [call('models', 'catboost'), call('rare', 'xgboost'), call('third', 'lightgbm')]
        with patch('parallel_research.available_memory', return_value=16 * 1024**3):
            results = run_batch(calls, self.folder, self.files, self.log, 90, jobs=2)
        for result in results.values():
            self.assertEqual(result['exit_code'], 0, result)
        first, second, third = [results[c.id] for c in calls]
        self.assertLess(max(first['started_at'], second['started_at']), min(first['ended_at'], second['ended_at']))
        self.assertGreaterEqual(third['started_at'], min(first['ended_at'], second['ended_at']))
        state = self.engine.state()
        self.assertEqual([e['id'] for e in state['experiments']], [1, 2, 3])
        self.assertEqual((self.folder / 'submission.csv').read_text(), 'unchanged')
        for record in state['experiments']:
            bundle = joblib.load(record['model_path'])
            self.assertTrue(np.isfinite(self.engine.predict_raw(bundle, self.engine.load()[0], state)).all())
        self.assertEqual(joblib.load(state['experiments'][0]['model_path'])['model'].get_param('thread_count'), first['threads'])
        duplicate = run_batch([calls[0]], self.folder, self.files, self.log, 10)
        self.assertIn('Duplicate', duplicate[calls[0].id]['error'])

    def test_failed_worker_does_not_lose_success_and_timeout_does_not_commit(self):
        bad = call('models', 'catboost', params=dict(depth=999))
        good = call('rare', 'catboost')
        with patch('parallel_research.available_memory', return_value=0):
            results = run_batch([bad, good], self.folder, self.files, self.log, 60, jobs=2)
        self.assertNotEqual(results[bad.id]['exit_code'], 0)
        self.assertEqual(results[good.id]['exit_code'], 0)
        self.assertEqual(len(self.engine.state()['experiments']), 1)
        result = run_batch([call('timeout', 'xgboost')], self.folder, self.files, self.log, .001)
        self.assertEqual(result['team_1_timeout']['exit_code'], -1)
        self.assertEqual(len(self.engine.state()['experiments']), 1)

    def test_feature_intent_is_checked_and_parameters_inherited(self):
        state = dict(experiments=[dict(id=49, algorithm='xgboost', params={'depth': 4}, feature_code='old')])
        with self.assertRaises(ValueError):
            resolve_recipe(dict(base_experiment_id=49, hypothesis='new'), state, 'features')
        with self.assertRaises(ValueError):
            resolve_recipe(dict(base_experiment_id=49, feature_code='old'), state, 'features')
        result = resolve_recipe(dict(base_experiment_id=49, feature_code='new'), state, 'features')
        self.assertEqual(result['params'], {'depth': 4})
        self.assertEqual(result['feature_code'], 'new')

    def test_controller_three_researchers_real_training_and_finalization(self):
        from solution import run_agent, fingerprint, validate_submission
        run = self.folder / 'agent_runs' / 'integration'
        engine = Research(run / 'research', self.files, self.folder)
        engine.configure('target', 'classification', 'f1_macro', 'random')
        engine.train('baseline', dict(iterations=10, depth=2))
        task = self.folder / 'task.md'
        task.write_text('Classification Macro F1')
        (run / 'manifest.json').write_text(json.dumps(fingerprint(task, self.files)))
        (run / 'final_selection.json').write_text('{"old":true}')
        args = NS(task=task, resume=run, reviewer=False, analyst=False, researchers=3, training_jobs=2,
                  new_selection=True, require_selection=False, time_budget=180, final_reserve=30,
                  max_steps=2, step_timeout=90, min_experiments=3)
        barrier, lock, counts = Barrier(3), Lock(), {}
        def completion(**kwargs):
            prompt = kwargs['messages'][0]['content']
            role = 'features' if 'Ты исследователь признаков.' in prompt else ('rare' if 'Ты исследователь редких классов.' in prompt else 'models')
            with lock:
                counts[role] = counts.get(role, 0) + 1
                first = counts[role] == 1
            if not first:
                return NS(choices=[NS(message=NS(content='Done', tool_calls=[]))])
            barrier.wait(timeout=10)
            arguments = dict(hypothesis=role, feature_code='')
            if role == 'features':
                arguments.update(base_experiment_id=1, feature_code='def transform(X):\n    X = X.copy()\n    X["square"] = X["x"] ** 2\n    return X')
            elif role == 'rare':
                arguments.update(base_experiment_id=1, params={'auto_class_weights': 'Balanced'})
            else:
                arguments.update(algorithm='xgboost', params={'iterations': 10, 'depth': 2})
            tool = NS(function=NS(name='train_model', arguments=json.dumps(arguments)))
            return NS(choices=[NS(message=NS(content='Training proposal', tool_calls=[tool]))])
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'OPENAI_BASE_URL': 'http://unused.invalid'}), \
             patch('openai.OpenAI') as client, patch('builtins.print'):
            client.return_value.__enter__.return_value.chat.completions.create.side_effect = completion
            run_agent(args, self.folder, self.files)
        self.assertEqual(set(counts), {'features', 'models', 'rare'})
        self.assertEqual(len(engine.state()['experiments']), 4)
        self.assertTrue(engine.state()['finalized'])
        self.assertTrue(validate_submission(self.folder, self.files)[0])
        self.assertTrue(list(run.glob('final_selection_before_*.json')))
        self.assertFalse((self.folder / '.agent.lock').exists())


if __name__ == '__main__':
    unittest.main()
