"""Small per-run audit, including failures, without changing model selection."""
import hashlib
import json
import os
from pathlib import Path
import time


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def write_summary(run, args, folder, elapsed, error_type=None):
    run, folder = Path(run), Path(folder)
    state = read_json(run / 'research/state.json') or {}
    selection = read_json(run / 'final_selection.json')
    output = read_json(run / 'final_output.json')
    recipes = [{k: e.get(k) for k in ('id', 'algorithm', 'score', 'metric', 'selected_iterations',
                'fit_seconds', 'seconds', 'timed_out')} for e in state.get('experiments', [])]
    submission = folder / 'submission.csv'
    config = {k: getattr(args, k, None) for k in ('time_budget', 'max_steps', 'step_timeout',
              'final_reserve', 'researchers', 'training_jobs', 'reviewer', 'analyst', 'require_selection',
              'api_timeout', 'api_retries')}
    config['threads'] = os.getenv('AUTOML_CPU_THREADS')
    config['autogluon_budget'] = int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0'))
    config['flaml_budget'] = int(os.getenv('AUTOML_FLAML_BUDGET', '0'))
    config['lightautoml_budget'] = int(os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0'))
    config['extended_research'] = os.getenv('AUTOML_EXTENDED_RESEARCH') == '1'
    summary = dict(status='failed' if error_type else 'completed', error_type=error_type,
        elapsed_seconds=round(elapsed, 3), configuration=config,
        model=os.getenv('OPENAI_MODEL', 'qwen/qwen3.8-27b'),
        task_type=state.get('task_type'), metric=state.get('metric'),
        experiments=recipes, final_selection=selection, final_output=output,
        research_progress=read_json(run / 'research_progress.json'),
        encoding_comparison=read_json(run / 'encoding_comparison.json'),
        temporal_search=read_json(run / 'temporal_search.json'),
        regression_anchors=read_json(run / 'regression_anchors.json'),
        autogluon=read_json(run / 'autogluon_summary.json'),
        autogluon_usage=read_json(run / 'autogluon_budget.json'),
        flaml=read_json(run / 'flaml_summary.json'),
        flaml_usage=read_json(run / 'flaml_budget.json'),
        lightautoml=read_json(run / 'lightautoml_summary.json'),
        lightautoml_usage=read_json(run / 'lightautoml_budget.json'),
        classification_experts=read_json(run / 'classification_experts_summary.json'),
        regression_specialists=read_json(run / 'regression_specialists_summary.json'),
        classification_selection=read_json(run / 'classification_selection.json'),
        adversarial=read_json(run / 'adversarial_summary.json'),
        classification_shift_audit=read_json(run / 'classification_shift_audit.json'),
        prediction_bounds=state.get('prediction_bounds'),
        validation_profile=read_json(run / 'validation_profile.json'),
        holdout_best=state.get('best', {}).get('id') if state.get('best') else None,
        submission_exists=submission.is_file(),
        submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest() if submission.is_file() else None,
        note='Submission presence on failure does not confirm a successful export; inspect final_output.')
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    history = run / f'run_summary_{time.time_ns()}.json'
    history.write_text(encoded, encoding='utf-8')
    temporary = run / f'.summary_{time.time_ns()}.tmp'
    temporary.write_text(encoded, encoding='utf-8')
    temporary.replace(run / 'run_summary.json')
