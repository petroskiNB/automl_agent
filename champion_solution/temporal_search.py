"""Global temporal ranking and bounded plateau detection on completed new trials."""
import json
import math
from pathlib import Path


def completed_reports(run, state, expected=None):
    run = Path(run)
    profile_path = run / 'validation_profile.json'
    if expected is None and profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding='utf-8'))
        chosen = next((v for v in profile.get('future_columns', []) if v['column'] == state.get('split_column')), None)
        expected = chosen['validation_periods'] if chosen else None
    result = []
    for plan_path in (run/'temporal').glob('*/plan.json'):
        report_path = plan_path.with_name('report.json')
        if not report_path.exists():
            continue
        plan = json.loads(plan_path.read_text(encoding='utf-8'))['config']
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if plan.get('time_column') != state.get('split_column') or (expected is not None and plan['validation_periods'] != expected):
            continue
        contract = plan.get('task')
        if contract and any(contract[k] != state[k] for k in ('target', 'task_type', 'metric', 'features', 'ids')):
            continue
        result.append((plan, report, plan_path.stat().st_mtime_ns))
    return sorted(result, key=lambda r: (r[2], r[1]['suite_id']))


def catalogue(run, state, expected=None):
    reports = completed_reports(run, state, expected)
    groups = {tuple(p['validation_periods']) for p, _, _ in reports}
    if len(groups) > 1:
        return dict(candidates=[], recommended_suite=None, reason='Validation periods differ; freeze one comparison plan')
    candidates = []
    for plan, report, _ in reports:
        row = next(r for r in report['results'] if r['mode'] == report['selected_mode'])
        if not all(math.isfinite(row[k]) for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')):
            continue
        candidates.append(dict(suite_id=report['suite_id'], mode=row['mode'],
            kind=plan.get('kind', 'single'), **{k: row[k] for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')}))
    candidates.sort(key=lambda v: (v['mean_rmse'], v['latest_rmse'], v['worst_rmse'], v['suite_id']))
    return dict(candidates=candidates, recommended_suite=candidates[0]['suite_id'] if candidates else None,
                criterion='Global minimum mean_rmse; exact ties: latest_rmse, worst_rmse, stable suite ID. Same temporal rows only.')


def search_status(run, state, expected=None, minimum_trials=8, patience=4, maximum_trials=16):
    """Re-reading reports never advances a trial or changes the stopping evidence."""
    reports = completed_reports(run, state, expected)
    best_reference = None
    since_gain = 0
    trials = 0
    entries = []
    for plan, report, _ in reports:
        row = next(r for r in report['results'] if r['mode'] == report['selected_mode'])
        if not math.isfinite(row['mean_rmse']):
            continue
        is_trial = plan.get('kind') != 'ensemble'
        trials += int(is_trial)
        gain = best_reference is None or row['mean_rmse'] < best_reference * .998
        if gain:
            best_reference, since_gain = row['mean_rmse'], 0
        elif is_trial:
            since_gain += 1
        entries.append(dict(suite_id=report['suite_id'], trial=is_trial, mean_rmse=row['mean_rmse'], substantial_gain=gain))
    plateau = trials >= minimum_trials and since_gain >= patience
    cap = trials >= maximum_trials
    return dict(trials=trials, minimum_trials=minimum_trials, patience=patience, maximum_trials=maximum_trials,
        trials_since_substantial_gain=since_gain, substantial_gain_relative=.002,
        plateau=plateau, trial_cap_reached=cap, continue_search=not (plateau or cap), entries=entries,
        note='Only completed distinct suites count; cache reads and equal-weight blends do not increase trial count. Time/steps/API limits still apply.')
