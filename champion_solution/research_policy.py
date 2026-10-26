"""Evidence-based split guard and bounded research stages, independent of the LLM."""
import hashlib
import json
from pathlib import Path
import re
import os
from copy import deepcopy
from reference_recipes import REFERENCE_RECIPES, REGRESSION_TEMPORAL_ANCHORS


def validation_profile(files):
    import numpy as np
    import pandas as pd
    columns = pd.read_csv(files['test.csv'], nrows=0).columns
    candidates = [c for c in columns if re.search(r'(^|_)(year|time|date|timestamp|period)(_|$)', c.lower())]
    evidence = []
    for column in candidates:
        train = pd.to_numeric(pd.read_csv(files['train.csv'], usecols=[column])[column], errors='coerce')
        test = pd.to_numeric(pd.read_csv(files['test.csv'], usecols=[column])[column], errors='coerce')
        if not len(train) or not len(test) or not np.isfinite(train).all() or not np.isfinite(test).all():
            continue
        periods = sorted(float(v) for v in train.unique())
        if len(periods) >= 3 and float(test.min()) > float(train.max()):
            evidence.append(dict(column=column, train_min=float(train.min()), train_max=float(train.max()),
                                 test_min=float(test.min()), test_max=float(test.max()),
                                 validation_periods=periods[2:][-3:]))
    recommended = evidence[0] if len(evidence) == 1 else None
    return dict(future_columns=evidence, recommended=recommended,
                note='Only named finite numeric chronological candidates are enforced. Multiple candidates require an explicit choice; no evidence does not prove i.i.d.')


def guard_split(profile, split, column):
    candidates = {e['column'] for e in profile['future_columns']}
    if candidates and (split != 'time' or column not in candidates):
        raise ValueError('Test is strictly later than train on ' + ', '.join(sorted(candidates)) +
                         '. Configure a time split on an evidenced column before freezing the run.')


def library():
    # Each caller gets independent data, without a deployment-time JSON asset.
    return deepcopy(REFERENCE_RECIPES)


def applicable_recipes(state):
    return [r for r in library() if r['task_type'] == state.get('task_type') and r['target'] == state.get('target')
            and set(r['required_columns']).issubset(state.get('features', []))]


def reference_recipe(reference_id, state):
    matches = [r for r in applicable_recipes(state) if r['reference_id'] == reference_id]
    if len(matches) != 1:
        raise ValueError('Reference recipe does not match this task/schema; inspect the research plan')
    return matches[0]


def recipe_key(recipe):
    return hashlib.sha256(json.dumps([recipe['algorithm'], recipe.get('params', {}),
               recipe.get('feature_code', '')], sort_keys=True).encode()).hexdigest()


def encoding_comparison(state, reports):
    """Pair only matching recipes/folds/history keys; prespecify the selection rule."""
    refs = applicable_recipes(state)
    variants = [r for r in refs if r.get('comparison_of')]
    if not variants:
        return None
    full = variants[0]
    base = next(r for r in refs if r['reference_id'] == full['comparison_of'])
    keys = {name: recipe_key(r) for name, r in (('legacy', base), ('full_strings', full))}
    measured = {recipe_key(e): e for e in state.get('experiments', []) if not e.get('timed_out')}
    holdout = {name: {k: measured[key][k] for k in ('id', 'score')} for name, key in keys.items() if key in measured}
    result = dict(reference_ids=[base['reference_id'], full['reference_id']], holdout=holdout,
        temporal_complete=False, temporal_pair=None,
        selection_rule='Prespecified: lower mean_rmse on identical temporal periods; latest_rmse then worst_rmse break exact ties. No significance claim.')
    grouped = {name: [(p, d) for p, d in reports if 'recipe' in p and recipe_key(p['recipe']) == key] for name, key in keys.items()}
    for base_plan, base_report in grouped['legacy']:
        for full_plan, full_report in grouped['full_strings']:
            compare_fields = ('time_column', 'validation_periods', 'key_columns', 'group_columns', 'history_gate', 'modes', 'task',
                              'key_variants', 'group_hierarchy', 'smoothing', 'area_rounding')
            if any(base_plan.get(k) != full_plan.get(k) for k in compare_fields):
                continue
            a = {r['mode']: r for r in base_report['results']}
            b = {r['mode']: r for r in full_report['results']}
            if set(a) != set(b):
                continue
            rows = []
            for mode in a:
                rows.append(dict(mode=mode, legacy={k: a[mode][k] for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')},
                    full_strings={k: b[mode][k] for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')},
                    mean_delta_full_minus_legacy=b[mode]['mean_rmse']-a[mode]['mean_rmse']))
            candidates = [(d['suite_id'], r) for d in (base_report, full_report) for r in d['results']
                          if r['mode'] == d['selected_mode']]
            winner_sid, winner = min(candidates, key=lambda v: (v[1]['mean_rmse'], v[1]['latest_rmse'], v[1]['worst_rmse']))
            result.update(temporal_complete=True, temporal_pair=dict(legacy_suite=base_report['suite_id'],
                full_strings_suite=full_report['suite_id'], rows=rows, recommended_suite=winner_sid,
                recommended_mode=winner['mode']))
            return result
    return result


def temporal_suggestions(profile, state):
    candidate = next((r for r in profile.get('future_columns', []) if r['column'] == state.get('split_column')), None)
    if not candidate:
        return None
    # These are explicitly hypotheses about keys, never a declaration of entity identity.
    available = state.get('features', [])
    keys = [c for c in ('State_Factor', 'facility_type', 'building_class', 'floor_area', 'year_built', 'ELEVATION') if c in available]
    groups = [c for c in ('State_Factor', 'facility_type') if c in available]
    return dict(time_column=candidate['column'], validation_periods=candidate['validation_periods'],
                key_columns=keys, group_columns=groups,
                key_variants=[v for v in ([c for c in keys if c not in {'ELEVATION', 'building_class'}],
                                          [c for c in keys if c not in {'ELEVATION', 'building_class', 'year_built'}]) if v and v != keys],
                group_hierarchy=[[c for c in ('facility_type',) if c in available]] if 'facility_type' in available else [],
                smoothing=20., history_hierarchy=True, residual=True,
                key_note='Candidate building key; examine coverage and collision risk. For another schema propose meaningful entity and backoff keys from its dictionary.')


def progress(run, state, profile, total_budget=9900):
    run = Path(run)
    experiments = [e for e in state.get('experiments', []) if not e.get('timed_out')]
    singles = [e for e in experiments if e.get('algorithm') != 'ensemble']
    refs = applicable_recipes(state)
    pending, skipped, improvement = [], [], None
    global_selection, exploration = None, None
    anchor_summary, flaml_blend, lightautoml_blend, shift_audit = None, None, None, None

    def current_json(name):
        # Run-local generated reports only; historical recipes are embedded in Python.
        path = run / name
        try:
            value = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None
    if not state.get('target'):
        pending.append('inspect_data, inspect_research_plan, затем configure_task по временному профилю')
    else:
        if len({e['algorithm'] for e in singles}) < 2:
            pending.append('Сравнить хотя бы два семейства моделей на фиксированном split')
        if len(singles) < 3:
            pending.append('Завершить минимум три разных одиночных рецепта, включая обоснованную доработку')
        measured = {recipe_key(e) for e in singles}
        if refs and not any(recipe_key(r) in measured for r in refs):
            pending.append('Повторно измерить один применимый сильный рецепт через train_reference_recipe')
        reference_keys = {recipe_key(r) for r in refs if not r.get('comparison_of')}
        anchors = [e for e in singles if recipe_key(e) in reference_keys]
        if anchors and state.get('best'):
            minimize = state.get('task_type') == 'regression'
            anchor = (min if minimize else max)(anchors, key=lambda e: e['score'])
            best = state['best']['score']
            achieved = best < anchor['score'] * .998 if minimize else best > anchor['score'] + .001
            later = {recipe_key(e) for e in singles if e['id'] > min(a['id'] for a in anchors)
                     and recipe_key(e) not in reference_keys}
            improvement = dict(current_reference_score=anchor['score'], best_holdout_score=best,
                target_achieved=achieved, new_hypotheses_after_reference=len(later),
                note='Bounded search goal on the same frozen holdout, not statistical proof; temporal mean scores are not compared with this anchor.')
            if not achieved and len(later) < 3 and total_budget >= 900 and state.get('split') != 'time':
                pending.append('После измеренного сильного baseline проверить до трёх новых разных гипотез улучшения; отсутствие выигрыша признать явно')
        is_temporal = state.get('task_type') == 'regression' and state.get('metric') == 'rmse' and state.get('split') == 'time'
        suggestion = temporal_suggestions(profile, state)
        expected = suggestion['validation_periods'] if suggestion else []
        anchor = REGRESSION_TEMPORAL_ANCHORS
        anchors_apply = (is_temporal and state.get('target') == anchor['target']
            and state.get('split_column') == anchor['time_column']
            and set(anchor['required_columns']).issubset(state.get('features', [])))
        if anchors_apply and total_budget >= 900 and len(expected) >= 2:
            candidate = current_json('regression_anchors.json')
            contract = {k: state[k] for k in ('target', 'task_type', 'metric', 'features', 'ids')}
            config = (candidate or {}).get('identity', {}).get('config', {})
            if (candidate and candidate.get('applicable') and config.get('task') == contract
                    and config.get('anchors') == anchor and candidate.get('validation_periods') == expected):
                anchor_summary = candidate
            if not anchor_summary or anchor_summary.get('status') != 'completed':
                pending.insert(0, 'Сразу после configure_task вызвать run_regression_anchors: заново измерить два закреплённых рецепта, '
                    'три temporal-источника и их фиксированный ансамбль. Частичный regression_anchors.json сохраняет готовые '
                    'источники; при повторе достроить недостающие, соблюдая резерв экспорта. Исторические 39.9274 не являются новым замером.')
        reports = []
        for plan_path in (run / 'temporal').glob('*/plan.json'):
            report_path = plan_path.with_name('report.json')
            if not report_path.exists():
                continue
            plan = json.loads(plan_path.read_text(encoding='utf-8'))['config']
            if (plan['time_column'] == state.get('split_column') and
                    (not expected or plan['validation_periods'] == expected)):
                reports.append((plan, json.loads(report_path.read_text(encoding='utf-8'))))
        if is_temporal:
            if len(expected) < 2:
                skipped.append('Недостаточно периодов для двух внешних фолдов с внутренним прошлым периодом')
            elif total_budget < 900:
                skipped.append('Короткий общий бюджет: multi-period/history этап не обязателен; это ограниченная проверка')
            else:
                distinct = {json.dumps({k: p.get(k) for k in ('recipe', 'key_columns', 'key_variants',
                    'group_columns', 'group_hierarchy', 'smoothing', 'area_rounding', 'modes')}, sort_keys=True)
                    for p, _ in reports if p.get('kind') != 'ensemble'}
                if len(distinct) < 2:
                    pending.append('Проверить два разных финалиста через run_temporal_suite на одинаковых validation_periods')
                if not any({'baseline', 'history_exact', 'history_groups'}.issubset({r['mode'] for r in d['results']}) for _, d in reports):
                    pending.append('Завершить сравнение baseline / history_exact / history_groups без будущих target')
                if not any('history_hierarchy' in {r['mode'] for r in d['results']} and p.get('key_variants') for p, d in reports):
                    pending.append('Проверить расширенные ключи и group_hierarchy через history_hierarchy=True с past-only сглаживанием')
                if not any('residual_history' in {r['mode'] for r in d['results']} for _, d in reports):
                    pending.append('Проверить residual=True: обучить остаток относительно причинного hist_baseline')
                if not any(p.get('kind') == 'ensemble' for p, _ in reports):
                    pending.append('Проверить blend_temporal_suites для двух сильных разных temporal-пайплайнов на одинаковых строках')
                from temporal_search import catalogue, search_status
                global_selection = catalogue(run, state, expected)
                exploration = search_status(run, state, expected)
                if exploration['continue_search']:
                    pending.append(f"Продолжить поиск: {exploration['trials']}/{exploration['minimum_trials']} начальных temporal-гипотез; "
                        f"после существенного выигрыша {exploration['trials_since_substantial_gain']}/{exploration['patience']} без улучшения. "
                        'Исследуй ключи, сглаживание, остатки и разные рецепты; повтор кеша не считается новым опытом')
                allocation = int(os.getenv('AUTOML_FLAML_BUDGET', '0'))
                if allocation > 0 and not (run / 'flaml_summary.json').exists():
                    ledger_path = run / 'flaml_budget.json'
                    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {}
                    used = ledger.get('charged_seconds', 0)
                    failures = [c for c in ledger.get('calls', []) if c.get('status') == 'failed']
                    if failures:
                        skipped.append('FLAML attempt failed; inspect flaml_budget.json and child logs before retrying. Export the validated reference if no fix is available.')
                    elif allocation - used < 120:
                        skipped.append('FLAML allocation exhausted; retain the best completed temporal pipeline')
                    else:
                        pending.append('После одного сильного одиночного temporal/history suite вызвать run_flaml_suite; сравнить FLAML и reference на тех же внешних строках до длительной серии похожих опытов')
                raw_flaml = {d['suite_id'] for p, d in reports
                    if p.get('backend') == 'flaml_v1' and any(r.get('mode') == 'flaml' for r in d.get('results', []))}
                tested_raw_blends = [d['suite_id'] for p, d in reports if p.get('kind') == 'ensemble'
                    and any(m.get('suite_id') in raw_flaml and m.get('mode') == 'flaml'
                            for m in p.get('members', []))]
                if raw_flaml:
                    flaml_blend = dict(raw_flaml_suite_ids=sorted(raw_flaml),
                        completed_fixed_blend_ids=tested_raw_blends,
                        note='Raw FLAML can diversify errors even when its standalone score loses. '
                             'Use fixed equal weights and matching temporal rows, never fitted outer-fold weights.')
                    if not tested_raw_blends:
                        pending.append('Обязательно один раз проверить фиксированный blend_temporal_suites с сырым mode="flaml", '
                            'даже если FLAML-suite выбрал reference. Передать member_modes явно: до трёх single anchor-источников '
                            'плюс FLAML (всего до четырёх); при отсутствии anchors использовать один сильный reference + raw FLAML. '
                            'Не передавать готовый ансамбль как вложенного участника и не дублировать FLAML reference с его источником.')
                allocation = int(os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0'))
                if allocation > 0 and not (run / 'lightautoml_summary.json').exists():
                    ledger = current_json('lightautoml_budget.json') or {}
                    used = ledger.get('charged_seconds', 0)
                    failures = [c for c in ledger.get('calls', []) if c.get('status') == 'failed']
                    if failures:
                        skipped.append('LightAutoML attempt failed; inspect lightautoml_budget.json and child logs before retrying. '
                            'Do not repeat an unchanged failing call; preserve the validated reference.')
                    elif allocation - used < 120:
                        skipped.append('LightAutoML allocation exhausted; retain the best completed temporal pipeline')
                    else:
                        pending.append('После anchors и одного сильного одиночного temporal/history suite один раз вызвать '
                            'run_lightautoml_suite; algorithms=["linear_l2","lgb","cb"], time_limit не больше остатка '
                            'LightAutoML-бюджета. Сравнить на тех же внешних строках; поиск и preprocessing только во внутреннем прошлом.')
                raw_lightautoml = {d['suite_id'] for p, d in reports
                    if p.get('backend') == 'lightautoml_v1'
                    and any(r.get('mode') in {'lightautoml', 'lightautoml_refit'} for r in d.get('results', []))}
                tested_lightautoml_blends = [d['suite_id'] for p, d in reports if p.get('kind') == 'ensemble'
                    and any(m.get('suite_id') in raw_lightautoml and m.get('mode') in {'lightautoml', 'lightautoml_refit'}
                            for m in p.get('members', []))]
                if raw_lightautoml:
                    lightautoml_blend = dict(raw_lightautoml_suite_ids=sorted(raw_lightautoml),
                        completed_fixed_blend_ids=tested_lightautoml_blends,
                        note='One fixed equal-weight blend checks complementary errors even when standalone LightAutoML loses. '
                             'No weight fitting or model selection on outer-fold targets.')
                    if not tested_lightautoml_blends:
                        pending.append('Один раз проверить фиксированный blend_temporal_suites с лучшим сырым mode="lightautoml" или "lightautoml_refit", '
                            'даже если LightAutoML-suite выбрал reference. Передать member_modes явно; использовать 2–4 '
                            'одиночных источника на тех же фолдах. Не вкладывать готовые ансамбли и не дублировать reference-алиас; '
                            'веса равные, без подгонки по внешней validation.')
                if (allocation >= 2700 and state.get('target') == 'site_eui' and
                        any(r['reference_id'] == 'regression_control_lgbm_compact' for r in refs)):
                    ledger = current_json('lightautoml_budget.json') or {}
                    if not any(c.get('status') == 'failed' for c in ledger.get('calls', [])):
                        profiles = [('inherit', False, False), ('compact_strings', False, False), ('compact_strings', True, True)]
                        completed = {(p.get('feature_mode', 'inherit'), p.get('tuning', False), p.get('refit_full', False))
                                     for p, _ in reports if p.get('backend') == 'lightautoml_v1'}
                        for mode, tune, refit in profiles:
                            if (mode, tune, refit) not in completed:
                                left = allocation - ledger.get('charged_seconds', 0)
                                if left >= 120:
                                    pending.append('Парное исследование LightAutoML: run_lightautoml_suite с тем же reference_suite_id, '
                                        f'feature_mode="{mode}", tuning={tune}, refit_full={refit}, max_tuning_trials=8, '
                                        f'time_limit={min(900, int(left))}. Сравнить только текущие одинаковые внешние годы; '
                                        'refit добавляет отдельный режим без подбора на внешней validation.')
                                else:
                                    skipped.append('LightAutoML paired profile skipped: allocation exhausted')
                selection_path = run / 'final_selection.json'
                selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else {}
                report_ids = {d['suite_id'] for _, d in reports}
                if not report_ids or selection.get('kind') != 'temporal' or selection.get('suite_id') != global_selection['recommended_suite']:
                    pending.append('Выбрать глобально лучший сопоставимый temporal-кандидат по mean/latest/worst, включая ансамбли')
        elif state.get('task_type') == 'classification' and state.get('metric') == 'f1_macro' and total_budget >= 900:
            if not (run / 'classification_summary.json').exists():
                pending.append('Завершить OOF-сравнение финалистов и ансамбля')
            selection_path = run / 'final_selection.json'
            selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else {}
            if selection.get('kind') not in {'classification_oof', 'autogluon_oof'}:
                pending.append('Явно выбрать проверенный classification_oof через select_final_candidate')
            allocation = int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0'))
            if allocation > 0 and state.get('split') == 'random' and not (run / 'autogluon_summary.json').exists():
                ledger = run / 'autogluon_budget.json'
                used = json.loads(ledger.read_text(encoding='utf-8'))['charged_seconds'] if ledger.exists() else 0
                if allocation - used >= 120:
                    pending.append('После OOF вызвать run_autogluon_suite на лучшем reference_suite_id; сравнить AutoGluon и смеси на тех же строках')
                else:
                    skipped.append('AutoGluon allocation exhausted; export the already validated reference')
            auto = current_json('autogluon_summary.json')
            if auto and auto.get('suite_id') and state.get('split') == 'random':
                shift = current_json('adversarial_summary.json')
                shift_audit = current_json('classification_shift_audit.json')
                if not shift or not shift.get('diagnostic_id'):
                    pending.append('После завершённого AutoGluon вызвать adversarial_validation, затем audit_classification_shift '
                        'для сохранённых OOF. Это условная диагностика test-like среза, не новый независимый holdout; '
                        'она не обучает модели, веса или Macro F1 множители и не меняет победителя.')
                elif (not shift_audit or shift_audit.get('suite_id') != auto['suite_id']
                        or shift_audit.get('shift_id') != shift['diagnostic_id']):
                    pending.append('Вызвать audit_classification_shift(suite_id="' + auto['suite_id'] +
                        '", shift_id="' + shift['diagnostic_id'] + '") по cached outer OOF; '
                        'сопоставить reference/AutoGluon/смеси и редкие классы. Это диагностический срез ранее '
                        'использованных OOF, не независимая оценка и не основание подбирать веса на срезе.')
        elif not any(e.get('algorithm') == 'ensemble' for e in experiments):
            pending.append('Проверить равновесный ансамбль разных кандидатов')
    if state.get('target') and total_budget >= 900:
        from extended_research import stages
        extended_pending, extended_skipped = stages(run, state, experiments)
        pending = extended_pending + pending
        skipped += extended_skipped
    return dict(pending=pending, skipped=skipped, complete=not pending, improvement_goal=improvement,
                global_temporal_selection=global_selection, temporal_search=exploration,
                regression_anchors=anchor_summary, flaml_fixed_blend=flaml_blend,
                lightautoml_fixed_blend=lightautoml_blend,
                classification_shift_audit=shift_audit,
                encoding_comparison=encoding_comparison(state, reports) if state.get('target') else None,
                reference_recipes=[{k: r[k] for k in ('reference_id', 'algorithm', 'historical_score')} for r in refs],
                temporal_suggestions=temporal_suggestions(profile, state),
                note='Historical scores never enter the current experiment registry. Stages are bounded by step/time/API limits and the export reserve.')


def save_progress(run, state, profile, total_budget):
    result = progress(run, state, profile, total_budget)
    Path(run, 'research_progress.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['encoding_comparison'] is not None:
        Path(run, 'encoding_comparison.json').write_text(json.dumps(result['encoding_comparison'], ensure_ascii=False, indent=2), encoding='utf-8')
    if result['global_temporal_selection'] is not None:
        Path(run, 'temporal_search.json').write_text(json.dumps(dict(ranking=result['global_temporal_selection'],
            search=result['temporal_search']), ensure_ascii=False, indent=2), encoding='utf-8')
    return result
