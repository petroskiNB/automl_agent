"""Opt-in stage tracking for diverse models; never compares incompatible metrics."""
import json
import os
from pathlib import Path


def stages(run, state, experiments):
    if os.getenv('AUTOML_EXTENDED_RESEARCH') != '1':
        return [], []
    run, pending, skipped = Path(run), [], []
    tabm = [e for e in experiments if e.get('algorithm') == 'torch_tabm']
    pairs = {}
    for experiment in tabm:
        params = dict(experiment['params'])
        classification = state.get('task_type') == 'classification'
        if classification:
            if params.get('numeric_preprocessing', 'robust') != 'quantile' or params.get('num_embedding', 'ple') != 'ple':
                continue
            embedding = params.pop('auto_class_weights', None) or 'None'
        else:
            embedding = params.pop('num_embedding', 'ple')
        key = json.dumps([params, experiment.get('feature_code', '')], sort_keys=True)
        pairs.setdefault(key, set()).add(embedding)
    if state.get('task_type') == 'classification' and not any({'None', 'SqrtBalanced'} <= modes for modes in pairs.values()):
        pending.append('Приоритет: парный TabM PLE numeric_preprocessing="quantile" без весов / '
            'auto_class_weights="SqrtBalanced", одинаковые feature_code/seed/параметры. Старую raw none/PLE пару не повторять.')
    elif state.get('task_type') != 'classification' and not any({'none', 'ple'} <= modes for modes in pairs.values()):
        pending.append('Приоритет: парный train_torch_tabm num_embedding="none" / "ple" с одинаковыми '
            'feature_code, seed и остальными параметрами; CPU k=8,width=128..256,layers=2,iterations=40..60. '
            'Сначала выполнить этот новый опыт, затем очередные вариации старых бустингов.')
    def reports(kind):
        values = []
        for path in sorted((run/kind).glob('*/report.json')):
            try:
                config = json.loads(path.with_name('plan.json').read_text(encoding='utf-8'))['config']
                report = json.loads(path.read_text(encoding='utf-8'))
                if all(config['task'].get(k) == state.get(k) for k in ('target', 'task_type', 'metric', 'features', 'ids')):
                    values.append((path.parent.name, config, report))
            except (OSError, ValueError, KeyError):
                continue
        return values
    if state.get('task_type') == 'classification':
        original = [r for r in reports('classification_oof') if not r[1].get('backend')]
        plans = sorted((run/'classification_fold_plans').glob('*/plan.json'))
        frozen_hint = ', split_plan_id="' + plans[0].parent.name + '"' if len(plans) == 1 else ''
        singles = [e for e in experiments if e.get('algorithm') != 'ensemble' and e.get('score') is not None]
        if singles:
            best = max(singles, key=lambda e: e['score'])
            def signature(recipe):
                params = dict(recipe.get('params', {}))
                params.setdefault('seed', 42)
                if params.get('auto_class_weights') is None:
                    params.pop('auto_class_weights', None)
                return json.dumps([recipe.get('algorithm'), params, recipe.get('feature_code', '')], sort_keys=True)
            if not any(any(signature(recipe) == signature(best) for recipe in config.get('recipes', []))
                       for _, config, _ in original):
                pending.append(f'Перепроверить текущий лучший single experiment_id={best["id"]} через '
                    'run_classification_suite с сильным независимым кандидатом' + frozen_hint +
                    '. Старый OOF состав не проверяет найденный позже рецепт; holdout и OOF числа не сравнивать напрямую.')
        if tabm and not any(any(r.get('algorithm') == 'torch_tabm' for r in config.get('recipes', []))
                            for _, config, _ in original):
            pending.append('Приоритет: run_classification_suite с сохранённым TabM и сильным бустингом; '
                           '3 одинаковых внешних фолда, сравнить сети, дерево, смеси и Macro F1 правила.')
        automatic = reports('autogluon_oof')
        profiles = {config.get('model_profile', 'boosting') for _, config, _ in automatic}
        cap = int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0'))
        ledger = run/'autogluon_budget.json'
        used = json.loads(ledger.read_text(encoding='utf-8')).get('charged_seconds', 0) if ledger.exists() else 0
        if original and cap > 0:
            for profile in ('diverse',):
                if profile not in profiles:
                    if cap-used >= 120:
                        reference = automatic[0][1]['reference_suite_id'] if automatic else '<один выбранный original OOF suite>'
                        pending.append(f'Приоритет: run_autogluon_suite(reference_suite_id="{reference}", '
                            f'model_profile="{profile}",time_limit={min(1800, int(cap-used))},bag_folds=3,stack_levels=0). '
                            'Оба профиля сравнивать на одном reference и внешних строках; общий AG бюджет не увеличивается.')
                    else:
                        skipped.append('Недостаточно AG allocation для второго профиля; результат не измерен')
        if original and not any(config.get('backend') == 'experts_v1' for _, config, _ in reports('classification_oof')):
            skipped.append('OVR уже дал минимальный выигрыш в предыдущем исследовании; в этом цикле необязателен. '
                'При остатке бюджета доступен run_classification_experts на лучшем совместимом источнике '
                '(reference_kind=classification_oof или autogluon_oof), class_labels до трёх слабых классов, '
                'time_limit=1200. Backdoor/Analysis/DoS — только если эти метки есть. Настройка на inner calibration.')
    elif state.get('task_type') == 'regression' and state.get('split') == 'time':
        temporal = reports('temporal')
        networks = {sid for sid, config, _ in temporal if config.get('recipe', {}).get('algorithm') == 'torch_tabm'}
        if tabm and not networks:
            pending.append('Приоритет: run_temporal_suite с сохранённым TabM на тех же validation_periods, '
                           'исходные числовые/строковые признаки, past-only preprocessing/history.')
        required = {'floor_area', 'State_Factor', 'facility_type', 'building_class', 'year_built'}
        if required <= set(state.get('features', [])) and temporal:
            if not any(config.get('backend') == 'specialists_v1' for _, config, _ in temporal):
                pending.append('Приоритет после anchors: run_regression_specialists на сильном SINGLE temporal source, '
                    'area_tolerance=0.01,tail_quantile=0.95,time_limit=1800; четыре парных режима, затем общий рейтинг.')
        if networks and not any(any(m.get('suite_id') in networks for m in config.get('members', []))
                                for _, config, _ in temporal):
            pending.append('Проверить фиксированный blend_temporal_suites: TabM и независимый сильный single '
                           'на тех же годах; явно member_modes, исходный победитель остаётся кандидатом.')
    return pending, skipped
