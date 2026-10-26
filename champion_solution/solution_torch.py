"""Isolated PyTorch version of the Qwen agent. See PYTORCH_README.md."""
import argparse
import copy
import hashlib
import importlib
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from torch_resources import configure

ROOT = Path(__file__).resolve().parent


def task_kind(task, requested="auto"):
    """Route optional tools from an explicit task description, never target dtype.

    Inspect the task section before the data/dictionary sections, where comparisons
    with other problem types can occur. Ambiguous descriptions keep AutoGluon off
    unless the caller explicitly chooses --task-type.
    """
    text = Path(task).read_text(encoding="utf-8-sig")
    section = re.search(r"(?im)^##\s+(?:задача|task|objective)\s*\r?$", text)
    intro = text[section.end():] if section else text
    intro = re.split(r"(?m)^##\s+", intro, maxsplit=1)[0][:6000]
    kinds = set()
    if re.search(r"(?i)\b(?:regress\w*|регресс\w*)", intro):
        kinds.add("regression")
    if re.search(r"(?i)\b(?:classif\w*|multiclass\w*|классифик\w*)", intro):
        kinds.add("classification")
    detected = next(iter(kinds)) if len(kinds) == 1 else "unknown"
    if requested != "auto":
        if detected != "unknown" and requested != detected:
            raise ValueError(f"--task-type {requested} conflicts with the task description ({detected})")
        return requested
    return detected


def effective_autogluon_budget(task_type, requested, platform, time_budget):
    if requested is not None and requested < 0:
        raise ValueError("AutoGluon budget cannot be negative")
    # Regression is isolated even if a wrapper or shell supplied a positive cap.
    if task_type == "regression":
        return 0
    budget = requested if requested is not None else (3600 if platform and task_type == "classification" else 0)
    if budget and task_type != "classification":
        raise ValueError("AutoGluon requires an explicit classification task; use --task-type classification if ambiguous")
    if budget and budget > min(7200, time_budget - 120):
        raise ValueError("AutoGluon budget must leave at least 120 seconds for controller/export")
    return budget


def effective_flaml_budget(task_type, requested, platform, time_budget):
    budget = requested if requested is not None else (1800 if platform and task_type == 'regression' else 0)
    if budget < 0 or (budget and (task_type != 'regression' or budget > min(7200, time_budget - 120))):
        raise ValueError('FLAML requires regression and a nonnegative budget leaving at least 120 seconds for export')
    return budget


def effective_lightautoml_budget(task_type, requested, time_budget):
    """Optional first-release backend; never consumes budget unless requested."""
    budget = requested
    if budget < 0 or (budget and (task_type != 'regression' or
            not 120 <= budget <= min(3600, time_budget - 180))):
        raise ValueError('LightAutoML requires regression and a budget of 120..3600 seconds '
                         'leaving at least 180 seconds for export; use 0 to disable')
    return budget


def controller():
    return importlib.import_module("legacy_solution" if (ROOT/"legacy_solution.py").is_file() else "solution")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_task(task, destination):
    """Independent input copies; no writes to original task/runs/submission."""
    discover_files = controller().discover_files
    task, destination = Path(task).resolve(), Path(destination).resolve()
    if destination == task.parent or task.is_relative_to(destination):
        raise ValueError("Experimental output must be separate from the source task directory")
    inputs = dict(discover_files(task.parent), **{"task.md": task})
    manifest = {name: sha256(path) for name, path in inputs.items()}
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "source_inputs.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Source data changed; use a new experimental output directory")
        if any(not (destination/name).is_file() or sha256(destination/name) != digest for name, digest in manifest.items()):
            raise ValueError("Staged inputs changed or missing; use a new output directory")
    else:
        if list(destination.iterdir()):
            raise ValueError("Refusing to stage inputs into a nonempty unrecognized directory")
        for name, source in inputs.items():
            shutil.copy2(source, destination / name)
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination / "task.md"


def limited_run_python(code, folder, log_dir, step, timeout):
    """Like legacy executor, but respects allocation and enforces a wall deadline."""
    try:
        compile(code, "<experiment>", "exec")
    except SyntaxError as error:
        return {"exit_code": 1, "error": str(error)}
    script = Path(log_dir) / f"step_{step:03d}.py"
    script.write_text(code, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    allocation = configure()
    env.update({k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS")})
    env.update(PYTHONIOENCODING="utf-8", AUTOML_CPU_THREADS=str(allocation),
               AUTOML_WORKER_DEADLINE=str(time.time()+max(0, timeout-3)))
    started = time.monotonic()
    out_path, err_path = script.with_suffix(".stdout.txt"), script.with_suffix(".stderr.txt")
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        try:
            process = subprocess.run([sys.executable, "-u", str(script)], cwd=folder, env=env,
                                     stdout=out, stderr=err, timeout=max(.001, timeout))
            result = {"exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            result = {"exit_code": -1, "error": f"Step exceeded {timeout:.1f}s; previous submission preserved"}
    for name, path in (("stdout", out_path), ("stderr", err_path)):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size-16000))
            result[name] = stream.read().decode("utf-8", errors="replace")
    result["seconds"] = time.monotonic()-started
    return result


def run_worker(action, arguments, folder, files, log_dir, step, timeout, module_name=None):
    solution = controller()
    if action == 'train':
        from training_contract import validate_arguments, rejection_memory
        rejected = rejection_memory(log_dir, arguments)
        if rejected:
            return dict(exit_code=1, repeated_failure=True,
                        error='Identical training request was already rejected: ' + rejected['error'])
        try:
            arguments = validate_arguments(arguments, solution.research_state(Path(log_dir)).get('task_type'))
        except (ValueError, TypeError, SyntaxError) as error:
            rejection_memory(log_dir, arguments, error)
            return dict(exit_code=1, error=str(error), repair='Use the algorithm-specific tool and remove unsupported fields.')
    original = solution.run_python
    solution.run_python = limited_run_python
    try:
        target = module_name or "xgb_backend.py"
        code_actions = {"inspect", "configure", "train", "blend", "finalize"}
        if action in code_actions or target == "diagnostics.py":
            target = "torch_backend.py"
        result = solution.run_research_original(action, arguments, folder, files, log_dir, step, timeout,
                                               module_name=target if target == "torch_backend.py" else "torch_suites.py")
        if action == 'train' and result.get('exit_code') == 1:
            failure = str(result.get('stderr', result.get('error', '')))
            if any(tag in failure for tag in ('ValueError:', 'TypeError:', 'SyntaxError:', 'KeyError:')):
                from training_contract import rejection_memory
                rejection_memory(log_dir, arguments, failure)
        if result.get("exit_code") == -1:
            if action == 'run_autogluon_suite':
                ledger_path = Path(log_dir)/'autogluon_budget.json'
                if ledger_path.is_file():
                    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
                    for entry in ledger.get('calls', []):
                        if entry.get('status') == 'running':
                            entry.update(status='timed_out', charged_seconds=entry.get('reserved_seconds', 0),
                                         reason='Worker killed by controller; complete fold checkpoints can be recovered')
                    temporary = ledger_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')
                    temporary.replace(ledger_path)
            task_lock = Path(folder)/".agent.lock"
            if task_lock.exists() and task_lock.read_text().strip() == str(os.getpid()):
                abandoned = {"run_classification_suite": ".classification.lock", "run_temporal_suite": ".temporal.lock", 'blend_temporal_suites': '.temporal.lock', 'run_regression_anchors': '.temporal.lock', 'run_autogluon_suite': '.autogluon.lock', 'run_flaml_suite': '.flaml.lock', 'run_lightautoml_suite': '.lightautoml.lock'}.get(action)
                if abandoned:
                    (Path(log_dir)/abandoned).unlink(missing_ok=True)
                if action == 'run_saved_recipe_oof':
                    (Path(log_dir)/'.classification.lock').unlink(missing_ok=True)
        return result
    finally:
        solution.run_python = original


def install_hooks(task_type="unknown"):
    solution = controller()
    if getattr(solution, "_torch_hooks", False):
        return
    solution._torch_hooks = True
    solution.run_research_original = solution.run_research
    solution.run_research = run_worker
    solution.run_python = limited_run_python
    original_fingerprint = solution.fingerprint
    def fingerprint(task, files):
        result = original_fingerprint(task, files)
        for name in ("torch_resources.py", "torch_model.py", "torch_backend.py", "torch_suites.py", "solution_torch.py", "diagnostics.py", "legacy_solution.py", "release_audit.py", "research_policy.py", "reference_recipes.py", "prediction_policy.py",
                     "classification_research.py", "classification_checks.py", "temporal_research.py",
                     "final_output.py", "parallel_research.py", "research_team.py", 'hierarchical_history.py', 'temporal_ensemble.py', 'temporal_search.py', 'temporal_anchors.py', 'training_contract.py', 'lightautoml_refit.py'):
            result[name] = sha256(ROOT/name)
        if task_type == "classification":
            result['autogluon_research.py'] = sha256(ROOT/'autogluon_research.py')
            result['classification_shift_audit.py'] = sha256(ROOT/'classification_shift_audit.py')
        if int(os.getenv('AUTOML_FLAML_BUDGET', '0')) > 0:
            result['flaml_research.py'] = sha256(ROOT/'flaml_research.py')
        if int(os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0')) > 0:
            result['lightautoml_research.py'] = sha256(ROOT/'lightautoml_research.py')
        for name in ('tabm_model.py', 'tabm_vendor.py', 'num_embeddings_vendor.py',
                     'classification_experts.py', 'regression_specialists.py', 'extended_research.py',
                     'classification_fold_plan.py'):
            result[name] = sha256(ROOT/name)
        result["cpu_threads"] = str(configure())
        result['task_profile'] = task_type
        result['autogluon_budget'] = os.getenv('AUTOML_AUTOGLUON_BUDGET', '0')
        result['flaml_budget'] = os.getenv('AUTOML_FLAML_BUDGET', '0')
        result['lightautoml_budget'] = os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0')
        result['extended_research'] = os.getenv('AUTOML_EXTENDED_RESEARCH', '0')
        return result
    solution.fingerprint = fingerprint
    solution.compatible_manifest = lambda saved, current, state: saved == current
    import parallel_research
    parallel_research.run_batch = partial(parallel_research.run_batch, backend="torch_backend", total_threads=configure())
    solution.TOOLS = [tool for tool in copy.deepcopy(solution.TOOLS)
                      if tool['function']['name'] not in {'run_regression_anchors', 'audit_classification_shift'}]
    if task_type == 'regression':
        solution.TOOLS.append(solution.tool_schema('run_regression_anchors',
            'Remeasure three embedded energy-use recipes and their fixed ensemble on current temporal folds. Applies only to site_eui / Year_Factor schema after configure_task. No historical models are loaded.', {}, []))
        solution.SYSTEM_PROMPT += '''
Для известной site_eui-регрессии с Year_Factor после чтения словаря и configure_task
со split="time", split_column="Year_Factor" сначала вызови run_regression_anchors().
Он заново измеряет три сохранённых рецепта CatBoost/LightGBM и их равновесный
ансамбль на текущих прошлых периодах. Исторический RMSE 39.9274 — ориентир,
а не измерение этого запуска и не готовые веса. Смотри regression_anchors.json.
Если инструмент неприменим к схеме, продолжай обычное исследование без повторов.
Если он завершился частично, используй завершённые suite и учитывай остаток времени.
'''
    if task_type == 'regression' and int(os.getenv('AUTOML_FLAML_BUDGET', '0')) > 0:
        solution.TOOLS.append(solution.tool_schema('run_flaml_suite',
            'FLAML regression search inside the past of an existing single temporal suite; compare on identical outer rows. time_limit covers ALL fold searches and refits together.', {
            'reference_suite_id': {'type': 'string'},
            'time_limit': {'type': 'integer', 'minimum': 120, 'maximum': 3600},
            'estimator_list': {'type': 'array', 'minItems': 1, 'maxItems': 5, 'uniqueItems': True,
                               'items': {'type': 'string', 'enum': ['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost']}},
            'feature_code': {'type': 'string', 'description': 'Omit to inherit reference features; empty string explicitly selects raw features.'}}, ['reference_suite_id']))
        solution.SYSTEM_PROMPT += '''
FLAML включён отдельным суммарным бюджетом только для временной regression/RMSE.
После стартовых anchors (если применимы) и одного сильного run_temporal_suite
с history_hierarchy вызови run_flaml_suite(reference_suite_id=<одиночный suite>,
time_limit=1800, estimator_list=["lgbm","xgboost","rf","extra_tree"]).
Сделай это ДО длительной серии похожих temporal-гипотез. Источник — завершённый
актуальный одиночный suite, выигравший в baseline/history_exact/history_groups/
history_hierarchy; ансамбль, history_gate и residual_history здесь не поддержаны.
Внешние фолды копируются из reference; поиск FLAML использует только внутреннее
прошлое, затем выбранный рецепт переобучается на всём прошлом внешнего фолда.
time_limit относится ко всем поискам и refit вместе. Встроенное ансамблирование
FLAML отключено. Если feature_code не задан, сохраняются признаки reference;
пустая строка означает сырые признаки. Не меняй сразу и признаки, и семейства.
Результат находится в обычном temporal-треке: сравни mean/latest/worst RMSE.
Даже если отдельный FLAML проиграл и selected_mode="reference", обязательно
один раз проверь равновесный blend_temporal_suites с сильным одиночным suite,
явно задав member_modes=[<режим сильного suite>, "flaml"]. Слабая отдельная
модель может улучшить ансамбль. Сначала проверь фиксированную смесь без подбора весов.
Допустимы 2–4 разных одиночных suite: три источника anchors и raw FLAML можно
объединить через их suite_ids и явные member_modes. Вложенные ансамбли запрещены.
Если выиграл mode="reference", это точная копия исходного кандидата, не новая
модель для дублирования в ансамбле. Все результаты учитывает общий temporal-рейтинг.
Выбирай через select_final_candidate(kind="temporal", suite_id=...). Финальный
refit повторяет рецепт последнего внешнего фолда на всём train без нового поиска.
Сохрани резерв на экспорт. При ошибке FLAML объясни причину и сохрани проверенный
reference; не обходи бюджет через run_python и не устанавливай пакеты.
'''
    if task_type == 'regression' and int(os.getenv('AUTOML_LIGHTAUTOML_BUDGET', '0')) > 0:
        solution.TOOLS.append(solution.tool_schema('run_lightautoml_suite',
            'LightAutoML regression candidate inside the past of an existing single temporal suite; '
            'identical outer rows and a cumulative budget across ALL fold fits plus a saved final inner-holdout pipeline.', {
            'reference_suite_id': {'type': 'string'},
            'time_limit': {'type': 'integer', 'minimum': 120, 'maximum': 3600},
            'algorithms': {'type': 'array', 'minItems': 1, 'maxItems': 5, 'uniqueItems': True,
                           'items': {'type': 'string', 'enum': ['linear_l2', 'lgb', 'cb', 'lgb_tuned', 'cb_tuned']}},
            'feature_mode': {'type': 'string', 'enum': ['inherit', 'compact_strings'],
                             'description': 'compact_strings uses the embedded compact energy-use recipe with original category strings; no manual rewriting.'},
            'tuning': {'type': 'boolean', 'description': 'Add bounded lgb_tuned/cb_tuned inside past only.'},
            'max_tuning_trials': {'type': 'integer', 'minimum': 1, 'maximum': 12},
            'refit_full': {'type': 'boolean', 'description': 'Also measure lightautoml_refit: fixed inner-selected iterations/C/weights, fresh target-free preprocessing, all past rows.'},
            'feature_code': {'type': 'string', 'description': 'Omit to inherit reference features; empty string explicitly selects raw features.'}}, ['reference_suite_id']))
        solution.SYSTEM_PROMPT += '''
LightAutoML включён отдельным суммарным бюджетом только для temporal regression/RMSE.
После anchors возьми один и тот же сильный одиночный source. Для site_eui и
бюджета >=2700 последовательно сравни три вызова, time_limit=900 каждый:
1) feature_mode="inherit", tuning=False, refit_full=False;
2) feature_mode="compact_strings", tuning=False, refit_full=False;
3) feature_mode="compact_strings", tuning=True, max_tuning_trials=8, refit_full=True.
Первое сравнение изолирует категориальные признаки, второе — внутренний подбор.
В третьем вызове есть парные mode="lightautoml" и "lightautoml_refit": один и
тот же внутренний выбор, затем фиксированный refit на всём прошлом без внешних
validation. Сравни оба режима. Для малого бюджета выбери один осмысленный вызов
с time_limit не больше остатка. Не называй весь выделенный бюджет потраченным:
фактические секунды находятся в lightautoml_budget.json.
Используй существующий завершённый source с baseline/history_exact/history_groups/
history_hierarchy; ансамбли, history_gate и residual_history не поддержаны.
Внешние фолды и признаки сохраняются от reference; вся предобработка, выбор
модели и внутреннее объединение LightAutoML используют только прошлое внутри
train внешнего фолда. Внешняя validation не передаётся для подбора.
time_limit покрывает все внешние fit и один финальный fit вместе, а не каждый fit.
Mode lightautoml использует внутренний holdout. При refit_full=True дополнительный
lightautoml_refit переобучает все удержанные компоненты на всём прошлом:
деревья/C/веса заморожены внутренним выбором, preprocessing заново обучается
только на past без target encoding. Нет train=valid и нового поиска на refit.
Оба финальных pipeline готовятся внутри бюджета suite; экспорт их не обучает.
История для query использует только строго прошлые targets.
При пропущенном
feature_code сохраняются признаки reference, пустая строка означает сырые признаки.
Даже если одиночный LightAutoML проиграл и selected_mode="reference", обязательно
один раз проверь фиксированный равновесный blend_temporal_suites с сильным single:
member_modes=[<режим сильного source>, <лучший lightautoml или lightautoml_refit>]. Допускаются 2–4 одиночных
компонента; вложенные ансамбли и дублирование reference-алиаса запрещены.
Не подбирай веса по внешним фолдам. Сравни mean/latest/worst, выбирай общий лучший
temporal suite; финальный экспорт использует тот же рецепт без нового поиска.
LightAutoML — дополнительный кандидат, а не гарантированное улучшение. При сбое
сохрани завершённый reference и проверь lightautoml_budget.json и логи ребёнка.
Не обходи бюджет через run_python, не устанавливай пакеты. Оставь резерв экспорта.
'''
    enabled = task_type == "classification" and int(os.getenv('AUTOML_AUTOGLUON_BUDGET', '0')) > 0
    if not enabled:
        selection = next(t['function'] for t in solution.TOOLS if t['function']['name'] == 'select_final_candidate')
        selection['parameters']['properties']['kind']['enum'].remove('autogluon_oof')
    if enabled:
        solution.TOOLS.append(solution.tool_schema('recover_autogluon_suite',
            'Reconstruct a report from ALL completed saved outer folds after interruption. Exact current runtime/data identity required; does not fit models or refund consumed budget.', {
                'suite_id': {'type': 'string'}}, ['suite_id']))
        solution.TOOLS.append(solution.tool_schema('run_autogluon_suite',
            'Compare AutoGluon and reference on identical outer classification folds; inner-only blend calibration. time_limit is for ALL fits together, not per model.', {
            'reference_suite_id': {'type': 'string'},
            'time_limit': {'type': 'integer', 'minimum': 120, 'maximum': 3600},
            'bag_folds': {'type': 'integer', 'enum': [3, 5]},
            'stack_levels': {'type': 'integer', 'enum': [0, 1]},
            'model_profile': {'type': 'string', 'enum': ['boosting', 'diverse']},
            'feature_code': {'type': 'string'}}, ['reference_suite_id']))
        solution.TOOLS.append(solution.tool_schema('audit_classification_shift',
            'Compare saved AutoGluon suite OOF probabilities on full OOF and the frozen test-like slice of an existing adversarial-validation run. Diagnostic only; does not train or change selection.', {
            'suite_id': {'type': 'string'}, 'shift_id': {'type': 'string'}}, ['suite_id', 'shift_id']))
        solution.SYSTEM_PROMPT += '''
AutoGluon включён отдельным суммарным бюджетом. Только для classification/f1_macro
с random stratified split: сначала baseline и завершённый run_classification_suite.
Затем обязательно run_autogluon_suite(reference_suite_id=<лучший OOF suite>,
time_limit=1800, bag_folds=3, stack_levels=0). Сравни reference, autogluon и смеси
по одним внешним строкам; правила смеси обучаются только на внутренней калибровке.
У time_limit общий смысл: все внутренние и внешние обучения вместе, а не каждый fit.
Не запускай близко к дедлайну. История и остаток регрессии этим инструментом пока
не поддержаны. Не используй AutoGluon через run_python и не устанавливай пакеты.
Если выиграл новый suite, select_final_candidate(kind="autogluon_oof", suite_id=...).
В этом suite есть неизменённый reference, поэтому при его победе сохраняется его
экспорт. Сохраняются целые папки predictor; final_model.joblib сам по себе недостаточен.
После завершённого AutoGluon suite вызови adversarial_validation, если его ещё нет,
затем audit_classification_shift(suite_id=<AG suite>, shift_id=<adversarial id>).
Инструмент использует только сохранённые OOF-вероятности. Это условный срез OOF
по сходству с test, а не независимый holdout. Сравни Macro F1, редкие классы и
support, опиши ухудшения. Не подбирай по этому срезу веса и не меняй автоматически
победителя; это диагностика устойчивости, а не доказательство роста на платформе.
'''
    train_schema = next(t["function"] for t in solution.TOOLS if t["function"]["name"] == "train_model")
    properties = train_schema["parameters"]["properties"]
    properties["algorithm"]["enum"] += ["torch_mlp", "torch_resnet", 'torch_tabm']
    properties["params"]["properties"].update({
        "width": {"type": "integer", "minimum": 8, "maximum": 1024},
        "layers": {"type": "integer", "minimum": 1, "maximum": 8},
        "batch_size": {"type": "integer", "minimum": 16, "maximum": 8192},
        "dropout": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
        "weight_decay": {"type": "number", "minimum": 0, "maximum": 1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "max_categories": {"type": "integer", "minimum": 2, "maximum": 50000},
        "loss": {"type": "string", "enum": ["auto", "mse", "mae", "huber"]},
        "target_transform": {"type": "string", "enum": ["standard", "log1p"]},
        'ensemble_size': {'type': 'integer', 'minimum': 2, 'maximum': 32},
        'num_embedding': {'type': 'string', 'enum': ['none', 'ple']},
        'numeric_preprocessing': {'type': 'string', 'enum': ['robust', 'quantile']},
        'embedding_dim': {'type': 'integer', 'minimum': 4, 'maximum': 64},
        'n_bins': {'type': 'integer', 'minimum': 2, 'maximum': 128}})
    from training_contract import split_tools
    original_train = next(t for t in solution.TOOLS if t['function']['name'] == 'train_model')
    solution.TOOLS = [t for t in solution.TOOLS if t['function']['name'] != 'train_model'] + split_tools(original_train)
    solution.SYSTEM_PROMPT = solution.SYSTEM_PROMPT.replace('train_model', 'train_<алгоритм>')
    solution.SYSTEM_PROMPT += '''
Обучение разделено на train_catboost, train_lightgbm, train_xgboost,
train_torch_mlp и train_torch_resnet. Алгоритм определяется именем инструмента.
Не передавай width/loss/target_transform бустингам. Повтор точно такого же
ошибочного запроса отклоняется без нового процесса; сначала исправь параметры.
'''
    solution.SYSTEM_PROMPT += """
Экспериментальная версия PyTorch. Дополнительные алгоритмы: torch_mlp и torch_resnet.
Это компактные MLP с embeddings категорий; torch_resnet содержит residual-блоки, это НЕ TabM.
После быстрого бустингового baseline обязательно сравни torch_mlp (iterations=80,
width=256, layers=3, batch_size=512, learning_rate=0.001, dropout=0.1,
weight_decay=0.0001, seed=42; early_stopping_rounds=10 означает терпение в ЭПОХАХ).
Затем проверь torch_resnet и для классификации SqrtBalanced против отсутствия весов.
Для нейросетей не передавай depth/l2_leaf_reg и параметры деревьев. Допустимы также
max_categories, loss (auto/mse/mae/huber), target_transform (standard/log1p).
Регрессионные loss/target_transform не применимы к классификации.
Preprocessing, словари категорий и статистики таргета обучаются backend только на train.
Сравни одиночную сеть и равновесный ансамбль с лучшим бустингом на том же split.
Финалистов перепроверь с seed=43 по бюджету, не трать всё время на подбор holdout-весов.
Непрерывное продолжение экспериментов доступно через resume; незавершённое обучение
одной сети после аварийного завершения не возобновляется с оптимизатора.
Поддерживаются три исследовательских направления (--researchers 3) и очередь обучения
до двух процессов (--training-jobs 2). Общие CPU-потоки делятся между процессами.
Для классификации сравни сеть и бустинг через run_classification_suite: число эпох,
preprocessing и правила решений выбираются только внутри внешних фолдов.
Временной suite поддерживает CatBoost, LightGBM, XGBoost, MLP и residual MLP.
Внутри каждого периода preprocessing и early stopping используют только прошлое.
Смотри fit_seconds, predict_seconds и peak_rss_mb в результатах. Качество не предполагается.
"""


    if task_type == 'classification':
        solution.TOOLS.append(solution.tool_schema('run_saved_recipe_oof',
            'Recheck imported historical recipe specifications on an audited frozen fold plan. Trains NEW models; historical holdout scores never enter the current registry. No API calls inside the tool.', {
                'split_plan_id': {'type': 'string'},
                'historical_recipe_ids': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1, 'maxItems': 3}},
            ['split_plan_id', 'historical_recipe_ids']))
        suite_tool = next(t['function'] for t in solution.TOOLS if t['function']['name'] == 'run_classification_suite')
        suite_tool['parameters']['properties']['split_plan_id'] = {'type': 'string',
            'description': 'Optional imported, audited exact outer+inner fold plan; permits a single recipe'}
        suite_tool['parameters']['properties']['experiment_ids']['minItems'] = 1
        solution.TOOLS.append(solution.tool_schema('run_classification_experts',
            'Compare OVR CatBoost experts for 1..3 weak classes with an original OOF or AutoGluon suite. Same outer folds; fusion and Macro F1 rules fit inner calibration only. time_limit covers all fits.', {
                'reference_suite_id': {'type': 'string'},
                'reference_kind': {'type': 'string', 'enum': ['classification_oof', 'autogluon_oof']},
                'class_labels': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
                'time_limit': {'type': 'integer', 'minimum': 120, 'maximum': 2400},
                'auto_class_weights': {'type': ['string', 'null'], 'enum': ['Balanced', 'SqrtBalanced', None]}}, ['reference_suite_id']))
    if task_type == 'regression':
        solution.TOOLS.append(solution.tool_schema('run_regression_specialists',
            'Paired energy-use temporal ablation: unchanged single reference, fuzzy building history, feature-gated high-consumption expert and combination. time_limit includes final full-past refits. Requires an original single temporal reference.', {
                'reference_suite_id': {'type': 'string'},
                'area_tolerance': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': .05},
                'tail_quantile': {'type': 'number', 'minimum': .8, 'maximum': .99},
                'time_limit': {'type': 'integer', 'minimum': 120, 'maximum': 2400}}, ['reference_suite_id']))
    solution.SYSTEM_PROMPT += '''
Доступен настоящий авторский TabM 0.0.3: train_torch_tabm, algorithm="torch_tabm".
Это ансамбль внутри одной сети, CPU, обучение с нуля. Старт: iterations=60,
width=256,layers=2,batch_size=256,ensemble_size=8,learning_rate=0.002,
weight_decay=0.0003,num_embedding="ple",embedding_dim=16,n_bins=32.
Backend считает loss каждого участника и усредняет вероятности на inference;
PLE-границы, категории и нормализация fit только на текущей training fold.
Проверь парно num_embedding="none" и "ple" с тем же seed и feature_code.
Подтверди кандидата через OOF/temporal suite и проверь фиксированную смесь с
деревом, даже если сеть отдельно слабее. Сохраняй исходные строковые категории
и числовые признаки; не хэшируй категории без причины. k не число CPU потоков.
'''
    if os.getenv('AUTOML_EXTENDED_RESEARCH') == '1':
        solution.SYSTEM_PROMPT = solution.SYSTEM_PROMPT.replace(
            'После быстрого бустингового baseline обязательно сравни torch_mlp',
            'После выполнения приоритетного расширенного плана при достаточном времени можно сравнить torch_mlp').replace(
            'Затем проверь torch_resnet и для классификации',
            'При оставшемся времени проверь torch_resnet и для классификации')
        solution.SYSTEM_PROMPT = solution.SYSTEM_PROMPT.replace(
            'Затем обязательно run_autogluon_suite(reference_suite_id=<лучший OOF suite>,',
            'Для отдельного старого профиля доступен run_autogluon_suite(reference_suite_id=<лучший OOF suite>,')
        solution.SYSTEM_PROMPT += '''
Включён приоритетный расширенный план. После чтения данных и baseline сначала
TabM none/PLE, затем честное OOF/temporal сравнение с деревом и ансамбль.
Приоритет этого плана выше очередных вариаций depth/seed и старого счётчика
temporal-гипотез. Дедлайн и экспорт приоритетны; явно сообщай о пропущенных этапах.
Не оценивай качество по одному holdout вместо общих фолдов.
'''
        if task_type == 'classification' and enabled:
            solution.SYSTEM_PROMPT += '''
После OOF сравни model_profile="boosting" и "diverse" через
run_autogluon_suite на ОДНОМ reference_suite_id: по 1800 секунд из общего бюджета
3600. diverse добавляет RF/XT/NN_TORCH и best_quality; bag_folds=3,stack_levels=0.
'''
        if task_type == 'classification':
            solution.SYSTEM_PROMPT += '''
Для Macro F1 приоритет — парный TabM PLE numeric_preprocessing="quantile",
auto_class_weights отсутствует против "SqrtBalanced"; остальные параметры,
feature_code и seed одинаковы. Это следующий опыт после уже проверенной пары
сырых none/PLE без весов. Не повторяй эту старую пару по умолчанию.
Если есть classification_fold_plan_summary в памяти, используй его split_plan_id
для всех новых run_classification_suite. Исторические recipes — только спецификации,
не новые метрики. run_saved_recipe_oof позволяет проверить их без прежних моделей.
После нового лучшего single перепроверь ЕГО рецепт на тех же OOF строках;
не экспортируй старое сравнение, игнорируя найденный позже сильный XGBoost.
Затем run_classification_experts на лучшем доступном OOF источнике, time_limit=1200,
class_labels=["Backdoor","Analysis","DoS"] только если эти метки есть в данных.
Исходный reference остаётся кандидатом. Сравни per_class, mean/worst Macro F1.
'''
        elif task_type == 'regression':
            solution.SYSTEM_PROMPT += '''
После anchors один раз run_regression_specialists на сильном single,
area_tolerance=0.01,tail_quantile=0.95,time_limit=1800, затем фиксированная смесь
нового режима и независимого TabM на тех же годах. Не используй ensemble как
reference специалиста.
'''


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--platform", action="store_true", help="Container mode: publish next to task.md")
    parser.add_argument("--time-budget", type=int, default=9900)
    parser.add_argument('--extended-research', action='store_true',
                        help='Prioritize TabM, diverse AutoGluon and nested classification/regression specialists')
    parser.add_argument('--task-type', choices=('auto', 'regression', 'classification'), default='auto',
                        help='Optional task routing override; auto reads the task description')
    parser.add_argument('--autogluon-budget', type=int, default=None,
                        help='Classification allocation in seconds; default local 0, platform classification 3600, regression always 0')
    parser.add_argument('--flaml-budget', type=int, default=None,
                        help='Cumulative temporal regression allocation; local 0, platform regression 1800, classification 0')
    parser.add_argument('--lightautoml-budget', type=int, default=0,
                        help='Opt-in cumulative temporal regression allocation, 120..3600 seconds; default 0 locally and on platform')
    options, remaining = parser.parse_known_args()
    os.environ['AUTOML_EXTENDED_RESEARCH'] = '1' if options.extended_research else '0'
    configure(options.threads)
    if not 0 < options.time_budget <= 10800:
        raise ValueError("Total platform time budget must be in 1..10800 seconds")
    solution = controller()
    if "--help" in remaining or "-h" in remaining:
        print("Experimental options: --threads TOTAL 1..8 (default 8), --output-dir PATH, --task-type auto|regression|classification,")
        print("--autogluon-budget SECONDS (local 0; platform classification 3600; regression always 0). Input copies/results stay separate.")
        print("--flaml-budget SECONDS (local 0; platform regression 1800; classification 0).")
        print("--lightautoml-budget SECONDS (default 0; temporal regression only, opt-in 120..3600).")
        print('--extended-research prioritizes paired TabM, diverse AutoGluon and nested specialists.')
        sys.argv = [sys.argv[0]] + remaining
        solution.main()
        return
    # Original CLI requires positional task first; all its other options remain available.
    if not remaining or remaining[0].startswith("-"):
        raise ValueError("Usage: solution_torch.py TASK.md [--threads 2] [--output-dir PATH] [agent options]")
    source = Path(remaining[0]).resolve()
    kind = task_kind(source, options.task_type)
    budget = effective_autogluon_budget(kind, options.autogluon_budget, options.platform, options.time_budget)
    os.environ['AUTOML_AUTOGLUON_BUDGET'] = str(budget)
    flaml_budget = effective_flaml_budget(kind, options.flaml_budget, options.platform, options.time_budget)
    os.environ['AUTOML_FLAML_BUDGET'] = str(flaml_budget)
    lightautoml_budget = effective_lightautoml_budget(kind, options.lightautoml_budget, options.time_budget)
    os.environ['AUTOML_LIGHTAUTOML_BUDGET'] = str(lightautoml_budget)
    if flaml_budget:
        from importlib.metadata import version
        if version('flaml') != '2.3.6':
            raise ValueError('Use FLAML 2.3.6: python -m pip install "flaml[automl]==2.3.6"')
    if lightautoml_budget:
        from importlib.metadata import version
        if version('lightautoml') != '0.4.2':
            raise ValueError('Use LightAutoML 0.4.2 in a separate compatible environment')
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
        except (ImportError, OSError, AttributeError) as error:
            raise ValueError('LightAutoML dependencies are not ready; run check_lightautoml.py in the same environment. '
                             + type(error).__name__ + ': ' + str(error)) from error
    if kind == 'regression' and options.autogluon_budget:
        print('Regression profile: supplied AutoGluon allocation disabled.', flush=True)
    if budget:
        from importlib.metadata import version
        if version('autogluon.tabular') != '1.3.1':
            raise ValueError('Use AutoGluon 1.3.1, locally via .venv_autogluon')
    install_hooks(kind)
    destination = options.output_dir or ROOT/"outputs"/"pytorch"/(source.parent.name+"_"+hashlib.sha256(str(source).encode()).hexdigest()[:10])
    if options.platform and options.output_dir:
        raise ValueError("--platform and --output-dir cannot be combined")
    staged = source if options.platform else stage_task(source, destination)
    print(f"Experimental task: {staged}\nTotal CPU allocation: {options.threads}; "
          + ("container output next to task.md." if options.platform else "independent input copies and results."), flush=True)
    print(f'Task profile: {kind}; AutoGluon allocation: {budget}s; FLAML allocation: {flaml_budget}s; '
          f'LightAutoML allocation: {lightautoml_budget}s.', flush=True)
    sys.argv = [sys.argv[0], str(staged)] + remaining[1:] + ["--time-budget", str(options.time_budget)]
    # Controller's default 9900 seconds leaves a platform margin below 10800 seconds.
    solution.main()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
