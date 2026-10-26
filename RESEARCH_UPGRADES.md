# Расширенные эксперименты, 25.10.2026

Реализованы четыре направления. Качество на реальных задачах ещё не измерено.
Сохранённый ZIP с платформенным баллом 2284.823 не изменялся.

- TabM: авторская реализация tabm 0.0.3, numerical embeddings 0.0.12, включённые
  в корневые Python-файлы вместе с полными Apache-2.0 лицензиями. Только CPU,
  обучение с нуля, без загрузки весов или pip install на платформе. Поддерживаются
  classification/regression, single split, OOF, temporal history, ансамбли, refit и reload.
  `num_embedding=none/ple`, `ensemble_size=2..32`; k — размер внутреннего ансамбля,
  число потоков по-прежнему задаётся `--threads`. PLE quantiles и preprocessing
  видят только training fold. Loss считается отдельно для каждого участника;
  classification inference усредняет softmax probabilities, не logits.
- AutoGluon: `model_profile=boosting` сохраняет GBM/CAT/XGB; `diverse` добавляет
  RF/XT/NN_TORCH и preset best_quality. Явные bag_folds/stack_levels/CPU/time_limit
  остаются. Это ограниченный профиль best_quality с перечисленными семействами,
  без pretrained foundation models. В отчёте есть реально обученные model_names.
- `run_classification_experts`: до трёх бинарных CatBoost OVR, те же внешние folds
  и вложенные calibration rows. Источник — original classification OOF или AG OOF.
  Сила вмешательства и Macro F1 multipliers на inner calibration, reference остаётся
  кандидатом. Если он победил, экспортирует точный исходный пакет без обучения.
- `run_regression_specialists`: прошлые объекты с близкой площадью внутри групп
  State_Factor/facility_type/building_class/year_built; только строго более ранние годы.
  Отдельный классификатор вероятности высокого потребления и регрессор для него.
  Порог высокого target — train quantile, inference gate видит только признаки/history.
  Четыре режима reference/fuzzy_history/tail_expert/tail_fuzzy. Смешивание калибруется
  внутри прошлого; последние коэффициенты фиксируются для полного refit. Все final
  модели готовятся в бюджете suite, export только читает их. Общий temporal ensemble
  умеет подключать новые режимы и исключает дублирование reference-алиаса.

Новый флаг `--extended-research` включает приоритеты этого исследования в плане Qwen.
Он не увеличивает CPU или время. Прежние платформенные defaults и сохранённые ZIP
не переключены на новый план. Сохранять полноценные run-папки: особенно AG нуждается
в каталогах predictor, а экспертам нужны исходные reference suites.

После повторных API-сбоев ожидание researcher увеличено до 180 секунд, не больше
остатка исследовательского бюджета. До четырёх повторов с паузами 5/10/20/40 секунд;
после них сохранение лучшего доступного результата. Настройки `--api-timeout`
(30..300) и `--api-retries` (0..8), актуальны и для нескольких исследователей.
Это не гарантирует доступность внешнего сервиса; API не вызывался при локальной проверке.

## Первый запуск: классификация

Из корня проекта в терминале с заданным API:

```powershell
.\.venv_autogluon\Scripts\python.exe champion_solution/solution_torch.py tasks/classification/task.md `
  --output-dir outputs/champion_extended_classification_v1 --extended-research `
  --threads 8 --researchers 1 --training-jobs 1 --reviewer --no-analyst `
  --autogluon-budget 3600 --flaml-budget 0 --lightautoml-budget 0 `
  --time-budget 9900 --max-steps 80 --step-timeout 1800 `
  --instructions extended_classification_plan.md
```

## Отдельно: регрессия

```powershell
.\.venv_autogluon\Scripts\python.exe champion_solution/solution_torch.py tasks/regression/task.md `
  --output-dir outputs/champion_extended_regression_v1 --extended-research `
  --threads 8 --researchers 1 --training-jobs 1 --reviewer --no-analyst `
  --autogluon-budget 0 --flaml-budget 0 --lightautoml-budget 0 `
  --time-budget 9900 --max-steps 80 --step-timeout 1800 `
  --instructions extended_regression_plan.md
```

Запускать последовательно. Для этой версии нужны новые run-папки: изменились
fingerprints кода, старые оценки не смешивать через --resume. Продолжение уже нового
запуска возможно с его точным --resume путём при неизменном коде/параметрах.

Отчёты: run_summary.json, research_progress.json, classification_experts_summary.json,
regression_specialists_summary.json, autogluon_summary.json, plan/report/pipelines
в classification_oof/autogluon_oof/temporal. Экспортированный submission.csv —
предсказания, не архив обученных моделей для переноса на платформу.

Авторские источники:
- https://github.com/yandex-research/tabm
- https://github.com/yandex-research/rtdl-num-embeddings
- https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html
