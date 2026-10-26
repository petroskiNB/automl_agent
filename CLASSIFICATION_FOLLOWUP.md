# Классификация: восстановление и продолжение, 26.10.2026

Отчёты обоих старых AutoGluon suites восстановлены без обучения из всех трёх
сохранённых фолдов. Для загрузки старых моделей используется точный исходный код
из неизменённого v9 ZIP. Проверены исходные identity, данные, индексы, порядок
классов, правила смешивания, метрики и SHA-256 файлов моделей. Старый запуск не менялся.

Лучший восстановленный вариант: `diverse + macro_decision`, mean Macro F1
**0.6328643**, worst **0.6289046**, OOF **0.6325865**.

- Отчёт: `outputs/champion_classification_recovered_20261026/agent_runs/recovered_20261026/autogluon_oof/67c6d1e058b17830/report.json`
- Экспорт: `outputs/champion_classification_recovered_20261026/submission.csv`
- Протокол: `outputs/champion_classification_recovered_20261026/recovery_result.json`

Большие native-модели восстановленного каталога связаны с исходными файлами
hard links для экономии примерно 6 ГБ. Этот каталог предназначен для чтения и
экспорта; не запускать в нём fit/save predictor. Все новые обучения идут отдельно.

В новом запуске `outputs/champion_classification_followup_v2/agent_runs/followup_20261026`
импортированы только точные внешние/внутренние индексы и спецификация старого exp5.
Старые модели и holdout-score не внесены в новый реестр как новые измерения.
Результат свежего OOF XGBoost сохраняется в `classification_recheck_summary.json`.

Перепроверка exp5 завершена: mean **0.6128243**, worst **0.5967548**. Преимущество
на одном прежнем holdout не подтвердилось на общих фолдах. Лучшим из сохранённых
результатов остаётся восстановленный AutoGluon-ансамбль **0.6328643**. CSV нового
followup-run с XGBoost — диагностический результат, а не замена лучшего контроля.

TabM получил параметр `numeric_preprocessing=quantile`: train-only QuantileTransformer,
PLE и категории сохраняются в модели; постоянные индикаторы пропусков исключены.
Старый `robust` остаётся совместимым со старыми моделями. Следующая парная проверка —
PLE/quantile без весов против SqrtBalanced при одинаковых признаках и seed.

Исправлено управление AutoGluon: резерв времени на отчёт, разные шкалы времени,
запас внешнего watchdog, повторное использование законченных фолдов, восстановление
завершённых checkpoints перед финальным выбором. Расходованный бюджет не обнуляется.

## Команда Qwen после подготовки

В терминале с заданными OPENAI_API_KEY / OPENAI_BASE_URL:

```powershell
.\.venv_autogluon\Scripts\python.exe champion_solution/solution_torch.py tasks/classification/task.md `
  --output-dir outputs/champion_classification_followup_v2 `
  --resume outputs/champion_classification_followup_v2/agent_runs/followup_20261026 `
  --extended-research --threads 8 --researchers 1 --training-jobs 1 `
  --reviewer --no-analyst --autogluon-budget 3600 --flaml-budget 0 --lightautoml-budget 0 `
  --time-budget 9900 --max-steps 80 --step-timeout 3600 `
  --api-timeout 180 --api-retries 4 --instructions classification_followup_plan.md
```

Не возобновлять старый v1-run новой версией кода и не запускать configure_task заново
в подготовленном followup-run. При изменении исходного кода после подготовки нужен
новый запуск: проверка fingerprints предохраняет от смешивания разных pipeline.
