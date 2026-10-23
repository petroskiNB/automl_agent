# Экспериментальная версия с PyTorch

Версия разработана на обновлённой базе с тремя исследовательскими направлениями,
очередью двух обучающих процессов, nested OOF-классификацией и явным выбором
финального pipeline. Обычный `solution.py` остаётся точкой входа прежней версии.
Актуальные команды объединённого проекта: [COMBINED_AGENT.md](../COMBINED_AGENT.md).
Результаты старого синтетического теста не являются оценкой реальной классификации.

## Модели и данные

- `torch_mlp`: MLP с embeddings категорий, LayerNorm, dropout и AdamW.
- `torch_resnet`: такой же ввод с residual-блоками. Это не реализация TabM.
- Обучение с нуля, CPU, без скачивания весов или дополнительных пакетов в runtime.
- Preprocessing сохраняется внутри модельного пакета: train-медиана, IQR scaling,
  missing indicators, clipping числовых признаков, словари категорий.
  Неизвестная категория получает код 0, пропуск - код 1. Словари ограничены
  `max_categories`; редкие значения за пределами словаря также получают код 0.
  Категориальные пропуски сохраняются до neural preprocessing, не смешиваясь
  с буквальной строкой `__MISSING__`.
- Регрессия: стандартизация таргета, MSE/MAE/Huber, опциональный log1p для
  неотрицательных targets. Early stopping всегда по исходной метрике, а не
  по преобразованному loss. log1p не обещает улучшения RMSE.
- Классификация: cross-entropy, None/Balanced/SqrtBalanced, фиксированный порядок
  классов, вероятности, ранняя остановка по невзвешенному Macro F1/Accuracy.
- Финальный refit обучает preprocessing заново на полном train и использует
  число эпох выбранной модели. Fold bagging из OOF экспортируется без full refit.
- `--resume` продолжает журнал экспериментов. Это не возобновление прерванной
  эпохи/оптимизатора; незавершённый опыт повторяется с нуля.

## Установка

Python 3.12 и зависимости из `requirements.txt`, в том числе torch 2.5.1.
Для локальной проверки подготовлено отдельное `.venv`.
Это окружение не включается в ZIP. Установка не выполняется внутри агента.

## Запуск трёх исследователей

Из корня репозитория (пути CSV ищутся относительно исходного task.md):

```powershell
.\.venv\Scripts\python.exe run_neural.py agent tasks\classification\task.md --researchers 3 --training-jobs 2 --threads 8 --analyst --output-dir outputs\pytorch_classification_v1
```

`--threads 8` - суммарная квота этого запуска, а не квота каждого агента.
Два процесса получают по 4 потока; exclusive suites и refit - до 8.
При недостатке свободной RAM очередь переходит к одному процессу.
PyTorch inter-op=1, intra-op ограничен квотой; DataLoader не используется,
дополнительных worker-процессов чтения нет. BLAS/OpenMP также ограничены.
Это лимит вычислительных пулов: служебные потоки Python/библиотек возможны.
Другие независимые старые запуски не подчиняются этой очереди; если они занимают
CPU, уменьшите `--threads`, чтобы сумма выделенных квот не превышала восемь.
Проверка RAM при admission является оценкой, не системным ограничением памяти.

Локально входные CSV копируются в отдельный каталог, результаты и submission
пишутся туда же. Повторное использование каталога проверяет хэши входов.
Изменившиеся входы требуют нового каталога. Старые запуски не импортируются
автоматически. --resume указывает на новый run внутри `outputs/.../agent_runs/`.
Хэши новой версии и квоты потоков входят в manifest.

Бюджет по умолчанию 9900 секунд с резервом на refit, максимум 10800 секунд.
Subprocess timeout ограничивает все backend. PyTorch дополнительно проверяет
дедлайн между батчами. Неудачное обучение не заменяет сохранённый результат;
незавершённый full refit не публикуется. При его сбое остаётся holdout-модель;
смотрите `finalized_experiment` и events.jsonl, а не только наличие CSV.

Для classification используйте `run_classification_suite` с рецептами сети и
бустинга: outer folds оценивают весь pipeline, внутренние части выбирают число
эпох, веса ансамбля и множители классов. `select_final_candidate` и
`finalize_selected` публикуют выбранный fold bagging. Adversarial и grouped
diagnostics доступны. Старый temporal suite истории остаётся CatBoost-only;
для сети используйте причинные фолды standalone benchmark ниже.

## Сравнение без Qwen/API

```powershell
.\.venv\Scripts\python.exe run_neural.py benchmark tasks\regression\task.md --target site_eui --task-type regression --metric rmse --split time --split-column Year_Factor --folds 3 --seeds 42 43 --threads 8 --output-dir outputs\torch_regression_comparison
.\.venv\Scripts\python.exe run_neural.py benchmark tasks\classification\task.md --target attack_cat --task-type classification --metric f1_macro --folds 3 --seeds 42 43 --threads 8 --output-dir outputs\torch_classification_comparison
```

Одинаковые фолды/seed и time cap на опыт для CatBoost, LightGBM, XGBoost,
MLP, residual MLP; классификация дополнительно сравнивает взвешенный MLP.
Сохраняются индексы разбиения, рецепты, модели, вероятности и метрики классов.
В comparison.json/md: качество, fit/predict time, peak RSS процесса с sampling
20 мс, квота потоков, seed и фиксированные ансамбли 0.5/0.5.
Память включает интерпретатор и импортированные библиотеки. fit_seconds включает
preprocessing внутри модели (у сетей - обучение словарей и масштабирования),
predict_seconds - преобразование внутри модели и сам прогноз. Внешний
feature_code, CSV IO и публикация в эти две величины не входят; worker_seconds
показывает полное время запуска опыта, включая старт интерпретатора.
В отчёте также записываются версии библиотек и хэши кода/входов.
Для крошечных синтетических данных `--smoke` переиспользует один процесс,
чтобы загрузка DLL не занимала большую часть проверки; эти времена и peak RSS
нельзя считать идентичными свежим subprocess для каждого реального опыта.

Standalone benchmark предназначен для разработки: early stopping выбирает
эпоху на том же validation fold, который выводится в отчёте. Это не nested OOF,
не независимый тест, а усреднение по seed не добавляет независимых разбиений.
Для окончательной classification-оценки используйте nested suite агента.
Сеть с меньшим отдельным score может улучшить ансамбль; это проверяется, а не
предполагается. Неполные наборы folds/seeds не участвуют в общем ранжировании.
Финальный standalone export использует первый заданный seed победившего рецепта
и full refit; bagging по всем seed не экспортируется этим benchmark.

Синтетическая проверка без исходных датасетов:

```powershell
.\.venv\Scripts\python.exe run_neural.py test -v
.\.venv\Scripts\python.exe run_neural.py benchmark --smoke regression --split time --split-column period --folds 2 --seeds 42 43 --epochs 30 --trees 60 --width 64 --threads 2 --time-budget 600 --final-reserve 60 --model-seconds 60 --output-dir outputs\torch_smoke_regression
```

Синтетические scores проверяют работоспособность, а не качество на Альфа-Хаке.
LLM-интеграция проверяется с mock API; запросы с реальным ключом и загрузка
архива на платформу не входят в автоматические тесты.

## Архив для платформы

```powershell
.\.venv\Scripts\python.exe run_neural.py build --output outputs\automl_pytorch_v1.zip
```

ZIP содержит только разрешённый список исходников и requirements, без данных,
ключей, журналов и моделей. Его `solution.py` вызывает новую версию в режиме
`--platform`: в чистом Docker-контейнере submission пишется рядом с переданным
task.md, как в исходном интерфейсе. Локальный `--platform` также пишет в каталог
исходной задачи, поэтому обычные эксперименты запускайте без этого флага.
