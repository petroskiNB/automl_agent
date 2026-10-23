# Бустинги и нейросети в одном агенте

Старый `solution.py` и старые запуски сохранены. Общий агент запускается через
`run_neural.py agent`: CatBoost, LightGBM, XGBoost, PyTorch MLP и residual MLP
доступны одному контроллеру. Это не TabM и не предобученная сеть.

`neural_solution/runtime.py` собирает полную версию из общих корневых модулей
и дополнений в `neural_solution` в `work/neural_runtime/<hash>`. Исходники не
дублируются вручную. Обновление кода создаёт новый runtime; уже работающий процесс
продолжает использовать свою версию. Не редактируйте сгенерированный runtime.

## Установка и проверка

В используемом виртуальном окружении дополнительно нужны:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 psutil==6.1.1 setuptools==80.9.0
.\.venv\Scripts\python.exe -m unittest test_neural_integration
.\.venv\Scripts\python.exe run_neural.py test
.\.venv\Scripts\python.exe run_neural.py agent tasks/classification/task.md --check
```

## Запуск Qwen

Настройте OPENAI_API_KEY, OPENAI_BASE_URL и OPENAI_MODEL в текущем терминале.
Затем запустите новый поиск в отдельной папке:

```powershell
.\.venv\Scripts\python.exe run_neural.py agent tasks/classification/task.md --researchers 3 --training-jobs 2 --threads 8 --time-budget 9900 --max-steps 60 --output-dir outputs/combined_classification_search --instructions combined_classification_plan.md
```

Три исследователя предлагают эксперименты, до двух обучений выполняются одновременно
с разделением 8 CPU-потоков. Все видят общую историю. Данные копируются в output-dir;
исходный submission и прежние журналы не перезаписываются. Новая история находится
в `output-dir/agent_runs/`. `--resume` принимает каталог именно нового запуска;
старые manifests нельзя переносить в новую версию подменой хешей.

## Перенос проверенного рецепта

Команда ниже без API повторно обучает старый рецепт и MLP на **точно том же split**,
проверяя хеши исходных данных. Старые метрики и веса не импортируются как новые результаты.
Каждый раз указывайте новую output-dir.

```powershell
.\.venv\Scripts\python.exe run_neural.py compare tasks/classification/task.md --source-run tasks/classification/agent_runs/20261022_143503_454151 --experiment 49 --output-dir outputs/combined_comparison --epochs 40 --threads 4 --time-budget 1800
```

Это предварительный holdout-тест, использующий validation для early stopping.
Он не доказывает прирост на платформе. `--oof` дополнительно запускает nested 3-fold
сравнение: ранняя остановка и настройка весов происходят внутри внешних фолдов.
Для такого запуска нужен больший time-budget. Сформированный журнал можно продолжить
через `agent ... --output-dir <та же папка> --resume <папка>/agent_runs/combined_comparison`.
Укажите те же `--threads`, что при сравнении: число потоков входит в manifest.

Продолжение уже проведённого реального пилота (4 потока, до двух обучений):

```powershell
.\.venv\Scripts\python.exe run_neural.py agent tasks/classification/task.md --output-dir outputs/combined_classification_pilot_20261023 --resume outputs/combined_classification_pilot_20261023/agent_runs/combined_comparison --threads 4 --researchers 3 --training-jobs 2 --time-budget 9900 --max-steps 60 --instructions combined_classification_plan.md --require-selection
```

Для переноса в Qwen сохраняется `imported_recipe.json`; обученные новые кандидаты
и их оценки уже находятся в обычном реестре экспериментов.

## Выбор и отправка

Сравниваем одиночный бустинг, одиночную сеть и смеси на одних внешних фолдах.
Для классификации дополнительно используем adversarial stress split и per-class F1.
Сеть включаем в финал только по результатам проверки. Старые и новые OOF-оценки
не объединяем по номерам строк без проверки разбиения, классов и происхождения данных.

```powershell
.\.venv\Scripts\python.exe run_neural.py build --output outputs/automl_combined.zip
```

ZIP содержит код и requirements, без датасетов, ключей и обученных моделей.
Контейнерная точка входа пишет submission рядом с task.md. Сборка ZIP сама по себе
не означает, что качество подтверждено. Временной suite исторических признаков пока
поддерживает CatBoost; перенос MLP в него требует отдельной реализации и проверки.
