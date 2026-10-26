# Технический план LightAutoML 0.4.2

## Порядок текущего исследования Qwen

1. Прочитать task и словарь. Настроить временной split на Year_Factor, если test
   строго позднее train. Для site_eui начать с run_regression_anchors.
2. Взять завершённый одиночный temporal suite с baseline/history_exact/
   history_groups/history_hierarchy, предпочтительно сильный history_hierarchy.
   Не использовать ensemble, residual_history или history_gate как source.
3. При бюджете LightAutoML 3600 сделать три вызова с одним и тем же source,
   algorithms=["linear_l2","lgb","cb"], time_limit=900 каждый:
   - feature_mode="inherit", tuning=False, refit_full=False;
   - feature_mode="compact_strings", tuning=False, refit_full=False;
   - feature_mode="compact_strings", tuning=True, max_tuning_trials=8, refit_full=True.
   Первый переход меняет только представление категорий. Во втором сравнить
   исходный mode=lightautoml с предыдущим — эффект внутреннего tuning.
   В третьем suite парно сравнить mode=lightautoml и mode=lightautoml_refit:
   одинаковые внутренние параметры/веса, разный объём refit-прошлого.
   900 — лимит всего suite вместе с финальными fit, не каждого фолда.
4. Сравнить mean/latest/worst на неизменённых внешних периодах. Независимо от
   отдельной победы проверить одну фиксированную смесь source + лучший режим LightAutoML:
   blend_temporal_suites(suite_ids=[source_id, lama_id],
   member_modes=[source_mode, <lightautoml или lightautoml_refit>]). Не оптимизировать веса по outer OOF.
5. Сравнить с ансамблем anchors и остальными сопоставимыми кандидатами; выбрать
   глобального победителя и сохранить резерв экспорта. Отсутствие выигрыша —
   допустимый результат, не повторять неизменённые вызовы ради улучшения метрики.

FLAML отключён, чтобы оценить LightAutoML отдельно. Для одиночных рецептов
использовать train_catboost / train_lightgbm / train_xgboost / train_torch_mlp /
train_torch_resnet; у каждого своя схема params. Не передавать width/loss/target_transform
бустингам и не повторять отклонённый запрос без исправления.
Финальные pipeline уже сохранены внутри бюджета suite: export не запускает fit.

## Зафиксированный интерфейс

Источники: тег `v.0.4.2`, а не изменяющаяся ветка master.

- `lightautoml/tasks/base.py`: `Task('reg', loss='mse', metric='mse')`.
- `lightautoml/automl/presets/tabular_presets.py`: `TabularAutoML`,
  `fit_predict(train_data, roles, cv_iter=None, valid_data=None, path_to_save=None)`;
  результат `.data`, `predict(frame).data`.
- `lightautoml/automl/base.py`: при переданном `valid_data` разрешён один
  уровень моделей; в конце fit доступен `joblib.dump` через `path_to_save`.
- `lightautoml/validation/base.py`: CustomIterator принимает пары позиционных
  индексов train/valid. Пустые или непокрытые временным CV точки могут вернуть
  NaN; это нельзя выдавать за полное OOF-покрытие.
- `lightautoml/automl/presets/tabular_config.yml`: явный список алгоритмов,
  reader advanced_roles, режим селекции, лимиты настройки.

Ссылки:

1. https://github.com/sb-ai-lab/LightAutoML/blob/v.0.4.2/lightautoml/tasks/base.py
2. https://github.com/sb-ai-lab/LightAutoML/blob/v.0.4.2/lightautoml/automl/presets/tabular_presets.py
3. https://github.com/sb-ai-lab/LightAutoML/blob/v.0.4.2/lightautoml/automl/base.py
4. https://github.com/sb-ai-lab/LightAutoML/blob/v.0.4.2/lightautoml/validation/base.py
5. https://github.com/sb-ai-lab/LightAutoML/blob/v.0.4.2/lightautoml/automl/presets/tabular_config.yml

## Минимальная первая конфигурация

```python
TabularAutoML(
    task=Task('reg', loss='mse', metric='mse'),
    timeout=seconds_for_one_fit,
    cpu_limit=threads,
    gpu_ids=None,
    general_params={'use_algos': [['lgb', 'linear_l2', 'cb']], 'nested_cv': False},
    reader_params={'n_jobs': threads, 'cv': None, 'random_state': 42,
                   'advanced_roles': False},
    selection_params={'mode': 0},
)
```

Это параметры создания кандидата, не обещание запуска всех трёх алгоритмов при
нехватке времени. Список реально обученных моделей нужно сохранять в отчёте.

Безопасный первоначальный режим: внутренний past holdout, а не автоматический
random CV. Для внешнего года 5 fit получает только прошлые периоды; последний
из этих периодов идёт во внутреннюю validation. Наружный год 5 используется
только через `predict` после fit. Исторические признаки также должны строиться
каузально, из строго более ранних меток.

Не передавать наружную валидацию в `valid_data`, не выбирать по ней внутренние
веса ансамбля. Не вызывать повторный `fit_predict` с train==valid ради якобы
полного refit: это даст оценку на обучении, ранний останов на train и другой
рецепт, чем проверялся снаружи.

## Экспорт и фиксированный refit

Время финального построения учитывать внутри общего бюджета suite. Сохранить
сам preset, параметры обработки и исторических признаков, границы предсказаний,
использованные периоды, хеши исходных файлов и списка зависимостей. Пакет после
загрузки обязан выдавать те же конечные предсказания. При использовании смеси
экспортировать явно указанный режим lightautoml, даже если внутри suite победил
reference. Сохранённый reference нельзя тихо заменять. Для refit существует
отдельный mode=lightautoml_refit с собственными предсказаниями и моделями.

Финальные базовые модели holdout-preset обучены на периодах до внутренней
валидации. Последний train-период применяется для раннего останова и внутреннего
blender. Это остаётся поведением mode=lightautoml.

Дополнительный mode=lightautoml_refit использует адаптер для версии 0.4.2:
LightGBM обучается на всём прошлом с выбранным внутри числом деревьев и параметрами;
CatBoost — с выбранным числом деревьев без eval_set; LinearLBFGS — с записанным
внутренним C, без нового перебора. Веса собственного ансамбля заморожены внутренним
holdout. Reader и те же семейства feature pipelines заново fit только на past;
cv=None отключает target encoding, категориальные роли заданы явно. Все обучающие
исторические признаки остаются причинными для каждой строки. На внешнюю validation
идёт только predict. Тот же протокол повторяется для финального train.

Не трактовать raw/refit как гарантированное улучшение; сравнение проводится в
одном suite на одинаковых годах. Сохранить достаточный запас времени для экспорта.

## Окружение

Не обновлять рабочую `.venv` вслепую. Сначала согласовать установку с текущими
пинами, особенно numpy1.26.4, scipy1.12.0, pandas2.2.3, sklearn1.5.2,
torch2.5.1, lightgbm4.6.0, catboost1.2.8, xgboost2.1.4, optuna4.2.1,
statsmodels0.14.0. Если создаётся отдельная среда, не забыть Qwen SDK и остальные
модули агента. Изменение импортируемого кода или списка зависимостей требует
новой папки запуска, старый resume не должен незаметно смешивать рецепты.

Прямые ограничения LAMA совместимы с этими пинами. Транзитивные зависимости и
импорт нужно проверять отдельно. `tabular_presets` импортирует TabICL; поэтому
потребуются его Python-зависимости, даже когда TabICL явно исключён из обучения.
Пакеты NLP/CV/report и скачивание весов для этой задачи не нужны.
