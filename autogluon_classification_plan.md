# Первый запуск Qwen с AutoGluon

Контрольный ZIP на платформе: 2157.225. Он сохранён отдельно и не меняется.
Это эксперимент с AutoGluon 1.3.1, не обещание прироста.

1. Прочитай данные, словарь и inspect_research_plan. Для классификации attack_cat
   зафиксируй Macro F1 и стратифицированное случайное разбиение. При признаках
   временного сдвига инструмент AutoGluon пока не применяй.
2. Быстро измерь CatBoost и сильный применимый classification_49 через
   train_reference_recipe. Сравни ещё одну обоснованную модель/модификацию.
3. Из лучших двух-трёх рецептов создай run_classification_suite с folds=3.
   Сохрани suite_id. Это исходный reference с внутренней калибровкой и внешним OOF.
4. Заблаговременно вызови run_autogluon_suite(reference_suite_id=..., time_limit=1800,
   bag_folds=3, stack_levels=0, feature_code=""). Все шесть AutoGluon fits делят
   указанные 1800 секунд. Внешние данные никогда не передаются в fit/tuning_data.
5. Сравни reference, autogluon, equal_ensemble, weighted_ensemble и macro_decision
   на одинаковых внешних строках: mean/worst/std Macro F1 и per-class отчёт на диске.
   Не сравнивай holdout 0.66 с OOF 0.63 как одинаковые оценки.
6. Если есть запас времени и суммарного AutoGluon budget, проверь одну отдельную
   гипотезу, например stack_levels=1, с тем же reference. Кеш не переобучает модели
   и не расходует бюджет. Не переходи к большому перебору.
7. Выбери лучший проверенный suite через select_final_candidate. Для нового трека
   kind="autogluon_oof"; для старого kind="classification_oof". Если в новом suite
   выиграл reference, экспорт в точности использует прежний pipeline. Оценки и
   ограничения сформулируй честно. Финальное обучение вручную не запускай.

Инструмент пока поддерживает только classification/f1_macro. GPU, предобученные
модели, установка пакетов и сеть внутри экспериментов не используются. AutoGluon
ограничен GBM/CAT/XGB, CPU, последовательным локальным bagging. Сохраняй весь
каталог suite/predictor, а не один final_model.joblib.
