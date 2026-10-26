# Расширенная классификация: TabM, AutoGluon и OVR

Контрольные ZIP и исходные отчёты не изменять. Использовать числовые признаки и
строковые категории без ID/target; fitted-преобразования обучать внутри training folds.

Если запуск подготовлен скриптом перепроверки XGBoost, продолжать его через выданный
--resume. Сначала inspect_research_plan и память: задача, frozen split и первая OOF-проверка
уже подготовлены. НЕ вызывать configure_task повторно и не пересоздавать фолды.
Для обычного нового запуска сначала inspect_data, словарь и configure_task; сопоставление
с прошлыми результатами допустимо только после импорта и проверки точного split_plan_id.

1. Первый приоритет — OOF-перепроверка лучшего сохранённого XGBoost на ТОЧНЫХ прежних
   outer и inner calibration folds. В исходном исследовании это historical recipe 5,
   но этот ID нельзя выдавать за experiment_id нового запуска. Читать фактические ID
   из памяти/ответов инструментов. Подготовительный скрипт уже выполняет
   run_saved_recipe_oof(split_plan_id=<проверенный импортированный план>,
   historical_recipe_ids=[5]); если отчёт существует, прочитать его, не повторять обучение.
   Вызывать этот инструмент вручную только если этап действительно не завершён.
2. Новый парный TabM: num_embedding="ple",numeric_preprocessing="quantile";
   iterations=40,width=128,layers=2,batch_size=512,ensemble_size=8,learning_rate=0.002,
   weight_decay=0.0003,embedding_dim=16,n_bins=32,dropout=0.1,seed=42,
   early_stopping_rounds=8. Первый — без auto_class_weights (None по умолчанию),
   второй auto_class_weights="SqrtBalanced". Полные остальные параметры и feature_code
   должны быть одинаковыми; передавать feature_code явно. Использовать один осмысленный
   набор признаков для пары. Не менять одновременно веса, признаки и размер сети.
   Не повторять старую сырую none/PLE пару без весов — она уже измерена.
3. run_classification_suite с актуальным XGBoost и новыми TabM experiment_ids,
   folds=3,split_plan_id=<ТОТ ЖЕ проверенный план>. Старый historical ID не подставлять
   вместо нового experiment_id. Все кандидаты и смеси сравниваются на одинаковых
   внешних строках; веса и Macro F1 правила — только на внутренних calibration rows.
   Если время позволяет, отдельная абляция weighted TabM с numeric_preprocessing="robust",
   тем же feature_code/seed/размером сети; затем подтвердить через те же OOF-фолды.
4. После сильного завершённого OOF-reference первым проверить run_autogluon_suite
   model_profile="diverse",time_limit=1800,bag_folds=3,stack_levels=0.
   reference_suite_id брать из текущих совместимых completed OOF-отчётов, включая
   перепроверку сильного XGBoost; не привязываться к прежнему слабому составу [1,2,4].
   При наличии места в общем дедлайне и AG allocation можно проверить model_profile="boosting"
   на ТОМ ЖЕ reference, но это вторичный этап. Общий AG-бюджет — 3600 секунд на все
   вызовы, не на каждый fit. Указать реально обученные семейства и пропущенные этапы.
5. OVR-эксперты — только при остатке бюджета и обоснованной гипотезе: предыдущий опыт
   с Backdoor/Analysis/DoS дал лишь около 0.00013 среднего Macro F1. Не запускать их
   снова автоматически вместо перепроверки XGBoost, TabM и полноценного AG-экспорта.
6. Сопоставить mean/worst Macro F1 и per_class. Выполнить adversarial_validation и
   audit_classification_shift для лучшего AG suite: диагностический test-like срез,
   не использовать его для подбора коэффициентов. Для expert OOF можно отдельно
   прочитать сохранённые probabilities, но не выдавать это за независимую проверку.
7. select_final_candidate(kind="classification_oof" или "autogluon_oof",suite_id=...),
   затем finalize_selected. Эксперты лежат в classification_oof, backend=experts_v1.
   Автоматический финальный selector также сравнит все совместимые completed suites.

Не сравнивать holdout 0.66515 напрямую со средним OOF: это разные проверки.
Новые сравнения приоритетнее десятков близких depth/seed рецептов и прежних MLP/resnet.
При дефиците времени сохранить лучший проверенный результат и перечислить пропущенные
этапы. Оценки внешних folds уже адаптивно используются для разработки, это не untouched test.
