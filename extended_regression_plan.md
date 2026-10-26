# Расширенная регрессия: TabM и крупные ошибки

Новый независимый output-dir. Сначала inspect_data, словарь, inspect_research_plan,
configure_task(split="time",split_column="Year_Factor"), затем run_regression_anchors.
Все последующие кандидаты проверять на одних validation_periods (обычно 3/4/5).

1. train_torch_tabm на исходных числовых/строковых признаках без ID/target.
   Парное none/ple: iterations=40,width=128,layers=2,batch_size=512,ensemble_size=8,
   embedding_dim=16,n_bins=32,learning_rate=0.002,weight_decay=0.0003,dropout=0.1,
   seed=42,early_stopping_rounds=8. Менять только num_embedding.
2. run_temporal_suite на лучшем TabM по тем же годам. Не исключать модель лишь
   за проигрыш на single holdout. Сначала baseline/history ablation, затем проверить
   history_hierarchy с прежними ключами при достаточном бюджете.
3. run_regression_specialists(reference_suite_id=<сильный одиночный temporal suite>,
   area_tolerance=0.01,tail_quantile=0.95,time_limit=1800). Нельзя передать ensemble,
   AutoML suite или residual/gate winner. Это четыре сопоставимых режима:
   reference, fuzzy_history, tail_expert, tail_fuzzy.
   Не выбирать gate по истинному target validation/test. Backend использует X/history.
   Абляция fuzzy сохраняет recipe/число итераций источника; совпадение по площади
   лишь гипотеза о похожем здании. Сравнить coverage, mean/latest/worst и большие ошибки.
4. blend_temporal_suites с сильным исходным single и TabM; затем при необходимости
   с лучшим новым специалистом. Фиксированные веса, 2..4 разных пайплайна, явно
   member_modes. Не дублировать reference-алиас исходного источника в ансамбле.
5. Выбрать глобальный минимум mean_rmse (ties latest/worst) через select_final_candidate
   и finalize_selected. Новые специалисты уже сохраняют full-past pipelines внутри
   своего time_limit; их экспорт не начинает поиск/обучение заново.

В этом сравнении AG/FLAML/LightAutoML отключены, чтобы изолировать новые направления.
Их готовые инструменты остаются в проекте и могут проверяться отдельными запусками.
При дефиците времени экспортировать лучший completed pipeline и перечислить пропуски.
