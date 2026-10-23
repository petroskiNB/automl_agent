# Новый поиск: 6 потоков, три исследователя и два советника

Это новый запуск. Старые метрики ниже — история, а не измерения текущего запуска.
Сначала inspect_data, затем configure_task по task.md. Читай переданный словарь:
ct_src_ltm/ct_dst_ltm — счётчики соединений, не длительности. Не объясняй признак
по имени. Не утверждай, что test не имеет сдвига, без проверяемого источника.

Воспроизведи рецепт old_id=1 как быстрый baseline с его полным feature_code.
Далее приоритет обязательным сетевым опытам из pending_experiments: MLP без весов
и SqrtBalanced, компактная residual MLP, сеть с log-преобразованиями. Один опыт
может закрыть несколько пунктов. Для начала iterations=40, width=128, layers=2,
batch_size=512, learning_rate=.001, dropout=.1, weight_decay=.0001, seed=42,
early_stopping_rounds=8. Для классификации не задавай regression loss.
Не наследуй параметры деревьев в сеть. Договоритесь по общей истории, кто какой
рецепт выполняет: models — residual/OOF, rare — веса MLP, features — преобразования.

После краткого отбора перепроверь лучшую сеть и бустинг через nested OOF 3 folds
с одним shift_id. Дополнительно сравни две тройки прежних рецептов [1,4,5]
и [6,4,5], если позволяет бюджет. Номера ниже ИСТОРИЧЕСКИЕ: используй новые ID
после обучения, не обращайся к старым ID как к текущим. Рецепты лежат ниже.
Для сравнения прежних троек их надо воспроизвести в новом запуске. Никакого
копирования старых моделей и метрик с подменой manifests.

Историческая тройка [1,4,5]: mean=0.63259965, worst=0.62489402, stress=0.58295342.
Историческая тройка [6,4,5]: mean=0.63283711, worst=0.62527108, stress=0.55922382.
Обе использовали macro_decision. Разница mean=0.00024 не доказывает преимущество
второй. Сохрани обе в сравнении; выбирай с явным учётом худшего фолда и stress.
Stress — диагностическая выборка, а не точная копия теста. Не подбирай веса
ансамбля по внешним меткам. Не выбирай сеть просто ради разнообразия.

Оставь резерв для select_final_candidate и finalize_selected. Финальный экспорт
OOF — fold bagging, не full-data refit. При исчерпании бюджета не скрывай
невыполненные проверки. Reviewer проверяет выводы/происхождение метрик,
analyst анализирует ошибки и редкие классы. Не трать несколько раундов на
повторное чтение одного неизменившегося рецепта.

## Исторические рецепты (переносить только параметры и преобразования)

```json
[
  {
    "id": 1,
    "algorithm": "xgboost",
    "params": {
      "auto_class_weights": "SqrtBalanced",
      "colsample_bytree": 0.6,
      "depth": 4,
      "gamma": 1.0,
      "iterations": 500,
      "l2_leaf_reg": 3,
      "learning_rate": 0.05,
      "min_child_weight": 5,
      "subsample": 0.8
    },
    "feature_code": "def transform(X):\n    X = X.copy()\n    log_cols = ['dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',\n                'stcpb', 'dtcpb', 'sbytes', 'dbytes', 'response_body_len']\n    for c in log_cols:\n        if c in X.columns:\n            X[c] = np.log1p(np.maximum(X[c], 0.0))\n    for c in ['spkts', 'dpkts']:\n        X['log1p_' + c] = np.log1p(np.maximum(X[c], 0.0))\n    X['bytes_per_pkt_s'] = X['sbytes'] / np.maximum(X['spkts'], 1)\n    X['bytes_per_pkt_d'] = X['dbytes'] / np.maximum(X['dpkts'], 1)\n    X['pkts_ratio'] = X['dpkts'] / np.maximum(X['spkts'], 1)\n    X['bytes_ratio'] = X['dbytes'] / np.maximum(X['sbytes'], 1)\n    X['load_ratio'] = X['dload'] / np.maximum(X['sload'], 1)\n    X['ttl_diff'] = X['dttl'] - X['sttl']\n    X['jit_ratio'] = X['djit'] / np.maximum(X['sjit'], 1)\n    X['pkt_gap'] = X['dinpkt'] - X['sinpkt']\n    return X",
    "score": 0.6651474102407088
  },
  {
    "id": 4,
    "algorithm": "xgboost",
    "params": {
      "auto_class_weights": null,
      "colsample_bytree": 0.6,
      "depth": 4,
      "gamma": 1.0,
      "iterations": 500,
      "l2_leaf_reg": 3,
      "learning_rate": 0.05,
      "min_child_weight": 5,
      "subsample": 0.8
    },
    "feature_code": "def transform(X):\n    X = X.copy()\n    log_cols = ['dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',\n                'stcpb', 'dtcpb', 'sbytes', 'dbytes', 'response_body_len']\n    for c in log_cols:\n        if c in X.columns:\n            X[c] = np.log1p(np.maximum(X[c], 0.0))\n    for c in ['spkts', 'dpkts']:\n        X['log1p_' + c] = np.log1p(np.maximum(X[c], 0.0))\n    X['bytes_per_pkt_s'] = X['sbytes'] / np.maximum(X['spkts'], 1)\n    X['bytes_per_pkt_d'] = X['dbytes'] / np.maximum(X['dpkts'], 1)\n    X['pkts_ratio'] = X['dpkts'] / np.maximum(X['spkts'], 1)\n    X['bytes_ratio'] = X['dbytes'] / np.maximum(X['sbytes'], 1)\n    X['load_ratio'] = X['dload'] / np.maximum(X['sload'], 1)\n    X['ttl_diff'] = X['dttl'] - X['sttl']\n    X['jit_ratio'] = X['djit'] / np.maximum(X['sjit'], 1)\n    X['pkt_gap'] = X['dinpkt'] - X['sinpkt']\n    return X",
    "score": 0.6349638146123098
  },
  {
    "id": 5,
    "algorithm": "xgboost",
    "params": {
      "auto_class_weights": "Balanced",
      "colsample_bytree": 0.6,
      "depth": 4,
      "gamma": 1.0,
      "iterations": 500,
      "l2_leaf_reg": 3,
      "learning_rate": 0.05,
      "min_child_weight": 5,
      "subsample": 0.8
    },
    "feature_code": "def transform(X):\n    X = X.copy()\n    log_cols = ['dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',\n                'stcpb', 'dtcpb', 'sbytes', 'dbytes', 'response_body_len']\n    for c in log_cols:\n        if c in X.columns:\n            X[c] = np.log1p(np.maximum(X[c], 0.0))\n    for c in ['spkts', 'dpkts']:\n        X['log1p_' + c] = np.log1p(np.maximum(X[c], 0.0))\n    X['bytes_per_pkt_s'] = X['sbytes'] / np.maximum(X['spkts'], 1)\n    X['bytes_per_pkt_d'] = X['dbytes'] / np.maximum(X['dpkts'], 1)\n    X['pkts_ratio'] = X['dpkts'] / np.maximum(X['spkts'], 1)\n    X['bytes_ratio'] = X['dbytes'] / np.maximum(X['sbytes'], 1)\n    X['load_ratio'] = X['dload'] / np.maximum(X['sload'], 1)\n    X['ttl_diff'] = X['dttl'] - X['sttl']\n    X['jit_ratio'] = X['djit'] / np.maximum(X['sjit'], 1)\n    X['pkt_gap'] = X['dinpkt'] - X['sinpkt']\n    return X",
    "score": 0.5860744597074043
  },
  {
    "id": 6,
    "algorithm": "xgboost",
    "params": {
      "auto_class_weights": "SqrtBalanced",
      "colsample_bytree": 0.6,
      "depth": 4,
      "gamma": 1.0,
      "iterations": 500,
      "l2_leaf_reg": 3,
      "learning_rate": 0.05,
      "min_child_weight": 5,
      "subsample": 0.8
    },
    "feature_code": "def transform(X):\n    X = X.copy()\n    log_cols = ['dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',\n                'stcpb', 'dtcpb', 'sbytes', 'dbytes', 'response_body_len']\n    for c in log_cols:\n        if c in X.columns:\n            X[c] = np.log1p(np.maximum(X[c], 0.0))\n    for c in ['spkts', 'dpkts']:\n        X['log1p_' + c] = np.log1p(np.maximum(X[c], 0.0))\n    X['bytes_per_pkt_s'] = X['sbytes'] / np.maximum(X['spkts'], 1)\n    X['bytes_per_pkt_d'] = X['dbytes'] / np.maximum(X['dpkts'], 1)\n    X['pkts_ratio'] = X['dpkts'] / np.maximum(X['spkts'], 1)\n    X['bytes_ratio'] = X['dbytes'] / np.maximum(X['sbytes'], 1)\n    X['load_ratio'] = X['dload'] / np.maximum(X['sload'], 1)\n    X['ttl_diff'] = X['dttl'] - X['sttl']\n    X['jit_ratio'] = X['djit'] / np.maximum(X['sjit'], 1)\n    X['pkt_gap'] = X['dinpkt'] - X['sinpkt']\n    for c in ['dur', 'sbytes', 'dbytes', 'tcprtt', 'synack', 'ackdat', 'swin', 'dwin']:\n        if c in X.columns:\n            X['zero_' + c] = (X[c] == 0).astype(float)\n    X['both_bytes_zero'] = ((X['sbytes'] == 0) & (X['dbytes'] == 0)).astype(float)\n    X['single_pkt'] = (X['spkts'] == 1).astype(float)\n    X['scanlike'] = ((X['spkts'] == 1) & (X['dbytes'] == 0)).astype(float)\n    return X",
    "score": 0.6638492771193918
  }
]
```
