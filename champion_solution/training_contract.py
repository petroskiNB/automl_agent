"""Algorithm-specific tool contracts and deterministic failed-request memory."""
import copy
import hashlib
import json
from pathlib import Path

KEYS = {
    'catboost': {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'random_strength', 'auto_class_weights', 'seed'},
    'lightgbm': {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'num_leaves', 'min_child_samples',
                'subsample', 'subsample_freq', 'colsample_bytree', 'reg_alpha', 'min_split_gain', 'seed'},
    'xgboost': {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'min_child_weight', 'subsample',
               'colsample_bytree', 'reg_alpha', 'gamma', 'auto_class_weights', 'seed'},
    'torch_mlp': {'iterations', 'width', 'layers', 'batch_size', 'dropout', 'weight_decay', 'learning_rate',
                  'seed', 'max_categories', 'loss', 'target_transform', 'auto_class_weights'},
}
KEYS['torch_resnet'] = set(KEYS['torch_mlp'])
KEYS['torch_tabm'] = KEYS['torch_mlp'] | {'ensemble_size', 'num_embedding', 'embedding_dim', 'n_bins', 'numeric_preprocessing'}
TRAINING_TOOLS = {'train_' + algorithm: algorithm for algorithm in KEYS}


def split_tools(template):
    tools = []
    for algorithm, allowed in KEYS.items():
        tool = copy.deepcopy(template)
        function = tool['function']
        function['name'] = 'train_' + algorithm
        function['description'] = 'Train ' + algorithm + ' on the frozen split. Only its own parameters are accepted.'
        props = function['parameters']['properties']
        props.pop('algorithm', None)
        props['params']['properties'] = {k: v for k, v in props['params']['properties'].items() if k in allowed}
        props['params']['additionalProperties'] = False
        tools.append(tool)
    return tools


def canonical_arguments(name, arguments):
    arguments = copy.deepcopy(arguments)
    if name in TRAINING_TOOLS:
        algorithm = TRAINING_TOOLS[name]
        if arguments.get('algorithm', algorithm) != algorithm:
            raise ValueError('Algorithm conflicts with tool ' + name)
        arguments['algorithm'] = algorithm
    return arguments


def validate_arguments(arguments, task_type=None):
    args = copy.deepcopy(arguments)
    algorithm = args.get('algorithm', 'catboost')
    if algorithm not in KEYS:
        raise ValueError('Unsupported algorithm: ' + str(algorithm))
    params = args.get('params') or {}
    if not isinstance(params, dict):
        raise ValueError('params must be an object')
    unsupported = sorted(set(params) - KEYS[algorithm])
    if unsupported:
        raise ValueError(f'{algorithm}: unsupported parameters {unsupported}; allowed: {sorted(KEYS[algorithm])}. '
                         'Remove irrelevant fields; do not repeat the unchanged request.')
    if params.get('auto_class_weights') is None:
        params.pop('auto_class_weights', None)
    if task_type == 'regression' and params.get('auto_class_weights'):
        raise ValueError('auto_class_weights requires classification')
    code = args.get('feature_code', '')
    if not isinstance(code, str):
        raise ValueError('feature_code must be a string')
    compile(code, '<features>', 'exec')
    args['params'] = params
    return args


def request_key(arguments):
    args = {k: arguments.get(k, default) for k, default in
            [('algorithm', 'catboost'), ('params', {}), ('feature_code', ''), ('early_stopping_rounds', 50)]}
    if isinstance(args['params'], dict):
        args['params'] = dict(args['params'])
        if args['params'].get('auto_class_weights') is None:
            args['params'].pop('auto_class_weights', None)
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def rejection_memory(log_dir, arguments, error=None):
    path = Path(log_dir) / 'rejected_training_requests.json'
    memory = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    key = request_key(arguments)
    if error is not None:
        memory[key] = dict(error=str(error)[-2000:], algorithm=arguments.get('algorithm', 'catboost'))
        path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding='utf-8')
    return memory.get(key)
