"""Reference recipes embedded in code: the platform transfers root Python files only.

Historical scores are provenance, not measurements for a new run.
"""

REFERENCE_RECIPES = [{'reference_id': 'regression_231',
  'task_type': 'regression',
  'target': 'site_eui',
  'required_columns': ['Year_Factor',
                       'State_Factor',
                       'building_class',
                       'facility_type',
                       'floor_area',
                       'year_built',
                       'energy_star_rating',
                       'ELEVATION',
                       'january_min_temp_c',
                       'january_avg_temp_c',
                       'january_max_temp_c',
                       'february_min_temp_c',
                       'february_avg_temp_c',
                       'february_max_temp_c',
                       'march_min_temp_c',
                       'march_avg_temp_c',
                       'march_max_temp_c',
                       'april_min_temp_c',
                       'april_avg_temp_c',
                       'april_max_temp_c',
                       'may_min_temp_c',
                       'may_avg_temp_c',
                       'may_max_temp_c',
                       'june_min_temp_c',
                       'june_avg_temp_c',
                       'june_max_temp_c',
                       'july_min_temp_c',
                       'july_avg_temp_c',
                       'july_max_temp_c',
                       'august_min_temp_c',
                       'august_avg_temp_c',
                       'august_max_temp_c',
                       'september_min_temp_c',
                       'september_avg_temp_c',
                       'september_max_temp_c',
                       'october_min_temp_c',
                       'october_avg_temp_c',
                       'october_max_temp_c',
                       'november_min_temp_c',
                       'november_avg_temp_c',
                       'november_max_temp_c',
                       'december_min_temp_c',
                       'december_avg_temp_c',
                       'december_max_temp_c',
                       'cooling_degree_days_c',
                       'heating_degree_days_c',
                       'precipitation_inches',
                       'snowfall_inches',
                       'snowdepth_inches',
                       'avg_temp_c',
                       'days_below_minus1C',
                       'days_below_minus7C',
                       'days_below_minus12C',
                       'days_below_minus18C',
                       'days_above_27C',
                       'days_above_32C',
                       'days_above_38C',
                       'days_above_43C',
                       'direction_max_wind_speed',
                       'direction_peak_wind_speed',
                       'max_wind_speed',
                       'days_with_fog'],
  'algorithm': 'catboost',
  'params': {'depth': 7, 'iterations': 2000, 'learning_rate': 0.045, 'l2_leaf_reg': 3, 'random_strength': 1},
  'feature_code': 'def _h(s, m=999983):\n'
                  "    return int(s.encode('utf-8').hex()[:14], 16) % m\n"
                  'def transform(X):\n'
                  '    X = X.copy()\n'
                  "    if 'id' in X.columns:\n"
                  "        X = X.drop(columns=['id'])\n"
                  "    for c in ['State_Factor', 'building_class', 'facility_type']:\n"
                  '        X[c] = X[c].astype(str).map(_h)\n'
                  "    X['year_built'] = X['year_built'].where(X['year_built'] > 0)\n"
                  "    la = np.log1p(X['floor_area'])\n"
                  "    X['floor_area_log'] = la\n"
                  "    X['hdd_log'] = np.log1p(X['heating_degree_days_c'])\n"
                  "    X['cdd_log'] = np.log1p(X['cooling_degree_days_c'])\n"
                  "    X['hdd_minus_cdd'] = X['heating_degree_days_c'] - X['cooling_degree_days_c']\n"
                  '    keep = '
                  "['Year_Factor','State_Factor','building_class','facility_type','floor_area','floor_area_log',\n"
                  '            '
                  "'year_built','energy_star_rating','ELEVATION','hdd_log','cdd_log','hdd_minus_cdd']\n"
                  '    X = X[keep]\n'
                  '    return X',
  'early_stopping_rounds': 50,
  'historical_score': 38.261194081354745,
  'metric': 'rmse',
  'source_data_sha256': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
  'provenance': 'Historical local recipe regression #231; must be retrained; no weights or predictions '
                'included'},
 {'reference_id': 'regression_213',
  'task_type': 'regression',
  'target': 'site_eui',
  'required_columns': ['Year_Factor',
                       'State_Factor',
                       'building_class',
                       'facility_type',
                       'floor_area',
                       'year_built',
                       'energy_star_rating',
                       'ELEVATION',
                       'january_min_temp_c',
                       'january_avg_temp_c',
                       'january_max_temp_c',
                       'february_min_temp_c',
                       'february_avg_temp_c',
                       'february_max_temp_c',
                       'march_min_temp_c',
                       'march_avg_temp_c',
                       'march_max_temp_c',
                       'april_min_temp_c',
                       'april_avg_temp_c',
                       'april_max_temp_c',
                       'may_min_temp_c',
                       'may_avg_temp_c',
                       'may_max_temp_c',
                       'june_min_temp_c',
                       'june_avg_temp_c',
                       'june_max_temp_c',
                       'july_min_temp_c',
                       'july_avg_temp_c',
                       'july_max_temp_c',
                       'august_min_temp_c',
                       'august_avg_temp_c',
                       'august_max_temp_c',
                       'september_min_temp_c',
                       'september_avg_temp_c',
                       'september_max_temp_c',
                       'october_min_temp_c',
                       'october_avg_temp_c',
                       'october_max_temp_c',
                       'november_min_temp_c',
                       'november_avg_temp_c',
                       'november_max_temp_c',
                       'december_min_temp_c',
                       'december_avg_temp_c',
                       'december_max_temp_c',
                       'cooling_degree_days_c',
                       'heating_degree_days_c',
                       'precipitation_inches',
                       'snowfall_inches',
                       'snowdepth_inches',
                       'avg_temp_c',
                       'days_below_minus1C',
                       'days_below_minus7C',
                       'days_below_minus12C',
                       'days_below_minus18C',
                       'days_above_27C',
                       'days_above_32C',
                       'days_above_38C',
                       'days_above_43C',
                       'direction_max_wind_speed',
                       'direction_peak_wind_speed',
                       'max_wind_speed',
                       'days_with_fog'],
  'algorithm': 'catboost',
  'params': {'depth': 9, 'iterations': 2000, 'l2_leaf_reg': 3, 'learning_rate': 0.045, 'random_strength': 1},
  'feature_code': 'def _h(s, m=999983):\n'
                  "    return int(s.encode('utf-8').hex()[:14], 16) % m\n"
                  'def transform(X):\n'
                  '    X = X.copy()\n'
                  "    if 'id' in X.columns:\n"
                  "        X = X.drop(columns=['id'])\n"
                  "    for c in ['State_Factor', 'building_class', 'facility_type']:\n"
                  '        X[c] = X[c].astype(str).map(_h)\n'
                  "    X['year_built'] = X['year_built'].where(X['year_built'] > 0)\n"
                  "    la = np.log1p(X['floor_area'])\n"
                  "    X['floor_area_log'] = la\n"
                  "    X['hdd_log'] = np.log1p(X['heating_degree_days_c'])\n"
                  "    X['cdd_log'] = np.log1p(X['cooling_degree_days_c'])\n"
                  "    X['hdd_minus_cdd'] = X['heating_degree_days_c'] - X['cooling_degree_days_c']\n"
                  '    keep = '
                  "['Year_Factor','State_Factor','building_class','facility_type','floor_area','floor_area_log',\n"
                  '            '
                  "'year_built','energy_star_rating','ELEVATION','hdd_log','cdd_log','hdd_minus_cdd']\n"
                  '    X = X[keep]\n'
                  '    return X',
  'early_stopping_rounds': 50,
  'historical_score': 38.596589415766374,
  'metric': 'rmse',
  'source_data_sha256': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
  'provenance': 'Historical local recipe regression #213; must be retrained; no weights or predictions '
                'included'},
 {'reference_id': 'regression_238',
  'task_type': 'regression',
  'target': 'site_eui',
  'required_columns': ['Year_Factor',
                       'State_Factor',
                       'building_class',
                       'facility_type',
                       'floor_area',
                       'year_built',
                       'energy_star_rating',
                       'ELEVATION',
                       'january_min_temp_c',
                       'january_avg_temp_c',
                       'january_max_temp_c',
                       'february_min_temp_c',
                       'february_avg_temp_c',
                       'february_max_temp_c',
                       'march_min_temp_c',
                       'march_avg_temp_c',
                       'march_max_temp_c',
                       'april_min_temp_c',
                       'april_avg_temp_c',
                       'april_max_temp_c',
                       'may_min_temp_c',
                       'may_avg_temp_c',
                       'may_max_temp_c',
                       'june_min_temp_c',
                       'june_avg_temp_c',
                       'june_max_temp_c',
                       'july_min_temp_c',
                       'july_avg_temp_c',
                       'july_max_temp_c',
                       'august_min_temp_c',
                       'august_avg_temp_c',
                       'august_max_temp_c',
                       'september_min_temp_c',
                       'september_avg_temp_c',
                       'september_max_temp_c',
                       'october_min_temp_c',
                       'october_avg_temp_c',
                       'october_max_temp_c',
                       'november_min_temp_c',
                       'november_avg_temp_c',
                       'november_max_temp_c',
                       'december_min_temp_c',
                       'december_avg_temp_c',
                       'december_max_temp_c',
                       'cooling_degree_days_c',
                       'heating_degree_days_c',
                       'precipitation_inches',
                       'snowfall_inches',
                       'snowdepth_inches',
                       'avg_temp_c',
                       'days_below_minus1C',
                       'days_below_minus7C',
                       'days_below_minus12C',
                       'days_below_minus18C',
                       'days_above_27C',
                       'days_above_32C',
                       'days_above_38C',
                       'days_above_43C',
                       'direction_max_wind_speed',
                       'direction_peak_wind_speed',
                       'max_wind_speed',
                       'days_with_fog'],
  'algorithm': 'catboost',
  'params': {'depth': 7, 'iterations': 2000, 'l2_leaf_reg': 5, 'learning_rate': 0.045, 'random_strength': 1},
  'feature_code': 'def _h(s, m=999983):\n'
                  "    return int(s.encode('utf-8').hex()[:14], 16) % m\n"
                  'def transform(X):\n'
                  '    X = X.copy()\n'
                  "    if 'id' in X.columns:\n"
                  "        X = X.drop(columns=['id'])\n"
                  "    for c in ['State_Factor', 'building_class', 'facility_type']:\n"
                  '        X[c] = X[c].astype(str).map(_h)\n'
                  "    X['year_built'] = X['year_built'].where(X['year_built'] > 0)\n"
                  "    la = np.log1p(X['floor_area'])\n"
                  "    X['floor_area_log'] = la\n"
                  "    X['hdd_log'] = np.log1p(X['heating_degree_days_c'])\n"
                  "    X['cdd_log'] = np.log1p(X['cooling_degree_days_c'])\n"
                  "    X['hdd_minus_cdd'] = X['heating_degree_days_c'] - X['cooling_degree_days_c']\n"
                  '    keep = '
                  "['Year_Factor','State_Factor','building_class','facility_type','floor_area','floor_area_log',\n"
                  '            '
                  "'year_built','energy_star_rating','ELEVATION','hdd_log','cdd_log','hdd_minus_cdd']\n"
                  '    X = X[keep]\n'
                  '    return X',
  'early_stopping_rounds': 50,
  'historical_score': 38.35483618353072,
  'metric': 'rmse',
  'source_data_sha256': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
  'provenance': 'Historical local recipe regression #238; must be retrained; no weights or predictions '
                'included'},
 {'reference_id': 'classification_49',
  'task_type': 'classification',
  'target': 'attack_cat',
  'required_columns': ['dur',
                       'proto',
                       'service',
                       'state',
                       'spkts',
                       'dpkts',
                       'sbytes',
                       'dbytes',
                       'rate',
                       'sttl',
                       'dttl',
                       'sload',
                       'dload',
                       'sloss',
                       'dloss',
                       'sinpkt',
                       'dinpkt',
                       'sjit',
                       'djit',
                       'swin',
                       'stcpb',
                       'dtcpb',
                       'dwin',
                       'tcprtt',
                       'synack',
                       'ackdat',
                       'smean',
                       'dmean',
                       'trans_depth',
                       'response_body_len',
                       'ct_srv_src',
                       'ct_state_ttl',
                       'ct_dst_ltm',
                       'ct_src_dport_ltm',
                       'ct_dst_sport_ltm',
                       'ct_dst_src_ltm',
                       'is_ftp_login',
                       'ct_ftp_cmd',
                       'ct_flw_http_mthd',
                       'ct_src_ltm',
                       'ct_srv_dst',
                       'is_sm_ips_ports'],
  'algorithm': 'xgboost',
  'params': {'auto_class_weights': 'SqrtBalanced',
             'colsample_bytree': 0.6,
             'depth': 4,
             'gamma': 1.0,
             'iterations': 500,
             'l2_leaf_reg': 3,
             'learning_rate': 0.05,
             'min_child_weight': 5,
             'subsample': 0.8},
  'feature_code': 'def transform(X):\n'
                  '    X = X.copy()\n'
                  "    log_cols = ['dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',\n"
                  "                'stcpb', 'dtcpb', 'sbytes', 'dbytes', 'response_body_len']\n"
                  '    for c in log_cols:\n'
                  '        if c in X.columns:\n'
                  '            X[c] = np.log1p(np.maximum(X[c], 0.0))\n'
                  "    for c in ['spkts', 'dpkts']:\n"
                  "        X['log1p_' + c] = np.log1p(np.maximum(X[c], 0.0))\n"
                  "    X['bytes_per_pkt_s'] = X['sbytes'] / np.maximum(X['spkts'], 1)\n"
                  "    X['bytes_per_pkt_d'] = X['dbytes'] / np.maximum(X['dpkts'], 1)\n"
                  "    X['pkts_ratio'] = X['dpkts'] / np.maximum(X['spkts'], 1)\n"
                  "    X['bytes_ratio'] = X['dbytes'] / np.maximum(X['sbytes'], 1)\n"
                  "    X['load_ratio'] = X['dload'] / np.maximum(X['sload'], 1)\n"
                  "    X['ttl_diff'] = X['dttl'] - X['sttl']\n"
                  "    X['jit_ratio'] = X['djit'] / np.maximum(X['sjit'], 1)\n"
                  "    X['pkt_gap'] = X['dinpkt'] - X['sinpkt']\n"
                  '    return X',
  'early_stopping_rounds': 50,
  'historical_score': 0.6651474102407088,
  'metric': 'f1_macro',
  'source_data_sha256': '171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0',
  'provenance': 'Historical local recipe classification #49; must be retrained; no weights or predictions '
                'included'},
 {'reference_id': 'regression_231_full_categories',
  'task_type': 'regression',
  'target': 'site_eui',
  'required_columns': ['Year_Factor',
                       'State_Factor',
                       'building_class',
                       'facility_type',
                       'floor_area',
                       'year_built',
                       'energy_star_rating',
                       'ELEVATION',
                       'january_min_temp_c',
                       'january_avg_temp_c',
                       'january_max_temp_c',
                       'february_min_temp_c',
                       'february_avg_temp_c',
                       'february_max_temp_c',
                       'march_min_temp_c',
                       'march_avg_temp_c',
                       'march_max_temp_c',
                       'april_min_temp_c',
                       'april_avg_temp_c',
                       'april_max_temp_c',
                       'may_min_temp_c',
                       'may_avg_temp_c',
                       'may_max_temp_c',
                       'june_min_temp_c',
                       'june_avg_temp_c',
                       'june_max_temp_c',
                       'july_min_temp_c',
                       'july_avg_temp_c',
                       'july_max_temp_c',
                       'august_min_temp_c',
                       'august_avg_temp_c',
                       'august_max_temp_c',
                       'september_min_temp_c',
                       'september_avg_temp_c',
                       'september_max_temp_c',
                       'october_min_temp_c',
                       'october_avg_temp_c',
                       'october_max_temp_c',
                       'november_min_temp_c',
                       'november_avg_temp_c',
                       'november_max_temp_c',
                       'december_min_temp_c',
                       'december_avg_temp_c',
                       'december_max_temp_c',
                       'cooling_degree_days_c',
                       'heating_degree_days_c',
                       'precipitation_inches',
                       'snowfall_inches',
                       'snowdepth_inches',
                       'avg_temp_c',
                       'days_below_minus1C',
                       'days_below_minus7C',
                       'days_below_minus12C',
                       'days_below_minus18C',
                       'days_above_27C',
                       'days_above_32C',
                       'days_above_38C',
                       'days_above_43C',
                       'direction_max_wind_speed',
                       'direction_peak_wind_speed',
                       'max_wind_speed',
                       'days_with_fog'],
  'algorithm': 'catboost',
  'params': {'depth': 7, 'iterations': 2000, 'learning_rate': 0.045, 'l2_leaf_reg': 3, 'random_strength': 1},
  'feature_code': 'def transform(X):\n'
                  '    X = X.copy()\n'
                  "    if 'id' in X.columns:\n"
                  "        X = X.drop(columns=['id'])\n"
                  "    for c in ['State_Factor', 'building_class', 'facility_type']:\n"
                  '        X[c] = X[c].astype(str)\n'
                  "    X['year_built'] = X['year_built'].where(X['year_built'] > 0)\n"
                  "    la = np.log1p(X['floor_area'])\n"
                  "    X['floor_area_log'] = la\n"
                  "    X['hdd_log'] = np.log1p(X['heating_degree_days_c'])\n"
                  "    X['cdd_log'] = np.log1p(X['cooling_degree_days_c'])\n"
                  "    X['hdd_minus_cdd'] = X['heating_degree_days_c'] - X['cooling_degree_days_c']\n"
                  '    keep = '
                  "['Year_Factor','State_Factor','building_class','facility_type','floor_area','floor_area_log',\n"
                  '            '
                  "'year_built','energy_star_rating','ELEVATION','hdd_log','cdd_log','hdd_minus_cdd']\n"
                  '    X = X[keep]\n'
                  '    return X',
  'early_stopping_rounds': 50,
  'historical_score': None,
  'metric': 'rmse',
  'source_data_sha256': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
  'provenance': 'Controlled candidate: same params and numeric transforms as regression_231; native full '
                'string categories instead of truncated numeric encoding. No measured score yet.',
  'comparison_of': 'regression_231'}]

# Frozen recipe from the successful local control comparison. No fitted objects or
# predictions are embedded; its historical score is provenance only.
REFERENCE_RECIPES.append({
    'reference_id': 'regression_control_lgbm_compact',
    'task_type': 'regression', 'target': 'site_eui', 'metric': 'rmse',
    'required_columns': ['Year_Factor', 'State_Factor', 'building_class',
        'facility_type', 'floor_area', 'year_built', 'energy_star_rating',
        'ELEVATION', 'heating_degree_days_c', 'cooling_degree_days_c'],
    'algorithm': 'lightgbm',
    # The source had no iterations override. Preserve the trusted engine default.
    'params': {'depth': 7, 'learning_rate': 0.05, 'min_child_samples': 50,
               'num_leaves': 63, 'reg_alpha': 0.0, 'seed': 42},
    'feature_code': "def transform(X):\n"
        "    X = X.copy()\n"
        "    if 'id' in X.columns:\n"
        "        X = X.drop(columns=['id'])\n"
        "    X['year_built'] = X['year_built'].where(X['year_built'] > 0)\n"
        "    la = np.log1p(X['floor_area'])\n"
        "    X['floor_area_log'] = la\n"
        "    X['hdd_log'] = np.log1p(X['heating_degree_days_c'])\n"
        "    X['cdd_log'] = np.log1p(X['cooling_degree_days_c'])\n"
        "    X['hdd_minus_cdd'] = X['heating_degree_days_c'] - X['cooling_degree_days_c']\n"
        "    keep = ['Year_Factor','State_Factor','building_class','facility_type','floor_area','floor_area_log',\n"
        "            'year_built','energy_star_rating','ELEVATION','hdd_log','cdd_log','hdd_minus_cdd']\n"
        "    X = X[keep]\n"
        "    return X",
    'early_stopping_rounds': 50,
    'historical_score': 44.70285083651112,
    'source_data_sha256': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
    'provenance': 'Local control comparison 20261025_132054_060709, experiment 5; '
                  'single-holdout score, not a new measurement or temporal score.'})

REGRESSION_TEMPORAL_ANCHORS = {
    'version': 'control_temporal_anchors_v1',
    'task_type': 'regression', 'target': 'site_eui', 'metric': 'rmse',
    'time_column': 'Year_Factor',
    'required_columns': ['Year_Factor', 'State_Factor', 'building_class',
        'facility_type', 'floor_area', 'year_built', 'energy_star_rating',
        'ELEVATION', 'heating_degree_days_c', 'cooling_degree_days_c'],
    'historical_mean_rmse': 39.927404,
    'historical_note': 'Local control comparison only. Always remeasure on this run; '
                       'this is neither a platform task score nor a guaranteed baseline.',
    'sources': [
        {'name': 'catboost_hierarchy', 'reference_id': 'regression_231',
         'source_suite_id': 'b9fbd01a55f44729', 'mode': 'history_hierarchy',
         'key_columns': ['State_Factor', 'facility_type', 'building_class',
                         'floor_area', 'year_built', 'ELEVATION']},
        {'name': 'catboost_exact_without_elevation', 'reference_id': 'regression_231',
         'source_suite_id': 'bbc11fdbff0ab187', 'mode': 'history_exact',
         'key_columns': ['State_Factor', 'facility_type', 'building_class',
                         'floor_area', 'year_built']},
        {'name': 'lightgbm_hierarchy', 'reference_id': 'regression_control_lgbm_compact',
         'source_suite_id': '1b2e3585bae78f0a', 'mode': 'history_hierarchy',
         'key_columns': ['State_Factor', 'facility_type', 'building_class',
                         'floor_area', 'year_built', 'ELEVATION']}
    ],
    'group_columns': ['State_Factor', 'facility_type'],
    'key_variants': [['State_Factor', 'facility_type', 'floor_area', 'year_built'],
                     ['State_Factor', 'facility_type', 'floor_area']],
    'group_hierarchy': [['facility_type']],
    'smoothing': 20.0,
    'area_rounding': None,
    'history_hierarchy': True,
    'history_gate': False,
    'residual': False
}
