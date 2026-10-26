"""Remeasure frozen regression recipes; historical scores never select a winner."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from reference_recipes import REGRESSION_TEMPORAL_ANCHORS
from research import write_json
from research_policy import recipe_key, reference_recipe
from temporal_research import TemporalResearch


class TemporalAnchors(TemporalResearch):
    """Trusted, schema-specific starting candidates on this run's temporal folds."""
    export_reserve_seconds = 180

    def _anchor_identity(self, config):
        identity = self._identity(config)
        identity['hashes']['temporal_anchors.py'] = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        return identity

    def _has_time(self):
        deadline = os.getenv('AUTOML_WORKER_DEADLINE')
        return not deadline or float(deadline) - time.time() > self.export_reserve_seconds

    def _read_current(self, suite_id, mode):
        if len(suite_id) != 16 or any(c not in '0123456789abcdef' for c in suite_id):
            raise ValueError('Invalid temporal anchor suite ID')
        folder = self.run / 'temporal' / suite_id
        identity = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
        if identity != self._identity(identity['config']):
            raise ValueError('Stale temporal anchor suite')
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        if report['suite_id'] != suite_id:
            raise ValueError('Temporal anchor report ID differs')
        result = next(r for r in report['results'] if r['mode'] == mode)
        if not all(math.isfinite(result[k]) for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')):
            raise ValueError('Nonfinite temporal anchor metrics')
        periods = identity['config']['validation_periods']
        if len(result['folds']) != len(periods):
            raise ValueError('Incomplete temporal anchor folds')
        if [r['period'] for r in result['folds']] != periods:
            raise ValueError('Temporal anchor fold periods differ')
        required = [folder / f'{mode}_oof.npy']
        required += [folder / f'{mode}_fold_{i}.npy' for i in range(len(periods))]
        required += [folder / f'fold_{i}.npz' for i in range(len(periods))]
        if not all(path.is_file() for path in required):
            raise ValueError('Incomplete temporal anchor prediction artifacts')
        return identity, report, result

    def _cached_source(self, arguments, recipe, mode):
        state = self.engine.state()
        task = {k: state[k] for k in ('target', 'task_type', 'metric', 'features', 'ids')}
        # The suite's own versioned identity validates data and implementation.
        # Do not reuse historical artifacts merely because a mode/score matches.
        for path in sorted((self.run / 'temporal').glob('*/plan.json')):
            try:
                identity, report, _ = self._read_current(path.parent.name, mode)
                config = identity['config']
                if config.get('kind') == 'ensemble' or 'recipe' not in config:
                    continue
                if config.get('task') != task or recipe_key(config['recipe']) != recipe_key(recipe):
                    continue
                fields = ('time_column', 'validation_periods', 'key_columns', 'group_columns',
                          'key_variants', 'group_hierarchy', 'history_gate', 'smoothing', 'area_rounding')
                if any(config.get(k) != arguments[k] for k in fields):
                    continue
                if config.get('modes') != ['baseline', 'history_exact', 'history_groups', 'history_hierarchy']:
                    continue
                return report
            except (OSError, ValueError, KeyError, StopIteration, TypeError):
                continue
        return None

    def run_regression_anchors(self):
        anchor = deepcopy(REGRESSION_TEMPORAL_ANCHORS)
        state = self.engine.state()
        report = dict(applicable=False, status='not_applicable', sources=[], singles=[],
            ensemble_id=None, errors=[], historical_reference={
                'mean_rmse': anchor['historical_mean_rmse'], 'note': anchor['historical_note']},
            note='Fresh measurements on current data. No historical models/predictions are loaded. '
                 'This tool does not publish a final task submission.')
        path = self.run / 'regression_anchors.json'

        def save():
            write_json(path, report)

        if any(state.get(k) != anchor[k] for k in ('task_type', 'target', 'metric')):
            report['reason'] = 'Anchors apply only to site_eui regression/RMSE.'
            save()
            return report
        if not set(anchor['required_columns']).issubset(state.get('features', [])):
            report['reason'] = 'Required energy-use schema is absent; no anchor training was started.'
            save()
            return report
        if state.get('split') != 'time' or state.get('split_column') != anchor['time_column']:
            report['reason'] = 'Configure a time split on Year_Factor first.'
            save()
            return report
        train, test = self.engine.load()
        column = anchor['time_column']
        values = pd.to_numeric(train[column], errors='coerce')
        future = pd.to_numeric(test[column], errors='coerce')
        if (not len(values) or not len(future) or not np.isfinite(values).all() or
                not np.isfinite(future).all() or not (future > values.max()).all()):
            report['reason'] = 'Anchors require complete chronological values and strictly future test periods.'
            save()
            return report
        periods = sorted(float(p) for p in values.unique())[2:][-3:]
        if len(periods) < 2:
            report['reason'] = 'At least four training periods are needed for two outer temporal folds.'
            save()
            return report
        # Both frozen singles must be applicable before any training begins.
        reference_ids = list(dict.fromkeys(s['reference_id'] for s in anchor['sources']))
        try:
            recipes = {rid: reference_recipe(rid, state) for rid in reference_ids}
        except ValueError as exc:
            report['reason'] = str(exc)
            save()
            return report
        report.update(applicable=True, status='running', validation_periods=periods,
            identity=self._anchor_identity(dict(kind='regression_anchors_v1', anchors=anchor,
                validation_periods=periods,
                task={k: state[k] for k in ('target', 'task_type', 'metric', 'features', 'ids')},
                recipes={rid: recipe_key(recipe) for rid, recipe in recipes.items()})))
        save()
        measured = {}
        for rid, recipe in recipes.items():
            experiment = next((e for e in self.engine.state()['experiments']
                if not e.get('timed_out') and recipe_key(e) == recipe_key(recipe)
                and e.get('early_stopping_rounds', 50) == recipe.get('early_stopping_rounds', 50)), None)
            cached = experiment is not None
            if experiment is None:
                if not self._has_time():
                    report['errors'].append(dict(stage='single', reference_id=rid,
                        reason='Deadline reserve reached; completed candidates remain available.'))
                    save()
                    continue
                try:
                    result = self.engine.train(
                        hypothesis='Remeasure frozen temporal anchor recipe ' + rid,
                        params=deepcopy(recipe['params']), algorithm=recipe['algorithm'],
                        feature_code=recipe.get('feature_code', ''),
                        early_stopping_rounds=recipe.get('early_stopping_rounds', 50))
                    experiment = result['experiment']
                    if experiment.get('timed_out'):
                        raise TimeoutError('Anchor single training timed out')
                except Exception as exc:
                    report['errors'].append(dict(stage='single', reference_id=rid,
                        reason=f'{type(exc).__name__}: {exc}'))
                    save()
                    continue
            measured[rid] = experiment
            report['singles'].append(dict(reference_id=rid, experiment_id=experiment['id'],
                score=experiment.get('score'), metric=state['metric'], cached=cached,
                note='Single frozen configured split, separate from temporal fold metrics.'))
            save()
        for source in anchor['sources']:
            rid = source['reference_id']
            if rid not in measured:
                continue
            arguments = dict(base_experiment_id=measured[rid]['id'], time_column=column,
                validation_periods=periods, key_columns=source['key_columns'],
                **{k: deepcopy(anchor[k]) for k in ('group_columns', 'key_variants',
                    'group_hierarchy', 'history_hierarchy', 'history_gate', 'residual',
                    'smoothing', 'area_rounding')})
            suite = self._cached_source(arguments, recipes[rid], source['mode'])
            cached = suite is not None
            try:
                if suite is None:
                    if not self._has_time():
                        raise TimeoutError('Deadline reserve reached before another temporal source')
                    suite = self.run_temporal_suite(**arguments)
                _, _, result = self._read_current(suite['suite_id'], source['mode'])
                report['sources'].append(dict(name=source['name'], reference_id=rid,
                    suite_id=suite['suite_id'], mode=source['mode'], cached=cached,
                    source_historical_suite_id=source['source_suite_id'],
                    measurements={k: result[k] for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')},
                    selected_mode=suite['selected_mode']))
            except Exception as exc:
                report['errors'].append(dict(stage='temporal', source=source['name'],
                    reason=f'{type(exc).__name__}: {exc}'))
            save()
        if len(report['sources']) == len(anchor['sources']):
            try:
                ensemble = self.blend_temporal_suites(
                    [s['suite_id'] for s in report['sources']],
                    member_modes=[s['mode'] for s in report['sources']])
                _, _, result = self._read_current(ensemble['suite_id'], ensemble['selected_mode'])
                report.update(ensemble_id=ensemble['suite_id'],
                    ensemble_mode=ensemble['selected_mode'],
                    ensemble_measurements={k: result[k] for k in ('mean_rmse', 'latest_rmse', 'worst_rmse')})
            except Exception as exc:
                report['errors'].append(dict(stage='ensemble', reason=f'{type(exc).__name__}: {exc}'))
        report['status'] = 'completed' if report['ensemble_id'] else 'partial'
        report['suite_ids'] = [s['suite_id'] for s in report['sources']]
        report['report_path'] = str(path)
        save()
        return report
