"""Past-only hierarchical history; wider matches are hypotheses, never entity IDs."""
import numpy as np
import pandas as pd
from temporal_research import HistoricalFeatures


class HierarchicalHistory(HistoricalFeatures):
    def __init__(self, time_column, target, key_columns, group_columns,
                 key_variants=None, group_hierarchy=None, smoothing=20., area_rounding=None):
        super().__init__(time_column, target, key_columns, group_columns, 'hierarchy')
        keys = [list(key_columns)] + [list(v) for v in (key_variants or [])]
        groups = [list(group_columns)] + [list(v) for v in (group_hierarchy or [])]
        self.levels = []
        for columns in keys + groups:
            if columns not in self.levels:
                self.levels.append(columns)
        self.key_levels = sum(v in keys for v in self.levels)
        self.smoothing, self.area_rounding = float(smoothing), area_rounding

    def _keys(self, frame):
        frame = frame.copy()
        if self.area_rounding is not None and 'floor_area' in frame:
            frame['floor_area'] = (pd.to_numeric(frame['floor_area'], errors='coerce') /
                                   self.area_rounding).round() * self.area_rounding
        return frame

    def fit(self, frame):
        columns = list(dict.fromkeys([self.time_column, self.target] + [c for level in self.levels for c in level]))
        self.reference = self._keys(frame[columns])
        if not np.isfinite(pd.to_numeric(self.reference[self.target], errors='coerce')).all():
            raise ValueError('Historical targets must be finite numeric values')
        return self

    def transform(self, frame):
        query = self._keys(frame.drop(columns=[self.target], errors='ignore')).reset_index(drop=True)
        periods = pd.to_numeric(query[self.time_column], errors='coerce')
        if not np.isfinite(periods).all():
            raise ValueError('Complete numeric time values are required')
        outputs = []
        for period in sorted(periods.unique()):
            positions = np.flatnonzero(periods.to_numpy() == period)
            part = query.iloc[positions].reset_index(drop=True)
            past = self.reference.loc[self.reference[self.time_column] < period]
            levels = []
            for i, columns in enumerate(self.levels):
                prefix = f'hier_{i}_'
                values = self._lookup(past, part, columns, prefix)
                values[prefix+'gap'] = period - values.pop(prefix+'last_period')
                # Dispersion exposes ambiguous coarse matches instead of calling them buildings.
                available = past.dropna(subset=columns)
                if len(available):
                    std = available.groupby(columns, as_index=False)[self.target].std().rename(columns={self.target: prefix+'std'})
                    values[prefix+'std'] = part[columns].merge(std, on=columns, how='left', validate='many_to_one')[prefix+'std'].to_numpy()
                else:
                    values[prefix+'std'] = np.nan
                levels.append(values)
            out = pd.concat(levels, axis=1)
            estimate = pd.Series(float(past[self.target].mean()) if len(past) else np.nan, index=out.index)
            selected = np.full(len(part), -1, dtype=int)
            # Shrink each narrower estimate toward its broader parent using past counts only.
            for i in range(len(levels)-1, -1, -1):
                count, mean = levels[i][f'hier_{i}_count'], levels[i][f'hier_{i}_mean']
                known = (count > 0) & mean.notna()
                weight = count / (count + self.smoothing) if self.smoothing else (count > 0).astype(float)
                shrunk = weight * mean + (1-weight) * estimate
                estimate = estimate.where(~known, shrunk)
                selected[known.to_numpy()] = i
            out['hist_baseline'] = estimate
            out['hist_global_count'] = len(past)
            out['hist_global_mean'] = float(past[self.target].mean()) if len(past) else np.nan
            out['hist_exact_seen'] = (levels[0]['hier_0_count'] > 0).astype(float)
            out['hist_any_key_seen'] = np.any([v[f'hier_{i}_count'].to_numpy() > 0
                                               for i, v in enumerate(levels[:self.key_levels])], axis=0).astype(float)
            out['hist_any_group_seen'] = np.any([v[f'hier_{i}_count'].to_numpy() > 0
                                                 for i, v in enumerate(levels)], axis=0).astype(float)
            out['hist_selected_level'] = selected
            out.index = positions
            outputs.append(out)
        if not outputs:
            raise ValueError('Empty history transform')
        result = pd.concat(outputs).sort_index()
        result.index = frame.index
        return result
