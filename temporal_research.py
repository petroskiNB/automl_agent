"""Frozen temporal ablations with strictly past-only fitted features.

Results live separately from the legacy single-holdout experiment registry.
No tool here overwrites the task's submission.csv or legacy best model.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research import Research, write_json


MODES = ("baseline", "history_exact", "history_groups")


class HistoricalFeatures:
    """fit stores available history; transform never reads query targets.

    Same-period peers are excluded even in the training transformation.
    Missing key components are deliberately not treated as an entity match.
    """
    def __init__(self, time_column, target, key_columns, group_columns, mode):
        self.time_column, self.target = time_column, target
        self.key_columns, self.group_columns = list(key_columns), list(group_columns)
        self.mode = mode

    def fit(self, frame):
        columns = list(dict.fromkeys([self.time_column, self.target] + self.key_columns + self.group_columns))
        self.reference = frame[columns].copy(deep=True)
        if not np.isfinite(pd.to_numeric(self.reference[self.target], errors="coerce")).all():
            raise ValueError("Historical targets must be finite numeric values")
        return self

    def _lookup(self, past, query, columns, prefix):
        names = [prefix + suffix for suffix in ("count", "mean", "last", "last_period")]
        out = pd.DataFrame(np.nan, index=np.arange(len(query)), columns=names)
        valid = query[columns].notna().all(axis=1).to_numpy()
        available = past.dropna(subset=columns)
        if available.empty or not valid.any():
            out[names[0]] = 0.
            return out
        stats = available.groupby(columns, dropna=False)[self.target].agg(["count", "mean"])
        # Multiple observations in the last year are averaged, never resolved by row order.
        yearly = available.groupby(columns + [self.time_column], as_index=False)[self.target].mean()
        last = yearly.sort_values(self.time_column).groupby(columns, sort=False).tail(1).set_index(columns)
        stats = stats.join(last[[self.target, self.time_column]].rename(
            columns={self.target: "last", self.time_column: "last_period"}))
        stats.columns = names
        request = query.loc[valid, columns].copy()
        request["_position"] = np.flatnonzero(valid)
        matched = request.merge(stats.reset_index(), on=columns, how="left", validate="many_to_one")
        out.loc[matched["_position"], names] = matched[names].to_numpy()
        out[names[0]] = out[names[0]].fillna(0.)
        return out

    def transform(self, frame):
        # Ignore targets even if the caller supplies labelled training/validation rows.
        query = frame.drop(columns=[self.target], errors="ignore").reset_index(drop=True)
        periods = pd.to_numeric(query[self.time_column], errors="coerce")
        if not np.isfinite(periods).all():
            raise ValueError("Complete finite numeric time values are required")
        outputs = []
        for period in sorted(periods.unique()):
            positions = np.flatnonzero(periods.to_numpy() == period)
            part = query.iloc[positions].reset_index(drop=True)
            past = self.reference.loc[self.reference[self.time_column] < period]
            exact = self._lookup(past, part, self.key_columns, "hist_exact_")
            exact["hist_exact_seen"] = (exact.hist_exact_count > 0).astype(float)
            exact["hist_exact_gap"] = period - exact.pop("hist_exact_last_period")
            if self.mode == "history_groups":
                group = self._lookup(past, part, self.group_columns, "hist_group_")
                group["hist_group_gap"] = period - group.pop("hist_group_last_period")
                exact = pd.concat([exact, group], axis=1)
            # An unknown past stays NaN; never fill with a full-train target mean.
            global_mean = float(past[self.target].mean()) if len(past) else np.nan
            exact["hist_global_mean"] = global_mean
            exact["hist_global_count"] = len(past)
            fallback = exact["hist_exact_mean"].copy()
            if self.mode == "history_groups":
                fallback = fallback.fillna(exact["hist_group_mean"])
            exact["hist_baseline"] = fallback.fillna(global_mean)
            exact.index = positions
            outputs.append(exact)
        if not outputs:
            raise ValueError("Empty transform input")
        result = pd.concat(outputs).sort_index()
        result.index = frame.index
        return result


def temporal_folds(frame, time_column, validation_periods):
    values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Time must be finite numeric chronological values")
    if not 2 <= len(validation_periods) <= 5 or len(set(validation_periods)) != len(validation_periods):
        raise ValueError("Choose 2..5 distinct validation periods")
    folds = []
    for period in sorted(validation_periods):
        if not np.isfinite(period):
            raise ValueError("Non-finite validation period")
        tr, va = np.flatnonzero(values < period), np.flatnonzero(values == period)
        if not len(tr) or not len(va):
            raise ValueError("Empty temporal training or validation fold")
        folds.append((period, tr, va))
    return folds


def metrics(y, pred):
    error = np.asarray(pred) - np.asarray(y)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Invalid predictions or targets")
    return {"rows": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "squared_error_sum": float(np.sum(error ** 2))}


class TemporalResearch:
    def __init__(self, root, files, folder):
        self.engine = Research(root, files, folder)
        self.run = Path(root).parent

    def _design(self, frame, state, recipe, history):
        x, cats = self.engine.prepare(frame, state, recipe.get("feature_code", ""))
        if history is not None:
            features = history.transform(frame)
            if set(features) & set(x):
                raise ValueError("Historical feature names collide with recipe columns")
            x = pd.concat([x, features], axis=1)
        return x, cats

    def _history(self, fit, state, config, mode):
        return None if mode == "baseline" else HistoricalFeatures(config["time_column"], state["target"],
            config["key_columns"], config["group_columns"], mode).fit(fit)

    def _fit(self, fit, state, recipe, config, mode, iterations):
        history = self._history(fit, state, config, mode)
        x, cats = self._design(fit, state, recipe, history)
        model = self.engine.model(state, dict(recipe["params"], iterations=int(iterations)), "catboost")
        model.fit(x, fit[state["target"]], cat_features=cats)
        return {"model": model, "history": history, "recipe": recipe, "state": state}

    def _predict(self, bundle, frame):
        if "gate_models" in bundle:
            baseline, historical = bundle["gate_models"]
            seen = historical["history"].transform(frame).hist_exact_seen.to_numpy().astype(bool)
            return np.where(seen, self._predict(historical, frame), self._predict(baseline, frame))
        x, _ = self._design(frame, bundle["state"], bundle["recipe"], bundle["history"])
        return np.asarray(bundle["model"].predict(x)).reshape(-1)

    def _iterations(self, fit, state, recipe, config, mode):
        # Select tree count on a period INSIDE the outer training fold.
        column = config["time_column"]
        periods = sorted(fit[column].unique())
        if len(periods) < 2:
            raise ValueError("Need at least two past periods for internal early stopping")
        inner_fit, inner_val = fit.loc[fit[column] < periods[-1]], fit.loc[fit[column] == periods[-1]]
        history = self._history(inner_fit, state, config, mode)
        x, cats = self._design(inner_fit, state, recipe, history)
        xv, _ = self._design(inner_val, state, recipe, history)
        model = self.engine.model(state, recipe["params"], "catboost")
        model.set_params(eval_metric="RMSE")
        model.fit(x, inner_fit[state["target"]], cat_features=cats,
            eval_set=(xv, inner_val[state["target"]]), early_stopping_rounds=50, use_best_model=True)
        return int(model.tree_count_)

    def _identity(self, config):
        paths = dict(self.engine.files)
        for name in ("research.py", "temporal_research.py"):
            paths[name] = Path(__file__).with_name(name)
        hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}
        return {"config": config, "hashes": hashes}

    def run_temporal_suite(self, base_experiment_id, time_column, key_columns, group_columns,
                           validation_periods, history_gate=False):
        lock = self.run / ".temporal.lock"
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.close(handle)
            return self._run_suite(base_experiment_id, time_column, key_columns, group_columns, validation_periods, history_gate)
        finally:
            lock.unlink(missing_ok=True)

    def _run_suite(self, base_experiment_id, time_column, key_columns, group_columns, validation_periods, history_gate):
        if not isinstance(history_gate, bool):
            raise ValueError("history_gate must be boolean")
        state = self.engine.state()
        if state["task_type"] != "regression" or state["metric"] != "rmse":
            raise ValueError("First temporal implementation supports regression/RMSE only")
        recipe = next((e for e in state["experiments"] if e["id"] == base_experiment_id), None)
        if recipe is None or recipe["algorithm"] != "catboost":
            raise ValueError("Select a saved single CatBoost recipe")
        for columns in (key_columns, group_columns):
            if not 1 <= len(columns) <= 8 or len(set(columns)) != len(columns):
                raise ValueError("Choose 1..8 distinct key/group features")
            if any(c not in state["features"] or c == time_column for c in columns):
                raise ValueError("Keys must be non-time features, never ID or target")
        if time_column not in state["features"]:
            raise ValueError("Time column must be a task feature")
        train, test = self.engine.load()
        folds = temporal_folds(train, time_column, validation_periods)
        # Freeze recipe before looking at any new metric. Old recipe selection is acknowledged.
        config = dict(base_experiment_id=base_experiment_id, time_column=time_column,
                      key_columns=key_columns, group_columns=group_columns,
                      validation_periods=sorted(validation_periods),
                      recipe={k: recipe[k] for k in ("params", "feature_code", "algorithm")},
                      task={k: state[k] for k in ("target", "task_type", "metric", "features", "ids")},
                      selection="lowest equal-weight mean of outer-fold RMSE; ties prefer baseline",
                      modes=list(MODES) + (["history_gate"] if history_gate else []), inner_early_stopping=50,
                      history_gate=history_gate)
        identity = self._identity(config)
        suite_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / "temporal" / suite_id
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        write_json(directory / "plan.json", identity)
        for i, (period, tr, va) in enumerate(folds):
            if train.iloc[tr][time_column].nunique() < 2:
                raise ValueError("Each fold needs at least two training periods for internal early stopping")
            np.savez(directory / f"fold_{i}.npz", train=tr, validation=va)
        results = []
        for mode in MODES:
            out = np.full(len(train), np.nan)
            fold_results = []
            for i, (period, tr, va) in enumerate(folds):
                result_path = directory / f"{mode}_fold_{i}.json"
                prediction_path = directory / f"{mode}_fold_{i}.npy"
                if result_path.exists() and prediction_path.exists():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    pred = np.load(prediction_path)
                else:
                    started = time.monotonic()
                    fit, val = train.iloc[tr], train.iloc[va]
                    iterations = self._iterations(fit, state, recipe, config, mode)
                    bundle = self._fit(fit, state, recipe, config, mode, iterations)
                    pred = self._predict(bundle, val)
                    row = dict(period=float(period), train_rows=len(tr), iterations=iterations,
                               **metrics(val[state["target"]], pred))
                    # Coverage is measured identically for baseline and history modes.
                    coverage = self._history(fit, state, config, "history_exact").transform(val)
                    seen = coverage.hist_exact_seen.to_numpy().astype(bool)
                    row["history_coverage"] = float(seen.mean())
                    row["segments"] = {label: metrics(val[state["target"]].to_numpy()[mask], pred[mask])
                        for label, mask in (("seen_key", seen), ("unseen_key", ~seen)) if mask.any()}
                    row["seconds"] = round(time.monotonic() - started, 2)
                    joblib.dump(bundle, directory / f"{mode}_fold_{i}.joblib")
                    np.save(prediction_path, pred)
                    write_json(result_path, row)
                out[va] = pred
                fold_results.append(row)
            np.save(directory / f"{mode}_oof.npy", out)
            scores = [r["rmse"] for r in fold_results]
            results.append(dict(mode=mode, folds=fold_results, mean_rmse=float(np.mean(scores)),
                worst_rmse=float(max(scores)), std_rmse=float(np.std(scores)), latest_rmse=scores[-1],
                pooled=metrics(train.loc[np.isfinite(out), state["target"]], out[np.isfinite(out)])))
        if history_gate:
            out = np.full(len(train), np.nan)
            rows = []
            for i, (period, tr, va) in enumerate(folds):
                baseline = joblib.load(directory / f"baseline_fold_{i}.joblib")
                historical = joblib.load(directory / f"history_exact_fold_{i}.joblib")
                val = train.iloc[va]
                seen = historical["history"].transform(val).hist_exact_seen.to_numpy().astype(bool)
                pred = np.where(seen, np.load(directory / f"history_exact_fold_{i}.npy"),
                                np.load(directory / f"baseline_fold_{i}.npy"))
                row = dict(period=float(period), train_rows=len(tr),
                    iterations={r["mode"]: r["folds"][i]["iterations"] for r in results[:2]},
                    **metrics(val[state["target"]], pred), history_coverage=float(seen.mean()))
                row["segments"] = {label: metrics(val[state["target"]].to_numpy()[mask], pred[mask])
                    for label, mask in (("seen_key", seen), ("unseen_key", ~seen)) if mask.any()}
                rows.append(row)
                out[va] = pred
                np.save(directory / f"history_gate_fold_{i}.npy", pred)
                write_json(directory / f"history_gate_fold_{i}.json", row)
                joblib.dump({"gate_models": [baseline, historical]}, directory / f"history_gate_fold_{i}.joblib")
            np.save(directory / "history_gate_oof.npy", out)
            scores = [r["rmse"] for r in rows]
            results.append(dict(mode="history_gate", folds=rows, mean_rmse=float(np.mean(scores)),
                worst_rmse=float(max(scores)), std_rmse=float(np.std(scores)), latest_rmse=scores[-1],
                pooled=metrics(train.loc[np.isfinite(out), state["target"]], out[np.isfinite(out)])))
        winner = min(results, key=lambda r: r["mean_rmse"])["mode"]
        baseline = results[0]
        for row in results:
            row["fold_rmse_delta_vs_baseline"] = [r["rmse"] - b["rmse"] for r, b in zip(row["folds"], baseline["folds"])]
        coverage = self._history(train, state, config, "history_exact").transform(test)
        report = dict(suite_id=suite_id, report_path=str(report_path), selected_mode=winner,
            results=results, test_key_coverage=float(coverage.hist_exact_seen.mean()),
            note="Temporal out-of-fold comparison. Early stopping uses an inner past period. "
                 "Recipe and periods were previously explored: this is not an untouched test. "
                 "No legacy best/submission changes. Earliest uncovered OOF rows are NaN.")
        write_json(report_path, report)
        # Compact agent memory, kept separate from old single-holdout scores.
        write_json(self.run / "temporal_summary.json", dict(suite_id=suite_id,
            selected_mode=winner, report_path=str(report_path), results=[{k: r[k] for k in
                ("mode", "mean_rmse", "latest_rmse", "worst_rmse", "fold_rmse_delta_vs_baseline")} for r in results], note=report["note"]))
        return report

    def finalize_temporal(self, suite_id):
        if len(suite_id) != 16 or any(c not in "0123456789abcdef" for c in suite_id):
            raise ValueError("Invalid suite id")
        directory = self.run / "temporal" / suite_id
        identity = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        if identity != self._identity(identity["config"]):
            raise ValueError("Data or temporal implementation changed; run a new suite")
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        config = identity["config"]
        state = config["task"]
        recipe = config["recipe"]
        mode = report["selected_mode"]
        result = next(r for r in report["results"] if r["mode"] == mode)
        # Prespecified latest-fold inner selection, not a new optimization on all labels.
        iterations = result["folds"][-1]["iterations"]
        train, test = self.engine.load()
        if not (test[config["time_column"]] > train[config["time_column"]].max()).all():
            raise ValueError("Final export requires test periods strictly after all training periods")
        if mode == "history_gate":
            bundle = {"gate_models": [self._fit(train, state, recipe, config, member, iterations[member])
                                       for member in ("baseline", "history_exact")]}
        else:
            bundle = self._fit(train, state, recipe, config, mode, iterations)
        pred = self._predict(bundle, test)
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise ValueError("Invalid final predictions")
        submission = test[state["ids"]].copy()
        submission[state["target"]] = pred
        joblib.dump(bundle, directory / "final_model.joblib")
        submission.to_csv(directory / "submission.csv", index=False)
        result = dict(suite_id=suite_id, mode=mode, rows=len(pred), iterations=iterations,
                      submission_path=str(directory / "submission.csv"),
                      note="Separate candidate export; task submission and legacy winner unchanged")
        write_json(directory / "finalized.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    # Stable class names for cross-process joblib reload.
    from temporal_research import TemporalResearch as Worker
    worker = Worker(request["root"], request["files"], request["folder"])
    if request["action"] not in {"run_temporal_suite", "finalize_temporal"}:
        raise ValueError("Unknown temporal tool")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})), ensure_ascii=False))
