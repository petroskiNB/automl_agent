"""Agent-requested diagnostics. Never publish or replace a winning model."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research import write_json
from xgb_backend import ExtendedResearch as Research


class Diagnostics:
    def __init__(self, root, files, folder):
        self.engine = Research(root, files, folder)
        self.run = Path(root).parent

    def _validation(self, experiment_id):
        state = self.engine.state()
        record = self.read_experiment(experiment_id)["experiment"]
        train, _ = self.engine.load()
        with np.load(self.engine.root / "split.npz") as split:
            frame = train.iloc[split["validation"]].copy()
        bundle = joblib.load(record["model_path"])
        path = self.engine.root / f"validation_{experiment_id}.npy"
        # Ensembles have no cached array. Use the saved HOLDOUT models, never final_model.
        raw = np.load(path) if path.exists() else self.engine.predict_raw(bundle, frame, state)
        if len(raw) != len(frame) or not np.isfinite(raw).all():
            raise ValueError("Invalid validation predictions")
        pred = self.engine.labels(raw, bundle, state)
        return state, frame, raw, pred

    def _save_analysis(self, kind, result):
        directory = self.run / "error_analysis"
        directory.mkdir(exist_ok=True)
        path = directory / f"{kind}_{time.time_ns()}.json"
        result["note"] = "Frozen holdout diagnostic; repeated selection can overfit it. No training or submission changes."
        write_json(path, result)
        index = self.run / "error_analysis.json"
        history = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
        summary = {k: v for k, v in result.items() if k not in {"groups", "confidence_bins", "per_class"}}
        if "per_class" in result:
            summary["weakest_classes"] = sorted(result["per_class"], key=lambda r: r["f1"])[:4]
        if result.get("groups"):
            summary["highest_error_groups"] = {column: sorted(rows,
                key=lambda r: r["error_rate"] if r["error_rate"] is not None else r["mae"], reverse=True)[:3]
                for column, rows in result["groups"].items()}
        summary["report_path"] = str(path)
        history.append(summary)
        write_json(index, history)
        return dict(result, report_path=str(path))

    def analyze_errors(self, experiment_id, group_columns=None, min_group_size=30):
        from sklearn.metrics import precision_recall_fscore_support
        state, frame, raw, pred = self._validation(experiment_id)
        y = frame[state["target"]].to_numpy()
        columns = group_columns or []
        if len(columns) > 3 or len(set(columns)) != len(columns) or any(c not in state["features"] for c in columns):
            raise ValueError("Choose up to three distinct feature columns")
        if min_group_size < 1:
            raise ValueError("min_group_size must be positive")
        classification = state["task_type"] == "classification"
        result = {"experiment_id": experiment_id, "rows": len(y), "metric": state["metric"],
                  "score": float(self.engine.score(state, y, pred))}
        if classification:
            labels = np.unique(np.concatenate([y, pred]))
            p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
            result["per_class"] = [dict(label=str(c), precision=float(a), recall=float(b), f1=float(d), support=int(e))
                                   for c, a, b, d, e in zip(labels, p, r, f, n)]
            wrong = pd.DataFrame({"actual": y[y != pred].astype(str), "predicted": pred[y != pred].astype(str)})
            counts = wrong.value_counts().head(10)
            result["confusions"] = [dict(actual=a, predicted=b, count=int(n)) for (a, b), n in counts.items()]
            confidence = raw.max(axis=1)
            result["confidence_bins"] = []
            for low, high in zip([0., .5, .7, .9], [.5, .7, .9, 1.000001]):
                mask = (confidence >= low) & (confidence < high)
                if mask.any():
                    result["confidence_bins"].append(dict(lower=low, upper=min(high, 1.), count=int(mask.sum()),
                        accuracy=float((y[mask] == pred[mask]).mean()), mean_confidence=float(confidence[mask].mean())))
        else:
            residual = pred - y
            result.update(mae=float(np.abs(residual).mean()), rmse=float(np.sqrt(np.mean(residual ** 2))),
                          mean_prediction_minus_target=float(residual.mean()),
                          absolute_error_quantiles=dict(zip(["50%", "90%", "99%"], np.quantile(np.abs(residual), [.5, .9, .99]).tolist())))
        result["groups"] = {}
        train, _ = self.engine.load()
        with np.load(self.engine.root / "split.npz") as split:
            fit = train.iloc[split["train"]]
        for column in columns:
            if pd.api.types.is_numeric_dtype(fit[column]):
                values = fit[column].replace([np.inf, -np.inf], np.nan).dropna()
                edges = np.unique(values.quantile([.2, .4, .6, .8]).to_numpy())
                groups = pd.cut(frame[column], [-np.inf, *edges, np.inf]).astype("string")
            else:
                groups = frame[column].astype("string")
            groups = groups.fillna("<MISSING>").to_numpy()
            rows = []
            for group, count in pd.Series(groups).value_counts().head(15).items():
                if count < min_group_size:
                    continue
                mask = groups == group
                rows.append(dict(group=str(group), count=int(count),
                    error_rate=float((pred[mask] != y[mask]).mean()) if classification else None,
                    mae=float(np.abs(pred[mask] - y[mask]).mean()) if not classification else None))
            result["groups"][column] = rows
        return self._save_analysis("errors", result)

    def compare_errors(self, experiment_ids):
        if len(experiment_ids) != 2 or len(set(experiment_ids)) != 2:
            raise ValueError("Choose two different experiments")
        state, frame, _, a = self._validation(experiment_ids[0])
        _, other, _, b = self._validation(experiment_ids[1])
        if not frame.index.equals(other.index):
            raise ValueError("Validation rows differ")
        y = frame[state["target"]].to_numpy()
        result = {"experiment_ids": experiment_ids, "rows": len(y), "metric": state["metric"],
                  "scores": [float(self.engine.score(state, y, p)) for p in (a, b)]}
        if state["task_type"] == "classification":
            ca, cb = a == y, b == y
            result.update(both_correct=int((ca & cb).sum()), only_first_correct=int((ca & ~cb).sum()),
                          only_second_correct=int((~ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
                          disagreements=int((a != b).sum()))
        else:
            ea, eb = a - y, b - y
            correlation = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else None
            result.update(residual_correlation=correlation,
                          first_lower_absolute_error=int((np.abs(ea) < np.abs(eb)).sum()),
                          second_lower_absolute_error=int((np.abs(eb) < np.abs(ea)).sum()))
        return self._save_analysis("comparison", result)

    def read_experiment(self, experiment_id):
        state = self.engine.state()
        matches = [e for e in state["experiments"] if e["id"] == experiment_id]
        if not matches:
            raise ValueError("Unknown experiment_id")
        return {"experiment": matches[0], "validation": {k: state.get(k) for k in
                ("metric", "split", "split_column", "train_rows", "validation_rows")}}

    def inspect_categories(self, columns=None):
        train, test = self.engine.load()
        columns = columns if columns is not None else [c for c in test if not pd.api.types.is_numeric_dtype(train[c])][:10]
        if not 1 <= len(columns) <= 10 or len(set(columns)) != len(columns):
            raise ValueError("Choose 1..10 distinct feature columns")
        result = {}
        for c in columns:
            if c not in test or c not in train:
                raise ValueError("Choose feature columns present in train and test")
            a = set(train[c].dropna().astype(str))
            b = set(test[c].dropna().astype(str))
            groups = {}
            for value in sorted(a | b):
                groups.setdefault(value.encode("utf-8")[:7].hex(), []).append(value)
            collisions = [values for values in groups.values() if len(values) > 1]
            result[c] = {"train_unique": len(a), "test_unique": len(b), "unseen_test_categories": len(b-a),
                         "unseen_test_rows": int((test[c].notna() & ~test[c].astype(str).isin(a)).sum()),
                         "train_missing": int(train[c].isna().sum()), "test_missing": int(test[c].isna().sum()),
                         "train_examples": sorted(a)[:12], "unseen_examples": sorted(b-a)[:12],
                         "seven_byte_collision_groups": len(collisions),
                         "collision_examples": [values[:6] for values in collisions[:5]]}
        return {"categories": result, "note": "Full-string native LightGBM encoding is already supported; "
                "categories are fitted on training rows only. No labels from test used."}

    def validate_experiment(self, experiment_id, split_column, validation_value):
        started = time.monotonic()
        state = self.engine.state()
        record = self.read_experiment(experiment_id)["experiment"]
        train, _ = self.engine.load()
        if split_column not in state["features"] or not pd.api.types.is_numeric_dtype(train[split_column]):
            raise ValueError("Use a numeric time feature present in the saved task")
        if train[split_column].isna().any() or not np.isfinite(validation_value):
            raise ValueError("Time values must be complete and validation_value finite")
        tr = train.loc[train[split_column] < validation_value]
        va = train.loc[train[split_column] == validation_value]
        if tr.empty or va.empty:
            raise ValueError("Empty temporal train/validation fold")
        if state["task_type"] == "classification" and set(va[state["target"]]) - set(tr[state["target"]]):
            raise ValueError("Validation has classes absent from training")
        bundle = joblib.load(record["model_path"])
        # Refit uses saved iteration counts and weights: never early-stop on this fold.
        fitted = self.engine.refit(bundle, tr, state)
        raw = self.engine.predict_raw(fitted, va, state)
        pred = self.engine.labels(raw, fitted, state)
        score = self.engine.score(state, va[state["target"]], pred)
        if not np.isfinite(score):
            raise ValueError("Non-finite secondary score")
        result = {"experiment_id": experiment_id, "split_column": split_column,
                  "validation_value": validation_value, "train_rows": len(tr), "validation_rows": len(va),
                  "metric": state["metric"], "score": score, "seconds": round(time.monotonic()-started, 2),
                  "note": "Secondary diagnostic, not an independent test. Fixed saved iterations and ensemble weights. "
                          "No change to primary score, saved models or submission.csv."}
        path = self.run / "diagnostics.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        history.append(result)
        write_json(path, history)
        return result


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker = Diagnostics(request["root"], request["files"], request["folder"])
    if request["action"] not in {"read_experiment", "inspect_categories", "validate_experiment", "analyze_errors", "compare_errors"}:
        raise ValueError("Unknown diagnostic")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})), ensure_ascii=False))
