"""Trusted tabular experiments. Each command runs in a separate worker process."""
import ast
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Research:
    def __init__(self, root, files, folder):
        self.root, self.folder = Path(root), Path(folder)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = files

    def load(self):
        return pd.read_csv(self.files["train.csv"]), pd.read_csv(self.files["test.csv"])

    def state(self):
        return json.loads((self.root / "state.json").read_text(encoding="utf-8"))

    def inspect(self):
        train, test = self.load()
        return {"train_shape": list(train.shape), "test_shape": list(test.shape),
                "numeric_ranges": {c: {"train_min": str(train[c].min()), "train_max": str(train[c].max()),
                                       "test_min": str(test[c].min()), "test_max": str(test[c].max())}
                                   for c in test if pd.api.types.is_numeric_dtype(test[c])},
                "target_candidates": list(set(train.columns) - set(test.columns)),
                "columns": {c: {"dtype": str(train[c].dtype),
                                 "missing": int(train[c].isna().sum()),
                                 "unique": int(train[c].nunique())} for c in train},
                "dictionary": pd.read_csv(self.files["data_dictionary.csv"]).fillna("").to_dict("records")}

    def configure(self, target, task_type, metric, split="random", split_column=None):
        if (self.root / "state.json").exists():
            raise ValueError("Task and split are already frozen for this run")
        valid = {"regression": {"rmse", "mae"}, "classification": {"f1_macro", "accuracy"}}
        if metric not in valid.get(task_type, set()):
            raise ValueError("Unsupported task/metric combination")
        train, test = self.load()
        sample = pd.read_csv(self.files["sample_submission.csv"])
        ids = [c for c in sample if c in test]
        if target not in train or target in test or list(sample.columns) != ids + [target] or not ids:
            raise ValueError("Expected ID columns followed by one target in submission")
        if train[target].isna().any():
            raise ValueError("Missing training targets")
        features = [c for c in test if c not in ids]
        if split == "group":
            if split_column not in features or train[split_column].isna().any():
                raise ValueError("Group split requires a complete feature column")
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(
                train, groups=train[split_column]))
            features.remove(split_column)
        elif split == "time":
            if split_column not in features or train[split_column].isna().any():
                raise ValueError("Time split requires a complete, chronologically sortable column")
            # Hold out complete final periods, never split the same timestamp across sets.
            counts = train.groupby(split_column).size().sort_index()
            if len(counts) < 2:
                raise ValueError("Time split needs at least two periods")
            threshold = counts.index[-1]
            for value in counts.index[::-1][:-1]:
                threshold = value
                if counts.loc[value:].sum() >= .2 * len(train):
                    break
            mask = train[split_column] >= threshold
            tr, va = np.flatnonzero(~mask), np.flatnonzero(mask)
        elif split == "random":
            tr, va = train_test_split(np.arange(len(train)), test_size=.2, random_state=42,
                                     stratify=train[target] if task_type == "classification" else None)
        else:
            raise ValueError("split must be random, time or group")
        if not len(tr) or not len(va):
            raise ValueError("Empty train/validation split")
        if task_type == "classification" and set(train.iloc[va][target]) - set(train.iloc[tr][target]):
            raise ValueError("Validation contains classes missing from training")
        np.savez(self.root / "split.npz", train=tr, validation=va)
        state = dict(target=target, task_type=task_type, metric=metric, split=split,
                     split_column=split_column, features=features, ids=ids,
                     experiments=[], best=None, train_rows=len(tr), validation_rows=len(va))
        write_json(self.root / "state.json", state)
        return state

    def prepare(self, frame, state, feature_code="", preserve_categorical_missing=False):
        x = frame[state["features"]].copy()
        if feature_code:
            tree = ast.parse(feature_code)
            forbidden = {"open", "exec", "eval", "compile", "getattr", "setattr", "globals", "locals", "__import__"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Feature code uses pre-provided np/pd, not imports")
                if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("__")):
                    raise ValueError("Forbidden feature code name")
                if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr.startswith(("read_", "to_"))):
                    raise ValueError("Private attributes and IO are not allowed in feature code")
            scope = {"np": np, "pd": pd}
            exec(compile(tree, "<features>", "exec"), scope)
            transform = scope.get("transform")
            if not callable(transform):
                raise ValueError("Define transform(X) returning a DataFrame")
            original = x
            x = transform(original.copy())
            if not isinstance(x, pd.DataFrame) or not x.index.equals(original.index) or not x.columns.is_unique or not len(x.columns):
                raise ValueError("Features must preserve row index and have unique, nonempty columns")
            # Reject common batch statistics (mean/rank/frequency) that would be fitted on validation.
            probe = original.iloc[:min(8, len(original))]
            individually = pd.concat([transform(probe.iloc[[i]].copy()) for i in range(len(probe))])
            try:
                pd.testing.assert_frame_equal(x.loc[probe.index], individually, check_dtype=False, check_exact=False)
            except AssertionError as error:
                raise ValueError("Only row-wise transformations are supported; batch-dependent features rejected") from error
        cats = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        for c in cats:
            if not preserve_categorical_missing:
                x[c] = x[c].fillna("__MISSING__").astype(str)
        for c in x.columns.difference(cats):
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        return x, cats

    def model(self, state, params, algorithm):
        allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength",
                   "auto_class_weights", "num_leaves", "min_child_samples"}
        extra = {}
        lgb_options = {"subsample", "subsample_freq", "colsample_bytree", "reg_alpha", "min_split_gain"}
        if algorithm == "lightgbm":
            allowed |= lgb_options
            for key in lgb_options & params.keys():
                value = params[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"{key} must be a finite number")
                if key in {"subsample", "colsample_bytree"} and not 0 < value <= 1:
                    raise ValueError(f"{key} must be in (0,1]")
                if key in {"reg_alpha", "min_split_gain"} and value < 0:
                    raise ValueError(f"{key} must be nonnegative")
                if key == "subsample_freq" and (not isinstance(value, int) or not 0 <= value <= 100):
                    raise ValueError("subsample_freq must be an integer in 0..100")
                extra[key] = value
            # Do not silently accept a row-sampling request that does nothing.
            if params.get("subsample", 1) < 1 and params.get("subsample_freq", 0) == 0:
                raise ValueError("LightGBM subsample < 1 requires explicit subsample_freq >= 1")
            if params.get("subsample_freq", 0) > 0 and params.get("subsample", 1) == 1:
                raise ValueError("LightGBM subsample_freq > 0 requires subsample < 1")
        if set(params) - allowed:
            raise ValueError(f"Unsupported parameters: {set(params) - allowed}")
        p = dict(iterations=200, depth=5, learning_rate=.1)
        p.update(params)
        if not 1 <= int(p["iterations"]) <= 2000 or not 2 <= int(p["depth"]) <= 10:
            raise ValueError("iterations must be 1..2000, depth 2..10")
        if not 0 < float(p["learning_rate"]) <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        classification = state["task_type"] == "classification"
        if not classification and "auto_class_weights" in p:
            raise ValueError("Class weights apply only to classification")
        if algorithm == "catboost":
            if set(params) & {"num_leaves", "min_child_samples"}:
                raise ValueError("num_leaves/min_child_samples apply only to LightGBM")
            cls = CatBoostClassifier if classification else CatBoostRegressor
            return cls(**p, thread_count=8, random_seed=42, verbose=False, allow_writing_files=False)
        if algorithm == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
            if set(params) & {"random_strength", "auto_class_weights"}:
                raise ValueError("These parameters are only supported for CatBoost")
            cls = LGBMClassifier if classification else LGBMRegressor
            leaves = int(p.get("num_leaves", min(31, 2 ** int(p["depth"]))))
            child = int(p.get("min_child_samples", 20))
            if not 2 <= leaves <= min(255, 2 ** int(p["depth"])) or not 1 <= child <= 1000:
                raise ValueError("Invalid num_leaves (2..min(255,2**depth)) or min_child_samples (1..1000)")
            return cls(n_estimators=int(p["iterations"]), max_depth=int(p["depth"]),
                       num_leaves=leaves, min_child_samples=child, learning_rate=p["learning_rate"],
                       reg_lambda=p.get("l2_leaf_reg", 0), n_jobs=8, random_state=42, verbosity=-1, **extra)
        raise ValueError("algorithm must be catboost or lightgbm")

    def encode(self, x, cats, algorithm, levels=None):
        if algorithm != "lightgbm":
            return x, None
        levels = levels if levels is not None else {c: sorted(x[c].unique()) for c in cats}
        for c in cats:
            x[c] = pd.Categorical(x[c], categories=levels[c])
        return x, levels

    def score(self, state, y, pred):
        metric = state["metric"]
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y, pred)))
        if metric == "mae":
            return float(mean_absolute_error(y, pred))
        if metric == "accuracy":
            return float(accuracy_score(y, pred))
        return float(f1_score(y, pred, average="macro", zero_division=0))

    def predict_raw(self, bundle, frame, state):
        if "members" in bundle:
            return sum(w * self.predict_raw(m, frame, state) for w, m in zip(bundle["weights"], bundle["members"]))
        x, cats = self.prepare(frame, state, bundle.get("feature_code", ""))
        x, _ = self.encode(x, cats, bundle["algorithm"], bundle["levels"])
        if state["task_type"] == "classification":
            probabilities = bundle["model"].predict_proba(x)
            order = [list(bundle["model"].classes_).index(c) for c in bundle["classes"]]
            return probabilities[:, order]
        return np.asarray(bundle["model"].predict(x)).reshape(-1)

    def labels(self, raw, bundle, state):
        return np.asarray(bundle["classes"])[raw.argmax(axis=1)] if state["task_type"] == "classification" else raw

    def publish(self, bundle):
        state = self.state()
        _, test = self.load()
        pred = self.labels(self.predict_raw(bundle, test, state), bundle, state)
        if len(pred) != len(test) or pd.isna(pred).any():
            raise ValueError("Invalid prediction shape or missing values")
        if state["task_type"] == "regression" and not np.isfinite(pred.astype(float)).all():
            raise ValueError("Non-finite predictions")
        if state["task_type"] == "classification" and not set(pred).issubset(set(bundle["classes"])):
            raise ValueError("Invalid class predictions")
        output = test[state["ids"]].copy()
        output[state["target"]] = pred
        tmp = self.folder / "submission_candidate.csv"
        output.to_csv(tmp, index=False)
        os.replace(tmp, self.folder / "submission.csv")

    def train(self, hypothesis, params=None, algorithm="catboost", feature_code="", early_stopping_rounds=50):
        started = time.monotonic()
        state = self.state()
        params = params or {}
        if not isinstance(early_stopping_rounds, int) or not 0 <= early_stopping_rounds <= 500:
            raise ValueError("early_stopping_rounds must be 0..500")
        if any(e["params"] == params and e["algorithm"] == algorithm and e.get("feature_code", "") == feature_code
               and e.get("early_stopping_rounds", 50) == early_stopping_rounds for e in state["experiments"]):
            raise ValueError("This exact experiment was already completed; choose another hypothesis")
        train, _ = self.load()
        with np.load(self.root / "split.npz") as indices:
            tr, va = indices["train"], indices["validation"]
        xtr, cats = self.prepare(train.iloc[tr], state, feature_code)
        xva, val_cats = self.prepare(train.iloc[va], state, feature_code)
        if list(xtr.columns) != list(xva.columns) or cats != val_cats:
            raise ValueError("Feature schema must match between train and validation")
        xtr, levels = self.encode(xtr, cats, algorithm)
        xva, _ = self.encode(xva, cats, algorithm, levels)
        model = self.model(state, params, algorithm)
        kwargs = {"cat_features": cats} if algorithm == "catboost" else {}
        if early_stopping_rounds:
            kwargs["eval_set"] = (xva, train.iloc[va][state["target"]])
            if algorithm == "catboost":
                metric_name = {"rmse": "RMSE", "mae": "MAE", "f1_macro": "TotalF1:average=Macro", "accuracy": "Accuracy"}[state["metric"]]
                if state["task_type"] == "classification" and params.get("auto_class_weights"):
                    metric_name += ";use_weights=false" if ":" in metric_name else ":use_weights=false"
                model.set_params(eval_metric=metric_name)
                kwargs.update(early_stopping_rounds=early_stopping_rounds, use_best_model=True)
            else:
                from lightgbm import early_stopping
                kwargs["eval_set"] = [kwargs["eval_set"]]
                classes = np.unique(train.iloc[tr][state["target"]])
                def metric_callback(y_true, predictions):
                    if state["task_type"] == "classification":
                        predicted = (predictions >= .5).astype(int) if predictions.ndim == 1 else predictions.argmax(axis=1)
                        value = self.score(state, classes[y_true.astype(int)], classes[predicted])
                    else:
                        value = self.score(state, y_true, predictions)
                    return state["metric"], value, state["task_type"] == "classification"
                model.set_params(metric="None")
                kwargs.update(eval_metric=metric_callback, callbacks=[early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)])
        model.fit(xtr, train.iloc[tr][state["target"]], **kwargs)
        pred = np.asarray(model.predict(xva)).reshape(-1)
        score = self.score(state, train.iloc[va][state["target"]], pred)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        eid = len(state["experiments"]) + 1
        selected = model.tree_count_ if algorithm == "catboost" else (model.best_iteration_ or model.n_estimators_)
        refit_params = dict(params, iterations=int(selected))
        bundle = dict(model=model, algorithm=algorithm, levels=levels, feature_code=feature_code,
                      classes=list(model.classes_) if state["task_type"] == "classification" else None,
                      refit_params=refit_params)
        raw = model.predict_proba(xva) if state["task_type"] == "classification" else pred
        np.save(self.root / f"validation_{eid}.npy", raw)
        path = self.root / f"model_{eid}.joblib"
        joblib.dump(bundle, path)
        record = dict(id=eid, hypothesis=hypothesis, params=params, algorithm=algorithm,
                      score=score, metric=state["metric"], seconds=round(time.monotonic()-started, 2),
                      model_path=str(path))
        record.update(feature_code=feature_code, early_stopping_rounds=early_stopping_rounds,
                      selected_iterations=int(selected), refit_params=refit_params)
        if state["task_type"] == "classification":
            from sklearn.metrics import classification_report
            record["class_metrics"] = classification_report(train.iloc[va][state["target"]], pred,
                                                            output_dict=True, zero_division=0)
        best = state["best"]
        better = best is None or (score < best["score"] if state["metric"] in {"rmse", "mae"}
                                 else score > best["score"])
        if better:
            # Write a usable prediction before committing the new winner.
            self.publish(bundle)
            state["best"] = record
        state["experiments"].append(record)
        write_json(self.root / "state.json", state)
        return dict(experiment=record, improved=better, best=state["best"])

    def blend(self, experiment_ids, weights=None):
        state = self.state()
        if len(experiment_ids) not in (2, 3) or len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Choose 2 or 3 distinct experiments")
        records = [next(e for e in state["experiments"] if e["id"] == i) for i in experiment_ids]
        if any(e["algorithm"] == "ensemble" for e in records):
            raise ValueError("Nested ensembles are not supported")
        weights = np.asarray(weights if weights is not None else [1] * len(records), dtype=float)
        if len(weights) != len(records) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Provide positive finite weights")
        weights /= weights.sum()
        members = [joblib.load(e["model_path"]) for e in records]
        if any(m["classes"] != members[0]["classes"] for m in members):
            raise ValueError("Class orders differ")
        bundle = dict(members=members, weights=weights.tolist(), classes=members[0]["classes"])
        raw = sum(w * np.load(self.root / f"validation_{e['id']}.npy") for w, e in zip(weights, records))
        train, _ = self.load()
        with np.load(self.root / "split.npz") as indices:
            va = indices["validation"]
        score = self.score(state, train.iloc[va][state["target"]], self.labels(raw, bundle, state))
        eid = len(state["experiments"]) + 1
        path = self.root / f"model_{eid}.joblib"
        joblib.dump(bundle, path)
        record = dict(id=eid, algorithm="ensemble", params={"ids": experiment_ids, "weights": weights.tolist()},
                      score=score, metric=state["metric"], model_path=str(path))
        best = state["best"]
        improved = score < best["score"] if state["task_type"] == "regression" else score > best["score"]
        if improved:
            self.publish(bundle)
            state["best"] = record
        state["experiments"].append(record)
        write_json(self.root / "state.json", state)
        return {"experiment": record, "improved": improved, "best": state["best"]}

    def refit(self, bundle, train, state):
        if "members" in bundle:
            return dict(bundle, members=[self.refit(m, train, state) for m in bundle["members"]])
        x, cats = self.prepare(train, state, bundle.get("feature_code", ""))
        x, levels = self.encode(x, cats, bundle["algorithm"])
        model = self.model(state, bundle["refit_params"], bundle["algorithm"])
        model.fit(x, train[state["target"]], **({"cat_features": cats} if bundle["algorithm"] == "catboost" else {}))
        return dict(bundle, model=model, levels=levels)

    def finalize(self):
        state = self.state()
        best = state["best"]
        if best is None:
            raise ValueError("No successful experiments")
        # Always restore the trusted winner before attempting a full refit.
        self.publish(joblib.load(best["model_path"]))
        train, _ = self.load()
        bundle = self.refit(joblib.load(best["model_path"]), train, state)
        joblib.dump(bundle, self.root / "final_model.joblib")
        self.publish(bundle)
        state["finalized"] = True
        write_json(self.root / "state.json", state)
        return {"finalized": True, "selected_experiment": best["id"],
                "validation_score": best["score"], "note": "Score is from holdout, not full refit/test"}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker = Research(request["root"], request["files"], request["folder"])
    result = getattr(worker, request["action"])(**request.get("arguments", {}))
    print(json.dumps(result, ensure_ascii=False, default=str))
