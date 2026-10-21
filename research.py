"""Trusted tabular experiments. Each command runs in a separate worker process."""
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

    def prepare(self, frame, state):
        x = frame[state["features"]].copy()
        cats = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        for c in cats:
            x[c] = x[c].fillna("__MISSING__").astype(str)
        for c in x.columns.difference(cats):
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        return x, cats

    def model(self, state, params, algorithm):
        allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength",
                   "auto_class_weights"}
        if set(params) - allowed:
            raise ValueError(f"Unsupported parameters: {set(params) - allowed}")
        p = dict(iterations=200, depth=5, learning_rate=.1)
        p.update(params)
        if not 10 <= int(p["iterations"]) <= 2000 or not 2 <= int(p["depth"]) <= 10:
            raise ValueError("iterations must be 10..2000, depth 2..10")
        if not 0 < float(p["learning_rate"]) <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        classification = state["task_type"] == "classification"
        if not classification and "auto_class_weights" in p:
            raise ValueError("Class weights apply only to classification")
        if algorithm == "catboost":
            cls = CatBoostClassifier if classification else CatBoostRegressor
            return cls(**p, thread_count=8, random_seed=42, verbose=False, allow_writing_files=False)
        if algorithm == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
            if set(params) & {"random_strength", "auto_class_weights"}:
                raise ValueError("These parameters are only supported for CatBoost")
            cls = LGBMClassifier if classification else LGBMRegressor
            return cls(n_estimators=int(p["iterations"]), max_depth=int(p["depth"]),
                       num_leaves=min(31, 2 ** int(p["depth"])), learning_rate=p["learning_rate"],
                       reg_lambda=p.get("l2_leaf_reg", 0), n_jobs=8, random_state=42, verbosity=-1)
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

    def publish(self, bundle):
        state = self.state()
        _, test = self.load()
        x, cats = self.prepare(test, state)
        x, _ = self.encode(x, cats, bundle["algorithm"], bundle["levels"])
        pred = np.asarray(bundle["model"].predict(x)).reshape(-1)
        if len(pred) != len(test) or pd.isna(pred).any():
            raise ValueError("Invalid prediction shape or missing values")
        if state["task_type"] == "regression" and not np.isfinite(pred.astype(float)).all():
            raise ValueError("Non-finite predictions")
        if state["task_type"] == "classification" and not set(pred).issubset(set(bundle["model"].classes_)):
            raise ValueError("Invalid class predictions")
        output = test[state["ids"]].copy()
        output[state["target"]] = pred
        tmp = self.folder / "submission_candidate.csv"
        output.to_csv(tmp, index=False)
        os.replace(tmp, self.folder / "submission.csv")

    def train(self, hypothesis, params=None, algorithm="catboost"):
        started = time.monotonic()
        state = self.state()
        params = params or {}
        if any(e["params"] == params and e["algorithm"] == algorithm for e in state["experiments"]):
            raise ValueError("This exact experiment was already completed; choose another hypothesis")
        train, _ = self.load()
        indices = np.load(self.root / "split.npz")
        x, cats = self.prepare(train, state)
        tr, va = indices["train"], indices["validation"]
        xtr, levels = self.encode(x.iloc[tr].copy(), cats, algorithm)
        xva, _ = self.encode(x.iloc[va].copy(), cats, algorithm, levels)
        model = self.model(state, params, algorithm)
        kwargs = {"cat_features": cats} if algorithm == "catboost" else {}
        model.fit(xtr, train.iloc[tr][state["target"]], **kwargs)
        pred = np.asarray(model.predict(xva)).reshape(-1)
        score = self.score(state, train.iloc[va][state["target"]], pred)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        eid = len(state["experiments"]) + 1
        bundle = dict(model=model, algorithm=algorithm, levels=levels)
        path = self.root / f"model_{eid}.joblib"
        joblib.dump(bundle, path)
        record = dict(id=eid, hypothesis=hypothesis, params=params, algorithm=algorithm,
                      score=score, metric=state["metric"], seconds=round(time.monotonic()-started, 2),
                      model_path=str(path))
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

    def finalize(self):
        state = self.state()
        best = state["best"]
        if best is None:
            raise ValueError("No successful experiments")
        # Always restore the trusted winner before attempting a full refit.
        self.publish(joblib.load(best["model_path"]))
        train, _ = self.load()
        x, cats = self.prepare(train, state)
        x, levels = self.encode(x, cats, best["algorithm"])
        model = self.model(state, best["params"], best["algorithm"])
        kwargs = {"cat_features": cats} if best["algorithm"] == "catboost" else {}
        model.fit(x, train[state["target"]], **kwargs)
        bundle = dict(model=model, algorithm=best["algorithm"], levels=levels)
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
