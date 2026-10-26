"""Optional XGBoost backend; existing CatBoost/LightGBM recipes stay unchanged."""
import json
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from research import Research, write_json


class XGBModel:
    def __init__(self, task_type, metric, params):
        self.task_type, self.metric = task_type, metric
        allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "min_child_weight",
                   "subsample", "colsample_bytree", "reg_alpha", "gamma", "auto_class_weights"}
        if set(params) - allowed:
            raise ValueError(f"Unsupported XGBoost parameters: {set(params) - allowed}")
        self.params = dict(iterations=200, depth=5, learning_rate=.1, l2_leaf_reg=1,
                           min_child_weight=1, subsample=1, colsample_bytree=1, reg_alpha=0, gamma=0)
        self.params.update(params)
        p = self.params
        if not 1 <= int(p["iterations"]) <= 2000 or not 2 <= int(p["depth"]) <= 10:
            raise ValueError("iterations must be 1..2000 and depth 2..10")
        for key in ("learning_rate", "subsample", "colsample_bytree"):
            if not 0 < float(p[key]) <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        for key in ("l2_leaf_reg", "min_child_weight", "reg_alpha", "gamma"):
            if not np.isfinite(float(p[key])) or float(p[key]) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if p.get("auto_class_weights") not in (None, "Balanced", "SqrtBalanced"):
            raise ValueError("auto_class_weights must be Balanced or SqrtBalanced")
        if task_type != "classification" and p.get("auto_class_weights"):
            raise ValueError("Class weights apply only to classification")

    def fit(self, x, y, eval_set=None, early_stopping_rounds=0):
        import xgboost as xgb
        p = self.params
        classification = self.task_type == "classification"
        y = np.asarray(y)
        weights = None
        if classification:
            self.classes_, encoded = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("Need at least two training classes")
            if p.get("auto_class_weights"):
                counts = np.bincount(encoded)
                per_class = counts.max() / counts.astype(float)
                if p["auto_class_weights"] == "SqrtBalanced":
                    per_class = np.sqrt(per_class)
                weights = per_class[encoded]
                weights /= weights.mean()
            y = encoded
        train = xgb.DMatrix(x, label=y, weight=weights, enable_categorical=True, nthread=8)
        settings = dict(tree_method="hist", device="cpu", nthread=8, seed=42,
                        max_depth=int(p["depth"]), eta=float(p["learning_rate"]),
                        reg_lambda=float(p["l2_leaf_reg"]), min_child_weight=float(p["min_child_weight"]),
                        subsample=float(p["subsample"]), colsample_bytree=float(p["colsample_bytree"]),
                        reg_alpha=float(p["reg_alpha"]), gamma=float(p["gamma"]), disable_default_eval_metric=1)
        if classification:
            settings.update(objective="multi:softprob", num_class=len(self.classes_))
        else:
            settings["objective"] = "reg:absoluteerror" if self.metric == "mae" else "reg:squarederror"
        evals = []
        if eval_set is not None:
            xv, yv = eval_set
            yv = np.asarray(yv)
            if classification:
                if set(yv) - set(self.classes_):
                    raise ValueError("Validation contains unknown classes")
                yv = np.searchsorted(self.classes_, yv)
            # Validation has no class weights: optimize the actual competition metric.
            evals = [(xgb.DMatrix(xv, label=yv, enable_categorical=True, nthread=8), "validation")]

        def metric(pred, data):
            truth = data.get_label()
            if classification:
                labels = pred.argmax(axis=1)
                score = f1_score(truth, labels, average="macro", zero_division=0) if self.metric == "f1_macro" else accuracy_score(truth, labels)
            else:
                score = mean_absolute_error(truth, pred) if self.metric == "mae" else np.sqrt(mean_squared_error(truth, pred))
            return self.metric, float(score)

        self.evaluation_history = {}
        self.booster = xgb.train(settings, train, num_boost_round=int(p["iterations"]), evals=evals,
                                 custom_metric=metric if evals else None, maximize=classification,
                                 early_stopping_rounds=early_stopping_rounds if evals and early_stopping_rounds else None,
                                 verbose_eval=False, evals_result=self.evaluation_history)
        self.selected_iterations = (self.booster.best_iteration + 1 if evals and early_stopping_rounds
                                    else self.booster.num_boosted_rounds())
        if self.selected_iterations < self.booster.num_boosted_rounds():
            self.booster = self.booster[:self.selected_iterations]
        return self

    def predict_proba(self, x):
        if self.task_type != "classification":
            raise ValueError("Probabilities require classification")
        return self._raw(x)

    def _raw(self, x):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(x, enable_categorical=True, nthread=8))

    def predict(self, x):
        raw = self._raw(x)
        return self.classes_[raw.argmax(axis=1)] if self.task_type == "classification" else raw


class ExtendedResearch(Research):
    def model(self, state, params, algorithm):
        if algorithm == "xgboost":
            return XGBModel(state["task_type"], state["metric"], params)
        return super().model(state, params, algorithm)

    def encode(self, x, cats, algorithm, levels=None):
        # Fit categorical dictionaries on train and reuse on validation/test.
        return super().encode(x, cats, "lightgbm" if algorithm == "xgboost" else algorithm, levels)

    def train(self, hypothesis, params=None, algorithm="catboost", feature_code="", early_stopping_rounds=50):
        if algorithm != "xgboost":
            return super().train(hypothesis, params, algorithm, feature_code, early_stopping_rounds)
        started = time.monotonic()
        state, params = self.state(), params or {}
        if not isinstance(early_stopping_rounds, int) or not 0 <= early_stopping_rounds <= 500:
            raise ValueError("early_stopping_rounds must be 0..500")
        if any(e["algorithm"] == algorithm and e["params"] == params and e.get("feature_code", "") == feature_code
               and e.get("early_stopping_rounds", 50) == early_stopping_rounds for e in state["experiments"]):
            raise ValueError("This exact experiment was already completed")
        train, _ = self.load()
        with np.load(self.root / "split.npz") as split:
            tr, va = split["train"], split["validation"]
        xt, cats = self.prepare(train.iloc[tr], state, feature_code)
        xv, vcats = self.prepare(train.iloc[va], state, feature_code)
        if list(xt.columns) != list(xv.columns) or cats != vcats:
            raise ValueError("Feature schemas differ between train and validation")
        xt, levels = self.encode(xt, cats, algorithm)
        xv, _ = self.encode(xv, cats, algorithm, levels)
        model = self.model(state, params, algorithm)
        model.fit(xt, train.iloc[tr][state["target"]],
                  eval_set=(xv, train.iloc[va][state["target"]]) if early_stopping_rounds else None,
                  early_stopping_rounds=early_stopping_rounds)
        pred = model.predict(xv)
        score = self.score(state, train.iloc[va][state["target"]], pred)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        eid = len(state["experiments"]) + 1
        refit_params = dict(params, iterations=int(model.selected_iterations))
        bundle = dict(model=model, algorithm=algorithm, levels=levels, feature_code=feature_code,
                      classes=model.classes_.tolist() if state["task_type"] == "classification" else None,
                      refit_params=refit_params)
        raw = model.predict_proba(xv) if state["task_type"] == "classification" else pred
        np.save(self.root / f"validation_{eid}.npy", raw)
        path = self.root / f"model_{eid}.joblib"
        joblib.dump(bundle, path)
        record = dict(id=eid, hypothesis=hypothesis, params=params, algorithm=algorithm, score=score,
                      metric=state["metric"], seconds=round(time.monotonic()-started, 2), model_path=str(path),
                      feature_code=feature_code, early_stopping_rounds=early_stopping_rounds,
                      selected_iterations=int(model.selected_iterations), refit_params=refit_params)
        if state["task_type"] == "classification":
            from sklearn.metrics import classification_report
            record["class_metrics"] = classification_report(train.iloc[va][state["target"]], pred,
                                                            output_dict=True, zero_division=0)
        best = state["best"]
        improved = best is None or (score < best["score"] if state["task_type"] == "regression" else score > best["score"])
        if improved:
            self.publish(bundle)
            state["best"] = record
        state["experiments"].append(record)
        write_json(self.root / "state.json", state)
        return {"experiment": record, "improved": improved, "best": state["best"]}


if __name__ == "__main__":
    # Stable module name is required for joblib across worker subprocesses.
    from xgb_backend import ExtendedResearch as Worker
    request = json.load(open(sys.argv[1], encoding="utf-8"))
    worker = Worker(request["root"], request["files"], request["folder"])
    if request["action"] not in {"inspect", "configure", "train", "blend", "finalize"}:
        raise ValueError("Unknown action")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})), ensure_ascii=False, default=str))
