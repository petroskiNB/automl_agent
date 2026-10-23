"""Experimental trusted workers. Legacy backend/recipes are not modified."""
import json
import os
import sys
import time
import threading
from pathlib import Path

from torch_resources import configure
THREADS = configure()

import joblib
import numpy as np
import psutil
from sklearn.metrics import classification_report
from threadpoolctl import threadpool_limits

from research import Research, write_json
from torch_model import TorchModel
from xgb_backend import ExtendedResearch


def deadline():
    value = os.getenv("AUTOML_WORKER_DEADLINE")
    # Wall-clock deadline survives subprocess boundaries.
    return time.monotonic() + max(0, float(value)-time.time()) if value else None


class MemoryMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


class CPUXGBoost:
    """XGBoost with explicit per-process allocation and actual metric stopping."""
    def __init__(self, state, params):
        from xgb_backend import XGBModel
        self.base = XGBModel(state["task_type"], state["metric"], params)
        self.task_type, self.metric = state["task_type"], state["metric"]

    def fit(self, x, y, eval_set=None, early_stopping_rounds=0):
        import xgboost as xgb
        p = self.base.params
        classification = self.task_type == "classification"
        weights = None
        if classification:
            self.classes_, y = np.unique(y, return_inverse=True)
            if p.get("auto_class_weights"):
                counts = np.bincount(y)
                w = counts.max()/counts.astype(float)
                if p["auto_class_weights"] == "SqrtBalanced":
                    w = np.sqrt(w)
                weights = w[y]/w[y].mean()
        matrix = xgb.DMatrix(x, label=y, weight=weights, enable_categorical=True, nthread=THREADS)
        settings = dict(tree_method="hist", nthread=THREADS, seed=self.seed,
            max_depth=int(p["depth"]), eta=p["learning_rate"], reg_lambda=p["l2_leaf_reg"],
            min_child_weight=p["min_child_weight"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], reg_alpha=p["reg_alpha"], gamma=p["gamma"],
            disable_default_eval_metric=1,
            objective="multi:softprob" if classification else ("reg:absoluteerror" if self.metric == "mae" else "reg:squarederror"))
        if classification:
            settings["num_class"] = len(self.classes_)
        evals = []
        if eval_set is not None:
            xv, yv = eval_set
            if classification:
                if set(yv)-set(self.classes_):
                    raise ValueError("Validation contains unseen target class")
                yv = np.searchsorted(self.classes_, yv)
            evals = [(xgb.DMatrix(xv, label=yv, enable_categorical=True, nthread=THREADS), "validation")]
        def metric(raw, data):
            truth = data.get_label()
            prediction = raw.argmax(1) if classification else raw
            return self.metric, Research.score(self, {"metric": self.metric}, truth, prediction)
        self.evaluation_history = {}
        self.booster = xgb.train(settings, matrix, num_boost_round=p["iterations"], evals=evals,
            custom_metric=metric if evals else None, maximize=classification,
            early_stopping_rounds=early_stopping_rounds if evals and early_stopping_rounds else None,
            verbose_eval=False, evals_result=self.evaluation_history)
        self.selected_iterations = self.booster.best_iteration+1 if evals and early_stopping_rounds else p["iterations"]
        self.booster = self.booster[:self.selected_iterations]
        return self

    def _raw(self, x):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(x, enable_categorical=True, nthread=THREADS))

    def predict(self, x):
        raw = self._raw(x)
        return self.classes_[raw.argmax(1)] if self.task_type == "classification" else raw

    def predict_proba(self, x):
        return self._raw(x)


class TorchResearch(ExtendedResearch):
    def prepare(self, frame, state, feature_code=""):
        return super().prepare(frame, state, feature_code, preserve_categorical_missing=True)

    def model(self, state, params, algorithm):
        params = dict(params)
        if algorithm in ("torch_mlp", "torch_resnet"):
            return TorchModel(state["task_type"], state["metric"], params, residual=algorithm == "torch_resnet")
        seed = params.pop("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be a nonnegative integer < 2**31")
        if algorithm == "xgboost":
            model = CPUXGBoost(state, params)
            model.seed = seed
            return model
        model = super().model(state, params, algorithm)
        if algorithm == "catboost":
            model.set_params(thread_count=THREADS, random_seed=seed)
        else:
            model.set_params(n_jobs=THREADS, random_state=seed)
        return model

    def encode(self, x, cats, algorithm, levels=None):
        if algorithm.startswith("torch_"):
            return x, None
        for c in cats:
            x[c] = x[c].fillna("__MISSING__").astype(str)
        return super().encode(x, cats, algorithm, levels)

    def _fit(self, model, algorithm, state, xt, yt, cats, xv=None, yv=None, patience=0):
        with threadpool_limits(limits=THREADS):
            if algorithm.startswith("torch_"):
                model.fit(xt, yt, eval_set=(xv, yv) if xv is not None else None,
                          early_stopping_rounds=patience, deadline=deadline())
            elif algorithm == "xgboost":
                model.fit(xt, yt, eval_set=(xv, yv) if xv is not None else None,
                          early_stopping_rounds=patience)
            else:
                kwargs = {"cat_features": cats} if algorithm == "catboost" else {}
                if xv is not None and patience:
                    if algorithm == "catboost":
                        metric = {"rmse": "RMSE", "mae": "MAE", "accuracy": "Accuracy",
                                  "f1_macro": "TotalF1:average=Macro"}[state["metric"]]
                        if state["task_type"] == "classification" and model.get_params().get("auto_class_weights"):
                            metric += ";use_weights=false" if ":" in metric else ":use_weights=false"
                        model.set_params(eval_metric=metric)
                        kwargs.update(eval_set=(xv, yv), early_stopping_rounds=patience, use_best_model=True)
                    else:
                        from lightgbm import early_stopping
                        classes = np.unique(yt)
                        def evaluate(truth, raw):
                            if state["task_type"] == "classification":
                                pred = (raw >= .5).astype(int) if raw.ndim == 1 else raw.argmax(1)
                                truth, pred = classes[truth.astype(int)], classes[pred]
                            else:
                                pred = raw
                            return state["metric"], self.score(state, truth, pred), state["task_type"] == "classification"
                        model.set_params(metric="None")
                        kwargs.update(eval_set=[(xv, yv)], eval_metric=evaluate,
                                      callbacks=[early_stopping(patience, first_metric_only=True, verbose=False)])
                model.fit(xt, yt, **kwargs)
        return model

    def train(self, hypothesis, params=None, algorithm="catboost", feature_code="", early_stopping_rounds=50):
        params = params or {}
        if type(early_stopping_rounds) is not int or not 0 <= early_stopping_rounds <= 500:
            raise ValueError("early_stopping_rounds must be in 0..500")
        state = self.state()
        if any(e["algorithm"] == algorithm and e["params"] == params and e.get("feature_code", "") == feature_code
               and e.get("early_stopping_rounds") == early_stopping_rounds for e in state["experiments"]):
            raise ValueError("This exact experiment was already completed")
        frame, _ = self.load()
        with np.load(self.root / "split.npz") as split:
            tr, va = split["train"], split["validation"]
        started = time.perf_counter()
        with MemoryMonitor() as memory:
            xt, cats = self.prepare(frame.iloc[tr], state, feature_code)
            xv, _ = self.prepare(frame.iloc[va], state, feature_code)
            xt, levels = self.encode(xt, cats, algorithm)
            xv, _ = self.encode(xv, cats, algorithm, levels)
            model = self.model(state, params, algorithm)
            fit_start = time.perf_counter()
            self._fit(model, algorithm, state, xt, frame.iloc[tr][state["target"]], cats,
                      xv, frame.iloc[va][state["target"]], early_stopping_rounds)
            fit_seconds = time.perf_counter()-fit_start
            prediction_start = time.perf_counter()
            classification = state["task_type"] == "classification"
            raw = model.predict_proba(xv) if classification else np.asarray(model.predict(xv)).reshape(-1)
            predict_seconds = time.perf_counter()-prediction_start
            classes = list(model.classes_) if classification else None
            pred = np.asarray(classes)[raw.argmax(1)] if classification else raw
            score = self.score(state, frame.iloc[va][state["target"]], pred)
            if not np.isfinite(score) or not np.isfinite(raw).all():
                raise ValueError("Non-finite result")
        selected = (model.selected_iterations if algorithm in ("torch_mlp", "torch_resnet", "xgboost") else
                    model.tree_count_ if algorithm == "catboost" else (model.best_iteration_ or model.n_estimators_))
        eid = len(state["experiments"])+1
        path = self.root / f"model_{eid}.joblib"
        bundle = dict(model=model, algorithm=algorithm, levels=levels, classes=classes,
                      feature_code=feature_code, refit_params=dict(params, iterations=int(selected)))
        np.save(self.root / f"validation_{eid}.npy", raw)
        joblib.dump(bundle, path)
        record = dict(id=eid, hypothesis=hypothesis, algorithm=algorithm, params=params, score=score,
            metric=state["metric"], feature_code=feature_code, early_stopping_rounds=early_stopping_rounds,
            selected_iterations=int(selected), refit_params=bundle["refit_params"], model_path=str(path),
            seconds=time.perf_counter()-started, fit_seconds=fit_seconds, predict_seconds=predict_seconds,
            peak_rss_mb=memory.peak/1024**2, threads=THREADS, seed=params.get("seed", 42),
            timed_out=getattr(model, "timed_out", False), split=state["split"], split_column=state["split_column"])
        if classification:
            record["class_metrics"] = classification_report(frame.iloc[va][state["target"]], pred,
                                                            output_dict=True, zero_division=0)
        best = state["best"]
        improved = best is None or (score > best["score"] if classification else score < best["score"])
        if improved:
            self.publish(bundle)
            state["best"] = record
        state["experiments"].append(record)
        state["finalized"] = False
        write_json(self.root / "state.json", state)
        return dict(experiment=record, improved=improved, best=state["best"])

    def blend(self, experiment_ids, weights=None):
        result = super().blend(experiment_ids, weights)
        record = result["experiment"]
        train, _ = self.load()
        with np.load(self.root / "split.npz") as split:
            va = train.iloc[split["validation"]]
        bundle = joblib.load(record["model_path"])
        np.save(self.root / f"validation_{record['id']}.npy", self.predict_raw(bundle, va, self.state()))
        state = self.state()
        if result["improved"]:
            state["finalized"] = False
        write_json(self.root / "state.json", state)
        return result

    def refit(self, bundle, train, state):
        if "members" in bundle:
            return dict(bundle, members=[self.refit(m, train, state) for m in bundle["members"]])
        x, cats = self.prepare(train, state, bundle.get("feature_code", ""))
        x, levels = self.encode(x, cats, bundle["algorithm"])
        model = self.model(state, bundle["refit_params"], bundle["algorithm"])
        self._fit(model, bundle["algorithm"], state, x, train[state["target"]], cats)
        return dict(bundle, model=model, levels=levels)

    def finalize(self):
        result = super().finalize()
        state = self.state()
        state["finalized_experiment"] = result["selected_experiment"]
        write_json(self.root / "state.json", state)
        return result


if __name__ == "__main__":
    # Ensure pickles always refer to an importable module, not __main__.
    from torch_backend import TorchResearch as Worker
    from diagnostics import Diagnostics
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker = Worker(request["root"], request["files"], request["folder"])
    if request["action"] in {"read_experiment", "inspect_categories", "validate_experiment", "analyze_errors", "compare_errors"}:
        diagnostic = Diagnostics.__new__(Diagnostics)
        diagnostic.engine, diagnostic.run = worker, Path(request["root"]).parent
        worker = diagnostic
    elif request["action"] not in {"inspect", "configure", "train", "blend", "finalize"}:
        raise ValueError("Unknown worker action")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})),
                     ensure_ascii=False, default=str))
