"""Nested classification validation, train/test shift diagnostics and deployable OOF ensembles."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from research import write_json
from xgb_backend import ExtendedResearch


def macro(y, probabilities, classes):
    return float(f1_score(y, np.asarray(classes)[np.asarray(probabilities).argmax(axis=1)],
                         labels=classes, average="macro", zero_division=0))


def fit_decision_rule(probabilities, y, classes):
    """Small regularized coordinate search on INNER validation predictions only."""
    count = len(probabilities)
    weights = np.full(count, 1 / count)
    def objective(w, multipliers):
        p = sum(a * b for a, b in zip(w, probabilities)) * multipliers
        return macro(y, p, classes) - .002 * float(np.mean(np.log(multipliers) ** 2))
    ones = np.ones(len(classes))
    # No unbounded optimizer and no negative ensemble coefficients.
    candidates = [weights] + [np.eye(count)[i] for i in range(count)]
    if count == 2:
        candidates += [np.array([.25, .75]), np.array([.75, .25])]
    weights = max(candidates, key=lambda w: objective(w, ones))
    multipliers = ones.copy()
    for _ in range(2):
        for k in range(len(classes)):
            best, value = objective(weights, multipliers), multipliers[k]
            for candidate in (.75, 1., 1.25, 1.5):
                trial = multipliers.copy()
                trial[k] = candidate
                score = objective(weights, trial)
                if score > best + 1e-10:
                    best, value = score, candidate
            multipliers[k] = value
    return weights.tolist(), multipliers.tolist()


class ClassificationResearch:
    def __init__(self, root, files, folder):
        self.engine = ExtendedResearch(root, files, folder)
        self.run, self.folder = Path(root).parent, Path(folder)

    def identity(self, config):
        paths = dict(self.engine.files)
        for name in ("research.py", "xgb_backend.py", "classification_research.py"):
            paths[name] = Path(__file__).with_name(name)
        return dict(config=config, hashes={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in paths.items()})

    def _state(self):
        state = self.engine.state()
        if state["task_type"] != "classification" or state["metric"] != "f1_macro":
            raise ValueError("Classification suites currently support Macro F1 only")
        return {k: state[k] for k in ("task_type", "metric", "features", "ids", "target")}

    def _fit(self, frame, state, recipe, seed):
        """Select iterations on internal data, then refit on the whole supplied frame."""
        tr, va = train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed,
                                 stratify=frame[state["target"]])
        algorithm = recipe["algorithm"]
        params, code = recipe["params"], recipe.get("feature_code", "")
        x, cats = self.engine.prepare(frame.iloc[tr], state, code)
        xv, _ = self.engine.prepare(frame.iloc[va], state, code)
        x, levels = self.engine.encode(x, cats, algorithm)
        xv, _ = self.engine.encode(xv, cats, algorithm, levels)
        model = self.engine.model(state, params, algorithm)
        y, yv = frame.iloc[tr][state["target"]], frame.iloc[va][state["target"]]
        if algorithm == "catboost":
            metric = "TotalF1:average=Macro" + (";use_weights=false" if params.get("auto_class_weights") else "")
            model.set_params(eval_metric=metric)
            model.fit(x, y, cat_features=cats, eval_set=(xv, yv), early_stopping_rounds=30, use_best_model=True)
            selected = model.tree_count_
        elif algorithm == "xgboost":
            model.fit(x, y, eval_set=(xv, yv), early_stopping_rounds=30)
            selected = model.selected_iterations
        elif algorithm == "lightgbm":
            from lightgbm import early_stopping
            labels = np.unique(y)
            def metric(truth, prediction):
                pred = (prediction >= .5).astype(int) if prediction.ndim == 1 else prediction.argmax(axis=1)
                return "f1_macro", float(f1_score(labels[truth.astype(int)], labels[pred], average="macro", zero_division=0)), True
            model.set_params(metric="None")
            model.fit(x, y, eval_set=[(xv, yv)], eval_metric=metric,
                      callbacks=[early_stopping(30, first_metric_only=True, verbose=False)])
            selected = model.best_iteration_ or model.n_estimators_
        else:
            raise ValueError("Unsupported base model")
        bundle = dict(model=model, algorithm=algorithm, levels=levels, feature_code=code,
                      classes=list(model.classes_), refit_params=dict(params, iterations=int(selected)))
        return self.engine.refit(bundle, frame, state)

    def _prob(self, bundle, frame, state, classes):
        raw = self.engine.predict_raw(bundle, frame, state)
        order = [list(bundle["classes"]).index(c) for c in classes]
        p = raw[:, order]
        if p.shape != (len(frame), len(classes)) or not np.isfinite(p).all():
            raise ValueError("Invalid class probabilities")
        return p

    def _views(self, probabilities, weights, multipliers):
        result = {f"model_{i}": p for i, p in enumerate(probabilities)}
        result["equal_ensemble"] = np.mean(probabilities, axis=0)
        result["weighted_ensemble"] = sum(w * p for w, p in zip(weights, probabilities))
        result["macro_decision"] = result["weighted_ensemble"] * np.asarray(multipliers)
        return result

    def adversarial_validation(self):
        state = self._state()
        identity = self.identity(dict(task=state, diagnostic="train_vs_test", folds=3, iterations=120, seed=42))
        rid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / "adversarial" / rid
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "report.json").exists():
            return json.loads((directory / "report.json").read_text(encoding="utf-8"))
        train, test = self.engine.load()
        data = pd.concat([train[state["features"]], test[state["features"]]], ignore_index=True)
        origin = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
        from catboost import CatBoostClassifier
        probabilities = np.zeros(len(data))
        importances, scores = [], []
        for i, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(data, origin)):
            x, cats = self.engine.prepare(data.iloc[tr], state)
            xv, _ = self.engine.prepare(data.iloc[va], state)
            model = CatBoostClassifier(iterations=120, depth=4, learning_rate=.08,
                thread_count=8, random_seed=42, verbose=False, allow_writing_files=False)
            model.fit(x, origin[tr], cat_features=cats)
            probabilities[va] = model.predict_proba(xv)[:, 1]
            scores.append(float(roc_auc_score(origin[va], probabilities[va])))
            importances.append(model.feature_importances_)
            np.savez(directory / f"fold_{i}.npz", train=tr, validation=va)
        # A prespecified stress subset, balanced by target class proportions, never used for ranking.
        y = train[state["target"]].to_numpy()
        stress = []
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            number = max(1, int(np.floor(.2 * len(rows))))
            stress.extend(rows[np.argsort(probabilities[rows], kind="stable")[-number:]].tolist())
        np.save(directory / "origin_oof.npy", probabilities)
        np.save(directory / "stress_validation.npy", np.array(sorted(stress), dtype=int))
        report = dict(diagnostic_id=rid, auc=float(roc_auc_score(origin, probabilities)), fold_auc=scores,
            excluded_ids=state["ids"], important_features=sorted(
                [{"feature": c, "importance": float(v)} for c, v in zip(state["features"], np.mean(importances, axis=0))],
                key=lambda row: row["importance"], reverse=True)[:10],
            duplicate_train_rows=int(train.duplicated(state["features"], keep=False).sum()),
            train_rows_matching_test=int(pd.util.hash_pandas_object(train[state["features"]], index=False).isin(
                pd.util.hash_pandas_object(test[state["features"]], index=False)).sum()),
            stress_rows=len(stress), report_path=str(directory / "report.json"),
            note="OOF train/test discrimination without target or ID features. Stress rows are a diagnostic proxy, "
                 "not the hidden test. High AUC does not establish target conditional shift.")
        write_json(directory / "plan.json", identity)
        write_json(directory / "report.json", report)
        write_json(self.run / "adversarial_summary.json", report)
        return report

    def run_classification_suite(self, experiment_ids, folds=3, shift_id=None):
        lock = self.run / ".classification.lock"
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.close(handle)
            return self._suite(experiment_ids, folds, shift_id)
        finally:
            lock.unlink(missing_ok=True)

    def _suite(self, experiment_ids, folds, shift_id):
        if not 2 <= len(experiment_ids) <= 3 or len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Select 2..3 distinct saved recipes")
        if folds not in (3, 4, 5):
            raise ValueError("Use 3..5 folds")
        state, full = self._state(), self.engine.state()
        recipes = []
        for eid in experiment_ids:
            recipe = next((e for e in full["experiments"] if e["id"] == eid), None)
            if not recipe or recipe["algorithm"] not in {"catboost", "lightgbm", "xgboost"}:
                raise ValueError("Select saved single-model recipes")
            recipes.append({k: recipe[k] for k in ("id", "algorithm", "params", "feature_code")})
        config = dict(task=state, recipes=recipes, folds=folds, seed=42, shift_id=shift_id,
                      selection="highest equal-weight outer-fold Macro F1; ties prefer first base model")
        identity = self.identity(config)
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.run / "classification_oof" / sid
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "report.json").exists():
            return json.loads((directory / "report.json").read_text(encoding="utf-8"))
        write_json(directory / "plan.json", identity)
        train, _ = self.engine.load()
        y, classes = train[state["target"]].to_numpy(), np.unique(train[state["target"]])
        if pd.Series(y).value_counts().min() < max(10, folds * 2):
            raise ValueError("Too few examples in a class for nested stratified validation")
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=42).split(train, y))
        if shift_id:
            self._check_id(shift_id)
            adir = self.run / "adversarial" / shift_id
            saved = json.loads((adir / "plan.json").read_text(encoding="utf-8"))
            if saved != self.identity(saved["config"]):
                raise ValueError("Shift diagnostic is stale")
            va = np.load(adir / "stress_validation.npy")
            splits.append((np.setdiff1d(np.arange(len(train)), va), va))
        oof, rows, stress_scores = {}, [], None
        for i, (tr, va) in enumerate(splits):
            fold_dir = directory / f"fold_{i}"
            fold_dir.mkdir(exist_ok=True)
            np.savez(fold_dir / "indices.npz", train=tr, validation=va)
            if (fold_dir / "scores.json").exists():
                row = json.loads((fold_dir / "scores.json").read_text(encoding="utf-8"))
                views = dict(np.load(fold_dir / "predictions.npz"))
            else:
                # Rule learning uses an internal calibration set, independent of outer validation.
                inner_tr, calibration = train_test_split(tr, test_size=.25, random_state=100 + i, stratify=y[tr])
                np.savez(fold_dir / "calibration_indices.npz", train=inner_tr, validation=calibration)
                calibration_prob, bundles, outer_prob = [], [], []
                for j, recipe in enumerate(recipes):
                    path = fold_dir / f"model_{j}.joblib"
                    cpath = fold_dir / f"calibration_{j}.npy"
                    if path.exists() and cpath.exists():
                        bundle, cp = joblib.load(path), np.load(cpath)
                    else:
                        calibration_model = self._fit(train.iloc[inner_tr], state, recipe, 200 + i)
                        cp = self._prob(calibration_model, train.iloc[calibration], state, classes)
                        bundle = self._fit(train.iloc[tr], state, recipe, 300 + i)
                        joblib.dump(bundle, path)
                        np.save(cpath, cp)
                    calibration_prob.append(cp)
                    bundles.append(bundle)
                    outer_prob.append(self._prob(bundle, train.iloc[va], state, classes))
                weights, multipliers = fit_decision_rule(calibration_prob, y[calibration], classes)
                views = self._views(outer_prob, weights, multipliers)
                row = dict(fold=i, rows=len(va), scores={name: macro(y[va], p, classes) for name, p in views.items()},
                           weights=weights, multipliers=multipliers)
                joblib.dump(dict(models=bundles, classes=classes.tolist(), state=state,
                                 weights=weights, multipliers=multipliers), fold_dir / "pipeline.joblib")
                np.savez(fold_dir / "predictions.npz", **views)
                write_json(fold_dir / "scores.json", row)
            if i == folds:
                stress_scores = row["scores"]
                continue
            rows.append(row)
            for name, pred in views.items():
                if name not in oof:
                    oof[name] = np.full((len(train), len(classes)), np.nan)
                oof[name][va] = pred
        results = []
        for name, pred in oof.items():
            if not np.isfinite(pred).all():
                raise ValueError("Incomplete OOF coverage")
            scores = [row["scores"][name] for row in rows]
            results.append(dict(candidate=name, mean_f1=float(np.mean(scores)), worst_f1=float(min(scores)),
                std_f1=float(np.std(scores)), fold_f1=scores, oof_f1=macro(y, pred, classes),
                stress_f1=stress_scores[name] if stress_scores else None,
                per_class=classification_report(y, classes[pred.argmax(axis=1)], labels=classes,
                                                output_dict=True, zero_division=0)))
        np.savez(directory / "oof.npz", **oof)
        winner = max(results, key=lambda r: r["mean_f1"])["candidate"]
        for result in results:
            result["weakest_classes"] = sorted([dict(label=str(c), **result["per_class"][str(c)]) for c in classes],
                                               key=lambda row: row["f1-score"])[:4]
        report = dict(suite_id=sid, selected_candidate=winner, results=results,
            report_path=str(directory / "report.json"), classes=classes.tolist(),
            note="Outer folds never tune tree counts, ensemble weights or class multipliers. "
                 "Recipes were previously searched: not a new untouched test. Shift stress fold is diagnostic only.")
        write_json(directory / "report.json", report)
        write_json(self.run / "classification_summary.json", dict(suite_id=sid, selected_candidate=winner,
            results=[{k: v for k, v in r.items() if k != "per_class"} for r in results], report_path=report["report_path"], note=report["note"]))
        return report

    @staticmethod
    def _check_id(value):
        if len(value) != 16 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid artifact id")

    def export_classification(self, suite_id):
        self._check_id(suite_id)
        directory = self.run / "classification_oof" / suite_id
        identity = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        if identity != self.identity(identity["config"]):
            raise ValueError("Stale classification suite")
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        candidate = report["selected_candidate"]
        pipelines = [joblib.load(directory / f"fold_{i}/pipeline.joblib") for i in range(identity["config"]["folds"])]
        package = dict(kind="classification_oof", candidate=candidate, folds=pipelines)
        _, test = self.engine.load()
        scores = self.predict_package(package, test)
        labels = np.asarray(pipelines[0]["classes"])[scores.argmax(axis=1)]
        state = pipelines[0]["state"]
        submission = test[state["ids"]].copy()
        submission[state["target"]] = labels
        submission.to_csv(directory / "submission.csv", index=False)
        joblib.dump(package, directory / "final_model.joblib")
        return dict(kind="classification_oof", suite_id=suite_id, candidate=candidate, rows=len(test),
                    submission_path=str(directory / "submission.csv"), model_path=str(directory / "final_model.joblib"),
                    note="Fold bagging, not full-data refit. Preserves fold-specific decision rules and class order.")

    def predict_package(self, package, frame):
        outputs = []
        for pipeline in package["folds"]:
            probabilities = [self._prob(b, frame, pipeline["state"], pipeline["classes"]) for b in pipeline["models"]]
            outputs.append(self._views(probabilities, pipeline["weights"], pipeline["multipliers"])[package["candidate"]])
        scores = np.mean(outputs, axis=0)
        if not np.isfinite(scores).all():
            raise ValueError("Invalid ensemble scores")
        return scores


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker = ClassificationResearch(request["root"], request["files"], request["folder"])
    if request["action"] not in {"adversarial_validation", "run_classification_suite", "export_classification"}:
        raise ValueError("Unknown classification action")
    result = getattr(worker, request["action"])(**request.get("arguments", {}))
    if "results" in result:
        result = dict(result, results=[{k: v for k, v in r.items() if k != "per_class"} for r in result["results"]])
    print(json.dumps(result, ensure_ascii=False))
