"""Budgeted, offline, same-split comparison; no LLM calls. --smoke uses synthetic data."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from torch_resources import configure


def smoke_data(folder, classification):
    import numpy as np
    import pandas as pd
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(71)
    n = 900
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    category = rng.choice(["low", "medium", "high"], n)
    y = x*3 + z*.7 + (category == "high")*2 + rng.normal(0, .2, n)
    if classification:
        y = np.where(y > 2, "Z", np.where(y < -1, "A", "M"))
    frame = pd.DataFrame(dict(id=np.arange(n), x=x, z=z, category=category,
                              period=np.repeat([1, 2, 3, 4, 5, 6], 150), target=y))
    frame.loc[::29, "x"] = np.nan
    frame.loc[::31, "category"] = None
    frame.to_csv(folder/"train.csv", index=False)
    test = frame.drop(columns="target").iloc[:80].copy()
    test.id += n
    test.loc[test.index[:3], "category"] = "never_seen"
    test.to_csv(folder/"test.csv", index=False)
    pd.DataFrame({"id": test.id, "target": "A" if classification else 0}).to_csv(folder/"sample_submission.csv", index=False)
    pd.DataFrame({"feature": frame.columns, "role": ["target" if c == "target" else "feature" for c in frame]}).to_csv(folder/"data_dictionary.csv", index=False)
    (folder/"task.md").write_text("Synthetic integration benchmark, NOT hackathon results.\n", encoding="utf-8")
    return folder/"task.md"


def make_splits(frame, target, task_type, split, split_column, folds):
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit
    indices = np.arange(len(frame))
    if split == "time":
        if not split_column or frame[split_column].isna().any():
            raise ValueError("Time split needs a complete chronological column")
        periods = sorted(frame[split_column].unique())
        if len(periods) <= folds:
            raise ValueError("Need more periods than temporal folds")
        return [(np.flatnonzero(frame[split_column] < p), np.flatnonzero(frame[split_column] == p)) for p in periods[-folds:]]
    if split == "group":
        if not split_column or frame[split_column].isna().any():
            raise ValueError("Group split requires a complete group column")
        if folds == 1:
            return list(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(frame, groups=frame[split_column]))
        return list(GroupKFold(folds).split(frame, groups=frame[split_column]))
    if folds == 1:
        return [train_test_split(indices, test_size=.2, random_state=42,
                                stratify=frame[target] if task_type == "classification" else None)]
    splitter = StratifiedKFold(folds, shuffle=True, random_state=42) if task_type == "classification" else KFold(folds, shuffle=True, random_state=42)
    return list(splitter.split(indices, frame[target]))


def write_report(folder, config, results, failures, final_status):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error
    classification = config["task_type"] == "classification"
    metric = config["metric"]
    def score(y, raw):
        pred = np.asarray(config["classes"])[raw.argmax(1)] if classification else raw
        if metric == "f1_macro":
            return float(f1_score(y, pred, labels=config["classes"], average="macro", zero_division=0))
        if metric == "accuracy":
            return float(accuracy_score(y, pred))
        if metric == "mae":
            return float(mean_absolute_error(y, pred))
        return float(np.sqrt(mean_squared_error(y, pred)))
    by_fold = {}
    for r in results:
        if r["algorithm"] == "ensemble":
            continue
        label = r["candidate"]
        by_fold.setdefault((label, r["fold"]), []).append(r)
    pooled = {}
    for (label, fold), rows in by_fold.items():
        # Only compare complete seeds; incomplete recipes are visible in raw results, not promoted.
        if len(rows) != len(config["seeds"]):
            continue
        raw = np.mean([np.load(r["prediction_path"]) for r in rows], axis=0)
        pooled.setdefault(label, {})[fold] = raw
    complete = {label: values for label, values in pooled.items() if len(values) == config["folds"]}
    all_folds = list(range(config["folds"]))
    y = np.concatenate([np.load(folder/f"fold_{i}"/"truth.npy", allow_pickle=False) for i in all_folds])
    rankings = []
    for label, values in complete.items():
        scores = [score(np.load(folder/f"fold_{i}"/"truth.npy", allow_pickle=False), values[i]) for i in all_folds]
        rankings.append(dict(candidate=label, pooled_score=score(y, np.concatenate([values[i] for i in all_folds])),
                             fold_scores=scores, fold_std=float(np.std(scores))))
    trees = [label for label in complete if not label.startswith("torch_")]
    nets = [label for label in complete if label.startswith("torch_")]
    # Fixed equal weights, no optimizer fitted on these held-out predictions.
    for tree in trees:
        for net in nets:
            raw = np.concatenate([.5*complete[tree][i]+.5*complete[net][i] for i in all_folds])
            rankings.append(dict(candidate=f"{tree} + {net} (0.5/0.5)", pooled_score=score(y, raw), members=[tree, net]))
    rankings.sort(key=lambda r: r["pooled_score"], reverse=classification)
    report = dict(configuration=config, experiments=results, rankings=rankings, failures=failures, final=final_status,
        limitation="Repeated model selection/early stopping on these folds is development validation, not an independent test. Temporal scores cover only held-out future periods. Seed averaging is not an independent repetition of data splits.")
    (folder/"comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PyTorch comparison", "", "SYNTHETIC DATA - not leaderboard results." if config["synthetic"] else "Provided task data; validation scores are not hidden-test scores.", "",
             f"Metric: {metric}; split: {config['split']}; folds: {config['folds']}; seeds: {config['seeds']}; threads: {config['threads']}", "",
             "| Candidate | Fold | Seed | Score | Fit s | Predict s | Peak RSS MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['candidate']} | {r['fold']} | {r.get('seed', '')} | {r['score']:.6f} | {r.get('fit_seconds', 0):.3f} | {r.get('predict_seconds', 0):.4f} | {r.get('peak_rss_mb', 0):.1f} |")
    lines += ["", "## Pooled development validation (seeds averaged)", "", "| Candidate | Score |", "|---|---:|"]
    lines += [f"| {r['candidate']} | {r['pooled_score']:.6f} |" for r in rankings]
    lines += ["", report["limitation"], "", "Peak RSS includes the Python process and imported libraries (20 ms sampling), not just model tensors.", "", f"Final refit: {final_status}", "", f"Failures/incomplete experiments: {len(failures)}"]
    (folder/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, nargs="?")
    parser.add_argument("--smoke", choices=["regression", "classification"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="target")
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--metric", choices=["rmse", "mae", "f1_macro", "accuracy"], default="rmse")
    parser.add_argument("--split", choices=["random", "time", "group"], default="random")
    parser.add_argument("--split-column")
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--model-seconds", type=float, default=600)
    parser.add_argument("--time-budget", type=float, default=9900)
    parser.add_argument("--final-reserve", type=float, default=900)
    args = parser.parse_args()
    configure(args.threads)
    import numpy as np
    import pandas as pd
    from solution_torch import stage_task, install_hooks, run_worker, sha256
    install_hooks()
    import solution
    if args.smoke:
        # Synthetic datasets have a fixed tiny size. Reuse one process so startup
        # does not dominate the smoke check; real-data runs use hard subprocess caps.
        from torch_backend import TorchResearch
        def run_worker(action, arguments, folder, files, log_dir, step, timeout, module_name=None):
            started = time.monotonic()
            previous = os.environ.get("AUTOML_WORKER_DEADLINE")
            os.environ["AUTOML_WORKER_DEADLINE"] = str(time.time()+timeout)
            try:
                engine = TorchResearch(Path(log_dir)/"research", files, folder)
                getattr(engine, action)(**arguments)
                return dict(exit_code=0, seconds=time.monotonic()-started)
            except Exception as error:
                return dict(exit_code=1, error=f"{type(error).__name__}: {error}")
            finally:
                if previous is None:
                    os.environ.pop("AUTOML_WORKER_DEADLINE", None)
                else:
                    os.environ["AUTOML_WORKER_DEADLINE"] = previous
    if not 1 <= args.folds <= 10 or len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("Use 1..10 folds and distinct seeds")
    if not 0 < args.time_budget <= 10800 or args.model_seconds <= 0 or not 0 < args.final_reserve < args.time_budget:
        parser.error("Require positive budgets, final reserve < total <= 10800s")
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("Use a new output directory (existing runs are never overwritten)")
    if args.smoke:
        args.task_type = args.smoke
        args.metric = "f1_macro" if args.smoke == "classification" else "rmse"
        args.task = smoke_data(destination/"synthetic_source", args.smoke == "classification")
    elif args.task is None:
        parser.error("Provide task.md or --smoke")
    if args.metric not in ({"f1_macro", "accuracy"} if args.task_type == "classification" else {"rmse", "mae"}):
        parser.error("Metric does not match task type")
    staged = stage_task(args.task, destination/"inputs")
    files = solution.discover_files(staged.parent)
    frame, test = pd.read_csv(files["train.csv"]), pd.read_csv(files["test.csv"])
    ids = [c for c in pd.read_csv(files["sample_submission.csv"]) if c in test]
    if args.target not in frame or args.target in test or frame[args.target].isna().any() or not ids:
        raise ValueError("Invalid target/ID schema")
    classes = sorted(frame[args.target].unique().tolist()) if args.task_type == "classification" else None
    config = dict(task_type=args.task_type, metric=args.metric, split=args.split, split_column=args.split_column,
                  folds=args.folds, seeds=args.seeds, threads=args.threads, synthetic=bool(args.smoke), classes=classes,
                  model_seconds=args.model_seconds, epochs=args.epochs, trees=args.trees)
    import importlib.metadata
    config["versions"] = {name: importlib.metadata.version(name) for name in
                          ("torch", "numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost")}
    config["code_hashes"] = {name: sha256(Path(__file__).with_name(name)) for name in
        ("research.py", "torch_resources.py", "torch_model.py", "torch_backend.py", "benchmark_torch.py")}
    config["input_hashes"] = {name: sha256(path) for name, path in files.items()}
    config["execution"] = "single_process_synthetic_smoke" if args.smoke else "isolated_subprocesses"
    plans = make_splits(frame, args.target, args.task_type, args.split, args.split_column, args.folds)
    tree_params = dict(iterations=args.trees, depth=5, learning_rate=.05)
    net_params = dict(iterations=args.epochs, width=args.width, layers=3, batch_size=256,
                      learning_rate=.001, dropout=.1, weight_decay=.0001)
    recipes = [(a, a, tree_params) for a in ("catboost", "lightgbm", "xgboost")]
    recipes += [(a, a, net_params) for a in ("torch_mlp", "torch_resnet")]
    if classes:
        recipes += [("torch_mlp_weighted", "torch_mlp", dict(net_params, auto_class_weights="SqrtBalanced"))]
    results, failures = [], []
    total_deadline = time.monotonic()+args.time_budget
    search_deadline = total_deadline-args.final_reserve
    # Freeze every fold before running any candidate.
    for fold, (tr, va) in enumerate(plans):
        root = destination/f"fold_{fold}"/"research"
        root.mkdir(parents=True)
        if classes and set(frame.iloc[va][args.target])-set(frame.iloc[tr][args.target]):
            raise ValueError("Validation has unseen class; revise split")
        features = [c for c in test if c not in ids and not (args.split == "group" and c == args.split_column)]
        state = dict(target=args.target, task_type=args.task_type, metric=args.metric, split=args.split,
            split_column=args.split_column, features=features, ids=ids, experiments=[], best=None,
            train_rows=len(tr), validation_rows=len(va))
        (root/"state.json").write_text(json.dumps(state), encoding="utf-8")
        np.savez(root/"split.npz", train=tr, validation=va)
        truth = frame.iloc[va][args.target].to_numpy()
        np.save(root.parent/"truth.npy", truth.astype(str) if truth.dtype == object else truth)
    for label, algorithm, params in recipes:
        for seed in args.seeds:
            for fold in range(args.folds):
                folder = destination/f"fold_{fold}"
                remaining = search_deadline-time.monotonic()
                if remaining < 5:
                    failures.append(dict(candidate=label, seed=seed, fold=fold, error="Research budget exhausted"))
                    continue
                result = run_worker("train", dict(hypothesis=f"Fixed comparison {label} seed={seed}", algorithm=algorithm,
                    params=dict(params, seed=seed), early_stopping_rounds=10), folder, files, folder,
                    len(results)+len(failures)+1, min(args.model_seconds, remaining))
                if result["exit_code"]:
                    failures.append(dict(candidate=label, fold=fold, seed=seed, error=result))
                else:
                    state = json.loads((folder/"research"/"state.json").read_text(encoding="utf-8"))
                    record = dict(state["experiments"][-1], candidate=label, fold=fold,
                        worker_seconds=result["seconds"],
                        prediction_path=str(folder/"research"/f"validation_{state['experiments'][-1]['id']}.npy"))
                    results.append(record)
                    print(f"{label} fold={fold} seed={seed}: {record['score']:.6f}; fit={record['fit_seconds']:.2f}s", flush=True)
                write_report(destination, config, results, failures, "pending")
    report = write_report(destination, config, results, failures, "pending")
    # Refit the pooled-development winner using its first prescribed seed, not the luckiest seed.
    final_status = "No complete candidates; fold-local validation submissions may exist."
    if report["rankings"]:
        winner = report["rankings"][0]
        labels = winner.get("members", [winner["candidate"]])
        primary = destination/"fold_0"
        members = [next(r for r in results if r["candidate"] == label and r["fold"] == 0 and r["seed"] == args.seeds[0]) for label in labels]
        state_path = primary/"research"/"state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # Use persisted holdout models to publish fallback before any refit attempt.
        if len(members) == 2:
            result = run_worker("blend", {"experiment_ids": [r["id"] for r in members]}, primary, files, primary,
                                90001, max(.01, min(args.model_seconds, total_deadline-time.monotonic())))
            if result["exit_code"] == 0:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["best"] = state["experiments"][-1]
            else:
                failures.append(dict(action="blend", error=result))
                final_status = "Winner blend failed; using best available fold-local fallback"
        else:
            state["best"] = next(e for e in state["experiments"] if e["id"] == members[0]["id"])
        state["finalized"] = False
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        remaining = total_deadline-time.monotonic()
        fallback_note = final_status if final_status.startswith("Winner blend failed") else ""
        if remaining > 5:
            result = run_worker("finalize", {}, primary, files, primary, 90002, remaining)
            final_status = "Full refit completed" if result["exit_code"] == 0 else "Refit failed/timed out; fallback validation submission preserved"
            if result["exit_code"]:
                failures.append(dict(action="finalize", error=result))
            if fallback_note:
                final_status = fallback_note+"; "+final_status
        else:
            final_status = "No time for full refit; existing fold-local fallback submission preserved"
        valid, detail = solution.validate_submission(primary, files)
        final_status += f"; submission structure valid={valid}: {detail}"
    write_report(destination, config, results, failures, final_status)
    print(f"Report: {destination/'comparison.md'}\n{final_status}", flush=True)


if __name__ == "__main__":
    main()
