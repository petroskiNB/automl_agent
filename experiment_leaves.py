"""Offline LightGBM comparison; preserves submissions and existing models."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import Research
from solution import discover_files


def main():
    project = Path(__file__).resolve().parent
    folder = project / "tasks" / "regression"
    source = folder / "agent_runs" / "20261021_220854_212189" / "research"
    state = json.loads((source / "state.json").read_text(encoding="utf-8"))
    indices = np.load(source / "split.npz")
    work = project / "work" / "leaves_comparison"
    engine = Research(work, discover_files(folder), folder)
    train, _ = engine.load()
    x, cats = engine.prepare(train, state)
    y = train[state["target"]]
    out = project / "outputs"
    out.mkdir(exist_ok=True)
    records = []

    def evaluate(leaves, child, split, tr, va):
        params = dict(state["best"]["params"], num_leaves=leaves, min_child_samples=child)
        started = time.monotonic()
        xt, levels = engine.encode(x.iloc[tr].copy(), cats, "lightgbm")
        xv, _ = engine.encode(x.iloc[va].copy(), cats, "lightgbm", levels)
        model = engine.model(state, params, "lightgbm")
        model.fit(xt, y.iloc[tr])
        rmse = engine.score(state, y.iloc[va], model.predict(xv))
        row = dict(split=split, num_leaves=leaves, min_child_samples=child,
                   rmse=rmse, seconds=round(time.monotonic()-started, 2),
                   train_rows=len(tr), validation_rows=len(va))
        records.append(row)
        (out / "lightgbm_leaves_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        return row

    tr, va = indices["train"], indices["validation"]
    first = [evaluate(n, 20, "original_random", tr, va) for n in (31, 63, 127)]
    if abs(first[0]["rmse"] - state["best"]["score"]) > 1e-5:
        raise RuntimeError("Baseline was not reproduced; investigate before comparing")
    winner = min(first, key=lambda r: r["rmse"])["num_leaves"]
    for child in (50, 100):
        evaluate(winner, child, "original_random", tr, va)
    finalists = sorted(records, key=lambda r: r["rmse"])[:2]
    configs = {(31, 20)} | {(r["num_leaves"], r["min_child_samples"]) for r in finalists}
    for period in (5, 4):
        tr = np.flatnonzero(train.Year_Factor < period)
        va = np.flatnonzero(train.Year_Factor == period)
        for leaves, child in sorted(configs):
            evaluate(leaves, child, f"past_to_period_{period}", tr, va)
    lines = ["# LightGBM: число листьев и минимальный размер листа", "",
             "Одинаковые признаки и параметры исходного лучшего эксперимента; меняются только два указанных параметра.",
             "Исходный RMSE 42.90935 относится к случайному разбиению. Временные проверки не использованы для первичного отбора.",
             "Это локальная валидация, не оценка платформы. API не вызывался, submission.csv и старые модели не изменялись.", "",
             "| Разбиение | Листья | min_child_samples | RMSE | Секунды |", "|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['split']} | {r['num_leaves']} | {r['min_child_samples']} | {r['rmse']:.6f} | {r['seconds']} |")
    (out / "lightgbm_leaves_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
