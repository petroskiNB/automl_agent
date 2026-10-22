"""Fixed regression experiment, no API. Does not overwrite submissions or models."""
import json
import time
from pathlib import Path

import numpy as np

from research import Research
from solution import discover_files


def main():
    project = Path(__file__).resolve().parent
    folder = project / "tasks" / "regression"
    engine = Research(project / "work" / "temporal_comparison", discover_files(folder), folder)
    train, test = engine.load()
    state = {"features": [c for c in test if c != "id"], "target": "site_eui",
             "task_type": "regression", "metric": "rmse"}
    x, cats = engine.prepare(train, state)
    y = train.site_eui
    out = project / "outputs"
    out.mkdir(exist_ok=True)
    configs = []
    for algorithm in ("catboost", "lightgbm"):
        for iterations in (200, 600):
            for regularization in (3, 10):
                params = dict(iterations=iterations, depth=7 if algorithm == "catboost" else 8,
                              learning_rate=.05, l2_leaf_reg=regularization)
                if algorithm == "lightgbm":
                    params.update(num_leaves=31, min_child_samples=20)
                configs.append((algorithm, params))
    # Include previous LightGBM baseline and best temporal leaf variant as controls.
    for leaves in (31, 127):
        configs.append(("lightgbm", dict(iterations=1500, depth=8, learning_rate=.05,
                                         l2_leaf_reg=0, num_leaves=leaves, min_child_samples=20)))
    rows = []

    def evaluate(index, period):
        algorithm, params = configs[index]
        tr, va = np.flatnonzero(train.Year_Factor < period), np.flatnonzero(train.Year_Factor == period)
        xt, levels = engine.encode(x.iloc[tr].copy(), cats, algorithm)
        xv, _ = engine.encode(x.iloc[va].copy(), cats, algorithm, levels)
        started = time.monotonic()
        model = engine.model(state, params, algorithm)
        model.fit(xt, y.iloc[tr], **({"cat_features": cats} if algorithm == "catboost" else {}))
        pred = np.asarray(model.predict(xv)).reshape(-1)
        score = engine.score(state, y.iloc[va], pred)
        assert np.isfinite(score)
        row = dict(config_id=index, algorithm=algorithm, params=params, period=period,
                   rmse=score, seconds=round(time.monotonic()-started, 2), train_rows=len(tr), validation_rows=len(va))
        rows.append(row)
        np.save(out / f"temporal_predictions_{index}_period{period}.npy", pred)
        (out / "temporal_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        return row

    for index in range(len(configs)):
        evaluate(index, 5)
    # Freeze top two per algorithm before observing the earlier-period scores.
    finalists = set()
    for algorithm in ("catboost", "lightgbm"):
        ranked = sorted([r for r in rows if r["algorithm"] == algorithm], key=lambda r: r["rmse"])
        finalists.update(r["config_id"] for r in ranked[:2])
    finalists.add(8)  # original LightGBM reference
    for index in sorted(finalists):
        evaluate(index, 4)
    lines = ["# Временная валидация: CatBoost и LightGBM", "",
             "Основной поиск: периоды 1–4 → 5. Перепроверка двух лучших вариантов каждого алгоритма: 1–3 → 4.",
             "Признаки одинаковые, включая Year_Factor. Кодирование категорий LightGBM обучается только на train части.",
             "Фиксированный seed 42, 8 потоков. Число деревьев фиксировано, early stopping не используется.",
             "Предыдущий случайный RMSE 42.90935 нельзя сравнивать напрямую с временными оценками.",
             "Периоды уже использовались в предыдущих исследованиях; это не независимый финальный тест.",
             "API не вызывался; старый ZIP, модели и submission.csv не изменены.", "",
             "| ID | Модель | Деревья | L2 | Листья | Период проверки | RMSE | Секунды |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        p = row["params"]
        lines.append(f"| {row['config_id']} | {row['algorithm']} | {p['iterations']} | {p['l2_leaf_reg']} | {p.get('num_leaves', '—')} | {row['period']} | {row['rmse']:.5f} | {row['seconds']} |")
    (out / "temporal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
