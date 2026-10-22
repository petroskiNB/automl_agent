"""Local integration check on public regression; no API, no submission overwrite."""
import json
from datetime import datetime
from pathlib import Path

from research import Research
from solution import discover_files


def main():
    project = Path(__file__).resolve().parent
    run = project / "work" / ("v2_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True)
    engine = Research(run / "research", discover_files(project / "tasks" / "regression"), run)
    engine.configure("site_eui", "regression", "rmse", "time", "Year_Factor")
    results = []
    for algo, params, code in [
        ("catboost", dict(iterations=600, depth=7, learning_rate=.05, l2_leaf_reg=3), ""),
        ("lightgbm", dict(iterations=600, depth=8, learning_rate=.05, l2_leaf_reg=10), ""),
        ("catboost", dict(iterations=600, depth=7, learning_rate=.05, l2_leaf_reg=3),
         "def transform(X):\n    return X.drop(columns=['Year_Factor'])"),
    ]:
        result = engine.train("Integration check: early stopping and optional removal of period", params, algo,
                              feature_code=code, early_stopping_rounds=50)
        results.append(result["experiment"])
        print(json.dumps(result["experiment"], ensure_ascii=False), flush=True)
    results.append(engine.blend([1, 2])["experiment"])
    results.append(engine.blend([2, 3])["experiment"])
    final = engine.finalize()
    report = dict(experiments=results, final=final, run_directory=str(run),
                  note="Public period-5 validation, no API, not an independent test or platform score")
    out = project / "outputs"
    out.mkdir(exist_ok=True)
    (out / "research_v2_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
