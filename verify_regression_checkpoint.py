"""Compare frozen old/new recipes on an earlier period, without publishing or API."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from research import Research
from solution import discover_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--period", type=int, default=4)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    run = args.run.resolve()
    state = json.loads((run / "research/state.json").read_text(encoding="utf-8"))
    folder = run.parent.parent
    engine = Research(project / "work" / "checkpoint_verification", discover_files(folder), folder)
    train, _ = engine.load()
    column = state["split_column"]
    tr = train.loc[train[column] < args.period]
    va = train.loc[train[column] == args.period]
    if tr.empty or va.empty:
        raise ValueError("Empty temporal fold")
    records = {e["id"]: e for e in state["experiments"]}
    ensembles = {"previous_79": records[79], "current_150": records[150]}
    ids = {i for e in ensembles.values() for i in e["params"]["ids"]}
    recipes = {str(i): records[i] for i in sorted(ids)}
    for i in (131, 135):
        e = dict(records[i])
        code = e["feature_code"]
        if ".astype(str).map(_h)" not in code:
            raise ValueError("Expected the saved hash recipe")
        e["feature_code"] = code.replace(".astype(str).map(_h)", ".astype(str)")
        recipes[f"native_{i}"] = e
    predictions, rows = {}, []
    for name, e in recipes.items():
        xt, cats = engine.prepare(tr, state, e.get("feature_code", ""))
        xv, _ = engine.prepare(va, state, e.get("feature_code", ""))
        xt, levels = engine.encode(xt, cats, e["algorithm"])
        xv, _ = engine.encode(xv, cats, e["algorithm"], levels)
        # Freeze iteration counts selected on period 5; never tune on this fold.
        model = engine.model(state, e["refit_params"], e["algorithm"])
        model.fit(xt, tr[state["target"]], **({"cat_features": cats} if e["algorithm"] == "catboost" else {}))
        predictions[name] = np.asarray(model.predict(xv)).reshape(-1)
        row = {"name": name, "rmse": engine.score(state, va[state["target"]], predictions[name]),
               "params": e["refit_params"], "categorical_columns": cats}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for name, e in ensembles.items():
        pred = sum(w * predictions[str(i)] for i, w in zip(e["params"]["ids"], e["params"]["weights"]))
        rows.append({"name": name, "rmse": engine.score(state, va[state["target"]], pred)})
    native = .6 * predictions["native_131"] + .15 * predictions["native_135"] + .25 * predictions["46"]
    rows.append({"name": "native_fixed_weights", "rmse": engine.score(state, va[state["target"]], native)})
    report = {"period": args.period, "train_rows": len(tr), "validation_rows": len(va),
              "note": "Secondary diagnostic, not an independent test: earlier periods have been inspected before. "
                      "Recipes, iteration counts and weights frozen before evaluating this fold. No submissions overwritten.",
              "rows": rows}
    out = project / "outputs" / f"checkpoint_verification_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics = run / "diagnostics.json"
    previous = json.loads(diagnostics.read_text(encoding="utf-8")) if diagnostics.exists() else []
    previous.append({"source": str(out), "period": args.period, "note": report["note"],
                     "iteration_policy": "Fixed counts from period 5, not separately tuned for native categories",
                     "results": [{"name": row["name"], "rmse": row["rmse"]} for row in rows]})
    diagnostics.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "ensembles": rows[-3:]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
