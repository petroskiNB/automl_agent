"""Explicit selection and atomic publication of a complete predictive pipeline."""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classification_research import ClassificationResearch
from temporal_research import TemporalResearch
from research import write_json


class FinalOutput:
    def __init__(self, root, files, folder):
        self.root, self.files, self.folder = Path(root), files, Path(folder)
        self.run = self.root.parent

    def _validate_selection(self, kind, suite_id):
        if kind not in {"classification_oof", "temporal"}:
            raise ValueError("Use classification_oof or temporal")
        ClassificationResearch._check_id(suite_id)
        directory = self.run / kind / suite_id
        identity = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        worker = (ClassificationResearch if kind == "classification_oof" else TemporalResearch)(self.root, self.files, self.folder)
        if kind == 'classification_oof' and identity['config'].get('backend') == 'checks_v1':
            from classification_checks import ClassificationChecks
            worker = ClassificationChecks(self.root, self.files, self.folder)
        current = worker.identity(identity["config"]) if kind == "classification_oof" else worker._identity(identity["config"])
        if current != identity:
            raise ValueError("Selected suite is stale")
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        if any(state[k] != identity["config"]["task"][k] for k in ("target", "task_type", "metric", "features", "ids")):
            raise ValueError("Task contract differs from selected suite")
        return worker, report

    def select_final_candidate(self, kind, suite_id):
        _, report = self._validate_selection(kind, suite_id)
        result = dict(kind=kind, suite_id=suite_id,
            candidate=report.get("selected_candidate", report.get("selected_mode")),
            note="Explicit switch of validation track; not a numeric comparison with legacy holdout score")
        write_json(self.run / "final_selection.json", result)
        return result

    def finalize_selected(self):
        selection = json.loads((self.run / "final_selection.json").read_text(encoding="utf-8"))
        kind, sid = selection["kind"], selection["suite_id"]
        worker, _ = self._validate_selection(kind, sid)
        result = worker.export_classification(sid) if kind == "classification_oof" else worker.finalize_temporal(sid)
        source = Path(result["submission_path"])
        output = pd.read_csv(source)
        test, sample = pd.read_csv(self.files["test.csv"]), pd.read_csv(self.files["sample_submission.csv"])
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        if list(output) != list(sample) or len(output) != len(test) or not output[state["ids"]].equals(test[state["ids"]]):
            raise ValueError("Invalid candidate output structure")
        if output[state["target"]].isna().any():
            raise ValueError("Missing candidate predictions")
        if state["task_type"] == "regression":
            if not np.isfinite(output[state["target"]]).all():
                raise ValueError("Non-finite regression predictions")
        else:
            train_labels = pd.read_csv(self.files["train.csv"], usecols=[state["target"]])[state["target"]]
            if not set(output[state["target"]]).issubset(set(train_labels)):
                raise ValueError("Unknown predicted classes")
        destination = self.folder / "submission.csv"
        if destination.exists():
            shutil.copy2(destination, self.run / f"submission_before_selected_{time.time_ns()}.csv")
        temporary = self.folder / "submission.selected.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        result.update(published=True, selected_track=kind, selected_suite=sid)
        write_json(self.run / "final_output.json", result)
        return result


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker = FinalOutput(request["root"], request["files"], request["folder"])
    if request["action"] not in {"select_final_candidate", "finalize_selected"}:
        raise ValueError("Unknown final output action")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})), ensure_ascii=False))
