"""Explicit selection and atomic publication of a complete predictive pipeline."""
import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classification_research import ClassificationResearch
from temporal_research import TemporalResearch
from research import write_json
from prediction_policy import prediction_bounds


class FinalOutput:
    def __init__(self, root, files, folder):
        self.root, self.files, self.folder = Path(root), files, Path(folder)
        self.run = self.root.parent

    def _validate_selection(self, kind, suite_id):
        if kind not in {"classification_oof", "temporal", 'autogluon_oof'}:
            raise ValueError("Use classification_oof, temporal or autogluon_oof")
        ClassificationResearch._check_id(suite_id)
        directory = self.run / kind / suite_id
        identity = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if kind == 'autogluon_oof':
            from autogluon_research import AutoGluonResearch
            worker = AutoGluonResearch(self.root, self.files, self.folder)
        else:
            worker = (ClassificationResearch if kind == "classification_oof" else TemporalResearch)(self.root, self.files, self.folder)
        if kind == 'classification_oof' and identity['config'].get('backend') == 'checks_v1':
            from classification_checks import ClassificationChecks
            worker = ClassificationChecks(self.root, self.files, self.folder)
        if kind == 'classification_oof' and identity['config'].get('backend') == 'experts_v1':
            from classification_experts import ClassificationExperts
            worker = ClassificationExperts(self.root, self.files, self.folder)
        current = worker._identity(identity['config']) if kind == 'temporal' else worker.identity(identity['config'])
        if current != identity:
            raise ValueError("Selected suite is stale")
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        if any(state[k] != identity["config"]["task"][k] for k in ("target", "task_type", "metric", "features", "ids")):
            raise ValueError("Task contract differs from selected suite")
        return worker, report

    def select_final_candidate(self, kind, suite_id):
        _, report = self._validate_selection(kind, suite_id)
        if kind == 'temporal':
            from temporal_search import catalogue
            state = json.loads((self.root/'state.json').read_text(encoding='utf-8'))
            ranking = catalogue(self.run, state)
            if ranking['recommended_suite'] and suite_id != ranking['recommended_suite']:
                raise ValueError('Select the global best comparable temporal suite: ' + ranking['recommended_suite'])
        result = dict(kind=kind, suite_id=suite_id,
            candidate=report.get("selected_candidate", report.get("selected_mode")),
            note="Explicit switch of validation track; not a numeric comparison with legacy holdout score")
        write_json(self.run / "final_selection.json", result)
        return result

    def select_best_classification(self):
        """Recover a validated OOF winner even when the LLM never selected it."""
        state = json.loads((self.root / 'state.json').read_text(encoding='utf-8'))
        if state.get('task_type') != 'classification' or state.get('metric') != 'f1_macro':
            return dict(selected=False, reason='Not a Macro F1 classification task')
        groups, skipped = {}, []
        for kind in ('classification_oof', 'autogluon_oof'):
            for path in sorted((self.run / kind).glob('*/report.json')):
                try:
                    _, report = self._validate_selection(kind, path.parent.name)
                    config = json.loads(path.with_name('plan.json').read_text(encoding='utf-8'))['config']
                    folds = config['folds']
                    if folds not in (3, 4, 5):
                        raise ValueError('Unsupported outer fold count')
                    signature = hashlib.sha256()
                    coverage = None
                    for i in range(folds):
                        with np.load(path.parent / f'fold_{i}/indices.npz') as split:
                            tr = np.asarray(split['train'], dtype=np.int64)
                            va = np.asarray(split['validation'], dtype=np.int64)
                        n = len(tr) + len(va)
                        if not np.array_equal(np.sort(np.r_[tr, va]), np.arange(n)):
                            raise ValueError('Outer fold is not a full disjoint partition')
                        if coverage is None:
                            coverage = np.zeros(n, dtype=np.int64)
                        if n != len(coverage):
                            raise ValueError('Outer fold row counts differ')
                        coverage[va] += 1
                        for indices in (tr, va):
                            signature.update(len(indices).to_bytes(8, 'little'))
                            signature.update(indices.astype('<i8').tobytes())
                    if not (coverage == 1).all():
                        raise ValueError('Outer rows are not covered exactly once')
                    winner = next(v for v in report['results'] if v['candidate'] == report['selected_candidate'])
                    scores = np.asarray(winner['fold_f1'], dtype=float)
                    if (len(scores) != folds or not np.isfinite(scores).all() or
                            (scores < 0).any() or (scores > 1).any() or
                            not np.isclose(float(np.mean(scores)), winner['mean_f1'], rtol=0, atol=1e-10)):
                        raise ValueError('Invalid selected OOF metric')
                    entry = dict(kind=kind, suite_id=path.parent.name, candidate=winner['candidate'],
                        mean_f1=float(np.mean(scores)), worst_f1=float(np.min(scores)), std_f1=float(np.std(scores)))
                    groups.setdefault(signature.hexdigest(), []).append(entry)
                except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                    skipped.append(dict(kind=kind, suite_id=path.parent.name, reason=type(error).__name__))
        if len(groups) > 1:
            # Never choose between scores measured on different outer rows.
            saved_path = self.run / 'final_selection.json'
            saved = json.loads(saved_path.read_text(encoding='utf-8')) if saved_path.exists() else {}
            groups = {k: values for k, values in groups.items() if any(
                v['kind'] == saved.get('kind') and v['suite_id'] == saved.get('suite_id') for v in values)}
        if len(groups) != 1:
            result = dict(selected=False, reason='No unique comparable OOF plan; preserve explicit selection or legacy fallback', skipped=skipped)
        else:
            candidates = sorted(next(iter(groups.values())),
                key=lambda v: (-v['mean_f1'], -v['worst_f1'], v['std_f1'], v['kind'] != 'classification_oof', v['suite_id']))
            best = candidates[0]
            selection = self.select_final_candidate(best['kind'], best['suite_id'])
            result = dict(selected=True, selection=selection, candidates=candidates, skipped=skipped,
                criterion='Maximum mean Macro F1 on identical outer rows; ties: worst, std, original classification track, stable ID. No holdout comparison.')
        write_json(self.run / 'classification_selection.json', result)
        return result

    def recover_classification_checkpoints(self):
        from autogluon_research import AutoGluonResearch
        recovered, skipped = [], []
        worker = AutoGluonResearch(self.root, self.files, self.folder)
        for path in sorted((self.run/'autogluon_oof').glob('*/plan.json')):
            if path.with_name('report.json').exists():
                continue
            try:
                config = json.loads(path.read_text(encoding='utf-8'))['config']
                if not all((path.parent/f'fold_{i}/scores.json').exists() for i in range(config['folds'])):
                    skipped.append(dict(suite_id=path.parent.name, reason='Incomplete fold checkpoints'))
                    continue
                report = worker.recover_autogluon_suite(path.parent.name)
                recovered.append(dict(suite_id=path.parent.name, selected_candidate=report['selected_candidate']))
            except (OSError, ValueError, KeyError, TypeError) as error:
                skipped.append(dict(suite_id=path.parent.name, reason=str(error)))
        return dict(recovered=recovered, skipped=skipped, training_performed=False)

    def finalize_selected(self):
        selection = json.loads((self.run / "final_selection.json").read_text(encoding="utf-8"))
        kind, sid = selection["kind"], selection["suite_id"]
        worker, _ = self._validate_selection(kind, sid)
        if kind == 'autogluon_oof':
            result = worker.export_autogluon(sid)
        else:
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
            bounds = prediction_bounds(state)
            if bounds and ((bounds.get('lower') is not None and (output[state['target']] < bounds['lower']).any()) or
                           (bounds.get('upper') is not None and (output[state['target']] > bounds['upper']).any())):
                raise ValueError('Selected pipeline violates persisted prediction bounds; regenerate its predictions')
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
    if request["action"] not in {"select_final_candidate", "finalize_selected", 'select_best_classification', 'recover_classification_checkpoints'}:
        raise ValueError("Unknown final output action")
    print(json.dumps(getattr(worker, request["action"])(**request.get("arguments", {})), ensure_ascii=False))
