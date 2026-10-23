"""Versioned adapters for the updated validation/selection tools."""
import hashlib
import json
import sys
from pathlib import Path

from torch_resources import configure
configure()


def install_suite_engines():
    import classification_research as classification
    import temporal_research as temporal
    import classification_checks as checks
    from torch_backend import TorchResearch
    from research import Research
    class TemporalEngine(TorchResearch):
        # The existing temporal design hands categories straight to CatBoost.
        def prepare(self, frame, state, feature_code=""):
            return Research.prepare(self, frame, state, feature_code)
    # These overrides are local to the experimental worker process.
    classification.ExtendedResearch = TorchResearch
    temporal.Research = TemporalEngine
    classification.ClassificationResearch.supported_algorithms |= {"torch_mlp", "torch_resnet"}
    original_classification = classification.ClassificationResearch.identity
    original_temporal = temporal.TemporalResearch._identity
    def extend(result):
        for name in ("torch_suites.py", "torch_backend.py", "torch_model.py", "torch_resources.py"):
            result["hashes"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        result["hashes"]["cpu_threads"] = str(configure())
        return result
    classification.ClassificationResearch.identity = lambda self, config: extend(original_classification(self, config))
    temporal.TemporalResearch._identity = lambda self, config: extend(original_temporal(self, config))
    return classification, temporal, checks


def main():
    classification, temporal, checks = install_suite_engines()
    from final_output import FinalOutput
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    action = request["action"]
    if action in {"adversarial_validation", "run_classification_suite"}:
        cls = classification.ClassificationResearch
    elif action in {"run_temporal_suite", "finalize_temporal"}:
        cls = temporal.TemporalResearch
    elif action in {"select_final_candidate", "finalize_selected"}:
        cls = FinalOutput
    elif action in {"reblend_suite", "stress_errors", "grouped_validation", "stress_ablation"}:
        cls = checks.ClassificationChecks
    else:
        raise ValueError("Unknown suite action")
    worker = cls(request["root"], request["files"], request["folder"])
    print(json.dumps(getattr(worker, action)(**request.get("arguments", {})), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
