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
    if getattr(temporal.TemporalResearch, '_hybrid_suite_hooks', False):
        return classification, temporal, checks
    from torch_backend import TorchResearch
    from research import Research
    class TemporalEngine(TorchResearch):
        # The existing temporal design hands categories straight to CatBoost.
        def prepare(self, frame, state, feature_code=""):
            return Research.prepare(self, frame, state, feature_code)
    # These overrides are local to the experimental worker process.
    classification.ExtendedResearch = TorchResearch
    temporal.Research = TemporalEngine
    classification.ClassificationResearch.supported_algorithms |= {"torch_mlp", "torch_resnet", 'torch_tabm'}
    original_classification = classification.ClassificationResearch.identity
    original_temporal = temporal.TemporalResearch._identity
    original_finalize = temporal.TemporalResearch.finalize_temporal
    original_predict = temporal.TemporalResearch._predict
    def extend(result):
        for name in ("torch_suites.py", "torch_backend.py", "torch_model.py", "torch_resources.py", "research_policy.py", "reference_recipes.py", "xgb_backend.py", "prediction_policy.py", 'hierarchical_history.py', 'temporal_ensemble.py', 'temporal_search.py', 'training_contract.py'):
            result["hashes"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        result["hashes"]["cpu_threads"] = str(configure())
        for name in ('tabm_model.py', 'tabm_vendor.py', 'num_embeddings_vendor.py'):
            result['hashes'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return result
    classification.ClassificationResearch.identity = lambda self, config: extend(original_classification(self, config))
    def flaml_worker(worker):
        from flaml_research import FlamlResearch
        return FlamlResearch(worker.engine.root, worker.engine.files, worker.engine.folder)
    def lightautoml_worker(worker):
        from lightautoml_research import LightAutoMLResearch
        return LightAutoMLResearch(worker.engine.root, worker.engine.files, worker.engine.folder)
    def specialist_worker(worker):
        from regression_specialists import RegressionSpecialists
        return RegressionSpecialists(worker.engine.root, worker.engine.files, worker.engine.folder)
    def identity(worker, config):
        if config.get('backend') == 'specialists_v1':
            return specialist_worker(worker)._identity(config)
        if config.get('backend') == 'flaml_v1':
            return flaml_worker(worker)._identity(config)
        if config.get('backend') == 'lightautoml_v1':
            return lightautoml_worker(worker)._identity(config)
        return extend(original_temporal(worker, config))
    def finalize(worker, suite_id, mode=None):
        if not isinstance(suite_id, str) or len(suite_id) != 16 or any(c not in '0123456789abcdef' for c in suite_id):
            raise ValueError('Invalid temporal suite ID')
        config = json.loads((worker.run/'temporal'/suite_id/'plan.json').read_text(encoding='utf-8'))['config']
        if config.get('backend') == 'specialists_v1':
            return specialist_worker(worker).finalize_temporal(suite_id, mode=mode)
        if config.get('backend') == 'flaml_v1':
            return flaml_worker(worker).finalize_temporal(suite_id, mode=mode)
        if config.get('backend') == 'lightautoml_v1':
            return lightautoml_worker(worker).finalize_temporal(suite_id, mode=mode)
        return original_finalize(worker, suite_id, mode=mode)
    def predict(worker, bundle, frame):
        if bundle.get('backend') == 'specialists_v1':
            return specialist_worker(worker)._predict(bundle, frame)
        if bundle.get('backend') == 'flaml_v1':
            return flaml_worker(worker)._predict(bundle, frame)
        if bundle.get('backend') == 'lightautoml_v1':
            return lightautoml_worker(worker)._predict(bundle, frame)
        return original_predict(worker, bundle, frame)
    temporal.TemporalResearch._identity = identity
    temporal.TemporalResearch.finalize_temporal = finalize
    temporal.TemporalResearch._predict = predict
    temporal.TemporalResearch._hybrid_suite_hooks = True
    return classification, temporal, checks


def main():
    classification, temporal, checks = install_suite_engines()
    from final_output import FinalOutput
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    action = request["action"]
    if action == 'run_flaml_suite':
        from flaml_research import FlamlResearch
        cls = FlamlResearch
    elif action == 'run_classification_experts':
        from classification_experts import ClassificationExperts
        cls = ClassificationExperts
    elif action == 'run_regression_specialists':
        from regression_specialists import RegressionSpecialists
        cls = RegressionSpecialists
    elif action == 'run_lightautoml_suite':
        from lightautoml_research import LightAutoMLResearch
        cls = LightAutoMLResearch
    elif action == 'run_regression_anchors':
        from temporal_anchors import TemporalAnchors
        cls = TemporalAnchors
    elif action == 'audit_classification_shift':
        from classification_shift_audit import ClassificationShiftAudit
        cls = ClassificationShiftAudit
    elif action in {'run_autogluon_suite', 'recover_autogluon_suite'}:
        from autogluon_research import AutoGluonResearch
        cls = AutoGluonResearch
    elif action in {"adversarial_validation", "run_classification_suite", 'run_saved_recipe_oof'}:
        cls = classification.ClassificationResearch
    elif action in {"run_temporal_suite", "finalize_temporal", 'blend_temporal_suites'}:
        cls = temporal.TemporalResearch
    elif action in {"select_final_candidate", "finalize_selected", 'select_best_classification', 'recover_classification_checkpoints'}:
        cls = FinalOutput
    elif action in {"reblend_suite", "stress_errors", "grouped_validation", "stress_ablation"}:
        cls = checks.ClassificationChecks
    else:
        raise ValueError("Unknown suite action")
    worker = cls(request["root"], request["files"], request["folder"])
    print(json.dumps(getattr(worker, action)(**request.get("arguments", {})), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
