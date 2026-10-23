"""Isolated PyTorch version of the Qwen agent. See PYTORCH_README.md."""
import argparse
import copy
import hashlib
import importlib
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from torch_resources import configure

ROOT = Path(__file__).resolve().parent


def controller():
    return importlib.import_module("legacy_solution" if (ROOT/"legacy_solution.py").is_file() else "solution")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_task(task, destination):
    """Independent input copies; no writes to original task/runs/submission."""
    discover_files = controller().discover_files
    task, destination = Path(task).resolve(), Path(destination).resolve()
    if destination == task.parent or task.is_relative_to(destination):
        raise ValueError("Experimental output must be separate from the source task directory")
    inputs = dict(discover_files(task.parent), **{"task.md": task})
    manifest = {name: sha256(path) for name, path in inputs.items()}
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "source_inputs.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Source data changed; use a new experimental output directory")
        if any(not (destination/name).is_file() or sha256(destination/name) != digest for name, digest in manifest.items()):
            raise ValueError("Staged inputs changed or missing; use a new output directory")
    else:
        if list(destination.iterdir()):
            raise ValueError("Refusing to stage inputs into a nonempty unrecognized directory")
        for name, source in inputs.items():
            shutil.copy2(source, destination / name)
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination / "task.md"


def limited_run_python(code, folder, log_dir, step, timeout):
    """Like legacy executor, but respects allocation and enforces a wall deadline."""
    try:
        compile(code, "<experiment>", "exec")
    except SyntaxError as error:
        return {"exit_code": 1, "error": str(error)}
    script = Path(log_dir) / f"step_{step:03d}.py"
    script.write_text(code, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    allocation = configure()
    env.update({k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS")})
    env.update(PYTHONIOENCODING="utf-8", AUTOML_CPU_THREADS=str(allocation),
               AUTOML_WORKER_DEADLINE=str(time.time()+max(0, timeout-3)))
    started = time.monotonic()
    out_path, err_path = script.with_suffix(".stdout.txt"), script.with_suffix(".stderr.txt")
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        try:
            process = subprocess.run([sys.executable, "-u", str(script)], cwd=folder, env=env,
                                     stdout=out, stderr=err, timeout=max(.001, timeout))
            result = {"exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            result = {"exit_code": -1, "error": f"Step exceeded {timeout:.1f}s; previous submission preserved"}
    for name, path in (("stdout", out_path), ("stderr", err_path)):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size-16000))
            result[name] = stream.read().decode("utf-8", errors="replace")
    result["seconds"] = time.monotonic()-started
    return result


def run_worker(action, arguments, folder, files, log_dir, step, timeout, module_name=None):
    solution = controller()
    original = solution.run_python
    solution.run_python = limited_run_python
    try:
        target = module_name or "xgb_backend.py"
        code_actions = {"inspect", "configure", "train", "blend", "finalize"}
        if action in code_actions or target == "diagnostics.py":
            target = "torch_backend.py"
        result = solution.run_research_original(action, arguments, folder, files, log_dir, step, timeout,
                                               module_name=target if target == "torch_backend.py" else "torch_suites.py")
        if result.get("exit_code") == -1:
            task_lock = Path(folder)/".agent.lock"
            if task_lock.exists() and task_lock.read_text().strip() == str(os.getpid()):
                abandoned = {"run_classification_suite": ".classification.lock", "run_temporal_suite": ".temporal.lock"}.get(action)
                if abandoned:
                    (Path(log_dir)/abandoned).unlink(missing_ok=True)
        return result
    finally:
        solution.run_python = original


def install_hooks():
    solution = controller()
    if getattr(solution, "_torch_hooks", False):
        return
    solution._torch_hooks = True
    solution.run_research_original = solution.run_research
    solution.run_research = run_worker
    solution.run_python = limited_run_python
    original_fingerprint = solution.fingerprint
    def fingerprint(task, files):
        result = original_fingerprint(task, files)
        for name in ("torch_resources.py", "torch_model.py", "torch_backend.py", "torch_suites.py", "solution_torch.py", "diagnostics.py",
                     "classification_research.py", "classification_checks.py", "temporal_research.py",
                     "final_output.py", "parallel_research.py", "research_team.py"):
            result[name] = sha256(ROOT/name)
        result["cpu_threads"] = str(configure())
        return result
    solution.fingerprint = fingerprint
    solution.compatible_manifest = lambda saved, current, state: saved == current
    import parallel_research
    parallel_research.run_batch = partial(parallel_research.run_batch, backend="torch_backend", total_threads=configure())
    solution.TOOLS = copy.deepcopy(solution.TOOLS)
    train_schema = next(t["function"] for t in solution.TOOLS if t["function"]["name"] == "train_model")
    properties = train_schema["parameters"]["properties"]
    properties["algorithm"]["enum"] += ["torch_mlp", "torch_resnet"]
    properties["params"]["properties"].update({
        "width": {"type": "integer", "minimum": 8, "maximum": 1024},
        "layers": {"type": "integer", "minimum": 1, "maximum": 8},
        "batch_size": {"type": "integer", "minimum": 16, "maximum": 8192},
        "dropout": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
        "weight_decay": {"type": "number", "minimum": 0, "maximum": 1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "max_categories": {"type": "integer", "minimum": 2, "maximum": 50000},
        "loss": {"type": "string", "enum": ["auto", "mse", "mae", "huber"]},
        "target_transform": {"type": "string", "enum": ["standard", "log1p"]}})
    solution.SYSTEM_PROMPT += """
Экспериментальная версия PyTorch. Дополнительные алгоритмы: torch_mlp и torch_resnet.
Это компактные MLP с embeddings категорий; torch_resnet содержит residual-блоки, это НЕ TabM.
После быстрого бустингового baseline обязательно сравни torch_mlp (iterations=80,
width=256, layers=3, batch_size=512, learning_rate=0.001, dropout=0.1,
weight_decay=0.0001, seed=42; early_stopping_rounds=10 означает терпение в ЭПОХАХ).
Затем проверь torch_resnet и для классификации SqrtBalanced против отсутствия весов.
Для нейросетей не передавай depth/l2_leaf_reg и параметры деревьев. Допустимы также
max_categories, loss (auto/mse/mae/huber), target_transform (standard/log1p).
Регрессионные loss/target_transform не применимы к классификации.
Preprocessing, словари категорий и статистики таргета обучаются backend только на train.
Сравни одиночную сеть и равновесный ансамбль с лучшим бустингом на том же split.
Финалистов перепроверь с seed=43 по бюджету, не трать всё время на подбор holdout-весов.
Непрерывное продолжение экспериментов доступно через resume; незавершённое обучение
одной сети после аварийного завершения не возобновляется с оптимизатора.
Поддерживаются три исследовательских направления (--researchers 3) и очередь обучения
до двух процессов (--training-jobs 2). Общие CPU-потоки делятся между процессами.
Для классификации сравни сеть и бустинг через run_classification_suite: число эпох,
preprocessing и правила решений выбираются только внутри внешних фолдов.
Временной suite истории пока принимает CatBoost; сети сравнивай на причинных фолдах
через benchmark_torch.py. Не отправляй сетевой рецепт в CatBoost-only temporal suite.
Смотри fit_seconds, predict_seconds и peak_rss_mb в результатах. Качество не предполагается.
"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--platform", action="store_true", help="Container mode: publish next to task.md")
    parser.add_argument("--time-budget", type=int, default=9900)
    options, remaining = parser.parse_known_args()
    configure(options.threads)
    if not 0 < options.time_budget <= 10800:
        raise ValueError("Total platform time budget must be in 1..10800 seconds")
    install_hooks()
    solution = controller()
    if "--help" in remaining or "-h" in remaining:
        print("Experimental options: --threads TOTAL 1..8 (default 8), --output-dir PATH. Input copies/results stay there.")
        sys.argv = [sys.argv[0]] + remaining
        solution.main()
        return
    # Original CLI requires positional task first; all its other options remain available.
    if not remaining or remaining[0].startswith("-"):
        raise ValueError("Usage: solution_torch.py TASK.md [--threads 2] [--output-dir PATH] [agent options]")
    source = Path(remaining[0]).resolve()
    destination = options.output_dir or ROOT/"outputs"/"pytorch"/(source.parent.name+"_"+hashlib.sha256(str(source).encode()).hexdigest()[:10])
    if options.platform and options.output_dir:
        raise ValueError("--platform and --output-dir cannot be combined")
    staged = source if options.platform else stage_task(source, destination)
    print(f"Experimental task: {staged}\nTotal CPU allocation: {options.threads}; "
          + ("container output next to task.md." if options.platform else "independent input copies and results."), flush=True)
    sys.argv = [sys.argv[0], str(staged)] + remaining[1:] + ["--time-budget", str(options.time_budget)]
    # Controller's default 9900 seconds leaves a platform margin below 10800 seconds.
    solution.main()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
