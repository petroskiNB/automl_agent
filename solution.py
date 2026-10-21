"""Minimal Qwen agent. Run: python solution.py tasks/regression/task.md.

The model requests run_python; this controller executes it and returns output.
Generated Python runs with your local user's permissions (not a sandbox).
"""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


SYSTEM_PROMPT = """Ты ML-инженер. Реши задачу из сообщения пользователя.
Используй inspect_data, затем configure_task, затем train_model для экспериментов.
configure_task фиксирует цель, метрику и разбиение до конца запуска. Выбирай их
по условию и данным, а не только по типу столбца. Поддерживаются regression
(rmse, mae) и classification (f1_macro, accuracy). Другие форматы пока не поддержаны.
train_model сам обучает, считает метрику, сохраняет модель и предсказания лучшего
эксперимента. Метрику и результат не придумывай. Начни с 100-200 итераций и
depth=4-5; затем сравни несколько гипотез, в том числе LightGBM, если установлен.
Финальное переобучение лучшего варианта выполняет контроллер автоматически.
run_python используй для дополнительного анализа; не перезаписывай state.json,
split.npz, модели и submission.csv и не обучай через него обходные эксперименты.
В train_model разрешены iterations, depth, learning_rate, l2_leaf_reg,
random_strength и auto_class_weights (CatBoost: Balanced или SqrtBalanced).
Для LightGBM random_strength и auto_class_weights не поддерживаются.
Все пояснения, сообщения о ходе работы и финальный отчёт пиши на русском языке.
Не переводи имена столбцов, метки классов, пути, названия библиотек и API.
У тебя есть инструмент run_python: выполни Python и получи stdout/stderr.
Каждый вызов — новый процесс: переменные не сохраняются, файлы сохраняются.
Рабочая папка — каталог task.md. Фактические пути CSV указаны в сообщении.
1. Изучи условие, словарь данных, размеры, типы, пропуски и целевую переменную.
2. Выбери честную валидацию с учётом времени/групп/классов. Зафиксируй seed=42.
Не обучай преобразования на валидации. Не используй id как признак без причины.
3. Обучи простую модель (для таблиц начни с CatBoost), измерь требуемую метрику.
Для категорий CatBoost замени пропуски строкой и приведи значения к строкам.
4. Как можно раньше получи предсказания и сохрани submission.csv рядом с task.md.
Соблюдай колонки sample_submission и порядок test. Шаблон не содержит ответов.
5. Если осталось время, проверь несколько обоснованных гипотез на том же разбиении.
Печатай гипотезу, метрику и время. Сохраняй лучший вариант, а не последний.
6. Не переобучай финальную модель вручную: это сделает контроллер.
Перед заменой ответа пиши временный CSV, затем используй os.replace.
Используй только предоставленные данные и установленные библиотеки.
Не устанавливай пакеты, не обращайся к сети, не читай секреты, не создавай
дочерние процессы и не меняй исходные CSV, task.md или код контроллера.
Не загружай предобученные модели. Ограничь обучение 8 CPU, GPU не доступен.
Не печатай целые таблицы; выводи краткие сводки. Сохраняй модели через joblib
при необходимости. Заверши только когда файл результата действительно готов.
"""

TOOLS = [{"type": "function", "function": {
    "name": "run_python",
    "description": "Execute Python in the task directory; return stdout, stderr and exit_code.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python source code"}
    }, "required": ["code"], "additionalProperties": False},
}}]


def tool_schema(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}}


TOOLS += [
    tool_schema("inspect_data", "Get dataset profile and dictionary", {}),
    tool_schema("configure_task", "Freeze task and validation split, once per run", {
        "target": {"type": "string"},
        "task_type": {"type": "string", "enum": ["regression", "classification"]},
        "metric": {"type": "string", "enum": ["rmse", "mae", "f1_macro", "accuracy"]},
        "split": {"type": "string", "enum": ["random", "time", "group"]},
        "split_column": {"type": "string"}}, ["target", "task_type", "metric", "split"]),
    tool_schema("train_model", "Run and score a model on the frozen split; automatically save winner", {
        "hypothesis": {"type": "string"},
        "algorithm": {"type": "string", "enum": ["catboost", "lightgbm"]},
        "params": {"type": "object", "properties": {
            "iterations": {"type": "integer", "minimum": 10, "maximum": 2000},
            "depth": {"type": "integer", "minimum": 2, "maximum": 10},
            "learning_rate": {"type": "number"}, "l2_leaf_reg": {"type": "number"},
            "random_strength": {"type": "number"},
            "auto_class_weights": {"type": "string", "enum": ["Balanced", "SqrtBalanced"]}
        }, "additionalProperties": False}}, ["hypothesis", "algorithm", "params"]),
]


def research_state(log_dir):
    path = log_dir / "research" / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"experiments": [], "best": None}


def run_research(action, arguments, folder, files, log_dir, step, timeout):
    request = {"root": str(log_dir / "research"), "folder": str(folder),
               "files": {k: str(v) for k, v in files.items()}, "action": action, "arguments": arguments}
    request_path = log_dir / f"request_{step}.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    module = str(Path(__file__).resolve().with_name("research.py"))
    code = f"import runpy, sys\nsys.argv = [{module!r}, {str(request_path)!r}]\nrunpy.run_path({module!r}, run_name='__main__')"
    result = run_python(code, folder, log_dir, step, timeout)
    with (log_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"action": action, "arguments": arguments, "result": result}, ensure_ascii=False) + "\n")
    return result


def discover_files(folder):
    files = {}
    for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        direct = folder / name
        matches = [direct] if direct.is_file() else list(folder.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} in {folder}; found {len(matches)}")
        files[name] = matches[0]
    return files


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def validate_submission(folder, files):
    """Structural validation for the provided single-target tabular tasks.

    This does not measure predictive quality. Extend for other output contracts.
    """
    path = folder / "submission.csv"
    if not path.is_file():
        return False, "submission.csv is missing"
    try:
        columns, rows = read_csv(path)
        expected, _ = read_csv(files["sample_submission.csv"])
        test_columns, test_rows = read_csv(files["test.csv"])
        if columns != expected or len(rows) != len(test_rows):
            return False, "Columns or row count do not match sample/test"
        ids = [column for column in expected if column in test_columns]
        targets = [column for column in expected if column not in test_columns]
        if not ids or not targets:
            return False, "Cannot infer ID/prediction columns; extend validator for this task"
        keys = [tuple(row[c] for c in ids) for row in rows]
        if keys != [tuple(row[c] for c in ids) for row in test_rows] or len(set(keys)) != len(keys):
            return False, "IDs are duplicated or do not follow test order"
        _, train = read_csv(files["train.csv"])
        for target in targets:
            observed = {row[target] for row in train if row.get(target, "").strip()}
            try:
                for value in observed:
                    float(value)
                numeric = True
            except ValueError:
                numeric = False
            for row in rows:
                value = row.get(target, "")
                if value is None or not value.strip():
                    return False, f"Missing prediction in {target}"
                if numeric:
                    if not math.isfinite(float(value)):
                        return False, f"Non-finite prediction in {target}"
                elif value not in observed:
                    return False, f"Unknown class in {target}: {value}"
        return True, f"Validated {len(rows)} predictions (format only, not quality)"
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        return False, f"Invalid submission: {error}"


def run_python(code, folder, log_dir, step, timeout):
    try:
        compile(code, "<experiment>", "exec")
    except SyntaxError as error:
        return {"exit_code": 1, "error": f"Syntax error on line {error.lineno}: {error.msg}"}
    script = log_dir / f"step_{step:03d}.py"
    script.write_text(code, encoding="utf-8")
    # Do not pass the API credentials to generated code. This is not isolation.
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    stdout_path, stderr_path = script.with_suffix(".stdout.txt"), script.with_suffix(".stderr.txt")
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        try:
            result = subprocess.run([sys.executable, "-u", str(script)], cwd=folder,
                                    env=env, stdout=out, stderr=err, timeout=timeout)
            status = {"exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            status = {"exit_code": -1, "error": f"Step exceeded {timeout:.0f} seconds"}
    for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            status[name] = stream.read().decode("utf-8", errors="replace")
    status["seconds"] = round(time.monotonic() - started, 2)
    return status


def run_agent(args, folder, files):
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise ValueError("Set OPENAI_API_KEY and OPENAI_BASE_URL in the environment")
    from openai import OpenAI

    log_dir = folder / "agent_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True)
    # An old result must not be mistaken for success of this run.
    previous = folder / "submission.csv"
    if previous.exists():
        previous.replace(log_dir / "previous_submission.csv")
    task = args.task.resolve().read_text(encoding="utf-8-sig")
    paths = {name: str(path.relative_to(folder)) for name, path in files.items()}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task + "\nФактические пути файлов:\n" +
                 json.dumps(paths, ensure_ascii=False) +
                 f"\nБюджет запуска: {args.time_budget} секунд, {args.max_steps} шагов."}]
    deadline = time.monotonic() + args.time_budget
    reserve = min(args.final_reserve, args.time_budget * .3)
    research_deadline = deadline - reserve
    with OpenAI(max_retries=0) as client:
        for step in range(1, args.max_steps + 1):
            remaining = research_deadline - time.monotonic()
            if remaining <= 5:
                break
            print(f"[{step}/{args.max_steps}] Qwen; remaining {remaining:.0f}s", flush=True)
            state = research_state(log_dir)
            messages.append({"role": "user", "content":
                f"До конца исследования {int(remaining)} секунд; финальное обучение выполнит контроллер. "
                f"Успешных экспериментов: {len(state['experiments'])}. "
                f"Доступность LightGBM: {bool(importlib.util.find_spec('lightgbm'))}."})
            try:
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b"),
                    messages=messages, tools=TOOLS, parallel_tool_calls=False,
                    timeout=min(90, remaining), max_tokens=8000,
                )
            except Exception as error:
                # SDK error strings may contain endpoint details: avoid logging credentials.
                print(f"API request failed: {type(error).__name__}. Checking saved output.", flush=True)
                break
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            (log_dir / "conversation.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            if message.content:
                print(message.content, flush=True)
            if not message.tool_calls:
                state = research_state(log_dir)
                if state["best"] and len(state["experiments"]) >= args.min_experiments:
                    break
                messages.append({"role": "user", "content":
                    f"Продолжи исследование через инструменты. Нужно хотя бы {args.min_experiments} "
                    "успешных эксперимента, если позволяет время. Не повторяй одинаковые параметры."})
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                    remaining = research_deadline - time.monotonic()
                    if remaining <= 1:
                        result = {"error": "Total time budget exhausted"}
                    elif call.function.name in {"inspect_data", "configure_task", "train_model"}:
                        action = {"inspect_data": "inspect", "configure_task": "configure", "train_model": "train"}[call.function.name]
                        result = run_research(action, arguments, folder, files, log_dir, step,
                                              min(args.step_timeout, remaining))
                    elif call.function.name == "run_python" and isinstance(arguments.get("code"), str):
                        result = run_python(arguments["code"], folder, log_dir, step,
                                            min(args.step_timeout, remaining))
                    else:
                        raise ValueError("Unknown tool or invalid arguments")
                except (ValueError, TypeError, OSError) as error:
                    result = {"error": str(error)}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                result["remaining_seconds"] = max(0, int(research_deadline - time.monotonic()))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            (log_dir / "conversation.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    state = research_state(log_dir)
    remaining = deadline - time.monotonic()
    if state["best"] and remaining > 5:
        print("Final refit of the best recorded experiment", flush=True)
        result = run_research("finalize", {}, folder, files, log_dir, args.max_steps + 1, remaining - 2)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if not state["best"]:
        raise RuntimeError(f"No trusted experiment completed. Inspect logs: {log_dir}")
    valid, detail = validate_submission(folder, files)
    if not valid:
        raise RuntimeError(f"Agent stopped without a valid output: {detail}. Logs: {log_dir}")
    print(detail)


def main():
    parser = argparse.ArgumentParser(description="Qwen ML agent")
    parser.add_argument("task", type=Path)
    parser.add_argument("--check", action="store_true", help="Check inputs without API or training")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--time-budget", type=int, default=1800)
    parser.add_argument("--step-timeout", type=int, default=600)
    parser.add_argument("--final-reserve", type=int, default=450,
                        help="Seconds reserved for final refit (at most 30 percent of total budget)")
    parser.add_argument("--min-experiments", type=int, default=4)
    args = parser.parse_args()
    if min(args.max_steps, args.time_budget, args.step_timeout, args.final_reserve, args.min_experiments) <= 0:
        parser.error("Limits must be positive")
    if not args.task.is_file():
        parser.error(f"Task file does not exist: {args.task}")
    folder = args.task.resolve().parent
    files = discover_files(folder)
    if args.check:
        print(f"Python: {sys.version.split()[0]} (platform requires 3.12)")
        for name, path in files.items():
            columns, rows = read_csv(path)
            print(f"{name}: {len(rows)} rows, {len(columns)} columns; {path}")
        for name in ("openai", "pandas", "numpy", "sklearn", "catboost"):
            print(f"{name}: {'available' if importlib.util.find_spec(name) else 'NOT INSTALLED'}")
        print("No API calls made. No models trained.")
        return
    run_agent(args, folder, files)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
