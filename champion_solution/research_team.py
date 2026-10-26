"""Independent proposal branches; the parent schedules isolated training jobs."""
import json
import time
from threading import Semaphore
from types import SimpleNamespace


class ResearchTeamUnavailable(RuntimeError):
    def __init__(self, failures):
        self.retryable = all(f['retryable'] for f in failures)
        super().__init__('; '.join(f["role"] + ': ' + f["type"] for f in failures))


class EmptyResearchResponse(RuntimeError):
    """Empty API messages do not indicate successful research completion."""


def validate_response(response):
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    if message is None or (not message.tool_calls and
            (not (message.content or '').strip() or getattr(choice, 'finish_reason', None) == 'length')):
        raise EmptyResearchResponse('Empty or truncated response without a tool call')
    return message


def transient_error(error):
    return (type(error).__name__ in {'EmptyResearchResponse', 'APITimeoutError', 'APIConnectionError', 'TimeoutError',
                                    'InternalServerError', 'RateLimitError'}
            or getattr(error, 'status_code', None) in {408, 429, 500, 502, 503, 504})


ROLES = {
    "features": "Ты исследователь признаков. Приоритет: анализ данных, преобразования, временная история. "
                "Не меняй одновременно признаки и гиперпараметры. Предложи максимум один инструмент за раунд.",
    "models": "Ты исследователь моделей и оценки. Приоритет: OOF, ансамбли, сдвиг train/test, Macro F1. "
              "Предложи максимум один инструмент за раунд. Ты отвечаешь за select_final_candidate после сравнения.",
    "rare": "Ты исследователь редких классов. Приоритет: веса классов, ошибки Backdoor/Analysis/DoS, "
            "Macro F1. Используй только поддерживаемые параметры инструментов. "
            "Предложи максимум один инструмент за раунд; проверку OOF и финальный выбор ведёт models."
}


class TeamMessage:
    def __init__(self, content, tool_calls):
        self.content, self.tool_calls = content, tool_calls

    def model_dump(self, **kwargs):
        result = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [dict(id=c.id, type="function", function=dict(name=c.function.name,
                                                                                 arguments=c.function.arguments)) for c in self.tool_calls]
        return result


def propose_round(client, pool, model, system, task, guidance, snapshot, recent, tools, remaining,
                  round_id, log_dir, usage_callback, researchers=2, sequential=False, api_timeout=180):
    roles = list(ROLES)[:researchers]
    deadline = time.monotonic() + remaining
    gate = Semaphore(1 if sequential else len(roles))
    allowed = [t for t in tools if t["function"]["name"] not in {"run_python", "configure_task"}]
    def request(role):
        role_tools = [t for t in allowed if role == "models" or t["function"]["name"] != "select_final_candidate"]
        messages = [{"role": "system", "content": system + "\n" + ROLES[role] +
            "\nТы работаешь параллельно с другими направлениями. Все видят один снимок. "
            "Обучения выполняет очередь контроллера; результаты увидишь в следующем снимке. "
            "Файлы напрямую не меняй. Не повторяй выполненные опыты. Все пояснения по-русски. "
            "Для train_<алгоритм> всегда передавай feature_code (пустой для сырых признаков) или base_experiment_id. "
            "В треке features обязательны base_experiment_id и полный изменённый feature_code. "
            "Не объявляй результат улучшенным до сравнения на одинаковых фолдах. "
            "Трек models сначала дождись результатов новых рецептов других направлений, затем проведи OOF-сравнение."},
            {"role": "user", "content": json.dumps(dict(task=task, instructions=guidance,
                snapshot=snapshot, recent_events=recent, remaining_seconds=int(remaining)), ensure_ascii=False)}]
        with gate:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('Research deadline exhausted')
            response = client.chat.completions.create(model=model, messages=messages, tools=role_tools,
                parallel_tool_calls=False, timeout=min(api_timeout, left), max_tokens=8000)
        usage_callback(response, log_dir, "researcher_" + role)
        return validate_response(response)
    futures = {role: pool.submit(request, role) for role in roles}
    calls, comments, seen = [], [], set()
    failures = []
    for role, future in futures.items():
        try:
            message = future.result()
        except Exception as error:
            failures.append(dict(role=role, type=type(error).__name__, retryable=transient_error(error)))
            comments.append(f"{role}: API unavailable ({type(error).__name__})")
            continue
        comments.append(f"{role}: {message.content or ''}")
        for call in (message.tool_calls or [])[:1]:
            try:
                args = json.loads(call.function.arguments)
                # Hypothesis prose is not part of an experiment's identity.
                key = (call.function.name, json.dumps({k: v for k, v in args.items() if k != "hypothesis"}, sort_keys=True))
            except (ValueError, AttributeError):
                key = (call.function.name, call.function.arguments)
            if key in seen:
                continue
            seen.add(key)
            calls.append(SimpleNamespace(id=f"team_{round_id}_{role}", function=SimpleNamespace(
                name=call.function.name, arguments=call.function.arguments)))
    with (log_dir / "research_team.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(round=round_id, comments=comments, failures=failures, sequential=sequential,
            queue=[dict(id=c.id, action=c.function.name, arguments=c.function.arguments) for c in calls]), ensure_ascii=False) + "\n")
    if len(failures) == len(roles) or (failures and not calls):
        raise ResearchTeamUnavailable(failures)
    return TeamMessage("\n".join(comments), calls)
