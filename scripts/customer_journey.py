"""Replay supplier discovery, feedback and a change in buying intent using CIB models."""
import asyncio
import getpass
import json
import os
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from live_model_check import ROOT, gateway_settings
from pydantic_ai.exceptions import ModelHTTPError

from business_match.app import create_app


def write_report(output, fixture, trace):
    names = {c["id"]: c["name"] for c in fixture["catalog"]}
    slates = trace["slates"]
    reranked = sum(s["response"]["diagnostics"]["reranked"] for s in slates)
    accepted = sum(e["outcome"]["status"] == "accepted" for e in trace["events"])
    lines = [
        "# Как меняется лента после действий закупщика", "",
        f"В каталоге {len(names)} вымышленных компаний. Пекарня сначала ищет муку, "
        "потом переключается на упаковку. Салон выбирает средства для волос. "
        "Среди альтернатив есть печи для пекарен, курьерские пакеты и шампунь для собак.",
        "", f"Qwen строит профили при индексации, Qwen3-Embedding подбирает кандидатов. "
        f"BGE отработал в {reranked} из {len(slates)} запросов ленты и поиска. "
        f"Сервис принял {accepted} из {len(trace['events'])} событий.",
        "", "В каждом поиске закупщик последовательно открывает двух подходящих поставщиков, "
        "смотрит каждую карточку 12 секунд, первую отмечает лайком, по второй открывает контакт. "
        "Длительность просмотра передаётся синтетическим событием. Затем он возвращается "
        "в обычную ленту поставщиков без поискового запроса.",
        "", "## Что появилось на первых трёх местах", "",
        "В таблице указано число компаний текущей закупочной категории в первой тройке. "
        "Для сравнения после действий повторно запрошена лента того же клиента с нулевым "
        "весом персонального ранкера. История показов и остальные настройки те же.", "",
        "| Закупка | До действий | После действий | После с нулевым весом |",
        "|---|---:|---:|---:|",
    ]
    labels = {"flour": "Пекарня · мука", "paper": "Пекарня · упаковка", "hair": "Салон · шампунь"}
    for stage in trace["stages"]:
        counts = [len(set(stage[k][:3]) & set(stage["relevant"]))
                  for k in ("before", "after", "no_learning")]
        lines.append(f"| {labels[stage['stage']]} | {counts[0]} | {counts[1]} | {counts[2]} |")
    for stage in trace["stages"]:
        lines.extend(["", "## " + labels[stage["stage"]], "",
                      "Запрос «" + stage["query"] + "». Действия по карточкам в порядке поиска — "
                      + ", ".join(names[i] for i in stage["touched"]) + ".", "",
                      "| Обычная лента | Первые три компании |",
                      "|---|---|"])
        for k, title in [("before", "До"), ("after", "После"), ("no_learning", "Нулевой вес")]:
            lines.append(f"| {title} | " + ", ".join(names[i] for i in stage[k][:3]) + " |")
        lines.extend(["", "Позиции подходящих поставщиков, по которым не было действий "
                      "на этом этапе. Прочерк означает отсутствие в возвращённой ленте.", "",
                      "| Компания | До | После | Нулевой вес |", "|---|---:|---:|---:|"])
        for i in stage["relevant"]:
            if i not in stage["touched"]:
                ranks = [str(stage[k].index(i) + 1) if i in stage[k] else "—"
                         for k in ("before", "after", "no_learning")]
                lines.append(f"| {names[i]} | " + " | ".join(ranks) + " |")
        unchanged = stage["control_unchanged"]
        lines.extend(["", "Порядок ленты второго клиента " +
                      ("сохранился." if unchanged else "изменился.")])
    hidden = trace.get("hide")
    if hidden:
        present = hidden["hidden"] in hidden["subsequent"]
        lines.extend(["", "## Скрытие карточки", "", f"Салон скрыл «{names[hidden['hidden']]}». "
                      + ("Компания осталась в ленте." if present else
                         "В следующей ленте этой компании уже не было.")])
    lines.extend(["", "## Условия опыта", "",
                  "Один каталог, два клиента, по четыре заранее размеченных поставщика "
                  "на каждый запрос. Действия заданы сценарием. Глобальный граф, случайное "
                  "исследование и бонус UCB отключены, seed равен 42. Опыт измеряет порядок "
                  "компаний внутри этой синтетической выдачи. Для оценки конверсии в контакты "
                  "нужны действия настоящих пользователей.", ""])
    (output / "report.md").write_text("\n".join(lines))


async def main():
    fixture = json.loads((ROOT / "fixtures/b2b_market.json").read_text())
    scenarios = json.loads((ROOT / "fixtures/customer_journeys.json").read_text())
    fixture["catalog"].extend(scenarios["additional_companies"])
    fixture["journeys"] = scenarios["journeys"]
    output = ROOT / "local_experiments" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f-journey")
    output.mkdir(parents=True)
    key = os.getenv("LLM_API_KEY") or getpass.getpass("LLM API key: ")
    # Only this cache is shared across runs. Events are written to a fresh experiment copy.
    cache = output.parent / "journey-catalog.db"
    settings = gateway_settings(cache, key)
    settings.ranker_graph_weight = 0
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as api:
        for card in fixture["catalog"]:
            for attempt in range(3):
                try:
                    response = await api.put(f"/v1/clients/{card['id']}", json=card)
                    response.raise_for_status()
                    break
                except ModelHTTPError as error:
                    if error.status_code != 429 or attempt == 2:
                        raise
                    await asyncio.sleep(61)
            print("Indexed", card["id"], flush=True)

    db = output / "market.db"
    with sqlite3.connect(cache) as source, sqlite3.connect(db) as dest:
        source.backup(dest)
    settings.database_url = f"sqlite:///{db}"
    app = create_app(settings)
    trace = {"slates": [], "events": [], "stages": []}

    def save():
        (output / "journey.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as api:
        async def slate(client, label, query=None):
            body = dict(client_id=client, intent="supplier", seed=42, limit=10)
            if query:
                body["query"] = query
            response = await api.post("/v1/search" if query else "/v1/feed", json=body)
            response.raise_for_status()
            result = response.json()
            trace["slates"].append(dict(label=label, body=body, response=result))
            save()
            return result

        def ids(result):
            return [row["client_id"] for row in result["items"]]

        async def action(client, result, target, kind, **extra):
            event = dict(client_id=client, request_id=result["request_id"], event_id=str(uuid4()),
                         target_client_id=target, kind=kind, **extra)
            response = await api.post("/v1/events", json={"events": [event]})
            response.raise_for_status()
            outcome = response.json()["results"][0]
            trace["events"].append(dict(event=event, outcome=outcome))
            save()
            if outcome["status"] != "accepted":
                raise ValueError(outcome)

        for journey in fixture["journeys"]:
            client, label = journey["client"], journey["stage"]
            control = "salon" if client == "bakery" else "bakery"
            control_before = ids(await slate(control, label + "-control-before"))
            before = ids(await slate(client, label + "-before"))
            found = await slate(client, label + "-search", journey["query"])
            targets = [i for i in ids(found) if i in journey["relevant"]][:2]
            for index, target in enumerate(targets):
                await action(client, found, target, "impression", visible_fraction=1)
                await action(client, found, target, "open")
                await action(client, found, target, "dwell", dwell_ms=12000)
                await action(client, found, target, "like" if index == 0 else "contact")
            last = await slate(client, label + "-after")
            after = ids(last)
            weight = settings.ranker_personal_weight
            settings.ranker_personal_weight = 0
            try:
                no_learning = ids(await slate(client, label + "-no-personal-weight"))
            finally:
                settings.ranker_personal_weight = weight
            control_after = ids(await slate(control, label + "-control-after"))
            trace["stages"].append(dict(**journey, before=before, after=after, touched=targets,
                                        no_learning=no_learning,
                                        control_unchanged=control_before == control_after))
            save()
        if last["items"]:
            hidden = ids(last)[0]
            await action(client, last, hidden, "hide")
            trace["hide"] = dict(hidden=hidden, subsequent=ids(await slate(client, "after-hide")))
            save()
    write_report(output, fixture, trace)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
