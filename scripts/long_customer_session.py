"""Continue a completed customer journey on an isolated snapshot of its database."""
import argparse
import asyncio
import getpass
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

import httpx
from live_model_check import ROOT, gateway_settings
from pydantic_ai.exceptions import ModelHTTPError

from business_match.app import create_app

# Additional suppliers give the buyer alternatives beyond the four from the short session.
EXTRA = {
    "flour": [
        ("Камский помол", "Мука пшеничная для хлеба в мешках по 50 кг"),
        ("Хлебная нива", "Мука для подового хлеба с доставкой дважды в неделю"),
        ("Первый сорт", "Мука хлебопекарная первого сорта с пробной партией"),
        ("Мельничный двор", "Пшеничная мука оптом с помолом под заказ"),
        ("Алтайский каравай", "Сильная хлебопекарная мука для долгой ферментации"),
        ("Утренний помол", "Мука для булочек и багетов в мешках по 25 кг"),
    ],
    "paper": [
        ("Окно для багета", "Длинные бумажные пакеты с прозрачным окном для багетов"),
        ("Пергамент", "Жиростойкие бумажные пакеты для круассанов"),
        ("Малая серия", "Бумажные пакеты для хлеба с логотипом от 500 штук"),
        ("Пакетная фабрика", "Пакеты для выпечки из бурого крафта оптом"),
        ("Бумага и хлеб", "Перфорированные бумажные пакеты для горячих булок"),
        ("Белый лист", "Белые бумажные пакеты для нарезного хлеба"),
    ],
}


async def main(previous):
    output = ROOT / "local_experiments" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f-long")
    output.mkdir(parents=True)
    db = output / "market.db"
    with sqlite3.connect(previous.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(db) as dest:
            src.backup(dest)
    key = os.getenv("LLM_API_KEY") or getpass.getpass("LLM API key: ")
    settings = gateway_settings(db, key)
    settings.ranker_graph_weight = 0
    app = create_app(settings)
    fixture = json.loads((ROOT / "fixtures/customer_journeys.json").read_text())
    relevant = {s["stage"]: set(s["relevant"]) for s in fixture["journeys"][:2]}
    trace = {"source": str(previous), "new_cards": [], "events": [], "slates": [], "checks": []}

    def save():
        (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as api:
        for family, offers in EXTRA.items():
            for index, (name, product) in enumerate(offers):
                cid = f"long-{family}-{index}"
                card = dict(id=cid, name=name, description=product,
                            segment="Мука оптом" if family == "flour" else "Бумажная упаковка",
                            products=[dict(id="main", name=product, stock=100)])
                for attempt in range(3):
                    try:
                        (await api.put(f"/v1/clients/{cid}", json=card)).raise_for_status()
                        break
                    except ModelHTTPError as error:
                        if error.status_code != 429 or attempt == 2:
                            raise
                        await asyncio.sleep(61)
                relevant[family].add(cid)
                trace["new_cards"].append(card)
                save()
                print(name, flush=True)

        previous_call = 0

        async def slate(query=None, client="bakery"):
            nonlocal previous_call
            body = dict(client_id=client, intent="supplier", limit=20, seed=42)
            if query:
                body["query"] = query
            for attempt in range(2):
                await asyncio.sleep(max(0, 2.5 - (monotonic() - previous_call)))
                previous_call = monotonic()
                resp = await api.post("/v1/search" if query else "/v1/feed", json=body)
                resp.raise_for_status()
                data = resp.json()
                trace["slates"].append(dict(body=body, weight=settings.ranker_personal_weight,
                                            result=data))
                save()
                if data["diagnostics"]["reranked"]:
                    return data
                if not attempt:
                    await asyncio.sleep(61)
            raise RuntimeError("BGE unavailable for paired comparison")

        async def checkpoint(label, family):
            actual = await slate()
            weight = settings.ranker_personal_weight
            settings.ranker_personal_weight = 0
            try:
                control = await slate()
            finally:
                settings.ranker_personal_weight = weight
            row = dict(label=label, family=family, events=len(trace["events"]),
                       observations=actual["diagnostics"]["ranker_observations"],
                       actual=[i["client_id"] for i in actual["items"]],
                       control=[i["client_id"] for i in control["items"]])
            trace["checks"].append(row)
            save()
            print(label, row["events"], row["actual"][:5], flush=True)

        trace["relevant"] = {k: sorted(v) for k, v in relevant.items()}
        control_start = await slate(client="salon")
        await checkpoint("start", "flour")
        visited = set()
        held_out = {"long-flour-4", "long-flour-5", "long-paper-4", "long-paper-5"}
        trace["held_out"] = sorted(held_out)
        for step, (family, query) in enumerate([
            ("flour", "Пшеничная хлебопекарная мука мешками"),
            ("flour", "Мука для багетов оптовая доставка"),
            ("flour", "Сильная мука для хлеба"),
            ("paper", "Бумажная упаковка хлеба оптом"),
            ("paper", "Жиростойкие пакеты для булочек"),
            ("paper", "Крафт пакеты для багетов"),
            ("paper", "Пакеты с логотипом для пекарни"),
            ("flour", "Мука пшеничная оптом"),
        ], 1):
            found = await slate(query)
            eligible = [i["client_id"] for i in found["items"]
                        if i["client_id"] in relevant[family] - held_out]
            targets = sorted(eligible, key=lambda cid: cid in visited)[:3]
            if not targets:
                raise ValueError(f"No matching suppliers for {query}")
            for index, target in enumerate(targets):
                visited.add(target)
                kinds = ["impression", "open", "dwell"]
                if index < 2:
                    kinds.append(["contact", "like"][index])
                for kind in kinds:
                    event = dict(event_id=str(uuid4()), request_id=found["request_id"],
                                 target_client_id=target, client_id="bakery", kind=kind)
                    if kind == "impression":
                        event["visible_fraction"] = 1
                    if kind == "dwell":
                        event["dwell_ms"] = (15 - index * 3) * 1000
                    resp = await api.post("/v1/events", json={"events": [event]})
                    resp.raise_for_status()
                    outcome = resp.json()["results"][0]
                    trace["events"].append(dict(event=event, outcome=outcome, round=step))
                    if outcome["status"] != "accepted":
                        raise RuntimeError(outcome)
            await checkpoint(str(step), family)
        control_end = await slate(client="salon")
        trace["salon_unchanged"] = ([i["client_id"] for i in control_start["items"]]
                                    == [i["client_id"] for i in control_end["items"]])
        save()
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path, help="SQLite from the earlier customer journey")
    asyncio.run(main(parser.parse_args().previous))
