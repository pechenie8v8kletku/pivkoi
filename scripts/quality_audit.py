"""Reproducible quality audit, with independent topic labels and visible failures.

No real Qwen/BGE: scripted extractions, hashing embeddings and optional hash reranker.
Every generated dataset, individual result and metric stays under local_experiments.
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from business_context_check import ScriptedGateway

from business_match.app import create_app
from business_match.config import Settings
from business_match.engine import Engine
from business_match.mock_api import mock_embedding
from business_match.providers import Providers
from business_match.schemas import utcnow
from business_match.storage import Repository

# Ground truth is business purpose, not similarity to an embedding or extractor output.
TOPICS = {
    "flour": (
        "Мука пшеничная хлебопекарная",
        ["Мука высший сорт для выпечки", "Пшеничная мука оптом", "Мука хлебопекарная в мешках"],
    ),
    "oven": (
        "Печи хлебопекарные",
        [
            "Печь конвекционная для пекарни",
            "Ротационная печь для хлеба",
            "Пекарские печи электрические",
        ],
    ),
    "pack": (
        "Упаковка пищевая",
        ["Бумажные пакеты для хлеба", "Коробки для выпечки", "Упаковка пищевая картонная"],
    ),
    "coffee": ("Кофе зерновой", ["Зерновой кофе арабика", "Кофе для кофейни", "Кофе оптом"]),
    "cold": (
        "Холодильные витрины",
        [
            "Витрина холодильная торговая",
            "Холодильные шкафы для магазина",
            "Холодильные витрины для продуктов",
        ],
    ),
    "metal": ("Металл стальной", ["Стальной лист", "Стальная арматура", "Металлопрокат стальной"]),
    "logistics": (
        "Перевозка грузов",
        [
            "Грузоперевозки автомобильные",
            "Перевозка грузов по России",
            "Транспортная доставка грузов",
        ],
    ),
    "cosmetics": (
        "Косметика для волос",
        ["Шампуни профессиональные", "Краска для волос салонная", "Косметика для парикмахерской"],
    ),
    "dental": (
        "Стоматологические материалы",
        ["Пломбировочные материалы", "Композиты стоматологические", "Расходники для стоматологии"],
    ),
    "wood": (
        "Древесина мебельная",
        ["Мебельный щит сосна", "Фанера для мебели", "Доска для мебели"],
    ),
    "fabric": ("Ткани для одежды", ["Хлопковая ткань", "Ткани швейные оптом", "Ткань трикотажная"]),
    "fertilizer": (
        "Удобрения сельскохозяйственные",
        ["Азотные удобрения", "Удобрения для фермеров", "Минеральные удобрения"],
    ),
    "tires": (
        "Шины автомобильные",
        ["Шины для легковых автомобилей", "Автомобильные покрышки", "Шины грузовые"],
    ),
    "cleaning": (
        "Уборка офисов",
        ["Клининг офисных помещений", "Профессиональная уборка", "Уборка коммерческих помещений"],
    ),
    "software": (
        "Программное обеспечение",
        ["CRM для бизнеса", "Разработка программного обеспечения", "Система складского учета"],
    ),
    "legal": (
        "Юридические услуги",
        [
            "Юридическое сопровождение бизнеса",
            "Юридическая проверка договора",
            "Юрист для организаций",
        ],
    ),
    "fuel": ("Дизельное топливо", ["Дизтопливо оптом", "Топливо для техники", "Дизель летний"]),
    "flowers": ("Цветы срезанные", ["Розы оптом", "Тюльпаны срезанные", "Цветы для флористов"]),
    "feed": (
        "Корм для животных",
        ["Комбикорм для птицы", "Корм для сельхозживотных", "Кормовая мука рыбная"],
    ),
    "sauna": ("Печи банные", ["Печь для сауны", "Дровяная печь для бани", "Печь банная"]),
    "industrial": (
        "Печи промышленные",
        ["Печь для обжига керамики", "Печь плавильная", "Термическая печь для металла"],
    ),
    "stone": (
        "Мука известняковая",
        ["Доломитовая мука", "Известняковая мука для почвы", "Минеральная мука"],
    ),
    "petcosmetics": (
        "Косметика для животных",
        ["Шампунь для собак", "Косметика для груминга", "Шампунь для кошек"],
    ),
    "bread": ("Хлеб и выпечка", ["Хлеб пшеничный", "Булочки свежие", "Хлеб и выпечка оптом"]),
}
PERSONAS = {
    "bakery": ("Пекарня", "bread", ["pack"], ["flour", "oven"]),
    "cafe": ("Кофейня", "bread", ["coffee", "pack"], ["cold"]),
    "salon": ("Парикмахерская", None, ["cosmetics"], ["cleaning"]),
    "dentist": ("Стоматологическая клиника", None, ["dental"], ["cleaning"]),
    "furniture": ("Мебельная фабрика", None, ["wood"], ["metal"]),
    "farm": ("Фермерское хозяйство", None, ["fertilizer", "feed"], ["fuel"]),
    "atelier": ("Швейная мастерская", None, ["fabric"], ["pack"]),
    "garage": ("Автосервис", None, ["tires"], ["metal"]),
}


class AuditGateway(ScriptedGateway):
    def respond(self, request):
        if request.url.path.endswith("/rerank"):
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "index": i,
                            "relevance_score": float(
                                np.dot(mock_embedding(p["query"]), mock_embedding(p["document"]))
                            ),
                        }
                        for i, p in enumerate(body["pairs"])
                    ]
                },
            )
        return super().respond(request)


async def run(output, per_topic=8):
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(
        _env_file=None,
        model_api_mode="mock",
        database_url=f"sqlite:///{output / 'audit.db'}",
        llm_rpm=100000,
        embedding_rpm=100000,
        embedding_tpm=100000000,
        rerank_rpm=100000,
        rerank_tpm=100000000,
        ranker_graph_weight=0,
        ranker_freshness_weight=0,
    )
    repo = Repository(settings.database_url)
    repo.initialize()
    provider = Providers(settings, repo)
    await provider.http.aclose()
    gateway = AuditGateway()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(gateway.respond))
    catalog, labels = [], {}

    def add(payload, offers, needs, peer):
        catalog.append(payload)
        labels[payload["id"]] = {"offers": set(offers), "needs": set(needs), "peer": peer}

    def context(client, topics, wrong=False):
        records = []
        for i, topic in enumerate(topics):
            raw = f"PRIVATE-{client}-{i}: оплата поставки {TOPICS[topic][0]}"
            gateway.answers[raw] = {
                "subject": TOPICS["cosmetics" if wrong else topic][0],
                "role": "buyer",
                "commercial": True,
                "confidence": 0.92,
            }
            records.append(
                {
                    "id": f"line-{i}",
                    "kind": "contract",
                    "role": "buyer",
                    "subject": raw,
                    "purpose": "trade",
                    "status": "active",
                    "occurred_at": (utcnow() - timedelta(days=3 + i * 11)).isoformat(),
                }
            )
        return {"source_id": "audit", "source_sequence": 1, "records": records}

    for topic, (title, variants) in TOPICS.items():
        for i in range(per_topic):
            name = variants[i % len(variants)]
            products = [{"id": topic, "name": name, "category": title, "stock": 20}]
            # Multi-category catalog: a relevant company can still show irrelevant products.
            offers = {topic}
            if i % 4 == 0 and topic != "software":
                products.append({"id": "software", "name": "CRM для бизнеса", "stock": 10})
                offers.add("software")
            add(
                {
                    "id": f"{topic}-{i}",
                    "name": f"{title} — компания {i}",
                    "description": name,
                    "segment": title,
                    "products": products,
                    "region": "Москва" if i % 2 else "Казань",
                    "explicit_needs": ["Перевозка грузов"],
                },
                offers,
                {"logistics"},
                topic,
            )
    for key, (name, offer, needs, observed) in PERSONAS.items():
        for suffix in ("", "-control"):
            client = key + suffix
            add(
                {
                    "id": client,
                    "name": name + suffix,
                    "description": name,
                    "segment": name,
                    "explicit_needs": [TOPICS[t][0] for t in needs],
                    "products": (
                        [{"id": offer, "name": TOPICS[offer][0], "stock": 10}] if offer else []
                    ),
                    "business_context": context(client, observed),
                },
                {offer} if offer else set(),
                set(needs + observed),
                key,
            )
    add(
        {
            "id": "wrong-extraction",
            "name": "Пекарня с ошибкой извлечения",
            "description": "Хлеб и выпечка",
            "segment": "Пекарня",
            "explicit_needs": ["Упаковка пищевая"],
            "business_context": context("wrong-extraction", ["flour"], wrong=True),
        },
        {"bread"},
        {"flour", "pack"},
        "bakery",
    )
    # Same keywords but explicitly unavailable, and a deleted company.
    add(
        {
            "id": "out-of-stock",
            "name": "Мука без остатка",
            "description": "Мука пшеничная",
            "products": [{"id": "flour", "name": "Мука пшеничная", "stock": 0}],
        },
        {"flour"},
        set(),
        "flour",
    )
    add(
        {
            "id": "inactive",
            "name": "Закрытая мельница",
            "description": "Мука пшеничная",
            "active": False,
        },
        {"flour"},
        set(),
        "flour",
    )
    cases = []
    for key in PERSONAS:
        for intent in ("supplier", None):
            cases.append(
                {
                    "id": f"feed_{key}_{intent or 'mixed'}",
                    "client_id": key,
                    "intent": intent,
                    "kind": "feed",
                }
            )
    searches = [
        ("flour_exact", "Мука пшеничная хлебопекарная", {"flour"}),
        ("flour_short", "мука", {"flour", "stone", "feed"}),
        ("flour_paraphrase", "сырье для замеса хлебного теста", {"flour"}),
        ("oven_exact", "Печи хлебопекарные", {"oven"}),
        ("oven_paraphrase", "оборудование чтобы выпекать батоны", {"oven"}),
        ("pack", "Бумажные пакеты для хлеба", {"pack"}),
        ("coffee", "Кофе зерновой", {"coffee"}),
        ("cosmetics", "Шампуни профессиональные для парикмахерской", {"cosmetics"}),
        ("dental", "Стоматологические материалы", {"dental"}),
        ("wood", "Мебельный щит сосна", {"wood"}),
        ("fuel", "Дизельное топливо", {"fuel"}),
        ("legal", "Юридическая проверка договора", {"legal"}),
        ("typo", "мука пшеничная хлебопекарнная", {"flour"}),
        ("no_match", "спутниковые двигатели для космических аппаратов", set()),
    ]
    for name, query, truth in searches:
        for intent in ("supplier", None):
            cases.append(
                {
                    "id": f"search_{name}_{intent or 'mixed'}",
                    "client_id": "bakery",
                    "intent": intent,
                    "query": query,
                    "truth": truth,
                    "kind": "search",
                }
            )
    cases.extend(
        [
            {"id": "buyer_flour", "client_id": "flour-1", "intent": "buyer", "kind": "buyer"},
            {
                "id": "wrong_extraction",
                "client_id": "wrong-extraction",
                "intent": "supplier",
                "kind": "adversarial",
            },
            {
                "id": "regional_search",
                "client_id": "bakery",
                "intent": "supplier",
                "query": "Мука пшеничная",
                "truth": {"flour"},
                "region": "Москва",
                "kind": "search",
            },
        ]
    )
    rows, responses, checks = [], [], {}

    def grade(case, item):
        source, target = labels[case["client_id"]], labels[item["client_id"]]
        role = item["intent"]
        if case.get("query"):
            # A plain product search asks for a supplier unless the caller says otherwise.
            return role == "supplier" and bool(target["offers"] & case["truth"])
        if role == "supplier":
            return bool(source["needs"] & target["offers"])
        if role == "buyer":
            return bool(source["offers"] & target["needs"])
        if role == "peer":
            return source["peer"] == target["peer"] or bool(source["offers"] & target["offers"])
        return False  # No labeled partnerships in this dataset.

    def metrics(mode, case, seed, data, elapsed):
        items = data["items"][:10]
        good = [grade(case, x) for x in items]
        wanted = case.get("truth", labels[case["client_id"]]["needs"])
        relevant_products = [
            p["product"]["id"] in wanted
            for x in items
            if x["intent"] == "supplier"
            for p in x["products"]
        ]
        row = {
            "mode": mode,
            "case": case["id"],
            "kind": case["kind"],
            "seed": seed,
            "returned": len(items),
            "p10": sum(good) / 10,
            "precision_returned": sum(good) / len(items) if items else None,
            "irrelevant": len(items) - sum(good),
            "product_precision": (
                sum(relevant_products) / len(relevant_products) if relevant_products else None
            ),
            "products_shown": len(relevant_products),
            "irrelevant_products": len(relevant_products) - sum(relevant_products),
            "segments": len({x["segment"] for x in items}),
            "roles": dict(Counter(x["intent"] for x in items)),
            "latency_ms": elapsed,
            "bad": [
                {
                    "id": x["client_id"],
                    "name": x["name"],
                    "role": x["intent"],
                    "match": x["matched_text"],
                }
                for x, ok in zip(items, good, strict=True)
                if not ok
            ],
        }
        rows.append(row)
        responses.append({"mode": mode, "case": case["id"], "seed": seed, "response": data})
        return row

    app = create_app(settings, providers=provider)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://service"
            ) as api:
                for payload in catalog:
                    (await api.put("/v1/clients/" + payload["id"], json=payload)).raise_for_status()
                for mode, enabled, rerank in [
                    ("off", False, True),
                    ("on", True, True),
                    ("on_no_rerank", True, False),
                ]:
                    settings.business_signals_enabled = enabled
                    settings.rerank_enabled = rerank
                    app.state.engine = Engine(app.state.repository, provider, settings)
                    for case in cases:
                        for seed in (11, 42, 79):
                            body = {
                                k: v
                                for k, v in case.items()
                                if k in ("client_id", "intent", "query", "region")
                            }
                            start = time.perf_counter()
                            response = await api.post(
                                "/v1/search" if case.get("query") else "/v1/feed",
                                json={**body, "limit": 10, "seed": seed},
                            )
                            response.raise_for_status()
                            data = response.json()
                            metrics(mode, case, seed, data, (time.perf_counter() - start) * 1000)
                            ids = [x["client_id"] for x in data["items"]]
                            assert len(ids) == len(set(ids)) and "inactive" not in ids
                            assert "PRIVATE-" not in response.text
                            if case.get("region"):
                                assert all(x["region"] == case["region"] for x in data["items"])
                checks["all_http_requests_succeeded"] = True
                checks["unique_active_private_safe_filtered"] = True
                # Real feedback sequence, then explicit new search; inspect personal spillover.
                settings.business_signals_enabled = True
                settings.rerank_enabled = True
                app.state.engine = Engine(app.state.repository, provider, settings)
                journey = []
                for topic, query in [
                    ("oven", "Печи хлебопекарные"),
                    ("pack", "Бумажные пакеты для хлеба"),
                    ("flour", "Мука пшеничная хлебопекарная"),
                ]:
                    response = await api.post(
                        "/v1/search",
                        json={
                            "client_id": "bakery",
                            "intent": "supplier",
                            "query": query,
                            "limit": 10,
                            "seed": 42,
                        },
                    )
                    response.raise_for_status()
                    slate = response.json()
                    targets = [
                        x for x in slate["items"] if topic in labels[x["client_id"]]["offers"]
                    ][:3]
                    for item in targets:
                        for kind in ("impression", "open", "like", "contact"):
                            event = {
                                "event_id": str(uuid4()),
                                "request_id": slate["request_id"],
                                "client_id": "bakery",
                                "target_client_id": item["client_id"],
                                "kind": kind,
                                "visible_fraction": 1,
                            }
                            result = await api.post("/v1/events", json={"events": [event]})
                            assert result.json()["results"][0]["status"] == "accepted"
                    feed = (
                        await api.post(
                            "/v1/feed",
                            json={
                                "client_id": "bakery",
                                "intent": "supplier",
                                "limit": 10,
                                "seed": 42,
                            },
                        )
                    ).json()
                    journey.append({"phase": topic, "contacts": len(targets), "feed": feed})
                    for case in [
                        c
                        for c in cases
                        if c["id"]
                        in (
                            "search_coffee_supplier",
                            "search_flour_exact_supplier",
                            "search_oven_exact_supplier",
                            "feed_bakery_supplier",
                        )
                    ]:
                        body = {
                            k: v for k, v in case.items() if k in ("client_id", "intent", "query")
                        }
                        result = await api.post(
                            "/v1/search" if case.get("query") else "/v1/feed",
                            json={**body, "limit": 10, "seed": 42},
                        )
                        result.raise_for_status()
                        metrics("after_" + topic, case, 42, result.json(), 0)
                checks["journey_events_accepted"] = True
        summary = {}
        for mode in sorted({r["mode"] for r in rows}):
            summary[mode] = {}
            for kind in sorted({r["kind"] for r in rows}):
                group = [r for r in rows if r["mode"] == mode and r["kind"] == kind]
                if group:
                    summary[mode][kind] = {
                        "requests": len(group),
                        "mean_p10": float(np.mean([r["p10"] for r in group])),
                        "bad_cards": sum(r["irrelevant"] for r in group),
                        "precision_returned": (
                            1
                            - sum(r["irrelevant"] for r in group)
                            / sum(r["returned"] for r in group)
                            if sum(r["returned"] for r in group)
                            else None
                        ),
                        "bad_products": sum(r["irrelevant_products"] for r in group),
                        "products_shown": sum(r["products_shown"] for r in group),
                        "mean_returned": float(np.mean([r["returned"] for r in group])),
                        "latency_p95_ms": float(
                            np.percentile([r["latency_ms"] for r in group], 95)
                        ),
                    }
        report = {
            "companies": len(catalog),
            "topics": len(TOPICS),
            "requests": len(rows),
            "checks": checks,
            "summary": summary,
            "rows": rows,
            "limitations": [
                "scripted extraction",
                "hash embeddings and reranker",
                "hand-labeled business relevance, not user judgments",
                "ambiguous flour query accepts feed/stone; unknown query expects empty",
            ],
        }
        for name, value in [
            ("report", report),
            ("catalog", catalog),
            ("labels", labels),
            ("cases", cases),
            ("responses", responses),
            ("journey", journey),
            ("extractions", gateway.answers),
        ]:
            (output / (name + ".json")).write_text(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda value: sorted(value) if isinstance(value, set) else str(value),
                ),
                encoding="utf-8",
            )
        print(json.dumps({"output": str(output), "summary": summary, "checks": checks}, indent=2))
    finally:
        await provider.close()
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(
        run(
            Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-quality"),
            args.per_topic,
        )
    )
