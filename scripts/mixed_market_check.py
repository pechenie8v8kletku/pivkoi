"""Directed mixed-market click journeys through the real API, with local HTTP mocks."""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

import httpx

from business_match.app import create_app
from business_match.config import Settings
from business_match.mock_api import app as mock_app
from business_match.providers import Providers
from business_match.ranking import LinUCBRanker, SemanticRanker
from business_match.schemas import utcnow
from business_match.storage import Repository, slates

# Explicit needs are ground truth in this fixture, not inferred by the hashing mock.
MARKETS = {
    "bakery": ("Пекарня", "Хлеб выпечка", ["Мука зерно", "Печи тестомесы", "Упаковка пакеты"]),
    "shop": ("Магазин", "Розничная торговля", ["Хлеб выпечка", "Холодильники витрины"]),
    "cafe": ("Кафе", "Готовые обеды", ["Хлеб выпечка", "Печи тестомесы", "Кофе зерновой"]),
    "flour": ("Мельница", "Мука зерно", ["Перевозка грузов"]),
    "oven": ("Оборудование", "Печи тестомесы", ["Металл сталь", "Перевозка грузов"]),
    "cold": ("Холодильное оборудование", "Холодильники витрины", ["Металл сталь"]),
    "pack": ("Упаковка", "Упаковка пакеты", ["Перевозка грузов"]),
    "coffee": ("Кофейный опт", "Кофе зерновой", ["Упаковка пакеты"]),
    "logistics": ("Логистика", "Перевозка грузов", ["Ремонт автомобилей"]),
    "metal": ("Металлобаза", "Металл сталь", ["Перевозка грузов"]),
    "garage": ("Автосервис", "Ремонт автомобилей", ["Металл сталь"]),
    "salon": ("Салон красоты", "Стрижка окрашивание", ["Косметика шампуни"]),
    "cosmetics": ("Косметика", "Косметика шампуни", ["Упаковка пакеты"]),
}


async def main():
    output = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-mixed")
    output.mkdir(parents=True)
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{output / 'market.db'}",
        model_api_mode="mock", llm_rpm=100000, embedding_rpm=100000,
        embedding_tpm=100000000, rerank_rpm=100000, rerank_tpm=100000000,
        exploration_rate=0, linucb_alpha=0, ranker_graph_weight=0,
    )
    repo = Repository(settings.database_url)
    repo.initialize()
    providers = Providers(settings, repo)
    await providers.http.aclose()
    providers.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app))
    catalog = []
    for key, (segment, offer, needs) in MARKETS.items():
        for i in range(20):
            catalog.append({"id": f"{key}-{i}", "name": f"{segment} {i}",
                            "segment": segment, "description": offer, "explicit_needs": needs,
                            "region": "Татарстан", "products": [{
                                "id": "main", "name": offer, "stock": 100,
                            }]})
    app = create_app(settings, providers=providers)
    checks, journey, events, timings = {}, [], [], []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://service") as api:
            for client in catalog:
                (await api.put(f"/v1/clients/{client['id']}", json=client)).raise_for_status()

            async def feed(client="bakery-0", intent="supplier", query=None):
                start = time.perf_counter()
                response = await api.post("/v1/search" if query else "/v1/feed", json={
                    "client_id": client, "intent": intent, "limit": 20, "seed": 42,
                    "query": query,
                })
                response.raise_for_status()
                timings.append((time.perf_counter() - start) * 1000)
                return response.json()

            async def click(slate, target, kind, client="bakery-0", **extra):
                event = {"event_id": str(uuid4()), "request_id": slate["request_id"],
                         "client_id": client, "target_client_id": target,
                         "kind": kind, **extra}
                result = await api.post("/v1/events", json={"events": [event]})
                result.raise_for_status()
                assert result.json()["results"][0]["status"] == "accepted", result.text
                events.append(event)

            def counts(slate):
                return dict(Counter(x["client_id"].split("-")[0] for x in slate["items"]))

            for client, intent, expected in [
                ("bakery-0", "supplier", {"flour", "oven", "pack"}),
                ("bakery-0", "buyer", {"shop", "cafe"}),
                ("oven-0", "buyer", {"bakery", "cafe"}),
                ("cold-0", "buyer", {"shop"}),
                ("salon-0", "supplier", {"cosmetics"}),
            ]:
                slate = await feed(client, intent)
                checks[f"direction_{client}_{intent}"] = (
                    len(slate["items"]) == 20 and set(counts(slate)) == expected)
                journey.append({"phase": f"cold_{client}_{intent}", "counts": counts(slate)})

            control = await feed("bakery-1")
            reference = await feed()
            rows = repo.get(slates, reference["request_id"])["items"]

            def fixed_scores():
                ranker = app.state.engine.ranker
                ranker.prepare("bakery-0")
                grouped = {}
                for row in rows:
                    group = row["client_id"].split("-")[0]
                    grouped.setdefault(group, []).append(
                        ranker.score(row["features"], "bakery-0", "")[0])
                return {k: sum(v) / len(v) for k, v in grouped.items()}

            baseline = fixed_scores()
            journey.append({"phase": "baseline", "counts": counts(reference),
                            "scores": baseline})
            probe = await feed(query="Мука зерно")
            probe_target = probe["items"][0]["client_id"]
            await click(probe, probe_target, "impression", visible_fraction=1)
            checks["impression_not_positive_label"] = fixed_scores() == baseline
            await click(probe, probe_target, "dwell", dwell_ms=15000)
            checks["dwell_updates_immediately"] = abs(
                fixed_scores()["flour"] - baseline["flour"]) > 1e-5
            # Search is an explicit navigation path into a specialty, then return to the feed.
            for phase, query, family, kind in [
                ("flour_opens", "Мука зерно", "flour", "open"),
                ("equipment_likes", "Печи тестомесы", "oven", "like"),
                ("equipment_contacts", "Печи тестомесы", "oven", "contact"),
                ("switch_packaging", "Упаковка пакеты", "pack", "contact"),
                ("return_flour", "Мука зерно", "flour", "contact"),
            ]:
                phase_before = fixed_scores()
                slate = await feed(query=query)
                targets = [x["client_id"] for x in slate["items"]
                           if x["client_id"].startswith(family + "-")][:8]
                assert len(targets) == 8
                for index, target in enumerate(targets):
                    await click(slate, target, "impression", visible_fraction=1)
                    await click(slate, target, kind)
                    current = await feed()
                    if index == 0 and kind in ("like", "contact"):
                        checks[f"immediate_{phase}"] = (
                            fixed_scores()[family] > phase_before[family] + 1e-5)
                    journey.append({"phase": phase, "clicked": target,
                                    "counts": counts(current), "scores": fixed_scores()})
                if phase == "equipment_contacts":
                    equipment_scores = fixed_scores()
                    checks["equipment_learning"] = equipment_scores["oven"] > baseline["oven"]
                    for target in targets:
                        await click(slate, target, "hide")
                    checks["hidden_removed"] = not set(targets) & {
                        x["client_id"] for x in (await feed())["items"]}
                    checks["negative_reversal"] = fixed_scores()["oven"] < equipment_scores["oven"]
                if phase == "switch_packaging":
                    checks["packaging_changes_feed"] = counts(current).get("pack", 0) > counts(
                        reference).get("pack", 0)
                    checks["supplier_diversity_preserved"] = max(counts(current).values()) <= 12
                if phase == "return_flour":
                    checks["return_interest_learned"] = (
                        fixed_scores()["flour"] > phase_before["flour"] + 0.01
                        and counts(current).get("flour", 0) > counts(reference).get("flour", 0))
            checks["packaging_learning"] = fixed_scores()["pack"] > baseline["pack"]
            checks["personal_isolation"] = counts(await feed("bakery-1")) == counts(control)
            mixed = await feed(intent=None)
            checks["mixed_unique"] = len({x["client_id"] for x in mixed["items"]}) == 20
            checks["mixed_multiple_roles"] = len({x["intent"] for x in mixed["items"]}) > 1
            semantic = SemanticRanker(settings)
            checks["semantic_baseline_unchanged"] = all(
                abs(semantic.score(row["features"], "bakery-0", "")[0]
                    - baseline[row["client_id"].split("-")[0]]) < 1e-6
                for row in rows if row["client_id"].startswith("flour-"))
            reopened = Repository(settings.database_url)
            restored = LinUCBRanker(settings, reopened)
            checks["posterior_survives_restart"] = all(
                abs(restored.score(row["features"], "bakery-0", "")[0]
                    - app.state.engine.ranker.score(row["features"], "bakery-0", "")[0]) < 1e-5
                for row in rows)
            reopened.close()
            settings.exploration_rate = 0.1
            settings.linucb_alpha = 0.15
            for seed in range(5):
                response = await api.post("/v1/feed", json={
                    "client_id": "bakery-0", "limit": 20, "seed": seed})
                response.raise_for_status()
                items = response.json()["items"]
                roles = Counter(x["intent"] for x in items)
                checks[f"exploration_relevant_balanced_{seed}"] = (
                    len({x["client_id"] for x in items}) == 20
                    and roles["peer"] <= 4 and roles["buyer"] >= 4
                    and set(counts(response.json())) <= {
                        "bakery", "shop", "cafe", "pack", "flour", "oven"})
                journey.append({"phase": "exploration", "seed": seed,
                                "counts": counts(response.json()), "roles": dict(roles)})
    report = {"clients": len(catalog), "checks": checks, "journey": journey,
              "feed_latency_ms": timings, "events": len(events),
              "scope": "HTTP ASGI, explicit needs, hashing mock; not real semantic evaluation"}
    for name, value in [("report", report), ("catalog", catalog), ("events", events)]:
        (output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    await providers.close()
    repo.close()
    print(json.dumps({"artifacts": str(output), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit("Mixed-market regression: inspect report")


if __name__ == "__main__":
    asyncio.run(main())
