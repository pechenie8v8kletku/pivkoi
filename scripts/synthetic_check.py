"""Run real service routes against local model HTTP mocks; artifacts stay git-ignored."""

import argparse
import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from business_match.app import create_app
from business_match.config import Settings
from business_match.mock_api import app as mock_app
from business_match.providers import Providers
from business_match.schemas import utcnow
from business_match.storage import Repository, ranker_states, slates


async def main(per_category=8):
    output = Path("local_experiments") / utcnow().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{output / 'experiment.db'}",
        model_api_mode="mock", rerank_enabled=True,
        llm_rpm=10000, embedding_rpm=10000, embedding_tpm=10000000,
        rerank_rpm=10000, rerank_tpm=10000000,
        exploration_rate=0, linucb_alpha=0, ranker_graph_weight=0,
    )
    repository = Repository(settings.database_url)
    repository.initialize()
    providers = Providers(settings, repository)
    await providers.http.aclose()
    providers.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app))
    catalog = []
    for viewer in ("viewer", "control"):
        catalog.append({"id": viewer, "name": "Пекарня", "segment": "пекарня",
                        "description": "Хлеб выпечка", "explicit_needs": [
                            "Мука пшеничная оптом", "Упаковка бумажная оптом"],
                        "region": "Татарстан", "city": "Казань"})
    categories = ["Мука пшеничная оптом", "Упаковка бумажная оптом",
                  "Кофе зерновой", "Металлопрокат стальной"]
    for category, text in enumerate(categories):
        for i in range(per_category):
            catalog.append({
                "id": f"supplier-{category}-{i}", "name": f"{text} {i}",
                "segment": text, "description": text,
                "region": "Татарстан" if i % 2 else "Москва", "city": "Казань",
                "products": [{"id": f"sku-{i}", "name": text, "stock": 10,
                              "description": f"Доставка партия {i}"}],
            })
    (output / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    app = create_app(settings, providers=providers)
    checks, timings, snapshots = {}, [], {}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://service") as api:
            for client in catalog:
                response = await api.put(f"/v1/clients/{client['id']}", json=client)
                response.raise_for_status()

            async def feed(viewer="viewer", **extra):
                start = time.perf_counter()
                result = await api.post("/v1/feed", json={
                    "client_id": viewer, "intent": "supplier", "limit": 50, "seed": 42,
                    **extra,
                })
                result.raise_for_status()
                timings.append((time.perf_counter() - start) * 1000)
                return result.json()

            before = await feed()
            control_before = await feed("control")
            snapshots["before"] = before
            # The other bakery is also an eligible company (description-derived offer).
            checks["feed_nonempty"] = len(before["items"]) == min(50, per_category * 4 + 1)
            checks["unique_clients"] = len({x["client_id"] for x in before["items"]}) == len(
                before["items"])
            checks["reranker_used"] = before["diagnostics"]["reranked"]
            rows = repository.get(slates, before["request_id"])["items"]
            training = [r for r in rows if r["client_id"].startswith("supplier-1-")][:6]
            checks["training_candidates_available"] = len(training) >= 4
            ranker = app.state.engine.ranker
            ranker.prepare("viewer")
            scores_before = {r["client_id"]: ranker.score(r["features"], "viewer", "")[0]
                             for r in rows}
            for row in training:
                event = {"event_id": str(uuid4()), "request_id": before["request_id"],
                         "client_id": "viewer", "target_client_id": row["client_id"],
                         "kind": "contact"}
                result = (await api.post("/v1/events", json={"events": [event, event]})).json()
                checks["event_dedup"] = [r["status"] for r in result["results"]] == [
                    "accepted", "duplicate"]
            ranker.prepare("viewer")
            deltas = {r["client_id"]: ranker.score(r["features"], "viewer", "")[0]
                      - scores_before[r["client_id"]] for r in rows}
            def average(prefix):
                values = [v for k, v in deltas.items() if k.startswith(prefix)]
                return sum(values) / len(values)
            checks["personal_content_learning"] = average("supplier-1-") > average("supplier-0-")
            after = await feed()
            snapshots["after"] = after
            trained_ids = {r["client_id"] for r in training}
            held_out = [value for key, value in deltas.items()
                        if key.startswith("supplier-1-") and key not in trained_ids]
            checks["unseen_product_transfer"] = bool(held_out) and sum(held_out) / len(
                held_out) > average("supplier-0-")
            control_after = await feed("control")
            checks["other_user_unchanged"] = [i["client_id"] for i in control_before["items"]] == [
                i["client_id"] for i in control_after["items"]]
            control_rows = repository.get(slates, control_before["request_id"])["items"]
            for row in [r for r in control_rows if r["client_id"].startswith("supplier-0-")][:6]:
                response = await api.post("/v1/events", json={"events": [{
                    "event_id": str(uuid4()), "request_id": control_before["request_id"],
                    "client_id": "control", "target_client_id": row["client_id"],
                    "kind": "contact",
                }]})
                assert response.json()["results"][0]["status"] == "accepted"
            ranker.prepare("control")
            control_deltas = {r["client_id"]: ranker.score(r["features"], "control", "")[0]
                              - scores_before[r["client_id"]] for r in rows}
            def control_mean(prefix):
                values = [v for k, v in control_deltas.items() if k.startswith(prefix)]
                return sum(values) / len(values)
            checks["opposite_preference_learned"] = control_mean("supplier-0-") > control_mean(
                "supplier-1-")
            snapshots["opposite_user"] = await feed("control")
            def target_count(slate, prefix, k):
                return sum(x["client_id"].startswith(prefix) for x in slate["items"][:k])
            checks["preferred_share_grows_or_saturates"] = (
                target_count(after, "supplier-1-", 50) > target_count(before, "supplier-1-", 50)
                or target_count(before, "supplier-1-", 50) == per_category
            )
            # Exercise production defaults too, independently of the deterministic ablation.
            settings.exploration_rate = 0.1
            settings.linucb_alpha = 0.15
            settings.ranker_graph_weight = 0.1
            default_feed = await feed()
            snapshots["default_policy"] = default_feed
            checks["default_policy_unique"] = (
                len({x["client_id"] for x in default_feed["items"]}) == len(default_feed["items"])
            )
            settings.exploration_rate = 0
            settings.linucb_alpha = 0
            settings.ranker_graph_weight = 0
            target = training[0]["client_id"]
            future = {"event_id": str(uuid4()), "request_id": before["request_id"],
                      "client_id": "viewer", "target_client_id": target,
                      "kind": "hide", "occurred_at": (utcnow() + timedelta(days=2)).isoformat()}
            rejected = (await api.post("/v1/events", json={"events": [future]})).json()
            checks["future_event_rejected"] = rejected["results"][0]["status"] == "rejected"
            hide = {**future, "event_id": str(uuid4()), "occurred_at": utcnow().isoformat()}
            await api.post("/v1/events", json={"events": [hide]})
            checks["hide_removes_card"] = target not in {
                x["client_id"] for x in (await feed())["items"]
            }
            observations = repository.get(ranker_states, "viewer")["observations"]
            checks["hide_overrides_contact"] = any(
                k.startswith(target + "|") and v["reward"] == 0 for k, v in observations.items()
            )
            search = await api.post("/v1/search", json={
                "client_id": "control", "query": "Мука пшеничная оптом", "intent": "supplier",
            })
            search.raise_for_status()
            snapshots["search"] = search.json()
            checks["search_top1_flour"] = search.json()["items"][0]["client_id"].startswith(
                "supplier-0-"
            )
            filtered = await feed(region="Москва")
            checks["region_filter"] = all(i["region"] == "Москва" for i in filtered["items"])
            final_state = repository.get(ranker_states, "viewer")
    restarted = Repository(settings.database_url)
    checks["state_survives_restart"] = restarted.get(ranker_states, "viewer") == final_state
    restarted.close()
    def mean_rank(slate):
        ranks = [i + 1 for i, item in enumerate(slate["items"])
                 if item["client_id"].startswith("supplier-1-")]
        return sum(ranks) / len(ranks)
    report = {"checks": checks, "clients": len(catalog),
              "mean_packaging_rank_before": mean_rank(before),
              "mean_packaging_rank_after": mean_rank(after),
              "packaging_top50_before": target_count(before, "supplier-1-", 50),
              "packaging_top50_after": target_count(after, "supplier-1-", 50),
              "packaging_top10_before": target_count(before, "supplier-1-", 10),
              "packaging_top10_after": target_count(after, "supplier-1-", 10),
              "fixed_features_score_delta_packaging": average("supplier-1-"),
              "fixed_features_score_delta_flour": average("supplier-0-"),
              "feed_latency_ms": timings,
              "scope": "ASGI HTTP contracts, SQLite, mocked models; deterministic policy ablation"}
    for name, data in {"report": report, "slates": snapshots}.items():
        (output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    await providers.close()
    repository.close()
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"Artifacts: {output}")
    if not all(checks.values()):
        raise SystemExit("Experiment exposed a regression; inspect report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=8)
    args = parser.parse_args()
    if args.per_category < 8:
        parser.error("--per-category must be at least 8")
    asyncio.run(main(args.per_category))
