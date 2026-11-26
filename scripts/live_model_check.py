"""Evaluate the service against the labelled synthetic B2B market on the CIB gateway."""
import asyncio
import getpass
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

from business_match.app import create_app
from business_match.config import Settings

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "https://llm-api.cibaa.raiffeisen.ru/v1"


def gateway_settings(database, key):
    return Settings(
        _env_file=None, database_url=f"sqlite:///{database}",
        model_api_mode="external", llm_base_url=GATEWAY, llm_model="qwen38-27b",
        embedding_base_url=GATEWAY, embedding_model="qwen3-embedding-4b",
        rerank_url=GATEWAY + "/score", rerank_model="bge-reranker-v2-m3",
        llm_api_key=key, embedding_api_key=key, rerank_api_key=key,
        http_timeout=120, exploration_rate=0, linucb_alpha=0,
    )


async def main():
    key = os.getenv("LLM_API_KEY") or getpass.getpass("LLM API key: ")
    fixture = json.loads((ROOT / "fixtures/b2b_market.json").read_text())
    output = ROOT / "local_experiments" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f-live")
    output.mkdir(parents=True)
    settings = gateway_settings(output / "market.db", key)
    app = create_app(settings)
    report = {"profiles": {}, "cases": []}
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service",
        headers={"X-API-Key": settings.service_api_key},
    ) as api:
        for card in fixture["catalog"]:
            response = await api.put(f"/v1/clients/{card['id']}", json=card)
            response.raise_for_status()
            report["profiles"][card["id"]] = response.json()["profile"]
        for case in fixture["cases"]:
            request = {k: v for k, v in case.items() if k != "relevant"}
            route = "/v1/search" if "query" in request else "/v1/feed"
            response = await api.post(route, json={**request, "limit": 3, "seed": 42})
            response.raise_for_status()
            slate = response.json()
            ids = [item["client_id"] for item in slate["items"]]
            hits = set(ids).intersection(case["relevant"])
            result = {**case, "top3": ids, "recall_at_3": len(hits) / len(case["relevant"]),
                      "reranked": slate["diagnostics"]["reranked"], "slate": slate}
            report["cases"].append(result)
            print(request, ids, "Recall@3", result["recall_at_3"], "BGE", result["reranked"])
    path = output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(path)
    if not all(case["reranked"] for case in report["cases"]):
        raise SystemExit("Часть выдач получена без BGE. Подробности в отчёте.")


if __name__ == "__main__":
    asyncio.run(main())
