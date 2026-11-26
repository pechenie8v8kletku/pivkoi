"""Exercise actual model HTTP inference before running the full catalog audit."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import numpy as np


async def run(args):
    async with httpx.AsyncClient(base_url=args.models, timeout=180) as api:

        async def embed(texts):
            r = await api.post(
                "/v1/embeddings", json={"model": "Qwen3-Embedding-4B", "input": texts}
            )
            r.raise_for_status()
            return np.asarray([x["embedding"] for x in r.json()["data"]])

        texts = ["Мука пшеничная для хлеба", "Промышленные насосы для перекачки воды"]
        batch = await embed(texts)
        single = np.concatenate([await embed([t]) for t in texts])
        checks = {
            "dimensions_2560": batch.shape == (2, 2560),
            "finite": bool(np.isfinite(batch).all()),
            "normalized": bool(np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5)),
            "batch_padding_consistent": bool(np.all(np.sum(batch * single, axis=1) > 0.999)),
        }
        query = "сырье для замеса хлебного теста"
        q = await embed(
            [
                "Instruct: Retrieve business offers or companies relevant "
                "to this B2B need.\nQuery: " + query
            ]
        )
        similarity = (q @ batch.T)[0].tolist()
        checks["relevant_over_unrelated_embedding"] = similarity[0] > similarity[1]
        r = await api.post(
            "/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "pairs": [{"query": query, "document": t} for t in texts],
            },
        )
        r.raise_for_status()
        scores = [x["relevance_score"] for x in r.json()["results"]]
        checks["relevant_over_unrelated_bge"] = scores[0] > scores[1]
        # Exercise a full configured batch with realistic longer descriptions.
        await embed(
            [
                (
                    "Поставка оборудования для производства и упаковки пищевых "
                    "продуктов. Монтаж, обслуживание, доставка. " * 10
                )
            ]
            * 2
        )
        health = (await api.get("/health")).json()
        checks["zero_oom"] = health["ooms"] == 0
        report = {
            "checks": checks,
            "cosines": similarity,
            "bge_logits": scores,
            "batch_single_cosines": np.sum(batch * single, axis=1).tolist(),
            "health": health,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not all(checks.values()):
            raise SystemExit("Model smoke failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, default=Path("local_experiments/model-smoke.json"))
    asyncio.run(run(parser.parse_args()))
