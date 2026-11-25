import json

import httpx
import pytest

from business_match.config import Settings
from business_match.providers import Providers
from business_match.storage import Repository


async def test_embedding_cache_query_prefix_and_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'cache.db'}"
    repository = Repository(url)
    repository.initialize()
    settings = Settings(database_url=url)
    provider = Providers(settings, repository)
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1, i + 1]} for i in range(len(body["input"]))
        ]})

    await provider.http.aclose()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    first = await provider.embed(["мука", "мука", "упаковка"])
    assert first[0] == first[1]
    assert len(calls) == 1 and len(calls[0]["input"]) == 2
    await provider.close()
    repository.close()
    repository = Repository(url)
    provider = Providers(settings, repository)
    await provider.http.aclose()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    assert await provider.embed(["мука"]) == [first[0]]
    assert len(calls) == 1
    await provider.embed(["мука"], query=True)
    assert len(calls) == 2
    assert calls[1]["input"][0].startswith("Instruct:")
    await provider.close()
    repository.close()


async def test_provider_bad_batch_rejected_and_reranker_fallback():
    repository = Repository("sqlite:///:memory:")
    repository.initialize()
    provider = Providers(Settings(rerank_enabled=True, rerank_url="http://test/rerank"),
                         repository)
    await provider.http.aclose()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [], "results": []})
    ))
    with pytest.raises(ValueError, match="batch indices"):
        await provider.embed(["мука"])
    assert await provider.rerank("мука", ["товар"]) is None
    await provider.close()
    repository.close()
