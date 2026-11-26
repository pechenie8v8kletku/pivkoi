import json

import httpx

from business_match.config import Settings
from business_match.indexing import Indexer
from business_match.mock_api import chat_completions, mock_embedding
from business_match.providers import Providers
from business_match.schemas import ClientInput, Intent
from business_match.storage import Repository


async def test_card_only_llm_search_needs_are_indexed_and_cached():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    settings = Settings(_env_file=None)
    provider = Providers(settings, repo)
    calls = []

    def respond(request):
        body = json.loads(request.content)
        if request.url.path.endswith("chat/completions"):
            calls.append(body)
            result = chat_completions(body)
            # Controlled LLM output: tests the real structured adapter, not Qwen quality.
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
                json.dumps({
                    "summary": "Пекарня выпускает хлеб", "offers": ["Хлеб"],
                    "needs": [
                        {"text": text, "intent": intent, "importance": 3,
                         "evidence": "inferred"}
                        for text, intent in [("Мука", "supplier"),
                                             ("Печи", "supplier"),
                                             ("Магазины хлеба", "buyer"),
                                             ("Доставка еды", "partner")]
                    ]}))
            return httpx.Response(200, json=result)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": mock_embedding(text)}
            for i, text in enumerate(body["input"])]})

    await provider.http.aclose()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        indexer = Indexer(repo, provider, settings)
        source = ClientInput(id="bakery", name="Пекарня", description="Выпекаем хлеб")
        indexed, changed = await indexer.upsert(source)
        assert changed and calls[0]["model"] == "Qwen3.8-38B"
        docs = [d for d in indexed.documents if d.kind == "need"]
        assert len(docs) == 4 and all(d.vector and d.weight == 0.7 for d in docs)
        assert {d.intent for d in docs} == {Intent.supplier, Intent.buyer, Intent.partner}
        assert not indexed.source.products
        assert not any(d.kind == "product" for d in indexed.documents)
        assert not (await indexer.upsert(source))[1]
        await indexer.upsert(source.model_copy(update={"region": "Москва"}))
        assert len(calls) == 1
        await indexer.upsert(source.model_copy(update={"description": "Выпекаем хлеб и торты"}))
        assert len(calls) == 2
    finally:
        await provider.close()
        repo.close()
