import httpx
import pytest
from httpx import ASGITransport

from business_match.config import Settings
from business_match.mock_api import app as mock_app
from business_match.providers import Providers
from business_match.schemas import ClientInput
from business_match.storage import Repository


@pytest.mark.asyncio
async def test_model_http_mock_exercises_profile_embedding_and_rerank():
    repository = Repository("sqlite:///:memory:")
    repository.initialize()
    settings = Settings(model_api_mode="mock", rerank_enabled=True)
    providers = Providers(settings, repository)
    await providers.http.aclose()
    providers.http = httpx.AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://mock.test"
    )
    try:
        profile = await providers.profile(ClientInput.model_validate({
            "id": "c1", "name": "Пекарня", "segment": "Пекарня",
            "description": "Пекарня в Казани", "explicit_needs": ["Мука оптом"],
            "products": [{"id": "p1", "name": "Хлеб", "description": "Ржаной"}],
            "active": True,
        }))
        assert profile.offers and profile.needs[0].text == "Мука оптом"
        vectors = await providers.embed(["поставщик муки", "мука оптом"])
        assert len(vectors[0]) == 384
        scores = await providers.rerank_pairs([("поставщик муки", "мука оптом")])
        assert scores is not None and len(scores) == 1
    finally:
        await providers.close()
        repository.close()
