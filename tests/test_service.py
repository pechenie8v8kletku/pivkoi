import hashlib

import pytest
from fastapi.testclient import TestClient

from business_match.app import create_app
from business_match.config import Settings
from business_match.providers import normalize
from business_match.ranking import FeatureBuilder, create_ranker
from business_match.schemas import BusinessProfile, Document, Need
from business_match.storage import clients, ranker_states, relation_observations, slates


class FakeProviders:
    embedding_base_url = "test://embedding"
    llm_base_url = "test://llm"

    def __init__(self):
        self.profile_calls = 0

    async def profile(self, source):
        self.profile_calls += 1
        return BusinessProfile(summary=source.description or source.name,
                               offers=[source.description or source.name],
                               needs=[Need(text=text, intent="supplier", evidence="explicit")
                                      for text in source.explicit_needs])

    async def embed(self, texts, query=False):
        # Test-only deterministic vectors: never used by the application.
        return [normalize([b + 1 for b in hashlib.sha256(t.encode()).digest()[:8]])
                for t in texts]

    async def rerank(self, query, documents):
        return None

    async def rerank_pairs(self, pairs):
        return None


@pytest.fixture
def service():
    remote = FakeProviders()
    app = create_app(Settings(database_url="sqlite:///:memory:"), providers=remote)
    with TestClient(app) as client:
        yield client, remote, app


def snapshot(key, **kwargs):
    return {"id": key, "name": key, "description": "мука", "segment": "food",
            "explicit_needs": ["мука"], "products": [
                {"id": "p", "name": "мука", "stock": 5}
            ], **kwargs}


def test_snapshot_cache_and_stock_update(service):
    client, remote, app = service
    body = snapshot("a")
    first = client.put("/v1/clients/a", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["changed"]
    assert not client.put("/v1/clients/a", json=body).json()["changed"]
    body["products"][0]["stock"] = 0
    assert client.put("/v1/clients/a", json=body).status_code == 200
    assert remote.profile_calls == 1
    assert not any(d.kind == "product" for d in app.state.repository.client("a").documents)


def test_large_catalog_updates_only_refresh_changed_profile_input(service):
    client, remote, app = service
    products = [{"id": f"p{i:03}", "name": f"Product {i}",
                 "category": "A", "stock": 5} for i in range(100)]
    body = snapshot("large", products=products)
    assert client.put("/v1/clients/large", json=body).status_code == 200
    initial = app.state.repository.client("large")
    assert remote.profile_calls == 1
    assert len([d for d in initial.documents if d.kind == "product"]) == 100
    products[-1]["description"] = "Changed outside the representative sample"
    assert client.put("/v1/clients/large", json=body).status_code == 200
    updated = app.state.repository.client("large")
    assert remote.profile_calls == 1
    assert updated.profile_revision == initial.profile_revision
    assert updated.content_updated_at > initial.content_updated_at
    assert any(d.product_id == "p099" and "Changed outside" in d.text
               for d in updated.documents)
    products[0]["name"] = "Changed representative product"
    assert client.put("/v1/clients/large", json=body).status_code == 200
    assert remote.profile_calls == 2
    products[0]["active"] = False
    assert client.put("/v1/clients/large", json=body).status_code == 200
    assert remote.profile_calls == 3
    assert not any(d.product_id == "p000"
                   for d in app.state.repository.client("large").documents)


def test_profile_sample_diversity_dedup_and_operational_stability():
    from business_match.indexing import profile_input
    from business_match.schemas import ClientInput

    source = ClientInput.model_validate(snapshot("sample", products=[
        {"id": "a1", "name": "Same", "category": "A", "price": 10, "stock": 2},
        {"id": "a2", "name": "Same", "category": "A"},
        {"id": "a3", "name": "Different", "category": "A"},
        {"id": "b1", "name": "Rare category", "category": "B"},
        {"id": "c1", "name": "Removed", "category": "C", "active": False},
    ]))
    before = profile_input(source, 3)
    assert [p.id for p in before.products] == ["a1", "b1", "a3"]
    source.products.reverse()
    for p in source.products:
        p.price, p.stock, p.unit = 999, 0, "new unit"
    source.region = "New region"
    assert profile_input(source, 3).model_dump() == before.model_dump()


def test_feed_event_attribution_dedup_hide_and_search(service):
    client, _, app = service
    for key in ("a", "b", "c"):
        assert client.put(f"/v1/clients/{key}", json=snapshot(key)).status_code == 200
    response = client.post("/v1/feed", json={"client_id": "a", "seed": 4}).json()
    assert {i["client_id"] for i in response["items"]} == {"b", "c"}
    item = response["items"][0]
    event = {"event_id": "e1", "request_id": response["request_id"], "client_id": "a",
             "target_client_id": item["client_id"], "kind": "contact"}
    results = client.post("/v1/events", json={"events": [event, event]}).json()["results"]
    assert [r["status"] for r in results] == ["accepted", "duplicate"]
    from sqlalchemy import select

    with app.state.repository.engine.connect() as conn:
        observation = conn.execute(select(relation_observations.c.payload)).scalar_one()
    assert observation["exposures"] == 1
    bad = {**event, "event_id": "bad", "target_client_id": "a"}
    assert client.post("/v1/events", json={"events": [bad]}).json()["results"][0][
        "status"
    ] == "rejected"
    hide = {**event, "event_id": "hide", "kind": "hide"}
    assert client.post("/v1/events", json={"events": [hide]}).status_code == 200
    next_feed = client.post("/v1/feed", json={"client_id": "a"}).json()
    assert item["client_id"] not in {i["client_id"] for i in next_feed["items"]}
    search = client.post("/v1/search", json={"client_id": "a", "query": "мука"})
    assert search.status_code == 200
    assert search.json()["mode"] == "search"
    assert search.json()["items"]
    assert all(item["intent"] == "supplier" for item in search.json()["items"])
    explicit = client.post("/v1/search", json={
        "client_id": "a", "query": "offers", "intent": "buyer"})
    assert explicit.status_code == 200
    assert explicit.json()["items"]
    assert all(item["intent"] == "buyer" for item in explicit.json()["items"])
    assert client.post("/v1/search", json={"client_id": "a"}).status_code == 422
    assert len(app.state.repository.user_events("a")) == 2


def test_ranker_learns_from_events_and_config_can_disable_it(service):
    client, _, app = service
    for key in ("a", "b", "c"):
        client.put(f"/v1/clients/{key}", json=snapshot(key))
    first = client.post("/v1/feed", json={"client_id": "a", "seed": 1}).json()
    selected = first["items"][0]
    selected_slate = app.state.repository.get(slates, first["request_id"])
    selected_row = next(row for row in selected_slate["items"]
                        if row["client_id"] == selected["client_id"])
    ranker = app.state.engine.ranker
    ranker.prepare("a")
    before, _ = ranker.score(selected_row["features"], "a", selected["client_id"])
    client.post("/v1/events", json={"events": [{
        "event_id": "contact-ranker", "request_id": first["request_id"], "client_id": "a",
        "target_client_id": selected["client_id"], "kind": "contact",
    }]})
    state = app.state.repository.get(ranker_states, "a")
    assert len(state["observations"]) == 1
    assert state["observations"][next(iter(state["observations"]))]["reward"] == 1
    assert first["diagnostics"]["ranker_mode"] == "linucb"
    ranker.prepare("a")
    after, _ = ranker.score(selected_row["features"], "a", selected["client_id"])
    assert after != before

    FeatureBuilder().build(
        semantic=0.8, rerank=0.5, graph=0, freshness=0.5,
        same_region=False, same_city=False, in_stock=True, catalog_size=1,
        seen=0, is_new=True, intent="supplier",
    )
    disabled = create_ranker(Settings(ranker_mode="semantic"), app.state.repository)
    linucb = create_ranker(Settings(ranker_mode="linucb"), app.state.repository)
    disabled.prepare("a")
    linucb.prepare("a")
    assert disabled.version == "semantic-v2-evidence"
    assert linucb.version.startswith("personal-linear-ucb-v5-")


def test_api_key():
    app = create_app(Settings(database_url="sqlite:///:memory:", service_api_key="secret"),
                     providers=FakeProviders())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/clients/a").status_code == 401
        assert client.get("/v1/clients/a", headers={"X-API-Key": "secret"}).status_code == 404


def test_supplier_retrieval_uses_need_not_viewer_offer(service):
    client, _, app = service
    for key in ("bakery", "flour", "other_bakery"):
        client.put(f"/v1/clients/{key}", json=snapshot(key))
    repository = app.state.repository
    for key in ("bakery", "flour", "other_bakery"):
        indexed = repository.client(key)
        offer = [1.0, 0.0] if key == "flour" else [0.0, 1.0]
        indexed.documents = [Document(key="offer", text=key, kind="offer", vector=offer)]
        if key == "bakery":
            indexed.documents.append(Document(key="need", text="flour", kind="need",
                                               intent="supplier", vector=[1.0, 0.0]))
        repository.put(clients, key, indexed.model_dump(mode="json"))
    response = client.post("/v1/feed", json={"client_id": "bakery", "intent": "supplier",
                                             "limit": 1, "seed": 4})
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["client_id"] == "flour"


def test_contract_validates_products_and_impression(service):
    client, _, _ = service
    body = snapshot("a")
    body["products"] *= 2
    assert client.put("/v1/clients/a", json=body).status_code == 422
    assert client.post("/v1/events", json={"events": [{
        "event_id": "e", "request_id": "r", "client_id": "a", "target_client_id": "b",
        "kind": "impression", "visible_fraction": 0.1,
    }]}).status_code == 422


def test_card_keeps_relevant_products_without_catalog_padding(service):
    client, _, app = service
    client.put("/v1/clients/viewer", json=snapshot("viewer"))
    client.put("/v1/clients/vendor", json=snapshot("vendor", products=[
        {"id": "relevant", "name": "Product A", "stock": 5},
        {"id": "irrelevant", "name": "Product B", "stock": 5},
        {"id": "unavailable", "name": "Product C", "stock": 0},
    ]))
    repository = app.state.repository
    viewer = repository.client("viewer")
    viewer.documents = [Document(key="need", text="need", kind="need",
                                 intent="supplier", vector=[1.0, 0.0])]
    vendor = repository.client("vendor")
    vendor.documents = [
        Document(key=key, text=key, kind="product", product_id=key, vector=vector)
        for key, vector in [("relevant", [1.0, 0.0]),
                            ("irrelevant", [0.0, 1.0]),
                            ("unavailable", [1.0, 0.0])]
    ]
    for indexed in (viewer, vendor):
        repository.put(clients, indexed.source.id, indexed.model_dump(mode="json"))
    response = client.post("/v1/feed", json={
        "client_id": "viewer", "intent": "supplier", "seed": 42})
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 1
    products = response.json()["items"][0]["products"]
    assert [p["product"]["id"] for p in products] == ["relevant"]


def test_http_feedback_expands_feed_without_changing_explicit_search(service):
    from business_match.interests import interests

    client, remote, app = service
    repo = app.state.repository
    for key in ('viewer', 'old', 'clicked', 'new', 'other'):
        assert client.put('/v1/clients/'+key, json=snapshot(key)).status_code == 200
        entity = repo.client(key)
        entity.documents = [Document(key='d', text=key+' topic',
            kind='need' if key in ('viewer', 'other') else 'product',
            intent='supplier' if key in ('viewer', 'other') else None,
            product_id=None if key in ('viewer', 'other') else 'p',
            vector=[1., 0.] if key in ('viewer', 'other', 'old') else [0., 1.])]
        repo.put(clients, key, entity.model_dump(mode='json'))

    async def embed(texts, query=False):
        return [[0., 1.] for _ in texts]

    remote.embed = embed
    body = {'client_id': 'viewer', 'intent': 'supplier', 'limit': 10, 'seed': 42}
    before = client.post('/v1/feed', json=body).json()
    assert {x['client_id'] for x in before['items']} == {'old'}
    search_body = {**body, 'query': 'new task'}
    search = client.post('/v1/search', json=search_body).json()
    event = {'event_id': 'personal-contact', 'request_id': search['request_id'],
             'client_id': 'viewer', 'target_client_id': 'clicked',
             'product_id': 'p', 'kind': 'contact'}
    result = client.post('/v1/events', json={'events': [event, event]})
    assert [x['status'] for x in result.json()['results']] == ['accepted', 'duplicate']
    after = client.post('/v1/feed', json=body).json()
    assert {x['client_id'] for x in after['items']} == {'old', 'clicked', 'new'}
    assert after['diagnostics']['interest_candidates'] == 2
    other = client.post('/v1/feed', json={**body, 'client_id':'other'}).json()
    assert {x['client_id'] for x in other['items']} == {'old'}
    searched = client.post('/v1/search', json=search_body).json()
    assert {x['client_id'] for x in searched['items']} == {x['client_id'] for x in search['items']}
    assert len(interests(repo, app.state.engine.settings, 'viewer', set())['supplier']) == 1
    hide = {**event, 'event_id': 'hide-contact', 'request_id': after['request_id'], 'kind':'hide'}
    hidden_result = client.post('/v1/events', json={'events':[hide]}).json()
    assert hidden_result['results'][0]['status'] == 'accepted'
    assert {x['client_id'] for x in client.post('/v1/feed', json=body).json()['items']} == {'old'}
