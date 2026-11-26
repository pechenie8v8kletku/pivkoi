from datetime import timedelta

import pytest

from business_match.config import Settings
from business_match.interests import embedding_signature, interests
from business_match.retrieval import ExactRetriever
from business_match.schemas import (
    BusinessProfile,
    ClientInput,
    Document,
    IndexedClient,
    Intent,
    RecommendRequest,
    utcnow,
)
from business_match.storage import Repository, clients, events


@pytest.fixture
def market():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    settings = Settings(_env_file=None)
    for key, vector in [
        ("viewer", [1.0, 0.0]),
        ("old", [1.0, 0.0]),
        ("clicked", [0.0, 1.0]),
        ("new", [0.0, 1.0]),
    ]:
        doc = Document(
            key="doc",
            kind="need" if key == "viewer" else "offer",
            intent=Intent.supplier if key == "viewer" else None,
            text=key + " offer",
            vector=vector,
        )
        indexed = IndexedClient(
            source=ClientInput(id=key, name=key),
            profile=BusinessProfile(summary=key, offers=[key]),
            documents=[doc],
            revision="r",
            profile_revision="p",
        )
        repo.put(clients, key, indexed.model_dump(mode="json"))
    yield repo, settings
    repo.close()


def record(
    repo,
    settings,
    kind="contact",
    user="viewer",
    intent="supplier",
    age=0,
    anchor=True,
    identifier="event",
):
    doc = repo.client("clicked").documents[0]
    now = utcnow() - timedelta(days=age)
    payload = {
        "event": {"kind": kind, "target_client_id": "clicked", "occurred_at": now.isoformat()},
        "received_at": utcnow().isoformat(),
        "intent": intent,
    }
    if anchor:
        payload["interest_anchor"] = {
            "text": doc.text,
            "vector": doc.vector,
            "product_id": None,
            "signature": embedding_signature(settings),
        }
    repo.put(events, identifier, payload, client_id=user)


def retrieve(repo, settings, query=None, hidden=None):
    req = RecommendRequest(
        client_id="viewer", intent=Intent.supplier, query="old" if query else None
    )
    return ExactRetriever(repo, settings).retrieve(
        repo.client("viewer"), req, hidden or set(), query
    )


def test_positive_retrieves_unseen_neighbors_and_preserves_baseline(market):
    repo, settings = market
    assert {m.target.source.id for m in retrieve(repo, settings)} == {"old"}
    record(repo, settings)
    result = retrieve(repo, settings)
    assert {m.target.source.id for m in result} == {"old", "clicked", "new"}
    assert next(m for m in result if m.target.source.id == "new").behavioral
    assert {m.target.source.id for m in retrieve(repo, settings, query=[1.0, 0.0])} == {"old"}
    assert {
        m.target.source.id
        for m in retrieve(repo, settings.model_copy(update={"interest_retrieval_enabled": False}))
    } == {"old"}


@pytest.mark.parametrize("kind", ["open", "dwell", "impression"])
def test_weak_actions_do_not_create_retrieval_interest(market, kind):
    repo, settings = market
    record(repo, settings, kind=kind)
    assert not interests(repo, settings, "viewer", set())


def test_ten_vector_global_bound_and_duplicate_collapse(market):
    import numpy as np

    repo, settings = market
    template = repo.client("clicked")
    for i in range(12):
        vector = np.eye(12)[i].tolist()
        target = template.model_copy(deep=True)
        target.source.id = f"topic-{i}"
        target.documents[0].text = f"Topic {i}"
        target.documents[0].vector = vector
        repo.put(clients, target.source.id, target.model_dump(mode="json"))
        for duplicate in range(3):
            now = utcnow().isoformat()
            repo.put(
                events,
                f"{i}-{duplicate}",
                {
                    "event": {
                        "kind": "contact",
                        "target_client_id": target.source.id,
                        "occurred_at": now,
                    },
                    "received_at": now,
                    "intent": "supplier" if i % 2 else "buyer",
                    "interest_anchor": {
                        "text": target.documents[0].text,
                        "vector": vector,
                        "product_id": None,
                        "signature": embedding_signature(settings),
                    },
                },
                client_id="viewer",
            )
    selected = interests(repo, settings, "viewer", set())
    assert sum(map(len, selected.values())) == 10
    assert {k for k, v in selected.items() if v} == {Intent.supplier, Intent.buyer}


def test_role_user_model_expiry_and_legacy_isolation(market):
    repo, settings = market
    record(repo, settings, user="other")
    assert not interests(repo, settings, "viewer", set())
    record(repo, settings, intent="buyer", identifier="buyer")
    assert {m.target.source.id for m in retrieve(repo, settings)} == {"old"}
    assert Intent.buyer in interests(repo, settings, "viewer", set())
    changed = settings.model_copy(update={"embedding_model": "different-model"})
    assert not interests(repo, changed, "viewer", set())
    record(repo, settings, age=59, identifier="old")
    assert Intent.supplier not in interests(repo, settings, "viewer", set())
    record(repo, settings, anchor=False, identifier="legacy")
    assert Intent.supplier not in interests(repo, settings, "viewer", set())


def test_hide_and_catalog_change_withdraw_interest(market):
    repo, settings = market
    record(repo, settings, age=1)
    assert interests(repo, settings, "viewer", set())
    assert not interests(repo, settings, "viewer", {"clicked"})
    record(repo, settings, kind="hide", anchor=False, identifier="hide")
    assert not interests(repo, settings, "viewer", set())
    record(repo, settings, identifier="new-contact")
    assert interests(repo, settings, "viewer", set())
    target = repo.client("clicked")
    target.documents[0].text = "changed catalog"
    repo.put(clients, "clicked", target.model_dump(mode="json"))
    assert not interests(repo, settings, "viewer", set())


def test_behavioral_pools_respect_filters_and_block_sizes(market):
    repo, settings = market
    record(repo, settings)
    request = RecommendRequest(
        client_id="viewer", intent=Intent.supplier, exclude_client_ids=["clicked"]
    )
    expected = None
    for page, block in [(1, 1), (128, 1024)]:
        configured = settings.model_copy(
            update={"retrieval_client_batch_size": page, "retrieval_document_batch_size": block}
        )
        actual = ExactRetriever(repo, configured).retrieve(
            repo.client("viewer"), request, set(), None
        )
        values = [(m.target.source.id, m.semantic, m.behavioral) for m in actual]
        assert {m.target.source.id for m in actual} == {"old", "new"}
        if expected is not None:
            assert [(v[0], v[2]) for v in values] == [(v[0], v[2]) for v in expected]
            assert [v[1] for v in values] == pytest.approx([v[1] for v in expected])
        expected = values
    request.region = "absent-region"
    assert not ExactRetriever(repo, settings).retrieve(repo.client("viewer"), request, set(), None)
