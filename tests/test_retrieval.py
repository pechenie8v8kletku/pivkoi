import numpy as np
import pytest

from business_match.config import Settings
from business_match.retrieval import ExactRetriever
from business_match.schemas import (
    BusinessProfile,
    ClientInput,
    Document,
    IndexedClient,
    Intent,
    RecommendRequest,
)
from business_match.storage import Repository, clients


def make_catalog(count=27):
    rng = np.random.default_rng(731)
    result = []
    for index in range(count):
        docs = []
        for number, (kind, intent) in enumerate([
            ("offer", None), ("product", None), ("product", None),
            ("need", Intent.supplier), ("need", Intent.supplier),
            ("need", Intent.partner),
        ]):
            vector = rng.normal(size=12)
            vector /= np.linalg.norm(vector)
            docs.append(Document(key=f"d-{number}", kind=kind, intent=intent,
                                 text=f"document {number}", vector=vector.tolist(),
                                 weight=0.7 if number % 2 else 1))
        result.append(IndexedClient(
            source=ClientInput(id=f"c-{index:03}", name=f"Company {index}",
                               region="A" if index % 2 else "B", city="City",
                               active=index != 3),
            profile=BusinessProfile(summary="Company", offers=["Offer"]),
            documents=docs, revision="test", profile_revision="test"))
    return result


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize("explicit_search", [False, True])
def test_chunked_retrieval_matches_brute_force(intent, explicit_search):
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    catalog = make_catalog()
    for item in reversed(catalog):
        repo.put(clients, item.source.id, item.model_dump(mode="json"))
    viewer = catalog[0]
    request = RecommendRequest(client_id=viewer.source.id, intent=intent,
                               exclude_client_ids=["c-005"], region="A", city="City",
                               query="query" if explicit_search else None)
    query_vector = viewer.documents[0].vector if explicit_search else None
    hidden = {"c-007"}
    if explicit_search:
        queries = [Document(key="search", kind="need", text="query", vector=query_vector)]
    else:
        queries = [d for d in viewer.documents if (
            d.kind in ("offer", "product") if intent == Intent.buyer else
            d.kind == "offer" if intent == Intent.peer else
            d.kind == "need" and d.intent == intent)]
    expected = []
    for target in catalog:
        if (not target.source.active or target.source.region != "A"
                or target.source.id in hidden | {"c-005", viewer.source.id}):
            continue
        corpus = [d for d in target.documents if (
            d.kind == "need" and d.intent == Intent.supplier if intent == Intent.buyer
            else d.kind in ("offer", "product"))]
        options = [((float(np.dot(q.vector, d.vector)) + 1) / 2 * (0.85 + 0.15 * q.weight),
                    qi, d.key, q.key) for qi, q in enumerate(queries) for d in corpus]
        best = sorted(options, key=lambda x: (-x[0], x[1], x[2]))[0]
        expected.append((target.source.id, best[0], best[3], best[2]))
    expected.sort(key=lambda x: (-x[1], x[0]))
    expected = expected[:10]
    for page, documents, qbatch in [(1, 1, 1), (4, 3, 1), (100, 1024, 64)]:
        settings = Settings(_env_file=None, candidate_limit=10, retrieval_score_tolerance=1,
                            retrieval_client_batch_size=page,
                            retrieval_document_batch_size=documents,
                            retrieval_query_batch_size=qbatch)
        matches = ExactRetriever(repo, settings).retrieve(viewer, request, hidden, query_vector)
        assert [m.target.source.id for m in matches] == [e[0] for e in expected]
        assert [m.semantic for m in matches] == pytest.approx([e[1] for e in expected])
        assert [(m.query.key, m.document.key) for m in matches] == [e[2:] for e in expected]
    repo.close()


def test_retrieval_reads_updates_and_rejects_mixed_dimensions():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    catalog = make_catalog(2)
    for item in catalog:
        repo.put(clients, item.source.id, item.model_dump(mode="json"))
    retriever = ExactRetriever(repo, Settings(_env_file=None))
    request = RecommendRequest(client_id=catalog[0].source.id, intent=Intent.supplier)
    assert len(retriever.retrieve(catalog[0], request, set(), None)) == 1
    catalog[1].source.active = False
    repo.put(clients, catalog[1].source.id, catalog[1].model_dump(mode="json"))
    assert retriever.retrieve(catalog[0], request, set(), None) == []
    catalog[1].source.active = True
    for document in catalog[1].documents:
        document.vector = [1.0, 0.0]
    repo.put(clients, catalog[1].source.id, catalog[1].model_dump(mode="json"))
    with pytest.raises(ValueError, match="Incompatible embedding dimensions"):
        retriever.retrieve(catalog[0], request, set(), None)
    repo.close()
