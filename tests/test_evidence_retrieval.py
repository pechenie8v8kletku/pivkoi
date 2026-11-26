from datetime import timedelta

import numpy as np
import pytest

from business_match.config import Settings
from business_match.retrieval import ExactRetriever
from business_match.schemas import (
    BusinessProfile,
    ClientInput,
    Document,
    IndexedClient,
    RecommendRequest,
    utcnow,
)
from business_match.storage import Repository, clients


def test_evidence_retrieval_matches_independent_pairwise_reference(monkeypatch):
    now = utcnow()
    monkeypatch.setattr("business_match.retrieval.utcnow", lambda: now)
    rng = np.random.default_rng(192)
    catalog = []
    for i in range(19):
        docs = []
        for j in range(6):
            vector = rng.normal(size=8)
            vector /= np.linalg.norm(vector)
            docs.append(
                Document(
                    key=f"d-{j}",
                    text="topic",
                    kind="need" if i == 0 else "offer",
                    intent="supplier" if i == 0 else None,
                    vector=vector.tolist(),
                    weight=0.8,
                    evidence=j % 2 == 1,
                    support=0.7,
                    observed_at=now - timedelta(days=j * 7),
                )
            )
        catalog.append(
            IndexedClient(
                source=ClientInput(id=f"c-{i:02}", name="Company"),
                profile=BusinessProfile(summary="Company", offers=["Offer"]),
                documents=docs,
                revision="test",
                profile_revision="test",
            )
        )
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    try:
        for item in catalog:
            repo.put(clients, item.source.id, item.model_dump(mode="json"))
        expected = []
        for target in catalog[1:]:
            pairs = []
            for qi, q in enumerate(catalog[0].documents):
                for d in target.documents:
                    semantic = (np.clip(np.dot(q.vector, d.vector), -1, 1) + 1) / 2
                    semantic *= 0.85 + 0.15 * q.weight

                    def strength(doc):
                        age = (now - doc.observed_at).total_seconds() / 86400
                        return 0.7 * 0.5 ** (age / 60) if doc.evidence else 0

                    evidence = max(strength(q), strength(d)) * semantic
                    pairs.append((semantic + 0.12 * evidence, qi, d.key, semantic, evidence))
            best = sorted(pairs, key=lambda row: (-row[0], row[1], row[2]))[0]
            expected.append((target.source.id, *best))
        expected.sort(key=lambda row: (-row[1], row[0]))
        expected = expected[:10]
        for page, block, queries in [(1, 1, 1), (4, 7, 2), (100, 1024, 64)]:
            settings = Settings(
                _env_file=None,
                candidate_limit=10,
                retrieval_score_tolerance=1,
                retrieval_client_batch_size=page,
                retrieval_document_batch_size=block,
                retrieval_query_batch_size=queries,
            )
            matches = ExactRetriever(repo, settings).retrieve(
                catalog[0], RecommendRequest(client_id="c-00", intent="supplier"), set(), None
            )
            assert [m.target.source.id for m in matches] == [r[0] for r in expected]
            assert [m.semantic for m in matches] == pytest.approx([r[4] for r in expected])
            assert [m.evidence_support for m in matches] == pytest.approx([r[5] for r in expected])
            assert [(m.query.key, m.document.key) for m in matches] == [
                (f"d-{r[2]}", r[3]) for r in expected
            ]
    finally:
        repo.close()
