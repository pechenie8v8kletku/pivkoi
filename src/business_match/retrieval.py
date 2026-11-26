"""Replaceable candidate retrieval with a bounded-memory, exact reference backend."""

import heapq
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from business_match.business_signals import support
from business_match.config import Settings
from business_match.schemas import Document, IndexedClient, Intent, RecommendRequest, utcnow
from business_match.storage import Repository


@dataclass
class CandidateMatch:
    target: IndexedClient
    intent: Intent
    query: Document
    document: Document
    semantic: float
    evidence_support: float = 0.0


class CandidateRetriever(Protocol):
    def retrieve(self, viewer: IndexedClient, request: RecommendRequest,
                 hidden: set[str], query_vector: list[float] | None) -> list[CandidateMatch]: ...


class ExactRetriever:
    """Scan SQL in pages, score vector blocks, retain only top-K companies per role.

    O(catalog documents) work remains: this is a reference backend, not an ANN index.
    The engine accepts another CandidateRetriever without changing ranking or feedback.
    """

    def __init__(self, repository: Repository, settings: Settings):
        self.repository, self.settings = repository, settings

    def retrieve(self, viewer, request, hidden, query_vector):
        now = utcnow()
        def usable(document):
            return not document.evidence or support(document, self.settings, now) > 0

        channels = [request.intent] if request.intent else list(Intent)
        queries = {}
        for intent in channels:
            if query_vector is not None:
                docs = [Document(key="search", text=request.query, kind="need",
                                 vector=query_vector, intent=intent)]
            elif intent == Intent.buyer:
                docs = [d for d in viewer.documents if d.kind in ("offer", "product")]
            elif intent == Intent.peer:
                docs = [d for d in viewer.documents if d.kind == "offer"]
            else:
                docs = [d for d in viewer.documents if d.kind == "need" and d.intent == intent]
            docs = [d for d in docs if usable(d)
                    and not (intent == Intent.peer and d.evidence)]
            if docs:
                queries[intent] = docs
        heaps = {intent: [] for intent in queries}
        if not queries:
            return []
        excluded = set(request.exclude_client_ids) | hidden | {request.client_id}
        sequence = 0
        for batch in self.repository.iter_client_batches(
            self.settings.retrieval_client_batch_size, region=request.region, city=request.city
        ):
            targets = [t for t in batch if t.source.active and t.source.id not in excluded]
            for intent, query_docs in queries.items():
                best = {}
                block = []

                def score_block(block=block, query_docs=query_docs, best=best, intent=intent):
                    if not block:
                        return
                    vectors = np.asarray([d.vector for _, d in block], dtype=float)
                    size = self.settings.retrieval_query_batch_size
                    for offset in range(0, len(query_docs), size):
                        qdocs = query_docs[offset:offset + size]
                        qvectors = np.asarray([d.vector for d in qdocs], dtype=float)
                        if vectors.shape[1] != qvectors.shape[1]:
                            raise ValueError(
                                "Incompatible embedding dimensions; reindex all clients")
                        weight = self.settings.query_importance_weight
                        weights = np.asarray([1 - weight + weight * q.weight for q in qdocs])
                        scores = (np.clip(qvectors @ vectors.T, -1, 1) + 1) / 2
                        scores *= weights[:, None]
                        evidence = np.maximum(
                            np.asarray([support(q, self.settings, now)
                                        for q in qdocs])[:, None],
                            np.asarray([support(d, self.settings, now)
                                        for _, d in block])[None, :])
                        evidence *= scores
                        effective = scores + self.settings.business_signal_boost * evidence
                        positions = effective.argmax(axis=0)
                        for column, (target, document) in enumerate(block):
                            qi = int(positions[column])
                            score = float(effective[qi, column])
                            old = best.get(target.source.id)
                            # Stable document-key order makes ties independent of block sizes.
                            order = (offset + qi, document.key)
                            if (old is None or score > old[0]
                                    or (score == old[0] and order < old[1])):
                                best[target.source.id] = (score, order, CandidateMatch(
                                    target, intent, qdocs[qi], document,
                                    float(scores[qi, column]), float(evidence[qi, column])))
                    block.clear()

                for target in targets:
                    for document in target.documents:
                        eligible = (document.kind == "need" and document.intent == Intent.supplier
                                    if intent == Intent.buyer
                                    else document.kind in ("offer", "product"))
                        if eligible and usable(document) and not (
                            intent == Intent.peer and document.evidence
                        ):
                            block.append((target, document))
                            if len(block) >= self.settings.retrieval_document_batch_size:
                                score_block()
                score_block()
                heap = heaps[intent]
                for score, _, match in best.values():
                    sequence += 1
                    entry = (score, -sequence, match)
                    if len(heap) < self.settings.candidate_limit:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
        result = []
        for heap in heaps.values():
            if heap:
                ordered = sorted(heap, key=lambda entry: entry[:2], reverse=True)
                threshold = ordered[0][0] - self.settings.retrieval_score_tolerance
                result.extend(match for score, _, match in ordered if score >= threshold)
        return result
