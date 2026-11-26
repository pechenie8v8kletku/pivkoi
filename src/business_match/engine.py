"""Multi-vector retrieval, anchored feedback, channel allocation and diversity."""

import math
import random
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np

from business_match.config import Settings
from business_match.providers import Providers
from business_match.ranking import (
    EVENT_REWARDS,
    FeatureBuilder,
    ScoredCandidate,
    SlatePolicy,
    create_ranker,
    observation_key,
)
from business_match.relations import RelationGraph
from business_match.schemas import (
    Document,
    FeedbackEvent,
    Intent,
    ProductMatch,
    Recommendation,
    RecommendationResponse,
    RecommendRequest,
    utcnow,
)
from business_match.storage import Repository, ranker_states, slates


def similarity(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("Incompatible embedding dimensions; reindex all clients")
    return float(np.clip(np.dot(a, b), -1, 1))


class Engine:
    def __init__(self, repository: Repository, providers: Providers, settings: Settings):
        self.repository, self.providers, self.settings = repository, providers, settings
        self.relation_graph = RelationGraph(repository)
        self.feature_builder = FeatureBuilder()
        self.ranker = create_ranker(settings, repository)
        self.slate_policy = SlatePolicy(settings)

    def feedback_state(self, client_id: str):
        seen, hidden = {}, set()
        now = utcnow()
        for record in self.repository.user_events(client_id):
            age = max(0, (now - datetime.fromisoformat(record["received_at"])).total_seconds())
            target = record["event"]["target_client_id"]
            if record["event"]["kind"] == "impression":
                seen[target] = max(seen.get(target, 0), math.exp(-age / 86400))
            if record["event"]["kind"] == "hide" and age < 7 * 86400:
                hidden.add(target)
        return seen, hidden

    async def recommend(self, request: RecommendRequest) -> RecommendationResponse:
        viewer = self.repository.client(request.client_id)
        if viewer is None or not viewer.source.active:
            raise LookupError("Active viewer not found")
        seen, hidden = self.feedback_state(request.client_id)
        rng = random.Random(request.seed)
        query_vector = None
        if request.query:
            query_vector = (await self.providers.embed([request.query], query=True))[0]
        candidates = []
        graph_edges_used = 0
        for target in self.repository.all_clients():
            source = target.source
            if (source.id == request.client_id or not source.active
                    or source.id in request.exclude_client_ids or source.id in hidden):
                continue
            if request.region and source.region != request.region:
                continue
            if request.city and source.city != request.city:
                continue
            target_offers = [d for d in target.documents if d.kind in ("offer", "product")]
            if not target_offers:
                continue
            channels = [request.intent] if request.intent else list(Intent)
            for intent in channels:
                if query_vector is not None:
                    queries = [Document(key="search", text=request.query, kind="need",
                                        vector=query_vector, intent=intent)]
                    corpus = target_offers if intent != Intent.buyer else [
                        d for d in target.documents
                        if d.kind == "need" and d.intent == Intent.supplier
                    ]
                elif intent == Intent.buyer:
                    queries = [d for d in viewer.documents if d.kind in ("offer", "product")]
                    corpus = [d for d in target.documents
                              if d.kind == "need" and d.intent == Intent.supplier]
                elif intent == Intent.peer:
                    queries = [d for d in viewer.documents if d.kind == "offer"]
                    corpus = target_offers
                else:
                    queries = [d for d in viewer.documents
                               if d.kind == "need" and d.intent == intent]
                    corpus = target_offers
                if not queries or not corpus:
                    continue
                best = None
                for query in queries:
                    q = query.vector
                    for document in corpus:
                        sim = (similarity(q, document.vector) + 1) / 2
                        value = sim * (0.85 + 0.15 * query.weight)
                        if best is None or value > best[0]:
                            best = (value, query, document, q)
                score, matched_query, matched, q = best
                published_at = target.content_updated_at or target.created_at
                age = max(0, (utcnow() - published_at).total_seconds() / 86400)
                freshness = 0.5 ** (age / self.settings.freshness_half_life_days)
                geo = bool(viewer.source.region and viewer.source.region == source.region)
                graph_delta, graph_used = self.relation_graph.score(
                    viewer.source.segment, source.segment, intent.value
                )
                graph_edges_used += int(graph_used)
                products = sorted(
                    [ProductMatch(product=p, score=(similarity(q, d.vector) + 1) / 2)
                     for p in source.products if p.active and p.stock != 0
                     for d in target.documents if d.product_id == p.id],
                    key=lambda p: p.score, reverse=True,
                )[:3]
                reason = {
                    Intent.supplier: "Potential supplier for this need",
                    Intent.buyer: "Potential buyer for your offer",
                    Intent.partner: "Potential complementary business partner",
                    Intent.peer: "Company with a similar offer",
                }[intent]
                candidates.append({
                    "item": Recommendation(
                        client_id=source.id, name=source.name, segment=source.segment,
                        region=source.region, city=source.city, intent=intent, score=score,
                        reason=reason, matched_need=matched_query.text, matched_text=matched.text,
                        products=products,
                    ),
                    "vector": matched.vector, "query": q,
                    "matched_query": matched_query.text,
                    "matched_text": matched.text,
                    "semantic": score,
                    "freshness": freshness,
                    "same_region": geo,
                    "same_city": bool(viewer.source.city and viewer.source.city == source.city),
                    "graph": graph_delta,
                    "graph_known": graph_used,
                    "seen": seen.get(source.id, 0),
                    "is_new": source.id not in seen,
                    "in_stock": bool(products),
                    "catalog_size": len(source.products),
                })
        candidates.sort(key=lambda c: c["semantic"], reverse=True)
        # Keep a candidate-generation budget per intent, separate from ranking.
        pool = []
        for intent in Intent:
            pool.extend([c for c in candidates if c["item"].intent == intent]
                        [:self.settings.candidate_limit])

        # Cross-encoder scores the exact need/offer (or query/offer) pair, including feed.
        reranked = False
        use_reranker = self.settings.rerank_enabled and (
            request.query or self.settings.rerank_feed_enabled
        )
        rerank_candidates = (
            sorted(pool, key=lambda c: c["semantic"], reverse=True)[:self.settings.rerank_top_k]
            if use_reranker else []
        )
        rerank_scores = None
        if rerank_candidates:
            pairs = [(c["matched_query"][:1000], c["matched_text"][:1500])
                     for c in rerank_candidates]
            rerank_scores = await self.providers.rerank_pairs(pairs)
            if rerank_scores is not None:
                # Per-request percentile avoids assuming provider scores are calibrated.
                # Equal cross-encoder scores must not create arbitrary ranking differences.
                ranks = {
                    i: (sum(s < score for s in rerank_scores)
                        + (sum(s == score for s in rerank_scores) - 1) / 2)
                    / (len(rerank_scores) - 1) if len(rerank_scores) > 1 else 0.5
                    for i, score in enumerate(rerank_scores)
                }
                for index, candidate in enumerate(rerank_candidates):
                    candidate["rerank"] = ranks[index]
                reranked = True
        prepared = []
        self.ranker.prepare(request.client_id)
        for candidate in pool:
            rerank_feature = candidate.get("rerank", 0.5)
            features = self.feature_builder.build(
                semantic=candidate["semantic"], rerank=rerank_feature,
                graph=candidate["graph"], freshness=candidate["freshness"],
                same_region=candidate["same_region"], same_city=candidate["same_city"],
                in_stock=candidate["in_stock"], catalog_size=candidate["catalog_size"],
                seen=candidate["seen"], is_new=candidate["is_new"],
                intent=candidate["item"].intent,
                content_vector=candidate["vector"],
            )
            score, _ = self.ranker.score(
                features, request.client_id, candidate["item"].client_id
            )
            candidate["item"].score = score
            prepared.append(ScoredCandidate(
                item=candidate["item"], vector=candidate["vector"], features=features,
                semantic=candidate["semantic"],
                novelty=max(candidate["freshness"], 1 - candidate["seen"]),
            ))
        selected = self.slate_policy.select(
            prepared, request.limit, rng, enable_exploration=not bool(request.query)
        )
        request_id = str(uuid4())
        ranker_payload = self.repository.get(ranker_states, request.client_id) or {}
        response = RecommendationResponse(
            request_id=request_id, mode="search" if request.query else "feed",
            items=[c.item for c in selected],
            diagnostics={"candidates": len(pool), "reranked": reranked,
                         "eligible_clients": len({c.item.client_id for c in prepared}),
                         "graph_edges_used": graph_edges_used,
                         "ranker_observations": len(ranker_payload.get("observations", {})),
                         "ranker_mode": self.settings.ranker_mode},
        )
        self.repository.put(slates, request_id, {
            "created_at": utcnow().isoformat(), "mode": response.mode,
            "policy_version": response.policy_version,
            "ranker_version": self.ranker.version,
            "items": [{"client_id": c.item.client_id, "intent": c.item.intent.value,
                       "position": i, "vector": c.vector,
                       "target_segment": c.item.segment,
                       "features": c.features,
                       "selection_probability": c.selection_probability,
                       "semantic_score": c.semantic,
                       "product_ids": [p.product.id for p in c.item.products],
                       "exploration": c.item.exploration,
                       "score": c.item.score} for i, c in enumerate(selected)],
            "client_id": request.client_id,
            "source_segment": viewer.source.segment,
        }, client_id=request.client_id)
        return response

    def ingest(self, event: FeedbackEvent) -> bool:
        slate = self.repository.get(slates, event.request_id)
        if not slate or slate["client_id"] != event.client_id:
            raise ValueError("Unknown request_id for this client")
        matched = next((c for c in slate["items"]
                        if c["client_id"] == event.target_client_id), None)
        if not matched:
            raise ValueError("Target was not included in this slate")
        if event.product_id and event.product_id not in matched["product_ids"]:
            raise ValueError("Product was not included in this card")
        if event.occurred_at > utcnow() + timedelta(minutes=5):
            raise ValueError("Event timestamp is in the future")
        if event.occurred_at < datetime.fromisoformat(slate["created_at"]):
            raise ValueError("Event predates recommendation request")
        payload = {"event": event.model_dump(mode="json"), "received_at": utcnow().isoformat(),
                   "intent": matched["intent"], "vector": matched["vector"],
                   "features": matched["features"],
                   "position": matched["position"], "mode": slate["mode"],
                   "exploration": matched["exploration"],
                   "selection_probability": matched["selection_probability"]}
        with self.repository.transaction() as conn:
            inserted = self.repository.insert_event(event.event_id, event.client_id, payload, conn)
            if inserted:
                self.relation_graph.record(event, slate, matched, conn)
                reward = EVENT_REWARDS.get(event.kind.value)
                if event.kind.value == "dwell":
                    reward = (min(0.35, 0.35 * event.dwell_ms / 15000)
                              if event.dwell_ms >= 2000 else None)
                if reward is not None:
                    self.ranker.update(
                        event.client_id,
                        observation_key(
                            event.target_client_id, matched["intent"], event.occurred_at
                        ),
                        matched["features"], reward, event.occurred_at,
                        terminal=event.kind.value in ("contact", "hide"), conn=conn,
                    )
        return inserted
