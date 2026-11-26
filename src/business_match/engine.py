"""Multi-vector retrieval, anchored feedback, channel allocation and diversity."""

import math
import random
from datetime import datetime, timedelta
from time import perf_counter
from uuid import uuid4

import numpy as np

from business_match.config import Settings
from business_match.interests import embedding_signature
from business_match.providers import Providers
from business_match.ranking import (
    FeatureBuilder,
    ScoredCandidate,
    SlatePolicy,
    create_ranker,
    observation_key,
)
from business_match.relations import RelationGraph
from business_match.retrieval import CandidateRetriever, ExactRetriever
from business_match.schemas import (
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
    def __init__(self, repository: Repository, providers: Providers, settings: Settings,
                 retriever: CandidateRetriever | None = None):
        self.repository, self.providers, self.settings = repository, providers, settings
        self.relation_graph = RelationGraph(repository, settings)
        self.retriever = (retriever if retriever is not None
                          else ExactRetriever(repository, settings))
        self.feature_builder = FeatureBuilder(settings)
        self.ranker = create_ranker(settings, repository)
        self.slate_policy = SlatePolicy(settings)

    def feedback_state(self, client_id: str):
        seen, hidden = {}, set()
        now = utcnow()
        for record in self.repository.user_events(client_id, self.settings.feedback_history_days):
            age = max(0, (now - datetime.fromisoformat(record["received_at"])).total_seconds())
            target = record["event"]["target_client_id"]
            if record["event"]["kind"] == "impression":
                seen[target] = max(seen.get(target, 0), math.exp(
                    -age / (self.settings.impression_decay_days * 86400)))
            if (record["event"]["kind"] == "hide"
                    and age < self.settings.hide_duration_days * 86400):
                hidden.add(target)
        return seen, hidden

    async def recommend(self, request: RecommendRequest) -> RecommendationResponse:
        started = perf_counter()
        viewer = self.repository.client(request.client_id)
        if viewer is None or not viewer.source.active:
            raise LookupError("Active viewer not found")
        seen, hidden = self.feedback_state(request.client_id)
        rng = random.Random(request.seed)
        query_vector = None
        if request.query:
            query_vector = (await self.providers.embed([request.query], query=True))[0]
        pool = []
        graph_edges_used = 0
        graph_cache = {}
        retrieval_started = perf_counter()
        matches = self.retriever.retrieve(viewer, request, hidden, query_vector)
        interest_candidates = sum(m.behavioral for m in matches)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        now = utcnow()
        for match in matches:
            target, intent = match.target, match.intent
            source = target.source
            matched_query, matched = match.query, match.document
            q, score = matched_query.vector, match.semantic
            published_at = target.content_updated_at or target.created_at
            age = max(0, (now - published_at).total_seconds() / 86400)
            freshness = 0.5 ** (age / self.settings.freshness_half_life_days)
            geo = bool(viewer.source.region and viewer.source.region == source.region)
            graph_key = (viewer.source.segment, source.segment, intent.value)
            if graph_key not in graph_cache:
                graph_cache[graph_key] = self.relation_graph.score(*graph_key)
            graph_delta, graph_used = graph_cache[graph_key]
            graph_edges_used += int(graph_used)
            by_id = {p.id: p for p in source.products if p.active and p.stock != 0}
            products = sorted(
                [ProductMatch(product=by_id[d.product_id],
                              score=(similarity(q, d.vector) + 1) / 2)
                 for d in target.documents if d.product_id in by_id],
                key=lambda p: p.score, reverse=True,
            )
            if products:
                threshold = products[0].score - self.settings.card_product_score_tolerance
                products = [p for p in products if p.score >= threshold]
            products = products[:self.settings.card_product_limit]
            reason = {
                Intent.supplier: "Возможный поставщик по вашему запросу",
                Intent.buyer: "Возможный покупатель вашего предложения",
                Intent.partner: "Возможный партнёр с дополняющим предложением",
                Intent.peer: "Компания с похожим предложением",
            }[intent]
            if match.behavioral:
                reason = "Похожее предложение по вашим лайкам и контактам"
            pool.append({
                "item": Recommendation(
                    client_id=source.id, name=source.name, segment=source.segment,
                    region=source.region, city=source.city, intent=intent, score=score,
                    reason=reason,
                    matched_need=("Сигнал хозяйственной деятельности" if matched_query.evidence
                                  else matched_query.text),
                    matched_text=("Сигнал хозяйственной деятельности" if matched.evidence
                                  else matched.text),
                    insights=(["Совпадение с наблюдаемой хозяйственной деятельностью"]
                              if match.evidence_support > 0 else []),
                    products=products,
                ),
                "vector": matched.vector, "query": q,
                "matched_query": matched_query.text, "matched_text": matched.text,
                "semantic": score, "freshness": freshness, "same_region": geo,
                "same_city": bool(viewer.source.city and viewer.source.city == source.city),
                "graph": graph_delta, "graph_known": graph_used,
                "seen": seen.get(source.id, 0), "is_new": source.id not in seen,
                "in_stock": bool(products), "catalog_size": len(source.products),
                "evidence_support": match.evidence_support,
                "interest_anchor": ({"text": matched.text, "vector": matched.vector,
                                     "product_id": matched.product_id,
                                     "signature": embedding_signature(self.settings)}
                                    if not matched.evidence else None),
                "product_anchors": {d.product_id: {"text": d.text, "vector": d.vector,
                    "product_id": d.product_id, "signature": embedding_signature(self.settings)}
                    for d in target.documents if not d.evidence
                    and d.product_id in {p.product.id for p in products}},
            })

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
        rerank_started = perf_counter()
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
        rerank_ms = (perf_counter() - rerank_started) * 1000
        ranking_started = perf_counter()
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
                evidence_support=candidate["evidence_support"],
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
        ranking_ms = (perf_counter() - ranking_started) * 1000
        slate_started = perf_counter()
        selected = self.slate_policy.select(
            prepared, request.limit, rng, enable_exploration=not bool(request.query),
            balance_segments=not bool(request.query),
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
                          "ranker_mode": self.settings.ranker_mode,
                          "retrieval_backend": type(self.retriever).__name__,
                          "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms,
                          "ranking_ms": ranking_ms,
                          "slate_ms": (perf_counter() - slate_started) * 1000},
        )
        anchors = {(c['item'].client_id, c['item'].intent): c for c in pool}
        response.diagnostics['interest_candidates'] = interest_candidates
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
                       "interest_anchor": anchors[c.item.client_id, c.item.intent][
                           'interest_anchor'],
                       "product_anchors": anchors[c.item.client_id, c.item.intent][
                           'product_anchors'],
                       "score": c.item.score} for i, c in enumerate(selected)],
            "client_id": request.client_id,
            "source_segment": viewer.source.segment,
        }, client_id=request.client_id)
        response.diagnostics["engine_total_ms"] = (perf_counter() - started) * 1000
        return response

    def ingest(self, event: FeedbackEvent) -> bool:
        slate = self.repository.get(slates, event.request_id)
        if not slate or slate["client_id"] != event.client_id:
            raise ValueError("Unknown request_id for this client")
        if slate.get("ranker_version") != self.ranker.version:
            raise ValueError("Recommendation model changed; refresh the feed before feedback")
        matched = next((c for c in slate["items"]
                        if c["client_id"] == event.target_client_id), None)
        if not matched:
            raise ValueError("Target was not included in this slate")
        if event.product_id and event.product_id not in matched["product_ids"]:
            raise ValueError("Product was not included in this card")
        if event.occurred_at > utcnow() + timedelta(
            seconds=self.settings.event_future_tolerance_seconds
        ):
            raise ValueError("Event timestamp is in the future")
        if event.occurred_at < datetime.fromisoformat(slate["created_at"]):
            raise ValueError("Event predates recommendation request")
        anchor = (matched.get('product_anchors', {}).get(event.product_id)
                  if event.product_id and matched['intent'] == Intent.supplier
                  else matched.get('interest_anchor'))
        payload = {"interest_anchor": anchor,
                   "event": event.model_dump(mode="json"), "received_at": utcnow().isoformat(),
                   "intent": matched["intent"], "vector": matched["vector"],
                   "features": matched["features"],
                   "position": matched["position"], "mode": slate["mode"],
                   "exploration": matched["exploration"],
                   "selection_probability": matched["selection_probability"]}
        with self.repository.transaction() as conn:
            inserted = self.repository.insert_event(event.event_id, event.client_id, payload, conn)
            if inserted:
                self.relation_graph.record(event, slate, matched, conn)
                reward = {kind: getattr(self.settings, f"reward_{kind}")
                          for kind in ("open", "like", "contact", "hide")}.get(event.kind.value)
                if event.kind.value == "dwell":
                    reward = (self.settings.reward_dwell_max * min(
                        1, event.dwell_ms / self.settings.dwell_saturation_ms)
                        if event.dwell_ms >= self.settings.dwell_min_ms else None)
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
