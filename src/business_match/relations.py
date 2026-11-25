"""Sparse, directed global graph of observed business-segment relationships."""

import hashlib
import json
import re
from datetime import datetime

from business_match.schemas import EventKind, FeedbackEvent, utcnow
from business_match.storage import Repository, relation_edges, relation_observations

PRIOR_RATE = 0.12
PRIOR_STRENGTH = 20.0
HALF_LIFE_DAYS = 60.0
POSITIVE_REWARD = {
    EventKind.dwell: 0.12,
    EventKind.open: 0.35,
    EventKind.like: 0.65,
    EventKind.contact: 1.0,
}


def canonical_segment(value: str) -> str:
    """Interim normalization only; replace with the product taxonomy when it exists."""
    return re.sub(r"\s+", " ", value.casefold()).strip() or "unknown"


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RelationGraph:
    """Accumulates bounded edge evidence transactionally with the original event."""

    def __init__(self, repository: Repository):
        self.repository = repository

    @staticmethod
    def edge_id(source_segment: str, target_segment: str, intent: str) -> str:
        key = json.dumps([
            canonical_segment(source_segment), canonical_segment(target_segment), intent
        ], ensure_ascii=False, separators=(",", ":"))
        return stable_id(key)

    def record(self, event: FeedbackEvent, slate: dict, item: dict, conn) -> None:
        source_segment = slate["source_segment"]
        target_segment = item["target_segment"]
        intent = item["intent"]
        edge_id = self.edge_id(source_segment, target_segment, intent)
        observation_id = stable_id(
            f"{event.request_id}:{event.target_client_id}:{intent}"
        )
        old = self.repository.get(relation_observations, observation_id, conn=conn)
        if old is None:
            updated = {
                "edge_id": edge_id,
                "source_segment": source_segment,
                "target_segment": target_segment,
                "intent": intent,
                "exposures": 1,
                "positive": 0.0,
                "negative": 0.0,
                "updated_at": utcnow().isoformat(),
            }
            exposure_delta = 1.0
            positive_delta = negative_delta = 0.0
        else:
            updated = old.copy()
            exposure_delta = 0.0
            positive_delta = negative_delta = 0.0

        reward = POSITIVE_REWARD.get(event.kind, 0.0)
        if event.kind == EventKind.dwell:
            reward *= min(event.dwell_ms / 15000, 1)
        if reward > updated["positive"]:
            positive_delta = reward - updated["positive"]
            updated["positive"] = reward
        if event.kind == EventKind.hide and updated["negative"] < 1:
            negative_delta = 1.0 - updated["negative"]
            updated["negative"] = 1.0
        if exposure_delta or positive_delta or negative_delta:
            updated["exposures"] += exposure_delta
            updated["updated_at"] = utcnow().isoformat()
            self.repository.put(
                relation_observations, observation_id, updated, conn=conn
            )
            edge = self.repository.get(relation_edges, edge_id, conn=conn) or {
                "source_segment": source_segment,
                "target_segment": target_segment,
                "intent": intent,
                "exposures": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "updated_at": utcnow().isoformat(),
            }
            elapsed = max(
                0.0,
                (utcnow() - datetime.fromisoformat(edge["updated_at"])).total_seconds(),
            )
            decay = 0.5 ** (elapsed / (HALF_LIFE_DAYS * 86400))
            edge["exposures"] = edge["exposures"] * decay + exposure_delta
            edge["positive"] = edge["positive"] * decay + positive_delta
            edge["negative"] = edge["negative"] * decay + negative_delta
            edge["updated_at"] = utcnow().isoformat()
            self.repository.put(relation_edges, edge_id, edge, conn=conn)

    def score(self, source_segment: str, target_segment: str, intent: str) -> tuple[float, bool]:
        edge_id = self.edge_id(source_segment, target_segment, intent)
        edge = self.repository.get(relation_edges, edge_id)
        if edge is None:
            return 0.0, False
        elapsed = max(
            0.0,
            (utcnow() - datetime.fromisoformat(edge["updated_at"])).total_seconds(),
        )
        decay = 0.5 ** (elapsed / (HALF_LIFE_DAYS * 86400))
        exposures = edge["exposures"] * decay
        positive = edge["positive"] * decay
        negative = edge["negative"] * decay
        # Smoothed global response rate; impressions only provide denominator/support,
        # never a per-user negative preference update.
        posterior = (PRIOR_RATE * PRIOR_STRENGTH + positive) / (
            PRIOR_STRENGTH + exposures + 2 * negative
        )
        confidence = exposures / (exposures + PRIOR_STRENGTH)
        return 0.12 * confidence * (posterior - PRIOR_RATE), True
