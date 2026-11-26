"""Replaceable feature, scoring, and slate-policy stages of the recommendation cascade."""

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np

from business_match.config import Settings
from business_match.providers import fingerprint
from business_match.schemas import Intent, utcnow
from business_match.storage import Repository, ranker_states

METADATA_FEATURE_NAMES = (
    "semantic", "rerank", "graph", "freshness", "same_region", "same_city",
    "in_stock", "catalog_size", "seen", "new_candidate",
    "intent_supplier", "intent_buyer", "intent_partner", "intent_peer",
    "evidence_support",
)


def feature_names(settings: Settings):
    return METADATA_FEATURE_NAMES + tuple(
        f"content_{intent.value}_{i}" for intent in Intent
        for i in range(settings.content_dimensions))


def training_signature(settings: Settings):
    """Prevent silently interpreting saved observations under a different feature/provider setup."""
    fields = (
        "content_dimensions", "content_projection_seed", "catalog_size_scale",
        "personal_metadata_scale", "model_api_mode", "embedding_model",
        "llm_model", "query_importance_weight",
        "baseline_ceiling", "baseline_slope", "baseline_midpoint",
        "reward_open", "reward_like", "reward_contact", "reward_hide",
        "reward_dwell_max", "dwell_min_ms", "dwell_saturation_ms",
        "ranker_rerank_weight", "ranker_graph_weight",
        "ranker_freshness_weight", "ranker_region_weight",
        "business_signals_enabled", "business_signal_boost",
        "business_signal_half_life_days", "business_signal_max_age_days",
        "business_signal_min_confidence", "business_signal_completed_weight",
    )
    values = {field: getattr(settings, field) for field in fields}
    values["provider"] = (settings.mock_api_base_url if settings.model_api_mode == "mock"
                          else settings.embedding_base_url)
    values["features"] = feature_names(settings)
    return fingerprint(values)[:16]


@lru_cache(maxsize=8)
def content_projection(dimensions: int, output_dimensions: int, seed: int):
    projection = np.random.default_rng(seed).standard_normal(
        (output_dimensions, dimensions)
    ) / np.sqrt(output_dimensions)
    projection.setflags(write=False)
    return projection


def personal_features(features, metadata_scale):
    """Keep shared relevance/quality metadata in the prior, learn personal content taste."""
    values = np.asarray(features, dtype=float).copy()
    values[:len(METADATA_FEATURE_NAMES)] *= metadata_scale
    return values


@dataclass
class ScoredCandidate:
    """Candidate plus decision-time features for training and audit."""

    item: object
    vector: list[float]
    features: list[float]
    semantic: float
    novelty: float
    selection_probability: float = 0.0


class FeatureBuilder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings(_env_file=None)

    def build(self, *, semantic: float, rerank: float, graph: float, freshness: float,
              same_region: bool, same_city: bool, in_stock: bool, catalog_size: int,
              seen: float, is_new: bool, intent: Intent,
              content_vector: list[float] | None = None,
              evidence_support: float = 0.0) -> list[float]:
        values = [
            semantic, rerank, graph, freshness, float(same_region), float(same_city),
            float(in_stock), min(1.0, np.log1p(catalog_size)
                                 / np.log1p(self.settings.catalog_size_scale)),
            seen, float(is_new),
        ]
        values.extend(float(intent == channel) for channel in Intent)
        values.append(evidence_support)
        # Stable random projection: let the personal model distinguish flour from packaging,
        # even if their scalar semantic scores and other metadata are identical.
        dimensions = self.settings.content_dimensions
        content = np.zeros(dimensions)
        if content_vector:
            projection = content_projection(
                len(content_vector), dimensions, self.settings.content_projection_seed)
            content = projection @ np.asarray(content_vector)
            norm = np.linalg.norm(content)
            if norm:
                content /= norm
        # Buying equipment must not teach the model to seek equipment vendors as buyers.
        for channel in Intent:
            values.extend(content.tolist() if channel == intent else [0.0] * dimensions)
        return values


class SemanticRanker:
    """Deterministic cold-start ranker; all optional priors are configurable."""

    version = "semantic-v2-evidence"

    def __init__(self, settings: Settings):
        self.settings = settings

    def prepare(self, client_id: str) -> None:
        return None

    def score(self, features: list[float], client_id: str, observation_key: str):
        semantic, rerank, graph, freshness, same_region = features[:5]
        score = semantic
        score += self.settings.ranker_rerank_weight * (2 * rerank - 1)
        score += self.settings.ranker_graph_weight * graph
        score += self.settings.ranker_freshness_weight * freshness
        score += self.settings.ranker_region_weight * same_region
        if self.settings.business_signals_enabled:
            score += self.settings.business_signal_boost * features[
                METADATA_FEATURE_NAMES.index("evidence_support")]
        return score, 0.0

    def update(self, client_id: str, key: str, features: list[float], reward: float,
               occurred_at: datetime, terminal: bool = False, conn=None) -> None:
        return None


class LinUCBRanker(SemanticRanker):
    """Personal residual linear UCB, with a semantic cold-start prior."""

    version = "personal-linear-ucb-v5"

    def __init__(self, settings: Settings, repository: Repository):
        super().__init__(settings)
        self.version = f"personal-linear-ucb-v5-{training_signature(settings)}"
        self.repository = repository
        self._request_model = None

    def prepare(self, client_id: str) -> None:
        """Build posterior once per recommendation request, rather than per candidate."""
        payload = self.repository.get(ranker_states, client_id)
        if payload and payload.get("version") != self.version:
            payload = None
        observations = (payload or {}).get("observations", {})
        dimension = len(feature_names(self.settings))
        matrix = self.settings.linucb_regularization * np.eye(dimension)
        target = np.zeros(dimension)
        now = utcnow()
        for observation in observations.values():
            features = personal_features(
                observation["features"], self.settings.personal_metadata_scale
            )
            if features.shape != (dimension,):
                continue
            age = max(
                0.0,
                (now - datetime.fromisoformat(observation["occurred_at"])).total_seconds(),
            )
            decay = 0.5 ** (age / (86400 * self.settings.history_half_life_days))
            matrix += decay * np.outer(features, features)
            target += decay * features * observation["residual"]
        inverse = np.linalg.inv(matrix)
        self._request_model = (client_id, inverse, inverse @ target)

    def score(self, features: list[float], client_id: str, key: str):
        prior, _ = super().score(features, client_id, key)
        if self._request_model is None or self._request_model[0] != client_id:
            self.prepare(client_id)
        _, inverse, theta = self._request_model
        x = personal_features(features, self.settings.personal_metadata_scale)
        personal_residual = float(x @ theta)
        uncertainty = float(np.sqrt(max(0.0, x @ inverse @ x)))
        # Do not treat the raw retrieval score as a probability or as a click.
        score = prior + self.settings.ranker_personal_weight * personal_residual
        score += self.settings.linucb_alpha * uncertainty
        return score, uncertainty

    def update(self, client_id: str, key: str, features: list[float], reward: float,
               occurred_at: datetime, terminal: bool = False, conn=None) -> None:
        payload = self.repository.get(ranker_states, client_id, conn=conn) or {
            "observations": {}, "version": self.version,
        }
        if payload.get("version") != self.version:
            payload = {"observations": {}, "version": self.version}
        observations = payload["observations"]
        previous = observations.get(key)
        timestamp = occurred_at.astimezone(UTC).isoformat()
        if previous:
            if previous.get("terminal") and not terminal:
                return
            if previous.get("occurred_at", "") > timestamp:
                return
            if not terminal and not previous.get("terminal"):
                reward = max(previous["reward"], reward)
            baseline = previous["baseline"]
            features = previous["features"]
        else:
            prior, _ = super().score(features, client_id, key)
            logit = self.settings.baseline_slope * (prior - self.settings.baseline_midpoint)
            baseline = float(self.settings.baseline_ceiling * np.exp(-np.logaddexp(0, -logit)))
        observations[key] = {
            "features": features,
            "reward": reward,
            "residual": reward - baseline,
            "baseline": baseline,
            "occurred_at": timestamp,
            "terminal": terminal,
        }
        # Keep online state bounded; raw event log remains the retraining source.
        if len(observations) > self.settings.ranker_max_observations:
            ordered = sorted(observations, key=lambda k: observations[k]["occurred_at"])
            for expired in ordered[:len(observations) - self.settings.ranker_max_observations]:
                del observations[expired]
        payload["observations"] = observations
        payload["version"] = self.version
        self.repository.put(ranker_states, client_id, payload, conn=conn)
        self._request_model = None


def create_ranker(settings: Settings, repository: Repository):
    if settings.ranker_mode == "linucb":
        return LinUCBRanker(settings, repository)
    return SemanticRanker(settings)


class SlatePolicy:
    """MMR diversity plus an epsilon-greedy, novelty-weighted exploration branch."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def select(self, candidates: list[ScoredCandidate], limit: int, rng,
               enable_exploration: bool = True) -> list[ScoredCandidate]:
        pool = list(candidates)
        if not pool:
            return []
        # One matrix-vector update per chosen slot: O(candidates * dimensions) memory.
        vectors = np.asarray([c.vector for c in pool])
        indices = {id(c): i for i, c in enumerate(pool)}
        redundancy_scores = np.zeros(len(pool))
        segment_counts = {}
        role_counts = {}
        selected: list[ScoredCandidate] = []
        used = set()
        while pool and len(selected) < limit:
            available = [candidate for candidate in pool
                         if candidate.item.client_id not in used]
            if not available:
                break
            # Caps are soft: use another eligible role when available, otherwise backfill.
            balanced = [c for c in available if role_counts.get(c.item.intent, 0) < max(
                1, int(np.ceil(limit * (self.settings.max_peer_share
                                       if c.item.intent == Intent.peer
                                       else self.settings.max_role_share))))]
            if balanced:
                available = balanced
            # Keep a relevant alternative visible without filling the feed with unrelated sectors.
            best_semantic = max(c.semantic for c in available)
            diverse = [c for c in available
                       if segment_counts.get(c.item.segment, 0)
                       < max(1, int(np.ceil(limit * self.settings.max_segment_share)))
                       and c.semantic >= best_semantic - self.settings.diversity_score_tolerance]
            if diverse:
                available = diverse

            def utility(candidate, redundancy_scores=redundancy_scores):
                redundancy = redundancy_scores[indices[id(candidate)]]
                repeats = segment_counts.get(candidate.item.segment, 0)
                return (candidate.item.score - self.settings.diversity_weight * redundancy
                        - self.settings.diversity_weight * self.settings.segment_repeat_weight
                        * repeats / max(1, limit))

            greedy_choice = max(available, key=utility)
            epsilon = self.settings.exploration_rate if enable_exploration else 0.0
            weights = [max(1e-9, c.novelty ** self.settings.exploration_freshness_weight)
                       for c in available]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            random_branch = epsilon > 0 and rng.random() < epsilon
            if random_branch:
                choice = rng.choices(available, weights=weights, k=1)[0]
            else:
                choice = greedy_choice
            index = available.index(choice)
            choice.selection_probability = (
                epsilon * probabilities[index]
                + (1 - epsilon) * float(choice is greedy_choice)
            )
            choice.item.exploration = random_branch
            selected.append(choice)
            used.add(choice.item.client_id)
            redundancy_scores = np.maximum(
                redundancy_scores, (vectors @ vectors[indices[id(choice)]] + 1) / 2
            )
            segment_counts[choice.item.segment] = segment_counts.get(choice.item.segment, 0) + 1
            role_counts[choice.item.intent] = role_counts.get(choice.item.intent, 0) + 1
        return selected


def observation_key(target_id: str, intent: str, occurred_at: datetime) -> str:
    return f"{target_id}|{intent}|{occurred_at.astimezone(UTC).date().isoformat()}"
