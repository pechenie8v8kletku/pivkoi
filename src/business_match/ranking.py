"""Replaceable feature, scoring, and slate-policy stages of the recommendation cascade."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from business_match.config import Settings
from business_match.schemas import Intent, utcnow
from business_match.storage import Repository, ranker_states

FEATURE_NAMES = (
    "semantic", "rerank", "graph", "freshness", "same_region", "same_city",
    "in_stock", "catalog_size", "seen", "new_candidate",
    "intent_supplier", "intent_buyer", "intent_partner", "intent_peer",
)
EVENT_REWARDS = {
    "open": 0.55,
    "like": 0.8,
    "contact": 1.0,
    "hide": 0.0,
}


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
    def build(self, *, semantic: float, rerank: float, graph: float, freshness: float,
              same_region: bool, same_city: bool, in_stock: bool, catalog_size: int,
              seen: float, is_new: bool, intent: Intent) -> list[float]:
        values = [
            semantic, rerank, graph, freshness, float(same_region), float(same_city),
            float(in_stock), min(1.0, np.log1p(catalog_size) / np.log(21)),
            seen, float(is_new),
        ]
        values.extend(float(intent == channel) for channel in Intent)
        return values


class SemanticRanker:
    """Deterministic cold-start ranker; all optional priors are configurable."""

    version = "semantic-v1"

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
        return score, 0.0

    def update(self, client_id: str, key: str, features: list[float], reward: float,
               occurred_at: datetime, terminal: bool = False, conn=None) -> None:
        return None


class LinUCBRanker(SemanticRanker):
    """Hybrid contextual linear bandit, with semantic score as the cold-start prior."""

    version = "hybrid-linucb-v1"

    def __init__(self, settings: Settings, repository: Repository):
        super().__init__(settings)
        self.repository = repository
        self._request_model = None

    def prepare(self, client_id: str) -> None:
        """Build posterior once per recommendation request, rather than per candidate."""
        payload = self.repository.get(ranker_states, client_id)
        observations = (payload or {}).get("observations", {})
        dimension = len(FEATURE_NAMES)
        matrix = self.settings.linucb_regularization * np.eye(dimension)
        target = np.zeros(dimension)
        now = utcnow()
        for observation in observations.values():
            features = np.asarray(observation["features"], dtype=float)
            if features.shape != (dimension,):
                continue
            age = max(
                0.0,
                (now - datetime.fromisoformat(observation["occurred_at"])).total_seconds(),
            )
            decay = 0.5 ** (age / (86400 * self.settings.history_half_life_days))
            matrix += decay * np.outer(features, features)
            target += decay * features * observation["residual"]
        self._request_model = (client_id, np.linalg.inv(matrix), target)

    def score(self, features: list[float], client_id: str, key: str):
        prior, _ = super().score(features, client_id, key)
        if self._request_model is None or self._request_model[0] != client_id:
            self.prepare(client_id)
        _, inverse, target = self._request_model
        x = np.asarray(features, dtype=float)
        personal_residual = float(x @ inverse @ target)
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
        observations = payload["observations"]
        previous = observations.get(key)
        timestamp = occurred_at.isoformat()
        if previous:
            if previous.get("terminal") and not terminal:
                return
            if previous.get("terminal") and terminal and previous["reward"] >= reward:
                return
            if previous.get("occurred_at", "") > timestamp:
                return
            if not terminal and not previous.get("terminal"):
                reward = max(previous["reward"], reward)
            baseline = previous["baseline"]
            features = previous["features"]
        else:
            prior, _ = super().score(features, client_id, key)
            baseline = float(0.8 / (1 + np.exp(-4 * (prior - 0.5))))
        observations[key] = {
            "features": features,
            "reward": reward,
            "residual": reward - baseline,
            "baseline": baseline,
            "occurred_at": timestamp,
            "terminal": terminal,
        }
        # Keep online state bounded; raw event log remains the retraining source.
        if len(observations) > 1000:
            ordered = sorted(observations, key=lambda k: observations[k]["occurred_at"])
            for expired in ordered[:len(observations) - 1000]:
                del observations[expired]
        payload["observations"] = observations
        payload["version"] = self.version
        self.repository.put(ranker_states, client_id, payload, conn=conn)


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
        selected: list[ScoredCandidate] = []
        used = set()
        while pool and len(selected) < limit:
            available = [candidate for candidate in pool
                         if candidate.item.client_id not in used]
            if not available:
                break

            def utility(candidate):
                redundancy = max((
                    (float(np.dot(candidate.vector, other.vector)) + 1) / 2
                    for other in selected
                ), default=0)
                repeats = sum(other.item.segment == candidate.item.segment for other in selected)
                return (candidate.item.score - self.settings.diversity_weight * redundancy
                        - self.settings.diversity_weight * 0.25 * repeats)

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
        return selected


def observation_key(target_id: str, intent: str, occurred_at: datetime) -> str:
    return f"{target_id}|{intent}|{occurred_at.date().isoformat()}"
