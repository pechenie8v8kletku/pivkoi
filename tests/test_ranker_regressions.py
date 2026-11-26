import random
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

import pytest

from business_match.config import Settings
from business_match.ranking import FeatureBuilder, LinUCBRanker, ScoredCandidate, SlatePolicy
from business_match.schemas import Intent, utcnow
from business_match.storage import Repository, ranker_states


def test_configured_features_invalidate_incompatible_saved_training():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    settings = Settings(_env_file=None, content_dimensions=8, ranker_max_observations=2,
                        linucb_alpha=0)
    builder = FeatureBuilder(settings)
    features = builder.build(semantic=0.6, rerank=0.5, graph=0, freshness=0.5,
                             same_region=False, same_city=False, in_stock=True, catalog_size=2,
                             seen=0, is_new=True, intent=Intent.supplier, content_vector=[1, 0])
    ranker = LinUCBRanker(settings, repo)
    assert len(features) == 14 + len(Intent) * 8
    for i in range(3):
        ranker.update("u", str(i), features, 1, utcnow(), terminal=True)
    assert set(repo.get(ranker_states, "u")["observations"]) == {"1", "2"}
    changed = settings.model_copy(update={"content_projection_seed": 7})
    other = LinUCBRanker(changed, repo)
    assert other.version != ranker.version
    assert other.score(features, "u", "")[0] == other.score(features, "fresh", "")[0]
    other.update("u", "new", features, 0, utcnow(), terminal=True)
    assert set(repo.get(ranker_states, "u")["observations"]) == {"new"}
    repo.close()


def test_content_learning_is_personal_and_hide_overrides_contact():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    settings = Settings(linucb_alpha=0)
    ranker = LinUCBRanker(settings, repo)
    builder = FeatureBuilder()

    def features(vector):
        return builder.build(
            semantic=0.6, rerank=0.5, graph=0, freshness=0.5,
            same_region=False, same_city=False, in_stock=True, catalog_size=1,
            seen=0, is_new=True, intent=Intent.supplier, content_vector=vector,
        )

    flour, packaging = features([1, 0]), features([0, 1])
    before = [ranker.score(x, "viewer", "")[0] for x in (flour, packaging)]
    now = utcnow()
    ranker.update("viewer", "flour", flour, 1, now, terminal=True)
    after = [ranker.score(x, "viewer", "")[0] for x in (flour, packaging)]
    assert after[0] - before[0] > after[1] - before[1]
    assert ranker.score(flour, "other", "")[0] == pytest.approx(before[0])
    ranker.update("viewer", "flour", flour, 0, now + timedelta(seconds=1), terminal=True)
    assert repo.get(ranker_states, "viewer")["observations"]["flour"]["reward"] == 0
    assert ranker.score(flour, "viewer", "")[0] < before[0]
    # Delayed positive event cannot undo a newer explicit hide.
    ranker.update("viewer", "flour", flour, 1, now, terminal=True)
    assert repo.get(ranker_states, "viewer")["observations"]["flour"]["reward"] == 0
    repo.close()


def test_content_feedback_is_directional():
    repo = Repository("sqlite:///:memory:")
    repo.initialize()
    ranker = LinUCBRanker(Settings(_env_file=None, linucb_alpha=0), repo)
    def features(intent):
        return FeatureBuilder().build(
            semantic=0.6, rerank=0.5, graph=0, freshness=0.5, same_region=False,
            same_city=False, in_stock=True, catalog_size=1, seen=0, is_new=True,
            intent=intent, content_vector=[1, 0])
    supplier, buyer = features(Intent.supplier), features(Intent.buyer)
    before = [ranker.score(x, "u", "")[0] for x in (supplier, buyer)]
    ranker.update("u", "equipment", supplier, 1, utcnow(), terminal=True)
    delta = [ranker.score(x, "u", "")[0] - b for x, b in zip(
        (supplier, buyer), before, strict=True)]
    assert delta[0] > 10 * abs(delta[1])
    repo.close()


def test_slate_preserves_relevant_alternatives_and_role_balance():
    settings = Settings(_env_file=None, exploration_rate=0)
    def candidate(index, segment, intent, score):
        return ScoredCandidate(
            item=SimpleNamespace(client_id=str(index), segment=segment, intent=intent,
                                 score=score, exploration=False),
            vector=[1.0, 0.0], features=[], semantic=0.8, novelty=1.0)
    suppliers = [candidate(i, "equipment" if i < 20 else "flour",
                            Intent.supplier, 1.0 if i < 20 else 0.65) for i in range(40)]
    policy = SlatePolicy(settings)
    selected = policy.select(suppliers, 20, random.Random(42))
    assert Counter(c.item.segment for c in selected) == {"equipment": 12, "flour": 8}
    mixed = suppliers + [candidate(40 + i, "bakery", Intent.peer, 2.0) for i in range(20)]
    mixed += [candidate(60 + i, "shop", Intent.buyer, 0.6) for i in range(20)]
    selected = policy.select(mixed, 20, random.Random(42))
    assert Counter(c.item.intent for c in selected) == {
        Intent.peer: 4, Intent.supplier: 12, Intent.buyer: 4}
    # A single eligible sector must still fill the page: the cap is soft.
    assert len(policy.select(suppliers[:20], 20, random.Random(42))) == 20
