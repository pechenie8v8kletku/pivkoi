from datetime import timedelta

import pytest

from business_match.config import Settings
from business_match.ranking import FeatureBuilder, LinUCBRanker
from business_match.schemas import Intent, utcnow
from business_match.storage import Repository, ranker_states


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
