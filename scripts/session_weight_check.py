"""Compare personal weights on complete candidate snapshots from a long session."""
import argparse
import json
import random
import sqlite3
from pathlib import Path

from business_match.config import Settings
from business_match.ranking import ScoredCandidate, SemanticRanker, SlatePolicy
from business_match.schemas import Recommendation


def compare(directory):
    trace = json.loads((directory / "trace.json").read_text())
    settings = Settings(_env_file=None, ranker_graph_weight=0, linucb_alpha=0, exploration_rate=0)
    prior = SemanticRanker(settings)
    policy = SlatePolicy(settings)
    feeds = [s for s in trace["slates"] if s["body"]["client_id"] == "bakery"
             and not s["body"].get("query") and s["weight"] == 0.5]
    rows = []
    with sqlite3.connect((directory / "market.db").resolve().as_uri() + "?mode=ro",
                         uri=True) as db:
        for stage, feed in zip(trace["checks"], feeds, strict=True):
            response = feed["result"]
            saved = db.execute("SELECT payload FROM bm_slates WHERE id = ?",
                               (response["request_id"],)).fetchone()
            items = json.loads(saved[0])["items"]
            cards = {i["client_id"]: i for i in response["items"]}
            if len(cards) != response["diagnostics"]["candidates"] or len(cards) != len(items):
                raise ValueError("The snapshot must cover the entire candidate pool")
            weights = {"0.5": stage["actual"]}
            for weight in (1.0, 1.5, 2.0):
                pool = []
                for item in items:
                    cid = item["client_id"]
                    features = item["features"]
                    base, _ = prior.score(features, "bakery", cid)
                    # With alpha=0 the recorded score is exactly base + weight * residual.
                    residual = (item["score"] - base) / feed["weight"]
                    card = Recommendation.model_validate(cards[cid])
                    card.score = base + weight * residual
                    pool.append(ScoredCandidate(card, item["vector"], features,
                                                item["semantic_score"],
                                                max(features[3], 1 - features[8])))
                rng = random.Random(feed["body"]["seed"])
                slate = policy.select(pool, feed["body"]["limit"], rng)
                weights[str(weight)] = [c.item.client_id for c in slate]
            rows.append(dict(stage=stage["label"], family=stage["family"], events=stage["events"],
                             weights=weights))
    path = directory / "weight-comparison.json"
    path.write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    compare(parser.parse_args().directory)
