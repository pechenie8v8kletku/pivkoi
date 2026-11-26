"""Evaluate personal score weights with frozen learned state over real HTTP."""

import argparse
import asyncio
import json
import math
import os
import sqlite3
from pathlib import Path

from real_model_audit import service

from business_match.schemas import utcnow


async def run(args):
    root = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-score-sweep")
    root.mkdir(parents=True)
    labels = json.loads((args.fixtures / "labels.json").read_text(encoding="utf-8"))
    catalog = json.loads((args.fixtures / "catalog.json").read_text(encoding="utf-8"))
    topics = {
        "bakery": ("flour", "oven"),
        "salon": ("cleaning", "cosmetics"),
        "furniture": ("metal", "wood"),
    }
    heldout = {x["id"] for x in catalog if x["id"].rsplit("-", 1)[-1] in {"4", "5", "6", "7"}}
    all_ids = {x["id"] for x in catalog}
    rows, responses = [], []
    original = os.environ.get("RANKER_PERSONAL_WEIGHT")
    try:
        for weight in args.weights:
            output = root / str(weight)
            output.mkdir()
            with sqlite3.connect(args.source) as src, sqlite3.connect(output / "service.db") as dst:
                src.backup(dst)
                initial_states = dict(dst.execute("select id,payload from bm_ranker_states"))
            os.environ["RANKER_PERSONAL_WEIGHT"] = str(weight)
            async with service(output, "on", args) as api:
                for user, (current, previous) in topics.items():
                    for kind in ["feed", "heldout", "search_current", "search_previous"]:
                        for seed in [11, 42]:
                            topic = previous if kind == "search_previous" else current
                            body = {
                                "client_id": user,
                                "intent": "supplier",
                                "seed": seed,
                                "limit": 10,
                            }
                            if kind == "heldout":
                                body["exclude_client_ids"] = sorted(all_ids - heldout)
                            if kind.startswith("search"):
                                body["query"] = next(
                                    x["description"] for x in catalog if x["id"] == topic + "-0"
                                )
                            r = await api.post(
                                "/v1/search" if "query" in body else "/v1/feed", json=body
                            )
                            r.raise_for_status()
                            data = r.json()
                            items = data["items"]
                            relevance = [
                                bool(
                                    set(labels[user]["needs"])
                                    & set(labels[x["client_id"]]["offers"])
                                )
                                for x in items
                            ]
                            gains = [int(topic in labels[x["client_id"]]["offers"]) for x in items]
                            # Same denominator across weights; hide state is identical.
                            eligible = [
                                x
                                for x in catalog
                                if x.get("active", True)
                                and x["id"] != user
                                and (kind != "heldout" or x["id"] in heldout)
                            ]
                            n = sum(topic in labels[x["id"]]["offers"] for x in eligible)
                            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, n)))
                            row = {
                                "weight": weight,
                                "user": user,
                                "kind": kind,
                                "seed": seed,
                                "returned": len(items),
                                "relevant": sum(relevance),
                                "focus_count": sum(gains),
                                "focus_ndcg": sum(g / math.log2(i + 2) for i, g in enumerate(gains))
                                / ideal
                                if ideal
                                else 0,
                                "topics_covered": len(
                                    set().union(
                                        *(
                                            set(labels[x["client_id"]]["offers"])
                                            & set(labels[user]["needs"])
                                            for x in items
                                        )
                                    )
                                )
                                if items
                                else 0,
                                "ids": [x["client_id"] for x in items],
                                "observations": data["diagnostics"]["ranker_observations"],
                                "reranked": data["diagnostics"]["reranked"],
                            }
                            rows.append(row)
                            responses.append({**row, "response": data})
            with sqlite3.connect(output / "service.db") as db:
                assert dict(db.execute("select id,payload from bm_ranker_states")) == initial_states
            (root / "rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (root / "responses.json").write_text(
                json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("DONE", weight, "requests", len(rows), flush=True)
    finally:
        if original is None:
            os.environ.pop("RANKER_PERSONAL_WEIGHT", None)
        else:
            os.environ["RANKER_PERSONAL_WEIGHT"] = original
    summary = []
    for weight in args.weights:
        for kind in ["feed", "heldout", "search_current", "search_previous"]:
            group = [r for r in rows if r["weight"] == weight and r["kind"] == kind]
            total = sum(r["returned"] for r in group)
            summary.append(
                {
                    "weight": weight,
                    "kind": kind,
                    "returned": total,
                    "precision": sum(r["relevant"] for r in group) / total if total else None,
                    "focus_ndcg": sum(r["focus_ndcg"] for r in group) / len(group),
                    "focus_count": sum(r["focus_count"] for r in group) / len(group),
                    "topics_covered": sum(r["topics_covered"] for r in group) / len(group),
                }
            )
    report = {
        "path": str(root),
        "requests": len(rows),
        "summary": summary,
        "checks": {
            "trained_state_used": all(r["observations"] > 0 for r in rows),
            "bge_used": all(r["reranked"] for r in rows if r["returned"]),
            "learned_state_unchanged": True,
        },
        "limitations": [
            "Frozen end-of-journey state, no retraining at each weight",
            "Three synthetic users, two evaluation seeds",
            "Identical graph, hides and exploration settings across weights",
            "Topic NDCG denominator includes hidden relevant suppliers",
            "No production setting changed",
        ],
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="http://127.0.0.1:8011")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.5, 1, 2])
    asyncio.run(run(parser.parse_args()))
