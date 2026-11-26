"""Paired real HTTP ablation: identical training, behavioral retrieval off/on."""

import argparse
import asyncio
import json
import math
import os
import sqlite3
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
from real_model_audit import service

from business_match.schemas import utcnow


async def run(args):
    root = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-interests")
    root.mkdir(parents=True)
    labels = json.loads((args.fixtures / "labels.json").read_text(encoding="utf-8"))
    catalog = json.loads((args.fixtures / "catalog.json").read_text(encoding="utf-8"))
    scenarios = {
        "bakery": ("pack", "flour"),
        "salon": ("cosmetics", "cleaning"),
        "furniture": ("wood", "metal"),
    }
    queries = {
        topic: next(c["description"] for c in catalog if c["id"] == topic + "-0")
        for pair in scenarios.values()
        for topic in pair
    }
    heldout = {c["id"] for c in catalog if c["id"].rsplit("-", 1)[-1] in {"4", "5", "6", "7"}}
    excluded = sorted({c["id"] for c in catalog} - heldout)
    rows, responses, events = [], [], []
    checks = {}
    original = os.environ.get("INTEREST_RETRIEVAL_ENABLED")

    def clone(source, folder, clear=False):
        folder.mkdir()
        with sqlite3.connect(source) as src, sqlite3.connect(folder / "service.db") as dst:
            src.backup(dst)
            if clear:
                for table in (
                    "bm_events",
                    "bm_slates",
                    "bm_ranker_states",
                    "bm_relation_edges",
                    "bm_relation_observations",
                ):
                    dst.execute("DELETE FROM " + table)
                dst.commit()

    def states(folder):
        with sqlite3.connect(folder / "service.db") as db:
            return dict(db.execute("select id,payload from bm_ranker_states"))

    async def call(api, body):
        r = await api.post("/v1/search" if body.get("query") else "/v1/feed", json=body)
        r.raise_for_status()
        return r.json()

    async def probe(api, mode):
        for user, (topic, previous) in scenarios.items():
            for kind in ("feed", "heldout", "search_current", "search_other"):
                for seed in (11, 42):
                    body = {"client_id": user, "intent": "supplier", "limit": 10, "seed": seed}
                    wanted = previous if kind == "search_other" else topic
                    if kind == "heldout":
                        body["exclude_client_ids"] = excluded
                    if kind.startswith("search"):
                        body["query"] = queries[wanted]
                    start = perf_counter()
                    data = await call(api, body)
                    items = data["items"]
                    gains = [int(wanted in labels[x["client_id"]]["offers"]) for x in items]
                    ideal_n = sum(
                        wanted in labels[c["id"]]["offers"]
                        for c in catalog
                        if c.get("active", True)
                        and c["id"] != user
                        and (kind != "heldout" or c["id"] in heldout)
                    )
                    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, ideal_n)))
                    good = [
                        bool(set(labels[user]["needs"]) & set(labels[x["client_id"]]["offers"]))
                        for x in items
                    ]
                    row = {
                        "mode": mode,
                        "user": user,
                        "kind": kind,
                        "seed": seed,
                        "returned": len(items),
                        "relevant": sum(good),
                        "focus_count": sum(gains),
                        "ndcg": sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / ideal
                        if ideal
                        else 0,
                        "latency_ms": (perf_counter() - start) * 1000,
                        "candidates": data["diagnostics"]["candidates"],
                        "interest_candidates": data["diagnostics"]["interest_candidates"],
                        "observations": data["diagnostics"]["ranker_observations"],
                        "reranked": data["diagnostics"]["reranked"],
                        "ids": [x["client_id"] for x in items],
                    }
                    rows.append(row)
                    responses.append({**row, "response": data})

    async with httpx.AsyncClient(timeout=30) as api:
        health_before = (await api.get(args.models + "/health")).json()
    try:
        os.environ["INTEREST_RETRIEVAL_ENABLED"] = "false"
        training = root / "training"
        clone(args.source, training, clear=True)
        async with service(training, "on", args) as api:
            await probe(api, "cold")
            for user, (topic, _) in scenarios.items():
                data = await call(
                    api,
                    {
                        "client_id": user,
                        "intent": "supplier",
                        "limit": 10,
                        "seed": 42,
                        "query": queries[topic],
                        "exclude_client_ids": sorted(heldout),
                    },
                )
                targets = [x for x in data["items"] if topic in labels[x["client_id"]]["offers"]][
                    :3
                ]
                assert targets, (user, topic)
                for target in targets:
                    for kind in ("impression", "open", "like", "contact"):
                        event = {
                            "event_id": str(uuid4()),
                            "client_id": user,
                            "kind": kind,
                            "target_client_id": target["client_id"],
                            "request_id": data["request_id"],
                            "visible_fraction": 1,
                        }
                        # Exercise SKU attribution where the relevant SKU is actually shown.
                        if any(p["product"]["id"] == topic for p in target["products"]):
                            event["product_id"] = topic
                        r = await api.post("/v1/events", json={"events": [event, event]})
                        r.raise_for_status()
                        assert [x["status"] for x in r.json()["results"]] == [
                            "accepted",
                            "duplicate",
                        ]
                        events.append(event)
                print("TRAINED", user, len(targets), flush=True)
        original_states = states(training)
        for mode in ("disabled", "enabled"):
            output = root / mode
            clone(training / "service.db", output)
            os.environ["INTEREST_RETRIEVAL_ENABLED"] = str(mode == "enabled").lower()
            async with service(output, "on", args) as api:
                await probe(api, mode)
            checks[mode + "_same_learned_state"] = states(output) == original_states
            print("EVALUATED", mode, len(rows), flush=True)
        os.environ["INTEREST_RETRIEVAL_ENABLED"] = "true"
        async with service(root / "enabled", "on", args) as api:
            body = {"client_id": "bakery", "intent": "supplier", "limit": 10, "seed": 42}
            restored = await call(api, body)
            prior = next(
                r
                for r in rows
                if r["mode"] == "enabled"
                and r["user"] == "bakery"
                and r["kind"] == "feed"
                and r["seed"] == 42
            )
            checks["restart_same_order"] = prior["ids"] == [
                x["client_id"] for x in restored["items"]
            ]
    finally:
        if original is None:
            os.environ.pop("INTEREST_RETRIEVAL_ENABLED", None)
        else:
            os.environ["INTEREST_RETRIEVAL_ENABLED"] = original
    index = {(r["mode"], r["user"], r["kind"], r["seed"]): r for r in rows}
    checks["search_identical"] = all(
        r["ids"] == index["disabled", r["user"], r["kind"], r["seed"]]["ids"]
        for r in rows
        if r["mode"] == "enabled" and r["kind"].startswith("search")
    )
    checks["heldout_no_training"] = not heldout.intersection(e["target_client_id"] for e in events)
    checks["bge_no_fallback"] = all(r["reranked"] for r in rows if r["returned"])
    async with httpx.AsyncClient(timeout=30) as api:
        health_after = (await api.get(args.models + "/health")).json()
    checks["zero_new_oom"] = health_after["ooms"] == health_before["ooms"]
    report = {
        "path": str(root),
        "requests": len(rows),
        "events": len(events),
        "checks": checks,
        "rows": rows,
        "models_before": health_before,
        "models_after": health_after,
        "limitations": [
            "Synthetic labels and scripted extraction",
            "Three clients, two seeds",
            "Same learned state and feedback in both ablations",
            "One current topic per client, not long-term drift validation",
        ],
    }
    for name, value in [("report", report), ("responses", responses), ("events", events)]:
        (root / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {"path": str(root), "requests": len(rows), "events": len(events), "checks": checks}
        ),
        flush=True,
    )
    if not all(checks.values()):
        raise SystemExit("Technical check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="http://127.0.0.1:8011")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
