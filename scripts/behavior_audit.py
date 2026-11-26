"""Compare scripted browsing policies on fresh SQL clones; no model tuning."""

import argparse
import asyncio
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path
from uuid import uuid4

import httpx
from real_model_audit import service

from business_match.schemas import utcnow


async def run(args):
    root = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-behavior")
    root.mkdir(parents=True)
    labels = json.loads((args.fixtures / "labels.json").read_text(encoding="utf-8"))
    catalog = json.loads((args.fixtures / "catalog.json").read_text(encoding="utf-8"))
    journeys = {
        "bakery": ["oven", "pack", "flour"],
        "salon": ["cosmetics", "cleaning"],
        "furniture": ["wood", "metal"],
    }
    queries = {
        topic: next(x["description"] for x in catalog if x["id"] == topic + "-0")
        for topics in journeys.values()
        for topic in topics
    }
    train_ids = {x["id"] for x in catalog if x["id"].rsplit("-", 1)[-1] in {"0", "1", "2", "3"}}
    heldout = {x["id"] for x in catalog if x["id"].rsplit("-", 1)[-1] in {"4", "5", "6", "7"}}
    rows, responses, events, checks = [], [], [], []

    def save():
        for name, value in [
            ("rows", rows),
            ("responses", responses),
            ("events", events),
            ("checks", checks),
        ]:
            (root / (name + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    async def request(api, body, endpoint="/v1/feed"):
        r = await api.post(endpoint, json=body)
        r.raise_for_status()
        return r.json()

    async def probe(api, mode, user, phase, focus, excluded, hidden):
        for kind in ["feed", "heldout", "explicit_search"]:
            for seed in [11, 42, 79]:
                body = {"client_id": user, "intent": "supplier", "seed": seed, "limit": 10}
                if kind == "heldout":
                    body["exclude_client_ids"] = sorted(excluded)
                if kind == "explicit_search":
                    body["query"] = queries[focus]
                data = await request(api, body, "/v1/search" if "query" in body else "/v1/feed")
                items = data["items"]
                gains = [int(focus in labels[x["client_id"]]["offers"]) for x in items]
                good = [
                    bool(set(labels[user]["needs"]) & set(labels[x["client_id"]]["offers"]))
                    for x in items
                ]
                ideal_n = sum(
                    focus in labels[x["id"]]["offers"]
                    for x in catalog
                    if x.get("active", True)
                    and x["id"] != user
                    and x["id"] not in hidden
                    and (kind != "heldout" or x["id"] not in excluded)
                )
                ideal = sum(1 / math.log2(i + 2) for i in range(min(10, ideal_n)))
                dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
                rows.append(
                    {
                        "mode": mode,
                        "user": user,
                        "phase": phase,
                        "focus": focus,
                        "kind": kind,
                        "seed": seed,
                        "returned": len(items),
                        "relevant": sum(good),
                        "focus_count": sum(gains),
                        "focus_ndcg10": dcg / ideal if ideal else 0,
                        "needs_covered": len(
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
                        "reranked": data["diagnostics"]["reranked"],
                    }
                )
                checks.append(
                    {
                        "name": "hidden_absent",
                        "ok": not hidden.intersection(x["client_id"] for x in items),
                    }
                )
                if kind == "heldout":
                    checks.append(
                        {
                            "name": "heldout_only",
                            "ok": all(x["client_id"] in heldout for x in items),
                        }
                    )
                responses.append(
                    {
                        "mode": mode,
                        "user": user,
                        "phase": phase,
                        "kind": kind,
                        "seed": seed,
                        "response": data,
                    }
                )

    async with httpx.AsyncClient(timeout=30) as api:
        health_before = (await api.get(args.models + "/health")).json()
    for mode in ["on_semantic", "on"]:
        output = root / mode
        output.mkdir()
        with sqlite3.connect(args.source) as src, sqlite3.connect(output / "service.db") as dst:
            src.backup(dst)
            for table in [
                "bm_events",
                "bm_slates",
                "bm_ranker_states",
                "bm_relation_edges",
                "bm_relation_observations",
            ]:
                dst.execute("DELETE FROM " + table)
            dst.commit()
        async with service(output, mode, args) as api:
            for user, topics in journeys.items():
                rng = random.Random(42)
                hidden, interacted = set(), set()
                excluded = {x["id"] for x in catalog} - heldout
                for focus in topics:
                    await probe(api, mode, user, "cold_" + focus, focus, excluded, hidden)
                for phase, focus in enumerate(topics):
                    for turn in range(2):
                        # Browse feed first, then use explicit search for current task.
                        body = {
                            "client_id": user,
                            "intent": "supplier",
                            "limit": 10,
                            "seed": 42 + turn,
                            "exclude_client_ids": sorted(heldout),
                        }
                        if turn:
                            body["query"] = queries[focus]
                        data = await request(api, body, "/v1/search" if turn else "/v1/feed")
                        batch = []
                        positive = 0
                        for item in data["items"][:7]:
                            target = item["client_id"]
                            kinds = [("impression", 0)]
                            wanted = focus in labels[target]["offers"]
                            if wanted and target in train_ids and positive < 2:
                                positive += 1
                                kinds += [("open", 0), ("dwell", 12000)]
                                kinds += [("contact" if turn else "like", 0)]
                                interacted.add(target)
                            elif not wanted and rng.random() < 0.15:
                                kinds += [("open", 0), ("dwell", 500)]
                            if (
                                not set(labels[target]["offers"]) & set(labels[user]["needs"])
                                and not hidden
                            ):
                                kinds += [("hide", 0)]
                                hidden.add(target)
                            for kind, dwell in kinds:
                                batch.append(
                                    {
                                        "event_id": str(uuid4()),
                                        "request_id": data["request_id"],
                                        "client_id": user,
                                        "target_client_id": target,
                                        "kind": kind,
                                        "visible_fraction": 1,
                                        "dwell_ms": dwell,
                                    }
                                )
                        if batch:
                            result = await api.post("/v1/events", json={"events": batch})
                            result.raise_for_status()
                            assert all(x["status"] == "accepted" for x in result.json()["results"])
                            events.extend({"mode": mode, "phase": phase, **e} for e in batch)
                    await probe(api, mode, user, "after_" + focus, focus, excluded, hidden)
                    print(mode, user, focus, "events", len(events), "probes", len(rows), flush=True)
                    save()
                checks.append(
                    {"name": "no_training_on_heldout", "ok": not interacted.intersection(heldout)}
                )
                # Recheck earlier needs after the final change of interest.
                for focus in topics[:-1]:
                    await probe(api, mode, user, "final_" + focus, focus, excluded, hidden)
    async with httpx.AsyncClient(timeout=30) as api:
        health_after = (await api.get(args.models + "/health")).json()
    checks.append({"name": "zero_new_oom", "ok": health_after["ooms"] == health_before["ooms"]})
    checks.append(
        {"name": "bge_no_fallback", "ok": all(r["reranked"] for r in rows if r["returned"])}
    )
    summary = []
    for mode in ["on_semantic", "on"]:
        for kind in ["feed", "heldout", "explicit_search"]:
            for stage in ["cold", "after", "final"]:
                group = [
                    r
                    for r in rows
                    if r["mode"] == mode
                    and r["kind"] == kind
                    and r["phase"].startswith(stage + "_")
                ]
                n = sum(r["returned"] for r in group)
                summary.append(
                    {
                        "mode": mode,
                        "kind": kind,
                        "stage": stage,
                        "requests": len(group),
                        "precision": sum(r["relevant"] for r in group) / n if n else None,
                        "focus_per10": sum(r["focus_count"] for r in group) / len(group),
                        "focus_ndcg10": sum(r["focus_ndcg10"] for r in group) / len(group),
                    }
                )
    save()
    report = {
        "output": str(root),
        "summary": summary,
        "checks_pass": all(c["ok"] for c in checks),
        "event_counts": dict(Counter(e["kind"] for e in events)),
        "requests": len(rows),
        "models_before": health_before,
        "models_after": health_after,
        "limitations": [
            "Synthetic policy, not measured CTR",
            "Three users, one behavior seed",
            "Closed-loop exposures differ by ranker; not identical event replay",
            "Same catalog, not heldout industries; scripted extraction, SQLite",
        ],
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["checks_pass"]:
        raise SystemExit("Behavior checks failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="http://127.0.0.1:8011")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
