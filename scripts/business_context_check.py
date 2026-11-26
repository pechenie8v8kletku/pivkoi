"""Synthetic contracts + deliberately imperfect extractor outputs, real service HTTP routes."""

import argparse
import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from business_match.app import create_app
from business_match.config import Settings
from business_match.mock_api import chat_completions, mock_embedding
from business_match.providers import Providers
from business_match.schemas import RecommendRequest, utcnow
from business_match.storage import Repository, ranker_states, slates


class ScriptedGateway:
    """Hand-authored fixture output; this does not evaluate Qwen's semantic accuracy."""

    def __init__(self):
        self.calls = {"extract": 0, "profile": 0, "embedding": 0}
        self.fail = False
        self.answers = {}

    def respond(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            self.calls["embedding"] += 1
            if self.fail:
                return httpx.Response(503, json={"error": "synthetic outage"})
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": mock_embedding(text)}
                        for i, text in enumerate(body["input"])
                    ]
                },
            )
        if request.url.path.endswith("chat/completions"):
            result = chat_completions(body)
            tool = result["choices"][0]["message"]["tool_calls"][0]["function"]
            args = json.loads(tool["arguments"])
            if "commercial" in args:
                self.calls["extract"] += 1
                source = json.loads(
                    next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
                )
                answer = self.answers.get(
                    source["subject"],
                    {"subject": "", "commercial": False, "role": "unknown", "confidence": 0},
                )
                tool["arguments"] = json.dumps(answer, ensure_ascii=False)
            else:
                self.calls["profile"] += 1
            return httpx.Response(200, json=result)
        return httpx.Response(503, json={"error": "rerank deliberately unavailable"})


async def run_scenario(output: Path, per_family: int = 10):
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{output / 'context.db'}",
        exploration_rate=0,
        linucb_alpha=0,
        ranker_graph_weight=0,
        ranker_freshness_weight=0,
        rerank_enabled=False,
        llm_rpm=100000,
        embedding_rpm=100000,
        embedding_tpm=100000000,
    )
    repo = Repository(settings.database_url)
    repo.initialize()
    provider = Providers(settings, repo)
    gateway = ScriptedGateway()
    await provider.http.aclose()
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(gateway.respond))
    families = {
        "flour": "Мука пшеничная",
        "oven": "Печи тестомесы",
        "pack": "Упаковка бумажная",
        "coffee": "Кофе зерновой",
        "metal": "Металл стальной",
        "cosmetics": "Косметика шампуни",
        "logistics": "Перевозка грузов",
        "cold": "Холодильные витрины",
    }
    catalog = [
        {
            "id": key,
            "name": "Пекарня " + key,
            "segment": "Пекарня",
            "description": "Хлеб выпечка",
            "explicit_needs": [families["pack"]],
        }
        for key in ("viewer", "control")
    ]
    for family, subject in families.items():
        for i in range(per_family):
            catalog.append(
                {
                    "id": f"{family}-{i}",
                    "name": f"{subject} {i}",
                    "segment": family,
                    "description": subject,
                    "region": "Казань" if i % 2 else "Москва",
                    "products": [{"id": "sku", "name": subject, "stock": 20}],
                }
            )
    catalog.append(
        {
            "id": "historical-seller",
            "name": "Промсервис",
            "description": "Производственное предприятие",
        }
    )
    snapshots = {c["id"]: c for c in catalog}
    checks, trace, timings, sent = {}, [], [], []

    def record(
        key,
        family,
        role="buyer",
        kind="contract",
        status=None,
        group=None,
        days=2,
        confidence=0.9,
        answer_role=None,
        purpose="trade",
    ):
        raw = f"Закрытый документ SECRET-{key}: {families.get(family, family)}"
        gateway.answers[raw] = {
            "subject": families.get(family, family),
            "role": answer_role or role,
            "commercial": True,
            "confidence": confidence,
        }
        return {
            "id": key,
            "kind": kind,
            "subject": raw,
            "role": role,
            "purpose": purpose,
            "status": status or ("active" if kind == "contract" else "settled"),
            "occurred_at": (utcnow() - timedelta(days=days)).isoformat(),
            "transaction_group_id": group,
        }

    app = create_app(settings, providers=provider)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://service"
            ) as api:

                async def put(client, records=None, sequence=1, source="erp", expected=200):
                    payload = dict(snapshots[client])
                    if records is not None:
                        payload["business_context"] = {
                            "source_id": source,
                            "source_sequence": sequence,
                            "records": records,
                        }
                    sent.append(payload)
                    response = await api.put(f"/v1/clients/{client}", json=payload)
                    assert response.status_code == expected, response.text
                    return response.json()

                async def feed(client="viewer", intent="supplier", **extra):
                    start = time.perf_counter()
                    response = await api.post(
                        "/v1/search" if extra.get("query") else "/v1/feed",
                        json={
                            "client_id": client,
                            "intent": intent,
                            "seed": 31,
                            "limit": 20,
                            **extra,
                        },
                    )
                    assert response.status_code == 200, response.text
                    timings.append((time.perf_counter() - start) * 1000)
                    data = response.json()
                    assert (
                        "SECRET-" not in response.text
                        and "transaction_group_id" not in response.text
                    )
                    return data

                def count(slate, family):
                    return sum(x["client_id"].startswith(family + "-") for x in slate["items"])

                async def click(slate, target, kind):
                    event = {
                        "event_id": str(uuid4()),
                        "request_id": slate["request_id"],
                        "client_id": "viewer",
                        "target_client_id": target,
                        "kind": kind,
                        "visible_fraction": 1,
                        "dwell_ms": 12000,
                    }
                    response = await api.post("/v1/events", json={"events": [event, event]})
                    assert [r["status"] for r in response.json()["results"]] == [
                        "accepted",
                        "duplicate",
                    ], response.text

                for client in snapshots:
                    await put(client)
                baseline, control = await feed(), await feed("control")
                profile_calls = gateway.calls["profile"]
                flour = record("flour-contract", "flour", group="purchase-flour")
                await put("viewer", [flour])
                first = await feed()
                checks["new_need_reaches_candidates"] = count(first, "flour") > count(
                    baseline, "flour"
                )
                checks["insight_visible"] = any(
                    x["insights"] for x in first["items"] if x["client_id"].startswith("flour-")
                )
                checks["profile_not_regenerated"] = gateway.calls["profile"] == profile_calls
                checks["no_fake_click_training"] = repo.get(ranker_states, "viewer") is None
                public = (await api.get("/v1/clients/viewer")).json()
                checks["context_private_and_no_fake_products"] = (
                    "business_context" not in public["source"]
                    and not public["source"]["products"]
                    and "SECRET-" not in json.dumps(public)
                )
                previous_calls = dict(gateway.calls)
                checks["same_snapshot_noop"] = not (await put("viewer", [flour]))["changed"]
                await put("viewer")
                checks["legacy_write_preserves_context"] = count(await feed(), "flour") == count(
                    first, "flour"
                )
                checks["retries_do_not_call_models"] = previous_calls == gateway.calls
                await put("viewer", [], sequence=0, expected=409)
                await put("viewer", [], sequence=1, expected=409)
                checks["stale_and_conflicting_versions_rejected"] = True
                # Contract + split payments: all settle the same trade, not independent intent.
                payments = [
                    record(f"payment-{i}", "flour", kind="transfer", group="purchase-flour")
                    for i in range(12)
                ]
                for payment in payments:
                    payment["occurred_at"] = flour["occurred_at"]
                await put("viewer", [flour, *payments], sequence=2)
                duplicated = await feed()
                checks["split_payments_do_not_multiply_boost"] = all(
                    abs(a["score"] - b["score"]) < 1e-5
                    for a, b in zip(first["items"], duplicated["items"], strict=True)
                )
                checks["single_group_document"] = (
                    len([d for d in repo.client("viewer").documents if d.evidence]) == 1
                )
                calls = gateway.calls["extract"]
                await put("viewer", [{**flour, "status": "cancelled"}, *payments], sequence=3)
                checks["cancel_contract_with_payments_withdraws"] = count(
                    await feed(), "flour"
                ) == count(baseline, "flour")
                checks["status_only_no_extraction"] = gateway.calls["extract"] == calls
                ambiguous = record("unknown", "Назначение не указано", role="unknown")
                weak = record("weak", "cosmetics", confidence=0.3)
                wrong = record("wrong-role", "metal", answer_role="seller")
                tax = record("tax", "coffee", purpose="noncommercial")
                pending = record("pending", "cold", kind="transfer", status="pending")
                refunded = record("refund", "oven", kind="transfer", status="refunded")
                expired = record("expired", "logistics", days=10)
                expired["valid_until"] = (utcnow() - timedelta(days=1)).isoformat()
                await put(
                    "viewer", [ambiguous, weak, wrong, tax, pending, refunded, expired], sequence=4
                )
                checks["noisy_extractions_do_not_pollute"] = not any(
                    d.evidence for d in repo.client("viewer").documents
                )
                old = record("old", "flour", days=500)
                await put("viewer", [old], sequence=5)
                checks["old_evidence_expires_at_read"] = count(await feed(), "flour") == count(
                    baseline, "flour"
                )
                oven = record("oven", "oven")
                await put("viewer", [oven], sequence=6)
                oven_feed = await feed()
                checks["switch_contract_changes_feed"] = count(oven_feed, "oven") > count(
                    baseline, "oven"
                )
                target = next(
                    x["client_id"] for x in oven_feed["items"] if x["client_id"].startswith("oven-")
                )
                rows = repo.get(slates, oven_feed["request_id"])["items"]
                features = next(r["features"] for r in rows if r["client_id"] == target)
                ranker = app.state.engine.ranker
                before_score = ranker.score(features, "viewer", "")[0]
                before_calls = dict(gateway.calls)
                for kind in ("impression", "dwell", "open", "like", "contact"):
                    await click(oven_feed, target, kind)
                    trace.append({"event": kind, "slate": await feed()})
                checks["live_contact_learning"] = (
                    ranker.score(features, "viewer", "")[0] > before_score
                )
                checks["browsing_without_model_calls"] = gateway.calls == before_calls
                await click(oven_feed, target, "hide")
                checks["hide_beats_evidence"] = target not in {
                    x["client_id"] for x in (await feed())["items"]
                }
                checks["other_user_isolated"] = [
                    x["client_id"] for x in (await feed("control"))["items"]
                ] == [x["client_id"] for x in control["items"]]
                await put("viewer", [flour], source="bank")
                await put("viewer", [], sequence=7)
                checks["source_scoped_clear"] = any(
                    d.evidence for d in repo.client("viewer").documents
                )
                await put("viewer", [], sequence=2, source="bank")
                checks["all_sources_clear"] = not any(
                    d.evidence for d in repo.client("viewer").documents
                )
                # Seller records create historical offers, never catalog inventory.
                await put("historical-seller", [record("sale", "flour", role="seller")])
                await put("viewer", [flour], sequence=8)
                selling = await feed(query="Мука пшеничная")
                checks["historical_offer_searchable"] = any(
                    x["client_id"] == "historical-seller" for x in selling["items"]
                )
                checks["seller_no_invented_sku"] = not repo.client(
                    "historical-seller"
                ).source.products
                buyers = await feed("flour-0", "buyer")
                checks["purchase_reverse_buyer_matching"] = any(
                    x["client_id"] == "viewer" for x in buyers["items"]
                )
                # Fixed query/pool: isolate the prior from candidate-generation effects.
                settings.business_signal_boost = 0
                without_boost = await feed("flour-0", "buyer")
                settings.business_signal_boost = 0.12
                supported = next(x["score"] for x in buyers["items"] if x["client_id"] == "viewer")
                unsupported = next(
                    x["score"] for x in without_boost["items"] if x["client_id"] == "viewer"
                )
                checks["bounded_prior_changes_matching_score"] = 0 < supported - unsupported <= 0.12
                settings.business_signals_enabled = False
                disabled = await feed("flour-0", "buyer")
                checks["disable_removes_evidence_contribution"] = all(
                    not x["insights"] for x in disabled["items"]
                )
                disabled_matches = app.state.engine.retriever.retrieve(
                    repo.client("flour-0"),
                    RecommendRequest(client_id="flour-0", intent="buyer"),
                    set(),
                    None,
                )
                checks["disable_excludes_private_documents"] = all(
                    not m.document.evidence and not m.query.evidence and m.evidence_support == 0
                    for m in disabled_matches
                )
                settings.business_signals_enabled = True
                suppliers = await feed("control", query="Мука пшеничная")
                checks["purchaser_not_flour_supplier"] = not any(
                    x["client_id"] == "viewer" for x in suppliers["items"]
                )
                filtered = await feed(region="Москва", exclude_client_ids=["flour-0"])
                checks["filters_beat_context"] = all(
                    x["region"] == "Москва" and x["client_id"] != "flour-0"
                    for x in filtered["items"]
                )
                saved_revision = repo.client("viewer").revision
                gateway.fail = True
                novel = record("outage", "Новая редкая линия оборудования")
                await put("viewer", [novel], sequence=9, expected=502)
                gateway.fail = False
                checks["failed_embedding_rolls_back_snapshot"] = (
                    repo.client("viewer").revision == saved_revision
                    and next(p for p in repo.contexts("viewer") if p["source_id"] == "erp")[
                        "source_sequence"
                    ]
                    == 8
                )
                malformed = {**snapshots["viewer"], "business_context": None}
                checks["null_context_rejected"] = (
                    await api.put("/v1/clients/viewer", json=malformed)
                ).status_code == 422
                await put("viewer", [flour, flour], sequence=9, expected=422)
                checks["duplicate_record_ids_rejected"] = True

                # Concurrent snapshots serialize at the API boundary; max sequence wins.
                async def race(seq):
                    return await api.put(
                        "/v1/clients/viewer",
                        json={
                            **snapshots["viewer"],
                            "business_context": {
                                "source_id": "erp",
                                "source_sequence": seq,
                                "records": [flour],
                            },
                        },
                    )

                results = await asyncio.gather(race(10), race(9))
                checks["concurrent_snapshot_monotonic"] = (
                    all(r.status_code in (200, 409) for r in results)
                    and next(p for p in repo.contexts("viewer") if p["source_id"] == "erp")[
                        "source_sequence"
                    ]
                    == 10
                )
                returned = await feed()
                checks["return_to_previous_interest"] = count(returned, "flour") > 0
                checks["hidden_stays_hidden_after_switch"] = target not in {
                    item["client_id"] for item in returned["items"]
                }
                trace.append({"phase": "return_to_flour", "slate": returned})
                settings.rerank_enabled = True
                settings.rerank_feed_enabled = True
                fallback = await feed()
                checks["reranker_outage_preserves_feed"] = (
                    bool(fallback["items"])
                    and not fallback["diagnostics"]["reranked"]
                    and count(fallback, "flour") > 0
                )
                settings.rerank_enabled = False
                # Multiple seeds with production exploration and UCB, same hard exclusions.
                settings.exploration_rate, settings.linucb_alpha = 0.1, 0.15
                for seed in range(5):
                    mixed = await feed(intent=None, seed=seed)
                    ids = [x["client_id"] for x in mixed["items"]]
                    checks[f"mixed_seed_{seed}"] = len(ids) == len(set(ids)) and target not in ids
                trace.extend(
                    [
                        {"phase": "before", "slate": baseline},
                        {"phase": "flour", "slate": first},
                        {"phase": "oven", "slate": oven_feed},
                    ]
                )
        # Recreate app/engine and use the same disk database, not just in-memory state.
        restarted = create_app(settings, providers=provider)
        async with restarted.router.lifespan_context(restarted):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted), base_url="http://restart"
            ) as api2:
                response = await api2.post(
                    "/v1/feed", json={"client_id": "viewer", "intent": "supplier", "seed": 31}
                )
                checks["restart_context_and_training"] = (
                    response.status_code == 200
                    and any(x["client_id"].startswith("flour-") for x in response.json()["items"])
                    and repo.get(ranker_states, "viewer") is not None
                )
        report = {
            "checks": checks,
            "companies": len(catalog),
            "calls": gateway.calls,
            "flour_before": count(baseline, "flour"),
            "flour_after": count(first, "flour"),
            "oven_before": count(baseline, "oven"),
            "oven_after": count(oven_feed, "oven"),
            "latency_ms": timings,
            "scope": "Hand-authored imperfect extractions; hashing embeddings; no real Qwen",
        }
        for name, value in [
            ("report", report),
            ("catalog", catalog),
            ("requests", sent),
            ("extractions", gateway.answers),
            ("journey", trace),
        ]:
            (output / f"{name}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return report
    finally:
        await provider.close()
        repo.close()


async def main(per_family):
    output = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-context")
    report = await run_scenario(output, per_family)
    print(json.dumps({**report, "artifacts": str(output)}, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("Business context regression")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=10)
    args = parser.parse_args()
    if args.per_family < 10:
        parser.error("--per-family must be at least 10")
    asyncio.run(main(args.per_family))
