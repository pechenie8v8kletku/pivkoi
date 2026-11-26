"""Real TCP service/model audit with durable state and controlled chat extraction."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from business_match.config import Settings
from business_match.ranking import LinUCBRanker
from business_match.schemas import utcnow
from business_match.storage import Repository, slates


@asynccontextmanager
async def service(output, mode, args):
    env = {
        **os.environ,
        "MODEL_API_MODE": "external",
        "DATABASE_URL": f"sqlite:///{(output / 'service.db').resolve().as_posix()}",
        "LLM_BASE_URL": args.models + "/v1",
        "EMBEDDING_BASE_URL": args.models + "/v1",
        "RERANK_URL": args.models + "/rerank",
        "BUSINESS_SIGNALS_ENABLED": str(mode != "off").lower(),
        "RERANK_ENABLED": str(mode != "on_no_rerank").lower(),
        "RANKER_MODE": "semantic" if mode == "on_semantic" else "linucb",
        "LLM_RPM": "100000",
        "EMBEDDING_RPM": "100000",
        "EMBEDDING_TPM": "100000000",
        "RERANK_RPM": "100000",
        "RERANK_TPM": "100000000",
        "HTTP_TIMEOUT": "120",
        "SERVICE_API_KEY": "local-audit-key",
        "PYTHONIOENCODING": "utf-8",
    }
    log = (output / (mode + "-service.log")).open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "business_match.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workers",
            "1",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{args.port}",
            headers={"X-API-Key": "local-audit-key"},
            timeout=180,
        ) as api:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError("Service exited; inspect " + str(log.name))
                try:
                    (await api.get("/health")).raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.25)
            else:
                raise RuntimeError("Service startup timed out")
            yield api
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()


async def run(args):
    output = Path("local_experiments") / (utcnow().strftime("%Y%m%d-%H%M%S") + "-real-models")
    output.mkdir(parents=True)

    def load(name):
        return json.loads((args.fixtures / (name + ".json")).read_text(encoding="utf-8"))

    catalog, cases, labels = load("catalog"), load("cases"), load("labels")
    responses, rows, events, learning = [], [], [], []
    checks = {}
    async with httpx.AsyncClient(timeout=180) as model_api:
        health_before = (await model_api.get(args.models + "/health")).json()
        assert health_before["ready"] and health_before["reranker_model"]
        assert health_before["embedding_model"] == "Qwen/Qwen3-Embedding-4B"

    def relevant(case, item):
        src, target = labels[case["client_id"]], labels[item["client_id"]]
        if case.get("query"):
            return item["intent"] == "supplier" and bool(set(target["offers"]) & set(case["truth"]))
        if item["intent"] == "supplier":
            return bool(set(src["needs"]) & set(target["offers"]))
        if item["intent"] == "buyer":
            return bool(set(src["offers"]) & set(target["needs"]))
        if item["intent"] == "peer":
            return src["peer"] == target["peer"] or bool(set(src["offers"]) & set(target["offers"]))
        return False

    async def evaluate(api, case, mode, seed=42):
        body = {k: v for k, v in case.items() if k in ("client_id", "intent", "query", "region")}
        start = time.perf_counter()
        response = await api.post(
            "/v1/search" if case.get("query") else "/v1/feed",
            json={**body, "seed": seed, "limit": 10},
        )
        response.raise_for_status()
        data = response.json()
        assert "PRIVATE-" not in response.text
        ids = [x["client_id"] for x in data["items"]]
        assert len(ids) == len(set(ids)) and "inactive" not in ids
        if case.get("region"):
            assert all(x["region"] == case["region"] for x in data["items"])
        good = [relevant(case, x) for x in data["items"]]
        rows.append(
            {
                "mode": mode,
                "case": case["id"],
                "kind": case["kind"],
                "seed": seed,
                "returned": len(good),
                "correct": sum(good),
                "p10": sum(good) / 10,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "reranked": data["diagnostics"]["reranked"],
                "bad": [x["client_id"] for x, g in zip(data["items"], good, strict=True) if not g],
            }
        )
        responses.append({"mode": mode, "case": case["id"], "seed": seed, "response": data})
        return data

    def save(name, value):
        (output / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async with service(output, "on", args) as api:
        for i, payload in enumerate(catalog):
            (await api.put("/v1/clients/" + payload["id"], json=payload)).raise_for_status()
            if i % 25 == 0:
                print("INDEXED", i + 1, "/", len(catalog), flush=True)
    for mode in ("off", "on_no_rerank", "on_semantic", "on"):
        async with service(output, mode, args) as api:
            for case in cases:
                for seed in (11, 42, 79):
                    await evaluate(api, case, mode, seed)
            print("EVALUATED", mode, len(rows), flush=True)
        save("rows", rows)
        save("responses", responses)

    # Fixed probes before/after real event ingestion: not a change in seed or inventory.
    probe = next(c for c in cases if c["id"] == "feed_bakery_supplier")
    control = {**probe, "id": "control", "client_id": "bakery-control"}
    hidden = None
    audit_repo = Repository(f"sqlite:///{(output / 'service.db').resolve().as_posix()}")
    audit_settings = Settings(
        model_api_mode="external",
        embedding_base_url=args.models + "/v1",
        llm_base_url=args.models + "/v1",
        business_signals_enabled=True,
    )

    def fixed_scores(feature_rows, client_id="bakery"):
        ranker = LinUCBRanker(audit_settings, audit_repo)
        ranker.prepare(client_id)
        return {r["client_id"]: ranker.score(r["features"], client_id, "")[0] for r in feature_rows}

    async with service(output, "on", args) as api:
        control_before = await evaluate(api, control, "control_before")
        control_rows = audit_repo.get(slates, control_before["request_id"])["items"]
        control_scores = fixed_scores(control_rows, "bakery-control")
        for topic, case_id in [
            ("oven", "search_oven_exact_supplier"),
            ("pack", "search_pack_supplier"),
            ("flour", "search_flour_exact_supplier"),
        ]:
            case = next(c for c in cases if c["id"] == case_id)
            before = await evaluate(api, case, "before_" + topic)
            targets = [x for x in before["items"] if topic in labels[x["client_id"]]["offers"]][:3]
            before_scores = {x["client_id"]: x["score"] for x in before["items"]}
            fixed_rows = audit_repo.get(slates, before["request_id"])["items"]
            fixed_before = fixed_scores(fixed_rows)
            for item in targets:
                for kind in ("impression", "dwell", "open", "like", "contact"):
                    event = {
                        "event_id": str(uuid4()),
                        "request_id": before["request_id"],
                        "client_id": "bakery",
                        "target_client_id": item["client_id"],
                        "kind": kind,
                        "visible_fraction": 1,
                        "dwell_ms": 15000,
                    }
                    result = await api.post("/v1/events", json={"events": [event, event]})
                    result.raise_for_status()
                    assert [r["status"] for r in result.json()["results"]] == [
                        "accepted",
                        "duplicate",
                    ]
                    events.append(event)
            after = await evaluate(api, case, "after_" + topic)
            after_scores = {x["client_id"]: x["score"] for x in after["items"]}
            fixed_after = fixed_scores(fixed_rows)
            learning.append(
                {
                    "topic": topic,
                    "trained_targets": len(targets),
                    "fixed_feature_deltas": {
                        x["client_id"]: fixed_after[x["client_id"]] - fixed_before[x["client_id"]]
                        for x in targets
                    },
                    "score_deltas": {
                        x["client_id"]: (
                            after_scores[x["client_id"]] - before_scores[x["client_id"]]
                            if x["client_id"] in after_scores
                            else None
                        )
                        for x in targets
                    },
                    "feed": await evaluate(api, probe, "feed_after_" + topic),
                }
            )
            # New explicit query must survive earlier personal preferences.
            await evaluate(
                api,
                next(c for c in cases if c["id"] == "search_coffee_supplier"),
                "coffee_after_" + topic,
            )
            if topic == "oven" and after["items"]:
                hidden = next(
                    (
                        x["client_id"]
                        for x in after["items"]
                        if "oven" in labels[x["client_id"]]["offers"]
                    ),
                    None,
                )
                if hidden:
                    event = {
                        "event_id": str(uuid4()),
                        "request_id": after["request_id"],
                        "client_id": "bakery",
                        "target_client_id": hidden,
                        "kind": "hide",
                    }
                    result = await api.post("/v1/events", json={"events": [event]})
                    assert result.json()["results"][0]["status"] == "accepted"
                    events.append(event)
        control_after = await evaluate(api, control, "control_after")
        checks["personal_isolation"] = control_scores == fixed_scores(
            control_rows, "bakery-control"
        )
        # The shared segment graph can legitimately change the control feed; report it
        # separately instead of confusing global learning with personal-state leakage.
        learning.append(
            {
                "control_feed_same_order": [x["client_id"] for x in control_before["items"]]
                == [x["client_id"] for x in control_after["items"]]
            }
        )
        saved = await evaluate(api, probe, "before_restart")
        checks["training_persisted"] = saved["diagnostics"]["ranker_observations"] > 0
        # Concurrent real TCP requests; service currently serializes compute in one worker.
        concurrent = await asyncio.gather(
            *[
                api.post(
                    "/v1/feed",
                    json={"client_id": "bakery", "intent": "supplier", "limit": 10, "seed": i},
                )
                for i in range(4)
            ]
        )
        checks["concurrent_requests_succeed"] = all(r.status_code == 200 for r in concurrent)
    async with service(output, "on", args) as api:
        restored = await evaluate(api, probe, "after_restart")
        checks["restart_same_order"] = [x["client_id"] for x in saved["items"]] == [
            x["client_id"] for x in restored["items"]
        ]
        checks["hide_survives_restart"] = hidden is not None and hidden not in {
            x["client_id"] for x in restored["items"]
        }
    async with httpx.AsyncClient(timeout=180) as model_api:
        health_after = (await model_api.get(args.models + "/health")).json()
    checks["zero_gpu_oom"] = health_after["ooms"] == health_before["ooms"]
    checks["contacts_update_personal_model"] = all(
        phase["trained_targets"] > 0
        and all(delta > 0 for delta in phase["fixed_feature_deltas"].values())
        for phase in learning
        if "trained_targets" in phase
    )
    audit_repo.close()
    checks["real_inference_used"] = (
        health_after["embedding_calls"] > health_before["embedding_calls"]
    )
    checks["bge_used"] = health_after["rerank_calls"] > health_before["rerank_calls"]
    checks["no_silent_reranker_fallback"] = all(
        r["reranked"] for r in rows if r["mode"] in ("off", "on") and r["returned"]
    )
    summary = {}
    for mode in ("off", "on_no_rerank", "on_semantic", "on"):
        summary[mode] = {}
        for kind in ("feed", "search", "buyer", "adversarial"):
            group = [r for r in rows if r["mode"] == mode and r["kind"] == kind]
            total = sum(r["returned"] for r in group)
            summary[mode][kind] = {
                "requests": len(group),
                "returned": total,
                "correct": sum(r["correct"] for r in group),
                "precision": sum(r["correct"] for r in group) / total if total else None,
                "p95_ms": float(np.percentile([r["latency_ms"] for r in group], 95)),
            }
    report = {
        "output": str(output),
        "companies": len(catalog),
        "evaluated_requests": len(rows),
        "summary": summary,
        "checks": checks,
        "models_before": health_before,
        "models_after": health_after,
        "limitations": [
            "SQLite not PostgreSQL",
            "one worker",
            "scripted chat/extraction",
            "synthetic event labels",
            "local TCP, no WAN or production concurrency",
        ],
    }
    for name, value in [
        ("report", report),
        ("rows", rows),
        ("responses", responses),
        ("learning", learning),
        ("events", events),
    ]:
        save(name, value)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Audit checks failed; inspect report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="http://127.0.0.1:8011")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--fixtures", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
