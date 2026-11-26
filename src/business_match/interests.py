"""Directional, source-anchored interests for candidate retrieval; no LLM calls."""

from datetime import datetime

import numpy as np

from business_match.providers import fingerprint
from business_match.schemas import Document, Intent, utcnow


def embedding_signature(settings):
    endpoint = (
        settings.mock_api_base_url
        if settings.model_api_mode == "mock"
        else settings.embedding_base_url
    )
    return fingerprint([settings.model_api_mode, endpoint, settings.embedding_model])


def interests(repository, settings, client_id, hidden):
    now = utcnow()
    signature = embedding_signature(settings)
    grouped = {}
    records = repository.user_events(client_id, settings.feedback_history_days)
    hides = {}
    for record in records:
        event = record["event"]
        if event["kind"] == "hide":
            key = (event["target_client_id"], record["intent"])
            timestamp = datetime.fromisoformat(event["occurred_at"])
            hides[key] = max(hides.get(key, timestamp), timestamp)
    for record in records:
        event = record["event"]
        anchor = record.get("interest_anchor")
        if not anchor or anchor.get("signature") != signature:
            continue
        if event["target_client_id"] in hidden:
            continue
        key = (event["target_client_id"], record["intent"], anchor["text"])
        grouped.setdefault(key, []).append(record)
    candidates = []
    targets = {}
    for (target_id, intent, text), history in grouped.items():
        last_hide = hides.get((target_id, intent))
        positive = [
            r
            for r in history
            if r["event"]["kind"] in ("like", "contact")
            and (last_hide is None or datetime.fromisoformat(r["event"]["occurred_at"]) > last_hide)
        ]
        if not positive:
            continue
        if target_id not in targets:
            targets[target_id] = repository.client(target_id)
        target = targets[target_id]
        if target is None or not target.source.active:
            continue
        choices = []
        for record in positive:
            anchor = record["interest_anchor"]
            vector = np.asarray(anchor["vector"], dtype=float)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                continue
            # Catalog changes, removals and stock withdrawal invalidate the exact anchor.
            if not any(
                not d.evidence
                and d.text == text
                and d.vector == anchor["vector"]
                and d.product_id == anchor.get("product_id")
                for d in target.documents
            ):
                continue
            age = max(
                0, (now - datetime.fromisoformat(record["event"]["occurred_at"])).total_seconds()
            )
            strength = (1 if record["event"]["kind"] == "contact" else 0.8) * (
                0.5 ** (age / (settings.interest_half_life_days * 86400))
            )
            if strength >= settings.interest_min_support:
                choices.append((strength, anchor))
        if choices:
            strength, anchor = max(choices, key=lambda x: x[0])
            candidates.append((strength, target_id, Intent(intent), anchor))
    selected = {}
    count = 0
    for strength, target_id, intent, anchor in sorted(
        candidates, key=lambda x: (-x[0], x[1], x[3]["text"])
    ):
        bucket = selected.setdefault(intent, [])
        if count >= settings.interest_max_vectors:
            break
        vector = np.asarray(anchor["vector"], dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            continue
        vector /= norm
        # Keep topics separate, merge near-duplicate anchors without averaging interests.
        if any(
            len(d.vector) == len(vector)
            and np.dot(d.vector, vector) >= settings.interest_merge_similarity
            for d in bucket
        ):
            continue
        bucket.append(
            Document(
                key="interest:" + fingerprint([target_id, anchor["text"]]),
                kind="offer" if intent in (Intent.buyer, Intent.peer) else "need",
                intent=intent,
                text=anchor["text"],
                vector=vector.tolist(),
                weight=strength,
            )
        )
        count += 1
    return selected
