"""Private, versioned source snapshots; no synthetic products or fabricated click labels."""

from datetime import datetime

from business_match.providers import fingerprint
from business_match.schemas import BusinessContext, Document, SignalExtraction, utcnow
from business_match.storage import business_contexts


class ContextConflict(ValueError):
    pass


class BusinessSignals:
    def __init__(self, repository, providers, settings):
        self.repository, self.providers, self.settings = repository, providers, settings

    async def stage(self, client_id, context):
        sources = {p["source_id"]: p for p in self.repository.contexts(client_id)}
        changed = None
        if context is not None:
            old = sources.get(context.source_id)
            content = context.model_dump(mode="json")
            content["records"].sort(key=lambda record: record["id"])
            digest = fingerprint(content)
            if old and context.source_sequence <= old["source_sequence"]:
                if context.source_sequence < old["source_sequence"] or digest != old["digest"]:
                    raise ContextConflict("Stale or conflicting business_context source_sequence")
            else:
                if not old and len(sources) >= self.settings.business_signal_sources_limit:
                    raise ContextConflict("Too many business_context sources")
                changed = {
                    **content,
                    "digest": digest,
                    "extractions": (old or {}).get("extractions", {}),
                }
                sources[context.source_id] = changed
        # Extraction caches are scoped to current records, not an unbounded raw text cache.
        updates = {}
        for source_id, payload in sorted(sources.items()):
            parsed = BusinessContext.model_validate(
                {k: payload[k] for k in ("source_id", "source_sequence", "records")}
            )
            previous = payload.get("extractions", {})
            extracted = {}
            for record in parsed.records:
                content = {
                    "subject": record.subject,
                    "role": record.role,
                    "purpose": record.purpose,
                    "kind": record.kind,
                }
                key = fingerprint(
                    [
                        content,
                        self.settings.llm_model,
                        self.providers.llm_base_url,
                        self.settings.model_api_mode,
                        "business-extractor-v1",
                    ]
                )
                if key in extracted:
                    continue
                if key in previous:
                    extracted[key] = previous[key]
                elif (
                    self.settings.business_signals_enabled
                    and record.purpose != "noncommercial"
                    and record.status in ("active", "completed", "settled")
                ):
                    value = await self.providers.extract_signal(content)
                    extracted[key] = SignalExtraction.model_validate(value).model_dump()
            updated = {**payload, "extractions": extracted}
            sources[source_id] = updated
            if updated != payload or (changed is not None and source_id == changed["source_id"]):
                updates[source_id] = updated
        revision = fingerprint(
            [
                sources,
                self.settings.business_signals_enabled,
                self.settings.business_signal_min_confidence,
                self.settings.business_signal_completed_weight,
            ]
        )
        return sources, updates, revision

    async def documents(self, sources):
        if not self.settings.business_signals_enabled:
            return []
        result = []
        for source_id, payload in sorted(sources.items()):
            groups = {}
            for record in payload["records"]:
                group = (
                    ("trade", record["transaction_group_id"])
                    if record["transaction_group_id"]
                    else ("record", record["id"])
                )
                groups.setdefault(group, []).append(record)
            for group, records in groups.items():
                # Conservatively withdraw the entire linked trade on cancellation/refund.
                if any(r["status"] in ("cancelled", "refunded") for r in records):
                    continue
                topics = {}
                for record in records:
                    if record["status"] not in ("active", "completed", "settled"):
                        continue
                    if record["purpose"] == "noncommercial":
                        continue
                    content = {k: record[k] for k in ("subject", "role", "purpose", "kind")}
                    key = fingerprint(
                        [
                            content,
                            self.settings.llm_model,
                            self.providers.llm_base_url,
                            self.settings.model_api_mode,
                            "business-extractor-v1",
                        ]
                    )
                    signal = SignalExtraction.model_validate(payload["extractions"].get(key, {}))
                    # An extractor cannot contradict a known source role.
                    if (
                        not signal.commercial
                        or not signal.subject.strip()
                        or signal.confidence < self.settings.business_signal_min_confidence
                        or signal.role == "unknown"
                        or record["role"] not in ("unknown", signal.role)
                    ):
                        continue
                    support = signal.confidence
                    if record["status"] == "completed":
                        support *= self.settings.business_signal_completed_weight
                    topic = (signal.role, signal.subject.strip().casefold())
                    observed = datetime.fromisoformat(record["occurred_at"])
                    expires = (
                        datetime.fromisoformat(record["valid_until"])
                        if record["valid_until"]
                        else None
                    )
                    if expires and expires <= utcnow():
                        continue
                    # Max, not sum: split payments and duplicate records cannot inflate support.
                    option = (support, observed, signal.subject.strip(), expires)
                    if topic not in topics or option[:2] > topics[topic][:2]:
                        topics[topic] = option
                for (role, _), (support, observed, text, expires) in topics.items():
                    vector = (await self.providers.embed([text], query=role == "buyer"))[0]
                    result.append(
                        Document(
                            key="evidence:" + fingerprint([source_id, group, role, text]),
                            text=text,
                            kind="need" if role == "buyer" else "offer",
                            intent="supplier" if role == "buyer" else None,
                            vector=vector,
                            evidence=True,
                            support=support,
                            observed_at=observed,
                            expires_at=expires,
                        )
                    )
        return result

    def commit(self, client_id, updates, conn):
        for source_id, payload in updates.items():
            self.repository.put(
                business_contexts,
                fingerprint([client_id, source_id]),
                payload,
                conn=conn,
                client_id=client_id,
            )


def support(document, settings, now):
    if not document.evidence or not settings.business_signals_enabled:
        return 0.0
    if document.expires_at and document.expires_at <= now:
        return 0.0
    if document.observed_at is None:
        return 0.0
    age = max(0, (now - document.observed_at).total_seconds() / 86400)
    if age > settings.business_signal_max_age_days:
        return 0.0
    return document.support * 0.5 ** (age / settings.business_signal_half_life_days)
