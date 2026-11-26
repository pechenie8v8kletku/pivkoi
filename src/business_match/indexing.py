import asyncio
from collections import deque

from business_match.business_signals import BusinessSignals
from business_match.config import Settings
from business_match.providers import Providers, fingerprint
from business_match.schemas import ClientInput, Document, IndexedClient, utcnow
from business_match.storage import Repository, clients

PROFILE_VERSION = "profile-v3-bounded-input"


def catalog_content(source: ClientInput):
    semantic = source.model_dump(exclude={"region", "city", "active", "products"})
    semantic["products"] = [
        p.model_dump(exclude={"stock", "price", "unit"})
        for p in sorted(source.products, key=lambda p: p.id)
    ]
    return semantic


def profile_input(source: ClientInput, limit: int) -> ClientInput:
    """Stable category round-robin, exact-text deduplication; no sales assumptions."""
    grouped, seen = {}, set()
    for product in sorted(source.products, key=lambda p: (p.category or "", p.id)):
        if not product.active:
            continue
        # Operational fields never influence the generated semantic company profile.
        product = product.model_copy(update={
            "description": product.description[:500],
            "price": None, "stock": None, "unit": None,
        })
        key = (product.name, product.description, product.category)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(product.category or "", deque()).append(product)
    sample = []
    while grouped and len(sample) < limit:
        for category in list(grouped):
            sample.append(grouped[category].popleft())
            if not grouped[category]:
                del grouped[category]
            if len(sample) >= limit:
                break
    return source.model_copy(update={
        "products": sample, "region": None, "city": None, "active": True,
    })


class Indexer:
    def __init__(self, repository: Repository, providers: Providers, settings: Settings):
        self.repository, self.providers, self.settings = repository, providers, settings
        self.lock = asyncio.Lock()
        self.signals = BusinessSignals(repository, providers, settings)

    async def upsert(self, source: ClientInput) -> tuple[IndexedClient, bool]:
        # Single-worker MVP: serialize indexing so concurrent snapshots cannot overwrite each other.
        async with self.lock:
            sources, context_updates, context_revision = await self.signals.stage(
                source.id, source.business_context)
            # Remove the write-only context before persisting or prompting the public profile.
            source = ClientInput.model_validate(source.model_dump())
            revision = fingerprint([
                source.model_dump(), context_revision, self.settings.embedding_model,
                self.settings.llm_model, self.settings.model_api_mode,
                self.providers.embedding_base_url, self.providers.llm_base_url,
                PROFILE_VERSION, self.settings.profile_product_limit,
            ])
            previous = self.repository.client(source.id)
            if previous and previous.revision == revision:
                return previous, False
            # Cache the actual bounded LLM input, not all unsampled catalog text.
            prompt_source = profile_input(source, self.settings.profile_product_limit)
            profile_revision = fingerprint([
                prompt_source.model_dump(), self.settings.llm_model, self.settings.model_api_mode,
                self.providers.llm_base_url, PROFILE_VERSION,
            ])
            if previous and previous.profile_revision == profile_revision:
                profile = previous.profile
            else:
                profile = await self.providers.profile(prompt_source)
            specs = [("summary", profile.summary, None, None, 1.0)]
            specs += [("offer", text, None, None, 1.0) for text in profile.offers]
            specs += [
                ("need", need.text, need.intent, None,
                 need.importance / 3 * (1 if need.evidence == "explicit" else 0.7))
                for need in profile.needs
            ]
            specs += [
                ("product", f"{p.name}. {p.description}. {p.category or ''}", None, p.id, 1.0)
                for p in source.products if p.active and p.stock != 0
            ]
            plain = [s for s in specs if s[0] != "need"]
            needs = [s for s in specs if s[0] == "need"]
            plain_vectors = await self.providers.embed([s[1] for s in plain])
            need_vectors = await self.providers.embed([s[1] for s in needs], query=True)
            documents = [
                Document(key=f"{kind}:{i}", kind=kind, text=text, intent=intent,
                         product_id=product_id, weight=weight, vector=vector)
                for i, ((kind, text, intent, product_id, weight), vector) in enumerate(
                    zip(plain + needs, plain_vectors + need_vectors, strict=True)
                )
            ]
            documents.extend(await self.signals.documents(sources))
            now = utcnow()
            content_updated_at = (
                (previous.content_updated_at or previous.created_at)
                if previous and catalog_content(previous.source) == catalog_content(source)
                and previous.profile_revision == profile_revision else now
            )
            indexed = IndexedClient(
                source=source, profile=profile, documents=documents, revision=revision,
                profile_revision=profile_revision,
                created_at=previous.created_at if previous else now,
                content_updated_at=content_updated_at,
                updated_at=now,
            )
            with self.repository.transaction() as conn:
                self.signals.commit(source.id, context_updates, conn)
                self.repository.put(clients, source.id, indexed.model_dump(mode="json"), conn=conn)
            return indexed, True
