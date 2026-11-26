import asyncio

from business_match.config import Settings
from business_match.providers import Providers, fingerprint
from business_match.schemas import ClientInput, Document, IndexedClient, utcnow
from business_match.storage import Repository, clients


class Indexer:
    def __init__(self, repository: Repository, providers: Providers, settings: Settings):
        self.repository, self.providers, self.settings = repository, providers, settings
        self.lock = asyncio.Lock()

    async def upsert(self, source: ClientInput) -> tuple[IndexedClient, bool]:
        # Single-worker MVP: serialize indexing so concurrent snapshots cannot overwrite each other.
        async with self.lock:
            revision = fingerprint([
                source.model_dump(), self.settings.embedding_model,
                self.settings.llm_model, self.settings.model_api_mode,
                self.providers.embedding_base_url, self.providers.llm_base_url, "profile-v1",
            ])
            previous = self.repository.client(source.id)
            if previous and previous.revision == revision:
                return previous, False
            # Stock, price, region changes should not spend another LLM call.
            semantic = source.model_dump(exclude={"region", "city", "active", "products"})
            semantic["products"] = [
                p.model_dump(exclude={"stock", "price", "active", "unit"})
                for p in sorted(source.products, key=lambda p: p.id)
            ]
            profile_revision = fingerprint([
                semantic, self.settings.llm_model, self.settings.model_api_mode,
                self.providers.llm_base_url, "profile-v1",
            ])
            if previous and previous.profile_revision == profile_revision:
                profile = previous.profile
            else:
                # Deterministic, category-diverse sample bounds the LLM prompt.
                ordered = sorted(source.products, key=lambda p: (p.category or "", p.id))
                grouped: dict[str, list] = {}
                for product in ordered:
                    grouped.setdefault(product.category or "", []).append(product)
                sample = []
                while grouped and len(sample) < self.settings.profile_product_limit:
                    for category in list(grouped):
                        sample.append(grouped[category].pop(0))
                        if not grouped[category]:
                            del grouped[category]
                        if len(sample) >= self.settings.profile_product_limit:
                            break
                prompt_source = source.model_copy(update={"products": sample})
                # Keep a conservative character budget for the shared API.
                prompt_source = prompt_source.model_copy(update={"products": [
                    p.model_copy(update={"description": p.description[:500]}) for p in sample
                ]})
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
            now = utcnow()
            content_updated_at = (
                (previous.content_updated_at or previous.created_at)
                if previous and previous.profile_revision == profile_revision else now
            )
            indexed = IndexedClient(
                source=source, profile=profile, documents=documents, revision=revision,
                profile_revision=profile_revision,
                created_at=previous.created_at if previous else now,
                content_updated_at=content_updated_at,
                updated_at=now,
            )
            self.repository.put(clients, source.id, indexed.model_dump(mode="json"))
            return indexed, True
