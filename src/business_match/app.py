import asyncio
import logging
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader

from business_match.business_signals import ContextConflict
from business_match.config import Settings, get_settings
from business_match.engine import Engine
from business_match.indexing import Indexer
from business_match.providers import Providers
from business_match.schemas import (
    ClientInput,
    FeedbackBatch,
    Intent,
    RecommendationResponse,
    RecommendRequest,
)
from business_match.storage import Repository

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, providers=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository = Repository(settings.database_url)
        repository.initialize()
        remote = providers if providers is not None else Providers(settings, repository)
        app.state.repository = repository
        app.state.indexer = Indexer(repository, remote, settings)
        app.state.engine = Engine(repository, remote, settings)
        # Sync SQL and exact retrieval run in this single process; no multiprocess budgets.
        app.state.recommend_lock = asyncio.Lock()
        yield
        if providers is None:
            await remote.close()
        repository.close()

    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize(key: str | None = Depends(key_header)):
        if settings.service_api_key and (
            key is None or not secrets.compare_digest(key, settings.service_api_key)
        ):
            raise HTTPException(401, "Invalid API key")

    app = FastAPI(
        title="Business Match", version="0.1.0", lifespan=lifespan,
        description="Adaptive B2B recommendations. Integration: docs/integration.md",
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.put("/v1/clients/{client_id}", dependencies=[Depends(authorize)])
    async def upsert(client_id: str, source: ClientInput, request: Request):
        if client_id != source.id:
            raise HTTPException(422, "Path and body client IDs must match")
        try:
            async with request.app.state.recommend_lock:
                indexed, changed = await request.app.state.indexer.upsert(source)
        except ContextConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("Indexing failed")
            raise HTTPException(502, "Profile/embedding provider failed; retry snapshot") from exc
        return {"client_id": indexed.source.id, "changed": changed,
                "revision": indexed.revision, "profile": indexed.profile}

    @app.get("/v1/clients/{client_id}", dependencies=[Depends(authorize)])
    def profile(client_id: str, request: Request):
        indexed = request.app.state.repository.client(client_id)
        if indexed is None:
            raise HTTPException(404, "Client not found")
        return {"source": indexed.source, "profile": indexed.profile,
                "revision": indexed.revision, "updated_at": indexed.updated_at}

    @app.post("/v1/feed", response_model=RecommendationResponse,
              dependencies=[Depends(authorize)])
    async def feed(body: RecommendRequest, request: Request):
        if body.query:
            raise HTTPException(422, "Use /v1/search for explicit queries")
        return await recommend(body, request)

    @app.post("/v1/search", response_model=RecommendationResponse,
              dependencies=[Depends(authorize)])
    async def search(body: RecommendRequest, request: Request):
        if not body.query or not body.query.strip():
            raise HTTPException(422, "Search requires a nonempty query")
        # A bare product query means finding a supplier; other roles remain explicit.
        if body.intent is None:
            body = body.model_copy(update={"intent": Intent.supplier})
        return await recommend(body, request)

    async def recommend(body, request):
        try:
            async with request.app.state.recommend_lock:
                return await request.app.state.engine.recommend(body)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Embedding provider unavailable") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/events", dependencies=[Depends(authorize)])
    async def events(body: FeedbackBatch, request: Request):
        # Per-event outcomes allow safe retry of mixed batches using the same event IDs.
        results = []
        async with request.app.state.recommend_lock:
            for event in body.events:
                try:
                    inserted = request.app.state.engine.ingest(event)
                    results.append({"event_id": event.event_id,
                                    "status": "accepted" if inserted else "duplicate"})
                except ValueError as exc:
                    results.append({"event_id": event.event_id, "status": "rejected",
                                    "detail": str(exc)})
        return {"results": results}

    return app


app = create_app()
