"""Portable SQL persistence. JSON vector storage + exact retrieval for the bounded MVP."""

from contextlib import contextmanager
from datetime import timedelta
from typing import Any

from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.pool import StaticPool

from business_match.schemas import IndexedClient, utcnow

metadata = MetaData()
clients = Table(
    "bm_clients", metadata,
    Column("id", String(128), primary_key=True), Column("payload", JSON, nullable=False),
)
cache = Table(
    "bm_embedding_cache", metadata,
    Column("id", String(64), primary_key=True), Column("payload", JSON, nullable=False),
)
slates = Table(
    "bm_slates", metadata,
    Column("id", String(128), primary_key=True),
    Column("client_id", String(128), index=True, nullable=False),
    Column("payload", JSON, nullable=False),
)
events = Table(
    "bm_events", metadata,
    Column("id", String(128), primary_key=True),
    Column("client_id", String(128), index=True, nullable=False),
    Column("payload", JSON, nullable=False),
)
relation_edges = Table(
    "bm_relation_edges",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("payload", JSON, nullable=False),
)
relation_observations = Table(
    "bm_relation_observations",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("payload", JSON, nullable=False),
)
ranker_states = Table(
    "bm_ranker_states",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("payload", JSON, nullable=False),
)


class Repository:
    def __init__(self, url: str):
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)

    def initialize(self):
        metadata.create_all(self.engine)

    def close(self):
        self.engine.dispose()

    @contextmanager
    def transaction(self):
        with self.engine.begin() as conn:
            yield conn

    def put(self, table, key: str, payload: dict, conn=None, **columns):
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        stmt = insert(table).values(id=key, payload=payload, **columns)
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.id], set_={"payload": payload, **columns}
        )
        if conn is not None:
            conn.execute(stmt)
        else:
            with self.transaction() as connection:
                connection.execute(stmt)

    def get(self, table, key: str, conn=None):
        stmt = select(table.c.payload).where(table.c.id == key)
        if conn is not None:
            return conn.execute(stmt).scalar_one_or_none()
        with self.engine.connect() as connection:
            return connection.execute(stmt).scalar_one_or_none()

    def all_clients(self) -> list[IndexedClient]:
        with self.engine.connect() as conn:
            return [
                IndexedClient.model_validate(p)
                for p in conn.execute(select(clients.c.payload)).scalars()
            ]

    def client(self, key: str) -> IndexedClient | None:
        payload = self.get(clients, key)
        return IndexedClient.model_validate(payload) if payload else None

    def user_events(self, client_id: str, days: int = 60) -> list[dict]:
        cutoff = (utcnow() - timedelta(days=days)).isoformat()
        with self.engine.connect() as conn:
            values = conn.execute(
                select(events.c.payload).where(events.c.client_id == client_id)
            ).scalars()
            return [p for p in values if p["received_at"] >= cutoff]

    def insert_event(self, event_id: str, client_id: str, payload: dict, conn) -> bool:
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        result = conn.execute(
            insert(events).values(id=event_id, client_id=client_id, payload=payload)
            .on_conflict_do_nothing(index_elements=[events.c.id])
        )
        return result.rowcount == 1

    def prune_slates(self, before: str) -> int:
        # Maintenance hook; exposed through CLI, not invoked on the online path.
        with self.transaction() as conn:
            rows = conn.execute(select(slates.c.id, slates.c.payload)).all()
            ids = [key for key, value in rows if value["created_at"] < before]
            if ids:
                conn.execute(delete(slates).where(slates.c.id.in_(ids)))
            return len(ids)
