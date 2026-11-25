from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


Id = Annotated[str, Field(min_length=1, max_length=128)]
Text = Annotated[str, Field(min_length=1, max_length=4000)]


class Intent(StrEnum):
    supplier = "supplier"
    buyer = "buyer"
    partner = "partner"
    peer = "peer"


class Product(BaseModel):
    id: Id
    name: Annotated[str, Field(min_length=1, max_length=300)]
    description: str = Field(default="", max_length=4000)
    category: str | None = Field(default=None, max_length=200)
    stock: float | None = Field(default=None, ge=0)
    price: float | None = Field(default=None, ge=0)
    unit: str | None = Field(default=None, max_length=100)
    active: bool = True


class ClientInput(BaseModel):
    """Full current snapshot; omitted products are deleted from this client's catalog."""

    id: Id
    name: Annotated[str, Field(min_length=1, max_length=300)]
    description: str = Field(default="", max_length=12000)
    segment: str = Field(default="unknown", max_length=200)
    region: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=200)
    explicit_needs: list[Text] = Field(default_factory=list, max_length=30)
    products: list[Product] = Field(default_factory=list, max_length=5000)
    active: bool = True

    @model_validator(mode="after")
    def unique_products(self):
        if len({p.id for p in self.products}) != len(self.products):
            raise ValueError("Product IDs must be unique within a client")
        return self


class Need(BaseModel):
    text: Text
    intent: Intent
    importance: int = Field(default=2, ge=1, le=3)
    evidence: Literal["explicit", "inferred"] = "inferred"


class BusinessProfile(BaseModel):
    summary: Text
    offers: list[Text] = Field(min_length=1, max_length=15)
    needs: list[Need] = Field(default_factory=list, max_length=20)


class Document(BaseModel):
    key: str
    text: str
    kind: Literal["summary", "offer", "need", "product"]
    intent: Intent | None = None
    product_id: str | None = None
    weight: float = 1
    vector: list[float]


class IndexedClient(BaseModel):
    source: ClientInput
    profile: BusinessProfile
    documents: list[Document]
    revision: str
    profile_revision: str
    created_at: datetime = Field(default_factory=utcnow)
    content_updated_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class RecommendRequest(BaseModel):
    client_id: Id
    limit: int = Field(default=20, ge=1, le=50)
    query: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    region: str | None = None
    city: str | None = None
    intent: Intent | None = None
    exclude_client_ids: list[Id] = Field(default_factory=list, max_length=1000)
    seed: int | None = None


class ProductMatch(BaseModel):
    product: Product
    score: float


class Recommendation(BaseModel):
    client_id: str
    name: str
    segment: str
    region: str | None
    city: str | None
    intent: Intent
    score: float
    reason: str
    matched_need: str
    matched_text: str
    products: list[ProductMatch]
    exploration: bool = False


class RecommendationResponse(BaseModel):
    request_id: str
    mode: Literal["feed", "search"]
    items: list[Recommendation]
    policy_version: str = "anchored-multivector-v1"
    diagnostics: dict[str, float | int | bool | str] = Field(default_factory=dict)


class EventKind(StrEnum):
    impression = "impression"
    dwell = "dwell"
    open = "open"
    like = "like"
    unlike = "unlike"
    contact = "contact"
    hide = "hide"


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: Id
    request_id: Id
    client_id: Id
    target_client_id: Id
    kind: EventKind
    occurred_at: datetime = Field(default_factory=utcnow)
    visible_fraction: float = Field(default=0, ge=0, le=1)
    dwell_ms: int = Field(default=0, ge=0, le=3600000)
    product_id: str | None = None

    @model_validator(mode="after")
    def timezone_required(self):
        if self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must have a timezone")
        if self.kind == EventKind.impression and self.visible_fraction < 0.5:
            raise ValueError("Impression requires visible_fraction >= 0.5")
        return self


class FeedbackBatch(BaseModel):
    events: list[FeedbackEvent] = Field(min_length=1, max_length=100)
