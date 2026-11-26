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


class BusinessRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Id
    kind: Literal["contract", "transfer"]
    role: Literal["buyer", "seller", "unknown"] = "unknown"
    subject: Text
    status: Literal["active", "completed", "cancelled", "pending", "settled", "refunded"]
    occurred_at: datetime
    valid_until: datetime | None = None
    transaction_group_id: Id | None = None
    purpose: Literal["trade", "noncommercial", "unknown"] = "unknown"

    @model_validator(mode="after")
    def valid_record(self):
        statuses = ({"active", "completed", "cancelled"} if self.kind == "contract"
                    else {"pending", "settled", "refunded"})
        if self.status not in statuses:
            raise ValueError("Status does not match record kind")
        for date in (self.occurred_at, self.valid_until):
            if date is not None and date.utcoffset() is None:
                raise ValueError("Business record dates require a timezone")
        if self.occurred_at > utcnow():
            raise ValueError("Business record is in the future")
        if self.valid_until is not None and self.valid_until < self.occurred_at:
            raise ValueError("valid_until precedes occurred_at")
        return self


class BusinessContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: Id
    source_sequence: int = Field(ge=0)
    records: list[BusinessRecord] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def unique_records(self):
        if len({r.id for r in self.records}) != len(self.records):
            raise ValueError("Business record IDs must be unique per source")
        return self


class SignalExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    subject: Annotated[str, Field(max_length=300)] = ""
    role: Literal["buyer", "seller", "unknown"] = "unknown"
    commercial: bool = False
    confidence: float = Field(default=0, ge=0, le=1)


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
    # Write-only: never serialize private source records in GET /clients or the profile prompt.
    business_context: BusinessContext | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def unique_products(self):
        if "business_context" in self.model_fields_set and self.business_context is None:
            raise ValueError("Omit business_context to preserve it; use records=[] to clear")
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
    evidence: bool = False
    support: float = 0
    observed_at: datetime | None = None
    expires_at: datetime | None = None


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
    insights: list[str] = Field(default_factory=list)


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
