from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", allow_inf_nan=False)

    database_url: str = "sqlite:///./business_match.db"
    model_api_mode: Literal["mock", "external"] = "mock"
    mock_api_base_url: str = "http://127.0.0.1:8010/v1"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "local"
    llm_model: str = "Qwen3.8-38B"
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_api_key: str = "local"
    embedding_model: str = "Qwen3-Embedding-4B"
    rerank_url: str = ""
    rerank_api_key: str = "local"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_feed_enabled: bool = True
    service_api_key: str = ""
    ranker_mode: Literal["semantic", "linucb"] = "linucb"
    linucb_regularization: float = Field(default=1.0, gt=0)
    linucb_alpha: float = Field(default=0.15, ge=0)
    ranker_personal_weight: float = Field(default=0.5, ge=0)
    personal_metadata_scale: float = Field(default=0.1, ge=0, le=1)
    ranker_rerank_weight: float = Field(default=0.35, ge=0, le=1)
    ranker_graph_weight: float = Field(default=0.1, ge=0)
    ranker_freshness_weight: float = Field(default=0.03, ge=0)
    ranker_region_weight: float = Field(default=0.02, ge=0)
    diversity_weight: float = Field(default=0.12, ge=0)
    max_segment_share: float = Field(default=0.6, gt=0, le=1)
    diversity_score_tolerance: float = Field(default=0.2, ge=0)
    retrieval_score_tolerance: float = Field(default=0.1, ge=0, le=1)
    max_role_share: float = Field(default=0.6, gt=0, le=1)
    max_peer_share: float = Field(default=0.2, gt=0, le=1)
    exploration_rate: float = Field(default=0.1, ge=0, le=1)
    exploration_freshness_weight: float = Field(default=1.0, ge=0)
    http_timeout: float = Field(default=45, gt=0)
    embedding_rpm: int = Field(default=28, ge=1)
    embedding_tpm: int = Field(default=60000, ge=1000)
    embedding_batch_tokens: int = Field(default=12000, ge=100, le=30000)
    llm_rpm: int = Field(default=9, ge=1)
    rerank_rpm: int = Field(default=28, ge=1)
    rerank_tpm: int = Field(default=60000, ge=1000)
    rerank_top_k: int = Field(default=12, ge=1, le=30)
    candidate_limit: int = Field(default=150, ge=10)
    retrieval_client_batch_size: int = Field(default=128, ge=1, le=10000)
    retrieval_document_batch_size: int = Field(default=1024, ge=1, le=50000)
    retrieval_query_batch_size: int = Field(default=64, ge=1, le=1000)
    query_importance_weight: float = Field(default=0.15, ge=0, le=1)
    card_product_limit: int = Field(default=3, ge=0, le=20)
    card_product_score_tolerance: float = Field(default=0.1, ge=0, le=1)
    business_signals_enabled: bool = True
    business_signal_boost: float = Field(default=0.12, ge=0, le=0.5)
    business_signal_half_life_days: float = Field(default=60, gt=0)
    business_signal_max_age_days: float = Field(default=365, gt=0)
    business_signal_min_confidence: float = Field(default=0.6, ge=0, le=1)
    business_signal_completed_weight: float = Field(default=0.5, ge=0, le=1)
    business_signal_sources_limit: int = Field(default=8, ge=1, le=100)
    content_dimensions: int = Field(default=32, ge=4, le=128)
    content_projection_seed: int = Field(default=42, ge=0)
    catalog_size_scale: int = Field(default=20, ge=1)
    ranker_max_observations: int = Field(default=1000, ge=1, le=100000)
    reward_open: float = Field(default=0.55, ge=0, le=1)
    reward_like: float = Field(default=0.8, ge=0, le=1)
    reward_contact: float = Field(default=1.0, ge=0, le=1)
    reward_hide: float = Field(default=0.0, ge=0, le=1)
    reward_dwell_max: float = Field(default=0.35, ge=0, le=1)
    dwell_min_ms: int = Field(default=2000, ge=0)
    dwell_saturation_ms: int = Field(default=15000, gt=0)
    baseline_ceiling: float = Field(default=0.8, gt=0, le=1)
    baseline_slope: float = Field(default=4.0, gt=0)
    baseline_midpoint: float = Field(default=0.5, ge=0, le=1)
    feedback_history_days: int = Field(default=60, ge=1)
    hide_duration_days: float = Field(default=7, gt=0)
    impression_decay_days: float = Field(default=1, gt=0)
    event_future_tolerance_seconds: int = Field(default=300, ge=0)
    segment_repeat_weight: float = Field(default=0.25, ge=0)
    graph_prior_rate: float = Field(default=0.12, ge=0, le=1)
    graph_prior_strength: float = Field(default=20, gt=0)
    graph_half_life_days: float = Field(default=60, gt=0)
    graph_negative_weight: float = Field(default=2, ge=0)
    graph_score_scale: float = Field(default=0.12, ge=0)
    graph_reward_dwell: float = Field(default=0.12, ge=0, le=1)
    graph_reward_open: float = Field(default=0.35, ge=0, le=1)
    graph_reward_like: float = Field(default=0.65, ge=0, le=1)
    graph_reward_contact: float = Field(default=1, ge=0, le=1)
    profile_product_limit: int = Field(default=40, ge=1, le=100)
    history_half_life_days: float = Field(default=14, gt=0)
    freshness_half_life_days: float = Field(default=14, gt=0)

    @model_validator(mode="after")
    def consistent_windows(self):
        if self.hide_duration_days > self.feedback_history_days:
            raise ValueError("feedback_history_days must cover hide_duration_days")
        if self.dwell_min_ms > self.dwell_saturation_ms:
            raise ValueError("dwell_min_ms must not exceed dwell_saturation_ms")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
