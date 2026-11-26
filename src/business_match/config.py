from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./business_match.db"
    model_api_mode: Literal["mock", "external"] = "mock"
    mock_api_base_url: str = "http://127.0.0.1:8010/v1"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "local"
    llm_model: str = "Qwen3.8-27B"
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
    profile_product_limit: int = Field(default=40, ge=1, le=100)
    history_half_life_days: float = Field(default=14, gt=0)
    freshness_half_life_days: float = Field(default=14, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
