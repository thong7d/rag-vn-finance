"""
config.py — Pydantic Settings for FastAPI backend.
All values are loaded from environment variables (set in Render dashboard).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # ── API Keys ─────────────────────────────────────────────────────────────
    openrouter_api_key: str
    gemini_api_key: str
    gemini_api_key_2: str | None = None
    mistral_api_key: str
    cohere_api_key: str
    hf_token: str

    # ── Qdrant Cloud ──────────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str

    # ── BM25 on HuggingFace Hub ───────────────────────────────────────────────
    hf_bm25_repo: str = "thong7d/rag-vn-finance-bm25"
    chunk_strategy: str = "sentence_aware"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_hybrid: int = 30        # candidates sent to Cohere Reranker
    top_k_rerank: int = 5         # final passages sent to LLM
    rrf_k: int = 60

    # ── Generation ───────────────────────────────────────────────────────────
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.2

    # ── Query Decomposition ──────────────────────────────────────────────────
    decompose_max_subqueries: int = 3

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Comma-separated list; set to your Vercel domain on production
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — called at import time by services."""
    return Settings()
