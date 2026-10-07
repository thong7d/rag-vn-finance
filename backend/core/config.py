"""
config.py — Pydantic Settings for FastAPI backend.
All values are loaded from environment variables (set in Render dashboard).
"""

from functools import lru_cache
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── API Keys ─────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    gemini_api_key: str = ""
    gemini_api_key_2: str | None = None
    mistral_api_key: str = ""
    cohere_api_key: str = ""
    hf_token: str = ""

    # ── Qdrant Cloud ──────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""

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

    # ── Database (PostgreSQL / Neon Serverless) ───────────────────────────────
    database_url: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def async_database_url(self) -> str | None:
        """
        Convert DATABASE_URL to asyncpg-compatible format:
          1. Switch scheme to postgresql+asyncpg://
          2. Strip params unsupported by asyncpg (sslmode, channel_binding)
             SSL is handled separately via connect_args in engine.py.
        """
        if not self.database_url:
            return None
        url = self.database_url
        # Switch scheme
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)

        # Strip asyncpg-incompatible query params (libpq-only)
        _STRIP_PARAMS = {"sslmode", "channel_binding"}
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        filtered_qs = {k: v for k, v in qs.items() if k not in _STRIP_PARAMS}
        clean_query = urlencode(filtered_qs, doseq=True)
        cleaned = parsed._replace(query=clean_query)
        return urlunparse(cleaned)

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — called at import time by services."""
    return Settings()
