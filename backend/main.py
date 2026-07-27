"""
main.py — FastAPI application entry point.

Startup sequence (lifespan):
  1. Download SQLite FTS5 database from HuggingFace Hub
  2. Fetch chunk metadata from Qdrant Cloud (used for text/title/url lookup)
  3. Initialize retrieval pipeline (Qdrant client + embedding model)

The backend is then ready to serve POST /api/ask with SSE streaming.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import get_settings
from core.logging import setup_logger
from routers.ask import router
from services import retrieval as retrieval_service
from services.sqlite_loader import get_sqlite_path

logger = setup_logger("Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager — runs startup/shutdown logic."""
    settings = get_settings()
    logger.info("=== RAG Backend Starting ===")

    # Step 1: Download SQLite FTS5 DB from HF Hub (cached after first download)
    logger.info("Step 1/3: Loading SQLite FTS5 DB from HuggingFace Hub...")
    get_sqlite_path()  # warms up the module-level cache

    # Step 2: Fetch chunk metadata from Qdrant Cloud IN BACKGROUND
    logger.info("Step 2/3: Starting background task to fetch chunk metadata...")
    import asyncio
    
    def fetch_metadata():
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        collection_name = f"vn_finance_{settings.chunk_strategy}"
        chunk_meta: dict[str, dict] = {}
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            batch, next_offset = result
            if not batch:
                break
            for pt in batch:
                cid = pt.payload.get("chunk_id", "")
                if cid:
                    chunk_meta[cid] = {
                        "text": pt.payload.get("text", ""),
                        "title": pt.payload.get("title", ""),
                        "url": pt.payload.get("url", ""),
                    }
            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"   → Loaded metadata for {len(chunk_meta):,} chunks")
        logger.info("Step 3/3: Initializing retrieval pipeline...")
        retrieval_service.init_retrieval(chunk_meta)

    # Chạy trên một thread riêng để hoàn toàn không block event loop
    asyncio.create_task(asyncio.to_thread(fetch_metadata))

    logger.info("=== RAG Backend Ready (Initializing in background) ===")
    yield

    # Shutdown
    logger.info("=== RAG Backend Shutting Down ===")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Vietnamese Financial News RAG API",
        description="Production-grade RAG backend with 3-layer LLM fallback and SSE streaming.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.api_route("/", methods=["GET", "HEAD"])
    def root_health_check():
        return {"status": "ok", "message": "Backend is running"}

    return app


app = create_app()
