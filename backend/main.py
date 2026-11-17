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
from db.engine import init_db, dispose_db
from routers.ask import router as ask_router
from routers.feedback import router as feedback_router
from routers.sessions import router as sessions_router
from routers.admin import router as admin_router
from services import retrieval as retrieval_service
from services.sqlite_loader import get_sqlite_path

logger = setup_logger("Main")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager — runs startup/shutdown logic."""
    settings = get_settings()
    logger.info("=== RAG Backend Starting ===")

    # Step 0: Initialize PostgreSQL connection pool (if DATABASE_URL set)
    init_db()

    # Step 1: Download SQLite FTS5 DB from HF Hub (cached after first download)
    logger.info("Step 1/3: Loading SQLite FTS5 DB from HuggingFace Hub...")
    get_sqlite_path()  # warms up the module-level cache

    # Step 2: Initialize retrieval pipeline (metadata loaded on-demand, saving ~350MB RAM)
    logger.info("Step 2/2: Initializing retrieval pipeline...")
    retrieval_service.init_retrieval()

    logger.info("=== RAG Backend Ready ===")
    yield

    # Shutdown
    logger.info("=== RAG Backend Shutting Down ===")
    await dispose_db()


def create_app() -> FastAPI:

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

    app.include_router(ask_router)
    app.include_router(feedback_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.api_route("/", methods=["GET", "HEAD"])
    def root_health_check():
        return {"status": "ok", "message": "Backend is running"}

    return app


app = create_app()
