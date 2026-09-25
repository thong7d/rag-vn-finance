"""
engine.py — Database engine & session management for PostgreSQL (Async).
If DATABASE_URL is not set, all database features will gracefully be disabled.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("Database")

engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def is_db_enabled() -> bool:
    """Return True if PostgreSQL engine is initialized and available."""
    return engine is not None and AsyncSessionLocal is not None


def init_db() -> bool:
    """Initialize DB engine if database_url is provided in environment/settings."""
    global engine, AsyncSessionLocal
    settings = get_settings()
    db_url = settings.async_database_url

    if not db_url:
        logger.info("DATABASE_URL not set — Enterprise PostgreSQL Observability disabled.")
        return False

    try:
        engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info("PostgreSQL Async Engine initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL engine: {e}")
        engine = None
        AsyncSessionLocal = None
        return False


async def dispose_db():
    """Dispose connection pool on app shutdown."""
    global engine, AsyncSessionLocal
    if engine:
        logger.info("Closing PostgreSQL connection pool...")
        await engine.dispose()
        engine = None
        AsyncSessionLocal = None
        logger.info("PostgreSQL connection pool closed.")


async def get_db() -> AsyncGenerator[AsyncSession | None, None]:
    """FastAPI dependency yielding an async DB session (or None if DB is disabled)."""
    if AsyncSessionLocal is None:
        yield None
        return

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
