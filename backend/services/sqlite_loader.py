"""
sqlite_loader.py — Download SQLite FTS5 index from HuggingFace Hub at startup.

Downloads:
  {strategy}/bm25.db
to a local temp dir and returns the path to the DB.
"""

import sqlite3
import tempfile
from pathlib import Path

from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("SQLiteLoader")

_SQLITE_DB_PATH: str | None = None  # module-level cache


def get_sqlite_path() -> str:
    """
    Download (if needed) and return the path to bm25.db.
    Uses module-level cache to avoid re-downloading on every request.
    """
    global _SQLITE_DB_PATH

    settings = get_settings()
    strategy = settings.chunk_strategy

    if _SQLITE_DB_PATH:
        return _SQLITE_DB_PATH

    logger.info(f"Downloading SQLite FTS5 DB for strategy='{strategy}' from HF Hub '{settings.hf_bm25_repo}'...")

    from huggingface_hub import hf_hub_download

    tmp_dir = Path(tempfile.gettempdir()) / "rag_sqlite" / strategy
    tmp_dir.mkdir(parents=True, exist_ok=True)

    db_path = hf_hub_download(
        repo_id=settings.hf_bm25_repo,
        filename=f"{strategy}/bm25.db",
        repo_type="dataset",
        token=settings.hf_token,
        local_dir=str(tmp_dir),
        local_dir_use_symlinks=False,
    )

    logger.info(f"SQLite DB loaded at: {db_path}")
    
    # Optional: Verify connection
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        logger.info(f"SQLite FTS5 DB verified — {count:,} chunks")
        conn.close()
    except Exception as e:
        logger.error(f"Error verifying SQLite DB: {e}")

    _SQLITE_DB_PATH = db_path
    return _SQLITE_DB_PATH
