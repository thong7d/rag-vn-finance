"""
bm25_loader.py — Download BM25 index from HuggingFace Hub at startup.

Downloads:
  {strategy}/bm25_index.pkl
  {strategy}/chunk_ids.json
to a local temp dir and returns the loaded BM25 + chunk_ids.
"""

import json
import pickle
import tempfile
from pathlib import Path

from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("BM25Loader")

_BM25_CACHE: dict = {}  # module-level cache so we only download once


def load_bm25():
    """
    Download (if needed) and return (bm25_index, chunk_ids).
    Uses module-level cache to avoid re-downloading on every request.
    """
    global _BM25_CACHE

    settings = get_settings()
    strategy = settings.chunk_strategy

    if strategy in _BM25_CACHE:
        logger.info("BM25 already loaded from cache.")
        return _BM25_CACHE[strategy]

    logger.info(f"Downloading BM25 index for strategy='{strategy}' from HF Hub '{settings.hf_bm25_repo}'...")

    from huggingface_hub import hf_hub_download

    tmp_dir = Path(tempfile.gettempdir()) / "rag_bm25" / strategy
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = hf_hub_download(
        repo_id=settings.hf_bm25_repo,
        filename=f"{strategy}/bm25_index.pkl",
        repo_type="dataset",
        token=settings.hf_token,
        local_dir=str(tmp_dir),
        local_dir_use_symlinks=False,
    )
    ids_path = hf_hub_download(
        repo_id=settings.hf_bm25_repo,
        filename=f"{strategy}/chunk_ids.json",
        repo_type="dataset",
        token=settings.hf_token,
        local_dir=str(tmp_dir),
        local_dir_use_symlinks=False,
    )

    with open(pkl_path, "rb") as f:
        bm25_index = pickle.load(f)

    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    logger.info(f"BM25 loaded — {len(chunk_ids):,} chunks, avgdl={bm25_index.avgdl:.1f}")
    _BM25_CACHE[strategy] = (bm25_index, chunk_ids)
    return _BM25_CACHE[strategy]
