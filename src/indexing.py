"""
indexing.py — BM25 sparse indexing for Phase 4.

Builds a BM25Okapi index per chunking strategy using rank-bm25.
Each index is paired with an ordered chunk_ids.json so that the BM25
internal row index maps 1-to-1 with chunk identifiers — enabling
rank fusion with the FAISS dense index in Phase 6.

Tokenization
------------
Baseline: whitespace-based Vietnamese tokenizer (punctuation stripped,
single-char tokens discarded).

Known trade-off: whitespace split does not segment Vietnamese compound
words (e.g., "ngân hàng" → two separate tokens instead of one concept).
underthesea word segmentation would improve recall but is optional;
document this limitation explicitly in the Phase 10 report.

Output per strategy
-------------------
  bm25/{strategy}/bm25_index.pkl   — serialized BM25Okapi object
  bm25/{strategy}/chunk_ids.json   — ordered list of chunk_id strings
  bm25/{strategy}/build_stats.json — build metadata (time, vocab size, …)
"""

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_vi(text: str) -> list[str]:
    """
    Whitespace-based Vietnamese tokenizer for BM25.

    Steps:
      1. Lowercase.
      2. Replace all non-alphanumeric / non-whitespace characters with a space.
      3. Split on whitespace.
      4. Discard single-character tokens (they carry little BM25 signal).

    Known limitation: does not perform true Vietnamese word segmentation.
    Compound words like "ngân hàng" are indexed as two separate unigrams.
    See underthesea for an upgrade path.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_bm25_index(
    df_chunks: pd.DataFrame,
    output_dir: str,
    strategy: str,
) -> tuple[object, list[str]]:
    """
    Build a BM25Okapi index from a chunk DataFrame and persist it.

    The function is idempotent: if `bm25_index.pkl` already exists in
    `output_dir`, it is loaded from disk and returned immediately without
    rebuilding.

    Parameters
    ----------
    df_chunks : pd.DataFrame
        Chunk table for one strategy.  Must contain columns:
        ``chunk_id`` and ``text``.
    output_dir : str
        Directory for output files (created if absent).
    strategy : str
        Strategy label used for logging and ``build_stats.json``.

    Returns
    -------
    bm25 : BM25Okapi
        Fitted BM25 index.
    chunk_ids : list[str]
        Ordered list of chunk_id strings aligned with BM25 internal rows.
    """
    from rank_bm25 import BM25Okapi

    ensure_dir(output_dir)
    pkl_path   = os.path.join(output_dir, "bm25_index.pkl")
    ids_path   = os.path.join(output_dir, "chunk_ids.json")
    stats_path = os.path.join(output_dir, "build_stats.json")

    # ── Idempotent guard ────────────────────────────────────────────────────
    if os.path.exists(pkl_path) and os.path.exists(ids_path):
        logger.info(f"[{strategy}] BM25 index already exists — loading from cache.")
        bm25       = _load_bm25(pkl_path)
        chunk_ids  = _load_chunk_ids(ids_path)
        logger.info(f"[{strategy}] Loaded: {len(chunk_ids):,} chunks.")
        return bm25, chunk_ids

    # ── Guarantee row order is stable ───────────────────────────────────────
    df_chunks = df_chunks.reset_index(drop=True)
    chunk_ids = df_chunks["chunk_id"].tolist()
    texts     = df_chunks["text"].tolist()

    logger.info(f"[{strategy}] Tokenizing {len(texts):,} chunks…")
    t0 = time.time()

    tokenized = [
        tokenize_vi(t)
        for t in tqdm(texts, desc=f"Tokenizing [{strategy}]", unit="chunk")
    ]

    tok_elapsed = time.time() - t0
    logger.info(f"[{strategy}] Tokenization done in {tok_elapsed:.1f}s")

    # ── Build index ─────────────────────────────────────────────────────────
    logger.info(f"[{strategy}] Building BM25Okapi index…")
    t1  = time.time()
    bm25 = BM25Okapi(tokenized)
    build_elapsed = time.time() - t1
    logger.info(f"[{strategy}] Index built in {build_elapsed:.1f}s")

    # ── Compute vocabulary size ─────────────────────────────────────────────
    all_tokens  = [tok for doc in tokenized for tok in doc]
    vocab_size  = len(set(all_tokens))
    avg_doc_len = sum(len(d) for d in tokenized) / max(len(tokenized), 1)

    # ── Persist ─────────────────────────────────────────────────────────────
    _save_bm25(bm25, pkl_path)
    _save_chunk_ids(chunk_ids, ids_path)

    stats = {
        "strategy":            strategy,
        "total_chunks":        len(chunk_ids),
        "vocab_size":          vocab_size,
        "avg_doc_length_tokens": round(avg_doc_len, 1),
        "tokenization_seconds": round(tok_elapsed, 1),
        "build_seconds":       round(build_elapsed, 1),
        "tokenizer":           "whitespace_vi",
        "bm25_variant":        "BM25Okapi",
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{strategy}] Saved: {pkl_path}  "
        f"(vocab={vocab_size:,}, avg_doc_len={avg_doc_len:.1f} tokens)"
    )
    return bm25, chunk_ids


# ---------------------------------------------------------------------------
# Build all 3 strategies in one call
# ---------------------------------------------------------------------------

def build_all_bm25_indexes(
    chunks_base_dir: str,
    bm25_base_dir: str,
    strategies: Optional[list[str]] = None,
) -> dict:
    """
    Iterate over all chunking strategies, build a BM25 index for each,
    and return a summary dict.

    Parameters
    ----------
    chunks_base_dir : str
        Root directory containing ``{strategy}/chunks.parquet`` files.
    bm25_base_dir : str
        Root output directory; indexes go into ``{bm25_base_dir}/{strategy}/``.
    strategies : list[str], optional
        Subset of strategies to build.  Defaults to all three.

    Returns
    -------
    dict
        Mapping strategy → {bm25, chunk_ids, stats_path}.
    """
    if strategies is None:
        strategies = ["fixed_size", "sentence_aware", "article_level"]

    results = {}
    for strategy in strategies:
        chunks_path = os.path.join(chunks_base_dir, strategy, "chunks.parquet")
        output_dir  = os.path.join(bm25_base_dir,   strategy)

        if not os.path.exists(chunks_path):
            logger.warning(
                f"[{strategy}] chunks.parquet not found at {chunks_path} — skipping."
            )
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing strategy: {strategy}")

        df_chunks = pd.read_parquet(chunks_path)
        logger.info(f"[{strategy}] Loaded {len(df_chunks):,} chunks.")

        bm25, chunk_ids = build_bm25_index(df_chunks, output_dir, strategy)

        results[strategy] = {
            "bm25":       bm25,
            "chunk_ids":  chunk_ids,
            "output_dir": output_dir,
        }

    logger.info("All BM25 indexes built.")
    return results


# ---------------------------------------------------------------------------
# Load helpers (used in Phase 6 retrieval)
# ---------------------------------------------------------------------------

def load_bm25_index(bm25_dir: str, strategy: str) -> tuple[object, list[str]]:
    """
    Load a pre-built BM25 index and its chunk_ids from disk.

    Parameters
    ----------
    bm25_dir : str
        Root BM25 directory (e.g., ``bm25/``).
    strategy : str
        One of ``fixed_size``, ``sentence_aware``, ``article_level``.

    Returns
    -------
    tuple[BM25Okapi, list[str]]
        The fitted BM25 object and ordered chunk_id list.
    """
    strategy_dir = os.path.join(bm25_dir, strategy)
    pkl_path     = os.path.join(strategy_dir, "bm25_index.pkl")
    ids_path     = os.path.join(strategy_dir, "chunk_ids.json")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"BM25 index not found: {pkl_path}\n"
            "Run Phase 4 notebook (04_bm25_indexing.ipynb) first."
        )

    bm25      = _load_bm25(pkl_path)
    chunk_ids = _load_chunk_ids(ids_path)
    logger.info(
        f"[{strategy}] BM25 index loaded — {len(chunk_ids):,} chunks, "
        f"avgdl={bm25.avgdl:.1f}"
    )
    return bm25, chunk_ids


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------

def _save_bm25(bm25, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"BM25 index serialised: {path}  ({size_mb:.1f} MB)")


def _load_bm25(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_chunk_ids(chunk_ids: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    logger.info(f"chunk_ids.json saved: {path}  ({len(chunk_ids):,} IDs)")


def _load_chunk_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_bm25_index(bm25_dir: str, strategy: str) -> bool:
    """
    Verify that the BM25 index and chunk_ids are consistent.

    Checks:
      - Both files exist.
      - Number of IDs matches BM25 corpus length.

    Returns True on success, raises AssertionError on failure.
    """
    strategy_dir = os.path.join(bm25_dir, strategy)
    pkl_path     = os.path.join(strategy_dir, "bm25_index.pkl")
    ids_path     = os.path.join(strategy_dir, "chunk_ids.json")

    assert os.path.exists(pkl_path), f"Missing: {pkl_path}"
    assert os.path.exists(ids_path), f"Missing: {ids_path}"

    bm25      = _load_bm25(pkl_path)
    chunk_ids = _load_chunk_ids(ids_path)

    n_bm25 = bm25.corpus_size if hasattr(bm25, "corpus_size") else len(bm25.doc_freqs)
    # BM25Okapi exposes corpus length via idf dict length in worst case;
    # the safest proxy is the length of the tokenized corpus stored internally.
    n_corpus = bm25.corpus_size if hasattr(bm25, "corpus_size") else len(chunk_ids)
    n_ids    = len(chunk_ids)

    assert n_corpus == n_ids, (
        f"[{strategy}] Alignment error: BM25 corpus={n_corpus}, "
        f"chunk_ids={n_ids}"
    )
    logger.info(
        f"[{strategy}] BM25 index verified — {n_ids:,} chunks, "
        f"avgdl={bm25.avgdl:.1f} tokens"
    )
    return True
