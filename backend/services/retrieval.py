"""
retrieval.py — Hybrid retrieval for the backend API.
Pipeline: Qdrant Cloud (dense) + BM25 (sparse) → RRF → Cohere Reranker.

Returns list of dicts:
  [{"chunk_id": str, "text": str, "title": str, "url": str, "score": float}, ...]
"""

import re
from typing import Any

import cohere
import numpy as np
from qdrant_client import QdrantClient

from backend.core.config import get_settings
from backend.core.logging import setup_logger
from backend.services.bm25_loader import load_bm25
from backend.services.embedding import OpenRouterEmbeddingAPI

logger = setup_logger("RetrievalService")

# ── Module-level singletons (initialized once at startup via init_retrieval()) ──
_qdrant_client: QdrantClient | None = None
_embedding_model: OpenRouterEmbeddingAPI | None = None
_chunk_meta: dict[str, dict] = {}  # chunk_id → {text, title, url}


def tokenize_vi(text: str) -> list[str]:
    """Whitespace-based Vietnamese tokenizer (mirrors implementation/src/indexing.py)."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def init_retrieval(chunk_meta: dict[str, dict]):
    """
    Called once at startup (lifespan) to initialize Qdrant client,
    embedding model, and store chunk metadata.

    Args:
        chunk_meta: dict mapping chunk_id → {"text": str, "title": str, "url": str}
    """
    global _qdrant_client, _embedding_model, _chunk_meta

    settings = get_settings()

    logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
    _qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    _embedding_model = OpenRouterEmbeddingAPI()
    _chunk_meta = chunk_meta

    collection_name = f"vn_finance_{settings.chunk_strategy}"
    info = _qdrant_client.get_collection(collection_name)
    logger.info(f"Qdrant collection '{collection_name}': {info.points_count} points")


def _dense_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    settings = get_settings()
    collection_name = f"vn_finance_{settings.chunk_strategy}"

    query_emb = _embedding_model.encode([query])[0].tolist()

    results = _qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
    )
    return [(hit.payload["chunk_id"], float(hit.score)) for hit in results.points if "chunk_id" in hit.payload]


def _sparse_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    bm25_index, chunk_ids = load_bm25()
    tokens = tokenize_vi(query)
    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def _rrf(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int,
    top_n: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(sparse):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _rerank(query: str, candidates: list[tuple[str, float]], top_n: int) -> list[tuple[str, float]]:
    settings = get_settings()

    docs = [_chunk_meta.get(cid, {}).get("text", "") for cid, _ in candidates]
    valid = [(c, d) for c, d in zip(candidates, docs) if d.strip()]
    if not valid:
        return candidates[:top_n]

    valid_candidates, valid_docs = zip(*valid)
    try:
        co = cohere.Client(api_key=settings.cohere_api_key)
        resp = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=list(valid_docs),
            top_n=top_n,
        )
        reranked = [(valid_candidates[r.index][0], r.relevance_score) for r in resp.results]
        logger.info(f"Cohere rerank OK — top-{top_n} from {len(candidates)} candidates")
        return reranked
    except Exception as e:
        logger.warning(f"Cohere rerank failed ({e}), falling back to RRF top-{top_n}")
        return candidates[:top_n]


def retrieve(query: str) -> list[dict[str, Any]]:
    """
    Full retrieval pipeline: Dense → Sparse → RRF → Cohere Rerank.

    Returns a list of source dicts:
      [{"chunk_id", "text", "title", "url", "score"}, ...]
    """
    settings = get_settings()

    # Step 1+2: Dense + Sparse (parallel semantics, sequential execution)
    dense_results = _dense_retrieve(query, top_k=settings.top_k_hybrid)
    sparse_results = _sparse_retrieve(query, top_k=settings.top_k_hybrid)

    # Step 3: RRF fusion
    fused = _rrf(dense_results, sparse_results, k=settings.rrf_k, top_n=settings.top_k_hybrid)

    # Step 4: Cohere Rerank → top-5
    reranked = _rerank(query, fused, top_n=settings.top_k_rerank)

    # Step 5: Build result dicts
    results = []
    for cid, score in reranked:
        meta = _chunk_meta.get(cid, {})
        text = meta.get("text", "")
        if text.strip():
            results.append({
                "chunk_id": cid,
                "text": text,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "score": round(score, 4),
            })

    return results
