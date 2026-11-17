"""
retrieval.py — Hybrid retrieval for the backend API.
Pipeline: Qdrant Cloud (dense) + BM25 (sparse) → RRF → Cohere Reranker.

Returns list of dicts:
  [{"chunk_id": str, "text": str, "title": str, "url": str, "score": float}, ...]
"""

import re
from typing import Any

import cohere
from qdrant_client import QdrantClient

import sqlite3
import time

from core.config import get_settings
from core.logging import setup_logger
from core.metrics import rag_retrieval_latency_seconds
from services.sqlite_loader import get_sqlite_path
from services.embedding import OpenRouterEmbeddingAPI

logger = setup_logger("RetrievalService")

# ── Module-level singletons (initialized once at startup via init_retrieval()) ──
_qdrant_client: QdrantClient | None = None
_embedding_model: OpenRouterEmbeddingAPI | None = None
_chunk_cache: dict[str, dict] = {}  # bounded in-memory cache: chunk_id → {text, title, url}
_MAX_CACHE_SIZE = 2000


def _cache_chunk_payload(chunk_id: str, payload: dict) -> None:
    """Store chunk payload in a bounded cache (prevent unbounded memory growth)."""
    global _chunk_cache
    if chunk_id and chunk_id not in _chunk_cache:
        if len(_chunk_cache) >= _MAX_CACHE_SIZE:
            to_remove = list(_chunk_cache.keys())[:500]
            for k in to_remove:
                _chunk_cache.pop(k, None)
        _chunk_cache[chunk_id] = {
            "text": payload.get("text", ""),
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
        }


def get_chunk_meta(chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch chunk metadata on-demand for specific IDs, using Qdrant filter if needed."""
    global _chunk_cache, _qdrant_client
    settings = get_settings()

    missing = [cid for cid in chunk_ids if cid and (cid not in _chunk_cache or not _chunk_cache[cid].get("text"))]
    if missing and _qdrant_client is not None:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            scroll_filter = Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=missing))])
            pts, _ = _qdrant_client.scroll(
                collection_name=f"vn_finance_{settings.chunk_strategy}",
                scroll_filter=scroll_filter,
                limit=len(missing),
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                cid = p.payload.get("chunk_id", "")
                if cid:
                    _cache_chunk_payload(cid, p.payload)
        except Exception as e:
            logger.warning(f"On-demand chunk metadata lookup error: {e}")

    return {cid: _chunk_cache.get(cid, {}) for cid in chunk_ids}


def tokenize_vi(text: str) -> list[str]:
    """Whitespace-based Vietnamese tokenizer (mirrors pipeline/src/indexing.py)."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def init_retrieval(chunk_meta: dict[str, dict] | None = None):
    """
    Called once at startup (lifespan) to initialize Qdrant client and embedding model.
    No longer holds 64k chunks in RAM; metadata is loaded on demand.
    """
    global _qdrant_client, _embedding_model, _chunk_cache

    settings = get_settings()

    logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
    _qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    _embedding_model = OpenRouterEmbeddingAPI()
    if chunk_meta:
        for k, v in list(chunk_meta.items())[:_MAX_CACHE_SIZE]:
            _chunk_cache[k] = v

    collection_name = f"vn_finance_{settings.chunk_strategy}"
    info = _qdrant_client.get_collection(collection_name)
    logger.info(f"Qdrant collection '{collection_name}': {info.points_count} points")


def _dense_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    settings = get_settings()
    collection_name = f"vn_finance_{settings.chunk_strategy}"
    t0 = time.perf_counter()

    query_emb = _embedding_model.encode([query])[0].tolist()

    results = _qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
        with_payload=True,
    )

    rag_retrieval_latency_seconds.labels(type="dense").observe(time.perf_counter() - t0)

    dense_results = []
    for hit in results.points:
        if hit.payload and "chunk_id" in hit.payload:
            cid = hit.payload["chunk_id"]
            score = float(hit.score) if isinstance(hit.score, (int, float, str)) else 0.0
            _cache_chunk_payload(cid, hit.payload)
            dense_results.append((cid, score))

    return dense_results


def _sparse_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    t0 = time.perf_counter()
    db_path = get_sqlite_path()
    tokens = tokenize_vi(query)

    if not tokens:
        return []

    # FTS5 matches using OR to sum up IDF exactly like rank-bm25
    fts_query = " OR ".join(tokens)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # SQLite bm25() returns negative score, lower is better. We negate it to get positive score.
        cur.execute('''
            SELECT chunk_id, -bm25(chunks) AS score
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY bm25(chunks) ASC
            LIMIT ?
        ''', (fts_query, top_k))

        results = [(row[0], float(row[1])) for row in cur.fetchall()]

        rag_retrieval_latency_seconds.labels(type="sparse").observe(time.perf_counter() - t0)
        return results
    except Exception as e:
        logger.error(f"SQLite FTS5 query failed: {e}")
        return []
    finally:
        conn.close()


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
    t0 = time.perf_counter()

    cids = [cid for cid, _ in candidates]
    meta_map = get_chunk_meta(cids)
    docs = [meta_map.get(cid, {}).get("text", "") for cid in cids]
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

        rag_retrieval_latency_seconds.labels(type="rerank").observe(time.perf_counter() - t0)
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
    cids = [cid for cid, _ in reranked]
    meta_map = get_chunk_meta(cids)
    results = []
    for cid, score in reranked:
        meta = meta_map.get(cid, {})
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


# ── Public step functions (for async pipeline orchestration in ask.py) ────────

def dense_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    """Public wrapper — dense vector search via Qdrant + HF embedding."""
    return _dense_retrieve(query, top_k)


def sparse_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    """Public wrapper — BM25 keyword search via SQLite FTS5."""
    return _sparse_retrieve(query, top_k)


def fuse_and_rerank(query: str, dense: list, sparse: list) -> list[dict[str, Any]]:
    """
    Public — RRF fusion + Cohere Rerank + build result dicts.
    Called by ask.py after emitting retrieval progress events.
    """
    settings = get_settings()
    fused = _rrf(dense, sparse, k=settings.rrf_k, top_n=settings.top_k_hybrid)
    reranked = _rerank(query, fused, top_n=settings.top_k_rerank)
    cids = [cid for cid, _ in reranked]
    meta_map = get_chunk_meta(cids)
    results = []
    for cid, score in reranked:
        meta = meta_map.get(cid, {})
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
