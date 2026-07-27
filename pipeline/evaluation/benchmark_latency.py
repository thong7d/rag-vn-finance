"""
benchmark_latency.py — Measure Retrieval Time & End-to-End Latency
====================================================================
Reads an existing generation results parquet (which already has LLM latency_s)
and augments it with:
  - retrieval_time_s  : Embedding + Vector Search + Reranking time
  - e2e_latency_s     : retrieval_time_s + latency_s (LLM generation)

The script reuses the EXACT same retrieval pipeline as app.py:
  1. Encode query via OpenRouterEmbeddingAPI or local SentenceTransformer
  2. HybridRetriever.retrieve(top_k=30) — Dense(Qdrant) + Sparse(BM25) + RRF
  3. rerank_with_cohere(top_n=5)

This ensures the measured retrieval time matches the production environment.

Usage:
  python evaluation/benchmark_latency.py --model gemini [--limit 5]
  python evaluation/benchmark_latency.py --model mistral
  python evaluation/benchmark_latency.py --model gemma
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env
from src.retrieval import HybridRetriever, SparseRetriever, create_dense_retriever
from src.indexing import load_bm25_index

logger = setup_logger("BenchLatency")
config = load_config()

# ── Cohere Reranker (same logic as app.py) ──────────────────────────────────
import cohere

def rerank_with_cohere(query: str, candidates: list, chunk_text_map: dict) -> list:
    cohere_key = get_env("COHERE_API_KEY")
    if not cohere_key:
        logger.warning("[Reranker] COHERE_API_KEY not set, skipping rerank.")
        return candidates[:5]

    docs = [chunk_text_map.get(cid, "") for cid, _ in candidates]
    valid_pairs = [(cs, d) for cs, d in zip(candidates, docs) if d.strip()]
    if not valid_pairs:
        return candidates[:5]

    valid_candidates, valid_docs = zip(*valid_pairs)
    try:
        co = cohere.Client(api_key=cohere_key)
        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=list(valid_docs),
            top_n=5,
        )
        reranked = [
            (valid_candidates[r.index][0], r.relevance_score)
            for r in response.results
        ]
        return reranked
    except Exception as e:
        logger.warning("[Reranker] Cohere failed (%s), fallback to top-5.", e)
        return candidates[:5]


def atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info("[Checkpoint] Saved %d rows -> %s", len(df), path)
    except Exception as exc:
        logger.error("[Checkpoint] Save failed: %s", exc)
        if tmp.exists():
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval + E2E latency")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gemini", "mistral", "gemma"],
                        help="Which model's parquet to augment")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-api-embedding", action="store_true", default=False,
                        help="Use OpenRouter API for embedding (like production). "
                             "Default: use local SentenceTransformer.")
    args = parser.parse_args()

    # ── Resolve file paths ──────────────────────────────────────────────────
    OUTPUT_DIR = ROOT / config["evaluation"]["output_dir"]
    backend = get_env("VECTOR_STORE_BACKEND", config["vector_store"]["backend"]).lower()

    if args.model == "gemini":
        PARQUET_PATH = OUTPUT_DIR / f"generation_results_{backend}.parquet"
    else:
        PARQUET_PATH = OUTPUT_DIR / f"generation_results_{args.model}.parquet"

    if not PARQUET_PATH.exists():
        logger.error("File not found: %s", PARQUET_PATH)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows from %s", len(df), PARQUET_PATH)

    # Add new columns if missing
    for col in ["retrieval_time_s", "e2e_latency_s"]:
        if col not in df.columns:
            df[col] = float("nan")

    # ── Initialize retrieval pipeline ────────────────────────────────────────
    STRATEGY = "sentence_aware"  # Best strategy from Phase 6
    index_dir = str(ROOT / config["indexing"]["output_dir"] / STRATEGY)
    bm25_dir = str(ROOT / config["indexing"]["bm25_dir"])

    # Load chunk metadata
    with open(os.path.join(index_dir, "chunk_ids.json"), "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    df_meta = pd.read_parquet(os.path.join(index_dir, "metadata.parquet"))
    chunk_text_map = dict(zip(df_meta["chunk_id"], df_meta["text"]))

    # Load embedding model
    if args.use_api_embedding:
        import requests as req_lib

        class OpenRouterEmbeddingAPI:
            def __init__(self):
                self.api_url = "https://openrouter.ai/api/v1/embeddings"
                self.api_key = get_env("OPENROUTER_API_KEY")
                self.model_name = "intfloat/multilingual-e5-large"
            def encode(self, texts, *a, **kw):
                if isinstance(texts, str): texts = [texts]
                prefixed = [t if t.startswith("query: ") else f"query: {t}" for t in texts]
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {"model": self.model_name, "input": prefixed}
                resp = req_lib.post(self.api_url, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    return np.array([item["embedding"] for item in data], dtype=np.float32)
                raise RuntimeError(f"OpenRouter API error: {resp.status_code} - {resp.text}")

        embedding_model = OpenRouterEmbeddingAPI()
        logger.info("Using OpenRouter API for embedding (production-like).")
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(config["embedding"]["model_name"], device=device)
        logger.info("Using local SentenceTransformer on %s.", device)

    # Load dense retriever
    if backend == "qdrant":
        from qdrant_client import QdrantClient
        qdrant_url = get_env("QDRANT_URL")
        qdrant_api_key = get_env("QDRANT_API_KEY")
        collection_name = f"{config.get('vector_store', {}).get('qdrant', {}).get('collection_name', 'vn_finance')}_{STRATEGY}"

        if qdrant_url:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            local_path = config.get("vector_store", {}).get("qdrant", {}).get("local_path", "qdrant_data")
            qdrant_dir = ROOT / local_path / STRATEGY
            qdrant_client = QdrantClient(path=str(qdrant_dir))

        dense_retriever = create_dense_retriever(
            backend="qdrant", client=qdrant_client,
            collection_name=collection_name, chunk_ids=chunk_ids,
            model=embedding_model
        )
    else:
        import faiss
        faiss_path = os.path.join(index_dir, "index.faiss")
        faiss_index = faiss.read_index(faiss_path)
        dense_retriever = create_dense_retriever(
            backend="faiss", index=faiss_index,
            chunk_ids=chunk_ids, model=embedding_model
        )

    # Load BM25
    bm25_index, bm25_chunk_ids = load_bm25_index(bm25_dir, STRATEGY)
    sparse_retriever = SparseRetriever(bm25_index, bm25_chunk_ids)

    # Hybrid
    retriever = HybridRetriever(dense_retriever, sparse_retriever, rrf_k=config["retrieval"]["rrf_k"])
    logger.info("Retrieval pipeline initialized (backend=%s, strategy=%s).", backend, STRATEGY)

    # ── Identify pending rows ────────────────────────────────────────────────
    needs_bench = df["retrieval_time_s"].isna()
    pending_indices = df[needs_bench].index.tolist()
    if args.limit:
        pending_indices = pending_indices[:args.limit]

    logger.info("Rows needing benchmark: %d", len(pending_indices))
    if not pending_indices:
        logger.info("All rows already benchmarked. Exiting.")
        return

    # ── Benchmark loop ───────────────────────────────────────────────────────
    for count, row_idx in enumerate(pending_indices):
        question = str(df.at[row_idx, "question"]).strip()
        llm_latency = df.at[row_idx, "latency_s"] if "latency_s" in df.columns else 0.0
        if pd.isna(llm_latency):
            llm_latency = 0.0

        logger.info("--- %d/%d --- Q: %.60s...", count + 1, len(pending_indices), question)

        try:
            # Measure full retrieval pipeline: embedding + search + rerank
            t0 = time.perf_counter()

            # Step 1+2: Hybrid retrieval (embedding + vector search + BM25 + RRF)
            retrieved = retriever.retrieve(question, top_k=30)

            # Step 3: Cohere reranking
            top_results = rerank_with_cohere(question, retrieved, chunk_text_map)

            retrieval_time = time.perf_counter() - t0
            e2e_latency = retrieval_time + llm_latency

            df.at[row_idx, "retrieval_time_s"] = retrieval_time
            df.at[row_idx, "e2e_latency_s"] = e2e_latency

            logger.info(
                "Retrieval: %.3fs | LLM: %.3fs | E2E: %.3fs",
                retrieval_time, llm_latency, e2e_latency
            )

        except Exception as e:
            logger.error("Benchmark failed for row %d: %s", row_idx, e)
            df.at[row_idx, "retrieval_time_s"] = float("nan")
            df.at[row_idx, "e2e_latency_s"] = float("nan")

        # Checkpoint every 10 rows
        if (count + 1) % 10 == 0 or count == len(pending_indices) - 1:
            atomic_save_parquet(df, PARQUET_PATH)

        # Small cooldown to respect Cohere rate limits
        time.sleep(1.0)

    # ── Print summary ────────────────────────────────────────────────────────
    valid = df["retrieval_time_s"].notna()
    if valid.sum() > 0:
        logger.info("="*60)
        logger.info("BENCHMARK SUMMARY for %s (%d samples)", args.model.upper(), valid.sum())
        logger.info("  Avg Retrieval Time : %.3fs", df.loc[valid, "retrieval_time_s"].mean())
        logger.info("  Avg LLM Latency    : %.3fs", df.loc[valid, "latency_s"].mean() if "latency_s" in df.columns else 0)
        logger.info("  Avg E2E Latency    : %.3fs", df.loc[valid, "e2e_latency_s"].mean())
        logger.info("  Min E2E            : %.3fs", df.loc[valid, "e2e_latency_s"].min())
        logger.info("  Max E2E            : %.3fs", df.loc[valid, "e2e_latency_s"].max())
        logger.info("  P50 E2E            : %.3fs", df.loc[valid, "e2e_latency_s"].quantile(0.50))
        logger.info("  P95 E2E            : %.3fs", df.loc[valid, "e2e_latency_s"].quantile(0.95))
        logger.info("============================================================")
    logger.info("Done. Results saved to %s", PARQUET_PATH)

    # Explicitly close qdrant client to avoid __del__ ImportError on shutdown
    if 'qdrant_client' in locals():
        qdrant_client.close()


if __name__ == "__main__":
    main()
