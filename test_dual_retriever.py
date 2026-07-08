"""
test_dual_retriever.py — Verification script to compare FAISS and Qdrant retrievers.
Using English prints to prevent Windows terminal encoding issues.
"""

import os
import json
import faiss
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils import load_config, get_env
from src.retrieval import create_dense_retriever
from app import OpenRouterEmbeddingAPI  # Reuse OpenRouter Embedding connection

def main():
    print("=== START DUAL RETRIEVER COMPARISON TEST ===")
    
    # Init embedding model
    embedding_model = OpenRouterEmbeddingAPI()
    
    # Load metadata and chunk IDs
    STRATEGY = "fixed_size"
    INDEXES_DIR = "indexes"
    faiss_dir = os.path.join(INDEXES_DIR, STRATEGY)
    
    with open(os.path.join(faiss_dir, 'chunk_ids.json'), 'r', encoding='utf-8') as f:
        dense_chunk_ids = json.load(f)
        
    print(f"Total chunk IDs loaded: {len(dense_chunk_ids)}")
    
    # 1. Initialize FAISS Retriever
    print("\n--- 1. Initializing FAISS Retriever ---")
    faiss_index = faiss.read_index(os.path.join(faiss_dir, 'index.faiss'))
    faiss_retriever = create_dense_retriever(
        backend="faiss",
        index=faiss_index,
        chunk_ids=dense_chunk_ids,
        model=embedding_model
    )
    print("FAISS Retriever initialized successfully.")
    
    # 2. Initialize Qdrant Retriever
    print("\n--- 2. Initializing Qdrant Retriever ---")
    from qdrant_client import QdrantClient
    config = load_config()
    qdrant_url = get_env("QDRANT_URL")
    qdrant_api_key = get_env("QDRANT_API_KEY")
    collection_name = f"{config.get('vector_store', {}).get('qdrant', {}).get('collection_name', 'vn_finance')}_{STRATEGY}"
    
    if qdrant_url:
        print(f"Connecting to remote Qdrant Cloud: {qdrant_url}")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        local_path = config.get("vector_store", {}).get("qdrant", {}).get("local_path", "qdrant_data")
        qdrant_dir = os.path.join(local_path, STRATEGY)
        print(f"Connecting to local Qdrant at: {qdrant_dir}")
        qdrant_client = QdrantClient(path=qdrant_dir)
        
    qdrant_retriever = create_dense_retriever(
        backend="qdrant",
        client=qdrant_client,
        collection_name=collection_name,
        chunk_ids=dense_chunk_ids,
        model=embedding_model
    )
    print("Qdrant Retriever initialized successfully.")
    
    # 3. Perform test query
    query = "đánh giá tác động của lãi suất đến thị trường chứng khoán"
    print(f"\n--- 3. Running test query: '{query}' ---")
    
    top_k = 10
    print("Retrieving from FAISS...")
    faiss_results = faiss_retriever.retrieve(query, top_k=top_k)
    
    print("Retrieving from Qdrant...")
    qdrant_results = qdrant_retriever.retrieve(query, top_k=top_k)
    
    # 4. Compare results
    print("\n--- 4. Comparison Results ---")
    
    print("\n[FAISS Results]")
    for rank, (cid, score) in enumerate(faiss_results, 1):
        print(f"  {rank}. {cid} | Score: {score:.5f}")
        
    print("\n[QDRANT Results]")
    for rank, (cid, score) in enumerate(qdrant_results, 1):
        print(f"  {rank}. {cid} | Score: {score:.5f}")
        
    faiss_ids = [cid for cid, _ in faiss_results]
    qdrant_ids = [cid for cid, _ in qdrant_results]
    
    overlap = len(set(faiss_ids).intersection(set(qdrant_ids)))
    overlap_pct = (overlap / top_k) * 100
    
    print(f"\nOverlap between FAISS and Qdrant top-{top_k}: {overlap}/{top_k} ({overlap_pct:.1f}%)")
    
    # Check rank alignment
    exact_rank_matches = 0
    for idx in range(min(len(faiss_ids), len(qdrant_ids))):
        if faiss_ids[idx] == qdrant_ids[idx]:
            exact_rank_matches += 1
    print(f"Exact rank position matches: {exact_rank_matches}/{top_k}")
    
    if overlap_pct >= 90:
        print("\n✅ SUCCESS: Similarity test passed (>=90%)!")
    else:
        print("\n⚠️ WARNING: Similarity test failed (<90%). Please check metrics/vectors.")

if __name__ == "__main__":
    main()
