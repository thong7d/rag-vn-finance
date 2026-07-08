"""
retrieval.py — Dense, sparse BM25, and hybrid RRF retrieval for Phase 6.
9 configs: 3 methods × 3 chunking strategies.
Implemented in Phase 6.
"""

import math
from typing import Dict, List, Tuple
import numpy as np
import faiss

from src.indexing import tokenize_vi

class DenseRetriever:
    def __init__(self, index: faiss.Index, chunk_ids: List[str], model):
        self.index = index
        self.chunk_ids = chunk_ids
        self.model = model

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # E5 requires the "query: " prefix
        prefixed_query = f"query: {query}"
        
        # Encode the query
        query_emb = self.model.encode(
            [prefixed_query],
            show_progress_bar=False,
            normalize_embeddings=False, # We normalize below
            convert_to_numpy=True,
        ).astype("float32")
        
        # L2 normalize to compute cosine similarity with Inner Product index
        faiss.normalize_L2(query_emb)
        
        # Search
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(dist)))
                
        return results

# Alias for backward compatibility and naming consistency in factory pattern
FaissDenseRetriever = DenseRetriever

class QdrantDenseRetriever:
    def __init__(self, client, collection_name: str, model, chunk_ids: List[str] = None):
        self.client = client
        self.collection_name = collection_name
        self.model = model
        self.chunk_ids = chunk_ids

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # E5 requires the "query: " prefix
        prefixed_query = f"query: {query}"
        
        # Encode the query
        query_emb = self.model.encode(
            [prefixed_query],
            show_progress_bar=False,
            normalize_embeddings=False,  # No manual L2 normalization (Qdrant handles Cosine)
            convert_to_numpy=True,
        )
        
        # Convert to a standard list of floats for Qdrant client
        if hasattr(query_emb, "tolist"):
            query_vector = query_emb[0].tolist()
        else:
            query_vector = list(query_emb[0])
            
        # Search Qdrant collection using the unified query_points API
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        # Format output as [(chunk_id, score), ...]
        results = []
        for hit in search_result.points:
            chunk_id = hit.payload.get("chunk_id")
            if chunk_id:
                results.append((chunk_id, float(hit.score)))
                
        return results

def create_dense_retriever(backend: str, **kwargs):
    if backend == "faiss":
        return FaissDenseRetriever(kwargs["index"], kwargs["chunk_ids"], kwargs["model"])
    elif backend == "qdrant":
        return QdrantDenseRetriever(kwargs["client"], kwargs["collection_name"], kwargs["model"], kwargs.get("chunk_ids"))
    else:
        raise ValueError(f"Unsupported vector store backend: {backend}")

class SparseRetriever:
    def __init__(self, bm25_index, chunk_ids: List[str]):
        self.bm25 = bm25_index
        self.chunk_ids = chunk_ids

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokenized_query = tokenize_vi(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx])))
                
        return results

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        dense_results = self.dense.retrieve(query, top_k=top_k)
        sparse_results = self.sparse.retrieve(query, top_k=top_k)
        
        return self._rrf(dense_results, sparse_results, top_k)
        
    def _rrf(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        rrf_scores: Dict[str, float] = {}
        
        # Rank dense
        for rank, (chunk_id, _) in enumerate(dense_results):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            
        # Rank sparse
        for rank, (chunk_id, _) in enumerate(sparse_results):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            
        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

def calculate_metrics(retrieved_chunk_ids: List[str], ground_truth_doc_id: str, k: int = 10) -> Dict[str, float]:
    """
    Evaluates retrieval performance.
    A hit is defined as: retrieved_chunk_id.startswith(ground_truth_doc_id)
    """
    hits = [1 if cid.startswith(ground_truth_doc_id) else 0 for cid in retrieved_chunk_ids[:k]]
    
    # Precision@K
    precision = sum(hits) / k if k > 0 else 0.0
    
    # Recall@K (Assuming exactly 1 relevant document per query since the QA is generated per document)
    recall = 1.0 if sum(hits) > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for i, hit in enumerate(hits):
        if hit:
            mrr = 1.0 / (i + 1)
            break
            
    # NDCG@10
    dcg = 0.0
    for i, hit in enumerate(hits):
        if hit:
            dcg += 1.0 / math.log2(i + 2) # i+2 because rank starts at 1, so log2(rank+1)
            
    # Calculate exact IDCG based on the actual number of relevant chunks found
    num_hits = sum(hits)
    idcg = sum([1.0 / math.log2(i + 2) for i in range(num_hits)]) if num_hits > 0 else 1.0
    ndcg = dcg / idcg
    
    return {
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        "MRR": mrr,
        f"NDCG@{k}": ndcg
    }
