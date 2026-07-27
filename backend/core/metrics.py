"""
metrics.py — Custom Prometheus metrics for RAG operations.
"""

from prometheus_client import Counter, Histogram

# Total RAG requests
rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG API requests",
    ["endpoint", "status"]
)

# Latency histograms
rag_embedding_latency_seconds = Histogram(
    "rag_embedding_latency_seconds",
    "Time to encode query embedding"
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time per retrieval step",
    ["type"]  # dense, sparse, rerank
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Time for LLM generation",
    ["model"]
)

# Fallbacks and Decomposition counters
rag_fallback_total = Counter(
    "rag_fallback_total",
    "Number of fallback events",
    ["from_model", "to_model"]
)

rag_decomposition_total = Counter(
    "rag_decomposition_total",
    "Number of decomposition usage",
    ["sub_query_count"]
)
