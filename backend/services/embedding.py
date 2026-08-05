"""
embedding.py — Hugging Face Inference API embedding service.
Uses intfloat/multilingual-e5-large directly via HuggingFace for fast response (<1s).
"""

import time
import numpy as np
import requests

from core.config import get_settings
from core.logging import setup_logger
from core.metrics import rag_embedding_latency_seconds

logger = setup_logger("EmbeddingService")

class OpenRouterEmbeddingAPI:
    """
    Embedding API service using Hugging Face Inference API.
    Maintains class name 'OpenRouterEmbeddingAPI' for backwards compatibility with retrieval.py.
    """

    MODEL_NAME = "intfloat/multilingual-e5-large"
    API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.hf_token

    def encode(self, texts: list[str] | str, **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # E5 requires "query: " prefix for retrieval queries
        prefixed = [t if t.startswith("query: ") else f"query: {t}" for t in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prefixed,
            "options": {"wait_for_model": True}
        }

        t0 = time.time()
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        latency = time.time() - t0

        rag_embedding_latency_seconds.observe(latency)

        if resp.status_code == 200:
            raw_data = resp.json()
            emb_array = np.array(raw_data, dtype=np.float32)

            # If 3D (batch, seq_len, 1024), perform mean pooling over token embeddings
            if emb_array.ndim == 3:
                emb_array = emb_array.mean(axis=1)
            elif emb_array.ndim == 1:
                emb_array = np.expand_dims(emb_array, axis=0)

            logger.info(f"Embedding OK (HF) | {len(texts)} texts | {latency:.2f}s | shape={emb_array.shape}")
            return emb_array
        else:
            logger.error(f"Hugging Face embedding error: {resp.status_code} — {resp.text}")
            raise RuntimeError(f"HF Inference API error {resp.status_code}: {resp.text}")

