"""
embedding.py — OpenRouter Embedding API service.
Ported from implementation/app.py::OpenRouterEmbeddingAPI.
"""

import time
import numpy as np
import requests

from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("EmbeddingService")


class OpenRouterEmbeddingAPI:
    """Call OpenRouter's /embeddings endpoint with intfloat/multilingual-e5-large."""

    MODEL_NAME = "intfloat/multilingual-e5-large"
    API_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.openrouter_api_key

    def encode(self, texts: list[str] | str, **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # E5 requires "query: " prefix for retrieval queries
        prefixed = [t if t.startswith("query: ") else f"query: {t}" for t in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.MODEL_NAME, "input": prefixed}

        t0 = time.time()
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=15)
        latency = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()["data"]
            embeddings = [item["embedding"] for item in data]
            emb_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Embedding OK | {len(texts)} texts | {latency:.2f}s | shape={emb_array.shape}")
            return emb_array
        else:
            logger.error(f"OpenRouter embedding error: {resp.status_code} — {resp.text}")
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
