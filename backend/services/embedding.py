"""
embedding.py — Hugging Face Spaces Gradio API embedding service.
Uses intfloat/multilingual-e5-large hosted on Gradio ZeroGPU Space (thong7d/e5-embedding-api).
"""

import json
import time
import numpy as np
from gradio_client import Client

from core.config import get_settings
from core.logging import setup_logger
from core.metrics import rag_embedding_latency_seconds

logger = setup_logger("EmbeddingService")


class OpenRouterEmbeddingAPI:
    """
    Embedding API service using Hugging Face Spaces Gradio client.
    Maintains class name 'OpenRouterEmbeddingAPI' for backwards compatibility with retrieval.py.
    """

    def __init__(self):
        settings = get_settings()
        self.space_id = getattr(settings, "hf_embedding_space", "thong7d/e5-embedding-api")
        self.hf_token = settings.hf_token if settings.hf_token else None
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            logger.info(f"Connecting to Gradio Space: {self.space_id}...")
            if self.hf_token:
                self._client = Client(self.space_id, hf_token=self.hf_token)
            else:
                self._client = Client(self.space_id)
            logger.info(f"Connected to Gradio Space: {self.space_id}")
        return self._client

    def encode(self, texts: list[str] | str, **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        t0 = time.time()
        client = self._get_client()

        embeddings = []
        for text in texts:
            # E5 requires "query: " prefix for retrieval queries
            prefixed = text if (text.startswith("query: ") or text.startswith("passage: ")) else f"query: {text}"

            try:
                raw_result = client.predict(
                    text=prefixed,
                    api_name="/encode",
                )
            except Exception as e:
                logger.warning(f"Gradio Space predict error ({e}), reconnecting client...")
                self._client = None
                client = self._get_client()
                raw_result = client.predict(
                    text=prefixed,
                    api_name="/encode",
                )

            # Handle if returned as stringified JSON or direct list
            if isinstance(raw_result, str):
                vec = json.loads(raw_result)
            else:
                vec = raw_result

            embeddings.append(vec)

        latency = time.time() - t0
        rag_embedding_latency_seconds.observe(latency)

        emb_array = np.array(embeddings, dtype=np.float32)

        # Ensure 2D shape (batch_size, 1024)
        if emb_array.ndim == 1:
            emb_array = np.expand_dims(emb_array, axis=0)
        elif emb_array.ndim == 3:
            emb_array = emb_array.mean(axis=1)

        logger.info(f"Embedding OK (Gradio Space) | {len(texts)} texts | {latency:.2f}s | shape={emb_array.shape}")
        return emb_array
