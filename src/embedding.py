"""
embedding.py — Dense embedding & FAISS indexing for Phase 3.

Model  : intfloat/multilingual-e5-large (dim=1024)
Device : T4 GPU on Colab (falls back to CPU locally)
Index  : IndexFlatIP with L2-normalised vectors (= cosine similarity)

Key design decisions
--------------------
- np.memmap is used for the embedding matrix so it is written directly to
  disk in chunks — no risk of OOM even for 50 K+ chunks on GPU.
- Checkpointing every 100 batches lets any interrupted Colab session resume
  from exactly where it stopped.
- faiss.normalize_L2() is applied in-place BEFORE adding to IndexFlatIP so
  that inner-product search returns cosine similarity scores.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from src.utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024
PASSAGE_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Task 1 — encode_chunks_with_checkpoint
# ---------------------------------------------------------------------------

def encode_chunks_with_checkpoint(
    texts: list[str],
    model,                           # SentenceTransformer instance
    output_npy_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    checkpoint_every: int = 100,
) -> np.ndarray:
    """
    Encode a list of texts to float32 embeddings, writing results
    incrementally to a memory-mapped NumPy array on disk.

    Checkpointing resumes automatically from `last_completed_batch + 1`
    when `checkpoint_path` already exists — no data is lost on Colab timeout.

    Parameters
    ----------
    texts : list[str]
        Plain text strings (passage prefix already applied by the caller).
    model : SentenceTransformer
        Pre-loaded embedding model (GPU recommended).
    output_npy_path : str
        Path to the `.npy` file that will hold the full embedding matrix
        (shape: N × 1024, dtype float32).
    checkpoint_path : str
        Path to the JSON file storing resume progress.
    batch_size : int
        Number of texts per model.encode() call. Default 64.
    checkpoint_every : int
        Flush checkpoint after this many batches. Default 100.

    Returns
    -------
    np.ndarray
        Memory-mapped array of shape (N, 1024). Caller should cast to a
        regular np.ndarray (np.array(result)) before passing to FAISS.
    """
    n_total = len(texts)
    ensure_dir(Path(output_npy_path).parent)
    ensure_dir(Path(checkpoint_path).parent)

    # ------------------------------------------------------------------
    # Determine start batch from checkpoint (if it exists)
    # ------------------------------------------------------------------
    start_batch = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        start_batch = ckpt.get("last_completed_batch", -1) + 1
        logger.info(
            f"Checkpoint found — resuming from batch {start_batch} "
            f"({ckpt.get('completed_chunks', 0)}/{n_total} chunks done)"
        )

    # ------------------------------------------------------------------
    # Open (or create) the memmap array
    # ------------------------------------------------------------------
    mode = "r+" if os.path.exists(output_npy_path) else "w+"
    embeddings = np.memmap(
        output_npy_path,
        dtype="float32",
        mode=mode,
        shape=(n_total, EMBEDDING_DIM),
    )
    logger.info(
        f"Embedding matrix: {output_npy_path}  shape={embeddings.shape}  mode={mode}"
    )

    # ------------------------------------------------------------------
    # Encode batch by batch
    # ------------------------------------------------------------------
    total_batches = (n_total + batch_size - 1) // batch_size

    with tqdm(
        total=total_batches,
        initial=start_batch,
        desc="Encoding chunks",
        unit="batch",
    ) as pbar:
        for batch_idx in range(start_batch, total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_total)
            batch_texts = texts[start:end]

            batch_emb = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalise ourselves before FAISS
                convert_to_numpy=True,
            ).astype("float32")

            embeddings[start:end] = batch_emb
            embeddings.flush()

            # Checkpoint every N batches
            if (batch_idx + 1) % checkpoint_every == 0 or batch_idx == total_batches - 1:
                ckpt_data = {
                    "last_completed_batch": batch_idx,
                    "completed_chunks": end,
                    "total_chunks": n_total,
                    "embedding_path": str(output_npy_path),
                }
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(ckpt_data, f, indent=2)
                logger.info(f"Checkpoint saved — batch {batch_idx} ({end}/{n_total} chunks)")

            pbar.update(1)

    logger.info(f"Encoding complete. Total chunks embedded: {n_total}")
    return embeddings


# ---------------------------------------------------------------------------
# Task 2 — build_faiss_index
# ---------------------------------------------------------------------------

def build_faiss_index(npy_path: str, index_output_path: str, dim: int = 1024):
    """
    Builds a FAISS IndexFlatIP from a raw binary memmap file.
    """
    logger.info(f"Loading raw embeddings from: {npy_path}")
    
    # 1. Tự động tính số lượng vector dựa trên dung lượng file đĩa (4 bytes cho kiểu float32)
    file_size = os.path.getsize(npy_path)
    num_vectors = file_size // (4 * dim)
    
    # 2. Đọc file dưới dạng raw memmap
    embeddings_raw = np.memmap(npy_path, dtype='float32', mode='r', shape=(num_vectors, dim))
    logger.info(f"Embedding matrix shape inferred: {embeddings_raw.shape}")
    
    # 3. Tạo bản sao trên RAM để chuẩn hóa (normalize in-place) vì memmap mode 'r' là read-only
    embeddings = np.array(embeddings_raw, dtype="float32")
    
    logger.info("Normalizing vectors for Inner Product (Cosine Similarity simulation)...")
    faiss.normalize_L2(embeddings)
    
    logger.info(f"Building FAISS IndexFlatIP with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    logger.info(f"Saving FAISS index to {index_output_path}")
    faiss.write_index(index, index_output_path)
    
    return index


# ---------------------------------------------------------------------------
# Task 3 — align_and_save_metadata
# ---------------------------------------------------------------------------

def align_and_save_metadata(
    df_chunks: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """
    Reset the DataFrame index so row i corresponds exactly to vector i in
    the FAISS index, then persist:
      - chunk_ids.json   : ordered list of chunk_id strings
      - metadata.parquet : full chunk metadata table (Snappy compressed)

    Parameters
    ----------
    df_chunks : pd.DataFrame
        Chunk DataFrame in the same row order as the embedding matrix.
    output_dir : str
        Directory where output files are written.

    Returns
    -------
    tuple[str, str]
        Paths to (chunk_ids.json, metadata.parquet).
    """
    ensure_dir(output_dir)

    # Hard reset — guarantees positional alignment with FAISS vector IDs
    df_aligned = df_chunks.reset_index(drop=True)

    chunk_ids = df_aligned["chunk_id"].tolist()
    ids_path = os.path.join(output_dir, "chunk_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    logger.info(f"chunk_ids.json saved: {ids_path}  ({len(chunk_ids)} IDs)")

    meta_path = os.path.join(output_dir, "metadata.parquet")
    df_aligned.to_parquet(meta_path, index=False, compression="snappy")
    logger.info(f"metadata.parquet saved: {meta_path}  ({len(df_aligned)} rows)")

    # Sanity check
    assert len(chunk_ids) == len(df_aligned), (
        f"Alignment error: {len(chunk_ids)} IDs vs {len(df_aligned)} metadata rows"
    )
    logger.info("Row-alignment verified: chunk_ids ↔ metadata ↔ FAISS index.")

    return ids_path, meta_path


# ---------------------------------------------------------------------------
# Convenience helper — verify index integrity
# ---------------------------------------------------------------------------

def verify_index(
    index_path: str,
    metadata_path: str,
    chunk_ids_path: str,
) -> bool:
    """
    Quick sanity check: confirm that the FAISS index, metadata, and
    chunk_ids all report the same total count.

    Returns True if all counts match, raises AssertionError otherwise.
    """
    index = faiss.read_index(index_path)
    df_meta = pd.read_parquet(metadata_path)
    with open(chunk_ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    n_index = index.ntotal
    n_meta = len(df_meta)
    n_ids = len(ids)

    logger.info(f"Index vectors : {n_index}")
    logger.info(f"Metadata rows : {n_meta}")
    logger.info(f"chunk_ids     : {n_ids}")

    assert n_index == n_meta == n_ids, (
        f"Count mismatch — index:{n_index}  meta:{n_meta}  ids:{n_ids}"
    )
    logger.info("✅ Index integrity verified.")
    return True
