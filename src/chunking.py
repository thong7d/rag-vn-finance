"""
chunking.py — Three chunking strategies for Phase 2.

Strategies: fixed_size, sentence_aware, article_level.
Tokenizer: intfloat/multilingual-e5-large (same as embedding model).

Each chunk inherits full article metadata for downstream filtering.
"""

import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

# Metadata columns that MUST be propagated from article to every chunk
METADATA_COLS = [
    "source", "category", "time", "year", "title", "url",
    "tickers", "is_historical", "numerical_density", "entities",
]


# ---------------------------------------------------------------------------
# Tokenizer singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer(model_name: str = "intfloat/multilingual-e5-large"):
    """Return a cached tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def tokenize(text: str, model_name: str = "intfloat/multilingual-e5-large") -> list[int]:
    """Tokenize text and return token IDs."""
    tok = get_tokenizer(model_name)
    return tok.encode(text, add_special_tokens=False)


def decode(token_ids: list[int], model_name: str = "intfloat/multilingual-e5-large") -> str:
    """Decode token IDs back to text."""
    tok = get_tokenizer(model_name)
    return tok.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Metadata extraction helper
# ---------------------------------------------------------------------------

def _extract_metadata(row: pd.Series) -> dict:
    """Extract metadata dict from a DataFrame row."""
    meta = {}
    for col in METADATA_COLS:
        val = row.get(col, "")
        # Convert Timestamp to ISO string for parquet compatibility
        if hasattr(val, "isoformat"):
            val = val.isoformat()
        meta[col] = val
    return meta


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking
# ---------------------------------------------------------------------------

def chunk_fixed_size(
    df: pd.DataFrame,
    chunk_size: int = 256,
    overlap: int = 32,
    model_name: str = "intfloat/multilingual-e5-large",
) -> pd.DataFrame:
    """
    Fixed-size token-based chunking with sliding window.

    Each chunk text is prefixed with "Title: {title}\n".
    """
    all_chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="fixed_size chunking"):
        doc_id = row["doc_id"]
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        meta = _extract_metadata(row)

        token_ids = tokenize(content, model_name)
        step = max(chunk_size - overlap, 1)

        if len(token_ids) == 0:
            continue

        chunks_for_doc = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_tokens = token_ids[start:end]
            chunk_text = decode(chunk_tokens, model_name)
            # Prepend title
            chunk_text = f"Title: {title}\n{chunk_text}"
            chunks_for_doc.append(chunk_text)
            if end >= len(token_ids):
                break
            start += step

        total = len(chunks_for_doc)
        for idx, text in enumerate(chunks_for_doc):
            all_chunks.append({
                "chunk_id": f"{doc_id}_c{idx:04d}",
                "doc_id": doc_id,
                "text": text,
                "chunk_index": idx,
                "total_chunks": total,
                "strategy": "fixed_size",
                **meta,
            })

    return pd.DataFrame(all_chunks)


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-aware chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using sentence-splitter with Vietnamese fallback."""
    try:
        from sentence_splitter import SentenceSplitter
        splitter = SentenceSplitter(language="vi")
        sentences = splitter.split(text)
    except Exception:
        # Fallback: split on Vietnamese sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence_aware(
    df: pd.DataFrame,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
    max_tokens_per_sentence: int = 256,
    absolute_max_tokens: int = 500, # Ngưỡng an toàn tuyệt đối cho e5-large (max 512)
    model_name: str = "intfloat/multilingual-e5-large",
) -> pd.DataFrame:
    """
    Sentence-aware chunking.

    Groups up to max_sentences per chunk with overlap_sentences overlap.
    STRICT CONSTRAINT: A combined chunk must not exceed absolute_max_tokens.
    If a single sentence exceeds max_tokens_per_sentence, it is split
    by token into sub-chunks (with 32-token overlap) as a fallback.
    """
    all_chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="sentence_aware chunking"):
        doc_id = row["doc_id"]
        content = str(row.get("content", ""))
        meta = _extract_metadata(row)

        sentences = _split_sentences(content)
        if not sentences:
            continue

        # Bước 1: Tách các câu quá dài và theo dõi độ dài token của từng câu
        expanded_with_lengths = []
        for sent in sentences:
            tids = tokenize(sent, model_name)
            tok_len = len(tids)
            
            if tok_len <= max_tokens_per_sentence:
                expanded_with_lengths.append((sent, tok_len))
            else:
                # Token-level fallback split
                step_tok = max(max_tokens_per_sentence - 32, 1)
                start_tok = 0
                while start_tok < tok_len:
                    end_tok = min(start_tok + max_tokens_per_sentence, tok_len)
                    sub_sent = decode(tids[start_tok:end_tok], model_name)
                    expanded_with_lengths.append((sub_sent, end_tok - start_tok))
                    if end_tok >= tok_len:
                        break
                    start_tok += step_tok

        # Bước 2: Gộp câu với ràng buộc khắt khe về tổng số Token
        chunks_for_doc = []
        start = 0
        while start < len(expanded_with_lengths):
            current_chunk = []
            current_length = 0
            end = start

            while end < len(expanded_with_lengths):
                sent, length = expanded_with_lengths[end]
                
                # Ngắt nếu đã đủ số câu tối đa (5)
                if len(current_chunk) == max_sentences:
                    break
                
                # Ngắt nếu thêm câu này sẽ làm nổ token limit (>500)
                # (Chỉ ngắt nếu chunk đã có ít nhất 1 câu)
                if current_chunk and (current_length + length > absolute_max_tokens):
                    break
                    
                current_chunk.append(sent)
                current_length += length
                end += 1

            chunk_text = " ".join(current_chunk)
            chunks_for_doc.append(chunk_text)

            # Tịnh tiến con trỏ start, tính toán độ chồng chéo (overlap)
            step = max(len(current_chunk) - overlap_sentences, 1)
            start += step

        # Bước 3: Format output
        total = len(chunks_for_doc)
        for idx, text in enumerate(chunks_for_doc):
            all_chunks.append({
                "chunk_id": f"{doc_id}_c{idx:04d}",
                "doc_id": doc_id,
                "text": text,
                "chunk_index": idx,
                "total_chunks": total,
                "strategy": "sentence_aware",
                **meta,
            })

    return pd.DataFrame(all_chunks)

# ---------------------------------------------------------------------------
# Strategy 3: Article-level chunking
# ---------------------------------------------------------------------------

def chunk_article_level(
    df: pd.DataFrame,
    max_tokens: int = 512,
    model_name: str = "intfloat/multilingual-e5-large",
) -> pd.DataFrame:
    """
    Article-level chunking: one chunk per article, truncated at max_tokens.
    """
    all_chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="article_level chunking"):
        doc_id = row["doc_id"]
        content = str(row.get("content", ""))
        meta = _extract_metadata(row)

        token_ids = tokenize(content, model_name)
        if len(token_ids) == 0:
            continue

        # Truncate to max_tokens
        truncated_ids = token_ids[:max_tokens]
        chunk_text = decode(truncated_ids, model_name)

        all_chunks.append({
            "chunk_id": f"{doc_id}_c0000",
            "doc_id": doc_id,
            "text": chunk_text,
            "chunk_index": 0,
            "total_chunks": 1,
            "strategy": "article_level",
            **meta,
        })

    return pd.DataFrame(all_chunks)


# ---------------------------------------------------------------------------
# Chunk statistics
# ---------------------------------------------------------------------------

def generate_chunk_stats(
    df_chunks: pd.DataFrame,
    strategy: str,
    model_name: str = "intfloat/multilingual-e5-large",
) -> dict:
    """
    Compute chunk statistics for a given strategy.

    Returns a dict suitable for saving as stats.json.
    """
    # Calculate token counts for each chunk
    logger.info(f"Computing token stats for {strategy}...")
    token_counts = []
    for text in tqdm(df_chunks["text"], desc=f"tokenizing {strategy} stats"):
        tids = tokenize(str(text), model_name)
        token_counts.append(len(tids))

    token_counts = np.array(token_counts)
    chunks_per_article = df_chunks.groupby("doc_id")["chunk_id"].count()

    # Determine truncation based on strategy
    if strategy == "fixed_size":
        max_allowed = 256
    elif strategy == "article_level":
        max_allowed = 512
    else:
        max_allowed = None

    truncated = 0
    if max_allowed is not None:
        truncated = int((token_counts >= max_allowed).sum())

    stats = {
        "strategy": strategy,
        "total_chunks": len(df_chunks),
        "total_articles": int(df_chunks["doc_id"].nunique()),
        "avg_tokens_per_chunk": round(float(token_counts.mean()), 1),
        "max_tokens_per_chunk": int(token_counts.max()),
        "min_tokens_per_chunk": int(token_counts.min()),
        "truncated_chunks": truncated,
        "chunks_per_article": {
            "mean": round(float(chunks_per_article.mean()), 1),
            "median": float(chunks_per_article.median()),
            "max": int(chunks_per_article.max()),
        },
    }

    logger.info(f"Stats for {strategy}: {stats['total_chunks']} chunks, "
                f"avg {stats['avg_tokens_per_chunk']} tokens")
    return stats


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_chunks(df_chunks: pd.DataFrame, output_dir: str) -> str:
    """Save chunks parquet (idempotent). Returns the output path."""
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "chunks.parquet")
    if not os.path.exists(output_path):
        df_chunks.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"Saved {len(df_chunks)} chunks -> {output_path}")
    else:
        logger.info(f"Cached chunks found: {output_path}")
    return output_path


def save_stats(stats: dict, output_dir: str) -> str:
    """Save stats JSON. Always overwrites."""
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Stats saved: {output_path}")
    return output_path
