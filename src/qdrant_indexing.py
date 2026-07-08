"""
qdrant_indexing.py — Offline indexing script for Qdrant.
Builds a Qdrant collection per chunking strategy.
"""

import argparse
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from dotenv import load_dotenv

# Add implementation/ directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, resolve_path, setup_logger, ensure_dir, get_env

load_dotenv()

logger = setup_logger("QdrantIndexing")

def main():
    parser = argparse.ArgumentParser(description="Build Qdrant index offline from existing embeddings/indexes.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_size",
        choices=["fixed_size", "sentence_aware", "article_level"],
        help="Chunking strategy to process (default: fixed_size)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for upserting points to Qdrant (default: 500)"
    )
    args = parser.parse_args()
    
    strategy = args.strategy
    batch_size = args.batch_size
    
    logger.info(f"Bắt đầu xây dựng chỉ mục Qdrant cho chiến lược: {strategy}")
    
    # Load config
    config = load_config()
    
    # 1. Resolve paths
    indexes_dir = Path(resolve_path(config["indexing"], "output_dir"))
    strategy_dir = indexes_dir / strategy
    
    index_path = strategy_dir / "index.faiss"
    chunk_ids_path = strategy_dir / "chunk_ids.json"
    metadata_path = strategy_dir / "metadata.parquet"
    
    # Check if necessary files exist
    if not index_path.exists():
        logger.error(f"Không tìm thấy tệp chỉ mục FAISS tại: {index_path}. Vui lòng chạy embedding & indexing cho FAISS trước.")
        return
        
    if not chunk_ids_path.exists():
        logger.error(f"Không tìm thấy chunk_ids.json tại: {chunk_ids_path}")
        return
        
    if not metadata_path.exists():
        logger.error(f"Không tìm thấy metadata.parquet tại: {metadata_path}")
        return
        
    # 2. Load chunk IDs and metadata
    logger.info(f"Đang đọc chunk_ids.json từ {chunk_ids_path}...")
    with open(chunk_ids_path, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
        
    logger.info(f"Đang đọc metadata.parquet từ {metadata_path}...")
    df_meta = pd.read_parquet(metadata_path)
    
    n_chunks = len(chunk_ids)
    logger.info(f"Tổng số lượng chunks cần xử lý: {n_chunks}")
    
    if len(df_meta) != n_chunks:
        logger.error(f"Kích thước không khớp: chunk_ids ({n_chunks}) vs metadata rows ({len(df_meta)})")
        return

    # 3. Load or reconstruct embeddings
    # We check if raw embeddings exist in the output_dir. Otherwise, we reconstruct from FAISS index.
    emb_dir = Path(resolve_path(config["embedding"], "output_dir")) / strategy
    npy_path = emb_dir / "embeddings.npy"
    
    embeddings = None
    if npy_path.exists():
        logger.info(f"Tìm thấy tệp embeddings.npy tại: {npy_path}. Đang tải...")
        try:
            # Load as read-only memmap to save memory
            embeddings = np.memmap(npy_path, dtype="float32", mode="r", shape=(n_chunks, 1024))
            logger.info("Đã tải embeddings.npy qua memmap thành công.")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc embeddings.npy qua memmap: {e}. Sẽ fallback sang reconstruct từ FAISS index.")
            embeddings = None
            
    if embeddings is None:
        logger.info(f"Không có embeddings.npy hoặc lỗi đọc. Tiến hành khôi phục vector từ FAISS index tại: {index_path}...")
        try:
            faiss_index = faiss.read_index(str(index_path))
            if faiss_index.ntotal != n_chunks:
                logger.error(f"Kích thước FAISS index ({faiss_index.ntotal}) không khớp với chunk_ids ({n_chunks})!")
                return
            
            # We will reconstruct in batches inside the upload loop to keep RAM usage low.
            logger.info("Đã tải FAISS index và sẵn sàng khôi phục vector theo batch.")
        except Exception as e:
            logger.error(f"Không thể đọc FAISS index để khôi phục vectors: {e}")
            return
    
    # 4. Initialize Qdrant Client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    qdrant_url = get_env("QDRANT_URL")
    qdrant_api_key = get_env("QDRANT_API_KEY")
    
    collection_name = f"{config['vector_store']['qdrant']['collection_name']}_{strategy}"
    
    if qdrant_url:
        logger.info(f"Đang kết nối tới remote Qdrant Cloud: {qdrant_url}")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        local_path = config["vector_store"]["qdrant"]["local_path"]
        # Store in root implementation folder
        project_root = Path(__file__).resolve().parent.parent
        qdrant_dir = project_root / local_path / strategy
        logger.info(f"Đang kết nối tới local Qdrant tại: {qdrant_dir}")
        ensure_dir(qdrant_dir)
        client = QdrantClient(path=str(qdrant_dir))
        
    # 5. Create collection
    logger.info(f"Đang kiểm tra và tạo collection '{collection_name}'...")
    try:
        # Recreate collection to ensure a clean start
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        logger.info(f"Tạo thành công collection '{collection_name}' với metric Distance.COSINE.")
    except Exception as e:
        logger.error(f"Không thể tạo collection '{collection_name}': {e}")
        return

    # 6. Upsert points in batches
    logger.info(f"Bắt đầu upsert dữ liệu lên Qdrant theo batch (size={batch_size})...")
    
    total_batches = (n_chunks + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Upserting to Qdrant"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_chunks)
        
        # 6.1 Get vectors for this batch
        if embeddings is not None:
            # From memmap
            batch_vectors = np.array(embeddings[start:end], dtype="float32")
        else:
            # Reconstruct from FAISS index (already normalized, which is fine for COSINE)
            batch_vectors = faiss_index.reconstruct_n(start, end - start)
            
        # 6.2 Build points list
        points = []
        for i in range(start, end):
            vec_idx = i - start
            vector = batch_vectors[vec_idx].tolist()
            
            # Get metadata row safely
            row = df_meta.iloc[i]
            
            title = row.get("title", "")
            if pd.isna(title):
                title = ""
                
            url = row.get("url", row.get("link", ""))
            if pd.isna(url):
                url = ""
                
            payload = {
                "chunk_id": str(row.get("chunk_id", "")),
                "text": str(row.get("text", "")),
                "title": str(title),
                "url": str(url),
                "source": str(row.get("source", "")),
                "category": str(row.get("category", "")),
                "doc_id": str(row.get("doc_id", ""))
            }
            
            # Qdrant accepts integer or UUID for point ID
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload=payload
                )
            )
            
        # 6.3 Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    logger.info(f"✅ Đã tải thành công {n_chunks} điểm dữ liệu lên collection '{collection_name}'!")
    
    # 7. Verification print
    try:
        coll_info = client.get_collection(collection_name)
        logger.info(f"Xác nhận collection status: {coll_info.status}")
        logger.info(f"Tổng số vectors lưu trong Qdrant: {coll_info.vectors_count}")
    except Exception as e:
        logger.warning(f"Không thể truy vấn collection info: {e}")

if __name__ == "__main__":
    main()
