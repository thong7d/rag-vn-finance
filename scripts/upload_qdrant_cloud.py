"""
upload_qdrant_cloud.py — Upload local Qdrant snapshot to Qdrant Cloud.

Usage:
    python scripts/upload_qdrant_cloud.py \\
        --local-path implementation/qdrant_data/sentence_aware \\
        --strategy sentence_aware \\
        --cloud-url https://YOUR_CLUSTER.qdrant.io \\
        --api-key YOUR_QDRANT_API_KEY

Or set env vars QDRANT_URL and QDRANT_API_KEY, then:
    python scripts/upload_qdrant_cloud.py --strategy sentence_aware
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Upload local Qdrant data to Qdrant Cloud")
    parser.add_argument("--local-path", type=str, default=None,
                        help="Path to local Qdrant directory (e.g. implementation/qdrant_data/sentence_aware)")
    parser.add_argument("--strategy", type=str, default="sentence_aware",
                        choices=["fixed_size", "sentence_aware", "article_level"],
                        help="Chunking strategy (used to name the Cloud collection)")
    parser.add_argument("--cloud-url", type=str, default=None,
                        help="Qdrant Cloud cluster URL (overrides QDRANT_URL env var)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Qdrant Cloud API key (overrides QDRANT_API_KEY env var)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Upload batch size (default: 200 points per request)")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the collection if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve Cloud credentials ──────────────────────────────────────────────
    cloud_url = args.cloud_url or os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_CLOUD_URL")
    api_key = args.api_key or os.environ.get("QDRANT_API_KEY") or os.environ.get("QDRANT_CLOUD_API_KEY")

    if not cloud_url:
        print("❌ ERROR: Missing Qdrant Cloud URL. Set --cloud-url or QDRANT_URL env var.")
        sys.exit(1)
    if not api_key:
        print("❌ ERROR: Missing Qdrant API key. Set --api-key or QDRANT_API_KEY env var.")
        sys.exit(1)

    # ── Resolve local path ─────────────────────────────────────────────────────
    strategy = args.strategy
    if args.local_path:
        local_dir = Path(args.local_path)
    else:
        local_dir = ROOT / "implementation" / "qdrant_data" / strategy

    if not local_dir.exists():
        print(f"❌ ERROR: Local Qdrant directory not found: {local_dir}")
        print(f"   Try: python scripts/upload_qdrant_cloud.py --local-path <path> --strategy {strategy}")
        sys.exit(1)

    # ── Load metadata & vectors from indexes/ (same source as app.py) ─────────
    indexes_dir = ROOT / "implementation" / "indexes" / strategy
    if not indexes_dir.exists():
        print(f"❌ ERROR: Indexes directory not found: {indexes_dir}")
        sys.exit(1)

    print(f"[1/5] Loading metadata from {indexes_dir}...")
    with open(indexes_dir / "chunk_ids.json", "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    df_meta = pd.read_parquet(indexes_dir / "metadata.parquet")
    chunk_text_map = dict(zip(df_meta["chunk_id"], df_meta["text"]))
    chunk_title_map = dict(zip(df_meta["chunk_id"], df_meta.get("title", pd.Series(dtype=str))))
    chunk_url_map = dict(zip(
        df_meta["chunk_id"],
        df_meta.get("url", df_meta.get("link", pd.Series("", index=df_meta.index)))
    ))

    print(f"   → Loaded {len(chunk_ids)} chunks")

    # ── Load vectors from local Qdrant ─────────────────────────────────────────
    print(f"[2/5] Connecting to LOCAL Qdrant at {local_dir}...")
    local_client = QdrantClient(path=str(local_dir))
    collection_name_local = f"vn_finance_{strategy}"

    collections = [c.name for c in local_client.get_collections().collections]
    if collection_name_local not in collections:
        print(f"❌ ERROR: Collection '{collection_name_local}' not found in local Qdrant.")
        print(f"   Available: {collections}")
        sys.exit(1)

    local_info = local_client.get_collection(collection_name_local)
    total_points = local_info.points_count
    vector_size = local_info.config.params.vectors.size
    print(f"   → Collection: {collection_name_local} | {total_points} points | dim={vector_size}")

    # ── Connect Cloud ─────────────────────────────────────────────────────────
    print(f"[3/5] Connecting to Qdrant Cloud at {cloud_url}...")
    # Tăng timeout lên 120s để tránh lỗi httpx.WriteTimeout khi upload vectors
    cloud_client = QdrantClient(url=cloud_url, api_key=api_key, timeout=120.0)

    collection_name_cloud = f"vn_finance_{strategy}"
    cloud_collections = [c.name for c in cloud_client.get_collections().collections]

    if collection_name_cloud in cloud_collections:
        if args.recreate:
            print(f"   ⚠️  Deleting existing cloud collection '{collection_name_cloud}'...")
            cloud_client.delete_collection(collection_name_cloud)
        else:
            existing = cloud_client.get_collection(collection_name_cloud)
            existing_count = existing.points_count
            print(f"   ℹ️  Cloud collection already exists with {existing_count} points.")
            if existing_count >= total_points:
                print("   ✅ Upload already complete. Use --recreate to force re-upload.")
                sys.exit(0)
            else:
                print(f"   → Partial upload detected ({existing_count}/{total_points}). Will upload missing points.")

    if collection_name_cloud not in [c.name for c in cloud_client.get_collections().collections]:
        print(f"   → Creating cloud collection '{collection_name_cloud}'...")
        cloud_client.create_collection(
            collection_name=collection_name_cloud,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # ── Upload in batches ──────────────────────────────────────────────────────
    print(f"[4/5] Uploading {total_points} points in batches of {args.batch_size}...")

    offset = None
    uploaded = 0
    pbar = tqdm(total=total_points, unit="pts")

    while True:
        result = local_client.scroll(
            collection_name=collection_name_local,
            limit=args.batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        points_batch, next_offset = result

        if not points_batch:
            break

        cloud_points = []
        for p in points_batch:
            chunk_id = p.payload.get("chunk_id", "")
            payload = {
                "chunk_id": chunk_id,
                "text": chunk_text_map.get(chunk_id, p.payload.get("text", "")),
                "title": chunk_title_map.get(chunk_id, p.payload.get("title", "")),
                "url": chunk_url_map.get(chunk_id, p.payload.get("url", "")),
            }
            cloud_points.append(PointStruct(
                id=p.id,
                vector=p.vector,
                payload=payload
            ))

        cloud_client.upsert(
            collection_name=collection_name_cloud,
            points=cloud_points
        )
        uploaded += len(cloud_points)
        pbar.update(len(cloud_points))

        if next_offset is None:
            break
        offset = next_offset

    pbar.close()

    # ── Verify ─────────────────────────────────────────────────────────────────
    print(f"[5/5] Verifying upload...")
    cloud_info = cloud_client.get_collection(collection_name_cloud)
    cloud_count = cloud_info.points_count
    print(f"\n✅ Upload complete!")
    print(f"   Collection: {collection_name_cloud}")
    print(f"   Local points: {total_points} | Cloud points: {cloud_count}")
    print(f"\nNext steps:")
    print(f"  1. Add to backend/.env:")
    print(f"     QDRANT_URL={cloud_url}")
    print(f"     QDRANT_API_KEY=<your_api_key>")
    print(f"  2. Proceed to BM25 upload (see instructions)")

    local_client.close()


if __name__ == "__main__":
    main()
