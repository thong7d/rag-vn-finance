"""
upload_bm25_hub.py — Upload BM25 index to HuggingFace Hub (private dataset).

Usage:
    python scripts/upload_bm25_hub.py \\
        --strategy sentence_aware \\
        --repo-id thong7d/rag-vn-finance-bm25 \\
        --hf-token hf_xxxx

Or set HF_TOKEN env var, then:
    python scripts/upload_bm25_hub.py --strategy sentence_aware
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Upload BM25 index to HuggingFace Hub")
    parser.add_argument("--strategy", type=str, default="sentence_aware",
                        choices=["fixed_size", "sentence_aware", "article_level"])
    parser.add_argument("--repo-id", type=str, default="thong7d/rag-vn-finance-bm25",
                        help="HuggingFace dataset repo ID (will be created if not exists)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace write token (overrides HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Create as private dataset (default: True)")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ ERROR: Missing HF_TOKEN. Set --hf-token or HF_TOKEN env var.")
        sys.exit(1)

    strategy = args.strategy
    bm25_dir = ROOT / "implementation" / "bm25" / strategy

    pkl_file = bm25_dir / "bm25_index.pkl"
    chunk_ids_file = bm25_dir / "chunk_ids.json"

    if not pkl_file.exists():
        print(f"❌ ERROR: BM25 index not found: {pkl_file}")
        sys.exit(1)
    if not chunk_ids_file.exists():
        print(f"❌ ERROR: chunk_ids.json not found: {chunk_ids_file}")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)

    # Create or ensure repo exists
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            token=hf_token,
            exist_ok=True,
        )
        print(f"✅ Repo ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"⚠️  Repo creation warning (may already exist): {e}")

    # Upload files
    for local_file, hub_path in [
        (pkl_file, f"{strategy}/bm25_index.pkl"),
        (chunk_ids_file, f"{strategy}/chunk_ids.json"),
    ]:
        print(f"Uploading {local_file.name} → {hub_path} ...")
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=hub_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        print(f"  ✅ Uploaded {local_file.name} ({local_file.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n✅ BM25 upload complete!")
    print(f"   Repo: https://huggingface.co/datasets/{args.repo_id}")
    print(f"   Files: {strategy}/bm25_index.pkl | {strategy}/chunk_ids.json")
    print(f"\nBackend will download on startup with:")
    print(f"   HF_BM25_REPO={args.repo_id}")
    print(f"   CHUNK_STRATEGY={strategy}")


if __name__ == "__main__":
    main()
