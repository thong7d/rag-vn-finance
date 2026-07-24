import argparse
import os
import re
import sqlite3
import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

def tokenize_vi(text: str) -> list[str]:
    """
    Giữ nguyên logic tokenizer cũ để tương thích ngược 100%.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

def build_sqlite_fts(parquet_path: str, db_path: str):
    """Đọc chunks.parquet, pre-tokenize và tạo SQLite FTS5 db."""
    print(f"[*] Đọc dữ liệu từ {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Xoá file db cũ nếu tồn tại
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng FTS5. chunk_id lưu trữ nhưng không đánh index FTS (UNINDEXED) để tiết kiệm dung lượng
    cursor.execute('''
        CREATE VIRTUAL TABLE chunks USING fts5(
            chunk_id UNINDEXED,
            text,
            tokenize="unicode61 remove_diacritics 0"
        )
    ''')
    
    print(f"[*] Bắt đầu pre-tokenize và insert {len(df)} chunks vào SQLite FTS5...")
    
    # Pre-tokenize để đảm bảo logic giống hệt rank-bm25 (bỏ từ 1 ký tự, bỏ dấu câu)
    # Bảng FTS5 sẽ index chuỗi pre-tokenized này.
    insert_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pre_tokenized = " ".join(tokenize_vi(row['text']))
        insert_data.append((row['chunk_id'], pre_tokenized))
        
    cursor.executemany("INSERT INTO chunks(chunk_id, text) VALUES (?, ?)", insert_data)
    
    conn.commit()
    
    # Optimize index
    print("[*] Tối ưu hoá FTS5 index (VACUUM & Optimize)...")
    cursor.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
    conn.commit()
    
    # VACUUM phải được chạy ngoài transaction
    conn.isolation_level = None
    cursor.execute("VACUUM")
    conn.isolation_level = ""
    
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"✅ Đã tạo thành công {db_path} (Kích thước: {db_size:.2f} MB)")
    conn.close()

def upload_to_hf(db_path: str, repo_id: str, strategy: str, token: str):
    """Upload file sqlite lên Hugging Face Hub (Private Dataset)."""
    print(f"[*] Đang chuẩn bị upload lên {repo_id}...")
    api = HfApi(token=token)
    
    try:
        api.dataset_info(repo_id)
    except Exception:
        print(f"[*] Repo {repo_id} chưa tồn tại. Đang tạo mới (Private)...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
        print(f"✅ Đã tạo repo {repo_id}")
        
    hf_path = f"{strategy}/bm25.db"
    print(f"[*] Đang upload {db_path} -> {hf_path} trên HF Hub...")
    
    api.upload_file(
        path_or_fileobj=db_path,
        path_in_repo=hf_path,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✅ Upload thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo SQLite FTS5 Database và upload lên HF Hub")
    parser.add_argument("--strategy", type=str, default="sentence_aware", help="Chunking strategy (vd: sentence_aware)")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (vd: thong7d/rag-vn-finance-bm25)")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face Write Token")
    
    args = parser.parse_args()
    
    # Đường dẫn cố định dựa trên cấu trúc thư mục
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parquet_path = os.path.join(base_dir, "pipeline", "data", "chunks", args.strategy, "chunks.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Không tìm thấy file {parquet_path}. Vui lòng chạy pipeline chunking trước.")
        exit(1)
        
    # Tạo thư mục output nếu chưa có
    output_dir = os.path.join(base_dir, "pipeline", "data", "bm25_sqlite", args.strategy)
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "bm25.db")
    
    build_sqlite_fts(parquet_path, db_path)
    upload_to_hf(db_path, args.repo_id, args.strategy, args.hf_token)
