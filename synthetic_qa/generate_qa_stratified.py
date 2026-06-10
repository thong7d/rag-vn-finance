"""
generate_qa_stratified.py — Phase 5: Stratified Sampling & Synthetic QA Generation
====================================================================================

Chiến lược lấy mẫu theo tầng (Stratified Sampling Strategy):
  - Tiêu chí phân tầng: Cột 'source' (Nguồn báo mạng/Domain) trong metadata.
    Nếu cột 'source' không tồn tại, tự động fallback sang cột 'ticker' hoặc 'category'.
  - Tỷ lệ phân bổ: Proportional Stratified Sampling (tỷ lệ mẫu ~ kích thước tầng).
  - Hạt giống ngẫu nhiên cố định: random_state=42 đảm bảo tính tái lập (Reproducibility).
  - Tổng mẫu mục tiêu: 500 bài báo (qa_per_article=2 → 1000 cặp QA).

Đầu ra:
  - synthetic_qa/stratified_sampling_log.json : Nhật ký phân tầng chi tiết.
  - synthetic_qa/qa_dataset_stratified.jsonl  : Tập QA định dạng chuẩn Ragas.

Chạy script này offline tại thư mục implementation/:
    python synthetic_qa/generate_qa_stratified.py
"""

import os
import sys
import json
import math
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Thêm thư mục gốc (implementation/) vào PYTHONPATH để import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, load_config, get_env
from src.generation import generate_synthetic_qa_batch

logger = setup_logger("Phase5_Stratified_QA")
config = load_config()

# ---------------------------------------------------------------------------
# Cấu hình chiến lược lấy mẫu
# ---------------------------------------------------------------------------

RANDOM_STATE    = 42          # Hạt giống cố định — bắt buộc để tái lập kết quả
TOTAL_SAMPLES   = 500         # Tổng số bài báo cần lấy mẫu
QA_PER_ARTICLE  = 2           # Số cặp QA sinh ra trên mỗi bài báo

# CẤU HÌNH ĐA CẤP:
PRIMARY_STRATA_COLS = ["year", "source"]
SECONDARY_STRATA_COL = "category"

OUTPUT_DIR = Path(config["synthetic_qa"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH    = OUTPUT_DIR / "stratified_sampling_log.json"
QA_OUT_PATH = OUTPUT_DIR / "qa_dataset_stratified.jsonl"

# Tệp dữ liệu chunk đầu vào (dùng fixed_size strategy của Phase 3)
METADATA_PATH = Path(config["indexing"]["output_dir"]) / "fixed_size" / "metadata.parquet"


# ---------------------------------------------------------------------------
# Bước 1 — Tải dữ liệu chunk và xác định cột phân tầng
# ---------------------------------------------------------------------------

def verify_strata_columns(df: pd.DataFrame):
    """Xác thực sự tồn tại của tất cả các cột phục vụ phân tầng đa cấp."""
    required_cols = PRIMARY_STRATA_COLS + [SECONDARY_STRATA_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Mô hình dữ liệu thiếu các cột bắt buộc sau để phân tầng: {missing_cols}")
    logger.info(f"Xác thực cấu trúc phân tầng đa cấp thành công: {PRIMARY_STRATA_COLS} -> {SECONDARY_STRATA_COL}")


def load_metadata() -> pd.DataFrame:
    """Tải tệp metadata.parquet và trả về DataFrame."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata.parquet tại: {METADATA_PATH}\n"
            f"Hãy chạy Phase 3 (Embedding & Indexing) trước."
        )
    df = pd.read_parquet(METADATA_PATH)
    logger.info(f"Đã tải metadata: {len(df)} chunks từ {METADATA_PATH}")
    return df


# ---------------------------------------------------------------------------
# Bước 2 — Proportional Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, primary_cols: list, secondary_col: str, total_samples: int, random_state: int) -> tuple:
    """
    Thực hiện phân tầng đa cấp tỷ lệ thuận (Multi-level Proportional Stratified Sampling).
    """
    df_docs = df.drop_duplicates(subset="doc_id").copy()
    
    # Xử lý chuẩn hóa dữ liệu khuyết thiếu đề phòng lỗi kiểu dữ liệu
    df_docs[primary_cols[0]] = df_docs[primary_cols[0]].fillna("Unknown").astype(str)
    df_docs[primary_cols[1]] = df_docs[primary_cols[1]].fillna("Unknown").astype(str)
    df_docs[secondary_col] = df_docs[secondary_col].fillna("Unknown").astype(str)
    
    total_population = len(df_docs)

    # Thống kê số lượng phần tử của từng tiểu phân vùng hạ tầng
    group_counts = df_docs.groupby(primary_cols + [secondary_col]).size().reset_index(name='pop')
    
    # BƯỚC A: Phân bổ mẫu cho tầng Primary (year + source)
    primary_counts = df_docs.groupby(primary_cols).size().reset_index(name='prim_pop')
    primary_counts['prim_samples'] = (primary_counts['prim_pop'] / total_population * total_samples).apply(math.floor)
    
    # Khấu trừ và bù sai số phần dư cho tầng Primary
    diff = total_samples - primary_counts['prim_samples'].sum()
    if diff > 0:
        for idx in primary_counts.sort_values(by='prim_pop', ascending=False).index[:diff]:
            primary_counts.loc[idx, 'prim_samples'] += 1

    # Trộn thông tin phân bổ Primary vào bảng phân phối tổng hợp
    group_counts = group_counts.merge(primary_counts, on=primary_cols, how='left')
    
    # BƯỚC B: Phân bổ mẫu xuống tầng Secondary (category) nội bộ
    sampled_frames = []
    strata_details = {}

    for _, prim_row in primary_counts.iterrows():
        prim_samples_allocated = prim_row['prim_samples']
        prim_pop = prim_row['prim_pop']
        
        if prim_samples_allocated == 0:
            continue
            
        # Lọc các sub-groups thuộc tầng primary hiện tại
        cond_prim = (group_counts[primary_cols[0]] == prim_row[primary_cols[0]]) & \
                    (group_counts[primary_cols[1]] == prim_row[primary_cols[1]])
        sub_groups = group_counts[cond_prim].copy()
        
        sub_groups['sub_samples'] = (sub_groups['pop'] / prim_pop * prim_samples_allocated).apply(math.floor)
        
        # Bù sai số phần dư cho tầng Secondary
        sub_diff = prim_samples_allocated - sub_groups['sub_samples'].sum()
        if sub_diff > 0:
            for idx in sub_groups.sort_values(by='pop', ascending=False).index[:sub_diff]:
                sub_groups.loc[idx, 'sub_samples'] += 1

        # Trích xuất mẫu ngẫu nhiên thực tế dựa trên hạt giống cố định
        for _, sub_row in sub_groups.iterrows():
            n_samples = int(sub_row['sub_samples'])
            if n_samples == 0:
                continue
                
            n_samples = min(n_samples, sub_row['pop'])
            
            cond_extract = (df_docs[primary_cols[0]] == sub_row[primary_cols[0]]) & \
                           (df_docs[primary_cols[1]] == sub_row[primary_cols[1]]) & \
                           (df_docs[secondary_col] == sub_row[secondary_col])
            
            stratum_df = df_docs[cond_extract]
            sampled = stratum_df.sample(n=n_samples, random_state=random_state)
            sampled_frames.append(sampled)
            
            strata_key = f"{sub_row[primary_cols[0]]}_{sub_row[primary_cols[1]]} | {sub_row[secondary_col]}"
            strata_details[strata_key] = {
                "population": int(sub_row['pop']),
                "samples_drawn": int(n_samples),
                "sampling_rate": round(n_samples / sub_row['pop'], 4) if sub_row['pop'] > 0 else 0
            }

    df_sampled = pd.concat(sampled_frames, ignore_index=True)
    return df_sampled, strata_details, total_population


# ---------------------------------------------------------------------------
# Bước 3 — Ghi nhật ký phân tầng
# ---------------------------------------------------------------------------

def save_sampling_log(
    strata_col: str,
    total_population: int,
    total_samples_drawn: int,
    strata_details: dict
) -> None:
    """Xuất nhật ký phân tầng ra tệp JSON có cấu trúc tường minh."""
    log_data = {
        "stratification_criterion": strata_col,
        "random_state": RANDOM_STATE,
        "total_population": total_population,
        "total_samples_drawn": total_samples_drawn,
        "qa_per_article": QA_PER_ARTICLE,
        "total_qa_pairs_expected": total_samples_drawn * QA_PER_ARTICLE,
        "strata_details": strata_details
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Nhật ký phân tầng đã lưu: {LOG_PATH}")

    # In tóm tắt ra console
    print("\n" + "="*60)
    print(f"  CHIẾN LƯỢC LẤY MẪU: Proportional Stratified Sampling")
    print(f"  Tiêu chí phân tầng : '{strata_col}'")
    print(f"  Hạt giống cố định  : random_state={RANDOM_STATE}")
    print(f"  Tổng quần thể      : {total_population} bài báo")
    print(f"  Tổng mẫu lấy       : {total_samples_drawn} bài báo")
    print(f"  Cặp QA dự kiến     : {total_samples_drawn * QA_PER_ARTICLE}")
    print("="*60)
    print(f"  {'Tầng':<40} {'Quần thể':>10} {'Mẫu':>8} {'Tỷ lệ':>8}")
    print("-"*60)
    for stratum, info in strata_details.items():
        label = stratum[:38] + ".." if len(stratum) > 40 else stratum
        print(f"  {label:<40} {info['population']:>10} {info['samples_drawn']:>8} {info['sampling_rate']:>7.1%}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Bước 4 — Sinh cặp QA định dạng Ragas
# ---------------------------------------------------------------------------

QA_GENERATION_PROMPT_TEMPLATE = """\
Bạn là một chuyên gia tổng hợp dữ liệu huấn luyện cho hệ thống hỏi đáp tài chính Việt Nam.

Đọc kỹ đoạn văn bản báo tài chính dưới đây và tạo ra {n} cặp Hỏi-Đáp:
- Câu hỏi: Phải rõ ràng, liên quan đến thông tin cụ thể trong đoạn văn (số liệu, tổ chức, sự kiện).
- Câu trả lời: Ngắn gọn, chính xác, dựa hoàn toàn vào nội dung đoạn văn bản.
- BẮT BUỘC phải trích xuất thông tin có thật từ đoạn văn bản. 
- Quy tắc bắt buộc: TUYỆT ĐỐI KHÔNG được sử dụng dấu ngoặc kép (") hoặc các ký tự điều khiển bên trong nội dung của câu hỏi và câu trả lời. Nếu cần trích dẫn tên tổ chức, thuật ngữ hoặc danh mục, bắt buộc phải thay thế bằng dấu nháy đơn (')
- TUYỆT ĐỐI KHÔNG sử dụng dấu ba chấm "...", ký tự giữ chỗ hoặc bỏ trống các trường dữ liệu.
Đoạn văn bản:
---
{passage}
---

Trả về JSON hợp lệ theo cấu trúc sau (không thêm markdown hay giải thích):
{{
  "qa_pairs": [
    {{"question": "Câu hỏi 1?", "ground_truth": "Câu trả lời 1"}},
    {{"question": "Câu hỏi 2?", "ground_truth": "Câu trả lời 2"}}
  ]
}}
"""


def generate_qa_for_doc(doc_row: pd.Series, df_chunks: pd.DataFrame) -> list[dict]:
    """
    Sinh cặp QA cho một tài liệu.
    Lấy chunk đầu tiên của tài liệu làm passage chính để sinh câu hỏi.

    Returns:
        Danh sách dict với cấu trúc Ragas: {question, ground_truth, contexts, doc_id}
    """
    doc_id = doc_row["doc_id"]
    doc_chunks = df_chunks[df_chunks["doc_id"] == doc_id]["text"].tolist()

    if not doc_chunks:
        logger.warning(f"Không tìm thấy chunk cho doc_id: {doc_id}, bỏ qua.")
        return []

    # Lấy chunk đầu tiên làm passage sinh câu hỏi (chunk_0 thường chứa thông tin chính)
    primary_chunk = doc_chunks[0]

    prompt = QA_GENERATION_PROMPT_TEMPLATE.format(
        n=QA_PER_ARTICLE,
        passage=primary_chunk[:2000]   # Giới hạn để tránh vượt context window
    )

    try:
        result = generate_synthetic_qa_batch(prompt)
        qa_pairs = result.get("qa_pairs", [])

        ragas_records = []
        for pair in qa_pairs:
            if "question" in pair and "ground_truth" in pair:
                ragas_records.append({
                    "question"    : pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "contexts"    : doc_chunks,   # Toàn bộ chunks của tài liệu
                    "doc_id"      : doc_id,
                })
        return ragas_records

    except Exception as e:
        # Thay đổi từ cảnh báo (warning) sang lỗi chí mạng (critical error)
        logger.error(f"\n[CRITICAL API ERROR] Phát hiện lỗi hệ thống nghiêm trọng: {e}")
        logger.error("Tiến trình tự động kích hoạt chế độ Fail-Fast để bảo vệ tài khoản và tránh lặp lỗi.")
        
        # Thoát toàn bộ chương trình Python ngay lập tức với mã lỗi 1
        sys.exit(1)


# ---------------------------------------------------------------------------
# Điểm kiểm soát (Checkpoint) — tiếp tục từ nơi dừng
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = OUTPUT_DIR / "stratified_checkpoint.json"


def load_checkpoint() -> set:
    """Tải danh sách doc_id đã xử lý từ tệp checkpoint."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(data.get("processed_doc_ids", []))
        logger.info(f"Checkpoint phát hiện: {len(processed)} doc_id đã xử lý.")
        return processed
    return set()


def save_checkpoint(processed_ids: set) -> None:
    """Lưu trạng thái tiến độ vào checkpoint."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_doc_ids": list(processed_ids)}, f)


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def run_stratified_qa_pipeline():
    logger.info("="*60)
    logger.info("PHASE 5: MULTI-LEVEL STRATIFIED SAMPLING")
    logger.info("="*60)

    df_meta = load_metadata()

    # Xác thực sự tồn tại của các cột dữ liệu hệ thống
    verify_strata_columns(df_meta)

    # Thực hiện lấy mẫu đa cấp
    logger.info("Đang tính toán phân bổ mẫu đa cấp tỷ lệ thuận...")
    df_sampled, strata_details, total_population = stratified_sample(
        df=df_meta,
        primary_cols=PRIMARY_STRATA_COLS,
        secondary_col=SECONDARY_STRATA_COL,
        total_samples=TOTAL_SAMPLES,
        random_state=RANDOM_STATE
    )
    total_samples_drawn = len(df_sampled)

    # Ghi nhật ký lưu trữ (Sửa đổi strata_col thành chuỗi mô tả cấu trúc)
    strata_desc = f"{PRIMARY_STRATA_COLS} -> {SECONDARY_STRATA_COL}"
    save_sampling_log(strata_desc, total_population, total_samples_drawn, strata_details)

    # Sinh QA với checkpoint để tiếp tục nếu bị gián đoạn
    processed_ids = load_checkpoint()
    docs_to_process = df_sampled[~df_sampled["doc_id"].isin(processed_ids)]
    logger.info(f"Còn {len(docs_to_process)} tài liệu cần xử lý (đã có {len(processed_ids)} từ checkpoint).")

    total_qa_generated = 0
    qa_write_mode = "a" if CHECKPOINT_PATH.exists() else "w"

    with open(QA_OUT_PATH, qa_write_mode, encoding="utf-8") as out_f:
        for idx, (_, doc_row) in enumerate(docs_to_process.iterrows()):
            records = generate_qa_for_doc(doc_row, df_meta)

            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_qa_generated += 1

            processed_ids.add(doc_row["doc_id"])

            # Checkpoint mỗi 5 tài liệu
            if (idx + 1) % 5 == 0:
                save_checkpoint(processed_ids)
                logger.info(f"[Progress] {idx+1}/{len(docs_to_process)} tài liệu | {total_qa_generated} cặp QA")

            # Rate limit cooldown giữa các lần gọi API
            time.sleep(12)

    # Checkpoint cuối cùng
    save_checkpoint(processed_ids)
    logger.info(f"\n✅ Hoàn thành! Tổng cặp QA sinh ra: {total_qa_generated}")
    logger.info(f"   Tệp QA: {QA_OUT_PATH}")
    logger.info(f"   Nhật ký phân tầng: {LOG_PATH}")


if __name__ == "__main__":
    run_stratified_qa_pipeline()
