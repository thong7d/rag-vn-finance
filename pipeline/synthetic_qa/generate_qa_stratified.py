"""
generate_qa_stratified.py — Phase 5: Stratified Sampling & Synthetic QA Generation
====================================================================================

Stratified Sampling Strategy:
  - Stratification criteria: 'source' column (news domain) in metadata.
    Falls back to 'ticker' or 'category' if 'source' is absent.
  - Allocation ratio: Proportional Stratified Sampling (sample size ~ stratum size).
  - Fixed random seed: random_state=42 ensures full reproducibility.
  - Target samples: 500 articles (qa_per_article=2 → 1000 QA pairs).

Outputs:
  - synthetic_qa/stratified_sampling_log.json : Detailed stratification log.
  - synthetic_qa/qa_dataset_stratified.jsonl  : QA dataset in standard Ragas format.

Run this script offline from the implementation/ directory:
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

# Add root directory (implementation/) to PYTHONPATH for src/ imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, load_config, get_env
from src.generation import generate_synthetic_qa_batch

logger = setup_logger("Phase5_Stratified_QA")
config = load_config()

# ---------------------------------------------------------------------------
# Sampling strategy configuration
# ---------------------------------------------------------------------------

RANDOM_STATE    = 42          # Fixed seed — required for reproducibility
TOTAL_SAMPLES   = 500         # Total number of articles to sample
QA_PER_ARTICLE  = 2           # Number of QA pairs to generate per article

# MULTI-LEVEL STRATIFICATION CONFIG:
PRIMARY_STRATA_COLS = ["year", "source"]
SECONDARY_STRATA_COL = "category"

OUTPUT_DIR = Path(config["synthetic_qa"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH    = OUTPUT_DIR / "stratified_sampling_log.json"
QA_OUT_PATH = OUTPUT_DIR / "qa_dataset_stratified.jsonl"

# Input chunk metadata file (uses fixed_size strategy from Phase 3)
METADATA_PATH = Path(config["indexing"]["output_dir"]) / "fixed_size" / "metadata.parquet"


# ---------------------------------------------------------------------------
# Step 1 — Load chunk metadata and verify stratification columns
# ---------------------------------------------------------------------------

def verify_strata_columns(df: pd.DataFrame):
    """Validate that all required stratification columns exist in the DataFrame."""
    required_cols = PRIMARY_STRATA_COLS + [SECONDARY_STRATA_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data model is missing required stratification columns: {missing_cols}")
    logger.info(f"Multi-level stratification schema validated: {PRIMARY_STRATA_COLS} -> {SECONDARY_STRATA_COL}")


def load_metadata() -> pd.DataFrame:
    """Load the metadata.parquet file and return a DataFrame."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.parquet not found at: {METADATA_PATH}\n"
            f"Please run Phase 3 (Embedding & Indexing) first."
        )
    df = pd.read_parquet(METADATA_PATH)
    logger.info(f"Metadata loaded: {len(df)} chunks from {METADATA_PATH}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Proportional Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, primary_cols: list, secondary_col: str, total_samples: int, random_state: int) -> tuple:
    """
    Perform multi-level proportional stratified sampling.
    """
    df_docs = df.drop_duplicates(subset="doc_id").copy()
    
    # Normalize missing values to prevent dtype errors
    df_docs[primary_cols[0]] = df_docs[primary_cols[0]].fillna("Unknown").astype(str)
    df_docs[primary_cols[1]] = df_docs[primary_cols[1]].fillna("Unknown").astype(str)
    df_docs[secondary_col] = df_docs[secondary_col].fillna("Unknown").astype(str)
    
    total_population = len(df_docs)

    # Count elements per sub-stratum
    group_counts = df_docs.groupby(primary_cols + [secondary_col]).size().reset_index(name='pop')
    
    # STEP A: Allocate samples for primary strata (year + source)
    primary_counts = df_docs.groupby(primary_cols).size().reset_index(name='prim_pop')
    primary_counts['prim_samples'] = (primary_counts['prim_pop'] / total_population * total_samples).apply(math.floor)
    
    # Round-up correction for primary strata remainder
    diff = total_samples - primary_counts['prim_samples'].sum()
    if diff > 0:
        for idx in primary_counts.sort_values(by='prim_pop', ascending=False).index[:diff]:
            primary_counts.loc[idx, 'prim_samples'] += 1

    # Merge primary allocation info into combined distribution table
    group_counts = group_counts.merge(primary_counts, on=primary_cols, how='left')
    
    # STEP B: Allocate samples down to secondary stratum (category) within each primary group
    sampled_frames = []
    strata_details = {}

    for _, prim_row in primary_counts.iterrows():
        prim_samples_allocated = prim_row['prim_samples']
        prim_pop = prim_row['prim_pop']
        
        if prim_samples_allocated == 0:
            continue
            
        # Filter sub-groups belonging to the current primary stratum
        cond_prim = (group_counts[primary_cols[0]] == prim_row[primary_cols[0]]) & \
                    (group_counts[primary_cols[1]] == prim_row[primary_cols[1]])
        sub_groups = group_counts[cond_prim].copy()
        
        sub_groups['sub_samples'] = (sub_groups['pop'] / prim_pop * prim_samples_allocated).apply(math.floor)
        
        # Round-up correction for secondary strata remainder
        sub_diff = prim_samples_allocated - sub_groups['sub_samples'].sum()
        if sub_diff > 0:
            for idx in sub_groups.sort_values(by='pop', ascending=False).index[:sub_diff]:
                sub_groups.loc[idx, 'sub_samples'] += 1

        # Draw random samples using fixed seed
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
# Step 3 — Save stratification log
# ---------------------------------------------------------------------------

def save_sampling_log(
    strata_col: str,
    total_population: int,
    total_samples_drawn: int,
    strata_details: dict
) -> None:
    """Export stratification log to a structured JSON file."""
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
    logger.info(f"Stratification log saved: {LOG_PATH}")

    # Print summary to console
    print("\n" + "="*60)
    print(f"  SAMPLING STRATEGY: Proportional Stratified Sampling")
    print(f"  Stratification criterion: '{strata_col}'")
    print(f"  Fixed random seed        : random_state={RANDOM_STATE}")
    print(f"  Total population         : {total_population} articles")
    print(f"  Total samples drawn      : {total_samples_drawn} articles")
    print(f"  Expected QA pairs        : {total_samples_drawn * QA_PER_ARTICLE}")
    print("="*60)
    print(f"  {'Stratum':<40} {'Population':>10} {'Samples':>8} {'Rate':>8}")
    print("-"*60)
    for stratum, info in strata_details.items():
        label = stratum[:38] + ".." if len(stratum) > 40 else stratum
        print(f"  {label:<40} {info['population']:>10} {info['samples_drawn']:>8} {info['sampling_rate']:>7.1%}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Step 4 — Generate QA pairs in Ragas format
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
    Generate QA pairs for a single document.
    Uses the first chunk of the document as the primary passage for question generation.

    Returns:
        List of dicts in Ragas format: {question, ground_truth, contexts, doc_id}
    """
    doc_id = doc_row["doc_id"]
    doc_chunks = df_chunks[df_chunks["doc_id"] == doc_id]["text"].tolist()

    if not doc_chunks:
        logger.warning(f"No chunks found for doc_id: {doc_id}. Skipping.")
        return []

    # Use the first chunk as the primary passage (chunk_0 typically contains the core info)
    primary_chunk = doc_chunks[0]

    prompt = QA_GENERATION_PROMPT_TEMPLATE.format(
        n=QA_PER_ARTICLE,
        passage=primary_chunk[:2000]   # Limit to avoid exceeding model context window
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
                    "contexts"    : doc_chunks,   # All chunks from the document
                    "doc_id"      : doc_id,
                })
        return ragas_records

    except Exception as e:
        # Treat as critical failure and trigger fail-fast to protect the account
        logger.error(f"\n[CRITICAL API ERROR] Severe system error detected: {e}")
        logger.error("Process activating Fail-Fast mode to avoid repeated errors and protect API quota.")
        
        # Exit the entire Python process immediately with error code 1
        sys.exit(1)


# ---------------------------------------------------------------------------
# Checkpoint — resume from last saved position
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = OUTPUT_DIR / "stratified_checkpoint.json"


def load_checkpoint() -> set:
    """Load the list of already-processed doc_ids from the checkpoint file."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(data.get("processed_doc_ids", []))
        logger.info(f"Checkpoint found: {len(processed)} doc_ids already processed.")
        return processed
    return set()


def save_checkpoint(processed_ids: set) -> None:
    """Save current progress to the checkpoint file."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_doc_ids": list(processed_ids)}, f)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stratified_qa_pipeline():
    logger.info("="*60)
    logger.info("PHASE 5: MULTI-LEVEL STRATIFIED SAMPLING")
    logger.info("="*60)

    df_meta = load_metadata()

    # Validate stratification column existence
    verify_strata_columns(df_meta)

    # Perform multi-level proportional sampling
    logger.info("Computing multi-level proportional sample allocation...")
    df_sampled, strata_details, total_population = stratified_sample(
        df=df_meta,
        primary_cols=PRIMARY_STRATA_COLS,
        secondary_col=SECONDARY_STRATA_COL,
        total_samples=TOTAL_SAMPLES,
        random_state=RANDOM_STATE
    )
    total_samples_drawn = len(df_sampled)

    # Save stratification log
    strata_desc = f"{PRIMARY_STRATA_COLS} -> {SECONDARY_STRATA_COL}"
    save_sampling_log(strata_desc, total_population, total_samples_drawn, strata_details)

    # Generate QA with checkpoint support for resumption
    processed_ids = load_checkpoint()
    docs_to_process = df_sampled[~df_sampled["doc_id"].isin(processed_ids)]
    logger.info(f"{len(docs_to_process)} documents remaining (already have {len(processed_ids)} from checkpoint).")

    total_qa_generated = 0
    qa_write_mode = "a" if CHECKPOINT_PATH.exists() else "w"

    with open(QA_OUT_PATH, qa_write_mode, encoding="utf-8") as out_f:
        for idx, (_, doc_row) in enumerate(docs_to_process.iterrows()):
            records = generate_qa_for_doc(doc_row, df_meta)

            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_qa_generated += 1

            processed_ids.add(doc_row["doc_id"])

            # Checkpoint every 5 documents
            if (idx + 1) % 5 == 0:
                save_checkpoint(processed_ids)
                logger.info(f"[Progress] {idx+1}/{len(docs_to_process)} documents | {total_qa_generated} QA pairs")

            # Rate-limit cooldown between API calls
            time.sleep(12)

    # Final checkpoint
    save_checkpoint(processed_ids)
    logger.info(f"\nCompleted! Total QA pairs generated: {total_qa_generated}")
    logger.info(f"   QA file: {QA_OUT_PATH}")
    logger.info(f"   Stratification log: {LOG_PATH}")


if __name__ == "__main__":
    run_stratified_qa_pipeline()
