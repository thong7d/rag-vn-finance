"""
evaluate_aio_prompt.py — Phase 8 Extension: All-in-One Prompt Judge
======================================================================================
Script offline độc lập — KHÔNG được nhập vào app.py.

Pipeline đánh giá:
  1. Tải kết quả Generation từ Phase 7 (generation_results_{backend}.parquet).
  2. Bỏ qua RAG pipeline, sử dụng trực tiếp các contexts và answers đã sinh sẵn.
  3. Đánh giá bằng All-in-One Prompt (LLM-as-a-Judge) thay vì Ragas để tiết kiệm Token:
       - faithfulness     : Câu trả lời có dựa trên ngữ cảnh được cung cấp không?
       - answer_relevancy : Câu trả lời có tập trung giải quyết câu hỏi không?
       - context_recall   : Ngữ cảnh trích xuất có chứa đầy đủ thông tin để trả lời câu hỏi không?
  4. Xuất báo cáo evaluation_report_aio_prompt_{backend}.csv sử dụng cơ chế ghi đè an toàn (Atomic Write).
"""

import os
import sys
import json
import re
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, load_config, get_env

logger = setup_logger("LLM_Judge")
config = load_config()

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn và tham số từ config
# ---------------------------------------------------------------------------
backend = os.environ.get("VECTOR_STORE_BACKEND", "qdrant").lower()
OUTPUT_DIR = Path(config["evaluation"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{backend}.parquet"
REPORT_PATH = OUTPUT_DIR / f"evaluation_report_aio_prompt_{backend}.csv"

# ---------------------------------------------------------------------------
# Khởi tạo LLM Judge (Llama 3.3 70B qua LangChain ChatGroq)
# ---------------------------------------------------------------------------
def build_llm_judge():
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY chưa được thiết lập!")

    from langchain_groq import ChatGroq

    judge_llm = ChatGroq(
        model=config["evaluation"].get("groq_model", "llama-3.3-70b-versatile"),
        groq_api_key=groq_key,
        temperature=0.0,
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    return judge_llm

# ---------------------------------------------------------------------------
# All-in-One Prompt và cấu trúc Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI Judge evaluating a Vietnamese financial news RAG (Retrieval-Augmented Generation) system.
You will receive a QUESTION, the GROUND TRUTH (correct answer), the RETRIEVED CONTEXTS (a list of source passages), and the GENERATED ANSWER.
Your task is to grade the RAG system on three metrics: Faithfulness, Answer Relevancy, and Context Recall.
Evaluate each metric on a continuous scale from 0.0 (worst) to 1.0 (best).

Metrics Definitions & Scoring Rubric:
1. Faithfulness (Faithfulness of the generated answer to the retrieved contexts):
   - Measure if the generated answer contains only facts that are directly supported by the retrieved contexts.
   - Score 1.0 if all claims in the generated answer are fully supported by and can be directly inferred from the retrieved contexts.
   - Penalize the score if there are hallucinations, fabrications, or claims not mentioned in the contexts.
   - Score 0.0 if the answer is completely unfaithful or contradicts the contexts.

2. Answer Relevancy (Relevancy of the generated answer to the question):
   - Measure if the generated answer directly addresses what was asked, without redundancy, tangents, or generic statements.
   - Score 1.0 if the generated answer directly, clearly, and completely answers the question.
   - Penalize the score if the answer is verbose, goes on tangents, or misses key parts of the question.
   - Score 0.0 if the answer does not address the question at all.

3. Context Recall (How well the retrieved contexts cover the ground truth answer):
   - Measure if the retrieved contexts contain all the necessary information to reconstruct the ground truth answer.
   - Score 1.0 if all facts and key information in the ground truth answer are present in the retrieved contexts.
   - Penalize the score if key facts or pieces of information from the ground truth answer are missing from the retrieved contexts.
   - Score 0.0 if none of the information in the ground truth answer is present in the retrieved contexts.

You must respond with a JSON object ONLY, containing the scores and short reasonings in English. Do not include markdown code blocks (e.g. ```json), and do not write any explanation outside the JSON.
Your JSON response must match this schema:
{
  "faithfulness": {
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation>"
  },
  "answer_relevancy": {
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation>"
  },
  "context_recall": {
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation>"
  }
}"""

USER_PROMPT_TEMPLATE = """### QUESTION
{question}

### GROUND TRUTH
{ground_truth}

### RETRIEVED CONTEXTS
{retrieved_contexts}

### GENERATED ANSWER
{generated_answer}"""

# ---------------------------------------------------------------------------
# Fuzzy Key JSON Parser
# ---------------------------------------------------------------------------
def parse_judge_json(raw_text: str) -> dict:
    try:
        clean_text = raw_text.strip()
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(pattern, clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
            
        data = json.loads(clean_text)
        
        faithfulness_val = None
        for k, v in data.items():
            if "faith" in k.lower():
                faithfulness_val = v
                break
        
        relevancy_val = None
        for k, v in data.items():
            if "relev" in k.lower() or "answer" in k.lower():
                relevancy_val = v
                break
                
        recall_val = None
        for k, v in data.items():
            if "recall" in k.lower() or "context" in k.lower():
                recall_val = v
                break
                
        def extract_score_reason(val) -> Tuple[float, str]:
            score = 0.0
            reasoning = "No reason provided"
            if isinstance(val, dict):
                for k, v in val.items():
                    if "score" in k.lower():
                        score = float(v)
                    elif "reason" in k.lower():
                        reasoning = str(v)
            elif isinstance(val, (int, float)):
                score = float(val)
            return score, reasoning

        f_score, f_reason = extract_score_reason(faithfulness_val) if faithfulness_val is not None else (0.0, "Key missing")
        r_score, r_reason = extract_score_reason(relevancy_val) if relevancy_val is not None else (0.0, "Key missing")
        c_score, c_reason = extract_score_reason(recall_val) if recall_val is not None else (0.0, "Key missing")
        
        return {
            "faithfulness_score": f_score,
            "faithfulness_reasoning": f_reason,
            "answer_relevancy_score": r_score,
            "answer_relevancy_reasoning": r_reason,
            "context_recall_score": c_score,
            "context_recall_reasoning": c_reason
        }
    except Exception as e:
        logger.warning(f"Lỗi khi giải mã JSON phản hồi: {e}. Chuỗi thô: {raw_text}")
        return {
            "faithfulness_score": 0.0,
            "faithfulness_reasoning": f"Failed to parse JSON: {str(e)}",
            "answer_relevancy_score": 0.0,
            "answer_relevancy_reasoning": f"Failed to parse JSON: {str(e)}",
            "context_recall_score": 0.0,
            "context_recall_reasoning": f"Failed to parse JSON: {str(e)}"
        }

# ---------------------------------------------------------------------------
# Gọi API Giám khảo với cơ chế Retry / Rate Limit
# ---------------------------------------------------------------------------
def call_judge_with_retry(judge_llm, question: str, ground_truth: str, retrieved_contexts_str: str, generated_answer: str, max_retries: int = 3) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    user_msg = HumanMessage(content=USER_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        retrieved_contexts=retrieved_contexts_str,
        generated_answer=generated_answer
    ))
    
    for attempt in range(max_retries):
        try:
            response = judge_llm.invoke([system_msg, user_msg])
            raw_text = response.content.strip()
            parsed_result = parse_judge_json(raw_text)
            
            if (parsed_result["faithfulness_score"] == 0.0 and 
                parsed_result["faithfulness_reasoning"].startswith("Failed to parse JSON")):
                raise ValueError(f"JSON Output không đúng định dạng mong muốn: {raw_text[:200]}")
                
            return parsed_result
        except Exception as e:
            err_str = str(e).lower()
            logger.warning(f"[Judge API] Thử lần {attempt + 1}/{max_retries} thất bại. Chi tiết: {e}")
            
            # Detect critical rate limit (Daily Quota) or Out of Funds
            if "limit 100000" in err_str or "tokens per day" in err_str:
                logger.error("!!! CRITICAL: Đã đạt giới hạn Token theo ngày (Daily Limit) của Groq !!! Dừng chương trình ngay.")
                sys.exit(1)
            if "402" in err_str or "out of credits" in err_str:
                logger.error("!!! CRITICAL: Hết tiền (Out of Credits/Funds) !!! Dừng chương trình ngay.")
                sys.exit(1)
                
            if "429" in err_str or "rate limit" in err_str or "tpm" in err_str or "rpm" in err_str:
                sleep_time = 30 + 10 * attempt
                logger.info(f"[Judge API] Phát hiện giới hạn cuộc gọi. Tạm nghỉ {sleep_time}s trước khi thử lại...")
                time.sleep(sleep_time)
            else:
                sleep_time = 5 * (attempt + 1)
                logger.info(f"[Judge API] Gặp lỗi thông thường. Thử lại sau {sleep_time}s...")
                time.sleep(sleep_time)
                
    # Nếu chạy hết 3 lượt vẫn lỗi, KHÔNG trả về 0.0 để tránh làm hỏng Checkpoint
    # Thay vào đó, ném ngoại lệ để main() bắt và dừng chương trình
    raise RuntimeError("Tất cả 3 lượt thử gọi API đều thất bại. Có thể do lỗi mạng hoặc Rate Limit kéo dài. Dừng chương trình để bảo toàn Checkpoint.")

# ---------------------------------------------------------------------------
# Lưu file an toàn (Atomic Save CSV)
# ---------------------------------------------------------------------------
def atomic_save_csv(df: pd.DataFrame, file_path: Path):
    temp_path = file_path.with_suffix(".csv.tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        if temp_path.exists():
            os.replace(temp_path, file_path)
            logger.info(f"[Atomic Save] Đã lưu báo cáo cập nhật thành công vào {file_path}")
    except Exception as e:
        logger.error(f"[Atomic Save ERROR] Không thể lưu file an toàn: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline RAG Evaluation sử dụng All-in-One Llama 3.3 Judge")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu câu hỏi đánh giá")
    parser.add_argument("--cooldown", type=int, default=20, help="Thời gian nghỉ giữa các lần chấm điểm (giây) để chống Rate Limit")
    args = parser.parse_args()
    
    # 1. Đọc kết quả đã sinh từ Phase 7
    if not INPUT_PARQUET.exists():
        logger.error(f"Không tìm thấy file kết quả Generation tại: {INPUT_PARQUET}")
        sys.exit(1)
        
    df_eval = pd.read_parquet(INPUT_PARQUET)
    records = df_eval.to_dict("records")
    logger.info(f"Đã tải {len(records)} mẫu từ {INPUT_PARQUET}")
    
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Giới hạn đánh giá: {len(records)} mẫu đầu tiên.")
        
    # 2. Đọc kết quả cũ từ CSV báo cáo (nếu có) để bỏ qua
    existing_df = None
    existing_questions = set()
    if REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(REPORT_PATH)
            completed_df = existing_df[existing_df["faithfulness_score"].notna()]
            existing_questions = set(completed_df["question"].astype(str).str.strip().tolist())
            logger.info(f"Đã tìm thấy {len(existing_questions)} mẫu đã được chấm điểm trong báo cáo cũ. Sẽ bỏ qua.")
        except Exception as e:
            logger.warning(f"Không thể đọc kết quả cũ: {e}. Sẽ chạy lại mới từ đầu.")
            
    pending_records = [r for r in records if r["question"].strip() not in existing_questions]
    logger.info(f"Tổng số mẫu cần xử lý thực tế: {len(pending_records)}")
    
    if not pending_records:
        logger.info("Không có mẫu mới nào cần đánh giá. Chương trình kết thúc.")
        return
        
    # 3. Khởi tạo LLM Judge
    try:
        judge_llm = build_llm_judge()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo LLM Judge: {e}")
        sys.exit(1)
        
    # Vòng lặp chính xử lý từng mẫu
    new_results = []
    total_processed = 0
    
    for idx, record in enumerate(pending_records):
        question = record["question"]
        ground_truth = record.get("ground_truth", "")
        retrieved_contexts_str = record.get("retrieved_context", "")
        generated_answer = record.get("generated_answer", "")
        
        logger.info(f"\n==========================================")
        logger.info(f"Xử lý mẫu {idx + 1}/{len(pending_records)} (Tổng số: {total_processed + 1})")
        logger.info(f"Câu hỏi: {question[:100]}...")
        
        # Đánh giá bằng LLM Judge (All-in-One Prompt)
        logger.info("Đang gọi Llama 3.3 Judge để đánh giá 3 chỉ số...")
        try:
            scores = call_judge_with_retry(
                judge_llm=judge_llm,
                question=question,
                ground_truth=ground_truth,
                retrieved_contexts_str=retrieved_contexts_str,
                generated_answer=generated_answer
            )
        except RuntimeError as e:
            logger.error(f"[FATAL] {e}")
            sys.exit(1)
        
        logger.info(f"-> Chấm điểm thành công:")
        logger.info(f"   - Faithfulness: {scores['faithfulness_score']} ({scores['faithfulness_reasoning']})")
        logger.info(f"   - Answer Relevancy: {scores['answer_relevancy_score']} ({scores['answer_relevancy_reasoning']})")
        logger.info(f"   - Context Recall: {scores['context_recall_score']} ({scores['context_recall_reasoning']})")
        
        # Gom kết quả mới vào bộ đệm
        res_row = {
            "question": question,
            "ground_truth": ground_truth,
            "rag_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts_str,
            "faithfulness_score": scores["faithfulness_score"],
            "faithfulness_reasoning": scores["faithfulness_reasoning"],
            "answer_relevancy_score": scores["answer_relevancy_score"],
            "answer_relevancy_reasoning": scores["answer_relevancy_reasoning"],
            "context_recall_score": scores["context_recall_score"],
            "context_recall_reasoning": scores["context_recall_reasoning"]
        }
        new_results.append(res_row)
        total_processed += 1
        
        # Checkpoint mỗi 5 mẫu (hoặc khi kết thúc tập dữ liệu)
        if total_processed % 5 == 0 or idx == len(pending_records) - 1:
            logger.info("Đang lưu checkpoint dữ liệu an toàn...")
            df_new = pd.DataFrame(new_results)
            if existing_df is not None:
                df_combined = pd.concat([existing_df, df_new], ignore_index=True)
            else:
                df_combined = df_new
                
            atomic_save_csv(df_combined, REPORT_PATH)
            existing_df = df_combined
            new_results = []
            
        # Sleep Cooldown chống Rate Limit
        if idx < len(pending_records) - 1:
            logger.info(f"Nghỉ cooldown {args.cooldown}s chống rate limit Groq...")
            time.sleep(args.cooldown)
            
    # 4. Hiển thị báo cáo thống kê cuối cùng
    try:
        final_df = pd.read_csv(REPORT_PATH)
        logger.info("\n==========================================")
        logger.info("📊 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG (ALL-IN-ONE RAGAS)")
        logger.info("==========================================")
        logger.info(f"Tổng số mẫu đã đánh giá: {len(final_df)}")
        logger.info(f"Faithfulness trung bình    : {final_df['faithfulness_score'].mean():.4f}")
        logger.info(f"Answer Relevancy trung bình: {final_df['answer_relevancy_score'].mean():.4f}")
        logger.info(f"Context Recall trung bình  : {final_df['context_recall_score'].mean():.4f}")
        logger.info("==========================================")
    except Exception as e:
        logger.error(f"Lỗi khi đọc tệp báo cáo cuối cùng để in thống kê: {e}")

if __name__ == "__main__":
    main()
