"""
check_groq_model.py — Script độc lập kiểm tra sự tồn tại và chạy thử mẫu đánh giá AIO (All-in-One) với mô hình Groq API.

Sử dụng:
    python scripts/check_groq_model.py
    python scripts/check_groq_model.py --model qwen/qwen3.6-27b
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Tự động nạp file .env từ thư mục gốc hoặc backend nếu có
try:
    from dotenv import load_dotenv
    env_paths = [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent / "backend" / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[*] Đã nạp biến môi trường từ: {env_path}")
            break
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("❌ Thư viện 'openai' chưa được cài đặt. Vui lòng chạy: pip install openai")
    sys.exit(1)


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


def strip_thinking_and_markdown(text: str) -> str:
    """Loại bỏ khối <think>...</think> (nếu là reasoning model như Qwen 3.6) và trích xuất JSON."""
    text = text.strip()

    # Stripping <think>...</think> block if present
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Stripping ```json ... ``` fences
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Extract JSON object substring if there's surrounding text
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    return text


def parse_judge_json(raw_text: str) -> dict:
    clean_text = strip_thinking_and_markdown(raw_text)

    try:
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

        def extract_score_reason(val):
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
        print(f"⚠️ Failed to parse JSON: {e}\nRaw output:\n{raw_text}")
        return {
            "faithfulness_score": 0.0,
            "faithfulness_reasoning": f"Failed to parse JSON: {str(e)}",
            "answer_relevancy_score": 0.0,
            "answer_relevancy_reasoning": f"Failed to parse JSON: {str(e)}",
            "context_recall_score": 0.0,
            "context_recall_reasoning": f"Failed to parse JSON: {str(e)}"
        }


def run_aio_sample_eval(client: OpenAI, target_model: str):
    print(f"\n=======================================================")
    print(f"🧪 BẮT ĐẦU CHẠY THỬ MẪU ĐÁNH GIÁ AIO VỚI MODEL: '{target_model}'")
    print(f"=======================================================")

    sample_question = "Lợi nhuận sau thuế năm 2023 của Vietcombank (VCB) đạt bao nhiêu?"
    sample_gt = "Năm 2023, Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank - VCB) ghi nhận lợi nhuận sau thuế hợp nhất đạt 33.054 tỷ đồng, tăng 10,2% so với năm 2022."
    sample_context = "Theo báo cáo tài chính hợp nhất năm 2023, Vietcombank (VCB) đạt lợi nhuận trước thuế kỷ lục 41.244 tỷ đồng và lợi nhuận sau thuế đạt 33.054 tỷ đồng."
    sample_answer = "Lợi nhuận sau thuế của Vietcombank (VCB) năm 2023 đạt 33.054 tỷ đồng, tăng trưởng 10,2% so với năm trước."

    print(f"📌 [Input Data]")
    print(f"  • Question        : {sample_question}")
    print(f"  • Ground Truth    : {sample_gt}")
    print(f"  • Context         : {sample_context}")
    print(f"  • Generated Answer: {sample_answer}\n")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=sample_question,
        ground_truth=sample_gt,
        retrieved_contexts=sample_context,
        generated_answer=sample_answer
    )

    print(f"[*] Gửi yêu cầu AIO Judge tới Groq ({target_model})...")

    raw_output = None

    # Thử Lần 1: Dùng response_format={"type": "json_object"}
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        if "json_validate_failed" in str(e) or "Failed to validate JSON" in str(e):
            print(f"⚠️ Groq JSON Mode thất bại (do mô hình Reasoning tự sinh khối <think> trước JSON).")
            print(f"🔄 Tự động chuyển sang chế độ gọi Standard (không ép JSON Mode trên Groq server)...")
            try:
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2048
                )
                raw_output = response.choices[0].message.content.strip()
            except Exception as e2:
                print(f"❌ Thử nghiệm AIO Evaluation thất bại ở chế độ Standard: {e2}")
                return
        else:
            print(f"❌ Thử nghiệm AIO Evaluation thất bại: {e}")
            return

    if not raw_output:
        print("❌ Không nhận được kết quả từ mô hình.")
        return

    print("\n--- [Raw Output từ Groq] ---")
    print(raw_output)
    print("-----------------------------\n")

    scores = parse_judge_json(raw_output)

    print("📊 [KẾT QUẢ ĐÁNH GIÁ AIO EVALUATION]:")
    print(f"  ✅ Faithfulness Score    : {scores['faithfulness_score']:.2f}")
    print(f"     Reasoning            : {scores['faithfulness_reasoning']}")
    print(f"  ✅ Answer Relevancy Score: {scores['answer_relevancy_score']:.2f}")
    print(f"     Reasoning            : {scores['answer_relevancy_reasoning']}")
    print(f"  ✅ Context Recall Score  : {scores['context_recall_score']:.2f}")
    print(f"     Reasoning            : {scores['context_recall_reasoning']}")
    print(f"\n🎉 Thử nghiệm AIO Evaluation với '{target_model}' thành công hoàn toàn!")


def check_groq_model(target_model: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Lỗi: Không tìm thấy biến môi trường GROQ_API_KEY.")
        print("   Vui lòng thiết lập biến môi trường hoặc khai báo trong file .env")
        sys.exit(1)

    print(f"\n[*] Đang kết nối tới Groq API (https://api.groq.com/openai/v1)...")
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    # Step 1: Lấy danh sách toàn bộ các mô hình hiện có trên Groq
    print("[*] Đang truy vấn danh sách mô hình từ Groq...")
    try:
        models_response = client.models.list()
        all_models = [m.id for m in models_response.data]
        all_models.sort()
    except Exception as e:
        print(f"❌ Không thể kết nối tới Groq API: {e}")
        sys.exit(1)

    print(f"✅ Đã tải thành công {len(all_models)} mô hình từ Groq.")

    # Step 2: Kiểm tra mô hình mục tiêu trong danh sách
    print(f"\n[*] Kiểm tra mô hình mục tiêu: '{target_model}'")
    exact_match = target_model in all_models

    if exact_match:
        print(f"✅ TÌM THẤY TÊN MÔ HÌNH CHÍNH XÁC: '{target_model}'")
    else:
        print(f"⚠️  Không tìm thấy tên chính xác '{target_model}' trong danh sách Groq.")

    # Step 3: Thử nghiệm AIO Evaluation Sample
    run_aio_sample_eval(client, target_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra mô hình và chạy mẫu AIO Evaluation trên Groq API")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3.6-27b",
        help="Tên mô hình cần kiểm tra (mặc định: qwen/qwen3.6-27b)"
    )
    args = parser.parse_args()
    check_groq_model(args.model)
