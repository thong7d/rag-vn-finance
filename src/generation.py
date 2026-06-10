"""
generation.py — Answer generation module for Phase 5 (Synthetic QA) and Phase 7 (RAG pipeline).

Model priority for Phase 7 (generate_answer):
  1. Gemini  : gemini-3.1-flash-lite  (if GEMINI_API_KEY is set)
  2. OpenRouter: google/gemma-4-31b-it:free (fallback, from config.yaml)

Model for Phase 5 (generate_synthetic_qa_batch):
  OpenRouter: google/gemma-4-31b-it:free (if OPENROUTER_API_KEY is set)
"""

from openai import OpenAI
import json
import re
import time
import logging
from typing import List
from src.utils import get_env, load_config

logger = logging.getLogger(__name__)
config = load_config()

# ── System Prompt (Phase 7) ────────────────────────────────────────────────────
# Three-component Vietnamese prompt architecture:
#   [System Instruction] + [Retrieved Context] + [User Question]
RAG_SYSTEM_PROMPT = """Bạn là một chuyên viên phân tích tài chính chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng **dựa hoàn toàn** vào các đoạn ngữ cảnh báo chí tài chính được cung cấp bên dưới.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp. Không bịa đặt số liệu hoặc sự kiện.
2. Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ: "Thông tin trong ngữ cảnh không đủ để trả lời câu hỏi này."
3. Trả lời ngắn gọn, chính xác bằng tiếng Việt (2–4 câu là đủ cho hầu hết câu hỏi).
4. Nếu câu hỏi liên quan đến mã chứng khoán hoặc tổ chức (ví dụ: VIC, HPG, Vietcombank), hãy trích dẫn chính xác mã/tên đó trong câu trả lời.
5. Khi trích dẫn số liệu tài chính (lãi suất, doanh thu, lợi nhuận,...), luôn đi kèm thời điểm nếu có trong ngữ cảnh."""


def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """
    Build the user-turn of the RAG prompt.

    Combines numbered retrieved passages (Component 2) with the user
    question (Component 3). The system prompt (Component 1) is passed
    separately in the 'system' role.

    Args:
        query:    The user's question (Vietnamese).
        contexts: Ordered list of retrieved chunk texts.

    Returns:
        Formatted user prompt string.
    """
    if not contexts:
        context_block = "[Không có ngữ cảnh nào được truy xuất.]"
    else:
        context_block = "\n\n---\n\n".join(
            [f"[Đoạn {i + 1}]:\n{ctx.strip()}" for i, ctx in enumerate(contexts) if ctx.strip()]
        )
    return (
        f"Dưới đây là các đoạn thông tin được truy xuất từ kho dữ liệu báo chí tài chính Việt Nam:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Câu hỏi: {query}\n\n"
        f"Hãy trả lời câu hỏi dựa vào các đoạn ngữ cảnh trên. "
        f"Nếu có mã chứng khoán hoặc tổ chức liên quan, hãy trích dẫn rõ trong câu trả lời."
    )


def strip_markdown_json(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def generate_answer(query: str, contexts: List[str], max_retries: int = 1) -> str:
    """
    Generate a RAG answer given a query and a list of retrieved context strings.
    Implements a 3-layer Auto-Fallback architecture:
      1. Gemini (gemini-3.1-flash-lite)
      2. OpenRouter (google/gemma-4-31b-it:free)
    """
    keys = {
        "gemini": get_env('GEMINI_API_KEY'),
        "openrouter": get_env('OPENROUTER_API_KEY')
    }

    backends = []
    if keys["gemini"]:
        backends.append({
            "name": "Gemini",
            "client": OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=keys["gemini"]),
            "model": "gemini-3.1-flash-lite"
        })
    if keys["openrouter"]:
        backends.append({
            "name": "OpenRouter",
            "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=keys["openrouter"]),
            "model": config['generation'].get('model', 'google/gemma-4-31b-it:free')
        })
        
    if not backends:
        raise ValueError("Vui lòng set ít nhất một API KEY (GROQ, OPENROUTER, hoặc GEMINI).")

    user_prompt = build_rag_prompt(query, contexts)

    last_exception = None
    for backend in backends:
        logger.info(f"Đang xử lý qua {backend['name']} ({backend['model']})...")
        for attempt in range(max_retries + 1):
            try:
                response = backend["client"].chat.completions.create(
                    model=backend["model"],
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config['generation'].get('temperature', 0.2),
                    max_tokens=config['generation'].get('max_tokens', 1024)
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_exception = e
                logger.warning(f"{backend['name']} thất bại (lần {attempt + 1}): {e}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    logger.info(f"Đang thử lại {backend['name']} sau {sleep_time}s...")
                    time.sleep(sleep_time)
        
        logger.warning(f"FALLBACK: Không thể gọi {backend['name']}, chuyển sang backend tiếp theo...")

    raise RuntimeError(f"Tất cả các backend đều sập. Lỗi cuối cùng: {last_exception}")


def generate_synthetic_qa_batch(prompt: str, max_retries: int = 2) -> dict:
    """
    Generate a batch of QA pairs using llama-4-scout via Groq Cloud API.
    """
    groq_key = get_env('GROQ_API_KEY')  # Đảm bảo đã khai báo GROQ_API_KEY trong file .env

    if not groq_key:
        raise ValueError("Vui lòng thiết lập biến môi trường GROQ_API_KEY để kết nối hạ tầng Groq.")

    # Khởi tạo kết nối thông qua cổng tương thích OpenAI của Groq
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key
    )
    
    # Cấu hình mã định danh mô hình chuẩn theo bảng Quota Groq
    model_name = "qwen/qwen3-32b"

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Ép logic toán học tối đa để kiểm soát cấu trúc JSON
                "response_format": {"type": "json_object"}  # Kích hoạt JSON Mode gốc của Groq
            }

            response = client.chat.completions.create(**kwargs)
            output_text = response.choices[0].message.content
            
            if not output_text or output_text.strip() == "":
                raise ValueError("Mô hình Groq phản hồi chuỗi trống.")

            start_idx = output_text.find("{")
            if start_idx == -1:
                raise ValueError(f"Phản hồi không chứa ký tự mở đầu JSON.")

            json_candidate = output_text[start_idx:]

            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(json_candidate)
            return data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Giải mã JSON thất bại tại lượt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise ValueError("Mô hình không thể xuất ra cấu trúc định dạng JSON sạch.")
        except Exception as e:
            logger.warning(f"Groq API gặp sự cố kết nối (lần {attempt + 1}): {e}")
            if attempt == max_retries:
                raise e

            sleep_time = (2 ** attempt) + 2
            time.sleep(sleep_time)