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
    Locked to Gemini (gemini-3.1-flash-lite) to synchronize with Phase 7.
    """
    gemini_key = get_env('GEMINI_API_KEY')
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
        api_key=gemini_key
    )
    
    model_name = "gemini-3.1-flash-lite"
    user_prompt = build_rag_prompt(query, contexts)

    last_exception = None
    logger.info(f"Sending request via Gemini ({model_name})...")
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
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
            logger.warning(f"Gemini failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying Gemini in {sleep_time}s...")
                time.sleep(sleep_time)

    raise RuntimeError(f"Gemini API call failed. Last error: {last_exception}")


def generate_synthetic_qa_batch(prompt: str, max_retries: int = 2) -> dict:
    """
    Generate a batch of QA pairs using qwen/qwen3-32b via Groq Cloud API.
    """
    groq_key = get_env('GROQ_API_KEY')  # Ensure GROQ_API_KEY is declared in .env

    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Required to connect to Groq.")

    # Initialize client via Groq's OpenAI-compatible endpoint
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key
    )
    
    # Model identifier per Groq's official quota table
    model_name = "qwen/qwen3-32b"

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic output for strict JSON structure control
                "response_format": {"type": "json_object"}  # Enable Groq's native JSON Mode
            }

            response = client.chat.completions.create(**kwargs)
            output_text = response.choices[0].message.content
            
            if not output_text or output_text.strip() == "":
                raise ValueError("Groq model returned an empty response.")

            start_idx = output_text.find("{")
            if start_idx == -1:
                raise ValueError("Response does not contain a JSON opening brace.")

            json_candidate = output_text[start_idx:]

            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(json_candidate)
            return data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON decode failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise ValueError("Model failed to produce a clean JSON-formatted output.")
        except Exception as e:
            logger.warning(f"Groq API connection error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise e

            sleep_time = (2 ** attempt) + 2
            time.sleep(sleep_time)