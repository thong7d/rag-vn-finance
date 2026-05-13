"""
generation.py — OpenRouter answer generation for Phase 7.
Model: google/gemma-4-31b-it:free (from config.yaml).
Implemented in Phase 5 and 7.
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

# ── Prompt Template ────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """Bạn là một chuyên viên phân tích tài chính chuyên nghiệp. \
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng **dựa hoàn toàn** vào các đoạn ngữ cảnh được cung cấp bên dưới. \
Hãy trả lời ngắn gọn, chính xác và bằng tiếng Việt. \
Nếu thông tin trong các ngữ cảnh không đủ để trả lời, hãy nói rõ điều đó thay vì bịa đặt."""

def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """Build a RAG user prompt by injecting retrieved context chunks."""
    context_block = "\n\n---\n\n".join(
        [f"[Ngữ cảnh {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)]
    )
    return f"""Dưới đây là các đoạn thông tin được truy xuất từ kho dữ liệu tài chính:

{context_block}

---

Câu hỏi: {query}

Hãy trả lời câu hỏi dựa vào các ngữ cảnh trên."""


def strip_markdown_json(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def generate_answer(query: str, contexts: List[str], max_retries: int = 2) -> str:
    """
    Generate a RAG answer given a query and a list of retrieved context strings.
    Prioritizes Groq if GROQ_API_KEY is available, otherwise falls back to OpenRouter.

    Args:
        query: The user's question.
        contexts: List of retrieved text chunks to use as context.
        max_retries: Number of retries on API failure.

    Returns:
        The generated answer as a string.
    """
    groq_key = get_env('GROQ_API_KEY')
    openrouter_key = get_env('OPENROUTER_API_KEY')
    
    if groq_key:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        model_to_use = config['evaluation'].get('groq_model', 'llama-3.3-70b-versatile')
    elif openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        model_to_use = config['generation']['model']
    else:
        raise ValueError("Vui lòng set GROQ_API_KEY hoặc OPENROUTER_API_KEY")

    user_prompt = build_rag_prompt(query, contexts)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config['generation']['temperature'],
                max_tokens=config['generation']['max_tokens']
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise e
            sleep_time = 2 ** attempt
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)


def generate_synthetic_qa_batch(prompt: str, max_retries: int = 2) -> dict:
    """
    Generate a batch of QA pairs using either Gemini (via OpenAI compatibility) or OpenRouter.
    """
    gemini_key = get_env('GEMINI_API_KEY')
    openrouter_key = get_env('OPENROUTER_API_KEY')

    if gemini_key:
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key
        )
        model_name = "gemini-3.1-flash-lite"
    elif openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        model_name = config['generation']['model']
    else:
        raise ValueError("Vui lòng set GEMINI_API_KEY hoặc OPENROUTER_API_KEY")

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            if not gemini_key:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)

            output_text = response.choices[0].message.content
            clean_json = strip_markdown_json(output_text)
            return json.loads(clean_json)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise ValueError("Response was not valid JSON after retries")
        except Exception as e:
            logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise e

        sleep_time = (2 ** attempt) + (4 if gemini_key else 0)
        logger.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
