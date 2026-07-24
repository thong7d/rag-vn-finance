"""
generation.py — 3-layer LLM generation with SSE streaming for the FastAPI backend.

Fallback order:
  1. Gemini (gemini-3.1-flash-lite)
  2. Mistral (mistral-small-2506 via Mistral API)
  3. Gemma (gemma-4-31b-it via Google AI Studio)

Each layer streams tokens as SSE events to the client.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("GenerationService")

# ── System Prompt (mirrors pipeline/src/generation.py) ──────────────────
RAG_SYSTEM_PROMPT = """Bạn là một chuyên viên phân tích tài chính chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng **dựa hoàn toàn** vào các đoạn ngữ cảnh báo chí tài chính được cung cấp bên dưới.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp. Không bịa đặt số liệu hoặc sự kiện.
2. Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ: "Thông tin trong ngữ cảnh không đủ để trả lời câu hỏi này."
3. Trả lời ngắn gọn, chính xác bằng tiếng Việt (2–4 câu là đủ cho hầu hết câu hỏi).
4. Nếu câu hỏi liên quan đến mã chứng khoán hoặc tổ chức (ví dụ: VIC, HPG, Vietcombank), hãy trích dẫn chính xác mã/tên đó trong câu trả lời.
5. Khi trích dẫn số liệu tài chính (lãi suất, doanh thu, lợi nhuận,...), luôn đi kèm thời điểm nếu có trong ngữ cảnh."""


def _build_user_prompt(question: str, contexts: list[str]) -> str:
    if not contexts:
        context_block = "[Không có ngữ cảnh nào được truy xuất.]"
    else:
        context_block = "\n\n---\n\n".join(
            [f"[Đoạn {i+1}]:\n{ctx.strip()}" for i, ctx in enumerate(contexts) if ctx.strip()]
        )
    return (
        f"Dưới đây là các đoạn thông tin được truy xuất từ kho dữ liệu báo chí tài chính Việt Nam:\n\n"
        f"{context_block}\n\n---\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời câu hỏi dựa vào các đoạn ngữ cảnh trên."
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_answer(question: str, sources: list[dict]) -> AsyncGenerator[str, None]:
    """
    SSE generator that:
      1. Yields 'sources' event immediately with retrieved passages.
      2. Tries Gemini → Mistral → Gemma in order, streaming 'token' events.
      3. Yields 'done' event with the full concatenated answer.
      4. On total failure, yields 'error' event.

    Each event follows the format:
        event: <type>
        data: <json>
    """
    settings = get_settings()
    contexts = [s["text"] for s in sources if s.get("text")]
    user_prompt = _build_user_prompt(question, contexts)

    # ── Send sources first ────────────────────────────────────────────────────
    yield _sse("sources", {"sources": sources})

    # ── Layer definitions ─────────────────────────────────────────────────────
    layers = [
        {
            "name": "Gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": settings.gemini_api_key,
            "model": "gemini-3.1-flash-lite",
        },
        {
            "name": "Mistral",
            "base_url": "https://api.mistral.ai/v1",
            "api_key": settings.mistral_api_key,
            "model": "mistral-small-2506",
        },
        {
            "name": "Gemma",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": settings.gemini_api_key,   # Google AI Studio for Gemma
            "model": "gemma-3-27b-it",
        },
    ]

    full_answer = ""
    last_error = ""

    for layer in layers:
        try:
            logger.info(f"Trying {layer['name']} ({layer['model']})...")
            client = AsyncOpenAI(base_url=layer["base_url"], api_key=layer["api_key"])

            stream = await client.chat.completions.create(
                model=layer["model"],
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.generation_temperature,
                max_tokens=settings.generation_max_tokens,
                stream=True,
            )

            full_answer = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_answer += delta
                    yield _sse("token", {"token": delta, "model": layer["name"]})

            if full_answer.strip():
                yield _sse("done", {"full_answer": full_answer, "model": layer["name"]})
                logger.info(f"{layer['name']} succeeded — {len(full_answer)} chars")
                return

            logger.warning(f"{layer['name']} returned empty response, trying next layer...")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"{layer['name']} failed: {e}")
            # Brief pause before trying next layer
            await __import__("asyncio").sleep(1.5)

    # All layers exhausted
    yield _sse("error", {
        "message": "Tất cả mô hình sinh câu trả lời đều không phản hồi. Vui lòng thử lại sau.",
        "detail": last_error,
    })
