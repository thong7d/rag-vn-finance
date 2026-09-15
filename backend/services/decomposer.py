"""
decomposer.py — Query decomposition service using Gemini Flash Lite.

Decomposes complex multi-hop questions into 2-3 independent sub-queries
for parallel retrieval. Falls back to the original question if decomposition
fails or the question is already simple.
"""

import json

from openai import OpenAI

from core.config import get_settings
from core.logging import setup_logger

logger = setup_logger("Decomposer")

DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition expert for a Vietnamese financial news RAG system.

Your job: decide whether a question needs to be broken into 2-3 independent sub-questions for parallel retrieval.

DECOMPOSE when the question involves ANY of:
- Comparing 2+ entities (companies, banks, products, indices)
- Asking about 2+ time periods (year-over-year, quarterly trends)
- Asking about both cause AND effect of an event
- Multiple metrics about the same entity (e.g., both revenue AND profit)

DO NOT DECOMPOSE when:
- The question asks for exactly ONE fact about ONE entity at ONE time
- It is already a simple lookup question

Return ONLY a JSON array of strings. No markdown, no explanation.

Examples:

Input: "So sánh lợi nhuận của Vietcombank và BIDV năm 2023, ngân hàng nào tăng trưởng mạnh hơn?"
Output: ["Lợi nhuận sau thuế của Vietcombank (VCB) năm 2023 là bao nhiêu?", "Lợi nhuận sau thuế của BIDV năm 2023 là bao nhiêu?", "Tăng trưởng lợi nhuận ngân hàng thương mại Việt Nam năm 2023"]

Input: "Doanh thu và lợi nhuận của Tập đoàn Hòa Phát thay đổi như thế nào giữa năm 2022 và 2023?"
Output: ["Doanh thu và lợi nhuận Hòa Phát năm 2022 là bao nhiêu?", "Doanh thu và lợi nhuận Hòa Phát năm 2023 là bao nhiêu?"]

Input: "Sự cố xử lý sai phạm tại Vạn Thịnh Phát ảnh hưởng thế nào đến thị trường trái phiếu và Ngân hàng SCB?"
Output: ["Vụ sai phạm của Tập đoàn Vạn Thịnh Phát là gì?", "Tác động của vụ Vạn Thịnh Phát đến thị trường trái phiếu doanh nghiệp", "Tình trạng Ngân hàng SCB sau vụ Vạn Thịnh Phát"]

Input: "Tỷ lệ nợ xấu và bao phủ nợ xấu của VPBank và Techcombank năm 2023 là bao nhiêu?"
Output: ["Tỷ lệ nợ xấu và bao phủ nợ xấu của VPBank năm 2023", "Tỷ lệ nợ xấu và bao phủ nợ xấu của Techcombank năm 2023"]

Input: "Lợi nhuận Vietcombank quý 1/2023 là bao nhiêu?"
Output: ["Lợi nhuận Vietcombank quý 1/2023 là bao nhiêu?"]"""


import re


def _extract_json_array(raw: str) -> str:
    """
    Robustly extract a JSON array string from a model response that may contain:
    - <think>...</think> reasoning blocks (Gemma 4 / Qwen reasoning models)
    - ```json ... ``` markdown code fences
    - Surrounding explanation text

    Returns the extracted JSON array string, or raises ValueError if not found.
    """
    # 1. Strip <think>...</think> reasoning block
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # 2. Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # 3. Regex: find the first [...] array in the output (handles extra prose)
    array_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if array_match:
        return array_match.group(0)

    # 4. Last resort: return as-is (will raise JSONDecodeError if invalid)
    return raw.strip()


def decompose_query(question: str) -> list[str]:
    """
    Decompose a complex question into sub-queries using Gemma 4 31B.

    Returns a list of 1-3 sub-query strings.
    Falls back to [question] on any failure.
    """
    settings = get_settings()
    max_sub = settings.decompose_max_subqueries

    try:
        api_key = settings.gemini_api_key_2 if settings.gemini_api_key_2 else settings.gemini_api_key
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )

        # Merge system prompt into user message: Gemma 4 follows user role more reliably
        # than a separate system role when using the Google AI Studio OpenAI-compat endpoint.
        merged_content = f"{DECOMPOSE_SYSTEM_PROMPT}\n\nInput: \"{question}\"\nOutput:"

        response = client.chat.completions.create(
            model="gemma-4-31b-it",
            messages=[
                {"role": "user", "content": merged_content},
            ],
            temperature=0.4,   # Slightly creative to encourage decomposition decisions
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        logger.debug(f"Decomposer raw output: {raw[:200]}")

        sub_queries = json.loads(_extract_json_array(raw))

        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            logger.warning(f"Decomposer returned invalid format, falling back to original: {raw}")
            return [question]

        # Cap to max sub-queries
        sub_queries = [str(q).strip() for q in sub_queries[:max_sub] if str(q).strip()]

        if not sub_queries:
            return [question]

        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        return sub_queries

    except Exception as e:
        logger.warning(f"Decomposition failed ({e}), falling back to original question")
        return [question]
