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
Given a complex question, break it down into 2-3 independent sub-questions that can each
be answered by searching a financial news database independently.

Rules:
1. If the question is already simple (single entity, single fact), return it as-is in a single-element array.
2. Each sub-question must be self-contained and understandable without context from other sub-questions.
3. Keep sub-questions in Vietnamese.
4. Maximum 3 sub-questions.
5. Return ONLY a JSON array of strings, no markdown, no explanation.

Example:
Input: "So sánh lợi nhuận của Vietcombank và BIDV năm 2023, ngân hàng nào tăng trưởng mạnh hơn?"
Output: ["Lợi nhuận của Vietcombank năm 2023 là bao nhiêu?", "Lợi nhuận của BIDV năm 2023 là bao nhiêu?", "Tăng trưởng lợi nhuận ngân hàng năm 2023"]

Example:
Input: "Lợi nhuận Vietcombank quý 1/2023 là bao nhiêu?"
Output: ["Lợi nhuận Vietcombank quý 1/2023 là bao nhiêu?"]"""


def decompose_query(question: str) -> list[str]:
    """
    Decompose a complex question into sub-queries using Gemini Flash Lite.

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

        response = client.chat.completions.create(
            model="gemma-4-31b-it",
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]  # remove first line
            raw = raw.rsplit("```", 1)[0]  # remove closing fence
            raw = raw.strip()

        sub_queries = json.loads(raw)

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
