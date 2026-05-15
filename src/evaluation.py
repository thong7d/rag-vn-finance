"""
evaluation.py — Reference metrics and LLM-as-Judge evaluation for Phase 8.

Implements two evaluation layers:
  Layer 1 — Traditional Metrics (no API calls required)
    - ROUGE-L       : lexical overlap (rouge_score library)
    - BERTScore     : semantic similarity using bert-base-multilingual-cased
                      (lang="vi" is set explicitly to avoid mislabeling Vietnamese)

  Layer 2 — LLM-as-Judge (Groq API, llama-3.3-70b-versatile)
    - Faithfulness   : Does the answer stay faithful to the retrieved context?
    - Answer Relevance: Does the answer actually address the question asked?

    Each dimension returns {"score": int (1-5), "reasoning": str}.
    Mandatory strip_markdown_json() is called before json.loads() to
    handle LLM outputs wrapped in ``` fences.

Rate-limit safety (REQUIRED):
  - Global Cooldown: time.sleep(GROQ_COOLDOWN) between every LLM judge call.
  - Checkpoint: results saved to disk every CHECKPOINT_EVERY rows.
  - Both constants are configurable at call time.

References:
  Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench"
  https://arxiv.org/abs/2306.05685
"""

import json
import logging
import re
import time
from typing import Optional

from openai import OpenAI

from src.utils import get_env, load_config, setup_logger

logger = setup_logger(__name__)
config = load_config()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL = config["evaluation"].get("groq_model", "llama-3.3-70b-versatile")
GROQ_COOLDOWN = 20       # seconds between every Groq judge call (RPM safety)
CHECKPOINT_EVERY = 20    # flush parquet to disk every N evaluated rows
MAX_RETRIES = 2          # retries on API / JSON parse failure per dimension


# ---------------------------------------------------------------------------
# Helper: Markdown fence stripper (mandatory before json.loads)
# ---------------------------------------------------------------------------

def strip_markdown_json(text: str) -> str:
    """
    Remove markdown code fences (```json ... ``` or ``` ... ```) from LLM output.

    This is MANDATORY before calling json.loads() because open-source LLMs
    frequently wrap their JSON output in markdown code fences, causing
    json.JSONDecodeError on a valid-looking response.

    Args:
        text: Raw LLM output string.

    Returns:
        Cleaned string ready for json.loads().
    """
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Layer 1 — Traditional metrics
# ---------------------------------------------------------------------------

def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between a generated answer and a reference answer.

    Args:
        prediction: Generated answer string.
        reference:  Ground-truth answer string.

    Returns:
        ROUGE-L F1 score (float in [0, 1]).
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, prediction)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception as e:
        logger.warning(f"ROUGE-L computation failed: {e}")
        return 0.0


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for a list of (prediction, reference) pairs.

    IMPORTANT: Uses lang="vi" and model_type="bert-base-multilingual-cased"
    to ensure correct tokenisation and scoring for Vietnamese text.
    Using the default English model would produce incorrect results for
    Vietnamese morphology and word boundaries.

    Args:
        predictions: List of generated answer strings.
        references:  List of ground-truth answer strings (same length).

    Returns:
        List of BERTScore F1 floats (one per pair).
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(
            cands=predictions,
            refs=references,
            lang="vi",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return [0.0] * len(predictions)


# ---------------------------------------------------------------------------
# Layer 2 — LLM-as-Judge (Groq, llama-3.3-70b-versatile)
# ---------------------------------------------------------------------------

# ── Judge prompt templates ──────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an expert evaluator for a Vietnamese financial news RAG (Retrieval-Augmented Generation) system.

Your task: Evaluate whether the GENERATED ANSWER is FAITHFUL to the RETRIEVED CONTEXT.
A faithful answer uses ONLY information present in the context — it does not hallucinate facts, numbers, or entities.

Scoring rubric (1–5):
5 — Fully faithful. Every claim in the answer can be directly traced to the context.
4 — Mostly faithful. Minor paraphrase but no fabricated facts.
3 — Partially faithful. Some claims match context; others are unverifiable.
2 — Mostly unfaithful. Significant fabricated content despite some context overlap.
1 — Completely unfaithful. The answer ignores or contradicts the context.

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Respond with a JSON object ONLY — no markdown, no preamble:
{{"score": <int 1-5>, "reasoning": "<one concise English sentence explaining your score>"}}"""

_ANSWER_RELEVANCE_PROMPT = """\
You are an expert evaluator for a Vietnamese financial news RAG system.

Your task: Evaluate whether the GENERATED ANSWER is RELEVANT to the QUESTION.
A relevant answer directly addresses what was asked — not a tangent or a generic response.

Scoring rubric (1–5):
5 — Perfectly relevant. The answer directly and completely addresses the question.
4 — Mostly relevant. Minor off-topic content but the core question is answered.
3 — Partially relevant. The answer touches on the topic but misses key aspects.
2 — Mostly irrelevant. The answer is mostly off-topic or generic.
1 — Completely irrelevant. The answer does not address the question at all.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

Respond with a JSON object ONLY — no markdown, no preamble:
{{"score": <int 1-5>, "reasoning": "<one concise English sentence explaining your score>"}}"""


def _call_groq_judge(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    Send a judge prompt to Groq and return the parsed JSON dict.

    Applies strip_markdown_json() before json.loads() — MANDATORY.
    Retries up to max_retries times on API or JSON parse failures.
    Does NOT sleep here; the caller is responsible for Global Cooldown.

    Args:
        prompt:      Fully-formatted judge prompt string.
        max_retries: Number of retry attempts on failure.

    Returns:
        dict with keys "score" (int) and "reasoning" (str).
        Returns {"score": 0, "reasoning": "ERROR: <msg>"} on total failure.
    """
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set. Cannot run LLM-as-Judge.")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # deterministic scoring
                max_tokens=256,
            )
            raw_text = response.choices[0].message.content
            clean_text = strip_markdown_json(raw_text)   # mandatory fence strip
            result = json.loads(clean_text)

            # Validate expected keys
            if "score" not in result or "reasoning" not in result:
                raise ValueError(f"Missing keys in judge response: {result}")

            result["score"] = int(result["score"])
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[Judge] JSON parse failed (attempt {attempt + 1}): {e}")
            logger.warning(f"[Judge] Raw output was: {raw_text!r}")
        except Exception as e:
            logger.warning(f"[Judge] API call failed (attempt {attempt + 1}): {e}")
            err_str = str(e).lower()
            if '429' in err_str or 'rate limit' in err_str or 'tokens per day' in err_str:
                # Ép hệ thống ném thẳng lỗi này ra cho Notebook xử lý, không nuốt lỗi nữa
                raise RuntimeError(f"FATAL RATE LIMIT: {e}")

        if attempt < max_retries:
            sleep_time = 2 ** attempt
            logger.info(f"[Judge] Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    return {"score": 0, "reasoning": f"ERROR: All {max_retries + 1} attempts failed."}


def evaluate_faithfulness(context: str, answer: str) -> dict:
    """
    Score how faithful the generated answer is to the retrieved context.

    Args:
        context: Concatenated retrieved passage text.
        answer:  Generated answer string.

    Returns:
        {"score": int (1-5), "reasoning": str}
    """
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:3000],   # truncate to avoid exceeding context window
        answer=answer[:1000],
    )
    return _call_groq_judge(prompt)


def evaluate_answer_relevance(question: str, answer: str) -> dict:
    """
    Score how relevant the generated answer is to the question.

    Args:
        question: The user question string.
        answer:   Generated answer string.

    Returns:
        {"score": int (1-5), "reasoning": str}
    """
    prompt = _ANSWER_RELEVANCE_PROMPT.format(
        question=question[:500],
        answer=answer[:1000],
    )
    return _call_groq_judge(prompt)


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------

def evaluate_single_row(
    question: str,
    ground_truth: str,
    retrieved_context: str,
    generated_answer: str,
    cooldown: int = GROQ_COOLDOWN,
) -> dict:
    """
    Run ALL evaluation metrics for a single QA result row.

    Layer 1 (no API): ROUGE-L, BERTScore (computed in batch externally,
      but this function accepts pre-computed values via direct call in
      the notebook; for single-row convenience it computes ROUGE-L only).
    Layer 2 (Groq API): Faithfulness, Answer Relevance.

    Global Cooldown: sleeps `cooldown` seconds BETWEEN the two Groq calls
    to avoid hitting the Groq free-tier RPM (30 req/min) limit.

    Args:
        question:           The evaluation question.
        ground_truth:       Reference answer from the test set.
        retrieved_context:  Concatenated retrieved chunk texts.
        generated_answer:   Answer produced by the RAG pipeline.
        cooldown:           Seconds to sleep between Groq judge calls.

    Returns:
        dict with keys:
          rouge_l, faithfulness_score, faithfulness_reasoning,
          answer_relevance_score, answer_relevance_reasoning
    """
    result = {}

    # ── Layer 1: ROUGE-L (no API) ──────────────────────────────────────────
    result["rouge_l"] = compute_rouge_l(generated_answer, ground_truth)

    # ── Layer 2a: Faithfulness ────────────────────────────────────────────
    faith = evaluate_faithfulness(retrieved_context, generated_answer)
    result["faithfulness_score"]     = faith.get("score", 0)
    result["faithfulness_reasoning"] = faith.get("reasoning", "")

    # Global Cooldown between two consecutive Groq calls
    logger.info(f"[Cooldown] Sleeping {cooldown}s between judge calls...")
    time.sleep(cooldown)

    # ── Layer 2b: Answer Relevance ────────────────────────────────────────
    relevance = evaluate_answer_relevance(question, generated_answer)
    result["answer_relevance_score"]     = relevance.get("score", 0)
    result["answer_relevance_reasoning"] = relevance.get("reasoning", "")

    return result
