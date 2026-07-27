"""
evaluate_aio.py — Unified AIO Evaluation for All Models
=========================================================
Evaluates any model (gemini, mistral, gemma) using AIO Prompt.
Judge: Groq Llama 3.3 70B
"""

import os
import sys
import argparse
import time
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env

logger = setup_logger("AIOEval")
config = load_config()

# ---------------------------------------------------------------------------
# All-in-One Prompt and prompt templates
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

def atomic_save_csv(df: pd.DataFrame, file_path: Path):
    temp_path = file_path.with_suffix(".csv.tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        if temp_path.exists():
            os.replace(temp_path, file_path)
            logger.info(f"[Atomic Save] Report successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"[Atomic Save ERROR] Failed to save file safely: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def build_llm_judge():
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set!")

    from langchain_groq import ChatGroq

    judge_llm = ChatGroq(
        model=config["evaluation"].get("groq_model", "llama-3.3-70b-versatile"),
        groq_api_key=groq_key,
        temperature=0.0,
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    return judge_llm


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
        logger.warning(f"Failed to decode judge JSON response: {e}. Raw string: {raw_text}")
        return {
            "faithfulness_score": 0.0,
            "faithfulness_reasoning": f"Failed to parse JSON: {str(e)}",
            "answer_relevancy_score": 0.0,
            "answer_relevancy_reasoning": f"Failed to parse JSON: {str(e)}",
            "context_recall_score": 0.0,
            "context_recall_reasoning": f"Failed to parse JSON: {str(e)}"
        }

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
                raise ValueError(f"JSON output did not match expected schema: {raw_text[:200]}")
                
            return parsed_result
        except Exception as e:
            err_str = str(e).lower()
            logger.warning(f"[Judge API] Attempt {attempt + 1}/{max_retries} failed. Details: {e}")
            
            # Detect critical rate limit (Daily Quota) or Out of Funds
            if "limit 100000" in err_str or "tokens per day" in err_str:
                logger.error("!!! CRITICAL: Groq daily token limit reached. Stopping immediately.")
                sys.exit(1)
            if "402" in err_str or "out of credits" in err_str:
                logger.error("!!! CRITICAL: Out of Credits/Funds. Stopping immediately.")
                sys.exit(1)
                
            if "429" in err_str or "rate limit" in err_str or "tpm" in err_str or "rpm" in err_str:
                sleep_time = 30 + 10 * attempt
                logger.info(f"[Judge API] Rate limit detected. Sleeping {sleep_time}s before retry...")
                time.sleep(sleep_time)
            else:
                sleep_time = 5 * (attempt + 1)
                logger.info(f"[Judge API] General error. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                
    raise RuntimeError("All 3 API call attempts failed. Possible network error or extended rate limit. Stopping to preserve checkpoint data.")

def main():
    parser = argparse.ArgumentParser(description="Unified AIO Prompt Evaluation for All Models")
    parser.add_argument("--model", type=str, required=True, choices=["mistral", "gemma", "gemini"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=15)
    args = parser.parse_args()

    OUTPUT_DIR = ROOT / config["evaluation"]["output_dir"]
    backend = get_env("VECTOR_STORE_BACKEND", config["vector_store"]["backend"]).lower()

    if args.model == "gemini":
        INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{backend}.parquet"
        REPORT_PATH = OUTPUT_DIR / f"evaluation_report_aio_{backend}.csv"
    else:
        INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{args.model}.parquet"
        REPORT_PATH = OUTPUT_DIR / f"evaluation_report_aio_{args.model}.csv"

    if not INPUT_PARQUET.exists():
        logger.error("Generation results not found at: %s", INPUT_PARQUET)
        sys.exit(1)

    df_eval = pd.read_parquet(INPUT_PARQUET)
    records = df_eval.to_dict("records")
    if args.limit: records = records[:args.limit]

    existing_df = None
    existing_qs = set()
    if REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(REPORT_PATH)
            completed_df = existing_df[existing_df["faithfulness_score"].notna()]
            existing_qs = set(completed_df["question"].astype(str).str.strip().tolist())
            logger.info("Found %d existing records.", len(existing_qs))
        except Exception as e:
            logger.warning("Could not read report: %s", e)

    pending = [r for r in records if r["question"].strip() not in existing_qs]
    logger.info("Total pending: %d", len(pending))
    if not pending: return

    try:
        judge_llm = build_llm_judge()
    except Exception as e:
        logger.error("Failed to init LLM Judge: %s", e)
        sys.exit(1)

    new_results = []
    total_processed = 0

    for idx, record in enumerate(pending):
        question = record["question"].strip()
        gt = record.get("ground_truth", "")
        retrieved_contexts_str = record.get("retrieved_context", "")
        generated_answer = record.get("generated_answer", "")

        logger.info("--- Sample %d/%d ---", idx + 1, len(pending))
        
        try:
            scores = call_judge_with_retry(
                judge_llm=judge_llm,
                question=question,
                ground_truth=gt,
                retrieved_contexts_str=retrieved_contexts_str,
                generated_answer=generated_answer
            )
        except Exception as e:
            logger.error("Fatal error: %s", e)
            sys.exit(1)

        logger.info("Faithfulness: %.4f | Relevancy: %.4f | Recall: %.4f", 
                    scores["faithfulness_score"], scores["answer_relevancy_score"], scores["context_recall_score"])

        res_row = {
            "question": question,
            "ground_truth": gt,
            "rag_answer": generated_answer,
            "retrieved_context": retrieved_contexts_str,
            "faithfulness_score": scores["faithfulness_score"],
            "faithfulness_reasoning": scores["faithfulness_reasoning"],
            "answer_relevancy_score": scores["answer_relevancy_score"],
            "answer_relevancy_reasoning": scores["answer_relevancy_reasoning"],
            "context_recall_score": scores["context_recall_score"],
            "context_recall_reasoning": scores["context_recall_reasoning"]
        }
        new_results.append(res_row)
        total_processed += 1

        if total_processed % 5 == 0 or idx == len(pending) - 1:
            df_new = pd.DataFrame(new_results)
            df_combined = pd.concat([existing_df, df_new], ignore_index=True) if existing_df is not None else df_new
            atomic_save_csv(df_combined, REPORT_PATH)
            existing_df = df_combined
            new_results = []

        if idx < len(pending) - 1:
            time.sleep(args.cooldown)

if __name__ == "__main__":
    main()
