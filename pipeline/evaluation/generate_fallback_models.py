"""
generate_fallback_models.py — Fallback Generation Pipeline
==========================================================
Generates answers using fallback models (Mistral or Gemma) for the 150 QA pairs.
Reuses the retrieved contexts from generation_results_qdrant.parquet (same context
as Gemini, ensuring fair comparison). Records performance metrics in the output file.

Output schema (must match generation_results_qdrant.parquet exactly):
  question, ground_truth, retrieved_context, generated_answer,
  doc_id, strategy, method, retrieved_chunk_ids,
  model_name, latency_s, input_tokens, output_tokens, total_tokens,
  throughput_tps, retrieval_time_s, e2e_latency_s, error

Usage:
  python evaluation/generate_fallback_models.py --model mistral
  python evaluation/generate_fallback_models.py --model gemma
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env
from src.generation import build_rag_prompt, RAG_SYSTEM_PROMPT

logger = setup_logger("FallbackGen")
config = load_config()

def atomic_save_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        logger.info("[Checkpoint] Saved %d rows -> %s", len(df), path)
    except Exception as exc:
        logger.error("[Checkpoint] Save failed: %s", exc)
        if tmp.exists():
            tmp.unlink()

def atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info("[Checkpoint] Saved %d rows -> %s", len(df), path)
    except Exception as exc:
        logger.error("[Checkpoint] Save failed: %s", exc)
        if tmp.exists():
            tmp.unlink()

def main():
    parser = argparse.ArgumentParser(description="Generate fallback RAG answers")
    parser.add_argument("--model", type=str, required=True, choices=["mistral", "gemma"], help="Fallback model to use")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run")
    args = parser.parse_args()

    fallback_conf = config.get("fallback_generation", {}).get(args.model)
    if not fallback_conf:
        logger.error("Configuration for model '%s' not found in fallback_generation", args.model)
        sys.exit(1)

    model_name = fallback_conf["model"]
    api_base = fallback_conf["api_base"]
    cooldown = fallback_conf.get("cooldown_between_calls", 2)
    max_tokens = fallback_conf.get("max_tokens", 1024)
    temperature = fallback_conf.get("temperature", 0.2)

    api_key_env_var = "MISTRAL_API_KEY" if args.model == "mistral" else "GEMINI_API_KEY"
    api_key = get_env(api_key_env_var)

    if not api_key:
        logger.error("Environment variable %s is not set!", api_key_env_var)
        sys.exit(1)

    OUTPUT_DIR = ROOT / config["evaluation"]["output_dir"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH = OUTPUT_DIR / f"generation_results_{args.model}.parquet"
    CSV_PATH = OUTPUT_DIR / f"generation_results_{args.model}.csv"

    # Reuse Qdrant generation results for the contexts
    backend = get_env("VECTOR_STORE_BACKEND", "qdrant").lower()
    BASE_PARQUET = OUTPUT_DIR / f"generation_results_{backend}.parquet"
    if not BASE_PARQUET.exists():
        logger.error("Base generation file not found: %s", BASE_PARQUET)
        sys.exit(1)

    df_base = pd.read_parquet(BASE_PARQUET)
    all_qa = df_base.to_dict("records")
    
    if args.limit:
        all_qa = all_qa[:args.limit]

    logger.info("Loaded %d QA samples for generation using %s", len(all_qa), model_name)

    records_map = {}
    if PARQUET_PATH.exists():
        try:
            existing_df = pd.read_parquet(PARQUET_PATH)
            for _, row in existing_df.iterrows():
                q_key = str(row["question"]).strip()
                records_map[q_key] = row.to_dict()
            logger.info("Found existing checkpoint with %d records.", len(records_map))
        except Exception as e:
            logger.warning("Could not read existing parquet checkpoint: %s. Starting fresh.", e)

    client = OpenAI(base_url=api_base, api_key=api_key)
    
    pending_qs = []
    for qa in all_qa:
        q_text = str(qa["question"]).strip()
        if q_text not in records_map:
            pending_qs.append(qa)

    logger.info("Pending samples to process: %d", len(pending_qs))
    if not pending_qs:
        logger.info("All samples processed. Exiting.")
        return

    for idx, qa in enumerate(pending_qs):
        query = str(qa["question"]).strip()
        gt = qa.get("ground_truth", "")
        retrieved_context_str = qa.get("retrieved_context", "")
        
        logger.info("--- Sample %d/%d ---", idx + 1, len(pending_qs))
        
        # Build prompt using the loaded retrieved contexts
        contexts = retrieved_context_str.split("\n---\n") if retrieved_context_str else []
        user_prompt = build_rag_prompt(query, contexts)
        
        ans = ""
        latency = 0.0
        in_tokens = 0
        out_tokens = 0
        tot_tokens = 0
        error_msg = ""
        
        for attempt in range(4):
            try:
                start_time = time.perf_counter()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                latency = time.perf_counter() - start_time
                ans = response.choices[0].message.content.strip()
                
                if response.usage:
                    in_tokens = getattr(response.usage, "prompt_tokens", 0)
                    out_tokens = getattr(response.usage, "completion_tokens", 0)
                    tot_tokens = getattr(response.usage, "total_tokens", 0)
                break
                
            except Exception as e:
                err_str = str(e).lower()
                logger.warning("[Attempt %d/4] API Error: %s", attempt + 1, e)
                
                if "insufficient" in err_str or "quota" in err_str or "402" in err_str:
                    logger.error("CRITICAL: Out of funds or quota exceeded. Stopping immediately.")
                    error_msg = "Quota exceeded"
                    sys.exit(1)
                    
                if attempt < 3:
                    wait_time = 15 * (2 ** attempt)
                    if "429" in err_str or "too many requests" in err_str:
                        logger.info("Rate limit hit. Waiting %ds...", wait_time)
                    else:
                        logger.info("Unknown error. Waiting %ds...", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("All attempts failed for this sample.")
                    error_msg = str(e)
                    
        throughput = (out_tokens / latency) if latency > 0 and out_tokens > 0 else 0.0
        
        # Column order must exactly match the updated schema from Gemini
        records_map[query] = {
            "question":            query,
            "ground_truth":        gt,
            "retrieved_context":   retrieved_context_str,
            "generated_answer":    ans,
            "doc_id":              qa.get("doc_id", ""),
            "strategy":            qa.get("strategy", ""),
            "method":              qa.get("method", ""),
            "retrieved_chunk_ids": qa.get("retrieved_chunk_ids", ""),
            "model_name":          model_name,
            "latency_s":           latency,
            "input_tokens":        in_tokens,
            "output_tokens":       out_tokens,
            "total_tokens":        tot_tokens,
            "throughput_tps":      throughput,
            "retrieval_time_s":    0.0,   # Will be filled by benchmark_latency.py
            "e2e_latency_s":       0.0,   # Will be filled by benchmark_latency.py
            "error":               error_msg,
        }
        
        logger.info("Latency: %.2fs | TPS: %.1f | Tokens: %d out", latency, throughput, out_tokens)
        
        if (idx + 1) % 5 == 0 or idx == len(pending_qs) - 1:
            df = pd.DataFrame(list(records_map.values()))
            atomic_save_parquet(df, PARQUET_PATH)
            atomic_save_csv(df, CSV_PATH)
            
        time.sleep(cooldown)

    logger.info("Generation complete for model %s. Results saved to %s", model_name, PARQUET_PATH)

if __name__ == "__main__":
    main()
