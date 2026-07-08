"""
evaluate_ragas_pure.py — Official Ragas Framework Evaluation
=============================================================

Evaluates the Vietnamese Financial News RAG system using the pure ragas library.

Key Features & Enhancements (Production-Ready):
  1. Rate Limit & Burst Prevention: Enforces sequential request evaluation using 
     Ragas RunConfig(max_workers=1). Avoids OpenRouter 429 errors.
  2. Batch-Based Dataset Processing: Groups evaluation samples into batches of 
     size N (default: 5).
  3. Uses pre-generated answers from Phase 7 (generation_results_{backend}.parquet).

LLM Judge   : qwen/qwen3-32b via OpenRouter (OpenAI-compatible endpoint)
Embeddings  : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local)
Cooldown    : 10 s between metric evaluations — safe for 60 RPM budget
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# ==============================================================================
# BẮT ĐẦU VÁ LỖI (MONKEY PATCH) CHO RAGAS 0.4.x VÀ LANGCHAIN-COMMUNITY
# ==============================================================================
import sys
from types import ModuleType

if 'langchain_community.chat_models.vertexai' not in sys.modules:
    dummy_vertexai = ModuleType('langchain_community.chat_models.vertexai')
    dummy_vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {}) 
    sys.modules['langchain_community.chat_models.vertexai'] = dummy_vertexai
    
if 'langchain_community.llms.vertexai' not in sys.modules:
    dummy_llms_vertexai = ModuleType('langchain_community.llms.vertexai')
    dummy_llms_vertexai.VertexAI = type('VertexAI', (object,), {}) 
    sys.modules['langchain_community.llms.vertexai'] = dummy_llms_vertexai
# ==============================================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env

logger = setup_logger("RagasEval")
config = load_config()

backend = (get_env("VECTOR_STORE_BACKEND") or "qdrant").lower()
OUTPUT_DIR    = ROOT / config["evaluation"]["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{backend}.parquet"
REPORT_PATH   = OUTPUT_DIR / f"evaluation_report_pure_ragas_{backend}.csv"

DEFAULT_BATCH_SIZE = 5
DEFAULT_COOLDOWN   = 10

def is_out_of_funds_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    insufficient_triggers = [
        "insufficient_quota", "insufficient balance", "out of credit",
        "payment required", "billing", "credit limit", "insufficient funds"
    ]
    return any(trigger in err_str for trigger in insufficient_triggers)

def build_ragas_metrics():
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openrouter_key = get_env("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    judge_lc_llm = ChatOpenAI(
        model="qwen/qwen3-32b",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "https://github.com/thong7d/rag-vn-finance",
            "X-Title": "Vietnamese Financial RAG Eval"
        }
    )
    ragas_llm = LangchainLLMWrapper(judge_lc_llm)

    hf_embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embed = LangchainEmbeddingsWrapper(hf_embed)

    faithfulness.llm            = ragas_llm
    answer_relevancy.llm        = ragas_llm
    answer_relevancy.embeddings = ragas_embed
    answer_relevancy.strictness = 1
    context_recall.llm          = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_recall]
    logger.info("Ragas metrics initialised: %s", [m.name for m in metrics])
    return metrics

def atomic_save_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        logger.info("[Checkpoint] Saved %d rows → %s", len(df), path)
    except Exception as exc:
        logger.error("[Checkpoint] Save failed: %s", exc)
        tmp.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Ragas evaluation from Phase 7 outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    args = parser.parse_args()

    if not INPUT_PARQUET.exists():
        logger.error(f"Generation results not found at: {INPUT_PARQUET}")
        sys.exit(1)
        
    df_eval = pd.read_parquet(INPUT_PARQUET)
    all_records = df_eval.to_dict("records")
    logger.info("Loaded %d generated records from %s", len(all_records), INPUT_PARQUET)
    
    if args.limit:
        all_records = all_records[:args.limit]
        logger.info("Evaluation capped at %d samples.", args.limit)

    records_map = {}
    if REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(REPORT_PATH)
            for _, row in existing_df.iterrows():
                q_key = str(row["question"]).strip()
                row_dict = row.to_dict()
                for k in ["faithfulness", "answer_relevancy", "context_recall"]:
                    if k in row_dict and pd.isna(row_dict[k]):
                        row_dict[k] = None
                if "rag_answer" in row_dict and pd.isna(row_dict["rag_answer"]):
                    row_dict["rag_answer"] = None
                
                if "retrieved_contexts" in row_dict and pd.notna(row_dict["retrieved_contexts"]):
                    try:
                        row_dict["retrieved_contexts"] = json.loads(row_dict["retrieved_contexts"])
                    except Exception:
                        row_dict["retrieved_contexts"] = None
                else:
                    row_dict["retrieved_contexts"] = None
                    
                records_map[q_key] = row_dict
            logger.info("Resume mode: loaded %d existing records from CSV.", len(records_map))
        except Exception as exc:
            logger.warning("Could not read existing report (%s); starting fresh.", exc)

    for rec in all_records:
        q_key = rec["question"].strip()
        if q_key not in records_map:
            ctx_str = rec.get("retrieved_context", "")
            ctx_list = ctx_str.split("\n---\n") if ctx_str else []
            
            records_map[q_key] = {
                "question": q_key,
                "ground_truth": rec.get("ground_truth", ""),
                "rag_answer": rec.get("generated_answer", ""),
                "retrieved_contexts": ctx_list,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "error": "",
            }

    pending_questions = []
    for q_key, row in records_map.items():
        is_missing = (
            row.get("faithfulness") is None or
            row.get("answer_relevancy") is None or
            row.get("context_recall") is None
        )
        if is_missing:
            pending_questions.append(q_key)

    logger.info("Total pending questions to evaluate: %d", len(pending_questions))

    if not pending_questions:
        logger.info("Nothing to do. Exiting.")
        try:
            final_df = pd.read_csv(REPORT_PATH)
        except Exception:
            final_df = pd.DataFrame(list(records_map.values()))
        _print_summary(final_df)
        return

    try:
        metrics = build_ragas_metrics()
    except Exception as exc:
        logger.error("Failed to initialise Ragas metrics: %s", exc)
        sys.exit(1)

    from ragas import evaluate
    try:
        from ragas import RunConfig
    except ImportError:
        from ragas.run_config import RunConfig

    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    run_config = RunConfig(max_workers=1, timeout=180)

    batch_size = args.batch_size
    num_batches = (len(pending_questions) - 1) // batch_size + 1

    for b_idx in range(0, len(pending_questions), batch_size):
        batch_qs = pending_questions[b_idx : b_idx + batch_size]
        logger.info("==========================================")
        logger.info("Processing Batch %d/%d (Size: %d) | Backend: %s", (b_idx // batch_size) + 1, num_batches, len(batch_qs), backend.upper())
        logger.info("==========================================")

        samples = []
        batch_valid_keys = []

        for q_key in batch_qs:
            row = records_map[q_key]
            ground_truth = row.get("ground_truth", "")
            ans = row.get("rag_answer", "")
            contexts = row.get("retrieved_contexts", [])
            
            sample = SingleTurnSample(
                user_input=q_key,
                response=ans,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            samples.append(sample)
            batch_valid_keys.append(q_key)

        for metric in metrics:
            metric_samples = []
            metric_keys = []
            
            for sample, q_key in zip(samples, batch_valid_keys):
                if records_map[q_key].get(metric.name) is None:
                    metric_samples.append(sample)
                    metric_keys.append(q_key)

            if not metric_samples:
                continue

            logger.info("Evaluating metric '%s' on %d samples...", metric.name, len(metric_samples))
            
            dataset = EvaluationDataset(samples=metric_samples)
            
            score_success = False
            for attempt in range(3):
                try:
                    result = evaluate(dataset=dataset, metrics=[metric], run_config=run_config)
                    df_res = result.to_pandas()
                    
                    if df_res.empty or df_res[metric.name].isna().all():
                        raise RuntimeError(f"Ragas trả về toàn bộ NaN cho metric {metric.name}. Khả năng cao do lỗi API (429/402) bị thư viện ẩn đi.")
                        
                    for idx, res_row in df_res.iterrows():
                        target_key = metric_keys[idx]
                        score_val = res_row.get(metric.name)
                        
                        if score_val is not None and not pd.isna(score_val):
                            records_map[target_key][metric.name] = float(score_val)
                            logger.info("  Q: %s... -> %s = %.4f", target_key[:30], metric.name, score_val)
                        else:
                            logger.warning("  Q: %s... -> %s is NaN", target_key[:30], metric.name)

                    score_success = True
                    break
                except Exception as exc:
                    err = str(exc)
                    logger.warning("[Metric %s failed, attempt %d/3] %s", metric.name, attempt + 1, err)
                    if is_out_of_funds_error(exc):
                        logger.error("!!! CRITICAL: OUT OF FUNDS DETECTED DURING RAGAS EVALUATION !!! Exiting immediately.")
                        sys.exit(1)
                    if attempt < 2:
                        if any(tok in err.lower() for tok in ("429", "rate limit", "tpm", "rpm")):
                            wait = 30 + 15 * attempt
                        else:
                            wait = 5 * (attempt + 1)
                        logger.info("Waiting %ds before retrying...", wait)
                        time.sleep(wait)

            if not score_success:
                logger.error("All 3 attempts failed for metric '%s'.", metric.name)
                logger.error("!!! Dừng chương trình ngay lập tức để bảo toàn dữ liệu Checkpoint !!!")
                sys.exit(1)

            logger.info("Cooldown %ds...", args.cooldown)
            time.sleep(args.cooldown)

        save_records = []
        for r_val in records_map.values():
            r_copy = r_val.copy()
            if isinstance(r_copy.get("retrieved_contexts"), list):
                r_copy["retrieved_contexts"] = json.dumps(r_copy["retrieved_contexts"], ensure_ascii=False)
            save_records.append(r_copy)

        current_df = pd.DataFrame(save_records)
        cols = ["question", "ground_truth", "rag_answer", "retrieved_contexts", "faithfulness", "answer_relevancy", "context_recall", "error"]
        current_df = current_df[[c for c in cols if c in current_df.columns]]
        atomic_save_csv(current_df, REPORT_PATH)

    try:
        final_df = pd.read_csv(REPORT_PATH)
    except Exception:
        final_df = pd.DataFrame(list(records_map.values()))
    _print_summary(final_df)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print(f"  RAGAS EVALUATION SUMMARY | Backend: {backend.upper()}")
    print("=" * 55)
    if df.empty:
        print("  No results available.")
    else:
        scored = df[
            df["faithfulness"].notna() & 
            df["answer_relevancy"].notna() & 
            df["context_recall"].notna()
        ]
        n = len(scored)
        print(f"  Total samples fully evaluated : {n}")
        if n > 0:
            print(f"  Faithfulness                  : {scored['faithfulness'].mean():.4f}")
            print(f"  Answer Relevancy              : {scored['answer_relevancy'].mean():.4f}")
            print(f"  Context Recall                : {scored['context_recall'].mean():.4f}")
        
        incomplete = len(df) - n
        if incomplete > 0:
            print(f"  Incomplete/failed samples    : {incomplete}")
    print("=" * 55)
    print(f"  Report saved to: {REPORT_PATH}\n")

if __name__ == "__main__":
    main()
