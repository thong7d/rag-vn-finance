"""
evaluate_ragas_pure.py — Official Ragas Framework Evaluation
=============================================================

Evaluates the Vietnamese Financial News RAG system using the pure ragas library.

Key Features & Enhancements (Production-Ready):
  1. Rate Limit & Burst Prevention: Enforces sequential request evaluation using 
     Ragas RunConfig(max_workers=1). Avoids OpenRouter 429 errors.
  2. Batch-Based Dataset Processing: Groups evaluation samples into batches of 
     size N (default: 5) to leverage Ragas' batch optimizations without risking 
     complete data loss on crash.
  3. Partial Metric Isolation: Evaluates metrics (faithfulness, answer_relevancy, 
     context_recall) sequentially with independent try-except blocks. If one 
     metric fails (e.g. context_recall due to context size), other completed 
     metric scores are preserved.
  4. Granular Resume Capability: Tracks completed metrics at the individual 
     question level in the CSV report. If evaluation is interrupted, only the 
     missing metrics for incomplete questions are evaluated on rerun, reusing 
     already-generated RAG answers to minimize API cost.
  5. Atomic Writes: Updates evaluation_report_pure_ragas.csv atomically using a 
     temp-file swap to prevent file corruption.

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
# Ragas 0.4.x đôi khi tìm kiếm module cũ đã bị xóa của langchain_community.
# Đoạn mã này tạo ra một "module ảo" để đánh lừa Ragas, tránh lỗi ImportError.
import sys
from types import ModuleType

if 'langchain_community.chat_models.vertexai' not in sys.modules:
    dummy_vertexai = ModuleType('langchain_community.chat_models.vertexai')
    dummy_vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {}) # Tạo class ảo
    sys.modules['langchain_community.chat_models.vertexai'] = dummy_vertexai
    
if 'langchain_community.llms.vertexai' not in sys.modules:
    dummy_llms_vertexai = ModuleType('langchain_community.llms.vertexai')
    dummy_llms_vertexai.VertexAI = type('VertexAI', (object,), {}) # Tạo class ảo
    sys.modules['langchain_community.llms.vertexai'] = dummy_llms_vertexai
# ==============================================================================
# KẾT THÚC VÁ LỖI
# ==============================================================================

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any local src.* import
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env
from src.generation import generate_answer  # auto-fallback Gemini→OpenRouter

# ---------------------------------------------------------------------------
# Logger & config
# ---------------------------------------------------------------------------
logger = setup_logger("RagasEval")
config = load_config()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QA_FILE       = ROOT / config["synthetic_qa"]["output_dir"] / "ground_truth_final.jsonl"
OUTPUT_DIR    = ROOT / config["evaluation"]["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH   = OUTPUT_DIR / "evaluation_report_pure_ragas.csv"

# Default configuration
DEFAULT_BATCH_SIZE = 5
DEFAULT_COOLDOWN   = 10  # Seconds between metric evaluations

# ---------------------------------------------------------------------------
# Ragas metric setup
# ---------------------------------------------------------------------------
def build_ragas_metrics():
    """
    Instantiate and return the three ragas metric objects.

    LLM  : qwen/qwen3-32b served through the OpenRouter OpenAI-compat endpoint.
    Embed: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local).
    """
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openrouter_key = get_env("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Cannot initialise Ragas judge.")

    # ── Judge LLM ────────────────────────────────────────────────────────────
    judge_lc_llm = ChatOpenAI(
        model="qwen/qwen3-32b",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096,  # Tối ưu hóa tránh lỗi cắt bớt JSON
        default_headers={
            "HTTP-Referer": "https://github.com/thong7d/rag-vn-finance",
            "X-Title": "Vietnamese Financial RAG Eval"
        }
    )
    ragas_llm = LangchainLLMWrapper(judge_lc_llm)

    # ── Embedding model (local, no API cost) ─────────────────────────────────
    hf_embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embed = LangchainEmbeddingsWrapper(hf_embed)

    # ── Metric configuration ─────────────────────────────────────────────────
    faithfulness.llm            = ragas_llm
    answer_relevancy.llm        = ragas_llm
    answer_relevancy.embeddings = ragas_embed
    answer_relevancy.strictness = 1          # per requirement
    context_recall.llm          = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_recall]
    logger.info("Ragas metrics initialised: %s", [m.name for m in metrics])
    return metrics


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_qa_records(limit: Optional[int] = None) -> List[Dict]:
    if not QA_FILE.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {QA_FILE}")

    records: List[Dict] = []
    with open(QA_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if all(k in rec for k in ("question", "ground_truth", "contexts")):
                records.append(rec)

    logger.info("Loaded %d QA records from %s", len(records), QA_FILE)
    if limit:
        records = records[:limit]
        logger.info("Evaluation capped at %d samples.", limit)
    return records


# ---------------------------------------------------------------------------
# Atomic CSV save (safe against mid-write crashes)
# ---------------------------------------------------------------------------
def atomic_save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write df to a temp file then atomically replace the target."""
    tmp = path.with_suffix(".csv.tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        logger.info("[Checkpoint] Saved %d rows → %s", len(df), path)
    except Exception as exc:
        logger.error("[Checkpoint] Save failed: %s", exc)
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ragas pure-library evaluation for Vietnamese Financial News RAG"
    )
    parser.add_argument("--limit",    type=int, default=None,
                        help="Cap the number of samples (default: all 150).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of samples evaluated together (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN,
                        help=f"Seconds to sleep between metric evaluations (default: {DEFAULT_COOLDOWN}).")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    all_records = load_qa_records(limit=args.limit)

    # ── Load existing data from CSV if present ──────────────────────────────
    records_map = {}  # Normalized question -> dict
    if REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(REPORT_PATH)
            for _, row in existing_df.iterrows():
                q_key = str(row["question"]).strip()
                # Use to_dict() but handle NaN values as None
                row_dict = row.to_dict()
                for k in ["faithfulness", "answer_relevancy", "context_recall"]:
                    if k in row_dict and pd.isna(row_dict[k]):
                        row_dict[k] = None
                if "rag_answer" in row_dict and pd.isna(row_dict["rag_answer"]):
                    row_dict["rag_answer"] = None
                records_map[q_key] = row_dict
            logger.info("Resume mode: loaded %d existing records from CSV.", len(records_map))
        except Exception as exc:
            logger.warning("Could not read existing report (%s); starting fresh.", exc)

    # Initialize records in the map if they do not exist
    for rec in all_records:
        q_key = rec["question"].strip()
        if q_key not in records_map:
            records_map[q_key] = {
                "question": q_key,
                "ground_truth": rec["ground_truth"],
                "rag_answer": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "error": "",
            }

    # Filter pending questions (those that are missing any metric score)
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
        # Load final results to print summary
        try:
            final_df = pd.read_csv(REPORT_PATH)
        except Exception:
            final_df = pd.DataFrame(list(records_map.values()))
        _print_summary(final_df)
        return

    # ── Initialise Ragas components ──────────────────────────────────────────
    try:
        metrics = build_ragas_metrics()
    except Exception as exc:
        logger.error("Failed to initialise Ragas metrics: %s", exc)
        sys.exit(1)

    # Ragas imports
    from ragas import evaluate
    try:
        from ragas import RunConfig
    except ImportError:
        from ragas.run_config import RunConfig

    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    # Configure run settings
    run_config = RunConfig(
        max_workers=1,  # Strictly serialize execution threads internally
        timeout=180     # Robust timeout against slow API responses
    )

    # ── Main Batch Loop ──────────────────────────────────────────────────────
    batch_size = args.batch_size
    num_batches = (len(pending_questions) - 1) // batch_size + 1

    for b_idx in range(0, len(pending_questions), batch_size):
        batch_qs = pending_questions[b_idx : b_idx + batch_size]
        logger.info("==========================================")
        logger.info("Processing Batch %d/%d (Size: %d)", (b_idx // batch_size) + 1, num_batches, len(batch_qs))
        logger.info("==========================================")

        samples = []
        batch_valid_keys = []

        # Step A: Answer generation and sample validation
        for q_key in batch_qs:
            row = records_map[q_key]
            
            # Retrieve details from original dataset
            orig_rec = next((r for r in all_records if r["question"].strip() == q_key), None)
            if not orig_rec:
                logger.error("Question '%s' not found in loaded QA records! Skipping.", q_key[:40])
                continue

            contexts = orig_rec["contexts"]
            ground_truth = orig_rec["ground_truth"]

            # Generate answer if not present
            ans = row.get("rag_answer")
            if not ans:
                try:
                    logger.info("Generating RAG answer for: %s...", q_key[:60])
                    ans = generate_answer(q_key, contexts)
                    row["rag_answer"] = ans
                except Exception as exc:
                    logger.error("Answer generation failed for Q: %s. Error: %s", q_key[:40], exc)
                    row["error"] = f"Generation failed: {str(exc)}"
                    continue

            # Construct Ragas SingleTurnSample
            sample = SingleTurnSample(
                user_input=q_key,
                response=ans,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            samples.append(sample)
            batch_valid_keys.append(q_key)

        if not samples:
            logger.warning("No valid samples in this batch. Continuing.")
            continue

        # Step B: Metric-by-metric evaluation
        for metric in metrics:
            # Lọc chỉ những mẫu thực sự chưa có điểm cho metric này
            metric_samples = []
            metric_keys = []
            
            for sample, q_key in zip(samples, batch_valid_keys):
                if records_map[q_key].get(metric.name) is None:
                    metric_samples.append(sample)
                    metric_keys.append(q_key)

            if not metric_samples:
                logger.info("Metric '%s' already completed for all questions in this batch. Skipping.", metric.name)
                continue

            logger.info("Evaluating metric '%s' on %d remaining samples...", metric.name, len(metric_samples))
            
            # Khởi tạo dataset ĐỘNG chỉ chứa các mẫu bị thiếu
            dataset = EvaluationDataset(samples=metric_samples)
            
            score_success = False
            for attempt in range(3):
                try:
                    result = evaluate(
                        dataset=dataset,
                        metrics=[metric],
                        run_config=run_config
                    )
                    df_res = result.to_pandas()
                    
                    # Store scores back using strict index alignment
                    for idx, res_row in df_res.iterrows():
                        target_key = metric_keys[idx] # Sử dụng ánh xạ mảng chính xác 100%
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
                    if attempt < 2:
                        if any(tok in err.lower() for tok in ("429", "rate limit", "tpm", "rpm")):
                            wait = 30 + 15 * attempt
                        else:
                            wait = 5 * (attempt + 1)
                        logger.info("Waiting %ds before retrying...", wait)
                        time.sleep(wait)

            if not score_success:
                logger.error("All 3 evaluation attempts failed for metric '%s' in this batch.", metric.name)
                for q_key in batch_valid_keys:
                    if records_map[q_key].get(metric.name) is None:
                        records_map[q_key]["error"] = f"Metric {metric.name} failed after 3 retries."

            # Cooldown between metric runs to strictly respect the 60 RPM limit
            logger.info("Cooldown %ds...", args.cooldown)
            time.sleep(args.cooldown)

        # Step C: Save checkpoint atomically after each batch
        current_df = pd.DataFrame(list(records_map.values()))
        # Sort columns to ensure beautiful output CSV
        cols = ["question", "ground_truth", "rag_answer", "faithfulness", "answer_relevancy", "context_recall", "error"]
        current_df = current_df[[c for c in cols if c in current_df.columns]]
        atomic_save_csv(current_df, REPORT_PATH)

    # ── Final summary ─────────────────────────────────────────────────────────
    try:
        final_df = pd.read_csv(REPORT_PATH)
    except Exception:
        final_df = pd.DataFrame(list(records_map.values()))
    _print_summary(final_df)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION SUMMARY (pure ragas library)")
    print("=" * 55)
    if df.empty:
        print("  No results available.")
    else:
        # Filter rows where all 3 metrics are successfully scored
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
        
        # Print status of partially failed or incomplete runs
        incomplete = len(df) - n
        if incomplete > 0:
            print(f"  Incomplete/failed samples    : {incomplete}")
    print("=" * 55)
    print(f"  Report saved to: {REPORT_PATH}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
