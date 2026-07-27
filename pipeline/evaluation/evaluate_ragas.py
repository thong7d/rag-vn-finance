"""
evaluate_ragas.py — Unified Dual-Judge RAGAS Evaluation
=========================================================
Evaluates generated answers for all models (gemini, mistral, gemma).
Primary Judge: Cerebras (llama3.1-70b)
Fallback Judge: OpenRouter Qwen3-32B
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger, load_config, get_env

logger = setup_logger("RagasEval")
config = load_config()

# Monkey patch for vertexai
from types import ModuleType
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    dummy_vertexai = ModuleType('langchain_community.chat_models.vertexai')
    dummy_vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {}) 
    sys.modules['langchain_community.chat_models.vertexai'] = dummy_vertexai
    
if 'langchain_community.llms.vertexai' not in sys.modules:
    dummy_llms_vertexai = ModuleType('langchain_community.llms.vertexai')
    dummy_llms_vertexai.VertexAI = type('VertexAI', (object,), {}) 
    sys.modules['langchain_community.llms.vertexai'] = dummy_llms_vertexai


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


def get_llm_judge(provider="cerebras"):
    from langchain_openai import ChatOpenAI
    
    if provider == "cerebras":
        api_key = get_env("CEREBRAS_API_KEY")
        if not api_key: raise ValueError("CEREBRAS_API_KEY missing")
        return ChatOpenAI(
            model="gpt-oss-120b",
            openai_api_key=api_key,
            openai_api_base="https://api.cerebras.ai/v1",
            temperature=0.0,
            max_tokens=8192,
            max_retries=1
        )
    elif provider == "openrouter":
        api_key = get_env("OPENROUTER_API_KEY")
        if not api_key: raise ValueError("OPENROUTER_API_KEY missing")
        return ChatOpenAI(
            model="qwen/qwen3-32b",
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=4096,
            max_retries=1,
            default_headers={"HTTP-Referer": "https://github.com/thong7d", "X-Title": "Fallback Eval"}
        )
    raise ValueError(f"Unknown provider: {provider}")


def build_ragas_metrics(judge_lc_llm):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(judge_lc_llm)
    
    hf_embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embed = LangchainEmbeddingsWrapper(hf_embed)

    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embed
    answer_relevancy.strictness = 1
    context_recall.llm = ragas_llm

    return [faithfulness, answer_relevancy, context_recall]

# TPD Tracker for Cerebras limit
class TokenTracker:
    def __init__(self, limit_per_day=950000, tpm_limit=28000):
        self.log_file = ROOT / "evaluation" / "cerebras_token_usage.json"
        self.limit_per_day = limit_per_day
        self.tpm_limit = tpm_limit
        self.tokens_used_today = 0
        self.tokens_used_this_minute = 0
        self.minute_start_time = time.time()
        self.load()

    def load(self):
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                    self.tokens_used_today = data.get("tokens", 0)
            except:
                pass

    def save(self):
        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "tokens": self.tokens_used_today
        }
        with open(self.log_file, "w") as f:
            json.dump(data, f)

    def add_tokens(self, amount):
        now = time.time()
        if now - self.minute_start_time > 60:
            self.minute_start_time = now
            self.tokens_used_this_minute = 0
            
        self.tokens_used_today += amount
        self.tokens_used_this_minute += amount
        self.save()

    def check_tpm_and_sleep(self, upcoming_amount):
        now = time.time()
        if now - self.minute_start_time > 60:
            self.minute_start_time = now
            self.tokens_used_this_minute = 0
            
        if self.tokens_used_this_minute + upcoming_amount > self.tpm_limit:
            sleep_needed = 60 - (now - self.minute_start_time) + 2
            if sleep_needed > 0:
                logger.info("TPM limit approaching. Sleeping for %.1fs...", sleep_needed)
                time.sleep(sleep_needed)
                self.minute_start_time = time.time()
                self.tokens_used_this_minute = 0

    def check_tpd(self):
        if self.tokens_used_today >= self.limit_per_day:
            return False
        return True

def main():
    parser = argparse.ArgumentParser(description="Dual-Judge Ragas Evaluation for All Models")
    parser.add_argument("--model", type=str, required=True, choices=["mistral", "gemma", "gemini"], help="Model to evaluate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1) # Forced to 1 due to strict TPM/RPM
    parser.add_argument("--cooldown", type=int, default=60) # Must be 60 to avoid >5 RPM limit
    args = parser.parse_args()

    OUTPUT_DIR = ROOT / config["evaluation"]["output_dir"]
    backend = get_env("VECTOR_STORE_BACKEND", config["vector_store"]["backend"]).lower()

    if args.model == "gemini":
        INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{backend}.parquet"
        REPORT_PATH = OUTPUT_DIR / f"evaluation_report_pure_ragas_{backend}.csv"
    else:
        INPUT_PARQUET = OUTPUT_DIR / f"generation_results_{args.model}.parquet"
        REPORT_PATH = OUTPUT_DIR / f"evaluation_report_ragas_{args.model}.csv"

    if not INPUT_PARQUET.exists():
        logger.error("Generation results not found at: %s", INPUT_PARQUET)
        sys.exit(1)

    df_eval = pd.read_parquet(INPUT_PARQUET)
    all_records = df_eval.to_dict("records")
    if args.limit: all_records = all_records[:args.limit]

    records_map = {}
    if REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(REPORT_PATH)
            for _, row in existing_df.iterrows():
                q_key = str(row["question"]).strip()
                row_dict = row.to_dict()
                for k in ["faithfulness", "answer_relevancy", "context_recall"]:
                    if k in row_dict and pd.isna(row_dict[k]): row_dict[k] = None
                
                if "retrieved_context" in row_dict and pd.notna(row_dict["retrieved_context"]):
                    ctx_str = row_dict["retrieved_context"]
                    row_dict["retrieved_contexts"] = ctx_str.split("\n---\n") if ctx_str else []
                elif "retrieved_contexts" in row_dict and pd.notna(row_dict["retrieved_contexts"]):
                    try:
                        row_dict["retrieved_contexts"] = json.loads(row_dict["retrieved_contexts"])
                    except:
                        row_dict["retrieved_contexts"] = []
                
                records_map[q_key] = row_dict
            logger.info("Resumed %d records from existing report.", len(records_map))
        except Exception as e:
            logger.warning("Could not read report: %s", e)

    # Populate map
    for rec in all_records:
        q_key = rec["question"].strip()
        if q_key not in records_map:
            ctx_str = rec.get("retrieved_context", "")
            ctx_list = ctx_str.split("\n---\n") if ctx_str else []
            
            records_map[q_key] = {
                "question": q_key,
                "ground_truth": rec.get("ground_truth", ""),
                "rag_answer": rec.get("generated_answer", ""),
                "retrieved_context": ctx_str,
                "retrieved_contexts": ctx_list,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "error": "",
            }

    pending_qs = [q for q, r in records_map.items() if r.get("faithfulness") is None or r.get("answer_relevancy") is None or r.get("context_recall") is None]
    logger.info("Total pending samples: %d", len(pending_qs))
    if not pending_qs: return

    current_judge = "cerebras"
    judge_llm = get_llm_judge(current_judge)
    metrics = build_ragas_metrics(judge_llm)

    from ragas import evaluate, RunConfig
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    run_config = RunConfig(max_workers=1, timeout=180)
    batch_size = args.batch_size
    
    tracker = TokenTracker(limit_per_day=950000, tpm_limit=25000)

    for b_idx in range(0, len(pending_qs), batch_size):
        # TPD check before processing batch
        if not tracker.check_tpd():
            logger.error("!!! CRITICAL: Reached daily token limit for Cerebras (1,000,000). Stopping gracefully. Run again tomorrow.")
            sys.exit(0)
            
        batch_qs = pending_qs[b_idx : b_idx + batch_size]
        logger.info("Batch %d - Size %d | Tokens Used Today: %d", (b_idx // batch_size) + 1, len(batch_qs), tracker.tokens_used_today)

        samples = []
        batch_keys = []
        for q_key in batch_qs:
            r = records_map[q_key]
            samples.append(SingleTurnSample(
                user_input=q_key,
                response=r["rag_answer"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["ground_truth"]
            ))
            batch_keys.append(q_key)

        for metric in metrics:
            m_samples = [s for s, k in zip(samples, batch_keys) if records_map[k].get(metric.name) is None]
            m_keys = [k for k in batch_keys if records_map[k].get(metric.name) is None]
            
            if not m_samples: continue
            dataset = EvaluationDataset(samples=m_samples)
            
            # Estimated tokens per metric call: ~8000
            estimated_tokens = 8000 * len(m_samples)
            tracker.check_tpm_and_sleep(estimated_tokens)
            
            score_success = False
            for attempt in range(4):
                try:
                    result = evaluate(dataset=dataset, metrics=[metric], run_config=run_config)
                    tracker.add_tokens(estimated_tokens) # Add tokens upon successful call
                    
                    df_res = result.to_pandas()
                    if df_res.empty or df_res[metric.name].isna().all():
                        raise RuntimeError("Ragas returned NaNs. API Error suspected.")
                    
                    for idx, res_row in df_res.iterrows():
                        target_key = m_keys[idx]
                        score_val = res_row.get(metric.name)
                        if pd.notna(score_val):
                            records_map[target_key][metric.name] = float(score_val)
                            logger.info("  %s -> %.4f", metric.name, score_val)
                    score_success = True
                    break
                    
                except Exception as e:
                    err = str(e).lower()
                    logger.warning("[%s Attempt %d/4] Error: %s", metric.name, attempt + 1, e)
                    
                    if "insufficient" in err or "402" in err or "balance" in err:
                        logger.error("CRITICAL: Out of funds on %s! Stopping.", current_judge)
                        sys.exit(1)
                        
                    if attempt < 3:
                        if current_judge == "cerebras" and ("429" in err or "limit" in err or "tpm" in err or "rpm" in err):
                            logger.info("[JUDGE FALLBACK] Rate limited. Sleeping 60s...")
                            time.sleep(60)
                            continue 
                            
                        wait = 15 * (2 ** attempt)
                        logger.info("Waiting %ds...", wait)
                        time.sleep(wait)

            if not score_success:
                logger.error("All 4 attempts failed for metric '%s'. Exiting.", metric.name)
                sys.exit(1)

        # Force a sleep after each sample to safely respect the 30k TPM/5 RPM limit
        logger.info("Batch cooldown %ds...", args.cooldown)
        time.sleep(args.cooldown)

        # Save checkpoint
        save_records = []
        for r_val in records_map.values():
            r_copy = r_val.copy()
            if "retrieved_contexts" in r_copy: del r_copy["retrieved_contexts"]
            save_records.append(r_copy)
        
        current_df = pd.DataFrame(save_records)
        atomic_save_csv(current_df, REPORT_PATH)

    logger.info("Evaluation complete! Results at %s", REPORT_PATH)

if __name__ == "__main__":
    main()
