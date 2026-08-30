import os
import json
import time
import requests
import csv
import sys

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

JUDGE_SYSTEM_PROMPT = """You are an impartial RAG evaluator. Your task is to evaluate a generated answer based on a given user question and retrieved context.

You must evaluate two metrics:
1. Faithfulness (0.0 to 1.0): Does the answer only use information found in the context? (1.0 = fully faithful, no hallucinations. 0.0 = completely hallucinated).
2. Answer Relevancy (0.0 to 1.0): Does the answer directly address the user's question? (1.0 = directly and fully answers. 0.0 = completely irrelevant).

Provide your evaluation as a JSON object with exactly these two keys: "faithfulness", "answer_relevancy". Do not output anything else.
"""

def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks (Qwen 3.6 is a reasoning model) and markdown fences."""
    text = text.strip()
    # Remove reasoning block
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Remove ```json ... ``` fences
    import re
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # Extract JSON substring if surrounded by other text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def evaluate_with_groq(question, context, answer, api_key, max_retries=3):
    # Truncate context to ~1500 chars (~400 tokens) to ensure total input tokens stay under ~800, well below 8K TPM
    prompt = f"Question: {question[:300]}\n\nContext: {context[:1500]}\n\nAnswer: {answer[:600]}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # NOTE: Do NOT use response_format=json_object with qwen/qwen3.6-27b.
    # It is a reasoning model that outputs <think>...</think> blocks before the JSON,
    # which causes Groq's server-side JSON validator to fail (400 json_validate_failed).
    # We strip the thinking block manually instead.
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                raw = response.json()["choices"][0]["message"]["content"]
                clean = _strip_thinking(raw)
                metrics = json.loads(clean)
                faith = metrics.get("faithfulness", 0.0)
                relev = metrics.get("answer_relevancy", 0.0)
                # Handle nested {score: ...} format
                if isinstance(faith, dict):
                    faith = faith.get("score", 0.0)
                if isinstance(relev, dict):
                    relev = relev.get("score", 0.0)
                return float(faith), float(relev)
            elif response.status_code == 429:
                sleep_time = 15 * (attempt + 1)
                print(f"Groq API 429 Rate/Token Limit (attempt {attempt+1}/{max_retries}). Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Groq API Error: {response.status_code} - {response.text}")
                time.sleep(5)
        except json.JSONDecodeError as e:
            print(f"Groq Evaluation JSON Error: {e}\nRaw output: {raw}")
            time.sleep(5)
        except Exception as e:
            print(f"Groq Evaluation Exception: {e}")
            time.sleep(5)
            
    return 0.0, 0.0

def get_answer_from_backend(backend_url, question):
    try:
        response = requests.post(
            f"{backend_url}/api/ask",
            json={"question": question, "decompose": False},
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return "", []
            
        full_answer = ""
        sources = []
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("event: sources"):
                    # Extract json data on next line
                    pass 
                elif decoded.startswith("data: "):
                    data_str = decoded[6:]
                    try:
                        data = json.loads(data_str)
                        if "sources" in data:
                            sources = [s.get("text", "") for s in data["sources"]]
                        elif "full_answer" in data:
                            full_answer = data["full_answer"]
                    except:
                        pass
                        
        return full_answer, sources
    except Exception as e:
        print(f"Backend API Exception: {e}")
        return "", []

def main():
    backend_url = os.environ.get("BACKEND_URL")
    if not backend_url:
        backend_url = "https://rag-vn-finance-backend.onrender.com"
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Warning: GROQ_API_KEY is not set. Evaluation will fail.")
        
    ground_truth_path = "pipeline/synthetic_qa/ground_truth_final.jsonl"
    report_path = "tests/regression_report.csv"
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    samples = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    import random
    random.seed(42) # For reproducibility if needed, or remove for true randomness
    # Pick 10 random samples to avoid LLM rate limits and token limits
    if len(samples) > 10:
        samples = random.sample(samples, 10)
                
    print(f"Running regression test on {len(samples)} samples...")
    
    results = []
    
    with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'question', 'faithfulness', 'answer_relevancy', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, sample in enumerate(samples):
            question = sample.get("question")
            print(f"[{i+1}/{len(samples)}] Testing: {question[:50]}...")
            
            # Call backend
            answer, sources = get_answer_from_backend(backend_url, question)
            context = "\n".join(sources)
            
            if not answer:
                print("  ❌ Failed to get answer from backend")
                writer.writerow({'id': i, 'question': question, 'faithfulness': 0, 'answer_relevancy': 0, 'status': 'failed_backend'})
                results.append((0.0, 0.0))
                continue
                
            # Call Groq Judge
            faithfulness, answer_relevancy = evaluate_with_groq(question, context, answer, groq_api_key)
            print(f"  ✅ Faithfulness: {faithfulness:.2f}, Relevancy: {answer_relevancy:.2f}")
            
            writer.writerow({
                'id': i, 
                'question': question, 
                'faithfulness': faithfulness, 
                'answer_relevancy': answer_relevancy, 
                'status': 'success'
            })
            
            results.append((faithfulness, answer_relevancy))
            
            # Rate & Token limit handling for qwen/qwen3.6-27b (8K TPM, 30 RPM) -> 10s delay
            time.sleep(10)
            
    # Compute averages
    if results:
        avg_faithfulness = sum(r[0] for r in results) / len(results)
        avg_relevancy = sum(r[1] for r in results) / len(results)
        
        print("\n--- Regression Test Summary ---")
        print(f"Total Samples: {len(results)}")
        print(f"Average Faithfulness: {avg_faithfulness:.4f}")
        print(f"Average Answer Relevancy: {avg_relevancy:.4f}")
        
        if avg_faithfulness < 0.7 or avg_relevancy < 0.6:
            print("❌ Regression test failed: Metrics below threshold.")
            sys.exit(1)
        else:
            print("✅ Regression test passed successfully.")
    else:
        print("No results to evaluate.")
        sys.exit(1)

if __name__ == "__main__":
    main()
