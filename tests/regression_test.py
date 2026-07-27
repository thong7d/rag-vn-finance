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

def evaluate_with_groq(question, context, answer, api_key):
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            metrics = json.loads(result["choices"][0]["message"]["content"])
            return float(metrics.get("faithfulness", 0.0)), float(metrics.get("answer_relevancy", 0.0))
        else:
            print(f"Groq API Error: {response.status_code} - {response.text}")
            return 0.0, 0.0
    except Exception as e:
        print(f"Groq Evaluation Exception: {e}")
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
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:10000")
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
            
            # Rate limit handling: max 30 RPM -> 2 seconds delay
            time.sleep(2)
            
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
