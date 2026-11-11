import os
import json
import time
import requests
import sys

def main():
    backend_url = os.environ.get("BACKEND_URL")
    if not backend_url:
        backend_url = "https://rag-vn-finance-backend.onrender.com"
    ground_truth_path = "pipeline/synthetic_qa/ground_truth_final.jsonl"
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: {ground_truth_path} not found.")
        sys.exit(1)
        
    samples = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= 5:
                break
                
    if not samples:
        print("Error: No samples found.")
        sys.exit(1)

    print(f"Running smoke test on {len(samples)} samples against {backend_url}...")
    
    success_count = 0
    
    for i, sample in enumerate(samples):
        question = sample.get("question")
        print(f"[{i+1}/{len(samples)}] Testing: {question[:50]}...")
        
        passed = False
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{backend_url}/api/ask",
                    json={"question": question, "decompose": False},
                    stream=True,
                    timeout=90
                )
                
                if response.status_code != 200:
                    print(f"  ⚠️ Attempt {attempt+1} HTTP {response.status_code}")
                    time.sleep(5)
                    continue
                    
                has_sources = False
                has_done = False
                
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("event: sources"):
                            has_sources = True
                        elif decoded.startswith("event: done"):
                            has_done = True
                        elif decoded.startswith("event: error"):
                            print(f"  ⚠️ Received SSE error event: {decoded}")
                            break
                
                if has_sources and has_done:
                    passed = True
                    break
                else:
                    print(f"  ⚠️ Attempt {attempt+1} missing events (sources: {has_sources}, done: {has_done})")
                    time.sleep(5)
            except Exception as e:
                print(f"  ⚠️ Attempt {attempt+1} Exception: {e}")
                time.sleep(5)
        
        if passed:
            print("  ✅ Passed")
            success_count += 1
        else:
            print("  ❌ Failed after retries")
            
        if i < len(samples) - 1:
            time.sleep(6) # Spacing requests to prevent hitting Gemini/Groq rate limits
        
    print(f"\nSmoke test complete: {success_count}/{len(samples)} passed.")
    if success_count < len(samples):
        print("Smoke test failed!")
        sys.exit(1)
    else:
        print("Smoke test passed successfully!")

if __name__ == "__main__":
    main()
