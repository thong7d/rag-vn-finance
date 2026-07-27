import os
import json
import time
import requests
import sys

def main():
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:10000")
    ground_truth_path = "pipeline/synthetic_qa/ground_truth_final.jsonl"
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: {ground_truth_path} not found.")
        sys.exit(1)
        
    samples = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= 10:
                break
                
    if not samples:
        print("Error: No samples found.")
        sys.exit(1)

    print(f"Running smoke test on {len(samples)} samples against {backend_url}...")
    
    success_count = 0
    
    for i, sample in enumerate(samples):
        question = sample.get("question")
        print(f"[{i+1}/{len(samples)}] Testing: {question[:50]}...")
        
        try:
            # We use stream=True to parse SSE
            response = requests.post(
                f"{backend_url}/api/ask",
                json={"question": question, "decompose": False},
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"  ❌ Failed: HTTP {response.status_code}")
                continue
                
            has_sources = False
            has_done = False
            
            # Read streaming response
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("event: sources"):
                        has_sources = True
                    elif decoded.startswith("event: done"):
                        has_done = True
                    elif decoded.startswith("event: error"):
                        print(f"  ❌ Failed: Received SSE error event: {decoded}")
                        break
            
            if has_sources and has_done:
                print("  ✅ Passed")
                success_count += 1
            else:
                print(f"  ❌ Failed: Missing expected SSE events (sources: {has_sources}, done: {has_done})")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            
        time.sleep(1) # Be nice to the API
        
    print(f"\nSmoke test complete: {success_count}/{len(samples)} passed.")
    if success_count < len(samples):
        print("Smoke test failed!")
        sys.exit(1)
    else:
        print("Smoke test passed successfully!")

if __name__ == "__main__":
    main()
