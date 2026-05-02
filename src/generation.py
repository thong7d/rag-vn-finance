"""
generation.py — OpenRouter answer generation for Phase 7.
Model: qwen/qwen-2.5-7b-instruct:free (from config.yaml).
Implemented in Phase 7.
"""

from openai import OpenAI
import json
from src.utils import get_env, load_config

config = load_config()

def generate_answer(prompt: str) -> str:
    """
    Generate an answer using OpenRouter's API with OpenAI client syntax.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_env('OPENROUTER_API_KEY')
    )
    
    response = client.chat.completions.create(
        model=config['generation']['model'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=config['generation']['temperature']
    )
    return response.choices[0].message.content

def generate_synthetic_qa_batch(prompt: str) -> dict:
    """
    Generate a batch of QA pairs using OpenRouter's API, enforcing JSON output.
    Used in Phase 5.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_env('OPENROUTER_API_KEY')
    )
    
    response = client.chat.completions.create(
        model=config['generation']['model'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        # CRITICAL FOR PHASE 5 (Synthetic QA): You must enforce JSON output.
        response_format={"type": "json_object"}
    )
    
    output_text = response.choices[0].message.content
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        # Fallback handling should be implemented by the caller
        raise ValueError("Response was not valid JSON")
