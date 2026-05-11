"""
generation.py — OpenRouter answer generation for Phase 7.
Model: google/gemma-4-31b-it:free (from config.yaml).
Implemented in Phase 5 and 7.
"""

from openai import OpenAI
import json
import re
import time
import logging
from src.utils import get_env, load_config

logger = logging.getLogger(__name__)
config = load_config()

def strip_markdown_json(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

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

def generate_synthetic_qa_batch(prompt: str, max_retries: int = 2) -> dict:
    """
    Generate a batch of QA pairs using either Gemini (via OpenAI compatibility) or OpenRouter.
    """
    gemini_key = get_env('GEMINI_API_KEY')
    openrouter_key = get_env('OPENROUTER_API_KEY')
    
    if gemini_key:
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key
        )
        model_name = "gemini-3.1-flash-lite" # Or Gemma 4
    elif openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        model_name = config['generation']['model']
    else:
        raise ValueError("Vui lòng set GEMINI_API_KEY hoặc OPENROUTER_API_KEY")

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            # OpenRouter requires this for JSON mode. Gemini works fine without it (and sometimes fails if format isn't strictly defined in their schema).
            if not gemini_key:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = client.chat.completions.create(**kwargs)
            
            output_text = response.choices[0].message.content
            clean_json = strip_markdown_json(output_text)
            return json.loads(clean_json)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise ValueError("Response was not valid JSON after retries")
        except Exception as e:
            logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise e
                
        # Exponential backoff
        sleep_time = (2 ** attempt) + (4 if gemini_key else 0) # Gemini limit is 15 RPM, so base sleep is higher
        logger.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
