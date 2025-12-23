import os
import httpx

from .config import settings

def chat(prompt: str)-> str:
    """
    NVIDIA chat wrapper. Always returns a STRING (never None).
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload= {
        "model": settings.gen_model,
        "messages":[
            {"role": "system", "content": "Follow instructions strictly and cite sources."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
        "stream": False,
    }
    
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{settings.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        return data.get("choices", [{}])[0].get("message", {}).get("content") or "ERROR: Empty model response."

    except Exception as e:
        return f"ERROR: LLM call failed: {e}"