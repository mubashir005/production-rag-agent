import os
from typing import List
import numpy as np
import httpx

from .config import settings


def embed_texts(texts: List[str], input_type: str) -> np.ndarray:
    """
    NVIDIA embeddings wrapper.
    input_type: "query" or "passage"
    Returns: np.ndarray shape (N, D)
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in environment.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.embed_model,
        "input": texts,
        "input_type": input_type,
    }

    with httpx.Client(timeout=60) as client:
        r = client.post(f"{settings.base_url}/embeddings", headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Embeddings error {r.status_code}: {r.text}")
        data = r.json()

    vectors = [item["embedding"] for item in data["data"]]
    return np.array(vectors, dtype=np.float32)
