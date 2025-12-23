from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import json

@dataclass(frozen=True)
class Settings:
    # --- NVIDIA endpoints ---
    base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # --- Models ---
    embed_model: str = "nvidia/nv-embedqa-e5-v5"
    gen_model: str = "nvidia/nvidia-nemotron-nano-9b-v2"
    
    # --- Retrieval ---
    top_k: int = 3
    
    # Confidence thresholds
    confident_score : float = 0.25
    confident_score_vague: float = 0.35  # for "he/his/she" type queries
    
    # --- Paths ---
    project_root = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"/ "public_docs"
    cache_dir: Path = project_root / "cache"
    metrics_dir: Path = project_root / "metrics"
    chunks_file: Path = project_root / "cache" / "chunks.json"
    
settings = Settings()